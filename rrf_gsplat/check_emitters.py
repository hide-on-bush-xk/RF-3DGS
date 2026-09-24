"""Smoke for --emitters (M3's placement oracle; train_rrf.RRF.add_emitters, sionna_port/path_emitters.py).

Criteria, written before the first run (2026-09-24):
  S1  at initialisation (emitter value 0.01 above the floor, opacity 0.01) the renders equal the plain model's: median |diff|
      <= 0.05 dB and max <= 0.5 dB over the smoke views (every face of 12 stratified positions, the flat view included)
  S2  after one step the emitters' SH (DC and rest) and opacity have a nonzero, finite gradient; their means and
      scales do not train
  S3  300 steps on the smoke views: no NaN, the mean loss of the last 20 steps below that of the first 20
Reported, not judged (S4, train_rrf.py runs on the same views, see rounds/win_m3_smoke.sh): <= 1 deg with and
without the emitters, and the step time.

    C:/Users/Ke/miniconda3/envs/rf-gsplat-win/python.exe rrf_gsplat/check_emitters.py --emitters output/rrf/m3/emitters_smoke.npz \
        --names output/rrf/m3/smoke_names.txt
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import train_rrf as T  # noqa: E402
from losses import l1_loss  # noqa: E402

APS = os.path.join(T.REPO, "RF-3DGS_dataset", "regenerated", "3dgs_APS_60_gpct")
CK = os.path.join(T.REPO, "RF-3DGS_dataset", "blender_visual_trained", "chkpnt30000.pth")


def model(em, dev, scale=0.025):
    m = T.RRF(CK, "power", 1, 3, dev)
    m.sh_backend = "gsplat"
    g = torch.Generator().manual_seed(3)          # a trained-looking colour, the same in both models
    m.params["sh0"].data.copy_(0.1 * torch.randn(m.params["sh0"].shape, generator=g).to(dev))
    m.params["shN"].data.copy_(0.05 * torch.randn(m.params["shN"].shape, generator=g).to(dev))
    if em is not None:
        m.add_emitters(em, scale)
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--emitters", required=True); ap.add_argument("--names", required=True)
    a = ap.parse_args()
    os.environ["GSPLAT_BWD_NO_GEOM"] = "1"
    torch.backends.cudnn.allow_tf32 = False
    dev = "cuda"
    meta = json.load(open(os.path.join(APS, "generation_meta.json")))
    vmin, span = meta["spec_min_db"], meta["spec_max_db"] - meta["spec_min_db"]
    names = [l.strip() for l in open(a.names) if l.strip()]
    views = T.read_colmap_text(os.path.join(APS, "sparse", "0"))
    data = T.load_views(APS, names, views, dev, want_float=True)
    em = np.load(a.emitters)["means"]
    res = {"views": len(names), "emitters": int(len(em))}
    # S1
    m0, m1 = model(None, dev), model(em, dev)
    diffs = []
    with torch.no_grad():
        for s in range(0, len(names), 4):
            sl = slice(s, s + 4)
            x = m0.render_batch(data["viewmats"][sl], data["Ks"][sl], data["width"], data["height"], span)
            y = m1.render_batch(data["viewmats"][sl], data["Ks"][sl], data["width"], data["height"], span)
            diffs.append(((y - x).abs() * span).flatten().cpu())
    d = torch.cat(diffs)
    res["S1_median_db"], res["S1_max_db"] = float(d.median()), float(d.max())
    res["S1"] = res["S1_median_db"] <= 0.05 and res["S1_max_db"] <= 0.5
    # S2 and S3
    lrs = {"sh0": 5e-3, "shN": 2.5e-4, "opacities": 0.1, "em_sh0": 5e-3, "em_shN": 2.5e-4, "em_opacities": 0.1}
    opt = {k: torch.optim.Adam([p], lr=lrs[k], eps=1e-15) for k, p in m1.params.items() if p.requires_grad}
    target = lambda i: ((data["float"][i].float() - vmin) / span).clamp(0, 1)[None]    # noqa: E731
    fixed = {k: m1.params[k].detach().clone() for k in ("em_means", "em_scales")}
    losses = []
    n_pos = len(names) // 4
    for it in range(300):
        s = 4 * (it % n_pos); sl = slice(s, s + 4)
        img = m1.render_batch(data["viewmats"][sl], data["Ks"][sl], data["width"], data["height"], span)
        gt = torch.stack([target(i) for i in range(s, s + 4)])
        loss = l1_loss(img, gt)
        for o in opt.values():
            o.zero_grad(set_to_none=True)
        loss.backward()
        if it == 0:
            g = {k: m1.params[k].grad for k in ("em_sh0", "em_shN", "em_opacities")}
            res["S2_grad_norms"] = {k: (None if v is None else float(v.norm())) for k, v in g.items()}
            res["S2"] = all(v is not None and torch.isfinite(v).all() and float(v.norm()) > 0 for v in g.values()) and \
                not m1.params["em_means"].requires_grad and not m1.params["em_scales"].requires_grad
        for o in opt.values():
            o.step()
        losses.append(float(loss.detach()))
    res["S2"] = bool(res["S2"]) and all(torch.equal(fixed[k], m1.params[k].detach()) for k in fixed)
    res["S3_first20"], res["S3_last20"] = float(np.mean(losses[:20])), float(np.mean(losses[-20:]))
    res["S3"] = bool(np.isfinite(losses).all()) and res["S3_last20"] < res["S3_first20"]
    op = torch.sigmoid(m1.params["em_opacities"]).detach()
    res["after_300_emitter_opacity_p50_p99"] = [float(op.median()), float(op.quantile(0.99))]
    print(json.dumps(res, indent=1))
    print("ALL PASS" if res["S1"] and res["S2"] and res["S3"] else
          "FAIL: " + ", ".join(k for k in ("S1", "S2", "S3") if not res[k]))
    json.dump(res, open(os.path.join(T.REPO, "output", "rrf", "m3", "check_emitters.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
