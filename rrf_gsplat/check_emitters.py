"""Smoke for --emitters (M3's placement oracle; train_rrf.RRF.add_emitters, sionna_port/path_emitters.py).

Criteria, written before the first run (2026-09-24):
  S1  at initialisation (emitter value near the floor) the renders equal the plain model's: median |diff|
      <= 0.05 dB and max <= 0.5 dB over the smoke views (every face of 12 stratified positions, the flat view included)
  S2  after one step the emitters' SH (DC and rest) have a nonzero, finite gradient; their means, scales and (since
      the dead-emitter fix) opacities do not train
  S3  300 steps on the smoke views: no NaN, the mean loss of the last 20 steps below that of the first 20
  S5  (added with the dead-emitter fix, before its run) liveness after the 300 steps: over one pass of all smoke views,
      among the emitters inside at least one view's frustum, the share whose DC gradient is exactly zero <= 1 %
      (the first full oracle run: DC value median 0 under the clamp, opacity median 0.0028 < gsplat's 1/255 cut-off)
With --em-pcolor W (round 2: a position-conditioned colour on the emitters; written before its run):
  S1  also: the renders equal the emitters-only model's bit for bit (the MLP's last layer starts at zero)
  S2  also: after the SECOND step (the last layer is nonzero after the first) the per-emitter latent and every MLP
      tensor have a nonzero, finite gradient
  and the mean step time is reported against the emitters-only model's.
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


def model(em, dev, scale=0.025, em_pc=0, rx=None):
    m = T.RRF(CK, "power", 1, 3, dev)
    m.sh_backend = "gsplat"
    g = torch.Generator().manual_seed(3)          # a trained-looking colour, the same in both models
    m.params["sh0"].data.copy_(0.1 * torch.randn(m.params["sh0"].shape, generator=g).to(dev))
    m.params["shN"].data.copy_(0.05 * torch.randn(m.params["shN"].shape, generator=g).to(dev))
    if em is not None:
        m.add_emitters(em, scale)
    if em_pc:
        m.add_em_pcolor(em_pc, 32, 4, rx)
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--emitters", required=True); ap.add_argument("--names", required=True)
    ap.add_argument("--em-pcolor", type=int, default=0)
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
    rx = torch.linalg.inv(data["viewmats"])[:, :3, 3]
    m0, m1 = model(None, dev), model(em, dev, em_pc=a.em_pcolor, rx=rx)
    me = model(em, dev) if a.em_pcolor else None
    diffs, exact = [], 0.0
    with torch.no_grad():
        for s in range(0, len(names), 4):
            sl = slice(s, s + 4)
            x = m0.render_batch(data["viewmats"][sl], data["Ks"][sl], data["width"], data["height"], span)
            y = m1.render_batch(data["viewmats"][sl], data["Ks"][sl], data["width"], data["height"], span)
            diffs.append(((y - x).abs() * span).flatten().cpu())
            if me is not None:
                exact = max(exact, float((me.render_batch(data["viewmats"][sl], data["Ks"][sl], data["width"], data["height"], span) - y).abs().max()))
    d = torch.cat(diffs)
    res["S1_median_db"], res["S1_max_db"] = float(d.median()), float(d.max())
    res["S1"] = res["S1_median_db"] <= 0.05 and res["S1_max_db"] <= 0.5
    if me is not None:
        res["S1_vs_emitters_only_max_abs"] = exact
        res["S1"] = res["S1"] and exact == 0.0
    # S2 and S3
    lrs = {"sh0": 5e-3, "shN": 2.5e-4, "opacities": 0.1, "em_sh0": 5e-2, "em_shN": 2.5e-3,      # train_rrf: emitters x10
           "em_pc_latent": 5e-3}
    opt = {k: torch.optim.Adam([p], lr=lrs[k], eps=1e-15) for k, p in m1.params.items() if p.requires_grad}
    if a.em_pcolor:
        opt["em_mlp"] = torch.optim.Adam(m1.em_pcolor_mlp.parameters(), lr=2e-3)
    import time
    t_steps = []
    target = lambda i: ((data["float"][i].float() - vmin) / span).clamp(0, 1)[None]    # noqa: E731
    fixed = {k: m1.params[k].detach().clone() for k in ("em_means", "em_scales", "em_opacities")}
    losses = []
    n_pos = len(names) // 4
    for it in range(300):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        s = 4 * (it % n_pos); sl = slice(s, s + 4)
        img = m1.render_batch(data["viewmats"][sl], data["Ks"][sl], data["width"], data["height"], span)
        gt = torch.stack([target(i) for i in range(s, s + 4)])
        loss = l1_loss(img, gt)
        for o in opt.values():
            o.zero_grad(set_to_none=True)
        loss.backward()
        if it == 0:
            g = {k: m1.params[k].grad for k in ("em_sh0", "em_shN")}
            res["S2_grad_norms"] = {k: (None if v is None else float(v.norm())) for k, v in g.items()}
            res["S2"] = all(v is not None and torch.isfinite(v).all() and float(v.norm()) > 0 for v in g.values()) and \
                not any(m1.params[k].requires_grad for k in ("em_means", "em_scales", "em_opacities"))
        if it == 1 and a.em_pcolor:
            gp = [m1.params["em_pc_latent"].grad] + [q.grad for q in m1.em_pcolor_mlp.parameters()]
            res["S2_pcolor_grad_norms"] = [None if v is None else float(v.norm()) for v in gp]
            res["S2_pcolor"] = all(v is not None and torch.isfinite(v).all() and float(v.norm()) > 0 for v in gp)
        for o in opt.values():
            o.step()
        losses.append(float(loss.detach()))
        torch.cuda.synchronize(); t_steps.append(time.perf_counter() - t0)
    res["S2"] = bool(res["S2"]) and all(torch.equal(fixed[k], m1.params[k].detach()) for k in fixed) and         bool(res.get("S2_pcolor", True))
    res["step_ms_median_last200"] = 1000 * float(np.median(t_steps[100:]))
    res["S3_first20"], res["S3_last20"] = float(np.mean(losses[:20])), float(np.mean(losses[-20:]))
    res["S3"] = bool(np.isfinite(losses).all()) and res["S3_last20"] < res["S3_first20"]
    # S5: one pass over every smoke view, DC gradients accumulated, no step
    for o in opt.values():
        o.zero_grad(set_to_none=True)
    for s in range(0, len(names), 4):
        sl = slice(s, s + 4)
        img = m1.render_batch(data["viewmats"][sl], data["Ks"][sl], data["width"], data["height"], span)
        l1_loss(img, torch.stack([target(i) for i in range(s, s + 4)])).backward()
    gdc = m1.params["em_sh0"].grad[:, 0, 0].abs()
    mu = m1.params["em_means"].detach()
    inside = torch.zeros(len(mu), dtype=torch.bool, device=mu.device)
    for vm, K in zip(data["viewmats"], data["Ks"]):
        xc = mu @ vm[:3, :3].T + vm[:3, 3]
        z = xc[:, 2].clamp_min(1e-6)
        u, v = K[0, 0] * xc[:, 0] / z + K[0, 2], K[1, 1] * xc[:, 1] / z + K[1, 2]
        inside |= (xc[:, 2] > 0.01) & (u >= 0) & (u < data["width"]) & (v >= 0) & (v < data["height"])
    res["S5_emitters_in_a_frustum"] = int(inside.sum())
    res["S5_zero_grad_share"] = float((gdc[inside] == 0).float().mean())
    res["S5"] = res["S5_zero_grad_share"] <= 0.01
    val = (T.RRF.EM_VMAX * torch.sigmoid(m1.params["em_sh0"][:, 0, 0] * 0.28209479177387814)).detach()
    res["after_300_emitter_dc_value_p50_p90_p99"] = [float(val.median()), float(val.quantile(0.9)), float(val.quantile(0.99))]
    print(json.dumps(res, indent=1))
    print("ALL PASS" if all(res[k] for k in ("S1", "S2", "S3", "S5")) else
          "FAIL: " + ", ".join(k for k in ("S1", "S2", "S3", "S5") if not res[k]))
    json.dump(res, open(os.path.join(T.REPO, "output", "rrf", "m3",
                                     f"check_emitters{'_pc' + str(a.em_pcolor) if a.em_pcolor else ''}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
