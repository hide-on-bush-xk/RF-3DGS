"""Smoke for --lobes (P2, spherical-Gaussian lobes on top of the SH colour; train_rrf.RRF.lobe_colours).

Criteria, written before the first run:
  a  at initialisation (weights zero) the renders equal the plain SH3 model's bit for bit, on 2 positions x 4 faces
     (every orientation) spread over the held-out list
  b  after one backward every lobe parameter (weight, axis, log-kappa) has a nonzero, finite gradient -- the
     weights directly; axis and kappa only once a weight is nonzero, so this is checked after one Adam step
  c  300 steps on 2 training positions (8 views) with --lobes 1: no NaN, final loss below the first
  d  one training step (4 faces, render + loss + backward + Adam) costs <= 1.3 x the SH3 step at 300 x 200
     (CUDA events, 20 steps after 5 warm-up, median; not a timing-contract measurement, no quiet gate)

    C:/Users/Ke/miniconda3/envs/rf-gsplat-win/python.exe rrf_gsplat/check_lobes.py
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import train_rrf as T  # noqa: E402
from losses import l1_loss, ssim  # noqa: E402

MV = os.path.join(T.REPO, "RF-3DGS_dataset", "regenerated", "3dgs_MVDR_100_gpct")
CK = os.path.join(T.REPO, "RF-3DGS_dataset", "blender_visual_trained", "chkpnt30000.pth")


def model(lobes, dev, rx):
    m = T.RRF(CK, "db", 1, 3, dev)
    m.sh_backend = "gsplat"
    g = torch.Generator().manual_seed(3)
    m.params["sh0"].data.copy_(0.1 * torch.randn(m.params["sh0"].shape, generator=g).to(dev))
    m.params["shN"].data.copy_(0.05 * torch.randn(m.params["shN"].shape, generator=g).to(dev))
    if lobes:
        m.add_lobes(lobes, 20.0, rx)
    return m


def main():
    os.environ["GSPLAT_BWD_NO_GEOM"] = "1"
    torch.backends.cudnn.allow_tf32 = False
    dev = "cuda"
    meta = json.load(open(os.path.join(MV, "generation_meta.json")))
    vmin, span = meta["spec_min_db"], meta["spec_max_db"] - meta["spec_min_db"]
    train_names, test_names = T.ensure_split(MV)
    views = T.read_colmap_text(os.path.join(MV, "sparse", "0"))
    tv = torch.tensor(np.stack([views[n + ".png"][0] for n in test_names]))
    test_idx = T.spread_positions(tv, 8)
    trv = torch.tensor(np.stack([views[n + ".png"][0] for n in train_names]))
    train_idx = T.spread_positions(trv, 8)
    test = T.load_views(MV, [test_names[i] for i in test_idx], views, dev, want_float=True)
    train = T.load_views(MV, [train_names[i] for i in train_idx], views, dev, want_float=True)
    rx = torch.linalg.inv(train["viewmats"])[:, :3, 3]
    res = {}
    # a
    m0, m1 = model(0, dev, rx), model(1, dev, rx)
    with torch.no_grad():
        worst = 0.0
        for s in range(0, 8, 4):
            sl = slice(s, s + 4)
            a = m0.render_batch(test["viewmats"][sl], test["Ks"][sl], test["width"], test["height"], span)
            b = m1.render_batch(test["viewmats"][sl], test["Ks"][sl], test["width"], test["height"], span)
            worst = max(worst, float((a - b).abs().max()))
    res["a_init_max_abs_diff"] = worst; res["a"] = worst == 0.0
    # b and c: 300 steps on the 2 training positions
    opt = {k: torch.optim.Adam([p], lr=lr, eps=1e-15) for k, p, lr in
           [(k, p, {"sh0": 5e-3, "shN": 2.5e-4, "opacities": 0.1, "lobe_w": 5e-3, "lobe_axis": 2e-3, "lobe_logk": 2e-2}[k])
            for k, p in m1.params.items() if p.requires_grad]}
    target = lambda i: ((train["float"][i].float() - vmin) / span).clamp(0, 1)[None]    # noqa: E731
    losses, grads_ok = [], None
    for it in range(300):
        s = 4 * (it % 2); sl = slice(s, s + 4)
        img = m1.render_batch(train["viewmats"][sl], train["Ks"][sl], train["width"], train["height"], span)
        gt = torch.stack([target(i) for i in range(s, s + 4)])
        loss = 0.8 * l1_loss(img, gt) + 0.2 * (1 - ssim(img, gt))
        for o in opt.values():
            o.zero_grad(set_to_none=True)
        loss.backward()
        if it == 1:
            grads_ok = {k: (m1.params[k].grad is not None and bool(torch.isfinite(m1.params[k].grad).all())
                            and float(m1.params[k].grad.abs().max()) > 0) for k in ("lobe_w", "lobe_axis", "lobe_logk")}
        for o in opt.values():
            o.step()
        losses.append(float(loss))
    res["b_grads"] = grads_ok; res["b"] = all(grads_ok.values())
    res["c_loss_first_last"] = [losses[0], losses[-1]]
    res["c"] = bool(np.isfinite(losses).all() and losses[-1] < losses[0])
    # d: step time, SH3 vs SH3 + 1 lobe
    def step_ms(m):
        ops = {k: torch.optim.Adam([p], lr=1e-3) for k, p in m.params.items() if p.requires_grad}
        ts = []
        for it in range(25):
            st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            st.record()
            img = m.render_batch(train["viewmats"][:4], train["Ks"][:4], train["width"], train["height"], span)
            gt = torch.stack([target(i) for i in range(4)])
            loss = 0.8 * l1_loss(img, gt) + 0.2 * (1 - ssim(img, gt))
            for o in ops.values():
                o.zero_grad(set_to_none=True)
            loss.backward()
            for o in ops.values():
                o.step()
            en.record(); torch.cuda.synchronize()
            if it >= 5:
                ts.append(st.elapsed_time(en))
        return float(np.median(ts))
    t0, t1 = step_ms(model(0, dev, rx)), step_ms(model(1, dev, rx))
    res["d_step_ms_sh3_lobe1"] = [t0, t1]; res["d"] = t1 <= 1.3 * t0
    res["pass"] = all(res[k] for k in "abcd")
    print(json.dumps(res, indent=1))
    print("ALL PASS" if res["pass"] else "FAIL: " + ", ".join(k for k in "abcd" if not res[k]))


if __name__ == "__main__":
    main()
