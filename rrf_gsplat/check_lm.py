"""Smoke for the LM phase (lm_optim.py, train_rrf.py --lm-after): the linear algebra before any timing.

On the MVDR dataset (db, SH3, frozen geometry), after 300 Adam steps of the train_rrf.py configuration, with three
receiver positions stratified over the training set (first, middle, last):
  L1  J p from linearity equals the finite difference (f(theta + e p) - f(theta)) / e, relative L2 <= 1e-3
  L2  J^T J is symmetric: |<p, JtJ q> - <q, JtJ p>| / |<p, JtJ q>| <= 1e-4
  L3  the residuals are the loss: sum(r^2) (less the eps terms) equals the training loss summed over pixels and
      faces, relative <= 1e-3
  L4  J^T r equals autograd's gradient of 0.5 sum(r^2) w.r.t. the SH coefficients, relative L2 <= 1e-4
  L5  known failure: the PCG step x is a descent direction for -x, and the flipped step (theta + g x) raises the
      cost on the subset for a small g
  L6  one LM step (3DGS-LM's settings) lowers the cost on its line-search positions or is undone by the trust
      region; its time is reported (no criterion: the timing belongs to the full comparison)
Written before the first run. Run 1 (cuDNN's default TF32) FAILED L2 (3.1e-3) and L3 (1.2e-2): with TF32 the SSIM
convolutions put a quarter of the pixels above SSIM 1 (the residuals' clamp then breaks L3) and make the residual's
jvp and vjp disagree (L2); the test also summed its inner products in float32. Run 2 runs as train_rrf.py runs LM
(--lm-after forces cuDNN TF32 off) and sums the inner products in float64; the criteria are unchanged.

    C:/Users/Ke/miniconda3/envs/rf-gsplat-win/python.exe rrf_gsplat/check_lm.py
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import train_rrf as T  # noqa: E402
import lm_optim as L  # noqa: E402
from check_gsplat_port import MV, CK  # noqa: E402
from utils.loss_utils import l1_loss, ssim  # noqa: E402


def main():
    dev = torch.device("cuda")
    torch.backends.cudnn.allow_tf32 = False          # as train_rrf.py --lm-after runs
    meta = json.load(open(os.path.join(MV, "generation_meta.json")))
    views = T.read_colmap_text(os.path.join(MV, "sparse", "0"))
    names = T.read_index(os.path.join(MV, "train_index.txt"))
    span, lo = meta["spec_max_db"] - meta["spec_min_db"], meta["spec_min_db"]
    lam = 0.2
    m = T.RRF(CK, "db", 1, 3, dev)
    m.sh_backend = "gsplat"
    npos = len(names) // 4
    data = T.load_views(MV, names, views, dev, want_float=True)

    def views_of(g):
        return data["viewmats"][g], data["Ks"][g], data["width"], data["height"]

    def targets_of(g):
        return ((data["float"][g].float() - lo) / span).clamp(0, 1)[:, None]

    # 300 Adam steps as train_rrf.py takes them (4 faces a step, lr x 2)
    opts = {k: torch.optim.Adam([p], lr=v * 2, eps=1e-15) for k, p, v in
            (("sh0", m.params["sh0"], 0.0025), ("shN", m.params["shN"], 0.0025 / 20), ("opacities", m.params["opacities"], 0.025))}
    rng = np.random.default_rng(0)
    for _ in range(300):
        p = int(rng.integers(npos)); g = list(range(4 * p, 4 * p + 4))
        out = m.render_batch(*views_of(g), span); y = targets_of(g)
        loss = sum((1 - lam) * l1_loss(out[k], y[k]) + lam * (1 - ssim(out[k][None], y[k][None])) for k in range(4)) / 4
        for o in opts.values():
            o.zero_grad(set_to_none=True)
        loss.backward()
        for o in opts.values():
            o.step()
    groups = [list(range(4 * p, 4 * p + 4)) for p in range(npos)]
    pick = [groups[0], groups[npos // 2], groups[-1]]
    win = L.ssim_window(dev)
    rep, ok = {}, True
    G = [L.Group(m, views_of(g), span, targets_of(g), win, lam) for g in pick]
    for Gi in G:
        Gi.linearise()
    gen = torch.Generator(device="cpu").manual_seed(1)
    rnd = lambda t: torch.randn(t.shape, generator=gen).to(dev) * float(t.abs().mean() + 1e-3)  # noqa: E731
    p0, pN = rnd(m.params["sh0"]), rnd(m.params["shN"])
    q0, qN = rnd(m.params["sh0"]), rnd(m.params["shN"])

    # L1
    e = 1e-2
    Gi = G[0]
    with torch.no_grad():
        f0 = Gi.render()
        with L.coefficients(m, m.params["sh0"].data + e * p0, m.params["shN"].data + e * pN):
            f1 = Gi.render()
    fd = (f1 - f0) / e; jp = Gi.Jp(p0, pN)
    rep["L1_Jp_vs_fd"] = float((jp - fd).norm() / fd.norm()); c1 = rep["L1_Jp_vs_fd"] <= 1e-3
    # L2
    Jq = [Gi.jtj(q0, qN) for Gi in G]; Jp_ = [Gi.jtj(p0, pN) for Gi in G]
    a = sum(float((p0.double() * j[0].double()).sum() + (pN.double() * j[1].double()).sum()) for j in Jq)
    b = sum(float((q0.double() * j[0].double()).sum() + (qN.double() * j[1].double()).sum()) for j in Jp_)
    rep["L2_symmetry"] = abs(a - b) / abs(a); c2 = rep["L2_symmetry"] <= 1e-4
    # L3
    with torch.no_grad():
        tot_r, tot_l = 0.0, 0.0
        for Gi in G:
            f = Gi.render(); n = f[0].numel()
            tot_r += float((L.residuals(f, Gi.y, win, lam) ** 2).sum()) - 2 * L.EPS * f.numel()
            tot_l += float(sum((1 - lam) * l1_loss(f[k], Gi.y[k]) + lam * (1 - ssim(f[k][None], Gi.y[k][None]))
                               for k in range(4))) * n
    rep["L3_residuals_vs_loss"] = abs(tot_r - tot_l) / tot_l; c3 = rep["L3_residuals_vs_loss"] <= 1e-3
    # L4
    b0 = torch.zeros_like(m.params["sh0"]); bN = torch.zeros_like(m.params["shN"])
    for Gi in G:
        g0, gN = Gi.gradient(); b0 += g0; bN += gN
    for p in m.params.values():
        p.grad = None
    cost = sum(0.5 * (L.residuals(Gi.render(), Gi.y, win, lam) ** 2).sum() for Gi in G)
    cost.backward()
    a0, aN = m.params["sh0"].grad, m.params["shN"].grad
    rep["L4_JTr_vs_autograd"] = float(torch.sqrt(((b0 - a0) ** 2).sum() + ((bN - aN) ** 2).sum())
                                      / torch.sqrt((a0 ** 2).sum() + (aN ** 2).sum()))
    c4 = rep["L4_JTr_vs_autograd"] <= 1e-4
    # L5
    d0 = torch.zeros_like(b0); dN = torch.zeros_like(bN)
    for Gi in G:
        x0_, xN_ = Gi.diag(T.sh_basis, 3); d0 += x0_; dN += xN_
    C0, CN = d0.clamp(1, 1e6), dN.clamp(1, 1e6)

    def Aop(p):
        a0_, aN_ = C0 / 1e-3 * p[0], CN / 1e-3 * p[1]
        for Gi in G:
            j0, jN = Gi.jtj(p[0], p[1]); a0_ = a0_ + j0; aN_ = aN_ + jN
        return (a0_, aN_)
    x, rel = L.pcg(Aop, (b0, bN), (1 / (d0 + C0), 1 / (dN + CN)), 8, 5e-2)
    rep["L5_descent_inner"] = float((b0 * x[0]).sum() + (bN * x[1]).sum())
    base = sum(0.5 * float((Gi.r ** 2).sum()) for Gi in G)
    s = 0.1 / max(float(x[0].abs().max()), float(x[1].abs().max()))
    with torch.no_grad():
        prev0, prevN = m.params["sh0"].data.clone(), m.params["shN"].data.clone()
        m.params["sh0"].data += s * x[0]; m.params["shN"].data += s * x[1]
        up = sum(0.5 * float((L.residuals(Gi.render(), Gi.y, win, lam) ** 2).sum()) for Gi in G)
        m.params["sh0"].data.copy_(prev0 - s * x[0]); m.params["shN"].data.copy_(prevN - s * x[1])
        down = sum(0.5 * float((L.residuals(Gi.render(), Gi.y, win, lam) ** 2).sum()) for Gi in G)
        m.params["sh0"].data.copy_(prev0); m.params["shN"].data.copy_(prevN)
    rep["L5_cost"] = {"base": base, "flipped": up, "descent": down, "pcg_rel_residual": rel}
    c5 = rep["L5_descent_inner"] > 0 and up > base and down < base
    del G
    torch.cuda.empty_cache()
    # L6
    lm = L.LM(m, groups, views_of, targets_of, span, lam, T.sh_basis,
              {"subset_positions": 6, "subsets": 4, "pcg_iters": 8, "pcg_rtol": 5e-2, "radius": 1e-3, "radius_min": 1e-4,
               "radius_max": 1e-2, "linesearch_frac": 0.3, "min_diag": 1.0, "max_diag": 1e6, "max_step": 10.0,
               "gamma0": 1.0, "gamma_alpha": 0.7, "min_rel_decrease": 1e-5})
    torch.cuda.synchronize(); t0 = time.time()
    rec = lm.step(1)
    torch.cuda.synchronize()
    rep["L6_step"] = rec | {"wall_seconds": time.time() - t0}
    rep["pass"] = {"L1": c1, "L2": c2, "L3": c3, "L4": c4, "L5": c5}
    ok = c1 and c2 and c3 and c4 and c5
    print(json.dumps(rep, indent=1, default=float))
    print("ALL PASS" if ok else "FAIL")
    os.makedirs(os.path.join(T.REPO, "output", "rrf", "lm_check"), exist_ok=True)
    json.dump(rep, open(os.path.join(T.REPO, "output", "rrf", "lm_check", "check_lm.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
