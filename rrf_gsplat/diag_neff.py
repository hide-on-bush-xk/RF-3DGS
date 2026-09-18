"""Effective number of Gaussians per ray.

A pixel's value is sum_i w_i c_i with compositing weights w_i = alpha_i T_i.
N_eff = (sum_i w_i)^2 / sum_i w_i^2 says how many Gaussians a ray really
averages: 1 when one Gaussian owns the pixel, k when k share it equally.
It decides whether compositing in dB and compositing in linear power can
differ at all (they coincide for N_eff = 1), which is what research-state
section 5.4 asks.

gsplat does not expose per-Gaussian weights, so sum w_i^2 is estimated with
Rademacher colours: with c_i = +-1 drawn independently, E[(sum w_i c_i)^2]
= sum w_i^2. 256 channels x 4 draws = 1024 samples per pixel, relative
error about 4 %. sum w_i is the rendered alpha.

    (WSL, rf-gsplat) python rrf_gsplat/diag_neff.py --runs e2_mvdr_db a_mvdr_db_geom
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    from train_rrf import RRF, ensure_split, read_colmap_text
    from gsplat import rasterization
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=["e2_mvdr_db", "a_mvdr_db_geom"])
    ap.add_argument("--source", default=os.path.join(REPO, "RF-3DGS_dataset/regenerated/3dgs_MVDR_100_gpct"))
    ap.add_argument("--views", type=int, default=20)
    ap.add_argument("--channels", type=int, default=256)
    ap.add_argument("--draws", type=int, default=4)
    cfg = ap.parse_args()
    dev = torch.device("cuda")
    torch.manual_seed(0)
    views = read_colmap_text(os.path.join(cfg.source, "sparse", "0"))
    _, test = ensure_split(cfg.source)
    names = test[:: max(1, len(test) // cfg.views)][: cfg.views]
    out = {}
    for run in cfg.runs:
        rdir = os.path.join(REPO, "output/rrf", run)
        rc = json.load(open(os.path.join(rdir, "results.json")))["config"]
        model = RRF(rc["checkpoint"], rc["mode"], 1 if rc["mode"] in ("db", "power") else 3, rc["sh_degree"], dev)
        model.load_state(torch.load(os.path.join(rdir, "rrf_state.pt"), map_location=dev))
        n = model.means.shape[0]
        neff_all, alpha_mean = [], []
        with torch.no_grad():
            for name in names:
                view, K, w, h = views[name + ".png"]
                viewmat, Km = torch.from_numpy(view).to(dev), torch.from_numpy(K).to(dev)
                s2 = None
                for _ in range(cfg.draws):
                    r = (torch.randint(0, 2, (n, cfg.channels), device=dev) * 2 - 1).float()
                    img, alpha, _ = rasterization(model.means, model.quats, model.scales, model.opacities, r,
                                                  viewmat[None], Km[None], w, h, sh_degree=None,
                                                  backgrounds=torch.zeros(1, cfg.channels, device=dev))
                    m = (img[0] ** 2).mean(-1)                                  # [H, W] ~ sum w_i^2
                    s2 = m if s2 is None else s2 + m
                s2 = s2 / cfg.draws
                a = alpha[0, ..., 0]                                            # sum w_i
                neff = a ** 2 / s2.clamp_min(1e-12)
                neff_all.append(neff[a > 0.5])
                alpha_mean.append(float(a.mean()))
        ne = torch.cat(neff_all)
        q = torch.quantile(ne, torch.tensor([0.1, 0.5, 0.9], device=dev))
        out[run] = {"gaussians": n, "views": len(names), "pixels_alpha_gt_0.5": int(ne.numel()),
                    "share_pixels_alpha_gt_0.5": float(ne.numel() / (len(names) * w * h)),
                    "mean_alpha": float(sum(alpha_mean) / len(alpha_mean)),
                    "neff_p10": float(q[0]), "neff_median": float(q[1]), "neff_p90": float(q[2]),
                    "neff_mean": float(ne.mean()), "share_neff_lt_1_5": float((ne < 1.5).float().mean()),
                    "share_neff_lt_2": float((ne < 2).float().mean()), "share_neff_gt_5": float((ne > 5).float().mean())}
        o = out[run]
        print(f"{run}: N_eff per ray over {o['pixels_alpha_gt_0.5']:,} pixels with alpha > 0.5 "
              f"({o['share_pixels_alpha_gt_0.5']:.0%} of {len(names)} views): P10 {o['neff_p10']:.2f}, median {o['neff_median']:.2f}, "
              f"P90 {o['neff_p90']:.2f}, mean {o['neff_mean']:.2f}; share < 1.5: {o['share_neff_lt_1_5']:.1%}, < 2: {o['share_neff_lt_2']:.1%}, "
              f"> 5: {o['share_neff_gt_5']:.1%}; mean alpha {o['mean_alpha']:.3f}")
    json.dump(out, open(os.path.join(REPO, "output/rrf/diag_neff.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
