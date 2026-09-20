"""Expected depth of a visual 3DGS checkpoint at the RF receiver views (gsplat, WSL side).

Writes output/rrf/depth_gs_<tag>.npz: per held-out view the expected depth
(sum_i w_i z_i / alpha, camera z in metres, as the delay decomposition's "ED"
range term uses), alpha, the view matrix and K. depth_gt.py (Windows, Sionna
env) intersects the same rays with the scene mesh and compares.

    python rrf_gsplat/depth_gs.py --checkpoint <chkpnt.pth> --dataset RF-3DGS_dataset/regenerated/<ds> --tag lobby [--max-views 200]
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True); ap.add_argument("--dataset", required=True); ap.add_argument("--tag", required=True)
    ap.add_argument("--max-views", type=int, default=None); ap.add_argument("--out", default="output/rrf")
    cfg = ap.parse_args()
    from train_rrf import RRF, read_colmap_text, read_index
    from gsplat import rasterization
    device = "cuda"
    model = RRF(cfg.checkpoint, "db", 1, 0, device)
    views = read_colmap_text(os.path.join(cfg.dataset, "sparse", "0"))
    names = read_index(os.path.join(cfg.dataset, "test_index.txt"))
    if cfg.max_views:
        names = names[:: max(1, len(names) // cfg.max_views)][: cfg.max_views]
    depths, alphas, vms, ks = [], [], [], []
    colours = torch.zeros(model.means.shape[0], 1, device=device)
    with torch.no_grad():
        for n in names:
            view, K, w, h = views[n] if n in views else views[n + ".png"]      # index names carry no extension
            vm = torch.as_tensor(view, device=device); Kt = torch.as_tensor(K, device=device)
            img, alpha, _ = rasterization(model.means, model.quats, model.scales, model.opacities, colours, vm[None], Kt[None], w, h,
                                          sh_degree=None, backgrounds=torch.zeros(1, 1, device=device), render_mode="ED")
            depths.append(img[0, ..., 0].cpu().numpy().astype(np.float32)); alphas.append(alpha[0, ..., 0].cpu().numpy().astype(np.float32))
            vms.append(view); ks.append(K)
    os.makedirs(cfg.out, exist_ok=True)
    p = os.path.join(cfg.out, f"depth_gs_{cfg.tag}.npz")
    np.savez_compressed(p, names=np.array(names), depth=np.stack(depths), alpha=np.stack(alphas), viewmat=np.stack(vms), K=np.stack(ks))
    d = np.stack(depths); a = np.stack(alphas)
    print(f"{len(names)} views, {model.means.shape[0]:,} Gaussians; depth median {np.median(d[a > 0.5]):.2f} m, alpha > 0.5 on {(a > 0.5).mean():.1%} of pixels -> {p}")


if __name__ == "__main__":
    main()
