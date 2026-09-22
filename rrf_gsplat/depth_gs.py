"""Expected depth of a visual 3DGS checkpoint at the RF receiver views (gsplat, WSL side).

Writes output/rrf/depth_gs_<tag>.npz: per held-out view the expected depth
(sum_i w_i z_i / alpha, camera z in metres, as the delay decomposition's "ED"
range term uses), alpha, the view matrix and K. depth_gt.py (Windows, Sionna
env) intersects the same rays with the scene mesh and compares.

Two halves because of the toolchain split: gsplat only builds under WSL, and
Sionna's OptiX only works on the Windows side, so the view matrices travel
through the npz rather than the two ever running in one process.

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
    """Rasterise depth for each held-out view and save it with its camera."""
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
    if cfg.max_views and cfg.max_views < len(names):
        # stratified, not a stride: the index lists a position's four yaws consecutively, so pick positions evenly
        # along the route and keep all four faces of each (the smoke rule: never [::k] over views)
        per = 4; n_pos = len(names) // per; keep = max(1, cfg.max_views // per)
        pos = np.linspace(0, n_pos - 1, keep).round().astype(int)
        names = [names[p * per + k] for p in pos for k in range(per)]
    depths, depths_d, alphas, vms, ks = [], [], [], [], []
    # Depth does not depend on appearance, so a single zero colour channel is
    # enough and the SH evaluation is skipped entirely.
    colours = torch.zeros(model.means.shape[0], 1, device=device)
    with torch.no_grad():
        for n in names:
            view, K, w, h = views[n] if n in views else views[n + ".png"]      # index names carry no extension
            vm = torch.as_tensor(view, device=device); Kt = torch.as_tensor(K, device=device)
            # Two passes because the two modes answer different questions: ED is
            # normalised by alpha (the depth of the surface that was hit), D is
            # not (and is therefore shortened wherever alpha < 1).
            img, alpha, _ = rasterization(model.means, model.quats, model.scales, model.opacities, colours, vm[None], Kt[None], w, h,
                                          sh_degree=None, backgrounds=torch.zeros(1, 1, device=device), render_mode="ED")
            img_d, _, _ = rasterization(model.means, model.quats, model.scales, model.opacities, colours, vm[None], Kt[None], w, h,
                                        sh_degree=None, backgrounds=torch.zeros(1, 1, device=device), render_mode="D")
            depths.append(img[0, ..., 0].cpu().numpy().astype(np.float32)); depths_d.append(img_d[0, ..., 0].cpu().numpy().astype(np.float32))
            alphas.append(alpha[0, ..., 0].cpu().numpy().astype(np.float32))
            # The camera travels with the depth, so depth_gt.py can cast exactly
            # these rays without re-reading the dataset.
            vms.append(view); ks.append(K)
    os.makedirs(cfg.out, exist_ok=True)
    p = os.path.join(cfg.out, f"depth_gs_{cfg.tag}.npz")
    # depth = expected depth (ED, sum w z / alpha); depth_d = accumulated depth (D, sum w z), the density runs' range term
    np.savez_compressed(p, names=np.array(names), depth=np.stack(depths), depth_d=np.stack(depths_d), alpha=np.stack(alphas), viewmat=np.stack(vms), K=np.stack(ks))
    d = np.stack(depths); a = np.stack(alphas)
    # Summarised over confident pixels only: expected depth where alpha is near
    # zero is a ratio of two small numbers and says nothing.
    print(f"{len(names)} views, {model.means.shape[0]:,} Gaussians; depth median {np.median(d[a > 0.5]):.2f} m, alpha > 0.5 on {(a > 0.5).mean():.1%} of pixels -> {p}")


if __name__ == "__main__":
    main()
