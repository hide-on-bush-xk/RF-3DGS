"""Where does a training step go? Times the pieces of one train_rrf.py iteration.

    /home/ke/miniconda3/envs/rf-gsplat/bin/python rrf_gsplat/profile_step.py
"""

from __future__ import annotations

import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_rrf import REPO, RRF, load_views, read_colmap_text, read_index  # noqa: E402
from utils.loss_utils import l1_loss, ssim  # noqa: E402


def timed(fn, n=20):
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(n):
        out = fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / n * 1000, out


def main():
    device = torch.device("cuda")
    src = os.path.join(REPO, "RF-3DGS_dataset/regenerated/3dgs_MVDR_100_gpct")
    views = read_colmap_text(os.path.join(src, "sparse", "0"))
    names = read_index(os.path.join(src, "train_index.txt"))[:8]
    data = load_views(src, names, views, device, want_float=True)
    for mode, ch in (("rgb", 3), ("db", 1)):
        model = RRF(os.path.join(REPO, "RF-3DGS_dataset/blender_visual_trained/chkpnt30000.pth"),
                    mode, ch, 3, device)
        vm, K = data["viewmats"][0], data["Ks"][0]
        cam = torch.linalg.inv(vm)[:3, 3]
        t_sh, col = timed(lambda: model.colours(cam))
        from gsplat import rasterization
        def fwd():
            return rasterization(model.means, model.quats, model.scales,
                                 torch.sigmoid(model.opacity_logit[:, 0]), col.detach(),
                                 vm[None], K[None], data["width"], data["height"], sh_degree=None,
                                 backgrounds=torch.zeros(1, ch, device=device))[0]
        t_fwd, img = timed(fwd)
        gt = data["rgb"][0].float() / 255.0 if mode == "rgb" else data["float"][0].float()[None]
        def full():
            im = model.render(vm, K, data["width"], data["height"], 74.0)
            loss = 0.8 * l1_loss(im, gt) + 0.2 * (1 - ssim(im[None], gt[None]))
            loss.backward()
            return loss
        t_full, _ = timed(full, n=10)
        def fwd_packed_false():
            return rasterization(model.means, model.quats, model.scales,
                                 torch.sigmoid(model.opacity_logit[:, 0]), col.detach(),
                                 vm[None], K[None], data["width"], data["height"], sh_degree=None,
                                 packed=False, backgrounds=torch.zeros(1, ch, device=device))[0]
        t_fwd2, _ = timed(fwd_packed_false)
        n_vis = int((rasterization(model.means, model.quats, model.scales,
                                   torch.sigmoid(model.opacity_logit[:, 0]), col.detach(),
                                   vm[None], K[None], data["width"], data["height"], sh_degree=None)[2]["radii"] > 0).sum())
        print(f"{mode}: SH eval {t_sh:.1f} ms | raster fwd {t_fwd:.1f} ms (packed) / {t_fwd2:.1f} ms (unpacked) "
              f"| full step fwd+bwd {t_full:.1f} ms | visible {n_vis:,} of {model.means.shape[0]:,}")
        if mode == "rgb":
            sh = model.sh                                                   # [N,K,3]
            def fwd_sh():
                return rasterization(model.means, model.quats, model.scales,
                                     torch.sigmoid(model.opacity_logit[:, 0]), sh,
                                     vm[None], K[None], data["width"], data["height"], sh_degree=3,
                                     backgrounds=torch.zeros(1, 3, device=device))[0]
            t_sh_cuda, _ = timed(fwd_sh)
            print(f"rgb: raster fwd with gsplat's own SH (sh_degree=3): {t_sh_cuda:.1f} ms")


if __name__ == "__main__":
    main()
