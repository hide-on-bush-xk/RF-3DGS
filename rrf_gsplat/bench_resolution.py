"""Rasteriser speed against resolution: gsplat and the INRIA rasteriser on the same Gaussians.

One training step (render + L1 loss + backward into SH and opacity) of the
visual checkpoint's 1.01M Gaussians at 300x200, 600x400 and 1200x800.
Runs wherever one of the two rasterisers imports:

    WSL     /home/ke/miniconda3/envs/rf-gsplat/bin/python rrf_gsplat/bench_resolution.py --which gsplat
    Windows C:/Users/Ke/miniconda3/envs/rf-3dgs/python.exe rrf_gsplat/bench_resolution.py --which inria

Writes output/rrf/bench_resolution_<which>.json.

Several resolutions on purpose. At 300x200 the step is dominated by kernel
launch and Python overhead rather than by rasterisation, so a single small
size would compare the harness, not the rasteriser.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)


def load_gaussians(device):
    """The visual checkpoint's Gaussians, with fresh zero SH and trainable opacity.

    Both rasterisers get the identical parameter tensors, which is what makes
    the comparison a comparison. The SH is zeroed rather than loaded: the
    timing does not depend on the values, only on the shapes.
    """
    (m, it) = torch.load(os.path.join(REPO, "RF-3DGS_dataset/blender_visual_trained/chkpnt30000.pth"),
                         weights_only=False, map_location="cpu")
    (_, xyz, f_dc, f_rest, scaling, rotation, opacity, *_) = m
    g = dict(means=xyz.detach().to(device), scaling=scaling.detach().to(device),
             rotation=rotation.detach().to(device), opacity=opacity.detach().to(device))
    n = xyz.shape[0]
    g["sh"] = torch.zeros(n, 16, 3, device=device, requires_grad=True)          # [N,K,3]
    g["opacity"].requires_grad_(True)
    return g


def camera(width, height, device):
    """A real pose at the requested resolution, 90 degrees horizontal."""
    # the first training view of the regenerated MVDR data (a real pose)
    # A real pose matters: a synthetic one could put the whole scene off screen
    # and make the rasteriser look fast for the wrong reason.
    view = torch.tensor([[0.0, -1.0, 0.0, 0.5319], [0.0, 0.0, -1.0, 0.0282], [1.0, 0.0, 0.0, -3.1997], [0, 0, 0, 1]],
                        device=device)
    f = (width / 2) / math.tan(math.radians(90) / 2)
    K = torch.tensor([[f, 0, width / 2], [0, f, height / 2], [0, 0, 1]], device=device)
    return view, K


def step_gsplat(g, view, K, width, height):
    """One gsplat step. Activations are applied here so both sides do the same work."""
    from gsplat import rasterization
    img, _, _ = rasterization(g["means"], torch.nn.functional.normalize(g["rotation"], dim=-1),
                              torch.exp(g["scaling"]), torch.sigmoid(g["opacity"][:, 0]), g["sh"],
                              view[None], K[None], width, height, sh_degree=3,
                              backgrounds=torch.zeros(1, 3, device=view.device))
    # A trivial loss: the point is to exercise the backward pass, not to train.
    loss = img.abs().mean()
    loss.backward()


def step_inria(g, view, K, width, height):
    """The same step through the INRIA rasteriser, from the same K and view.

    The extra setup is convention translation, not extra work: INRIA wants FoV
    tangents and transposed matrices where gsplat takes K directly.
    """
    from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
    from utils.graphics_utils import getProjectionMatrix
    fovx = fovy = 2 * math.atan((width / 2) / float(K[0, 0]))
    fovy = 2 * math.atan((height / 2) / float(K[1, 1]))
    world_view = view.transpose(0, 1)                                    # INRIA stores transposed
    proj = getProjectionMatrix(0.01, 100.0, fovx, fovy).transpose(0, 1).to(view.device)
    full_proj = (world_view.unsqueeze(0).bmm(proj.unsqueeze(0))).squeeze(0)
    cam_center = world_view.inverse()[3, :3]
    settings = GaussianRasterizationSettings(
        image_height=height, image_width=width, tanfovx=math.tan(fovx * 0.5), tanfovy=math.tan(fovy * 0.5),
        bg=torch.zeros(3, device=view.device), scale_modifier=1.0, viewmatrix=world_view, projmatrix=full_proj,
        sh_degree=3, campos=cam_center, prefiltered=False, debug=False)
    rast = GaussianRasterizer(raster_settings=settings)
    # INRIA requires the screen-space gradient tensor even when it is unused.
    means2d = torch.zeros_like(g["means"], requires_grad=True)
    out = rast(means3D=g["means"], means2D=means2d, shs=g["sh"], colors_precomp=None,
               opacities=torch.sigmoid(g["opacity"]), scales=torch.exp(g["scaling"]),
               rotations=torch.nn.functional.normalize(g["rotation"], dim=-1), cov3D_precomp=None)
    # The return shape differs between forks; both are handled.
    img = out[0] if isinstance(out, tuple) else out
    loss = img.abs().mean()
    loss.backward()


def main():
    """Time one rasteriser across the resolution ladder and save the result."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--which", choices=["gsplat", "inria"], required=True)
    ap.add_argument("--reps", type=int, default=20)
    cfg = ap.parse_args()
    device = torch.device("cuda")
    g = load_gaussians(device)
    step = step_gsplat if cfg.which == "gsplat" else step_inria
    rows = []
    for width, height in ((300, 200), (600, 400), (1200, 800), (2400, 1600)):
        view, K = camera(width, height, device)
        for _ in range(3):                                   # warm-up, excluded
            # Gradients are cleared by hand each time: without this they would
            # accumulate and the backward cost would drift upward.
            step(g, view, K, width, height); g["sh"].grad = None; g["opacity"].grad = None
        times = []
        for _ in range(cfg.reps):
            # synchronize on both sides, or this would time the CUDA queue.
            torch.cuda.synchronize(); t0 = time.time()
            step(g, view, K, width, height); g["sh"].grad = None; g["opacity"].grad = None
            torch.cuda.synchronize(); times.append((time.time() - t0) * 1000)
        times.sort()
        ms = times[len(times) // 2]                          # median, per the reporting contract
        # min and max are recorded too, so a noisy measurement is visible rather
        # than hidden behind the median.
        rows.append({"width": width, "height": height, "ms_per_step_median": ms, "ms_per_step_mean": sum(times) / len(times),
                     "ms_min": times[0], "ms_max": times[-1], "it_per_s": 1000 / ms, "reps": cfg.reps})
        print(f"{cfg.which:7s} {width}x{height}: median {ms:6.1f} ms/step ({1000/ms:5.1f} it/s); mean {sum(times)/len(times):6.1f}, "
              f"min {times[0]:.1f}, max {times[-1]:.1f} over {cfg.reps} after 3 warm-ups")
    os.makedirs(os.path.join(REPO, "output", "rrf"), exist_ok=True)
    # The Gaussian count travels with the timings: they are meaningless without it.
    json.dump({"which": cfg.which, "gaussians": int(g["means"].shape[0]), "rows": rows},
              open(os.path.join(REPO, "output", "rrf", f"bench_resolution_{cfg.which}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
