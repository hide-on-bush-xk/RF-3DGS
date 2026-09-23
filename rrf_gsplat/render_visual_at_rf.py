"""Render the stage-1 visual model at the RF receiver poses.

Why this exists
---------------
The RF datasets and the visual dataset are two different captures of the same
NIST lobby: 3200 receiver views for the spectra, 809 camera views for the
geometry.  Their poses do not coincide, so there is no visual image sitting
beside a spectrum that could just be looked up.

But the two live in the same world frame -- that is the whole premise of the
two-stage training, where stage 2 freezes the geometry stage 1 learned -- so the
visual model can simply be rendered at a receiver's pose.  The RF reconstruction
uses a single PINHOLE camera, 300x200 with f=150 (90.0 x 67.4 degrees), so the
result is pixel-aligned in direction with the spectrum: whatever sits at pixel
(u, v) of the spectrum is the surface at pixel (u, v) of this render.

All six spectrum datasets share one set of poses and one split -- images.txt,
cameras.txt, train_index.txt and test_index.txt are byte-identical across
3dgs_{AoD,CBF,Delay,MPC,MVDR,TCBF}_100 -- so ONE run of this script serves all
six, and the output is written to a single directory rather than once per
spectrum.

Output is named by the ORIGINAL view name (00005.png), which is what our own
runs use.  The released models number their renders by position in the sorted
held-out split instead; viewer.py holds that mapping.
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch
from PIL import Image

import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from gaussian_renderer import render
from scene.cameras import Camera
from scene.gaussian_model import GaussianModel
from utils.graphics_utils import focal2fov


class Pipe:
    """The three flags gaussian_renderer.render actually reads."""
    convert_SHs_python = False
    compute_cov3D_python = False
    debug = False


def qvec2rotmat(w, x, y, z):
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def read_colmap(sparse_dir):
    """(cameras by name, intrinsics) from a COLMAP text reconstruction.

    Returns {name: (R, T)} with R camera-to-world and T world-to-camera, the
    same convention readColmapCameras builds, plus (w, h, fx, fy).
    """
    cams = {}
    with open(os.path.join(sparse_dir, "images.txt")) as fid:
        for line in fid:
            p = line.split()
            # image lines carry a name in field 9; the POINTS2D lines that follow
            # each one do not, which is what this filter drops
            if len(p) < 10 or not p[9].lower().endswith(".png"):
                continue
            R = qvec2rotmat(*(float(v) for v in p[1:5])).T
            cams[os.path.splitext(p[9])[0]] = (R, np.array([float(v) for v in p[5:8]]))

    with open(os.path.join(sparse_dir, "cameras.txt")) as fid:
        for line in fid:
            if line.startswith("#") or not line.strip():
                continue
            p = line.split()
            if p[1] != "PINHOLE":
                raise ValueError(f"expected a PINHOLE camera, got {p[1]}")
            w, h, fx, fy = int(p[2]), int(p[3]), float(p[4]), float(p[5])
            break
    return cams, (w, h, fx, fy)


def newest_ply(model_dir):
    """The highest-iteration point cloud, compared numerically.

    Sorting these names as strings picks iteration_7000 over iteration_30000.
    """
    pc = os.path.join(model_dir, "point_cloud")
    best = None
    for name in os.listdir(pc):
        if not name.startswith("iteration_"):
            continue
        it = int(name.split("_")[1])
        path = os.path.join(pc, name, "point_cloud.ply")
        if os.path.isfile(path) and (best is None or it > best[0]):
            best = (it, path)
    if best is None:
        raise FileNotFoundError(f"no point_cloud.ply under {pc}")
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=os.path.join(REPO, "RF-3DGS_dataset", "blender_visual_trained"),
                    help="the trained stage-1 visual model")
    ap.add_argument("--source", default=os.path.join(REPO, "RF-3DGS_dataset", "training-rf-spectrum", "3dgs_MVDR_100"),
                    help="any RF dataset; all six carry identical poses")
    ap.add_argument("--out", default=os.path.join(REPO, "output", "visual_at_rf"))
    ap.add_argument("--split", default="test", choices=["test", "train", "all"])
    ap.add_argument("--white", action="store_true", help="white background instead of black")
    args = ap.parse_args()

    cams, (w, h, fx, fy) = read_colmap(os.path.join(args.source, "sparse", "0"))
    if args.split == "all":
        names = sorted(cams)
    else:
        idx = os.path.join(args.source, f"{args.split}_index.txt")
        names = sorted(l.strip() for l in open(idx) if l.strip())
    missing = [n for n in names if n not in cams]
    if missing:
        raise KeyError(f"{len(missing)} split names absent from the reconstruction, e.g. {missing[:3]}")

    it, ply = newest_ply(args.model)
    gs = GaussianModel(3)
    gs.load_ply(ply)
    bg = torch.ones(3, device="cuda") if args.white else torch.zeros(3, device="cuda")
    pipe = Pipe()

    os.makedirs(args.out, exist_ok=True)
    print(f"model      {os.path.relpath(args.model, REPO)}  iteration {it}  "
          f"{gs.get_xyz.shape[0]} gaussians")
    print(f"poses      {os.path.relpath(args.source, REPO)}  split={args.split}  {len(names)} views")
    print(f"camera     PINHOLE {w}x{h}  f={fx:g}")
    print(f"out        {os.path.relpath(args.out, REPO)}")

    # warm-up, excluded from the timing: the first launch pays CUDA context and
    # kernel compilation and is not representative
    R, T = cams[names[0]]
    with torch.no_grad():
        render(Camera(0, R, T, focal2fov(fx, w), focal2fov(fy, h),
                      torch.zeros((3, h, w)), None, names[0], 0), gs, pipe, bg)
    torch.cuda.synchronize()

    per_view = []
    t0 = time.time()
    for k, n in enumerate(names):
        R, T = cams[n]
        t = time.time()
        with torch.no_grad():
            img = render(Camera(0, R, T, focal2fov(fx, w), focal2fov(fy, h),
                                torch.zeros((3, h, w)), None, n, k), gs, pipe, bg)["render"]
        torch.cuda.synchronize()
        per_view.append(time.time() - t)
        a = (img.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
        Image.fromarray(a).save(os.path.join(args.out, f"{n}.png"))
        if (k + 1) % 160 == 0:
            print(f"  {k + 1}/{len(names)}")
    total = time.time() - t0

    per_view.sort()
    med = per_view[len(per_view) // 2]
    print(f"\nrendered {len(names)} views at {w}x{h} in {total:.1f}s wall-clock")
    print(f"  median {med * 1000:.1f} ms/view (warm-up excluded), "
          f"{total / len(names) * 1000:.1f} ms/view including PNG encode and disk")
    print(f"  a full spherical spectrum is 4 faces, so {med * 4 * 1000:.1f} ms of "
          f"rasterising per receiver")


if __name__ == "__main__":
    main()
