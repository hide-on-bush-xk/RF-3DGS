"""Render the visual dataset of a procedural scene with Mitsuba, in the
Blender / NeRF-synthetic layout the INRIA 3DGS trainer reads
(transforms_train.json + train/*.png, camera_angle_x = 90 deg, RGBA).

Poses: every route position x the four yaws the RF datasets use, plus random
poses inside the spaces (random yaw, small pitch) so the ceilings and the
upper walls are seen too. points3d.ply is sampled on the meshes with their
texture colour, which is what the trainer initialises the Gaussians from.

    PYTHONUTF8=1 python scene2/render_visual.py --scene scene2/corridor --out scene2/visual_dataset --spp 128 --extra 300
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np

VIEW_YAWS = (-math.pi / 2, 0.0, math.pi / 2, math.pi)


def c2w_from(pos, yaw, pitch=0.0):
    """Camera-to-world in the Blender convention (camera looks down -z, y up),
    looking along world +x rotated by yaw about z, then pitched."""
    forward = np.array([math.cos(yaw) * math.cos(pitch), math.sin(yaw) * math.cos(pitch), math.sin(pitch)])
    up_w = np.array([0.0, 0.0, 1.0])
    right = np.cross(forward, up_w); right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    m = np.eye(4); m[:3, 0] = right; m[:3, 1] = up; m[:3, 2] = -forward; m[:3, 3] = pos
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--width", type=int, default=1600); ap.add_argument("--height", type=int, default=900)
    ap.add_argument("--spp", type=int, default=128); ap.add_argument("--extra", type=int, default=300, help="random poses in addition to the route x 4 yaws")
    ap.add_argument("--route-every", type=int, default=2, help="use every k-th route position")
    ap.add_argument("--test-frac", type=float, default=0.05)
    ap.add_argument("--limit", type=int, default=None, help="render only the first N frames (smoke)")
    cfg = ap.parse_args()
    import mitsuba as mi
    mi.set_variant("cuda_ad_rgb")
    from PIL import Image
    layout = json.load(open(os.path.join(cfg.scene, "layout.json")))
    rx_z = layout["rx_z"]
    route = [(float(p[1]) / 1000, float(p[2]) / 1000) for p in (l.split() for l in open(os.path.join(cfg.scene, "rx_route.txt"))) if len(p) == 4]
    rng = np.random.default_rng(0)
    poses = []
    for (x, y) in route[:: cfg.route_every]:
        for yaw in VIEW_YAWS:
            poses.append(c2w_from(np.array([x, y, rx_z]), yaw, 0.0))
    spaces = layout["spaces"]
    for _ in range(cfg.extra):
        s = spaces[rng.integers(len(spaces))]
        pos = np.array([rng.uniform(s["x0"] + 0.5, s["x1"] - 0.5), rng.uniform(s["y0"] + 0.5, s["y1"] - 0.5), rng.uniform(layout["floor_z"] + 1.0, layout["ceil_z"] - 0.6)])
        poses.append(c2w_from(pos, rng.uniform(-math.pi, math.pi), rng.uniform(-0.35, 0.35)))
    if cfg.limit:
        poses = poses[: cfg.limit]
    n_test = int(round(cfg.test_frac * len(poses)))
    test_idx = set(rng.permutation(len(poses))[:n_test].tolist())
    os.makedirs(os.path.join(cfg.out, "train"), exist_ok=True); os.makedirs(os.path.join(cfg.out, "test"), exist_ok=True)
    scene = mi.load_file(os.path.join(cfg.scene, "corridor_visual.xml"), spp=cfg.spp, resx=cfg.width, resy=cfg.height)
    frames = {"train": [], "test": []}
    t0 = time.time()
    for i, c2w in enumerate(poses):
        # Mitsuba's camera frame is x LEFT, y up, z forward (perspective.cpp maps camera +x to the left half of
        # the film and +y to the top; look_at builds [left, up, dir]). Blender's is x right, y up, z backward:
        # flip the x and z columns. The first three datasets flipped y and z (the OpenCV frame), which is the
        # Blender frame rotated 180 deg about the optical axis: every frame was upside-down and mirrored with
        # respect to the pose the trainer reads, the ceiling lights sat in the bottom half, and 3DGS could not
        # fit the set (loss flat at 0.2, 17 dB). Checked with output/scene2_densify_test/conv_*.png: lights at
        # the top and the corridor mouth on the camera's left only with the x,z flip.
        # A fresh sensor per pose: mutating the XML sensor's to_world through traverse() is not relied upon.
        m = c2w.copy(); m[:3, 0] *= -1; m[:3, 2] *= -1
        sensor = mi.load_dict({"type": "perspective", "fov": 90.0, "fov_axis": "x", "near_clip": 0.05, "far_clip": 1000.0,
                               "to_world": mi.ScalarTransform4f(m.tolist()),
                               "sampler": {"type": "independent", "sample_count": cfg.spp},
                               "film": {"type": "hdrfilm", "width": cfg.width, "height": cfg.height, "pixel_format": "rgba",
                                        "rfilter": {"type": "gaussian"}}})
        img = mi.render(scene, sensor=sensor, spp=cfg.spp, seed=i)
        rgba = np.clip(np.asarray(mi.Bitmap(img).convert(mi.Bitmap.PixelFormat.RGBA, mi.Struct.Type.UInt8, srgb_gamma=True)), 0, 255)
        split = "test" if i in test_idx else "train"
        name = f"{split}/{i:04d}.png"
        Image.fromarray(rgba).save(os.path.join(cfg.out, name))
        frames[split].append({"file_path": name, "transform_matrix": c2w.tolist()})   # with extension, as the lobby's json has it
        if i % 50 == 0:
            print(f"  {i}/{len(poses)} frames, {time.time() - t0:.0f} s")
    fov_x = math.radians(90.0)
    for split in ("train", "test"):
        json.dump({"camera_angle_x": fov_x, "frames": frames[split]}, open(os.path.join(cfg.out, f"transforms_{split}.json"), "w"), indent=1)
    # points3d.ply: surface samples with their texture tint (the trainer's initial Gaussians)
    from make_corridor import box
    pts, cols = [], []
    tint = {"wall": (190, 184, 166), "floor": (115, 107, 102), "ceiling": (217, 217, 209)}
    for name, cls, b in layout["pieces"]:
        verts, _, tris = box(*b)
        v = np.array(verts)
        for a, bb, c in tris:
            area = 0.5 * np.linalg.norm(np.cross(v[bb] - v[a], v[c] - v[a]))
            n = max(1, int(area * 40))
            r1, r2 = rng.random(n), rng.random(n); s = np.sqrt(r1)
            p = (1 - s)[:, None] * v[a] + (s * (1 - r2))[:, None] * v[bb] + (s * r2)[:, None] * v[c]
            pts.append(p); cols.append(np.tile(np.array(tint[cls], np.uint8), (n, 1)))
    pts, cols = np.concatenate(pts), np.concatenate(cols)
    with open(os.path.join(cfg.out, "points3d.ply"), "w") as f:
        # the trainer's fetchPly wants x y z nx ny nz red green blue (it silently falls back to None otherwise)
        f.write(f"ply\nformat ascii 1.0\nelement vertex {len(pts)}\nproperty float x\nproperty float y\nproperty float z\n"
                "property float nx\nproperty float ny\nproperty float nz\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        for p, c in zip(pts, cols):
            f.write(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f} 0 0 0 {c[0]} {c[1]} {c[2]}\n")
    print(f"{len(frames['train'])} train + {len(frames['test'])} test frames, {len(pts):,} initial points, {time.time() - t0:.0f} s -> {cfg.out}")


if __name__ == "__main__":
    main()
