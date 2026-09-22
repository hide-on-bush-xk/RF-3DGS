"""A leave-one-region split: hold out a contiguous piece of the route.

The random 20 % split leaves every held-out position 0.23 m from a training
one, so it measures interpolation, not the "extrapolation to unvisited
locations" the paper credits the visual geometry with. This makes a sibling
dataset (hard links, no copies of the spectra) whose test set is every
position inside a region and whose training set is everything else.

    PYTHONUTF8=1 python rrf_gsplat/make_region_split.py --source RF-3DGS_dataset/regenerated/3dgs_MULTI_24ghz_tut_cs \
        --out RF-3DGS_dataset/regenerated/3dgs_MULTI_24ghz_tut_cs_region --xmin 4 --ymax -7
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO, "rrf_gsplat"))


def main():
    """Write the sibling dataset and report how far the held-out region really is."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True); ap.add_argument("--out", required=True)
    # Defaults are effectively unbounded, so naming one edge is enough to cut a half-plane.
    ap.add_argument("--xmin", type=float, default=-1e9); ap.add_argument("--xmax", type=float, default=1e9)
    ap.add_argument("--ymin", type=float, default=-1e9); ap.add_argument("--ymax", type=float, default=1e9)
    cfg = ap.parse_args()
    from eval_baselines import read_poses
    poses = read_poses(os.path.join(cfg.source, "sparse", "0", "images.txt"))
    names = sorted(poses)
    inside = lambda n: (cfg.xmin <= poses[n][0][0] <= cfg.xmax) and (cfg.ymin <= poses[n][0][1] <= cfg.ymax)
    test = [n for n in names if inside(n)]; train = [n for n in names if not inside(n)]
    # A position contributes four yaw views. If a split broke a position apart,
    # three of its faces would be in training and the fourth "held out" from
    # one metre away -- which is exactly the leak this split exists to avoid.
    assert len(test) % 4 == 0 and len(train) % 4 == 0, "a position's four views must stay together"
    for sub in ("images", "spectra_float"):
        os.makedirs(os.path.join(cfg.out, sub), exist_ok=True)
        for f in os.listdir(os.path.join(cfg.source, sub)):
            dst = os.path.join(cfg.out, sub, f)
            # Hard links: the spectra are large and identical, only the split differs.
            if not os.path.exists(dst):
                os.link(os.path.join(cfg.source, sub, f), dst)
    os.makedirs(os.path.join(cfg.out, "sparse", "0"), exist_ok=True)
    # Poses and meta are copied rather than linked: they are small, and a copy
    # cannot be corrupted by an edit to the source dataset.
    for f in os.listdir(os.path.join(cfg.source, "sparse", "0")):
        shutil.copy(os.path.join(cfg.source, "sparse", "0", f), os.path.join(cfg.out, "sparse", "0", f))
    shutil.copy(os.path.join(cfg.source, "generation_meta.json"), os.path.join(cfg.out, "generation_meta.json"))
    # The only real difference between the two datasets.
    open(os.path.join(cfg.out, "train_index.txt"), "w").write("\n".join(train) + "\n")
    open(os.path.join(cfg.out, "test_index.txt"), "w").write("\n".join(test) + "\n")
    # how far is each held-out position from the nearest training one, per yaw (the copy baseline's reach)
    # Filtered to one yaw so positions are counted once; this distance is the
    # number that says whether the split is really extrapolation.
    tp = np.array([poses[n][0] for n in train if poses[n][1] == 0]); d = [np.linalg.norm(tp - poses[n][0], axis=1).min() for n in test if poses[n][1] == 0]
    print(f"region x in [{cfg.xmin}, {cfg.xmax}], y in [{cfg.ymin}, {cfg.ymax}]: {len(test)//4} held-out positions ({len(test)} views), "
          f"{len(train)//4} training positions; nearest training position from a held-out one: median {np.median(d):.2f} m, "
          f"min {np.min(d):.2f}, max {np.max(d):.2f} m -> {cfg.out}")


if __name__ == "__main__":
    main()
