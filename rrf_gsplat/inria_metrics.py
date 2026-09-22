"""Score a gsplat run with RF-3DGS's own metrics.py (PSNR / SSIM / VGG-LPIPS, per image, averaged).

Stages the run's saved jet-RGB renders and the dataset's ground-truth PNGs in
the INRIA layout (<run>/test/ours_<it>/{renders,gt}) and calls the fork's
metrics.py in the rf-3dgs environment, so every row of the SOTA table is
scored by the identical implementation the published numbers came from.
Refuses a run whose renders do not cover the whole held-out set.

Using the published implementation rather than a reimplementation is the point:
PSNR averaged per image differs from PSNR of the pooled error, and a table that
mixes the two conventions is not a comparison.

    PYTHONUTF8=1 python rrf_gsplat/inria_metrics.py --run output/rrf/<run> --source RF-3DGS_dataset/training-rf-spectrum/3dgs_MVDR_100
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# The 3DGS fork lives in its own environment; this script may run from another.
PY_3DGS = r"C:\Users\Ke\miniconda3\envs\rf-3dgs\python.exe"


def main():
    """Stage the renders into the INRIA layout, run metrics.py, keep both results."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True); ap.add_argument("--source", required=True)
    ap.add_argument("--iteration", type=int, default=None, help="label of the staged folder (default: the run's iterations)")
    ap.add_argument("--allow-partial", action="store_true")
    cfg = ap.parse_args()
    run = os.path.abspath(cfg.run); src = os.path.abspath(cfg.source)
    res = json.load(open(os.path.join(run, "results.json")))
    it = cfg.iteration if cfg.iteration is not None else int(res["config"]["iterations"])
    test = [l.strip() for l in open(os.path.join(src, "test_index.txt")) if l.strip()]
    have = [n for n in test if os.path.exists(os.path.join(run, "renders", n + ".png"))]
    # Refusing by default matters: a partial set would produce a headline number
    # computed over an easier subset than the other rows of the table.
    if len(have) < len(test) and not cfg.allow_partial:
        raise SystemExit(f"refusing: {len(have)} of {len(test)} held-out renders in {run}; re-render with --save-renders -1")
    stage = os.path.join(run, "test", f"ours_{it}")
    # Cleared first, so a stale render from an earlier iteration cannot be scored.
    for sub in ("renders", "gt"):
        shutil.rmtree(os.path.join(stage, sub), ignore_errors=True); os.makedirs(os.path.join(stage, sub))
    # Copied under the SAME name on both sides, which is how metrics.py pairs them.
    for n in have:
        shutil.copy(os.path.join(run, "renders", n + ".png"), os.path.join(stage, "renders", n + ".png"))
        shutil.copy(os.path.join(src, "images", n + ".png"), os.path.join(stage, "gt", n + ".png"))
    env = dict(os.environ, PYTHONUTF8="1")
    out = subprocess.run([PY_3DGS, os.path.join(REPO, "metrics.py"), "-m", run], cwd=REPO, env=env, capture_output=True, text=True)
    if out.returncode != 0:
        # metrics.py swallows its own exceptions, so its stderr is the only
        # diagnostic; the tail of it is printed before giving up.
        sys.stderr.write(out.stderr[-2000:]); raise SystemExit("metrics.py failed")
    m = json.load(open(os.path.join(run, "results.json")))     # metrics.py overwrites results.json with its own dict
    # keep the trainer's results too: metrics.py writes {"ours_<it>": {...}} over the trainer's file, so restore both
    json.dump(res, open(os.path.join(run, "results.json"), "w"), indent=1)
    key = f"ours_{it}"
    # `partial` travels with the numbers, so a subset score can never be read as a full one.
    scored = {"views": len(have), "partial": len(have) < len(test), "iteration": it, **m[key]}
    json.dump(scored, open(os.path.join(run, "inria_metrics.json"), "w"), indent=1)
    print(f"{os.path.basename(run)}: {len(have)} views, metrics.py PSNR {scored['PSNR']:.4f}  SSIM {scored['SSIM']:.4f}  LPIPS {scored['LPIPS']:.4f}")


if __name__ == "__main__":
    main()
