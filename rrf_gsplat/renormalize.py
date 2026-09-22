"""Re-map a generated dataset's float spectra to jet PNGs under a chosen normalisation.

The generator writes PNGs with the global min/max of every spectrum, which for
MVDR spans ~140 dB because of a numerical floor nobody cares about. This
rewrites images/ from spectra_float/ so the normalisation itself can be the
experiment:

  --norm global-pct   one range for the whole dataset, from percentiles
                      (what a radiance field needs: the same value means the
                      same dB in every view)
  --norm global-minmax  the generator's default
  --norm per-view     each image scaled by its own min/max -- what the
                      tutorial does for CBF/TCBF. The absolute level is gone;
                      per_view_ranges.csv keeps it so an oracle can put it back.

sparse/ and spectra_float/ are linked or copied, so the output is a complete
dataset for train_rrf.py; generation_meta.json records the range used.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil

import numpy as np


def main():
    """Recompute the PNGs under the chosen range; share everything else."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source")
    ap.add_argument("out")
    ap.add_argument("--norm", choices=["global-pct", "global-minmax", "per-view"], default="global-pct")
    ap.add_argument("--pct", type=float, nargs=2, default=[1.0, 99.9],
                    help="lower/upper percentiles for global-pct")
    ap.add_argument("--range", type=float, nargs=2, default=None, help="explicit min max dB")
    ap.add_argument("--copy-float", action="store_true", help="copy spectra_float instead of linking")
    cfg = ap.parse_args()

    from matplotlib import colormaps
    from PIL import Image
    jet = colormaps["jet"]

    src_float = os.path.join(cfg.source, "spectra_float")
    names = sorted(f[:-4] for f in os.listdir(src_float) if f.endswith(".npy"))
    os.makedirs(os.path.join(cfg.out, "images"), exist_ok=True)

    if cfg.norm.startswith("global"):
        if cfg.range:
            vmin, vmax = cfg.range                # explicit wins over any computation
        elif cfg.norm == "global-minmax":
            # Streamed, so the whole dataset never has to be in memory at once.
            vmin, vmax = np.inf, -np.inf
            for n in names:
                a = np.load(os.path.join(src_float, n + ".npy"))
                vmin, vmax = min(vmin, float(a.min())), max(vmax, float(a.max()))
        else:
            # percentiles over a subsample of pixels from every image
            # 2000 pixels per image rather than all of them: the percentiles of
            # a large uniform sample are accurate enough, and this keeps the
            # pool small. Every image contributes equally, so a single extreme
            # view cannot set the range.
            rng = np.random.default_rng(0)
            pool = []
            for n in names:
                a = np.load(os.path.join(src_float, n + ".npy")).ravel()
                pool.append(a[rng.integers(0, a.size, 2000)])
            pool = np.concatenate(pool)
            vmin, vmax = (float(v) for v in np.percentile(pool, cfg.pct))
        print(f"{cfg.norm}: range {vmin:.2f} .. {vmax:.2f} dB (span {vmax-vmin:.1f})")
    ranges = []
    for n in names:
        a = np.load(os.path.join(src_float, n + ".npy")).astype(np.float32)
        if cfg.norm == "per-view":
            lo, hi = float(a.min()), float(a.max())
        else:
            lo, hi = vmin, vmax
        # Recorded for every image regardless of mode, so the CSV below always
        # describes what was actually applied.
        ranges.append((n, lo, hi))
        norm = np.clip((a - lo) / max(hi - lo, 1e-9), 0.0, 1.0)
        rgb = (jet(norm)[..., :3] * 255).round().astype(np.uint8)
        Image.fromarray(rgb).save(os.path.join(cfg.out, "images", n + ".png"))

    for sub in ("sparse",):
        dst = os.path.join(cfg.out, sub)
        if not os.path.exists(dst):
            shutil.copytree(os.path.join(cfg.source, sub), dst)
    # Hard links, not a symlink: WSL cannot follow a Windows symlink, and the
    # trainer runs there.
    dst_float = os.path.join(cfg.out, "spectra_float")
    if not os.path.exists(dst_float):
        os.makedirs(dst_float)
        for f in os.listdir(src_float):
            s, d = os.path.join(src_float, f), os.path.join(dst_float, f)
            if cfg.copy_float:
                shutil.copy(s, d)
            else:
                # Hard links fail across volumes; a copy is the fallback.
                try:
                    os.link(s, d)
                except OSError:
                    shutil.copy(s, d)
    # The split travels with the dataset, so a renormalised sibling is scored on
    # exactly the same held-out views.
    for idx in ("train_index.txt", "test_index.txt"):
        p = os.path.join(cfg.source, idx)
        if os.path.exists(p):
            shutil.copy(p, os.path.join(cfg.out, idx))

    meta_path = os.path.join(cfg.source, "generation_meta.json")
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
    meta.update(normalization=cfg.norm, renormalized_from=os.path.abspath(cfg.source))
    if cfg.norm.startswith("global"):
        meta.update(spec_min_db=vmin, spec_max_db=vmax)
    else:
        # There is no single range in per-view mode. The mean is recorded only
        # so the fields exist, and the note says not to trust it; the CSV is the
        # real record, and it is what an oracle needs to restore absolute level.
        meta.update(spec_min_db=float(np.mean([r[1] for r in ranges])),
                    spec_max_db=float(np.mean([r[2] for r in ranges])),
                    note="per-view: spec_min/max are the mean per-image range; see per_view_ranges.csv")
        with open(os.path.join(cfg.out, "per_view_ranges.csv"), "w", newline="") as fid:
            w = csv.writer(fid); w.writerow(["image", "min_db", "max_db"]); w.writerows(ranges)
    with open(os.path.join(cfg.out, "generation_meta.json"), "w") as fid:
        json.dump(meta, fid, indent=1)
    print(f"wrote {len(names)} images to {cfg.out}")


if __name__ == "__main__":
    main()
