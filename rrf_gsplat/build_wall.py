"""Put the regenerated-data comparisons on the viewer's wall (viewer.py): T4's predictors next to our fields.

viewer.py's wall shows only runs on the six RELEASED datasets, in the INRIA layout (test/ours_<it>/renders). Every
experiment since round 24 is on regenerated datasets, and T4's baselines (NN, LOS, 3GPP InH, the float64 truth) write
only native-dB .npy renders. This builds a curated set of groups for the wall:

  - PNGs for any predictor that has only .npy renders, in the dataset's own jet colormap and dB range (LOS and InH
    carry their own absolute level: they are shifted by the one global offset t4_score.py used, and say so)
  - per view: PSNR(jet) against the dataset's PNG, and the main-peak direction error (deg) against its float truth
  - output/rrf/wall_groups.json, which viewer.py reads at start-up (restart it after rebuilding)

    python rrf_gsplat/build_wall.py        (rf-sionna-win: needs matplotlib and imageio)
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
RRF = os.path.join(REPO, "output", "rrf")
REGEN = os.path.join(REPO, "RF-3DGS_dataset", "regenerated")
sys.path.insert(0, HERE)
from mvdr_peaks import angle, pixel_dirs  # noqa: E402

# (group name, dataset, scores file for the level offsets, [(run dir, label, kind, shown by default)])
GROUPS = [
    ("MVDR regenerated (stored labels)", "3dgs_MVDR_100_gpct", "t4_scores.json", [
        ("t6_truth64", "float64 recomputation = the labels' own noise", "baseline", True),
        ("t4_nn", "NN: nearest training position's label", "baseline", True),
        ("r45_base", "field SH3 (round 45)", "ours", True),
        ("r43_B6_sh3_head", "field SH3 + shading head (B6)", "ours", True),
        ("r48_fit_640_20k", "field SH3, 20k steps", "ours", False),
        ("t4_los", "LOS only (level aligned)", "baseline", False),
        ("t4_inh_open_r0", "3GPP InH-open, realisation 0 (level aligned)", "baseline", False),
        ("t4_inh_mixed_r0", "3GPP InH-mixed, realisation 0 (level aligned)", "baseline", False),
        ("r49_k10", "field trained on 10 positions", "ours", False),
        ("t4_nn_k10", "NN from the same 10 positions", "baseline", False),
    ]),
    ("MVDR regenerated (float64 labels)", "3dgs_MVDR_100_f64_gpct", "t4_scores_f64labels.json", [
        ("t4f64_nn", "NN: nearest training position's label", "baseline", True),
        ("r50_f64", "field SH3 on float64 labels", "ours", True),
        ("r50_f64_s1", "field SH3 on float64 labels, seed 1", "ours", False),
    ]),
    ("Bartlett regenerated (round 54)", "3dgs_CBFF_60_gpct", "t4_scores_cbf.json", [
        ("t4cbf_nn", "NN: nearest training position's label", "baseline", True),
        ("r54_cbf_full_power", "field SH3, power mode", "ours", True),
    ]),
]


def main():
    import imageio.v2 as imageio
    from matplotlib import colormaps
    jet = colormaps["jet"]
    out = []
    for gname, ds, scores_file, preds in GROUPS:
        src = os.path.join(REGEN, ds)
        meta = json.load(open(os.path.join(src, "generation_meta.json")))
        lo, hi = meta["spec_min_db"], meta["spec_max_db"]
        names = sorted(l.strip() for l in open(os.path.join(src, "test_index.txt")) if l.strip())
        dirs = pixel_dirs(src)
        truth = {n: np.load(os.path.join(src, "spectra_float", n + ".npy")).astype(np.float64) for n in names}
        gt_png = {n: imageio.imread(os.path.join(src, "images", n + ".png"))[..., :3].astype(np.float64) / 255.0 for n in names}
        sp = os.path.join(RRF, scores_file)
        scores = json.load(open(sp)) if os.path.isfile(sp) else {}
        entries = []
        for run, label, kind, default in preds:
            rdir = os.path.join(RRF, run, "renders")
            if not all(os.path.isfile(os.path.join(rdir, n + ".npy")) for n in names):
                print(f"{gname}: {run} MISSING renders, skipped"); continue
            off = 0.0
            if "level aligned" in label:
                off = float(((scores.get(run) or {}).get("summary") or {}).get("offset_db", 0.0))
            psnr, peak = {}, {}
            for n in names:
                p = np.load(os.path.join(rdir, n + ".npy")).astype(np.float64) + off
                png = os.path.join(rdir, n + ".png")
                if kind == "baseline" or not os.path.isfile(png):
                    rgb = (jet(np.clip((p - lo) / (hi - lo), 0, 1))[..., :3] * 255).round().astype(np.uint8)
                    imageio.imwrite(png, rgb)
                a = imageio.imread(png)[..., :3].astype(np.float64) / 255.0
                mse = float(((a - gt_png[n]) ** 2).mean())
                psnr[n] = 10 * np.log10(1.0 / max(mse, 1e-12))
                t = truth[n]
                peak[n] = None if t.max() < -250 else angle(dirs, int(t.argmax()), int(p.argmax()))
            pk = np.array([v for v in peak.values() if v is not None])
            res_p = os.path.join(RRF, run, "results.json")
            its = (json.load(open(res_p)).get("config") or {}).get("iterations") if os.path.isfile(res_p) else None
            wall = {"PSNR": float(np.mean(list(psnr.values()))), "peak_median": float(np.median(pk)),
                    "peak_within_1deg": float((pk <= 1).mean()), "per_view_psnr": psnr, "peak_deg": peak,
                    "level_offset_db": off, "iterations": its}
            json.dump(wall, open(os.path.join(RRF, run, f"wall_{ds}.json"), "w"))
            entries.append({"id": run, "label": label, "kind": kind, "default": default,
                            "dir": os.path.relpath(rdir, REPO).replace(os.sep, "/"),
                            "wall": os.path.relpath(os.path.join(RRF, run, f"wall_{ds}.json"), REPO).replace(os.sep, "/")})
            print(f"{gname}: {run:<22} PSNR(jet) {wall['PSNR']:.2f}  main peak median {wall['peak_median']:.2f} deg, "
                  f"<= 1 deg {100 * wall['peak_within_1deg']:.1f} %" + (f"  (level +{off:.1f} dB)" if off else ""))
        out.append({"name": gname, "dataset": os.path.relpath(src, REPO).replace(os.sep, "/"), "preds": entries})
    json.dump(out, open(os.path.join(RRF, "wall_groups.json"), "w"), indent=1)
    print(f"wrote {os.path.join(RRF, 'wall_groups.json')}; restart viewer.py to see the groups")


if __name__ == "__main__":
    main()
