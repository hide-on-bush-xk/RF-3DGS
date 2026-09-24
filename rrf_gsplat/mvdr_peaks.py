"""Peaks of a beamformed spectrum: does a prediction keep the paths, not just the picture?

PSNR and dB RMSE average over every pixel; a spectrum's use is its peaks (a peak is a path direction). For
every held-out view, against the float truth at the truth's own resolution:

  main peak    the angle between the truth's and the prediction's argmax pixel directions (deg), the error of
               the peak power (pred max - truth max, dB), and the prediction's power AT the true peak (dB)
  top peaks    the truth's local maxima (3x3) within 20 dB of its main peak, greedily kept at least 3 deg apart,
               up to 3: detected if the prediction has a local maximum within 1.5 deg; and the prediction's
               power at each of them
  false peaks  the prediction's own top peaks (same rule, up to 5): false if no local maximum of the truth within
               20 dB of the truth's main peak lies within 1.5 deg -- a peak the model made up, which is what a
               learned upsampler or shading head could do and no pixel metric would show

The prediction is a run's saved renders (native dB, .npy), which must cover every held-out view.

    python rrf_gsplat/mvdr_peaks.py --run output/rrf/r27_live_new --truth RF-3DGS_dataset/regenerated/r27_live_300_gpct
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np


def pixel_dirs(truth):
    """Unit direction per pixel [H, W, 3] in the camera frame, from the dataset's PINHOLE camera."""
    p = open(os.path.join(truth, "sparse", "0", "cameras.txt")).read().split()
    w, h, fx, fy, cx, cy = int(p[2]), int(p[3]), *map(float, p[4:8])
    u = (np.arange(w) + 0.5 - cx) / fx; v = (np.arange(h) + 0.5 - cy) / fy
    d = np.stack(np.broadcast_arrays(u[None, :], v[:, None], np.ones((h, w))), axis=-1)
    return d / np.linalg.norm(d, axis=-1, keepdims=True)


def local_maxima(x):
    """Boolean [H, W]: the pixel is >= its 8 neighbours and > the lowest of them (edges padded with -inf).

    The second condition excludes plateaus: without it every pixel of a flat region is a "maximum", and a
    constant prediction detected every true peak (caught by the constant-image control).
    """
    pad = np.pad(x, 1, constant_values=-np.inf)
    nb = np.stack([pad[1 + dy:1 + dy + x.shape[0], 1 + dx:1 + dx + x.shape[1]]
                   for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy, dx) != (0, 0)])
    inner = np.where(np.isinf(nb), np.inf, nb).min(axis=0)       # the lowest real neighbour
    return (x >= nb).all(axis=0) & (x > inner)


def top_peaks(x, dirs, k, within_db=20.0, sep_deg=3.0):
    """Flat indices of x's local maxima within within_db of its maximum, strongest first, greedily kept at least
    sep_deg apart, at most k."""
    cand = np.flatnonzero(local_maxima(x) & (x >= x.max() - within_db))
    cand = cand[np.argsort(-x.flat[cand])]
    kept = []
    for c in cand:
        if all(angle(dirs, c, j) >= sep_deg for j in kept):
            kept.append(int(c))
        if len(kept) == k:
            break
    return kept


def angle(dirs, a, b):
    """Degrees between the directions of flat pixel indices a and b."""
    da, db = dirs.reshape(-1, 3)[a], dirs.reshape(-1, 3)[b]
    return float(np.degrees(np.arccos(np.clip((da * db).sum(-1), -1.0, 1.0))))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True); ap.add_argument("--truth", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--protocol", default=None, help="score the protocol's --eval-set instead of test_index.txt")
    ap.add_argument("--eval-set", default="val")
    ap.add_argument("--final-test", action="store_true", help="allow a sealed test set (logged)")
    a = ap.parse_args()
    dirs = pixel_dirs(a.truth); flat = dirs.reshape(-1, 3)
    if a.protocol:
        import protocol as PR
        PR.check_dataset(a.protocol, a.truth)
        test = PR.eval_names(a.protocol, a.eval_set, allow_test=a.final_test)
    else:
        test = [l.strip() for l in open(os.path.join(a.truth, "test_index.txt")) if l.strip()]
    missing = [n for n in test if not os.path.exists(os.path.join(a.run, "renders", n + ".npy"))]
    if missing:
        raise SystemExit(f"{len(missing)} of {len(test)} held-out views have no saved render in {a.run}")
    main_ang, main_pow, at_true, det, top_pow, skipped, distinct = [], [], [], [], [], 0, []
    false_flags = []
    for n in test:
        t = np.load(os.path.join(a.truth, "spectra_float", n + ".npy")).astype(np.float64)
        p = np.load(os.path.join(a.run, "renders", n + ".npy")).astype(np.float64)
        if p.shape != t.shape:
            raise SystemExit(f"{n}: render {p.shape} vs truth {t.shape}; render at the truth's resolution first")
        if t.max() < -250 or t.max() - t.min() < 1e-3:   # an all-floor view (no paths; APS: flat at N0): nothing to find
            skipped += 1; continue
        kt, kp = int(t.argmax()), int(p.argmax())
        main_ang.append(angle(dirs, kt, kp)); main_pow.append(p.max() - t.max()); at_true.append(p.flat[kt] - t.flat[kt])
        # the truth's top peaks, >= 3 deg apart
        kept = top_peaks(t, dirs, 3)
        # is the truth's main peak distinct? its best rival >= 3 deg away is >= 1 dB lower (or there is none within
        # 20 dB). On near-ties the argmax flips between two lattices of the same scene (APS smoke: 73 % within 1 deg
        # over all views, 94 % on the distinct ones), so the main-peak angle is only well posed on these
        distinct.append(len(kept) < 2 or t.flat[kept[0]] - t.flat[kept[1]] >= 1.0)
        pm = np.flatnonzero(local_maxima(p))
        # the prediction's top peaks, each checked against every true local maximum within 20 dB
        tm = np.flatnonzero(local_maxima(t) & (t >= t.max() - 20.0))
        for c in top_peaks(p, dirs, 5):
            dmin = np.degrees(np.arccos(np.clip(flat[tm] @ flat[c], -1, 1))).min() if len(tm) else 180.0
            false_flags.append(dmin > 1.5)
        for c in kept:
            dmin = np.degrees(np.arccos(np.clip(flat[pm] @ flat[c], -1, 1))).min() if len(pm) else 180.0
            det.append(dmin <= 1.5); top_pow.append(p.flat[c] - t.flat[c])
    q = lambda v, f: float(f(np.asarray(v)))  # noqa: E731
    res = {"views": len(test) - skipped, "all_floor_views_skipped": skipped,
           "main_peak_angle_deg": {"median": q(main_ang, np.median), "p90": q(main_ang, lambda x: np.percentile(x, 90)),
                                   "within_1deg": q(main_ang, lambda x: (x <= 1.0).mean())},
           "main_peak_power_err_db": {"median": q(main_pow, np.median), "p10": q(main_pow, lambda x: np.percentile(x, 10)),
                                      "p90": q(main_pow, lambda x: np.percentile(x, 90))},
           "power_at_true_peak_err_db": {"median": q(at_true, np.median), "p10": q(at_true, lambda x: np.percentile(x, 10)),
                                         "p90": q(at_true, lambda x: np.percentile(x, 90))},
           "top3_peaks": {"count": len(det), "detected_within_1p5deg": q(det, np.mean),
                          "power_err_db_median": q(top_pow, np.median), "power_err_db_p10": q(top_pow, lambda x: np.percentile(x, 10))},
           "false_peaks": {"predicted_peaks": len(false_flags),
                           "false_rate": q(false_flags, np.mean) if false_flags else None}}
    ang_d = np.asarray(main_ang)[np.asarray(distinct, dtype=bool)]
    res["main_peak_angle_deg_distinct"] = {"views": int(len(ang_d)),
                                           "median": float(np.median(ang_d)) if len(ang_d) else None,
                                           "within_1deg": float((ang_d <= 1.0).mean()) if len(ang_d) else None}
    res["eval_set"] = a.eval_set if a.protocol else "test_index"
    out = a.out or os.path.join(os.path.dirname(os.path.abspath(a.run)), f"peaks_{os.path.basename(os.path.abspath(a.run))}.json")
    json.dump(res, open(out, "w"), indent=1)
    m, pw, at, tp = res["main_peak_angle_deg"], res["main_peak_power_err_db"], res["power_at_true_peak_err_db"], res["top3_peaks"]
    md = {k: (float("nan") if v is None else v) for k, v in res["main_peak_angle_deg_distinct"].items()}
    print(f"{os.path.basename(a.run)}: {res['views']} views | main peak angle median {m['median']:.2f} P90 {m['p90']:.2f} deg "
          f"(<=1 deg {100 * m['within_1deg']:.0f}%; distinct {md['views']} views: median {md['median']:.2f}, "
          f"<=1 deg {100 * md['within_1deg']:.0f}%) | peak power err median {pw['median']:+.2f} P10 {pw['p10']:+.2f} dB | "
          f"at true peak {at['median']:+.2f} (P10 {at['p10']:+.2f}) | top-3 peaks detected {100 * tp['detected_within_1p5deg']:.0f}% "
          f"of {tp['count']}, power err {tp['power_err_db_median']:+.2f} dB | false peaks "
          f"{'n/a' if res['false_peaks']['false_rate'] is None else format(100 * res['false_peaks']['false_rate'], '.0f') + '%'} "
          f"of {res['false_peaks']['predicted_peaks']}")


if __name__ == "__main__":
    main()
