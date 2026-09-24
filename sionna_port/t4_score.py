"""T4 level 1: score every predictor's held-out spectra against the truth on the same per-view metrics
(docs/t4_channel_model_plan.md). A predictor is any directory with renders/<view>.npy in native dB: the radio
radiance fields' runs, and t4_baselines.py's los / nn / inh_* runs.

Per view, against the float truth:
  direction   angle between the truth's and the prediction's main peaks; <= 1 deg share
  beam loss   truth max - truth at the prediction's main peak (dB): what steering to the predicted peak costs on
              the true spectrum -- level-free, the downstream reading of the direction error
  top-3       the truth's top peaks detected within 1.5 deg (mvdr_peaks' rule)
  level       at the true peak and RMSE (both clipped to the dataset's dB range), raw and after one global offset
              per predictor (the median over views of truth max - prediction max): InH and LOS carry their own
              absolute level, which is a convention, not a prediction
  LOS check   at the views whose geometric transmitter direction falls inside the image: the angle from the
              main peak to that direction. For the truth itself this checks the pixel <-> world mapping; for InH it
              checks its element ordering / phase convention (control (b) of the plan)

    python sionna_port/t4_score.py --truth RF-3DGS_dataset/regenerated/3dgs_MVDR_100_gpct \
        --runs output/rrf/t4_los output/rrf/t4_nn "output/rrf/t4_inh_open_r*" output/rrf/r45_base
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(REPO, "rrf_gsplat"))

from mvdr_peaks import angle, local_maxima, pixel_dirs, top_peaks  # noqa: E402


TRUTH_RENDERS = None     # --truth-renders: score against another truth (e.g. t6's float64 recomputation)


def truth_file(truth, n):
    """The truth spectrum of view n: the dataset's stored one, or --truth-renders' if given."""
    if TRUTH_RENDERS:
        return os.path.join(TRUTH_RENDERS, "renders", n + ".npy")
    return os.path.join(truth, "spectra_float", n + ".npy")


def camera_rotations(truth):
    """name -> the world-to-camera rotation the trainer uses (COLMAP: x_cam = R x_world + t), and the centre."""
    out = {}
    for line in open(os.path.join(truth, "sparse", "0", "images.txt")):
        p = line.split()
        if len(p) < 10 or not p[9].lower().endswith(".png"):
            continue
        w, x, y, z = map(float, p[1:5])
        R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                      [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                      [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
        t = np.array(list(map(float, p[5:8])))
        out[p[9][:-4]] = (R, -R.T @ t)
    return out


def los_pixel(truth, R, c, tx):
    """Flat pixel index of the transmitter's direction in this view, or None if it is outside the image."""
    q = open(os.path.join(truth, "sparse", "0", "cameras.txt")).read().split()
    w, h, fx, fy, cx, cy = int(q[2]), int(q[3]), *map(float, q[4:8])
    d = R @ (tx - c)
    if d[2] <= 0:
        return None
    u, v = fx * d[0] / d[2] + cx, fy * d[1] / d[2] + cy
    if not (0 <= u < w and 0 <= v < h):
        return None
    return int(v) * w + int(u)


def spread_deg(x, dirs, within_db=20.0):
    """Power-weighted RMS angle (deg) from the main peak over the pixels within within_db of it: one spectrum's
    angular spread, the same estimator for the truth and every predictor (a spectrum-level LSP)."""
    flat = dirs.reshape(-1, 3); v = x.reshape(-1); k = int(v.argmax())
    m = v >= v[k] - within_db
    w = 10.0 ** ((v[m] - v[k]) / 10.0)
    th = np.degrees(np.arccos(np.clip(flat[m] @ flat[k], -1, 1)))
    return float(np.sqrt((w * th * th).sum() / w.sum()))


def ks(a, b):
    """Two-sample Kolmogorov-Smirnov distance."""
    a, b = np.sort(np.asarray(a, float)), np.sort(np.asarray(b, float))
    z = np.concatenate([a, b])
    return float(np.abs(np.searchsorted(a, z, side="right") / len(a) - np.searchsorted(b, z, side="right") / len(b)).max())


def bartlett_maps(cov_npz, truth):
    """view -> the truth channel's matched-beam gain map in dB, a^H R a / |a|^2 per pixel (R the tap covariance of
    the ray-traced channel, from t6_incoherent_mvdr.py --positions 0): the received power of a beam steered there."""
    import torch
    from generate_dataset import element_gain_fn
    from rf_spectra import ArrayGrid
    meta = json.load(open(os.path.join(truth, "generation_meta.json")))
    grid = ArrayGrid.build(meta["M"], meta["width"], meta["height"], meta["fov_deg"], element_gain_fn=element_gain_fn, device="cuda")
    man = grid.manifold
    norm = (man.abs() ** 2).sum(0)
    z = np.load(cov_npz)
    out = {}
    for n, R in zip(z["names"], z["R"]):
        R = torch.as_tensor(R, device="cuda")
        q = torch.einsum("mhw,mn,nhw->hw", man.conj(), R, man).real / norm
        out[str(n)] = (10 * torch.log10(q.clamp_min(1e-300))).cpu().numpy().astype(np.float64)
    return out


def score(run, truth, test, dirs, lo, hi, cams, tx, offset=None, bart=None):
    flat = dirs.reshape(-1, 3)
    T, P = [], []
    for n in test:
        t = np.load(truth_file(truth, n)).astype(np.float64)
        if t.max() < -250 or t.max() - t.min() < 1e-3:   # no paths (MVDR: EMPTY_VIEW_DB; APS: flat at N0)
            continue
        T.append((n, t)); P.append(np.load(os.path.join(run, "renders", n + ".npy")).astype(np.float64))
    off_est = float(np.median([t.max() - p.max() for (_, t), p in zip(T, P)]))
    off = off_est if offset is None else offset
    rows = []
    for (n, t), p in zip(T, P):
        kt, kp = int(t.argmax()), int(p.argmax())
        kept = top_peaks(t, dirs, 3); pm = np.flatnonzero(local_maxima(p))
        det = [(np.degrees(np.arccos(np.clip(flat[pm] @ flat[c], -1, 1))).min() <= 1.5) if len(pm) else False for c in kept]
        tc, pc, pa = np.clip(t, lo, hi), np.clip(p, lo, hi), np.clip(p + off, lo, hi)
        R, c = cams[n]; kl = los_pixel(truth, R, c, tx)
        rows.append({"view": n, "angle": angle(dirs, kt, kp), "beam_loss": float(t.max() - t.flat[kp]),
                     "top3": [bool(d) for d in det], "at_true": float(p.flat[kt] - t.flat[kt]), "at_true_aligned": float(p.flat[kt] + off - t.flat[kt]),
                     "rmse": float(np.sqrt(np.mean((pc - tc) ** 2))), "rmse_aligned": float(np.sqrt(np.mean((pa - tc) ** 2))),
                     "los_in_view": kl is not None,
                     "peak_to_los": None if kl is None else angle(dirs, kp, kl),
                     "truth_peak_to_los": None if kl is None else angle(dirs, kt, kl),
                     "spread_truth": spread_deg(t, dirs), "spread_pred": spread_deg(p, dirs),
                     # level 3: a beam steered to the predicted peak, on the true channel (dB below the best beam)
                     "beam_gain_loss": None if not bart or n not in bart else float(bart[n].max() - bart[n].flat[kp]),
                     "beam_gain_loss_truth_peak": None if not bart or n not in bart else float(bart[n].max() - bart[n].flat[kt])})
    a = np.array([r["angle"] for r in rows]); bl = np.array([r["beam_loss"] for r in rows])
    det = [d for r in rows for d in r["top3"]]
    los_rows = [r for r in rows if r["los_in_view"]]
    # the views where the truth's own main peak is the direct path (within 2 deg of the geometric direction)
    los_dom = [r for r in los_rows if r["truth_peak_to_los"] <= 2.0]
    med = lambda k, rs=rows: float(np.median([r[k] for r in rs])) if rs else None  # noqa: E731
    return {"run": os.path.basename(os.path.abspath(run)), "views": len(rows), "offset_db": off, "offset_estimated_db": off_est,
            "angle_median": float(np.median(a)), "angle_p90": float(np.percentile(a, 90)), "within_1deg": float((a <= 1).mean()),
            "beam_loss_median": float(np.median(bl)), "beam_loss_p90": float(np.percentile(bl, 90)),
            "beam_loss_within_3db": float((bl <= 3).mean()),
            "top3_detected": float(np.mean(det)), "at_true_median": med("at_true"), "at_true_aligned_median": med("at_true_aligned"),
            "rmse_mean": float(np.mean([r["rmse"] for r in rows])), "rmse_aligned_mean": float(np.mean([r["rmse_aligned"] for r in rows])),
            "los_views": len(los_rows), "los_dominant_views": len(los_dom),
            "peak_to_los_median_at_los_dominant": med("peak_to_los", los_dom),
            "peak_within_2deg_of_los_at_los_dominant": float(np.mean([r["peak_to_los"] <= 2.0 for r in los_dom])) if los_dom else None,
            "angle_median_at_los_dominant": med("angle", los_dom),
            "angle_median_elsewhere": med("angle", [r for r in rows if r not in los_dom]),
            "spread_truth_median": med("spread_truth"), "spread_pred_median": med("spread_pred"),
            "spread_ks": ks([r["spread_truth"] for r in rows], [r["spread_pred"] for r in rows]),
            "spread_abs_err_median": float(np.median([abs(r["spread_pred"] - r["spread_truth"]) for r in rows])),
            "beam_gain_loss_views": sum(r["beam_gain_loss"] is not None for r in rows),
            "beam_gain_loss_median": med("beam_gain_loss", [r for r in rows if r["beam_gain_loss"] is not None]),
            "beam_gain_loss_p90": (float(np.percentile([r["beam_gain_loss"] for r in rows if r["beam_gain_loss"] is not None], 90))
                                   if any(r["beam_gain_loss"] is not None for r in rows) else None),
            "beam_gain_loss_truth_peak_median": med("beam_gain_loss_truth_peak", [r for r in rows if r["beam_gain_loss"] is not None])}, rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--truth", required=True); ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--db-range", nargs=2, type=float, default=None, help="default: the truth dataset's generation_meta range")
    ap.add_argument("--out", default=os.path.join(REPO, "output", "rrf", "t4_scores.json"))
    ap.add_argument("--cov", default=None, help="t6's t4_rt_cov_*.npz: adds the level-3 beam-gain loss")
    ap.add_argument("--rt-stats", default=None, help="t6's t4_rt_stats_*.json: adds the level-2 channel LSPs vs InH")
    ap.add_argument("--partial", action="store_true", help="smoke: only the held-out views every run has")
    ap.add_argument("--protocol", default=None, help="score the protocol's --eval-set instead of test_index.txt")
    ap.add_argument("--eval-set", default="val")
    ap.add_argument("--final-test", action="store_true", help="allow a sealed test set (logged)")
    ap.add_argument("--truth-renders", default=None, help="a run directory whose renders/ are the truth "
                    "(t6's float64 recomputation); geometry still from --truth")
    a = ap.parse_args()
    global TRUTH_RENDERS
    TRUTH_RENDERS = a.truth_renders
    bart = bartlett_maps(a.cov, a.truth) if a.cov else None
    # the truth dataset's own normalisation range (the one its training images and every run on it use); until
    # 2026-09-24 04:10 this defaulted to r45_base's MVDR range, which clipped the Bartlett / power datasets' RMSE
    meta_t = json.load(open(os.path.join(a.truth, "generation_meta.json")))
    lo, hi = a.db_range or (meta_t["spec_min_db"], meta_t["spec_max_db"])
    if a.protocol:
        import protocol as PR
        PR.check_dataset(a.protocol, a.truth)
        test = PR.eval_names(a.protocol, a.eval_set, allow_test=a.final_test)
    else:
        test = [l.strip() for l in open(os.path.join(a.truth, "test_index.txt")) if l.strip()]
    dirs = pixel_dirs(a.truth); cams = camera_rotations(a.truth)
    tx = np.array(json.load(open(os.path.join(a.truth, "generation_meta.json")))["tx_loc"], dtype=np.float64)
    runs = sorted({r for pat in a.runs for r in (glob.glob(pat) or [pat]) if os.path.isdir(r)})
    if a.partial:
        # a smoke: score the held-out views every run has
        test = [n for n in test if all(os.path.exists(os.path.join(r, "renders", n + ".npy")) for r in runs)]
        print(f"partial: {len(test)} views common to every run")
    out = {}
    for run in runs:
        if not all(os.path.exists(os.path.join(run, "renders", n + ".npy")) for n in test):
            print(f"{run}: MISSING renders, skipped"); continue
        s, rows = score(run, a.truth, test, dirs, lo, hi, cams, tx, bart=bart)
        if a.rt_stats and os.path.exists(os.path.join(run, "stats.json")):
            # level 2 on the channels: the model's LSP distributions against the ray tracer's, same views
            rt = json.load(open(a.rt_stats)); mdl = json.load(open(os.path.join(run, "stats.json")))
            common = [n for n in mdl if n in rt]
            s["lsp"] = {k: {"rt_median": float(np.median([rt[n][k] for n in common])),
                            "model_median": float(np.median([mdl[n][k] for n in common])),
                            "ks": ks([rt[n][k] for n in common], [mdl[n][k] for n in common])}
                        for k in ("gain_db", "ds_ns", "first_tap_ratio_db")} if common else None
            s["lsp_views"] = len(common)
        out[s["run"]] = {"summary": s, "views": rows}
        print(f"{s['run']:<22} dir med {s['angle_median']:6.2f} (<=1 {100 * s['within_1deg']:4.1f}%)  beam loss med "
              f"{s['beam_loss_median']:5.2f} P90 {s['beam_loss_p90']:5.2f} dB  top-3 {100 * s['top3_detected']:4.1f}%  "
              f"at-true {s['at_true_median']:+7.2f} (aligned {s['at_true_aligned_median']:+6.2f})  RMSE {s['rmse_mean']:5.2f} "
              f"(aligned {s['rmse_aligned_mean']:5.2f})  offset {s['offset_db']:+6.1f}  | LOS-dominant views {s['los_dominant_views']}"
              f"/{s['los_views']}: peak within 2 deg of LOS "
              f"{'n/a' if s['peak_within_2deg_of_los_at_los_dominant'] is None else format(100 * s['peak_within_2deg_of_los_at_los_dominant'], '.0f') + '%'}, "
              f"dir med {s['angle_median_at_los_dominant']}, elsewhere {s['angle_median_elsewhere']}")
        if bart:
            print(f"{'':<22} beam steered to the predicted peak: loss median {s['beam_gain_loss_median']:.2f} P90 "
                  f"{s['beam_gain_loss_p90']:.2f} dB on {s['beam_gain_loss_views']} views (to the true MVDR peak: "
                  f"{s['beam_gain_loss_truth_peak_median']:.2f})")
        print(f"{'':<22} spectral spread median truth {s['spread_truth_median']:.2f} / pred {s['spread_pred_median']:.2f} deg, "
              f"KS {s['spread_ks']:.2f}" + ("" if not s.get("lsp") else "; LSP " + ", ".join(
                  f"{k} RT {v['rt_median']:.2f} / model {v['model_median']:.2f} (KS {v['ks']:.2f})" for k, v in s["lsp"].items())))
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
