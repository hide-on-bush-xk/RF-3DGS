"""The floor under the decoder: naive baselines and beam-selection accuracy.

The RRF's decoded angle-of-departure error (median 0.59 deg azimuth on the
held-out views) has a ceiling (the target's own encoding, 0.20 deg) but no
floor. This adds it, on exactly the evaluator's pixels and views:

  nearest     copy the spectrum of the nearest TRAINING position with the
              same yaw and decode it (no model at all);
  interp2     inverse-distance blend of the two nearest training positions'
              channels (cos/sin blend = circular mean of the azimuth);
  rrf         the trained field's render (as encoding_comparison reports);

and translates every method's angle errors into beam-selection accuracy:
a square codebook of theta_3dB cells on (azimuth, zenith), theta_3dB =
101.5 deg / M (10.2 deg for the M = 10 array of these datasets, 1.59 deg for
the 64 x 64 array the paper headlines). top-k = the true beam is among the k
codebook cells nearest to the decoded direction. Unweighted over hit pixels
and weighted by each pixel's linear power.

    PYTHONUTF8=1 python rrf_gsplat/eval_baselines.py --multi output/rrf/m_multi_24_tut_cs \
        --truth RF-3DGS_dataset/regenerated/3dgs_MULTI_24ghz_tut_cs
"""

from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
VIEW_YAWS = (-math.pi / 2, 0.0, math.pi / 2, math.pi)
ARRAYS = {"M=10 (these datasets)": 101.5 / 10, "64x64 (paper headline)": 101.5 / 64}


def read_poses(images_txt):
    """name -> (rx [3], yaw index)."""
    import sys
    sys.path.insert(0, os.path.join(REPO, "sionna_port"))
    from generate_dataset import euler_to_quaternion

    def rotmat(q):
        w, x, y, z = q
        return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                         [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                         [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    yaw_R = [euler_to_quaternion([yaw, 0.0, 0.0])[0].as_matrix() for yaw in VIEW_YAWS]
    poses = {}
    for line in open(images_txt):
        p = line.split()
        if len(p) < 10 or not p[9].lower().endswith(".png"):
            continue
        R = rotmat([float(v) for v in p[1:5]]); t = np.array([float(v) for v in p[5:8]])
        rx = -R.T @ t
        yaw = int(np.argmin([np.abs(m - R).sum() for m in yaw_R]))
        poses[p[9][:-4]] = (rx, yaw)
    return poses


def decode(arr, ch):
    az = np.degrees(np.arctan2(arr[ch["aod_az_sin"]], arr[ch["aod_az_cos"]]))
    return az, arr[ch["aod_zen"]] * 180.0, arr[ch["delay_ns"]]


def topk_hits(az_p, zen_p, az_t, zen_t, cell, ks=(1, 3, 5)):
    """For each pixel: is the true beam among the k codebook cells nearest to the decoded direction?"""
    # cell centres at (i + 0.5) * cell; the true cell and the decoded point
    it_az, it_zen = np.floor(az_t / cell), np.floor(zen_t / cell)
    ip_az, ip_zen = np.floor(az_p / cell), np.floor(zen_p / cell)
    # distance from the decoded point to the true cell's centre
    d_true = np.hypot(az_p - (it_az + 0.5) * cell, zen_p - (it_zen + 0.5) * cell)
    # count cells (in a 7x7 neighbourhood of the decoded cell) whose centre is closer than the true cell's
    closer = np.zeros(az_p.shape, dtype=np.int32)
    for da in range(-3, 4):
        for dz in range(-3, 4):
            ca, cz = ip_az + da, ip_zen + dz
            d = np.hypot(az_p - (ca + 0.5) * cell, zen_p - (cz + 0.5) * cell)
            is_true = (ca == it_az) & (cz == it_zen)
            closer += ((d < d_true) & ~is_true).astype(np.int32)
    far = d_true > 3.5 * cell                       # true cell outside the neighbourhood: rank > 49
    rank = np.where(far, 10_000, closer)
    return {k: rank < k for k in ks}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--multi", required=True); ap.add_argument("--truth", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--subset-mode", choices=["route", "fps"], default="route")
    ap.add_argument("--layout", default=None, help="scene layout.json (scene 2): also report every method per space class "
                                                   "(corridor / room / hall) of the held-out receiver, on the same pixels")
    ap.add_argument("--max-train-views", type=int, default=None,
                    help="use the same training subset as train_rrf --max-train-views (whole positions, the trainer's rule)")
    cfg = ap.parse_args()
    meta = json.load(open(os.path.join(cfg.truth, "generation_meta.json")))
    ch = {n: i for i, n in enumerate(meta["channels"])}; lo0, hi0 = meta["channel_ranges"][0]
    poses = read_poses(os.path.join(cfg.truth, "sparse", "0", "images.txt"))
    train = [l.strip() for l in open(os.path.join(cfg.truth, "train_index.txt")) if l.strip()]
    if cfg.max_train_views:                 # the trainer's rule: whole positions, evenly spaced along the training list, or fps
        per_pos, n_pos = 4, len(train) // 4
        keep = max(1, cfg.max_train_views // per_pos)
        if cfg.subset_mode == "fps":
            pos = np.array([poses[train[p * per_pos]][0] for p in range(n_pos)])
            chosen = [0]; dmin = np.linalg.norm(pos - pos[0], axis=1)
            while len(chosen) < keep:
                j = int(dmin.argmax()); chosen.append(j); dmin = np.minimum(dmin, np.linalg.norm(pos - pos[j], axis=1))
            pos_idx = np.array(sorted(chosen))
        else:
            pos_idx = np.linspace(0, n_pos - 1, keep).round().astype(int)
        train = [train[p * per_pos + k] for p in pos_idx for k in range(per_pos)]
    test = [l.strip() for l in open(os.path.join(cfg.truth, "test_index.txt")) if l.strip()]
    test = [n for n in test if os.path.exists(os.path.join(cfg.multi, "renders", n + ".npy"))]
    # constant predictor: the training pixels' circular-mean azimuth, mean zenith, mean delay
    cs = np.zeros(2); zs = ds = cnt = 0.0
    for n in train:
        a = np.load(os.path.join(cfg.truth, "spectra_float", n + ".npy")).astype(np.float64)
        m = (a[0] - lo0) / (hi0 - lo0) > 0.02
        if not m.any():
            continue
        cs += np.array([a[ch["aod_az_cos"]][m].sum(), a[ch["aod_az_sin"]][m].sum()])
        zs += a[ch["aod_zen"]][m].sum(); ds += a[ch["delay_ns"]][m].sum(); cnt += m.sum()
    const = {"az": float(np.degrees(np.arctan2(cs[1], cs[0]))), "zen": float(zs / cnt * 180.0), "delay": float(ds / cnt)}
    print(f"training subset: {len(train)} views ({len(train)//4} positions); constant predictor az {const['az']:.1f} deg, zen {const['zen']:.1f} deg, delay {const['delay']:.1f} ns")
    # training positions per yaw
    by_yaw = {y: [] for y in range(4)}
    for n in train:
        rx, y = poses[n]; by_yaw[y].append((rx, n))
    tpos = {y: np.array([r for r, _ in v]) for y, v in by_yaw.items()}
    tnames = {y: [n for _, n in v] for y, v in by_yaw.items()}
    load = lambda n: np.load(os.path.join(cfg.truth, "spectra_float", n + ".npy")).astype(np.float64)

    methods = ("constant", "nearest", "interp2", "rrf")
    err = {m: {"az": [], "zen": [], "delay": []} for m in methods}
    hits = {m: {a: {k: [] for k in (1, 3, 5)} for a in ARRAYS} for m in methods}
    weights, nn_dist, n_pix, view_rx = [], [], 0, []
    for n in test:
        truth = load(n); rx, y = poses[n]
        mask = (truth[0] - lo0) / (hi0 - lo0) > 0.02
        if not mask.any():
            continue
        d = np.linalg.norm(tpos[y] - rx, axis=1); order = np.argsort(d)[:2]
        nn_dist.append(float(d[order[0]])); view_rx.append(np.asarray(rx, float))
        near = load(tnames[y][order[0]]); second = load(tnames[y][order[1]])
        w1, w2 = 1.0 / max(d[order[0]], 1e-6), 1.0 / max(d[order[1]], 1e-6)
        interp = (w1 * near + w2 * second) / (w1 + w2)
        preds = {"nearest": near, "interp2": interp, "rrf": np.load(os.path.join(cfg.multi, "renders", n + ".npy")).astype(np.float64)}
        t_az, t_zen, t_dl = decode(truth, ch)
        pw = 10.0 ** (truth[0][mask] / 10.0); weights.append(pw); n_pix += int(mask.sum())
        for m in methods:
            if m == "constant":
                p_az = np.full(t_az.shape, const["az"]); p_zen = np.full(t_az.shape, const["zen"]); p_dl = np.full(t_az.shape, const["delay"])
            else:
                p_az, p_zen, p_dl = decode(preds[m], ch)
            d_az = ((p_az - t_az + 180) % 360 - 180)[mask]
            err[m]["az"].append(np.abs(d_az)); err[m]["zen"].append(np.abs(p_zen - t_zen)[mask]); err[m]["delay"].append(np.abs(p_dl - t_dl)[mask])
            for a, cell in ARRAYS.items():
                h = topk_hits(p_az[mask], p_zen[mask], t_az[mask], t_zen[mask], cell)
                for k in (1, 3, 5):
                    hits[m][a][k].append(h[k])
    w = np.concatenate(weights); w = w / w.sum()
    out = {"views": len(test), "pixels": n_pix, "nearest_train_distance_m": {"median": float(np.median(nn_dist)), "p90": float(np.percentile(nn_dist, 90)), "max": float(np.max(nn_dist))},
           "arrays_theta3db_deg": ARRAYS, "errors": {}, "topk": {}}
    print(f"{len(test)} held-out views, {n_pix:,} hit pixels; nearest training position: median {out['nearest_train_distance_m']['median']:.2f} m, "
          f"P90 {out['nearest_train_distance_m']['p90']:.2f} m, max {out['nearest_train_distance_m']['max']:.2f} m")
    print(f"{'method':8s} {'az med/P90/RMSE':>22s} {'zen med/P90/RMSE':>22s} {'delay med/P90/RMSE (ns)':>26s}")
    for m in methods:
        row = {}
        for q in ("az", "zen", "delay"):
            e = np.concatenate(err[m][q]); order = np.argsort(e); cw = np.cumsum(w[order])
            row[q] = {"median": float(np.median(e)), "p90": float(np.percentile(e, 90)), "rmse": float(np.sqrt((e ** 2).mean())),
                      "median_pw": float(e[order][np.searchsorted(cw, 0.5)]), "p90_pw": float(e[order][np.searchsorted(cw, 0.9)])}
        out["errors"][m] = row
        f = lambda q: f"{row[q]['median']:6.2f}/{row[q]['p90']:6.2f}/{row[q]['rmse']:6.2f}"
        print(f"{m:8s} {f('az'):>22s} {f('zen'):>22s} {f('delay'):>26s}")
    print("\nbeam selection (true beam among the k cells nearest to the decoded direction), unweighted | power-weighted:")
    for a, cell in ARRAYS.items():
        print(f"  {a}, theta_3dB {cell:.2f} deg:")
        out["topk"][a] = {}
        for m in methods:
            r = {}
            for k in (1, 3, 5):
                h = np.concatenate(hits[m][a][k]); r[f"top{k}"] = float(h.mean()); r[f"top{k}_pw"] = float((w * h).sum())
            out["topk"][a][m] = r
            print(f"    {m:8s} top-1 {r['top1']:.3f} | {r['top1_pw']:.3f}   top-3 {r['top3']:.3f} | {r['top3_pw']:.3f}   top-5 {r['top5']:.3f} | {r['top5_pw']:.3f}")
    if cfg.layout:
        # the same errors split by the space class of the held-out receiver (corridor / room / hall), so a
        # crossover can be read per propagation regime; a jittered position outside every box goes to the nearest one
        spaces = json.load(open(cfg.layout))["spaces"]
        def space_of(rx):
            for s in spaces:
                if s["x0"] - 0.3 <= rx[0] <= s["x1"] + 0.3 and s["y0"] - 0.3 <= rx[1] <= s["y1"] + 0.3:
                    return s["name"]
            c = [((rx[0] - (s["x0"] + s["x1"]) / 2) ** 2 + (rx[1] - (s["y0"] + s["y1"]) / 2) ** 2, s["name"]) for s in spaces]
            return min(c)[1]
        cls = ["room" if space_of(rx).startswith("room") else space_of(rx) for rx in view_rx]
        out["by_space"] = {}
        print("\nby space class of the held-out receiver (median / P90 / RMSE):")
        for c in sorted(set(cls)):
            idx = [i for i, k in enumerate(cls) if k == c]
            wc = np.concatenate([weights[i] for i in idx]); wc = wc / wc.sum()
            grp = {"views": len(idx), "nearest_train_distance_m": {"median": float(np.median([nn_dist[i] for i in idx])),
                                                                   "p90": float(np.percentile([nn_dist[i] for i in idx], 90))}, "errors": {}}
            for m in methods:
                grp["errors"][m] = {}
                for q in ("az", "zen", "delay"):
                    e = np.concatenate([err[m][q][i] for i in idx]); order = np.argsort(e); cw = np.cumsum(wc[order])
                    grp["errors"][m][q] = {"median": float(np.median(e)), "p90": float(np.percentile(e, 90)), "rmse": float(np.sqrt((e ** 2).mean())),
                                           "median_pw": float(e[order][np.searchsorted(cw, 0.5)])}
            out["by_space"][c] = grp
            f = lambda m, q: f"{grp['errors'][m][q]['median']:.2f}/{grp['errors'][m][q]['p90']:.1f}"
            print(f"  {c:9s} {len(idx):4d} views, nearest {grp['nearest_train_distance_m']['median']:.2f} m: "
                  f"az copy {f('nearest', 'az')} field {f('rrf', 'az')} | zen copy {f('nearest', 'zen')} field {f('rrf', 'zen')} | "
                  f"delay copy {f('nearest', 'delay')} field {f('rrf', 'delay')}")
    json.dump(out, open(cfg.out or os.path.join(cfg.multi, "baselines.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
