"""G1 of the generalisation diagnostics: do the training positions' emitters cover the validation positions' energy
sources? (docs/stage2_notes.md, "协议 v1" §13; no training.)

Step 1 put emitters at the ray-traced interaction points of the 467 training positions and still lost to the look-ups
on new positions. One explanation is placement: a new position's energy may come from surface points no training
position saw (a specular point moves with the receiver; a diffuse one does not). G1 measures that directly from the
path cache (sionna_port/path_emitters.py --cache: per position, the power and centroid of every 5 cm voxel of
interaction points; the direct path's point is the transmitter):

  main-peak source   per face with a distinct main peak (the dataset's own rule: >= 1 dB above the rival >= 3 deg
                     away), the voxel contributing most power at the true peak direction (power x the dataset's 1 deg
                     splat kernel at the angle between the voxel and the peak); covered at radius r if a training
                     emitter lies within r of its centroid
  power coverage     per position, the share of all voxel power within r of a training emitter
Radii 5 / 10 / 20 / 50 cm; val_random (interpolation) and val_segment (extrapolation) reported apart.

Criteria, written before the first run (2026-09-24):
  c1  source agreement: the source voxel's direction within 1 deg of the true peak direction, median over the
      distinct validation views (the peak-to-world mapping is right and a single voxel explains the peak)
  c2  trivial input: the training positions against their own emitters (60 positions spread along the training
      list): main-peak source covered at 10 cm on >= 95 % of distinct views
  c3  known failure: the training emitters shifted 1 m along x cover <= 10 % of the validation sources at 10 cm
  degenerate: positions without paths, flat views and views without a distinct peak are counted; coverage is also
      reported over all views
Reading, written before the run (main-peak source coverage at 10 cm, validation, distinct views):
  >= 90 %   the training placement covers the new positions' sources -> placement is not what step 1 lacked;
            expect G2 (emitters from the validation positions too) close to step 1's A
  <  60 %   a placement gap -> G2 should improve markedly over A
  between   mixed; read G2

    PYTHONUTF8=1 python rrf_gsplat/g1_coverage.py --truth RF-3DGS_dataset/regenerated/3dgs_APS_60_gp100 \\
        --cache output/rrf/m3/path_cache --emitters output/rrf/m3/emitters_train467.npz
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import protocol as PR                                        # noqa: E402
from mvdr_peaks import pixel_dirs, top_peaks                 # noqa: E402

RADII = (0.05, 0.10, 0.20, 0.50)


def read_views(truth):
    """name -> (R camera->world [3, 3], centre [3]) from COLMAP images.txt (world->camera quaternion and t)."""
    out = {}
    for line in open(os.path.join(truth, "sparse", "0", "images.txt")):
        f = line.split()
        if len(f) < 10 or not f[9].endswith(".png"):
            continue
        qw, qx, qy, qz, tx, ty, tz = map(float, f[1:8])
        R = np.array([[1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
                      [2 * (qx * qy + qw * qz), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qw * qx)],
                      [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx * qx + qy * qy)]])
        out[f[9][:-4]] = (R.T, -R.T @ np.array([tx, ty, tz]))
    return out


def load_position(cache, i, voxel):
    f = os.path.join(cache, f"pos_{i:04d}_v{int(round(voxel * 1000))}mm.npz")
    if not os.path.isfile(f):
        raise SystemExit(f"{f} missing: build the cache first (path_emitters.py --sets ... --cache {cache})")
    z = np.load(f)
    p = z["p"].astype(np.float64)
    c = z["s"] / np.maximum(p, 1e-300)[:, None]
    return p, c, int(z["n_paths"])


def peak_rows(positions, set_of, truth, views, cache, voxel, dirs, sigma_deg):
    """One row per face of each position: distinct flag, the main-peak source's centroid, its agreement angle and
    its share of the power at the peak direction."""
    rows, no_paths = [], []
    for i in positions:
        p, c, n_paths = load_position(cache, i, voxel)
        if n_paths == 0 or len(p) == 0:
            no_paths.append(i); continue
        for j in range(4):
            name = f"{4 * i + j + 1:05d}"                   # 1-based, position-major, face-minor
            t = np.load(os.path.join(truth, "spectra_float", name + ".npy")).astype(np.float64)
            if t.max() - t.min() < 1e-3:
                rows.append({"set": set_of(i), "position": i, "face": j, "flat": True}); continue
            kept = top_peaks(t, dirs, 3)
            distinct = len(kept) < 2 or t.flat[kept[0]] - t.flat[kept[1]] >= 1.0
            Rcw, rx = views[name]
            d_peak = Rcw @ dirs.reshape(-1, 3)[int(t.argmax())]
            v = c - rx[None]
            v /= np.linalg.norm(v, axis=1, keepdims=True).clip(1e-9)
            ang = np.degrees(np.arccos(np.clip(v @ d_peak, -1, 1)))
            contrib = p * np.exp(-0.5 * (ang / sigma_deg) ** 2)
            k = int(np.argmax(contrib))
            rows.append({"set": set_of(i), "position": i, "face": j, "flat": False, "distinct": bool(distinct),
                         "src": c[k].tolist(), "agree_deg": float(ang[k]), "top_share": float(contrib[k] / contrib.sum()),
                         "src_is_tx_voxel": False})
    return rows, no_paths


def power_coverage(positions, cache, voxel, tree):
    out = {}
    for i in positions:
        p, c, n = load_position(cache, i, voxel)
        if n == 0 or len(p) == 0:
            continue
        d = tree.query(c)[0]
        out[i] = {f"{r:.2f}": float(p[d <= r].sum() / p.sum()) for r in RADII}
    return out


def summarise(rows, tree, only_distinct=True):
    rr = [r for r in rows if not r["flat"] and (r["distinct"] or not only_distinct)]
    if not rr:
        return None
    d = tree.query(np.array([r["src"] for r in rr]))[0]
    return {"views": len(rr), **{f"covered_{r:.2f}m": float((d <= r).mean()) for r in RADII},
            "src_to_emitter_median_m": float(np.median(d)),
            "agree_deg_median": float(np.median([r["agree_deg"] for r in rr])),
            "top_share_median": float(np.median([r["top_share"] for r in rr]))}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--truth", required=True); ap.add_argument("--protocol", default="rrf_gsplat/protocol_v1")
    ap.add_argument("--cache", required=True); ap.add_argument("--emitters", required=True)
    ap.add_argument("--train-control", type=int, default=60)
    ap.add_argument("--out", default="output/rrf/m3/g1_coverage.json")
    a = ap.parse_args()
    from scipy.spatial import cKDTree
    z = np.load(a.emitters); em = z["means"].astype(np.float64); voxel = float(z["voxel"])
    meta = json.load(open(os.path.join(a.truth, "generation_meta.json")))
    sigma_deg = float(meta.get("splat_sigma", 3.0)) / 3.0     # aps_splat: sigma in equirect pixels at 3 px per degree
    views = read_views(a.truth); dirs = pixel_dirs(a.truth)
    sets = {s: sorted({(int(n) - 1) // 4 for n in PR.eval_names(a.protocol, s)}) for s in ("val_random", "val_segment")}
    set_of = {i: s for s, ps in sets.items() for i in ps}
    val_pos = sorted(set_of)
    tr_all = sorted({(int(n) - 1) // 4 for n in PR.train_names(a.protocol)})
    tr_ctl = [tr_all[k] for k in np.linspace(0, len(tr_all) - 1, a.train_control).round().astype(int)]
    tree = cKDTree(em); tree_shift = cKDTree(em + np.array([1.0, 0.0, 0.0]))
    tx = np.asarray(meta["tx_loc"], dtype=np.float64)

    rows, no_paths = peak_rows(val_pos, set_of.get, a.truth, views, a.cache, voxel, dirs, sigma_deg)
    for r in rows:
        if not r["flat"]:
            r["src_is_tx_voxel"] = bool(np.linalg.norm(np.array(r["src"]) - tx) < voxel)
    rows_tr, no_paths_tr = peak_rows(tr_ctl, lambda i: "train", a.truth, views, a.cache, voxel, dirs, sigma_deg)
    res = {"emitters": a.emitters, "n_emitters": int(len(em)), "voxel_m": voxel, "sigma_deg": sigma_deg,
           "validation": {s: summarise([r for r in rows if r["set"] == s], tree) for s in sets},
           "validation_all_distinct": summarise(rows, tree),
           "validation_all_views": summarise(rows, tree, only_distinct=False),
           "train_control": summarise(rows_tr, tree),
           "known_failure_shift_1m": summarise(rows, tree_shift),
           "degenerate": {"val_positions_without_paths": no_paths, "val_flat_views": sum(r["flat"] for r in rows),
                          "val_views_not_distinct": sum((not r["flat"]) and (not r["distinct"]) for r in rows),
                          "val_sources_at_tx": sum(r.get("src_is_tx_voxel", False) for r in rows if not r["flat"] and r["distinct"]),
                          "train_control_positions_without_paths": no_paths_tr}}
    pc = power_coverage(val_pos, a.cache, voxel, tree)
    res["power_coverage"] = {s: {f"{r:.2f}m": float(np.median([pc[i][f"{r:.2f}"] for i in ps if i in pc])) for r in RADII} for s, ps in sets.items()}
    v, t, k = res["validation_all_distinct"], res["train_control"], res["known_failure_shift_1m"]
    res["pass"] = {"c1_agreement": v["agree_deg_median"] <= 1.0, "c2_train_self": t["covered_0.10m"] >= 0.95,
                   "c3_known_failure": k["covered_0.10m"] <= 0.10}
    cov10 = v["covered_0.10m"]
    res["reading"] = ("covers: placement is not what step 1 lacked" if cov10 >= 0.90 else
                      "placement gap: G2 should improve markedly over A" if cov10 < 0.60 else "mixed: read G2")
    json.dump({**res, "rows": rows}, open(a.out, "w"), indent=1)
    print(f"G1: {len(em):,} training emitters; {len(val_pos)} validation positions, {len(rows)} faces")
    for name, s in (("val_random", res["validation"]["val_random"]), ("val_segment", res["validation"]["val_segment"]),
                    ("validation (distinct)", v), ("validation (all views)", res["validation_all_views"]),
                    ("train control (distinct)", t), ("known failure: shifted 1 m", k)):
        if s is None:
            print(f"  {name}: no views"); continue
        print(f"  {name:28s} {s['views']:4d} views: main-peak source covered at 5 / 10 / 20 / 50 cm "
              f"{s['covered_0.05m']:.1%} / {s['covered_0.10m']:.1%} / {s['covered_0.20m']:.1%} / {s['covered_0.50m']:.1%}; "
              f"median distance {s['src_to_emitter_median_m']:.3f} m; agreement {s['agree_deg_median']:.2f} deg; top share {s['top_share_median']:.2f}")
    for s, d in res["power_coverage"].items():
        print(f"  power coverage {s:12s} (median over positions): " + " / ".join(f"{d[f'{r:.2f}m']:.1%}" for r in RADII))
    print(f"  degenerate: {res['degenerate']}")
    print(f"  c1 agreement {v['agree_deg_median']:.2f} deg (<= 1) -> {'PASS' if res['pass']['c1_agreement'] else 'FAIL'}; "
          f"c2 train self {t['covered_0.10m']:.1%} (>= 95 %) -> {'PASS' if res['pass']['c2_train_self'] else 'FAIL'}; "
          f"c3 shifted {k['covered_0.10m']:.1%} (<= 10 %) -> {'PASS' if res['pass']['c3_known_failure'] else 'FAIL'}")
    print(f"  reading (validation distinct, 10 cm = {cov10:.1%}): {res['reading']}")
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
