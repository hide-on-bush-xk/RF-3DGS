"""Validation-set comparison (docs/cluster_log.md §7): V-plain and V-G1 (3 seeds each) against the look-ups, with the
readings written before the run.

    nice -n 10 ~/envs/rf-gsplat/bin/python rrf_gsplat/cluster/val_eval.py

Fields: output/rrf/val_falcon/peaks_<arm>_s<seed>_<set>.json (mvdr_peaks.py; distinct views, main peak <= 1 deg).
Look-ups recomputed here on the same views with the same distinct rule: nearest neighbour (same face of the nearest of
the 467 training positions) and IDW-2 (the 2 nearest, blended in linear power with weights 1/d), argmax as prediction.
Writes output/rrf/val_falcon/summary.json.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ladder_step1 as L                                          # noqa: E402

OUT = os.path.join(L.REPO, "output", "rrf", "val_falcon")
SETS = ("val", "val_random", "val_segment")
ARMS, SEEDS = ("plain", "G1"), (0, 1, 2)


def lookups():
    info = L.view_info()
    pool = {}
    for g in L.GROUPS:
        for n in g:
            pool.setdefault(tuple(np.round(L.POSES[n][0][2], 2)), []).append(n)
    arr = {k: (v, np.array([L.POSES[m][2] for m in v])) for k, v in pool.items()}
    res = {}
    for es in SETS:
        names = L.PR.eval_names(os.path.join(L.REPO, "rrf_gsplat", "protocol_v1"), es)
        hit = {"nn": [], "idw2": []}; ang = {"nn": [], "idw2": []}
        for n in names:
            t = L.truth(n)
            if t.max() < -250 or t.max() - t.min() < 1e-3:
                continue
            kept = L.top_peaks(t, L.DIRS, 3)
            if not (len(kept) < 2 or t.flat[kept[0]] - t.flat[kept[1]] >= 1.0):
                continue
            kt = int(t.argmax())
            v, P = arr[tuple(np.round(L.POSES[n][0][2], 2))]
            dd = np.linalg.norm(P - L.POSES[n][2], axis=1)
            o = np.argsort(dd)
            kp_nn = info[v[o[0]]]["kt"]
            w = 1.0 / np.maximum(dd[o[:2]], 1e-6)
            lin = sum(wi * 10 ** (L.truth(v[j]) / 10) for wi, j in zip(w, o[:2])) / w.sum()
            kp_idw = int(lin.argmax())
            for key, kp in (("nn", kp_nn), ("idw2", kp_idw)):
                a = L.angle(L.DIRS, kt, kp); ang[key].append(a); hit[key].append(a <= 1.0)
        res[es] = {k: {"distinct_views": len(hit[k]), "within_1deg": 100 * float(np.mean(hit[k])),
                       "median_deg": float(np.median(ang[k]))} for k in hit}
    return res


def main():
    fields = {}
    for arm in ARMS:
        for sd in SEEDS:
            for es in SETS:
                p = os.path.join(OUT, f"peaks_{arm}_s{sd}_{es}.json")
                if os.path.exists(p):
                    d = json.load(open(p))["main_peak_angle_deg_distinct"]
                    fields.setdefault(arm, {}).setdefault(es, {})[sd] = {"within_1deg": 100 * d["within_1deg"], "median_deg": d["median"],
                                                                          "views": d["views"]}
    lk = lookups()
    stat = lambda v: {"mean": float(np.mean(v)), "min": float(min(v)), "max": float(max(v)), "values": [float(x) for x in v]}
    tab = {arm: {es: stat([fields[arm][es][s]["within_1deg"] for s in sorted(fields[arm][es])]) for es in fields[arm]} for arm in fields}
    med = {arm: {es: stat([fields[arm][es][s]["median_deg"] for s in sorted(fields[arm][es])]) for es in fields[arm]} for arm in fields}
    rd = {}
    best = max(lk["val"]["nn"]["within_1deg"], lk["val"]["idw2"]["within_1deg"])
    if "G1" in tab and len(tab["G1"].get("val", {}).get("values", [])) == 3:
        g = tab["G1"]["val"]["mean"]
        rd["V1"] = {"G1_val_mean": g, "best_lookup": best, "margin": g - best,
                    "reading": "G1 beats the look-ups on new positions" if g - best > 3 else "G1 does not beat the look-ups",
                    "val_random_vs_best_lookup": tab["G1"]["val_random"]["mean"] - max(lk["val_random"][k]["within_1deg"] for k in ("nn", "idw2")),
                    "val_segment_vs_best_lookup": tab["G1"]["val_segment"]["mean"] - max(lk["val_segment"][k]["within_1deg"] for k in ("nn", "idw2"))}
    if all(a in tab and len(tab[a].get("val", {}).get("values", [])) == 3 for a in ARMS):
        va, vb = tab["G1"]["val"]["values"], tab["plain"]["val"]["values"]
        d = float(np.mean(va) - np.mean(vb)); ra, rb = max(va) - min(va), max(vb) - min(vb)
        rd["V2"] = {"mean_diff": d, "range_G1": ra, "range_plain": rb, "holds": bool(abs(d) > max(ra, rb))}
    json.dump({"fields": tab, "field_median_deg": med, "lookups": lk, "readings": rd, "per_seed": fields},
              open(os.path.join(OUT, "summary.json"), "w"), indent=1)
    print("look-ups:", json.dumps(lk, indent=1))
    for arm in tab:
        print(arm, {es: f"{v['mean']:.1f} [{v['min']:.1f}, {v['max']:.1f}]" for es, v in tab[arm].items()})
    print("READINGS:", json.dumps(rd, indent=1))


if __name__ == "__main__":
    main()
