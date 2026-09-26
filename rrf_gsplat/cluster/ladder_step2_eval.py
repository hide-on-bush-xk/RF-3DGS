"""Information ladder, step 2: score the k-position curves by face class and apply the readings written before the
run (docs/cluster_log.md §5).

    nice -n 10 ~/envs/rf-gsplat/bin/python rrf_gsplat/cluster/ladder_step2_eval.py

Runs: output/rrf/ladder_step2/<ARM>/k<K>_r<R>/ with <run>_insample/renders and <run>_heldout/renders (diag_ladder.py).
Classes (scoring only, never used in training): ring-max face = the position's brightest face; weak_image = another
distinct face whose true peak lies within 1 deg of the transmitter or a mirror image fitted on the route views
(ladder_step1.fit_images); weak_other = the remaining distinct faces. Per class: <= 1 deg, bits, hallucination (the
predicted peak > 1 deg off where the truth is >= 10 dB under its maximum), power error at the true peak (dB),
steering loss (truth max - truth at the predicted peak, dB; an array-free stand-in for the beam-gain loss).
Writes output/cluster/ladder/step2/summary.json and prints the tables and the readings.
"""

from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ladder_step1 as L                          # noqa: E402

RUNS = os.path.join(L.REPO, "output", "rrf", "ladder_step2")
PREP = os.path.join(L.REPO, "output", "cluster", "ladder", "step2", "prep")
OUT = os.path.join(L.REPO, "output", "cluster", "ladder", "step2")
ARMS = ("A0", "A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "G1")
KS = (1, 2, 5, 20, 160)
CLASSES = ("ringmax", "weak_image", "weak_other")
INFO = L.view_info()
IMAGES = L.fit_images(INFO, L.distinct_views(INFO))
_TRUTH = {}


def truth(n):
    if n not in _TRUTH:
        _TRUTH[n] = L.truth(n)
    return _TRUTH[n]


def classify(names):
    groups = {}
    for n in names:
        groups.setdefault(tuple(np.round(L.POSES[n][2], 4)), []).append(n)
    out = {}
    for g in groups.values():
        assert len(g) == 4, "every position must bring its 4 faces"
        rm = max(g, key=lambda n: INFO[n]["max"])
        for n in g:
            if INFO[n]["flat"] or not INFO[n]["distinct"]:
                continue
            out[n] = "ringmax" if n == rm else ("weak_image" if L.image_explains(INFO, n, IMAGES) else "weak_other")
    return out


def view_metrics(n, pred):
    t = truth(n)
    kt, kp = INFO[n]["kt"], int(pred.argmax())
    eps = L.angle(L.DIRS, kt, kp)
    return {"eps": eps, "bits": float(L.bits(eps)), "hit": eps <= 1.0,
            "halluc": bool(eps > 1.0 and t.flat[kp] <= t.max() - 10.0),
            "peak_err": float(pred.flat[kt] - t.flat[kt]), "steer": float(t.max() - t.flat[kp])}


def score_dir(render_dir, names):
    cls = classify(names)
    rows = {c: [] for c in CLASSES}
    for n, c in cls.items():
        rows[c].append(view_metrics(n, np.load(os.path.join(render_dir, f"{n}.npy"))))
    return rows


def pool(rowsets):
    out = {}
    for c in CLASSES:
        rs = [r for rows in rowsets for r in rows[c]]
        if not rs:
            out[c] = None
            continue
        out[c] = {"n": len(rs), "hit": 100 * float(np.mean([r["hit"] for r in rs])),
                  "bits": float(np.mean([r["bits"] for r in rs])), "halluc": 100 * float(np.mean([r["halluc"] for r in rs])),
                  "peak_err_med": float(np.median([r["peak_err"] for r in rs])),
                  "steer_med": float(np.median([r["steer"] for r in rs]))}
    return out


def main():
    held = [l.strip() for l in open(os.path.join(PREP, "heldout.txt")) if l.strip()]
    res = {}
    for arm in ARMS:
        for k in KS:
            runs = sorted(glob.glob(os.path.join(RUNS, arm, f"k{k}_r*")))
            # seed 0 only (other seeds end in _s<N>); finished = both diag JSONs written (they come after all renders)
            runs = [r for r in runs if os.path.basename(r)[len(f"k{k}_r"):].isdigit()
                    and os.path.isfile(r + "_heldout.json") and os.path.isfile(r + "_insample.json")]
            planned = {1: 6, 2: 4, 5: 4, 20: 2, 160: 1}[k] if arm not in ("A6", "A7", "A8") else None     # G1 runs every k
            if planned and len(runs) < planned:
                print(f"WARNING {arm} k={k}: {len(runs)} of {planned} repeats finished", flush=True)
            if not runs:
                continue
            ins, ho, per, secs = [], [], [], []
            for r in runs:
                cfg = json.load(open(os.path.join(r, "results.json")))["config"]
                nf = cfg["train_names_file"]
                nf = nf if os.path.isabs(nf) else os.path.join(L.REPO, nf)
                names = [l.strip() for l in open(nf) if l.strip()]
                a = score_dir(os.path.join(r + "_insample", "renders"), names)
                ins.append(a)
                ho.append(score_dir(os.path.join(r + "_heldout", "renders"), held))
                per.append({c: (100 * float(np.mean([x["hit"] for x in a[c]])) if a[c] else None) for c in CLASSES})
                w = os.path.join(r, "wall.json")
                secs.append(json.load(open(w)) if os.path.exists(w) else None)
            res[f"{arm}_k{k}"] = {"runs": len(runs), "insample": pool(ins), "heldout": pool(ho), "per_run_insample_hit": per,
                                  "wall": secs}
            print(f"{arm} k={k}: {len(runs)} runs", flush=True)

    def g(arm, k, split, c, key="hit"):
        e = res.get(f"{arm}_k{k}")
        return None if e is None or e[split][c] is None else e[split][c][key]

    rd = {}
    a0r, a0w = g("A0", 160, "insample", "ringmax"), g("A0", 160, "insample", "weak_image")
    if a0r is not None:
        rd["K0"] = {"ringmax": a0r, "weak_image": a0w, "pass": 80 <= a0r <= 90 and 4 <= a0w <= 17}
    a4 = g("A4", 160, "insample", "weak_image")
    if a4 is not None:
        pe = g("A4", 160, "insample", "weak_image", "peak_err_med")
        rd["K1_A4"] = {"weak_image": a4, "peak_err_med": pe, "power_ok_ge_-15dB": pe >= -15,
                       "hit_level": "strong (>= 50)" if a4 >= 50 else "pass level (>= 30)" if a4 >= 30 else "fail (< 17)" if a4 < 17 else "between",
                       "reading": "strong" if a4 >= 50 and pe >= -15 else "pass" if a4 >= 30 and pe >= -15 else "fail" if a4 < 17 else "between",
                       "note": "strong read as pass conditions with >= 50 %; hit_level and power_ok reported separately"}
    a2 = g("A2", 160, "insample", "weak_image")
    if a2 is not None:
        rd["K1_A2"] = {"weak_image": a2, "predicted_below_17": a2 < 17, "refutes_sh_wrong_form": a2 >= 30}
    a0h = {c: g("A0", 160, "heldout", c) for c in CLASSES}
    a0hh = {c: g("A0", 160, "heldout", c, "halluc") for c in CLASSES}
    if a0r is not None:
        rd["K2_A0"] = {"insample_ringmax": a0r, "insample_weak_other": g("A0", 160, "insample", "weak_other"),
                       "pass": a0r >= 80 and g("A0", 160, "insample", "weak_other") >= 28}
    for arm in ARMS[1:]:
        if f"{arm}_k160" not in res:
            continue
        r = {"insample_ringmax": g(arm, 160, "insample", "ringmax"), "insample_weak_other": g(arm, 160, "insample", "weak_other"),
             "heldout": {c: g(arm, 160, "heldout", c) for c in CLASSES},
             "heldout_halluc": {c: g(arm, 160, "heldout", c, "halluc") for c in CLASSES}}
        if a0h["ringmax"] is not None:
            r["K2_pass"] = (r["insample_ringmax"] >= 80 and r["insample_weak_other"] >= 28
                            and r["heldout"]["ringmax"] >= a0h["ringmax"] - 5)
            dh = {c: r["heldout_halluc"][c] - a0hh[c] for c in CLASSES}
            r["K3_pass"] = r["heldout"]["weak_image"] >= a0h["weak_image"] + 10 and all(v <= 3 for v in dh.values())
            r["K3_beats_lookup_47.9"] = r["heldout"]["weak_image"] >= 47.9
            if arm in ("A2", "A3", "A6"):
                r["K3_sh_fail_halluc_gt5"] = any(v > 5 for v in dh.values())
        rd[f"K2_K3_{arm}"] = r
    if all(f"{a}_k160" in res for a in ("A0", "A1", "A4", "A5")):
        d54 = g("A5", 160, "heldout", "weak_image") - g("A4", 160, "heldout", "weak_image")
        d10 = g("A1", 160, "insample", "weak_image") - g("A0", 160, "insample", "weak_image")
        hh = max(g("A1", 160, "heldout", c, "halluc") - a0hh[c] for c in CLASSES)
        rd["K4"] = {"A5_minus_A4_heldout_weak_image": d54, "A1_minus_A0_insample_weak_image": d10, "A1_heldout_halluc_rise": hh,
                    "claim_holds": d54 >= -3 and (d10 < 7.3 or hh >= 5),
                    "loss_alone_suffices": g("A1", 160, "insample", "weak_image") >= 30 and hh <= 3}
        if all(f"{a}_k160" in res for a in ("A2", "A3")):
            d32 = g("A3", 160, "insample", "weak_image") - g("A2", 160, "insample", "weak_image")
            h32 = max(g("A3", 160, "heldout", c, "halluc") - g("A2", 160, "heldout", c, "halluc") for c in CLASSES)
            rd["K4"]["post_hoc_not_in_plan"] = {"A3_minus_A2_insample": d32, "A3_minus_A2_heldout_halluc_rise": h32}
    if "A0_k1" in res and "A0_k2" in res and "A0_k160" in res:
        rs = [x for k in (1, 2) for x in [res[f"A0_k{k}"]["insample"]["weak_image"]] if x]
        n = sum(x["n"] for x in rs)
        small = sum(x["hit"] * x["n"] for x in rs) / n if n else None
        rd["K5a"] = {"A0_weak_image_k_le_2": small, "n_views": n, "A0_weak_image_k160": a0w,
                     "reading": ("conflict between neighbouring receivers (sampling)" if small is not None and small >= 50 and a0w <= 17
                                 else "single views not representable (A7/A8 decide)" if small is not None and small < 30 else "between")}
    if "A4_k20" in res and "A4_k160" in res:
        rd["K5b"] = {"A4_k20": g("A4", 20, "insample", "weak_image"), "A4_k160": a4,
                     "pass": a4 >= g("A4", 20, "insample", "weak_image") - 15}
        curve = {k: g("A4", k, "heldout", "weak_image") for k in KS if f"A4_k{k}" in res}
        vals = [curve[k] for k in KS if k in curve]
        rises = all(b >= a for a, b in zip(vals, vals[1:]))
        rd["K5c"] = {"A4_heldout_weak_image_by_k": curve, "rises_with_k": rises,
                     "pass": rises and curve.get(20, 0) >= 16.5 and curve.get(160, 0) >= 40}
    if "A6_k160" in res and "A2_k160" in res:
        d = g("A6", 160, "insample", "weak_image") - a2
        hh = max(g("A6", 160, "heldout", c, "halluc") - g("A2", 160, "heldout", c, "halluc") for c in CLASSES)
        rd["K6"] = {"A6_minus_A2_insample": d, "halluc_rise": hh,
                    "reading": "dark-face labels add information" if d >= 7.3 and hh <= 3 else "dark-face selection adds nothing beyond geometry"}
    if all(f"{a}_k160" in res for a in ("A4", "A7", "A8")):
        a7, a8, a4h = (g(a, 160, "heldout", "weak_image") for a in ("A7", "A8", "A4"))
        rd["K7"] = {"A7": a7, "A8": a8, "A4": a4h, "image_depth_matters": a7 - a8 >= 10,
                    "surface_lobe_enough": a4h >= a7 - 10, "surface_anchoring_bottleneck": a7 - a4h > 20}
    json.dump({"results": res, "readings": rd}, open(os.path.join(OUT, "summary.json"), "w"), indent=1)
    print("\n| arm | k | in-sample ring / weak_image / weak_other | held-out ring / weak_image / weak_other | held-out halluc (w_img) |")
    print("| --- | --- | --- | --- | --- |")
    for key, e in res.items():
        arm, k = key.split("_k")
        f = lambda s: " / ".join("—" if e[s][c] is None else f"{e[s][c]['hit']:.1f}" for c in CLASSES)
        hw = e["heldout"]["weak_image"]
        hwv = "—" if hw is None else f"{hw['halluc']:.1f}"
        print(f"| {arm} | {k} | {f('insample')} | {f('heldout')} | {hwv} |")
    print("\nREADINGS:\n" + json.dumps(rd, indent=1))


if __name__ == "__main__":
    main()
