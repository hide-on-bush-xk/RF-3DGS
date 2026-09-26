"""Information ladder, step 2: the 3-seed confirmation at k = 160 (docs/cluster_log.md §5, readings written before the
run). Seeds 0 (k160_r0), 1 (k160_r0_s1), 2 (k160_r0_s2) of every arm; per seed the in-sample and held-out class scores
(as ladder_step2_eval.py) plus the pooled held-out score over all distinct views; mean [min, max]; the §12 rule:
a difference holds if the 3-seed mean difference exceeds both arms' seed ranges and has seed 0's sign.

    nice -n 10 ~/envs/rf-gsplat/bin/python rrf_gsplat/cluster/ladder_step2_seeds.py
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ladder_step2_eval as E                     # noqa: E402

ARMS = E.ARMS
SEEDS = {0: "k160_r0", 1: "k160_r0_s1", 2: "k160_r0_s2"}
LOOKUP_POOLED = 40.8                               # nearest neighbour from the 160 route positions, same 916 views


def run_scores(run, held):
    cfg = json.load(open(os.path.join(run, "results.json")))["config"]
    nf = cfg["train_names_file"]
    nf = nf if os.path.isabs(nf) else os.path.join(E.L.REPO, nf)
    names = [l.strip() for l in open(nf) if l.strip()]
    ins = E.pool([E.score_dir(os.path.join(run + "_insample", "renders"), names)])
    ho_rows = E.score_dir(os.path.join(run + "_heldout", "renders"), held)
    ho = E.pool([ho_rows])
    allr = [r for c in E.CLASSES for r in ho_rows[c]]
    pooled = {"hit": 100 * float(np.mean([r["hit"] for r in allr])), "bits": float(np.mean([r["bits"] for r in allr])),
              "halluc": 100 * float(np.mean([r["halluc"] for r in allr]))}
    return {"insample": ins, "heldout": ho, "pooled_heldout": pooled, "seed_in_config": cfg["seed"]}


def main():
    held = [l.strip() for l in open(os.path.join(E.PREP, "heldout.txt")) if l.strip()]
    per = {}
    for arm in ARMS:
        for sd, tag in SEEDS.items():
            r = os.path.join(E.RUNS, arm, tag)
            if os.path.isfile(r + "_heldout.json") and os.path.isfile(r + "_insample.json"):
                per.setdefault(arm, {})[sd] = run_scores(r, held)
                assert per[arm][sd]["seed_in_config"] == sd, (arm, sd)
        print(arm, "seeds", sorted(per.get(arm, {})), flush=True)

    def vals(arm, split, c, key="hit"):
        if arm not in per or len(per[arm]) < 3:
            return None
        return [per[arm][s][split][c][key] if split != "pooled_heldout" else per[arm][s][split][key] for s in (0, 1, 2)]

    def stat(v):
        return None if v is None else {"mean": float(np.mean(v)), "min": float(min(v)), "max": float(max(v)), "values": v}

    def rule(a, b, split, c, key="hit"):
        va, vb = vals(a, split, c, key), vals(b, split, c, key)
        if va is None or vb is None:
            return None
        d = float(np.mean(va) - np.mean(vb)); ra, rb = max(va) - min(va), max(vb) - min(vb)
        return {"a": a, "b": b, "split": split, "class": c, "key": key, "mean_diff": d, "range_a": ra, "range_b": rb,
                "seed0_diff": float(va[0] - vb[0]), "holds": bool(abs(d) > max(ra, rb) and np.sign(d) == np.sign(va[0] - vb[0]))}

    table = {arm: {"insample": {c: stat(vals(arm, "insample", c)) for c in E.CLASSES},
                   "heldout": {c: stat(vals(arm, "heldout", c)) for c in E.CLASSES},
                   "heldout_halluc": {c: stat(vals(arm, "heldout", c, "halluc")) for c in E.CLASSES},
                   "insample_weak_image_peak_err": stat(vals(arm, "insample", "weak_image", "peak_err_med")),
                   "pooled_heldout": stat(vals(arm, "pooled_heldout", None)),
                   "pooled_heldout_bits": stat(vals(arm, "pooled_heldout", None, "bits")),
                   "pooled_heldout_halluc": stat(vals(arm, "pooled_heldout", None, "halluc"))} for arm in per}
    rd = {}
    a4 = table.get("A4")
    if a4 and a4["insample"]["weak_image"]:
        rd["K1_A4_threshold"] = {"mean": a4["insample"]["weak_image"]["mean"], "peak_err_mean": a4["insample_weak_image_peak_err"]["mean"],
                                 "pass": a4["insample"]["weak_image"]["mean"] >= 30 and a4["insample_weak_image_peak_err"]["mean"] >= -15}
    rd["K1_A4_vs_A0"] = rule("A4", "A0", "insample", "weak_image")
    rd["K1_A2_vs_A0"] = rule("A2", "A0", "insample", "weak_image")
    if table.get("A7") and table.get("A0") and table["A7"]["heldout"]["weak_image"]:
        dh = {c: table["A7"]["heldout_halluc"][c]["mean"] - table["A0"]["heldout_halluc"][c]["mean"] for c in E.CLASSES}
        rd["K3_A7_thresholds"] = {"weak_image_gain": table["A7"]["heldout"]["weak_image"]["mean"] - table["A0"]["heldout"]["weak_image"]["mean"],
                                  "halluc_rise": dh, "pass": (table["A7"]["heldout"]["weak_image"]["mean"] - table["A0"]["heldout"]["weak_image"]["mean"] >= 10
                                                               and all(v <= 3 for v in dh.values()))}
        rd["K3_A7_pooled_vs_lookup"] = {"A7_pooled_mean": table["A7"]["pooled_heldout"]["mean"], "lookup": LOOKUP_POOLED,
                                        "A7_min_above_lookup": table["A7"]["pooled_heldout"]["min"] > LOOKUP_POOLED}
    rd["K3_A7_vs_A0_weak_image"] = rule("A7", "A0", "heldout", "weak_image")
    rd["K3_A7_vs_A0_pooled"] = rule("A7", "A0", "pooled_heldout", None)
    rd["K7_A7_vs_A8"] = rule("A7", "A8", "heldout", "weak_image")
    rd["K7_A7_vs_A4"] = rule("A7", "A4", "heldout", "weak_image")
    rd["K4_A1_vs_A0_halluc_weak_image"] = rule("A1", "A0", "heldout", "weak_image", "halluc")
    rd["K4_A5_vs_A4"] = rule("A5", "A4", "heldout", "weak_image")
    rd["K6_A6_vs_A2"] = rule("A6", "A2", "insample", "weak_image")
    json.dump({"table": table, "readings": rd}, open(os.path.join(E.OUT, "seeds.json"), "w"), indent=1)
    print("\n| arm | in-sample weak_image | held-out weak_image | held-out pooled (916) | pooled halluc |")
    print("| --- | --- | --- | --- | --- |")
    f = lambda s: "—" if s is None else f"{s['mean']:.1f} [{s['min']:.1f}, {s['max']:.1f}]"
    for arm, t in table.items():
        print(f"| {arm} | {f(t['insample']['weak_image'])} | {f(t['heldout']['weak_image'])} | {f(t['pooled_heldout'])} | {f(t['pooled_heldout_halluc'])} |")
    print("\nREADINGS:\n" + json.dumps(rd, indent=1, default=float))


if __name__ == "__main__":
    main()
