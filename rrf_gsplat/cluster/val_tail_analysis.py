"""Where does the beam-gain-loss tail of E-db (and G1) come from? Validation views, no training (docs/cluster_log.md §8).

    nice -n 10 ~/envs/rf-gsplat/bin/python rrf_gsplat/cluster/val_tail_analysis.py

Per view (480): beam-gain loss of E-db (3 seeds), G1 (3 seeds), IDW-2, IDW-8, NN (t4_scores_val*.json), and the loss of
a beam steered to the TRUE APS main peak (beam_gain_loss_truth_peak: what a perfect spectrum would still lose).
Tail = beam-gain loss > 8 dB in >= 2 of the 3 E-db seeds. Tail views are described by: face class (ring-max face /
mirror-explained by one of G1's 11 geometric emitters within 1 deg / other), distinct or not, val_random or
val_segment, distance to the nearest training position, where E-db's predicted peak lies (within 1 deg of a G1 emitter
direction or not), and whether the look-ups fail there too.
Writes output/rrf/val_falcon/tail_analysis.json.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ladder_step1 as L                                          # noqa: E402

V = os.path.join(L.REPO, "output", "rrf", "val_falcon")
PROT = os.path.join(L.REPO, "rrf_gsplat", "protocol_v1")
TAIL_DB = 8.0


def rows(fname, run):
    return {r["view"]: r for r in json.load(open(os.path.join(V, fname)))[run]["views"]}


def main():
    info = L.view_info() if False else None
    em = np.load(os.path.join(L.REPO, "output", "cluster", "ladder", "step3", "prep", "emit_geo.npz"))["means"]
    names = L.PR.eval_names(PROT, "val")
    vr = set(L.PR.eval_names(PROT, "val_random"))
    E = [rows("t4_scores_val_ens.json", f"ens_db_G1_s{s}") for s in (0, 1, 2)]
    G = [rows("t4_scores_val.json", f"G1_s{s}") for s in (0, 1, 2)]
    I2, I8, NN = (rows("t4_scores_val.json", r) for r in ("lookup_idw2", "lookup_idw8", "lookup_nn"))
    train_c = np.stack([L.POSES[g[0]][2] for g in L.GROUPS])
    groups = {}
    for n in names:
        groups.setdefault(tuple(np.round(L.POSES[n][2], 4)), []).append(n)
    ringmax = {max(g, key=lambda m: L.truth(m).max()) for g in groups.values()}
    out = []
    for n in names:
        if n not in E[0]:
            continue                                              # flat truth, skipped by t4_score
        t = L.truth(n)
        kt = int(t.argmax())
        R, _, c = L.POSES[n]
        d = R.T @ L.FLAT[kt]
        u = em - c; u /= np.linalg.norm(u, axis=1, keepdims=True)
        mirror = bool((np.degrees(np.arccos(np.clip(u @ d, -1, 1))) <= 1.0).any())
        kept = L.top_peaks(t, L.DIRS, 3)
        distinct = bool(len(kept) < 2 or t.flat[kept[0]] - t.flat[kept[1]] >= 1.0)
        # E-db seed-0 predicted peak: at a G1 emitter direction?
        p = np.load(os.path.join(V, "ens_db_G1_s0", "renders", f"{n}.npy"))
        dp = R.T @ L.FLAT[int(p.argmax())]
        pred_on_emitter = bool((np.degrees(np.arccos(np.clip(u @ dp, -1, 1))) <= 1.0).any())
        out.append({"view": n, "split": "val_random" if n in vr else "val_segment",
                    "cls": "ringmax" if n in ringmax else ("mirror" if mirror else "other"), "distinct": distinct,
                    "dist_train_m": float(np.linalg.norm(train_c - c, axis=1).min()),
                    "e": [e[n]["beam_gain_loss"] for e in E], "g": [g[n]["beam_gain_loss"] for g in G],
                    "idw2": I2[n]["beam_gain_loss"], "idw8": I8[n]["beam_gain_loss"], "nn": NN[n]["beam_gain_loss"],
                    "truth_peak": E[0][n]["beam_gain_loss_truth_peak"], "e_angle": [e[n]["angle"] for e in E],
                    "pred_on_emitter": pred_on_emitter})
    tail = [r for r in out if sum(x > TAIL_DB for x in r["e"]) >= 2]
    rest = [r for r in out if r not in tail]

    def describe(rs):
        n = len(rs)
        if not n:
            return {"n": 0}
        f = lambda cond: round(100 * sum(cond(r) for r in rs) / n, 1)   # noqa: E731
        return {"n": n,
                "cls_ringmax_mirror_other_pct": [f(lambda r: r["cls"] == k) for k in ("ringmax", "mirror", "other")],
                "distinct_pct": f(lambda r: r["distinct"]), "val_segment_pct": f(lambda r: r["split"] == "val_segment"),
                "dist_train_median_m": round(float(np.median([r["dist_train_m"] for r in rs])), 2),
                "truth_peak_loss_median_db": round(float(np.median([r["truth_peak"] for r in rs])), 2),
                "truth_peak_loss_gt8_pct": f(lambda r: r["truth_peak"] > TAIL_DB),
                "idw8_gt8_pct": f(lambda r: r["idw8"] > TAIL_DB), "idw2_gt8_pct": f(lambda r: r["idw2"] > TAIL_DB),
                "nn_gt8_pct": f(lambda r: r["nn"] > TAIL_DB), "g1_gt8_2of3_pct": f(lambda r: sum(x > TAIL_DB for x in r["g"]) >= 2),
                "e_pred_on_g1_emitter_pct": f(lambda r: r["pred_on_emitter"]),
                "e_angle_median_deg": round(float(np.median([np.median(r["e_angle"]) for r in rs])), 2)}
    res = {"tail_threshold_db": TAIL_DB, "views_scored": len(out), "tail": describe(tail), "rest": describe(rest),
           "tail_by_class": {k: describe([r for r in tail if r["cls"] == k]) for k in ("ringmax", "mirror", "other")},
           "all_by_class_counts": {k: sum(r["cls"] == k for r in out) for k in ("ringmax", "mirror", "other")},
           "idw8_tail_overlap": {"idw8_gt8": sum(r["idw8"] > TAIL_DB for r in out),
                                 "both": sum(r["idw8"] > TAIL_DB for r in tail)},
           "views": out}
    json.dump(res, open(os.path.join(V, "tail_analysis.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "views"}, indent=1))


if __name__ == "__main__":
    main()
