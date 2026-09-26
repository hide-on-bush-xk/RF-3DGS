"""Information ladder, step 3 (docs/cluster_log.md §6): readings G-a..G-d for the geometry-only mirror emitters (G1),
against A0 / A7 (seed 0) and the look-ups on the same 916 held-out distinct views. Run ladder_step2_eval.py first.

    nice -n 10 ~/envs/rf-gsplat/bin/python rrf_gsplat/cluster/ladder_step3_eval.py
"""

from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ladder_step1 as L                                          # noqa: E402

S = os.path.join(L.REPO, "output", "cluster", "ladder", "step2", "summary.json")
NN = {20: 11.7, 160: 40.8}                                        # nearest neighbour, same held-out views (kcurve_lookup)
C = ("ringmax", "weak_image", "weak_other")


def pooled(e, key="hit"):
    n = sum(e["heldout"][c]["n"] for c in C)
    return sum(e["heldout"][c][key] * e["heldout"][c]["n"] for c in C) / n


def main():
    r = json.load(open(S))["results"]
    rd = {}
    g160, a7, a0 = r.get("G1_k160"), r.get("A7_k160"), r.get("A0_k160")
    if g160 and a7 and a0:
        gp, ap, zp = pooled(g160), pooled(a7), pooled(a0)
        rd["G-a"] = {"G1_pooled": gp, "A7_pooled": ap, "A0_pooled": zp,
                     "reading": "geometry replaces the RF-fitted walls" if gp >= ap - 5 else "fail" if gp < zp + 10 else "between"}
        dh = {c: g160["heldout"][c]["halluc"] - a0["heldout"][c]["halluc"] for c in C}
        gain = g160["heldout"]["weak_image"]["hit"] - a0["heldout"]["weak_image"]["hit"]
        rd["G-b"] = {"weak_image_gain": gain, "halluc_rise": dh, "pass": gain >= 10 and all(v <= 3 for v in dh.values())}
        rd["G-d"] = {"G1_pooled": gp, "lookup": NN[160], "pass": gp >= NN[160] + 10}
    g20, a720 = r.get("G1_k20"), r.get("A7_k20")
    if g20 and a720:
        rd["G-c"] = {"G1_pooled_k20": pooled(g20), "A7_pooled_k20": pooled(a720), "nn_k20": NN[20],
                     "pass": pooled(g20) >= pooled(a720) - 5 and pooled(g20) >= NN[20] + 20}
    rows = {k: {"pooled": pooled(v), "pooled_bits": pooled(v, "bits"), "pooled_halluc": pooled(v, "halluc"),
                "heldout": {c: v["heldout"][c]["hit"] for c in C}, "insample": {c: (v["insample"][c]["hit"] if v["insample"][c] else None) for c in C}}
            for k, v in r.items() if k.startswith(("G1_", "A7_", "A0_", "A8_"))}
    json.dump({"readings": rd, "rows": rows}, open(os.path.join(L.REPO, "output", "cluster", "ladder", "step3", "readings.json"), "w"), indent=1)
    for k, v in rows.items():
        print(f"{k:8s} held-out pooled {v['pooled']:5.1f} % ({v['pooled_bits']:.2f} bit, halluc {v['pooled_halluc']:.1f} %) | held-out "
              + " / ".join(f"{v['heldout'][c]:.1f}" for c in C))
    print(json.dumps(rd, indent=1))


if __name__ == "__main__":
    main()
