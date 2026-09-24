"""Round 45 table and decisions (rrf_gsplat/win_round45.sh): keep or drop each of the night's features.

Quality = PSNR(jet) on the 640 held-out views and mvdr_peaks' power at the true peak; time = the pure training
time (train_seconds_excl_running_eval: the running evaluations excluded, the final one not included), all as
3-seed means with the min..max across seeds. The decisions are the ones written in win_round45.sh before the run;
this script only applies them.

    python rrf_gsplat/r45_table.py
"""

from __future__ import annotations

import json
import os

import numpy as np

from dlss_ablation import RRF, one

VARIANTS = ["base", "nogeom", "pg", "pgng", "tf32off", "lm", "lmr"]
PREFIX = {"lmr": "r46"}          # round 46: LM with the trust region sized for a linear problem


def runs(v, n_seeds=3):
    out = []
    for s in range(n_seeds):
        name = f"{PREFIX.get(v, 'r45')}_{v}" + ("" if s == 0 else f"_s{s}")
        r = one(name)
        if r is None:
            continue
        res = json.load(open(os.path.join(RRF, name, "results.json")))
        r["train_s"] = res.get("train_seconds_excl_running_eval")
        r["name"] = name
        out.append(r)
    return out


def stat(rs, key):
    v = [r[key] for r in rs if r.get(key) is not None]
    return (float(np.mean(v)), float(np.min(v)), float(np.max(v)), len(v)) if v else (None, None, None, 0)


def cell(s, fmt="{:.2f}"):
    m, lo, hi, n = s
    if m is None:
        return "MISSING"
    return fmt.format(m) if n <= 1 else f"{fmt.format(m)} [{fmt.format(lo)}..{fmt.format(hi)}]"


def main():
    R = {v: runs(v) for v in VARIANTS}
    keys = [("psnr", "PSNR(jet)"), ("rmse", "RMSE dB"), ("at_true", "at true peak dB"), ("ang_med", "peak dir med deg"),
            ("ang_1deg", "<=1 deg %"), ("top3", "top-3 det %"), ("false", "false peaks %"), ("train_s", "train s")]
    print("| variant | seeds | " + " | ".join(k[1] for k in keys) + " |")
    print("|" + " --- |" * (len(keys) + 2))
    S = {}
    for v in VARIANTS:
        S[v] = {k: stat(R[v], k) for k, _ in keys}
        print(f"| {v} | {len(R[v])} | " + " | ".join(cell(S[v][k]) for k, _ in keys) + " |")

    def m(v, k):
        return S[v][k][0]

    def rng(v, k):
        return None if S[v][k][0] is None else S[v][k][2] - S[v][k][1]

    dec = {}
    complete = all(len(R[v]) == 3 for v in VARIANTS)
    # nogeom: quality within 0.05 dB of base (PSNR and at-true-peak), time >= 5 % shorter
    if all(m(v, "psnr") is not None for v in ("base", "nogeom")):
        dq = max(abs(m("nogeom", "psnr") - m("base", "psnr")), abs(m("nogeom", "at_true") - m("base", "at_true")))
        dt = 1 - m("nogeom", "train_s") / m("base", "train_s")
        dec["nogeom"] = {"quality_diff_db": dq, "time_saving": dt, "keep": bool(dq <= 0.05 and dt >= 0.05)}
    # pg: keep only if pgng is >= 5 % faster than nogeom at equal quality
    if all(m(v, "psnr") is not None for v in ("nogeom", "pgng")):
        dq = max(abs(m("pgng", "psnr") - m("nogeom", "psnr")), abs(m("pgng", "at_true") - m("nogeom", "at_true")))
        dt = 1 - m("pgng", "train_s") / m("nogeom", "train_s")
        dec["pg"] = {"quality_diff_db": dq, "time_saving_vs_nogeom": dt, "keep": bool(dq <= 0.05 and dt >= 0.05)}
    # tf32off: the default unless PSNR or at-true-peak is worse than base by more than base's seed range
    if all(m(v, "psnr") is not None for v in ("base", "tf32off")):
        worse_psnr = m("base", "psnr") - m("tf32off", "psnr"); worse_at = m("base", "at_true") - m("tf32off", "at_true")
        dec["tf32off"] = {"psnr_worse_by": worse_psnr, "base_psnr_range": rng("base", "psnr"),
                          "at_true_worse_by": worse_at, "base_at_true_range": rng("base", "at_true"),
                          "default": bool(worse_psnr <= rng("base", "psnr") and worse_at <= rng("base", "at_true"))}
    # lm (round 45, 3DGS-LM's radius settings) and lmr (round 46, Ceres' defaults): time <= 0.85 x tf32off,
    # PSNR >= tf32off - 0.05, at-true-peak within tf32off's seed range
    for v in ("lm", "lmr"):
        if all(m(w, "psnr") is not None for w in ("tf32off", v)):
            ratio = m(v, "train_s") / m("tf32off", "train_s")
            dpsnr = m(v, "psnr") - m("tf32off", "psnr"); dat = m(v, "at_true") - m("tf32off", "at_true")
            hist = [json.load(open(os.path.join(RRF, r["name"], "results.json"))).get("lm_history") or [] for r in R[v]]
            dec[v] = {"time_ratio": ratio, "psnr_diff": dpsnr, "at_true_diff": dat, "tf32off_at_true_range": rng("tf32off", "at_true"),
                      "gamma_at_max_share": float(np.mean([h["gamma"] >= 1.0 for hh in hist for h in hh])) if any(hist) else None,
                      "final_radius": [hh[-1]["radius"] for hh in hist if hh],
                      "keep": bool(ratio <= 0.85 and dpsnr >= -0.05 and abs(dat) <= rng("tf32off", "at_true"))}
    print("\ndecisions" + ("" if complete else " (PRELIMINARY: not every variant has 3 seeds)") + ":")
    for k, d in dec.items():
        print(f"  {k}: " + ", ".join(f"{a} {b:.4g}" if isinstance(b, float) else f"{a} {b}" for a, b in d.items()))
    # the S5 head runs (one seed each): base / nogeom / pg
    print("\nS5 head, one seed:")
    for v in ("S5_base", "S5_nogeom", "S5_pg"):
        r = one(f"r45_{v}")
        if r is None:
            print(f"  {v}: MISSING"); continue
        res = json.load(open(os.path.join(RRF, f"r45_{v}", "results.json")))
        print(f"  {v}: PSNR {r['psnr']:.3f}, at true peak {r.get('at_true', float('nan')):+.2f} dB, "
              f"train {res.get('train_seconds_excl_running_eval', float('nan')):.1f} s")
    json.dump({"stats": {v: {k: S[v][k] for k, _ in keys} for v in VARIANTS}, "decisions": dec, "complete": complete,
               "runs": {v: [r["name"] for r in R[v]] for v in VARIANTS}},
              open(os.path.join(RRF, "r45_table.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
