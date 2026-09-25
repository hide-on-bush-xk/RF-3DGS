"""Summary of the direction-loss capacity benchmark on Falcon (dir_loss_capacity.sh; cluster_handover.md §6).

    ~/envs/rf-gsplat/bin/python rrf_gsplat/cluster/dir_loss_capacity_summary.py

Reading, written before the run (cluster_handover.md §6, 2026-09-25):
  - an arm "helps" if its 3-seed mean of training-position distinct <= 1 deg beats base's by more than both arms'
    seed ranges (max - min); otherwise "no difference" (the rule of stage2_notes §12);
  - the target line: training-position distinct <= 1 deg >= 63 % (the look-ups' level on the interpolation part);
    reaching it, or an arm clearly approaching it, leads to the validation set -- discussed with Ke first;
  - reported beside: the pixel fit (training-view RMSE) and the power error at the true peak, to see whether a
    better direction costs level.
Timing per run (the job owns its A30): load + training (of which live evaluations) + final evaluation, the
training-fit diagnostic, and the median step time over live.jsonl's 50-step windows after step 200; a step = one
position's 4 faces at 300 x 200. Data generation was done on the PC: MISSING.
Prints a markdown table per seed and per arm (mean [min, max]); writes output/rrf/dirloss_cap_falcon/summary.json.
"""

from __future__ import annotations

import json
import os
import statistics

OUT = os.path.join(os.path.dirname(__file__), "..", "..", "output", "rrf", "dirloss_cap_falcon")
ARMS = ("base", "eg", "eg3", "ce", "ce3")
SEEDS = (0, 1, 2)
TARGET = 63.0


def load(arm, seed):
    d = os.path.join(OUT, f"{arm}_s{seed}")
    try:
        res = json.load(open(os.path.join(d, "results.json")))
        fit = json.load(open(d + "_trainfit.json"))["train"]
    except (OSError, ValueError, KeyError):
        return None
    wall = json.load(open(os.path.join(d, "wall.json"))) if os.path.exists(os.path.join(d, "wall.json")) else {}
    win = [json.loads(l) for l in open(os.path.join(d, "live.jsonl"))]
    it_s = [w["it_s"] for w in win if "it_s" in w and w["it"] > 200]
    return {"distinct": 100 * fit["distinct_within_1deg"], "distinct_views": fit["distinct_views"], "views": fit["views"],
            "angle": fit["distinct_angle_median"], "rmse": fit["rmse_db"], "at_true": fit["at_true_median"],
            "iterations": res["iterations_run"], "visits": res["visits_per_view"], "stopped_early": res["stopped_early"],
            "stop_reason": res["stop_reason"], "gpu": res["gpu"], "load_s": res["load_seconds"],
            "train_s": res["train_seconds"], "running_eval_s": res["running_eval_seconds"], "eval_s": res["eval_seconds"],
            "trainfit_s": wall.get("trainfit_wall_s"), "step_ms": 1000 / statistics.median(it_s) if it_s else None,
            "monitor_last": sum(v for _, v in res["dir_monitor_db"][-3:]) / 3 if res.get("dir_monitor_db") else None}


def f(v, spec, unit=""):
    return "MISSING" if v is None else f"{v:{spec}}{unit}"


def agg(vals):
    return statistics.mean(vals), min(vals), max(vals)


def main():
    runs = {(a, s): load(a, s) for a in ARMS for s in SEEDS}
    missing = [f"{a}_s{s}" for (a, s), r in runs.items() if r is None]
    print("### Per run\n")
    print("| run | distinct <= 1° | distinct views | main-peak angle median | train RMSE | at true peak | monitor (last 300) "
          "| steps (visits/view) | early stop | load + train (live eval) + final eval + trainfit, s | step ms (median) |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for (a, s), r in runs.items():
        if r is None:
            print(f"| {a}_s{s} | MISSING | | | | | | | | | |")
            continue
        tf = f(r["trainfit_s"], ".0f")
        print(f"| {a}_s{s} | {r['distinct']:.1f} % | {r['distinct_views']}/{r['views']} | {r['angle']:.2f}° | {r['rmse']:.2f} dB "
              f"| {r['at_true']:+.2f} dB | {f(r['monitor_last'], '.2f', ' dB')} | {r['iterations']} ({r['visits']:.0f}) "
              f"| {'yes' if r['stopped_early'] else 'no'} | {r['load_s']:.0f} + {r['train_s']:.0f} ({r['running_eval_s']:.0f}) "
              f"+ {r['eval_s']:.0f} + {tf} | {f(r['step_ms'], '.2f')} |")
    print("\n### Per arm: 3-seed mean [min, max]; reading against base\n")
    print("| arm | seeds | distinct <= 1° | vs base (mean diff; ranges arm / base) | reading | ≥ 63 %? | train RMSE | at true peak | angle median |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    table = {}
    base = [runs[("base", s)] for s in SEEDS if runs[("base", s)] is not None]
    for a in ARMS:
        rs = [runs[(a, s)] for s in SEEDS if runs[(a, s)] is not None]
        if not rs:
            print(f"| {a} | 0 | MISSING | | | | | | |")
            continue
        m, lo, hi = agg([r["distinct"] for r in rs])
        rm, rlo, rhi = agg([r["rmse"] for r in rs])
        tm, tlo, thi = agg([r["at_true"] for r in rs])
        am, alo, ahi = agg([r["angle"] for r in rs])
        verdict, cmp = "N/A", "N/A"
        if a != "base" and len(rs) == 3 and len(base) == 3:
            bm, blo, bhi = agg([r["distinct"] for r in base])
            diff = m - bm
            verdict = "helps" if diff > max(hi - lo, bhi - blo) else "no difference"
            cmp = f"{diff:+.1f} pts; {hi - lo:.1f} / {bhi - blo:.1f}"
        elif a != "base":
            verdict = "N/A (fewer than 3 seeds)"
        table[a] = {"seeds": len(rs), "distinct": [m, lo, hi], "rmse": [rm, rlo, rhi], "at_true": [tm, tlo, thi],
                    "angle": [am, alo, ahi], "reading": verdict}
        print(f"| {a} | {len(rs)} | {m:.1f} % [{lo:.1f}, {hi:.1f}] | {cmp} | {verdict} | {'yes' if m >= TARGET else 'no'} "
              f"| {rm:.2f} dB [{rlo:.2f}, {rhi:.2f}] | {tm:+.2f} dB [{tlo:+.2f}, {thi:+.2f}] | {am:.2f}° [{alo:.2f}, {ahi:.2f}] |")
    gpus = sorted({r["gpu"] for r in runs.values() if r is not None})
    print(f"\nGPU: {', '.join(gpus)}; data generation MISSING (done on the PC); missing runs: {', '.join(missing) or 'none'}")
    json.dump({"runs": {f"{a}_s{s}": r for (a, s), r in runs.items()}, "arms": table, "missing": missing},
              open(os.path.join(OUT, "summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
