"""Guided start + gradient refinement: the benchmark's gradient method from a cheaper starting cell.

benchmark_placement.py's gradient rows start from the exhaustive 1 m sweep's best cell, so their real cost is
the sweep plus the 30 gradient solves. guided_placement.py finds as good a grid cell with 37 solves (2 m coarse
solves + the analytic LoS guide + 5 verifications). This runs the same gradient() from the guided placement and
charges both budgets, so the two starts compare at their total solve counts. Re-scored with the common
evaluator (200k samples), as every benchmark row.

    set PYTHONUTF8=1
    cd tx_planning && python guided_refine.py --bench ../output/tx_planning/benchmark_lobby.json \
        --guided ../output/tx_planning/guided_lobby.json --row guided_2m
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benchmark_placement import Evaluator, gradient  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bench", required=True); ap.add_argument("--guided", required=True)
    ap.add_argument("--row", default="guided_2m", help="guided_placement row whose placement starts the gradient")
    ap.add_argument("--k", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--steps", type=int, default=None, help="default: the benchmark's --steps")
    a = ap.parse_args()
    bench = json.load(open(a.bench)); guided = json.load(open(a.guided))
    cfg = SimpleNamespace(**bench["config"]); cfg.flush_every = 25
    steps = a.steps or cfg.steps
    ev = Evaluator(cfg)
    out = {"bench": a.bench, "guided": a.guided, "row": a.row, "steps": steps, "rows": {}}
    for K in a.k:
        g = guided["rows"][f"{a.row}_K{K}"]
        start = [np.array(p) for p in g["placement"]]
        s0, t0 = ev.solves, time.time()
        pl = gradient(ev, start, steps)
        used = ev.solves - s0; secs = time.time() - t0
        kp = ev.kpis(ev.gains_db(pl, cfg.eval_samples))
        ref = bench["methods"].get(f"gradient_K{K}", {})
        out["rows"][f"{a.row}+gradient_K{K}"] = {"placement": [list(map(float, p)) for p in pl], "start_solves": g["solves"],
                                                 "gradient_solves": int(used), "total_solves": int(g["solves"] + used),
                                                 "gradient_seconds": secs, **kp}
        print(f"K={K}: {a.row} start {g['coverage']:.3f} ({g['solves']} solves) -> gradient {kp['coverage']:.3f} "
              f"(+{used} solves = {g['solves'] + used}); benchmark gradient from the 1 m exhaustive cell: "
              f"{ref.get('coverage', float('nan')):.3f} ({ref.get('solves', 0)} + the sweep's 128-253)")
    path = a.guided.replace(".json", f"_refine_{a.row}.json")
    json.dump(out, open(path, "w"), indent=1)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
