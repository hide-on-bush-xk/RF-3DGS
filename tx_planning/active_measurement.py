"""Stage 3b: choose where to measure next so the twin learns what it lacks.

fit_materials.py showed that one transmitter constrains only the materials
its depth-1 paths touch; the rest stay at the prior. Which second transmitter
position would teach the twin the most? Sionna's gradient answers that
before anything is measured: the sensitivity ||d PDP / d s_n||^2 at a
candidate position says how strongly each material would show in the data
there. The candidate scoring highest on the materials transmitter A left
unconstrained is where to measure next.

The experiment: fit from A alone, from A + the chosen candidate, and from
A + a median-scoring candidate (an arbitrary second position), and score all
three at a held-out transmitter B and on the material estimates themselves.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from fit_materials import MaterialTwin, add_twin_args
from scene_common import LOBBY_X, LOBBY_Y, indoor_mask, rx_grid


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_twin_args(ap)
    ap.add_argument("--out", default="../output/tx_planning/active_measurement.json")
    ap.add_argument("--tx-a", type=float, nargs=3, default=[8.0, -7.0, 2.0])
    ap.add_argument("--tx-b", type=float, nargs=3, action="append", default=None,
                    help="held-out transmitter(s); repeatable. Default: one in "
                         "the main room and one at each end of the corridor")
    ap.add_argument("--cand-step", type=float, default=4.0,
                    help="grid step of candidate second-transmitter positions")
    ap.add_argument("--cand-height", type=float, default=2.0)
    ap.add_argument("--probes", type=int, default=4,
                    help="random projections per candidate for the sensitivity")
    cfg = ap.parse_args()
    if cfg.tx_b is None:
        cfg.tx_b = [[0.0, -3.0, 2.0], [8.0, -15.0, 2.0], [8.0, 1.0, 2.0]]

    twin = MaterialTwin(cfg)
    names = twin.names
    t0 = time.time()
    print(f"{len(twin.rx_positions)} indoor receivers, {len(names)} used materials")

    # Candidates: indoor grid at AP height, excluding A and B themselves.
    cands = rx_grid(LOBBY_X, LOBBY_Y, cfg.cand_height, cfg.cand_step)
    cands = cands[indoor_mask(twin.scene, cands)]
    keep = [np.linalg.norm(c - cfg.tx_a) > 1.0
            and all(np.linalg.norm(c - b) > 1.0 for b in cfg.tx_b) for c in cands]
    cands = cands[keep]
    print(f"{len(cands)} candidate positions for the second transmitter")

    # --- what A alone teaches ------------------------------------------------
    obs_a = twin.measure(cfg.tx_a, twin.truth)
    true_b = [twin.measure(b, twin.truth, noisy=False) for b in cfg.tx_b]
    init = {n: cfg.init for n in names}

    def rmse_b(values):
        """Mean over the held-out transmitters."""
        return float(np.mean([twin.rmse(values, b, ref)
                              for b, ref in zip(cfg.tx_b, true_b)]))
    print("\nfit from A alone")
    est_a, hist_a = twin.fit([(cfg.tx_a, obs_a)], verbose=False)
    ident_a, _ = twin.identifiable(hist_a)
    sens_a = twin.sensitivity(init, cfg.tx_a, cfg.probes)
    floor = 0.02 ** 2 * max(sens_a.values())     # the identifiability threshold, squared

    # --- score every candidate by what it adds -------------------------------
    # log(1 + new/old) per material: large where A saw nothing, saturating
    # where A already constrains the material well. Evaluated at the prior,
    # as a planner would before measuring.
    scores = []
    for i, c in enumerate(cands):
        s = twin.sensitivity(init, c, cfg.probes, seed=i + 1)
        gain = sum(np.log1p(s[n] / (sens_a[n] + floor)) for n in names)
        new = [n for n in names if s[n] >= floor and n not in ident_a]
        scores.append({"pos": c.tolist(), "score": float(gain), "newly_seen": new})
        print(f"  cand {i:2d} ({c[0]:5.1f},{c[1]:6.1f})  score {gain:6.2f}  "
              f"newly constrained {len(new):2d}  {time.time()-t0:5.0f} s")
    order = np.argsort([-s["score"] for s in scores])
    best, median = scores[order[0]], scores[order[len(order) // 2]]
    print(f"\nbest candidate {np.round(best['pos'], 1).tolist()} score {best['score']:.2f}; "
          f"median candidate {np.round(median['pos'], 1).tolist()} score {median['score']:.2f}")

    # --- fit with the second transmitter, chosen vs arbitrary ---------------
    arms = {"A only": (est_a, hist_a)}
    for label, c in (("A + chosen C", best["pos"]), ("A + median C", median["pos"])):
        obs_c = twin.measure(c, twin.truth)
        print(f"\nfit from {label}")
        arms[label] = twin.fit([(cfg.tx_a, obs_a), (c, obs_c)], verbose=False)

    print(f"\n{'arm':14s} {'RMSE @ held-out Bs':>19s} {'|err| identifiable':>20s} "
          f"{'|err| all 29':>13s}  identifiable")
    rows = {}
    for label, (est, hist) in arms.items():
        ident, _ = twin.identifiable(hist)
        e_b = rmse_b(est)
        rows[label] = {"rmse_b": e_b, "err_ident": twin.material_error(est, ident),
                       "err_all": twin.material_error(est), "n_ident": len(ident),
                       "est": est}
        print(f"{label:14s} {e_b:15.2f} dB {rows[label]['err_ident']:20.3f} "
              f"{rows[label]['err_all']:13.3f}  {len(ident)}/{len(names)}")
    e_base = rmse_b(init)
    print(f"{'prior':14s} {e_base:15.2f} dB {twin.material_error(init):20.3f} "
          f"{twin.material_error(init):13.3f}  0/{len(names)}")

    os.makedirs(os.path.dirname(cfg.out) or ".", exist_ok=True)
    with open(cfg.out, "w", encoding="utf-8") as fid:
        json.dump({"config": vars(cfg), "truth": twin.truth, "candidates": scores,
                   "sensitivity_a": sens_a, "arms": rows, "prior_rmse_b": e_base,
                   "seconds": time.time() - t0}, fid, indent=1)
    print(f"\n{time.time()-t0:.0f} s")


if __name__ == "__main__":
    main()
