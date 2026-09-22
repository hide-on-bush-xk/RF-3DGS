"""Depth bias of the placement, measured in the regime where the solver is
trustworthy: one receiver per solve, 1M samples (as multibounce_fraction.py).

depth_check.py / depth_orders.py showed that the planner's many-receiver
solve is sample-limited: with 59 receivers in the scene a depth-3 solve at
100k-400k samples LOSES first-order power at a tenth of the receivers (up
to 28 dB) instead of adding bounces, and even the depth-1 coverage moves by
7 points between 20k and 400k samples. So the depth question is answered
here per receiver: for each candidate transmitter (the refine run's start
and final, and the depth-2 run's final), each grid point is solved alone at
depth 1 and depth 3 with 1M samples; coverage above the threshold and the
candidate ranking are compared between depths.

    PYTHONUTF8=1 python tx_planning/depth_bias_perrx.py
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
REPO = os.path.abspath(os.path.join(HERE, ".."))


def main():
    """One solve per (candidate, receiver, depth). Slow but not sample-starved.

    Cost is len(cands) x len(rx) x 2 solves at 1M samples each -- the price of
    removing the many-receiver confound. The elapsed total is printed per
    candidate so a long run can be judged while it is still going.
    """
    import mitsuba as mi
    mi.set_variant("cuda_ad_mono_polarized")
    import drjit as dr
    from sionna.rt import Receiver, Transmitter
    from scene_common import LOBBY_X, LOBBY_Y, RX_HEIGHT, indoor_mask, load_radio_scene, make_solver, rx_grid
    r1 = json.load(open(os.path.join(REPO, "output/tx_planning/optimize_tx_refine.json")))
    r2 = json.load(open(os.path.join(REPO, "output/tx_planning/optimize_tx_refine_depth2.json")))
    c = r1["config"]
    # Three candidates spanning the question: where the search started, where a
    # depth-1 objective sent it, and where a depth-2 objective sent it.
    cands = [("refine start", r1["history"][0]["position"]), ("refine final (depth-1 optimum)", r1["history"][-1]["position"]),
             ("depth-2 final", r2["history"][-1]["position"])]
    scene = load_radio_scene(os.path.join(HERE, c["scene_xml"]), scattering=c["scattering"])
    solver = make_solver()
    # The grid is computed but receivers are NOT all added: they go in one at a time below.
    rx = rx_grid(LOBBY_X, LOBBY_Y, RX_HEIGHT, c["rx_step"]); rx = rx[indoor_mask(scene, rx)]
    noise = 10.0 ** (c["noise_floor_db"] / 10.0); thr = c["threshold_db"]
    tx = Transmitter(name="tx", position=[float(v) for v in cands[0][1]]); scene.add(tx)
    out = {"threshold_db": thr, "samples": 1_000_000, "n_rx": int(len(rx)), "candidates": []}
    t_all = time.time()
    for label, pos in cands:
        tx.position = [float(v) for v in pos]
        g = {1: np.zeros(len(rx)), 3: np.zeros(len(rx))}
        for i, p in enumerate(rx):
            # Swap the single receiver: remove the previous one, add this point.
            # Keeping exactly one receiver in the scene is the whole point -- it is
            # what gives each solve the full sample budget.
            if "rx" in scene.receivers:
                scene.remove("rx")
            scene.add(Receiver(name="rx", position=[float(v) for v in p]))
            for depth in (1, 3):
                # seed varies with the receiver but not with depth, so the depth
                # comparison at one point uses the same Monte-Carlo draw.
                paths = solver(scene=scene, max_depth=depth, samples_per_src=1_000_000, los=True, specular_reflection=True,
                               diffuse_reflection=True, refraction=False, synthetic_array=True, seed=42 + i)
                a_re, a_im = paths.a
                # One receiver and one antenna, so a flat sum is the total power.
                pw = float(np.asarray(dr.square(a_re) + dr.square(a_im)).sum())
                g[depth][i] = 10 * np.log10(pw + noise)
        d = g[3] - g[1]
        # min is reported alongside max: a negative minimum would mean depth 3 lost
        # power somewhere, which is the pathology this regime is meant to rule out.
        row = {"label": label, "tx": [round(float(v), 3) for v in pos],
               "coverage_depth1": float((g[1] > thr).mean()), "coverage_depth3": float((g[3] > thr).mean()),
               "mean_db_depth1": float(g[1].mean()), "mean_db_depth3": float(g[3].mean()),
               "gain_depth3_minus_depth1_db": {"median": float(np.median(d)), "p90": float(np.percentile(d, 90)), "max": float(d.max()), "min": float(d.min())},
               "newly_covered_at_depth3": int(((g[3] > thr) & ~(g[1] > thr)).sum()), "lost_at_depth3": int((~(g[3] > thr) & (g[1] > thr)).sum()),
               # Per-receiver gains are kept so the spatial pattern can be mapped later.
               "gain_depth1": g[1].round(2).tolist(), "gain_depth3": g[3].round(2).tolist()}
        out["candidates"].append(row)
        print(f"{label} {row['tx']}: depth 1 coverage {row['coverage_depth1']:.3f} (mean {row['mean_db_depth1']:.1f} dB) | depth 3 coverage {row['coverage_depth3']:.3f} "
              f"(mean {row['mean_db_depth3']:.1f} dB); depth-3 gain over depth 1 per receiver: median {row['gain_depth3_minus_depth1_db']['median']:+.2f} dB, "
              f"P90 {row['gain_depth3_minus_depth1_db']['p90']:+.2f}, max {row['gain_depth3_minus_depth1_db']['max']:+.2f}, min {row['gain_depth3_minus_depth1_db']['min']:+.2f}; "
              f"+{row['newly_covered_at_depth3']} / -{row['lost_at_depth3']} points  [{time.time() - t_all:.0f} s]")
    out["rx"] = rx.round(3).tolist()
    json.dump(out, open(os.path.join(REPO, "output/tx_planning/depth_bias_perrx.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
