"""Does the depth-1 objective bias the transmitter placement?

The planner's coverage objective solves at max_depth 1, and the multi-bounce
audit found the depth-1 twin misses 23-82 % of the received power at the
corridor ends -- the points a placement is supposed to help. This scores
transmitter positions (the refine run's start and final, and any others
given, e.g. the final of a depth-2 optimisation) with the same receiver
grid and threshold at depths 1, 2 and 3, with a larger sample budget than
the optimiser uses, and reports hard coverage, mean gain, and which grid
points flip between depths.

    PYTHONUTF8=1 python tx_planning/depth_check.py --runs ../output/tx_planning/optimize_tx_refine.json \
        ../output/tx_planning/optimize_tx_refine_depth2.json --samples 100000
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
REPO = os.path.abspath(os.path.join(HERE, ".."))


def main():
    """Score every candidate transmitter at every requested depth on one grid.

    Read the "flips" counts rather than only the coverage fraction: a placement
    whose coverage is unchanged but that gains and loses different grid points
    between depths is being evaluated on a different set of receivers, which is
    the bias the script is looking for.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="optimize_tx result files; their start and final Tx are scored")
    ap.add_argument("--tx", nargs="*", default=[], metavar="x,y,z", help="extra transmitter positions")
    ap.add_argument("--depths", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--samples", type=int, default=100_000)
    ap.add_argument("--out", default=os.path.join(REPO, "output/tx_planning/depth_check.json"))
    cfg = ap.parse_args()
    import mitsuba as mi
    mi.set_variant("cuda_ad_mono_polarized")
    import drjit as dr
    from sionna.rt import Receiver, Transmitter
    from scene_common import LOBBY_X, LOBBY_Y, RX_HEIGHT, indoor_mask, load_radio_scene, make_solver, rx_grid

    runs = [json.load(open(p)) for p in cfg.runs]
    # The FIRST run's config defines the grid, threshold and scene for every
    # candidate, so all of them are scored on identical terms even when they came
    # from optimisations that used different settings.
    c0 = runs[0]["config"]
    cands = []
    for p, r in zip(cfg.runs, runs):
        # Both ends of each optimisation: the starting guess and what it converged
        # to. The label carries that run's own reported coverage for comparison.
        h = r["history"]; tag = os.path.splitext(os.path.basename(p))[0]
        cands.append((f"{tag} start (depth {r['config']['max_depth']})", h[0]["position"]))
        cands.append((f"{tag} final (depth {r['config']['max_depth']}, its own coverage {h[-1]['coverage_frac']:.3f})", h[-1]["position"]))
    for s in cfg.tx:
        cands.append((f"tx {s}", [float(v) for v in s.split(",")]))

    scene = load_radio_scene(c0["scene_xml"] if os.path.isabs(c0["scene_xml"]) else os.path.join(HERE, c0["scene_xml"]),
                             scattering=c0["scattering"])
    solver = make_solver()
    rx = rx_grid(LOBBY_X, LOBBY_Y, RX_HEIGHT, c0["rx_step"]); rx = rx[indoor_mask(scene, rx)]
    for i, p in enumerate(rx):
        scene.add(Receiver(name=f"rx{i}", position=[float(v) for v in p]))
    # One Transmitter object, repositioned per candidate: cheaper than rebuilding
    # the scene, and it keeps the receiver set byte-identical across candidates.
    tx = Transmitter(name="tx", position=[float(v) for v in cands[0][1]]); scene.add(tx)
    noise = 10.0 ** (c0["noise_floor_db"] / 10.0); thr = c0["threshold_db"]
    print(f"{len(rx)} indoor receivers on a {c0['rx_step']} m grid, threshold {thr} dB, {cfg.samples:,} samples")
    out = {"config": c0, "samples": cfg.samples, "depths": cfg.depths, "rx": rx.round(3).tolist(), "candidates": []}
    for label, pos in cands:
        tx.position = [float(v) for v in pos]
        row = {"label": label, "tx": [round(float(v), 3) for v in pos], "by_depth": {}}
        gains = {}
        for d in cfg.depths:
            t0 = time.time()
            paths = solver(scene=scene, max_depth=d, max_num_paths_per_src=10_000_000, samples_per_src=cfg.samples, los=True, specular_reflection=True,
                           diffuse_reflection=True, refraction=False, synthetic_array=True, seed=42)
            a_re, a_im = paths.a
            p_rx = dr.square(a_re) + dr.square(a_im)
            # Collapse every axis except the receiver axis (0), from the back
            # forward so the axis indices stay valid as the array shrinks.
            for axis in range(p_rx.ndim - 1, 0, -1):
                p_rx = dr.sum(p_rx, axis=axis)
            g = 10.0 * np.log10(np.asarray(p_rx).reshape(-1) + noise)
            gains[d] = g
            row["by_depth"][str(d)] = {"coverage": float((g > thr).mean()), "mean_db": float(g.mean()),
                                       "median_db": float(np.median(g)), "seconds": time.time() - t0}
        d0 = cfg.depths[0]      # everything is expressed relative to the first depth
        for d in cfg.depths[1:]:
            # Two separate counts, not a net figure: a placement can gain and lose
            # the same number of points and look unchanged in coverage.
            flips_up = int(((gains[d] > thr) & ~(gains[d0] > thr)).sum()); flips_down = int((~(gains[d] > thr) & (gains[d0] > thr)).sum())
            row["by_depth"][str(d)]["gain_vs_depth1_db"] = {"mean": float((gains[d] - gains[d0]).mean()), "p90": float(np.percentile(gains[d] - gains[d0], 90)),
                                                             "max": float((gains[d] - gains[d0]).max())}
            row["by_depth"][str(d)]["flips_vs_depth1"] = {"newly_covered": flips_up, "lost": flips_down}
        out["candidates"].append(row)
        print(f"{label}: " + "; ".join(f"depth {d}: coverage {row['by_depth'][str(d)]['coverage']:.3f}, mean {row['by_depth'][str(d)]['mean_db']:.1f} dB"
                                       + (f", +{row['by_depth'][str(d)]['flips_vs_depth1']['newly_covered']}/-{row['by_depth'][str(d)]['flips_vs_depth1']['lost']} pts vs depth 1"
                                          if d != d0 else "") for d in cfg.depths))
    json.dump(out, open(cfg.out, "w"), indent=1)
    print(f"wrote {cfg.out}")


if __name__ == "__main__":
    main()
