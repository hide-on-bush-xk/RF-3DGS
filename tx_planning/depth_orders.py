"""Does a deeper solve keep the first-order power?

depth_check.py found LOWER coverage and 4 dB LESS mean gain at max_depth 2
and 3 than at depth 1 for the same transmitter, which no physical model
allows: extra bounces can only add power. So either the solver spreads its
sample budget across depths (fewer samples per order, noisier per-receiver
estimates, Jensen bias in the dB mean) or the per-order weighting changes
with max_depth. This solves once at depth 3 and splits the received power
by number of interactions, and compares the order-<=1 part with a separate
depth-1 solve, per receiver, for the refine run's final transmitter.

    PYTHONUTF8=1 python tx_planning/depth_orders.py
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
REPO = os.path.abspath(os.path.join(HERE, ".."))


def main():
    import mitsuba as mi
    mi.set_variant("cuda_ad_mono_polarized")
    import drjit as dr
    from sionna.rt import Receiver, Transmitter
    from scene_common import LOBBY_X, LOBBY_Y, RX_HEIGHT, indoor_mask, load_radio_scene, make_solver, rx_grid
    run = json.load(open(os.path.join(REPO, "output/tx_planning/optimize_tx_refine.json")))
    c = run["config"]; tx_pos = run["history"][-1]["position"]
    scene = load_radio_scene(os.path.join(HERE, c["scene_xml"]), scattering=c["scattering"])
    solver = make_solver()
    rx = rx_grid(LOBBY_X, LOBBY_Y, RX_HEIGHT, c["rx_step"]); rx = rx[indoor_mask(scene, rx)]
    for i, p in enumerate(rx):
        scene.add(Receiver(name=f"rx{i}", position=[float(v) for v in p]))
    scene.add(Transmitter(name="tx", position=[float(v) for v in tx_pos]))
    noise = 10.0 ** (c["noise_floor_db"] / 10.0); thr = c["threshold_db"]
    out = {"tx": tx_pos, "threshold_db": thr, "n_rx": int(len(rx))}
    for samples in (100_000, 400_000):
        res = {}
        for depth in (1, 3):
            paths = solver(scene=scene, max_depth=depth, max_num_paths_per_src=10_000_000, samples_per_src=samples, los=True, specular_reflection=True,
                           diffuse_reflection=True, refraction=False, synthetic_array=True, seed=42)
            a_re, a_im = paths.a
            p = np.asarray(dr.square(a_re) + dr.square(a_im))               # [rx, rx_ant, tx, tx_ant, paths]
            p = p.reshape(p.shape[0], -1, p.shape[-1]).sum(1)              # [rx, paths]
            inter = np.asarray(paths.interactions)                          # [depth, rx, ..., paths]
            n_int = (inter.reshape(inter.shape[0], inter.shape[1], -1, inter.shape[-1]) != 0).sum(0)[:, 0, :]  # [rx, paths]
            by_order = {k: (p * (n_int == k)).sum(1) for k in range(depth + 1)}
            tot = p.sum(1)
            res[depth] = {"total": tot, "order_le1": by_order[0] + by_order[1], "orders": by_order, "n_paths": int(p.shape[1])}
        g1 = 10 * np.log10(res[1]["total"] + noise); g3 = 10 * np.log10(res[3]["total"] + noise); g3le1 = 10 * np.log10(res[3]["order_le1"] + noise)
        d = g3le1 - g1
        row = {"samples": samples, "paths_depth1": res[1]["n_paths"], "paths_depth3": res[3]["n_paths"],
               "coverage_depth1": float((g1 > thr).mean()), "coverage_depth3_total": float((g3 > thr).mean()), "coverage_depth3_order_le1": float((g3le1 > thr).mean()),
               "mean_db_depth1": float(g1.mean()), "mean_db_depth3_total": float(g3.mean()), "mean_db_depth3_order_le1": float(g3le1.mean()),
               "order_le1_minus_depth1_db": {"mean": float(d.mean()), "median": float(np.median(d)), "p10": float(np.percentile(d, 10)), "p90": float(np.percentile(d, 90))},
               "share_beyond_order1_at_depth3": float(1 - res[3]["order_le1"].sum() / res[3]["total"].sum())}
        out[str(samples)] = row
        print(f"{samples:,} samples: depth-1 solve {res[1]['n_paths']:,} paths, coverage {row['coverage_depth1']:.3f}, mean {row['mean_db_depth1']:.1f} dB | "
              f"depth-3 solve {res[3]['n_paths']:,} paths: total coverage {row['coverage_depth3_total']:.3f}, mean {row['mean_db_depth3_total']:.1f} dB; "
              f"its order<=1 part: coverage {row['coverage_depth3_order_le1']:.3f}, mean {row['mean_db_depth3_order_le1']:.1f} dB; "
              f"order<=1(depth 3) - depth-1 per receiver: median {row['order_le1_minus_depth1_db']['median']:+.1f} dB (P10 {row['order_le1_minus_depth1_db']['p10']:+.1f}, P90 {row['order_le1_minus_depth1_db']['p90']:+.1f}); "
              f"power beyond order 1 at depth 3: {row['share_beyond_order1_at_depth3']:.1%}")
    json.dump(out, open(os.path.join(REPO, "output/tx_planning/depth_orders.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
