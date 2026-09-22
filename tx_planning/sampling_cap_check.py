"""Why does the many-receiver solve get WORSE with more samples?

depth_orders.py: with 59 receivers in one scene, coverage of one transmitter
falls from 57.6 % (100k samples) to 44.1 % (400k), and a depth-3 solve loses
first-order power at a tenth of the receivers. Monotone degradation is not
variance. Candidate cause: the solver caps the candidate paths per source
(max_num_paths_per_src, default 1e6); more samples x more receivers x
diffuse reflection overflow the cap and paths are dropped. Test: the same
solve with the cap raised.

    PYTHONUTF8=1 python tx_planning/sampling_cap_check.py
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)          # so `from scene_common import ...` works when run directly
REPO = os.path.abspath(os.path.join(HERE, ".."))


def main():
    """Sweep (cap x samples x depth) on one fixed transmitter and report coverage.

    The prediction being tested: at cap 1e6 coverage falls as samples rise, and
    at cap 1e7 it does not. Anything else means the cap is not the mechanism.
    """
    import mitsuba as mi
    # Polarized mono variant: what Sionna's RT needs, and it must be set before
    # any Sionna import touches Dr.Jit.
    mi.set_variant("cuda_ad_mono_polarized")
    import drjit as dr
    from sionna.rt import Receiver, Transmitter
    from scene_common import LOBBY_X, LOBBY_Y, RX_HEIGHT, indoor_mask, load_radio_scene, make_solver, rx_grid
    # Config and transmitter are taken from a previous optimisation run, so this
    # diagnostic reproduces exactly the setting where the anomaly appeared.
    run = json.load(open(os.path.join(REPO, "output/tx_planning/optimize_tx_refine.json")))
    c = run["config"]; tx_pos = run["history"][-1]["position"]
    scene = load_radio_scene(os.path.join(HERE, c["scene_xml"]), scattering=c["scattering"])
    solver = make_solver()
    rx = rx_grid(LOBBY_X, LOBBY_Y, RX_HEIGHT, c["rx_step"]); rx = rx[indoor_mask(scene, rx)]
    for i, p in enumerate(rx):
        scene.add(Receiver(name=f"rx{i}", position=[float(v) for v in p]))
    scene.add(Transmitter(name="tx", position=[float(v) for v in tx_pos]))
    # Linear-power noise floor, added before the dB conversion so an unreached
    # receiver gives the floor rather than -inf.
    noise = 10.0 ** (c["noise_floor_db"] / 10.0); thr = c["threshold_db"]
    out = []
    print(f"{len(rx)} receivers, Tx {np.round(tx_pos, 2).tolist()}, threshold {thr} dB")
    for cap in (1_000_000, 10_000_000):          # default cap, then 10x
        for samples in (20_000, 100_000, 400_000):
            for depth in (1, 3):
                t0 = time.time()
                try:
                    # seed fixed across the sweep, so differences are the settings
                    # and not the Monte-Carlo draw.
                    paths = solver(scene=scene, max_depth=depth, samples_per_src=samples, max_num_paths_per_src=cap, los=True,
                                   specular_reflection=True, diffuse_reflection=True, refraction=False, synthetic_array=True, seed=42)
                except Exception as exc:
                    # The large-cap combinations can run out of GPU memory; that is
                    # itself a result, so the sweep continues rather than aborting.
                    print(f"cap {cap:.0e} samples {samples:>7,} depth {depth}: {type(exc).__name__}: {str(exc)[:120]}"); continue
                # paths.a is (real, imag); |a|^2 is the per-path power.
                a_re, a_im = paths.a
                p = np.asarray(dr.square(a_re) + dr.square(a_im)); n_paths = p.shape[-1]
                # Collapse antennas, transmitters and paths into one total per receiver.
                p = p.reshape(p.shape[0], -1).sum(1)
                g = 10 * np.log10(p + noise)
                row = {"cap": cap, "samples": samples, "depth": depth, "paths": int(n_paths), "coverage": float((g > thr).mean()),
                       "mean_db": float(g.mean()), "seconds": time.time() - t0}
                out.append(row)
                print(f"cap {cap:.0e} samples {samples:>7,} depth {depth}: {n_paths:>8,} paths, coverage {row['coverage']:.3f}, mean {row['mean_db']:.1f} dB, {row['seconds']:.1f} s")
    json.dump(out, open(os.path.join(REPO, "output/tx_planning/sampling_cap_check.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
