"""Is a single-receiver 60 GHz solve truncated by the solver's default max_num_paths_per_src (1e6)?

generate_dataset.py solves one receiver at a time with the default cap. tx_planning found that many receivers
overflow it (sampling_cap_check.py); this asks whether ONE receiver already does at the dataset's setting
(60 GHz, uniform scattering 0.7, depth 1, 1M samples, 10x10 synthetic array): same pose, same seed, the
default cap against 1e7, and a four-receiver solve (one per face) against single-receiver solves at 1e7.

    set PYTHONUTF8=1
    python sionna_port/probe_path_cap.py
"""

from __future__ import annotations

import dataclasses
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
from generate_dataset import Config, VIEW_YAWS, build_scene, read_pose_groups  # noqa: E402

SCENE = os.path.join(REPO, "sionna_tutorial/RF-3DGS_Sionna_simulation_tutorial/NIST_lobby_v1.0/NIST_lobby_V1.1_sionna12.xml")


def main():
    from sionna.rt import PathSolver, Receiver, Transmitter
    for freq, materials, label in ((60e9, "uniform", "60 GHz uniform 0.7 (MVDR_100, live)"), (2.4e9, "tutorial", "2.4 GHz tutorial (MULTI)")):
        cfg = dataclasses.replace(Config(scene_xml=SCENE, rx_loc_file="", out_dir=""), frequency=freq, materials=materials)
        scene = build_scene(cfg)
        scene.add(Transmitter(name="tx", position=list(cfg.tx_loc)))
        solver = PathSolver()
        groups = read_pose_groups(os.path.join(REPO, "output/r26_smoke/images.txt"))
        print(f"\n{label}")
        rows = []
        for gi, (rx, _) in enumerate(groups[:6]):
            def solve(orients, cap):
                for n in list(scene.receivers):
                    scene.remove(n)
                for j, y in enumerate(orients):
                    scene.add(Receiver(name=f"rx{j}", position=list(rx), orientation=[y, 0.0, 0.0]))
                kw = {} if cap is None else {"max_num_paths_per_src": cap}
                p = solver(scene=scene, max_depth=1, samples_per_src=1_000_000, los=True, specular_reflection=True,
                           diffuse_reflection=True, refraction=False, diffraction=False, synthetic_array=True, seed=42 + gi, **kw)
                a, tau = p.cir(normalize_delays=False, out_type="numpy")
                a = a[:, :, 0, 0, :, 0]                                   # [rx, ant, paths]
                tau = tau.reshape(a.shape[0], -1)
                valid = np.isfinite(tau) & (tau >= 0) & (np.abs(a).sum(1) > 0)
                pw = 10 * np.log10((np.abs(a) ** 2).sum(axis=(1, 2)) + 1e-300)
                return valid.sum(1), pw
            n_def, p_def = solve([VIEW_YAWS[0]], None)
            n_big, p_big = solve([VIEW_YAWS[0]], 10_000_000)
            n_b4, p_b4 = solve(list(VIEW_YAWS), 10_000_000)
            singles = [solve([y], 10_000_000) for y in VIEW_YAWS]
            r = {"position": gi, "paths_default_cap": int(n_def[0]), "paths_cap_1e7": int(n_big[0]),
                 "power_db_default_cap": float(p_def[0]), "power_db_cap_1e7": float(p_big[0]),
                 "batch4_paths": n_b4.tolist(), "single_paths": [int(s[0][0]) for s in singles],
                 "batch4_power_db": p_b4.round(3).tolist(), "single_power_db": [round(float(s[1][0]), 3) for s in singles]}
            rows.append(r)
            print(f"  pos {gi}: one receiver, default cap {n_def[0]:>8d} paths {p_def[0]:7.2f} dB | cap 1e7 {n_big[0]:>8d} paths "
                  f"{p_big[0]:7.2f} dB || 4 receivers in one solve {n_b4.tolist()} vs single {r['single_paths']}; "
                  f"power {np.round(p_b4, 2).tolist()} vs {[round(v, 2) for v in r['single_power_db']]}")
        json.dump(rows, open(os.path.join(REPO, "output", "rrf", f"probe_path_cap_{int(freq / 1e8)}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
