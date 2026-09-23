"""Does one Sionna solve with B receivers cost about one solve, as the planner found at 20k samples?

The dataset generator solves every receiver position on its own (one solve per MULTI position, per face for
MVDR). tx_planning measured that the solver's cost is per source, not per receiver, at 20k samples and 16
receivers. This measures it at the generator's setting (MULTI: 2.4 GHz, tutorial materials, depth 1, 1M
samples, synthetic 10x10 array) for B = 1, 2, 4, 8, 16 positions spread by farthest-point sampling, and checks
that the paths of receiver r in a batched solve reproduce a single-receiver solve's power for the specular /
LoS part (deterministic) -- the diffuse part is sampled and differs with the lattice.

    set PYTHONUTF8=1
    python sionna_port/probe_rx_batching.py
"""

from __future__ import annotations

import dataclasses
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
from generate_dataset import Config, VIEW_YAWS, build_scene, read_pose_groups  # noqa: E402

SCENE = os.path.join(REPO, "sionna_tutorial/RF-3DGS_Sionna_simulation_tutorial/NIST_lobby_v1.0/NIST_lobby_V1.1_sionna12.xml")
DS = os.path.join(REPO, "RF-3DGS_dataset/regenerated/3dgs_MULTI_24ghz_tut_cs")


def main():
    meta = json.load(open(os.path.join(DS, "generation_meta.json")))
    fields = {f.name for f in dataclasses.fields(Config)}
    cfg = dataclasses.replace(Config(**{k: v for k, v in meta.items() if k in fields}), scene_xml=SCENE)
    from sionna.rt import PathSolver, Receiver, Transmitter
    import drjit as dr
    scene = build_scene(cfg)
    scene.add(Transmitter(name="tx", position=list(cfg.tx_loc)))
    solver = PathSolver()
    pos = np.array([g[0] for g in read_pose_groups(os.path.join(DS, "sparse", "0", "images.txt"))])
    chosen = [0]; d = np.linalg.norm(pos - pos[0], axis=1)
    while len(chosen) < 16:
        j = int(d.argmax()); chosen.append(j); d = np.minimum(d, np.linalg.norm(pos - pos[j], axis=1))

    def solve(rxs, cap=None):
        for name in list(scene.receivers):
            scene.remove(name)
        for i, p in enumerate(rxs):
            scene.add(Receiver(name=f"rx{i}", position=[float(v) for v in p], orientation=[VIEW_YAWS[0], 0.0, 0.0]))
        kw = {} if cap is None else {"max_num_paths_per_src": cap}
        torch.cuda.synchronize(); t0 = time.perf_counter()
        paths = solver(scene=scene, max_depth=1, samples_per_src=cfg.samples_per_src, los=True, specular_reflection=True,
                       diffuse_reflection=True, refraction=False, diffraction=False, synthetic_array=True, seed=42, **kw)
        a, tau = paths.cir(normalize_delays=False, out_type="torch")
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) * 1000, a, tau

    for _ in range(2):
        solve(pos[chosen[:1]])
    rows = []
    for B in (1, 2, 4, 8, 16):
        for cap in (None, 10_000_000):
            ms = []
            for rep in range(3):
                t, a, tau = solve(pos[chosen[:B]], cap)
                ms.append(t)
            # a: [num_rx, rx_ant, num_tx, tx_ant, paths, time]; power per receiver summed over paths
            p = (a[:, :, 0, 0, :, 0].abs() ** 2).sum(dim=(1, 2)).cpu().numpy()
            n_valid = int(torch.isfinite(tau).sum()) if tau.numel() else 0
            rows.append({"B": B, "cap": cap, "ms_median": float(np.median(ms)), "ms_per_rx": float(np.median(ms)) / B,
                         "power_db_per_rx": (10 * np.log10(p + 1e-30)).round(2).tolist(), "paths_axis": int(a.shape[-2])})
            print(f"B={B:2d} cap={cap}: {np.median(ms):7.1f} ms ({np.median(ms) / B:6.1f} ms per receiver), paths axis {a.shape[-2]}, "
                  f"power per rx (dB) {np.round(10 * np.log10(p + 1e-30), 1)[:4]}...")
    # single-receiver reference powers for the first 4 positions (same seed)
    single = [float(10 * np.log10((solve(pos[[c]])[1][0, :, 0, 0, :, 0].abs() ** 2).sum().cpu().numpy() + 1e-30)) for c in chosen[:4]]
    print("single-receiver total power (dB), first 4 positions:", np.round(single, 2))
    json.dump({"rows": rows, "single_power_db_first4": single, "samples": cfg.samples_per_src},
              open(os.path.join(REPO, "output", "rrf", "probe_rx_batching.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
