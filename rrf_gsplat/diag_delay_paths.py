"""Who carries the delay tail: pixels with one path or pixels with several?

After the delay decomposition the delay channel's median is 0.95 ns but
its RMSE is still 6.9 ns. The candidate mechanism is the one that carries
the azimuth tail: a pixel whose kernel covers several paths of different
delay gets their power-weighted mean, which no single Gaussian composite
reproduces. This re-solves the paths of a sample of held-out positions,
counts the paths whose arrival direction falls in each equirect cell,
carries the count to the four pinhole faces (nearest cell), and reports the
delay error of a model's renders by number-of-paths class on the hit pixels,
together with each class's share of the pixels and of the squared error.

The share-of-squared-error column is the one that answers the question: a
class can hold most of the pixels and little of the error, or the reverse.

    PYTHONUTF8=1 python rrf_gsplat/diag_delay_paths.py --runs m_multi_24_tut_cs m_multi_24_tut_cs_depth --positions 24
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO, "sionna_port"))
sys.path.insert(0, os.path.join(REPO, "rrf_gsplat"))
# (label, min paths, max paths). The open-ended top class uses a large finite
# bound rather than inf so the comparison stays integer.
CLASSES = [("1", 1, 1), ("2", 2, 2), ("3-5", 3, 5), (">=6", 6, 10 ** 9)]


def main():
    """Re-solve sampled positions, bin pixels by path count, and score each bin."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--truth", default=os.path.join(REPO, "RF-3DGS_dataset/regenerated/3dgs_MULTI_24ghz_tut_cs"))
    ap.add_argument("--runs", nargs="+", default=["m_multi_24_tut_cs", "m_multi_24_tut_cs_depth"])
    ap.add_argument("--positions", type=int, default=24)
    cfg = ap.parse_args()
    import mitsuba as mi
    mi.set_variant("cuda_ad_mono_polarized")
    from sionna.rt import PathSolver, Receiver, Transmitter
    from generate_dataset import Config, VIEW_YAWS, build_scene, solve_paths
    from rf_spectra import _path_arrays, compute_angle_matrices
    from eval_baselines import read_poses
    meta = json.load(open(os.path.join(cfg.truth, "generation_meta.json")))
    ch = {n: i for i, n in enumerate(meta["channels"])}; lo0, hi0 = meta["channel_ranges"][0]
    # Frequency, materials and transmitter come from the dataset's own meta, so
    # the re-solve reproduces the paths that dataset was built from.
    gcfg = Config(scene_xml=os.path.join(REPO, "sionna_tutorial/RF-3DGS_Sionna_simulation_tutorial/NIST_lobby_v1.0/NIST_lobby_V1.1_sionna12.xml"),
                  rx_loc_file="", out_dir="", spectrum="MULTI", frequency=meta["frequency"], materials=meta["materials"], dashboard=False,
                  tx_loc=tuple(meta["tx_loc"]))
    scene = build_scene(gcfg); scene.add(Transmitter(name="tx", position=list(gcfg.tx_loc))); solver = PathSolver()
    poses = read_poses(os.path.join(cfg.truth, "sparse", "0", "images.txt"))
    test = [l.strip() for l in open(os.path.join(cfg.truth, "test_index.txt")) if l.strip()]
    # held-out positions (4 consecutive names each), evenly spaced
    # Evenly spaced positions with all four faces kept: the smoke rule again,
    # never a plain stride over views.
    pos_names = [test[i:i + 4] for i in range(0, len(test), 4)]
    pick = [pos_names[i] for i in np.linspace(0, len(pos_names) - 1, cfg.positions).round().astype(int)]
    dev = torch.device("cuda")
    scale, H, W, FOV = 3, gcfg.height, gcfg.width, gcfg.fov_deg
    # face pixel -> equirect cell, per yaw (the resampler's own mapping, nearest cell)
    # Precomputed once per yaw: the mapping is a property of the camera, not of
    # the position, so it is reused across every sampled receiver.
    theta, phi = compute_angle_matrices(W, H, FOV, device=dev)
    cell = {}
    for yaw in VIEW_YAWS:
        ph = phi + yaw
        y = (torch.rad2deg(theta) * scale).round().long().clamp(0, 180 * scale - 1)
        # Modulo, not clamp, on azimuth: it wraps.
        x = (((-torch.rad2deg(ph) + 180.0) * scale).round().long()) % (360 * scale)
        cell[yaw] = (y * (360 * scale) + x).reshape(-1)
    acc = {r: {c[0]: {"e2": [], "e": [], "n": 0} for c in CLASSES} for r in cfg.runs}
    for names in pick:
        rx, _ = poses[names[0]]
        if "rx" in scene.receivers:
            scene.remove("rx")
        scene.add(Receiver(name="rx", position=[float(v) for v in rx]))
        # view_index derived from the name, so the sampling lattice matches the
        # one the dataset was generated with at this position.
        paths = solve_paths(solver, scene, gcfg, view_index=int(names[0]) // 4)
        amp, tau, th_r, ph_r, _, _ = _path_arrays(paths, dev)
        ti = (torch.rad2deg(th_r) * scale).round().long().clamp(0, 180 * scale - 1)
        pi = ((-torch.rad2deg(ph_r) + 180.0) * scale).round().long().clamp(0, 360 * scale - 1)
        # Paths per equirect cell, scattered in one pass.
        count = torch.zeros(180 * scale * 360 * scale, device=dev).index_add_(0, ti * (360 * scale) + pi, torch.ones_like(amp))
        for name in names:
            _, yidx = poses[name]; yaw = VIEW_YAWS[yidx]
            # Carry the per-cell count onto this face's pixel grid.
            n_face = count[cell[yaw]].reshape(H, W).cpu().numpy()
            truth = np.load(os.path.join(cfg.truth, "spectra_float", name + ".npy")).astype(np.float64)
            mask = (truth[0] - lo0) / (hi0 - lo0) > 0.02
            t_dl = truth[ch["delay_ns"]]
            for r in cfg.runs:
                pred = np.load(os.path.join(REPO, "output/rrf", r, "renders", name + ".npy")).astype(np.float64)[ch["delay_ns"]]
                e = np.abs(pred - t_dl)
                for label, lo, hi in CLASSES:
                    m = mask & (n_face >= lo) & (n_face <= hi)
                    if m.any():
                        acc[r][label]["e"].append(e[m]); acc[r][label]["n"] += int(m.sum())
    out = {"positions": len(pick), "runs": {}}
    for r in cfg.runs:
        tot_n = sum(v["n"] for v in acc[r].values()); tot_e2 = sum(float((np.concatenate(v["e"]) ** 2).sum()) for v in acc[r].values() if v["e"])
        print(f"\n{r}: {tot_n:,} hit pixels on {len(pick)} positions x 4 faces; overall delay RMSE {math.sqrt(tot_e2 / tot_n):.2f} ns")
        print(f"  {'paths/pixel':>11} {'share px':>9} {'median':>7} {'P90':>7} {'RMSE':>7} {'share of squared error':>23}")
        out["runs"][r] = {}
        for label, _, _ in CLASSES:
            v = acc[r][label]
            if not v["e"]:
                continue
            e = np.concatenate(v["e"]); row = {"share_pixels": v["n"] / tot_n, "median": float(np.median(e)), "p90": float(np.percentile(e, 90)),
                                              "rmse": float(np.sqrt((e ** 2).mean())), "share_sq_error": float((e ** 2).sum() / tot_e2)}
            out["runs"][r][label] = row
            print(f"  {label:>11} {row['share_pixels']:9.1%} {row['median']:7.2f} {row['p90']:7.2f} {row['rmse']:7.2f} {row['share_sq_error']:23.1%}")
    json.dump(out, open(os.path.join(REPO, "output/rrf/diag_delay_paths.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
