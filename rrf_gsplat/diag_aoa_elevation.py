"""Where do the paths arrive from? Elevation distribution of the AoA.

Hypothesis behind the azimuth / zenith asymmetry of the sigma sweep
(power-weighted: azimuth prefers the narrowest kernel, zenith the widest):
indoor paths concentrate near the horizontal plane, so the azimuth axis is
crowded (a narrow kernel keeps neighbours apart) while the zenith axis is
sparse (a wide kernel buys coverage). This solves a few route positions at
the multi-channel setting and prints the AoA elevation histogram, by path
count and by power, plus the nearest-neighbour spacing of paths along
azimuth and along zenith.

    PYTHONUTF8=1 python rrf_gsplat/diag_aoa_elevation.py
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO, "sionna_port"))


def main():
    import mitsuba as mi
    mi.set_variant("cuda_ad_mono_polarized")
    from sionna.rt import PathSolver, Receiver, Transmitter
    from generate_dataset import Config, build_scene, read_pose_groups, solve_paths
    from rf_spectra import _path_arrays
    xml = os.path.join(REPO, "sionna_tutorial/RF-3DGS_Sionna_simulation_tutorial/NIST_lobby_v1.0/NIST_lobby_V1.1_sionna12.xml")
    route = os.path.join(REPO, "sionna_tutorial/RF-3DGS_Sionna_simulation_tutorial/NIST_rx_loc.txt")
    cfg = Config(scene_xml=xml, rx_loc_file=route, out_dir="", spectrum="MULTI", frequency=2.4e9, materials="tutorial", dashboard=False)
    scene = build_scene(cfg); scene.add(Transmitter(name="tx", position=list(cfg.tx_loc))); solver = PathSolver()
    groups = read_pose_groups(os.path.join(REPO, "RF-3DGS_dataset/regenerated/3dgs_MVDR_100/sparse/0/images.txt"))
    positions = [groups[i][0] for i in range(0, len(groups), len(groups) // 12)][:12]
    edges = np.array([-90, -60, -45, -34, -25, -15, -5, 5, 15, 25, 34, 45, 60, 90], float)
    cnt, pw = np.zeros(len(edges) - 1), np.zeros(len(edges) - 1)
    spacing = {"az": [], "zen": []}
    for k, pos in enumerate(positions):
        if "rx" in scene.receivers:
            scene.remove("rx")
        scene.add(Receiver(name="rx", position=[float(v) for v in pos]))
        paths = solve_paths(solver, scene, cfg, view_index=k)
        amp, tau, th_r, ph_r, _, _ = (t.cpu().numpy() for t in _path_arrays(paths))
        el = 90.0 - np.degrees(th_r); p = amp ** 2
        c, _ = np.histogram(el, edges); w, _ = np.histogram(el, edges, weights=p)
        cnt += c; pw += w / p.sum()
        # nearest-neighbour spacing of the strongest 2000 paths, along each axis, on the equirect grid (deg)
        top = np.argsort(p)[-2000:]
        az_d, ze_d = np.degrees(ph_r[top]), np.degrees(th_r[top])
        for key, a, b in (("az", az_d, ze_d), ("zen", ze_d, az_d)):
            same = np.abs(b[:, None] - b[None]) < 1.0                 # neighbours in the other coordinate
            da = np.abs(a[:, None] - a[None]); da[~same] = np.inf; np.fill_diagonal(da, np.inf)
            nn = da.min(1); spacing[key].append(nn[np.isfinite(nn)])
    cnt /= cnt.sum(); pw /= len(positions)
    print("AoA elevation histogram over 12 positions (share of paths / share of power):")
    for i in range(len(edges) - 1):
        print(f"  [{edges[i]:4.0f}, {edges[i+1]:4.0f}) deg: {cnt[i]:6.1%} / {pw[i]:6.1%}")
    within = lambda lim: float(pw[(edges[:-1] >= -lim) & (edges[1:] <= lim)].sum())
    print(f"power within +-25 deg: {within(25):.1%}; within +-34 deg: {within(34):.1%}; within +-15 deg: {within(15):.1%}")
    out = {"edges_deg": edges.tolist(), "share_paths": cnt.round(4).tolist(), "share_power": pw.round(4).tolist()}
    for key in ("az", "zen"):
        s = np.concatenate(spacing[key])
        out[f"nn_spacing_{key}_deg"] = {"median": float(np.median(s)), "p10": float(np.percentile(s, 10)), "share_lt_1deg": float((s < 1).mean())}
        print(f"nearest-neighbour spacing along {key} among the 2000 strongest paths (within 1 deg in the other axis): "
              f"median {np.median(s):.2f} deg, P10 {np.percentile(s, 10):.2f}, share < 1 deg {(s < 1).mean():.1%}")
    json.dump(out, open(os.path.join(REPO, "output/rrf/diag_aoa_elevation.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
