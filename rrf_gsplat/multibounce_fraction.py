"""How much received power arrives after more than one interaction?

Everything in this repository solves at max_depth 1 (the tutorial's setting),
and the Tx-conditioning argument rests on first-order scattering. This solves
at depth 3 at eight receiver positions of the route and reports the share of
the received power carried by paths with 0, 1, 2 and 3 interactions, for the
released data's setting (2.4 GHz, tutorial materials, both the as-is and the
unit-fixed variant) and for the 60 GHz uniform-scattering setting the Tx
experiments use. Same solver settings as the generator except max_depth.

This is an audit of an assumption, not a tuning run: if the power beyond depth
1 is small, the depth-1 datasets are defensible; if it is not, that is a stated
limitation rather than something to fix by raising the depth everywhere.

    PYTHONUTF8=1 python rrf_gsplat/multibounce_fraction.py
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO, "sionna_port"))

# Both tutorial variants are measured, because the as-is one has near-PEC walls
# and would plausibly bounce very differently from the unit-fixed one.
SETTINGS = [
    ("2.4 GHz, tutorial materials (unit-fixed)", dict(frequency=2.4e9, materials="tutorial")),
    ("2.4 GHz, tutorial materials as-is (released data)", dict(frequency=2.4e9, materials="tutorial-asis")),
    ("60 GHz, uniform scattering 0.7 (Tx experiments)", dict(frequency=60e9, materials="uniform")),
]


def main():
    """Solve at depth 3 for each setting and split the power by interaction count."""
    import mitsuba as mi
    mi.set_variant("cuda_ad_mono_polarized")
    from sionna.rt import PathSolver, Receiver, Transmitter
    from generate_dataset import Config, build_scene, read_pose_groups, solve_paths

    xml = os.path.join(REPO, "sionna_tutorial/RF-3DGS_Sionna_simulation_tutorial/NIST_lobby_v1.0/NIST_lobby_V1.1_sionna12.xml")
    route = os.path.join(REPO, "sionna_tutorial/RF-3DGS_Sionna_simulation_tutorial/NIST_rx_loc.txt")
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--positions", type=int, default=8, help="route positions, evenly spaced (8 in the first run; 40 for the LoS / NLoS split)")
    args = ap.parse_args()
    groups = read_pose_groups(os.path.join(REPO, "RF-3DGS_dataset/regenerated/3dgs_MVDR_100/sparse/0/images.txt"))
    # Evenly spaced along the route, so both open and enclosed parts are covered.
    positions = [groups[i][0] for i in range(0, len(groups), max(1, len(groups) // args.positions))][:args.positions]
    out = {}
    for label, kw in SETTINGS:
        # Everything matches the generator except max_depth: the comparison is
        # against the datasets as they were actually made.
        cfg = Config(scene_xml=xml, rx_loc_file=route, out_dir="", spectrum="MULTI", dashboard=False, max_depth=3, **kw)
        scene = build_scene(cfg)
        scene.add(Transmitter(name="tx", position=list(cfg.tx_loc)))
        solver = PathSolver()
        rows = []
        for k, pos in enumerate(positions):
            if "rx" in scene.receivers:
                scene.remove("rx")
            scene.add(Receiver(name="rx", position=[float(v) for v in pos]))
            paths = solve_paths(solver, scene, cfg, view_index=k)
            a, tau = paths.cir(normalize_delays=False, out_type="numpy")
            a = np.asarray(a)                                   # [rx, rx_ant, tx, tx_ant, paths, time]
            p_path = (np.abs(a[0, :, 0, 0, :, 0]) ** 2).mean(0)  # power per path, averaged over the array
            tau = np.asarray(tau).reshape(-1)
            inter = np.asarray(paths.interactions)              # [depth, ..., paths], 0 = no interaction
            n_int = (inter.reshape(inter.shape[0], -1) != 0).sum(0)
            # Guards the reshape above: if the layout ever changes, this fails
            # loudly instead of silently mis-attributing paths to orders.
            assert n_int.shape[0] == p_path.shape[0], (inter.shape, p_path.shape)
            valid = np.isfinite(tau) & (tau >= 0) & (p_path > 0)
            p, n = p_path[valid], n_int[valid]
            tot = p.sum()
            frac = [float(p[n == d].sum() / tot) for d in range(cfg.max_depth + 1)]
            # "los" = a zero-interaction path exists at this position, which is
            # what splits the summary below.
            rows.append({"position": [round(float(v), 3) for v in pos], "paths": int(valid.sum()),
                         "paths_by_interactions": [int((n == d).sum()) for d in range(cfg.max_depth + 1)],
                         "power_fraction_by_interactions": frac, "total_power_db": float(10 * np.log10(tot)),
                         "los": bool((n == 0).any())})
        f = np.array([r["power_fraction_by_interactions"] for r in rows])
        beyond1 = f[:, 2:].sum(1)                  # orders 2 and 3: what depth 1 misses
        los = np.array([r["los"] for r in rows])
        def cls(mask):
            """Summary of `beyond1` over a subset, or {"n": 0} if it is empty."""
            if not mask.any():
                return {"n": 0}
            v = beyond1[mask]
            return {"n": int(mask.sum()), "median": float(np.median(v)), "mean": float(v.mean()), "p90": float(np.percentile(v, 90)), "max": float(v.max())}
        # min and max as well as the median: a low average with a high maximum
        # would mean depth 1 is fine on average and wrong exactly where it matters.
        summary = {"config": kw, "max_depth": cfg.max_depth, "samples_per_src": cfg.samples_per_src,
                   "mean_power_fraction_by_interactions": f.mean(0).round(4).tolist(),
                   "power_beyond_depth1_median": float(np.median(beyond1)), "power_beyond_depth1_min": float(beyond1.min()),
                   "power_beyond_depth1_max": float(beyond1.max()), "los": cls(los), "nlos": cls(~los), "rows": rows}
        out[label] = summary
        print(f"{label}: mean power share by interactions 0/1/2/3 = {summary['mean_power_fraction_by_interactions']}; "
              f"beyond depth 1: median {summary['power_beyond_depth1_median']:.1%} "
              f"(range {beyond1.min():.1%}..{beyond1.max():.1%}) over {len(rows)} positions; "
              f"paths per position {np.mean([r['paths'] for r in rows]):,.0f}")
        # The LoS / NLoS split is the expected confound: a position with no
        # direct path has to get its power from somewhere.
        for name, c in (("LoS", summary["los"]), ("NLoS", summary["nlos"])):
            if c["n"]:
                print(f"   {name} ({c['n']} positions): beyond depth 1 median {c['median']:.1%}, mean {c['mean']:.1%}, P90 {c['p90']:.1%}, max {c['max']:.1%}")
            else:
                print(f"   {name}: no positions")
    os.makedirs(os.path.join(REPO, "output/rrf"), exist_ok=True)
    json.dump(out, open(os.path.join(REPO, "output/rrf/multibounce_fraction.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
