"""Ground truth for transmitter planning: candidate Tx positions x a receiver grid.

Output is structured, not pictures. Per (Tx, Rx) pair it stores what Aerial's
CIRResultsRequest hands to its RAN simulator -- complex path gains, delays,
angles of departure and arrival -- plus the total path gain the planner scores
coverage on. This is the table an indoor placement optimiser evaluates against,
and the observation set the material-fitting stage later inverts.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from scene_common import (LOBBY_X, LOBBY_Y, RX_HEIGHT, indoor_mask,
                          load_radio_scene, make_solver, rx_grid)


def solve_tx(solver, scene, cfg):
    """One solve for every receiver in the scene; returns per-Rx path lists.

    All receivers share a single launch, which is ~15x faster than one solve
    per (Tx, Rx) pair: the sampling cost is per source, not per receiver.
    """
    paths = solver(scene=scene, max_depth=cfg.max_depth,
                   samples_per_src=cfg.samples, los=True,
                   specular_reflection=True, diffuse_reflection=True,
                   refraction=False, synthetic_array=True, seed=cfg.seed)
    a, tau = paths.cir(normalize_delays=False, out_type="numpy")
    a = np.asarray(a)[:, 0, 0, 0, :, 0]             # [num_rx, paths]
    tau = np.asarray(tau)[:, 0, :]
    angles = [np.asarray(getattr(paths, k))[:, 0, :]
              for k in ("theta_t", "phi_t", "theta_r", "phi_r")]
    valid = np.asarray(paths.valid)[:, 0, :] & np.isfinite(tau) & (tau >= 0)
    out = []
    for r in range(a.shape[0]):
        keep = valid[r]
        out.append((a[r, keep], tau[r, keep],
                    np.stack([angles[0][r, keep], angles[1][r, keep]], axis=1),
                    np.stack([angles[2][r, keep], angles[3][r, keep]], axis=1)))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene-xml", required=True)
    ap.add_argument("--out", default="../output/tx_planning/tx_sweep.npz")
    ap.add_argument("--tx", type=float, nargs=3, action="append", default=None,
                    help="candidate transmitter position; repeatable")
    ap.add_argument("--tx-grid-step", type=float, default=None,
                    help="instead of --tx, sweep a coarse grid at --tx-height")
    ap.add_argument("--tx-height", type=float, default=2.0)
    ap.add_argument("--rx-step", type=float, default=1.0)
    ap.add_argument("--max-depth", type=int, default=1)
    ap.add_argument("--samples", type=int, default=200_000)
    ap.add_argument("--scattering", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-paths-stored", type=int, default=2000,
                    help="keep only the strongest paths per pair on disk")
    ap.add_argument("--variant", default="cuda_ad_mono_polarized")
    cfg = ap.parse_args()

    import mitsuba as mi
    mi.set_variant(cfg.variant)
    from sionna.rt import Receiver, Transmitter

    if cfg.tx is None and cfg.tx_grid_step is None:
        cfg.tx = [[6.905, 0.0, 0.287]]           # the NIST measurement Tx
    tx_positions = (np.array(cfg.tx, dtype=float) if cfg.tx is not None else
                    rx_grid(LOBBY_X, LOBBY_Y, cfg.tx_height, cfg.tx_grid_step))
    scene = load_radio_scene(cfg.scene_xml, scattering=cfg.scattering)
    solver = make_solver()
    # Only points inside the building can be served or serve.
    tx_positions = tx_positions[indoor_mask(scene, tx_positions)]
    rx_positions = rx_grid(LOBBY_X, LOBBY_Y, RX_HEIGHT, cfg.rx_step)
    rx_positions = rx_positions[indoor_mask(scene, rx_positions)]
    print(f"{len(tx_positions)} Tx candidates x {len(rx_positions)} indoor Rx points")

    n_tx, n_rx = len(tx_positions), len(rx_positions)
    gain_db = np.full((n_tx, n_rx), np.nan, dtype=np.float32)
    n_paths = np.zeros((n_tx, n_rx), dtype=np.int32)
    cir = {}                                       # (ti, ri) -> arrays
    t0 = time.time()
    for i, p in enumerate(rx_positions):
        scene.add(Receiver(name=f"rx{i}", position=[float(v) for v in p]))
    for ti, tx_pos in enumerate(tx_positions):
        if "tx" in scene.transmitters:
            scene.remove("tx")
        scene.add(Transmitter(name="tx", position=[float(v) for v in tx_pos]))
        for ri, (a, tau, aod, aoa) in enumerate(solve_tx(solver, scene, cfg)):
            n_paths[ti, ri] = a.size
            if a.size == 0:
                continue
            power = (np.abs(a) ** 2)
            gain_db[ti, ri] = 10 * np.log10(power.sum())
            order = np.argsort(-power)[:cfg.max_paths_stored]
            cir[(ti, ri)] = (a[order].astype(np.complex64),
                             tau[order].astype(np.float32),
                             aod[order].astype(np.float32),
                             aoa[order].astype(np.float32))
        rate = (ti + 1) / (time.time() - t0)
        print(f"  tx {ti+1}/{n_tx}: mean gain {np.nanmean(gain_db[ti]):.1f} dB, "
              f"coverage>-85 dB {np.mean(gain_db[ti] > -85):.0%}, {rate:.2f} tx/s")

    os.makedirs(os.path.dirname(cfg.out) or ".", exist_ok=True)
    # Ragged CIRs go in as object arrays keyed by pair index.
    keys = np.array(sorted(cir.keys()), dtype=np.int32)
    np.savez_compressed(
        cfg.out,
        tx_positions=tx_positions.astype(np.float32),
        rx_positions=rx_positions.astype(np.float32),
        gain_db=gain_db, n_paths=n_paths, cir_keys=keys,
        cir_a=np.array([cir[tuple(k)][0] for k in keys], dtype=object),
        cir_tau=np.array([cir[tuple(k)][1] for k in keys], dtype=object),
        cir_aod=np.array([cir[tuple(k)][2] for k in keys], dtype=object),
        cir_aoa=np.array([cir[tuple(k)][3] for k in keys], dtype=object),
        meta=json.dumps(vars(cfg)))
    print(f"wrote {cfg.out}: gain table {gain_db.shape}, "
          f"{len(keys)} CIRs, {time.time()-t0:.0f} s")


if __name__ == "__main__":
    main()
