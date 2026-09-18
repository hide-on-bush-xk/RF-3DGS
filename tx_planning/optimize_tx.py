"""Place the transmitter by gradient descent on a coverage objective.

The objective is evaluated over a grid of receiver positions inside the
differentiable ray tracer, so the gradient with respect to the transmitter
position comes straight out of Sionna -- the same gradient
probe_differentiability.py checked against finite differences.

Two objectives:
  mean-db     mean received power in dB over the grid (smooth, always defined)
  coverage    soft fraction of grid points above a threshold, a sigmoid of
              (P_dB - threshold) / width, which is what a planner actually wants

This is the "known digital twin" case: geometry and materials are given. It is
the planning loop that the material-fitting stage later feeds with fitted
materials, and the result to compare against a brute-force sweep.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np

from scene_common import (LOBBY_X, LOBBY_Y, RX_HEIGHT, clearance,
                          enable_reverse_mode, indoor_mask, load_radio_scene,
                          make_solver, rx_grid)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene-xml", required=True)
    ap.add_argument("--out", default="../output/tx_planning/optimize_tx.json")
    ap.add_argument("--tx-init", type=float, nargs=3, default=[6.905, 0.0, 0.287])
    ap.add_argument("--init-from", default=None,
                    help="tx_sweep.py output; start from its best-coverage Tx")
    ap.add_argument("--tx-height", type=float, default=None,
                    help="fix the transmitter height; otherwise z is optimised too")
    ap.add_argument("--rx-step", type=float, default=2.0)
    ap.add_argument("--objective", choices=["mean-db", "coverage"], default="mean-db")
    ap.add_argument("--threshold-db", type=float, default=-90.0)
    ap.add_argument("--noise-floor-db", type=float, default=-130.0,
                    help="power added to every point so an unreached one "
                         "costs a finite amount, not -300 dB")
    ap.add_argument("--width-db", type=float, default=5.0)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--margin", type=float, default=0.2,
                    help="minimum distance from the transmitter to any surface")
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--samples", type=int, default=20_000)
    ap.add_argument("--max-depth", type=int, default=1)
    ap.add_argument("--scattering", type=float, default=0.7)
    ap.add_argument("--variant", default="cuda_ad_mono_polarized")
    cfg = ap.parse_args()
    if cfg.init_from:
        sweep = np.load(cfg.init_from, allow_pickle=True)
        covered = np.nan_to_num(sweep["gain_db"], nan=-np.inf) > cfg.threshold_db
        cfg.tx_init = sweep["tx_positions"][np.argmax(covered.mean(1))].tolist()
        print(f"starting from sweep's best Tx {np.round(cfg.tx_init, 2).tolist()} "
              f"(coverage {covered.mean(1).max():.0%} on the sweep grid)")

    import drjit as dr
    import mitsuba as mi
    mi.set_variant(cfg.variant)
    enable_reverse_mode()
    from sionna.rt import Receiver, Transmitter

    scene = load_radio_scene(cfg.scene_xml, scattering=cfg.scattering)
    solver = make_solver(reverse_mode=True)

    # All receivers live in the scene at once, so one solve covers the grid.
    rx_positions = rx_grid(LOBBY_X, LOBBY_Y, RX_HEIGHT, cfg.rx_step)
    rx_positions = rx_positions[indoor_mask(scene, rx_positions)]
    for i, p in enumerate(rx_positions):
        scene.add(Receiver(name=f"rx{i}", position=[float(v) for v in p]))
    tx = Transmitter(name="tx", position=[float(v) for v in cfg.tx_init])
    scene.add(tx)
    print(f"{len(rx_positions)} indoor receivers on a {cfg.rx_step} m grid")
    noise = 10.0 ** (cfg.noise_floor_db / 10.0)

    def objective():
        paths = solver(scene=scene, max_depth=cfg.max_depth,
                       samples_per_src=cfg.samples, los=True,
                       specular_reflection=True, diffuse_reflection=True,
                       refraction=False, synthetic_array=True, seed=42)
        a_re, a_im = paths.a
        if a_re.shape[-1] == 0:
            return None, None                         # no path from anywhere
        # a: [num_rx, rx_ant, num_tx, tx_ant, paths]; sum every axis but rx
        p_rx = dr.square(a_re) + dr.square(a_im)
        for axis in range(p_rx.ndim - 1, 0, -1):
            p_rx = dr.sum(p_rx, axis=axis)                    # [num_rx]
        p_db = 10.0 * dr.log(p_rx + noise) / math.log(10.0)
        if cfg.objective == "mean-db":
            return dr.mean(p_db), p_db
        soft = 1.0 / (1.0 + dr.exp(-(p_db - cfg.threshold_db) / cfg.width_db))
        return dr.mean(soft), p_db

    def as_np(t):
        return np.asarray(t).reshape(-1)

    history = []
    pos = np.array(cfg.tx_init, dtype=float)
    m = np.zeros(3); v = np.zeros(3)                  # Adam state
    b1, b2, eps = 0.9, 0.999, 1e-8
    t0 = time.time()
    for step in range(1, cfg.steps + 1):
        tx.position = [float(x) for x in pos]
        p = tx.position
        dr.enable_grad(p)
        obj, p_db = objective()
        if obj is None:
            print(f"  step {step:3d}  no paths at all from "
                  f"({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}): the transmitter "
                  "moved inside an object, stopping")
            break
        dr.backward(obj)
        g = as_np(dr.grad(p)).astype(float)
        obj_v = float(as_np(obj)[0])
        cov = float(np.mean(as_np(p_db) > cfg.threshold_db))
        history.append({"step": step, "position": pos.tolist(),
                        "objective": obj_v, "coverage_frac": cov,
                        "grad": g.tolist()})
        print(f"  step {step:3d}  obj {obj_v:9.4f}  coverage>{cfg.threshold_db:.0f}dB "
              f"{cov:5.1%}  pos ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})  "
              f"|g| {np.linalg.norm(g):.2e}")
        if not np.all(np.isfinite(g)):
            print("  non-finite gradient, stopping")
            break
        # Adam, ascent on the objective.
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        mh, vh = m / (1 - b1 ** step), v / (1 - b2 ** step)
        upd = cfg.lr * mh / (np.sqrt(vh) + eps)
        if cfg.tx_height is not None:
            upd[2] = 0.0
            pos[2] = cfg.tx_height
        # Keep the transmitter indoors and off the walls: the gradient knows
        # nothing about the wall it is about to push through (step 6 of the
        # first run walked straight out of the building at x = 8.48).
        def admissible(cand):
            return (indoor_mask(scene, cand[None])[0]
                    and clearance(scene, cand) >= cfg.margin)
        # Try the full step, then the step with the wall-normal component
        # removed (sliding along the wall), then shorter steps.
        trials = [upd, upd * [0, 1, 1], upd * [1, 0, 1], upd * [0, 0, 1]]
        trials += [upd / 2 ** k for k in (1, 2, 3)]
        for trial in trials:
            cand = pos + trial
            cand[0] = np.clip(cand[0], *LOBBY_X)
            cand[1] = np.clip(cand[1], *LOBBY_Y)
            if np.any(trial != 0) and admissible(cand):
                pos = cand
                break
        else:
            print("  blocked by a surface in every direction tried, stopping")
            break

    os.makedirs(os.path.dirname(cfg.out) or ".", exist_ok=True)
    with open(cfg.out, "w", encoding="utf-8") as fid:
        json.dump({"config": vars(cfg), "history": history,
                   "seconds": time.time() - t0}, fid, indent=1)
    best = max(history, key=lambda h: h["objective"])
    print(f"\nbest: obj {best['objective']:.4f} at step {best['step']}, "
          f"pos {np.round(best['position'], 2).tolist()}  "
          f"({time.time()-t0:.0f} s)")


if __name__ == "__main__":
    main()
