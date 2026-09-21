"""Indoor transmitter placement benchmark: our gradient planner against the standard baselines, same KPIs as the
city-scale placement literature (coverage, mean rate, 5th-percentile "edge" rate), same ray tracer, same budget book.

Every method places K transmitters at a fixed height on the same scene and is charged one "solve" per ray-tracer
evaluation of the whole receiver grid (all receivers in one launch). Placements are then re-evaluated with a common
evaluator at a higher sample count, so the KPIs are not the optimiser's own noisy estimate.

  exhaustive   every admissible cell of a candidate grid (step --cand-step at --tx-z), one solve each; K = 1 takes the
               best cell, K > 1 is the submodular greedy on the same precomputed candidate gains (max-serving per Rx),
               which costs no extra solve (the structure of Taus/Tsai/Andrews, arXiv 2604.28153, on an indoor grid)
  random       --budget admissible positions drawn uniformly, one solve each (K positions drawn jointly)
  nelder-mead  scipy Nelder-Mead on the soft coverage from the exhaustive best cell, --budget evaluations
  gradient     Adam on Sionna's own gradient of the soft coverage (optimize_tx.py's rule: wall-normal component removed
               when blocked), --steps steps from the exhaustive best cell and from --restarts random starts; K positions
               are optimised jointly on the per-Rx maximum over transmitters

KPIs (rate in bit/s/Hz): coverage = share of receivers whose best-serving gain exceeds --threshold-db; SNR = gain + P_tx
- N with N = -174 dBm/Hz + 10 log10(B) + NF; rate = log2(1 + SNR); mean rate and the 5th percentile over receivers.

    PYTHONUTF8=1 python tx_planning/benchmark_placement.py --scene-xml ../sionna_tutorial/.../NIST_lobby_V1.1_sionna12.xml --k 1 2
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scene_common import LOBBY_X, LOBBY_Y, RX_HEIGHT, clearance, enable_reverse_mode, indoor_mask, load_radio_scene, make_solver, rx_grid  # noqa: E402


class Evaluator:
    """One scene, all receivers resident; K transmitters swapped by position. Counts solves."""

    def __init__(self, cfg):
        import mitsuba as mi
        mi.set_variant(cfg.variant); enable_reverse_mode()
        from sionna.rt import Receiver, Transmitter
        self.cfg = cfg; self.Transmitter = Transmitter
        self.scene = load_radio_scene(cfg.scene_xml, scattering=cfg.scattering)
        self.solver = make_solver(reverse_mode=True)
        rx = rx_grid(cfg.x_range, cfg.y_range, cfg.rx_z, cfg.rx_step)
        self.rx = rx[indoor_mask(self.scene, rx)]
        for i, p in enumerate(self.rx):
            self.scene.add(Receiver(name=f"rx{i}", position=[float(v) for v in p]))
        self.txs = []; self.solves = 0; self.solve_seconds = 0.0
        self.noise_lin = 10.0 ** (cfg.noise_floor_db / 10.0)
        n_dbm = -174.0 + 10.0 * math.log10(cfg.bandwidth_hz) + cfg.noise_figure_db
        self.snr_offset_db = cfg.tx_power_dbm - n_dbm          # SNR_dB = gain_dB + offset

    def set_txs(self, positions):
        for t in self.txs:
            self.scene.remove(t.name)
        self.txs = []
        for k, p in enumerate(positions):
            t = self.Transmitter(name=f"tx{k}", position=[float(v) for v in p]); self.scene.add(t); self.txs.append(t)

    def solve(self, samples, seed=42):
        import drjit as dr
        t0 = time.time()
        paths = self.solver(scene=self.scene, max_depth=self.cfg.max_depth, max_num_paths_per_src=10_000_000, samples_per_src=samples,
                            los=True, specular_reflection=True, diffuse_reflection=True, refraction=False, synthetic_array=True, seed=seed)
        self.solves += 1; self.solve_seconds += time.time() - t0
        return paths

    def gains_db(self, positions, samples):
        """[n_rx, K] gain in dB (numpy) for a placement."""
        import drjit as dr
        self.set_txs(positions)
        paths = self.solve(samples)
        a_re, a_im = paths.a
        if a_re.shape[-1] == 0:
            return np.full((len(self.rx), len(positions)), -300.0)
        p = dr.square(a_re) + dr.square(a_im)                    # [rx, rx_ant, tx, tx_ant, paths]
        p = dr.sum(dr.sum(dr.sum(p, axis=4), axis=3), axis=1)      # [rx, tx]
        g = np.asarray(p).reshape(len(self.rx), len(positions))
        return 10.0 * np.log10(g + self.noise_lin)

    def kpis(self, gains_db):
        best = gains_db.max(axis=1)
        snr = 10.0 ** ((best + self.snr_offset_db) / 10.0)
        rate = np.log2(1.0 + snr)
        return {"coverage": float((best > self.cfg.threshold_db).mean()), "mean_rate": float(rate.mean()),
                "edge_rate_p5": float(np.percentile(rate, 5)), "rate_p25": float(np.percentile(rate, 25)), "rate_p50": float(np.percentile(rate, 50)),
                "mean_gain_db": float(best.mean())}

    def admissible(self, p):
        return bool(indoor_mask(self.scene, np.asarray(p, float)[None])[0]) and clearance(self.scene, p) >= self.cfg.margin

    def random_position(self, rng):
        while True:
            p = np.array([rng.uniform(*self.cfg.x_range), rng.uniform(*self.cfg.y_range), self.cfg.tx_z])
            if self.admissible(p):
                return p

    def soft_objective_and_grad(self, positions):
        """Soft coverage of the per-Rx maximum over transmitters, and its gradient w.r.t. every Tx position."""
        import drjit as dr
        self.set_txs(positions)
        ps = [t.position for t in self.txs]
        for p in ps:
            dr.enable_grad(p)
        paths = self.solve(self.cfg.samples)
        a_re, a_im = paths.a
        if a_re.shape[-1] == 0:
            return None, None, None
        p = dr.square(a_re) + dr.square(a_im)
        p = dr.sum(dr.sum(dr.sum(p, axis=4), axis=3), axis=1)      # [rx, tx]
        best = p[:, 0] if len(positions) == 1 else dr.max(p, axis=1)
        p_db = 10.0 * dr.log(best + self.noise_lin) / math.log(10.0)
        soft = 1.0 / (1.0 + dr.exp(-(p_db - self.cfg.threshold_db) / self.cfg.width_db))
        obj = dr.mean(soft)
        dr.backward(obj)
        grads = [np.asarray(dr.grad(q)).reshape(-1).astype(float) for q in ps]
        return float(np.asarray(obj).reshape(-1)[0]), np.asarray(p_db).reshape(-1), grads


def candidates(ev, step):
    grid = rx_grid(ev.cfg.x_range, ev.cfg.y_range, ev.cfg.tx_z, step)
    return np.array([p for p in grid if ev.admissible(p)])


def exhaustive(ev, cand, samples):
    """Per-candidate gains [n_cand, n_rx] (one solve each) and the greedy placements for every K."""
    G = np.stack([ev.gains_db([c], samples)[:, 0] for c in cand])
    out = {}
    chosen = []; combined = np.full(len(ev.rx), -300.0)
    for k in range(1, max(ev.cfg.k) + 1):
        best_j, best_val = None, None
        for j in range(len(cand)):
            if j in chosen:
                continue
            kp = ev.kpis(np.maximum(combined, G[j])[:, None])
            key = (kp["coverage"], kp["mean_rate"])
            if best_val is None or key > best_val:
                best_j, best_val = j, key
        chosen.append(best_j); combined = np.maximum(combined, G[best_j])
        out[k] = [cand[j].tolist() for j in chosen]
    return G, out


def gradient(ev, start, steps, lr=0.1):
    pos = [np.array(p, float) for p in start]
    m = [np.zeros(3) for _ in pos]; v = [np.zeros(3) for _ in pos]; b1, b2, eps = 0.9, 0.999, 1e-8
    best = (None, -1.0)
    for step in range(1, steps + 1):
        obj, p_db, grads = ev.soft_objective_and_grad(pos)
        if obj is None:
            break
        cov = float((p_db > ev.cfg.threshold_db).mean())
        if obj > best[1]:
            best = ([p.copy() for p in pos], obj)
        for k in range(len(pos)):
            g = grads[k]; m[k] = b1 * m[k] + (1 - b1) * g; v[k] = b2 * v[k] + (1 - b2) * g * g
            upd = lr * (m[k] / (1 - b1 ** step)) / (np.sqrt(v[k] / (1 - b2 ** step)) + eps)     # ascent
            upd[2] = 0.0
            for trial in [upd, upd * [0, 1, 1], upd * [1, 0, 1]] + [upd / 2 ** j for j in (1, 2, 3)]:
                cand = pos[k] + trial; cand[0] = np.clip(cand[0], *ev.cfg.x_range); cand[1] = np.clip(cand[1], *ev.cfg.y_range)
                if np.any(trial != 0) and ev.admissible(cand):
                    pos[k] = cand; break
    return best[0] if best[0] is not None else pos


def nelder_mead(ev, start, budget):
    from scipy.optimize import minimize
    K = len(start)
    def f(x):
        pts = [np.array([x[2 * k], x[2 * k + 1], ev.cfg.tx_z]) for k in range(K)]
        if not all(ev.admissible(p) for p in pts):
            return 1.0
        obj, _, _ = ev.soft_objective_and_grad(pts)
        return 1.0 - (obj if obj is not None else 0.0)
    x0 = np.concatenate([[p[0], p[1]] for p in start])
    r = minimize(f, x0, method="Nelder-Mead", options={"maxfev": budget, "xatol": 0.05, "fatol": 1e-4, "initial_simplex": None})
    return [np.array([r.x[2 * k], r.x[2 * k + 1], ev.cfg.tx_z]) for k in range(K)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene-xml", required=True); ap.add_argument("--out", default="../output/tx_planning/benchmark_placement.json")
    ap.add_argument("--x-range", type=float, nargs=2, default=list(LOBBY_X)); ap.add_argument("--y-range", type=float, nargs=2, default=list(LOBBY_Y))
    ap.add_argument("--rx-z", type=float, default=RX_HEIGHT); ap.add_argument("--tx-z", type=float, default=2.0); ap.add_argument("--rx-step", type=float, default=1.0)
    ap.add_argument("--cand-step", type=float, default=1.0); ap.add_argument("--k", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--budget", type=int, default=150, help="solves for random search and Nelder-Mead"); ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--restarts", type=int, default=2); ap.add_argument("--samples", type=int, default=20_000); ap.add_argument("--eval-samples", type=int, default=200_000)
    ap.add_argument("--threshold-db", type=float, default=-85.0); ap.add_argument("--width-db", type=float, default=5.0); ap.add_argument("--noise-floor-db", type=float, default=-130.0)
    ap.add_argument("--tx-power-dbm", type=float, default=20.0); ap.add_argument("--bandwidth-hz", type=float, default=400e6); ap.add_argument("--noise-figure-db", type=float, default=7.0)
    ap.add_argument("--margin", type=float, default=0.2); ap.add_argument("--max-depth", type=int, default=1); ap.add_argument("--scattering", type=float, default=0.7)
    ap.add_argument("--variant", default="cuda_ad_mono_polarized"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--control", type=float, nargs=3, default=None, help="known placement to evaluate first (e.g. the earlier optimum)")
    cfg = ap.parse_args()
    rng = np.random.default_rng(cfg.seed)
    ev = Evaluator(cfg)
    print(f"{len(ev.rx)} indoor receivers; SNR = gain + {ev.snr_offset_db:.1f} dB; threshold {cfg.threshold_db} dB")
    results = {"config": vars(cfg), "n_rx": int(len(ev.rx)), "snr_offset_db": ev.snr_offset_db, "methods": {}}
    if cfg.control:
        kp = ev.kpis(ev.gains_db([np.array(cfg.control)], cfg.samples)); print(f"control {cfg.control} at {cfg.samples} samples: {kp}")
        results["control"] = {"position": cfg.control, "samples": cfg.samples, **kp}

    def record(name, K, placement, solves, seconds):
        g = ev.gains_db(placement, cfg.eval_samples); kp = ev.kpis(g)
        results["methods"][f"{name}_K{K}"] = {"placement": [list(map(float, p)) for p in placement], "solves": int(solves), "seconds": float(seconds), **kp}
        print(f"  {name:12s} K={K}: coverage {kp['coverage']:.3f}  mean rate {kp['mean_rate']:.2f}  edge rate {kp['edge_rate_p5']:.2f} bit/s/Hz  "
              f"({solves} solves, {seconds:.0f} s)  at {np.round(placement, 2).tolist()}")

    # exhaustive + greedy
    cand = candidates(ev, cfg.cand_step); print(f"{len(cand)} admissible candidate cells at {cfg.cand_step} m")
    s0, t0 = ev.solves, time.time(); G, greedy = exhaustive(ev, cand, cfg.samples); ex_solves, ex_sec = ev.solves - s0, time.time() - t0
    for K in cfg.k:
        record("exhaustive" if K == 1 else "greedy", K, [np.array(p) for p in greedy[K]], ex_solves, ex_sec)
    best_cell = np.array(greedy[1][0])
    for K in cfg.k:
        # random search
        s0, t0 = ev.solves, time.time(); best = (None, None)
        for _ in range(cfg.budget):
            pl = [ev.random_position(rng) for _ in range(K)]; kp = ev.kpis(ev.gains_db(pl, cfg.samples)); key = (kp["coverage"], kp["mean_rate"])
            if best[1] is None or key > best[1]:
                best = (pl, key)
        record("random", K, best[0], ev.solves - s0, time.time() - t0)
        # Nelder-Mead from the exhaustive best cell(s)
        start = [np.array(p) for p in greedy[K]]
        s0, t0 = ev.solves, time.time(); pl = nelder_mead(ev, start, cfg.budget); record("nelder-mead", K, pl, ev.solves - s0, time.time() - t0)
        # gradient from the exhaustive best cell(s), then random restarts; report the best and every start
        s0, t0 = ev.solves, time.time(); pl = gradient(ev, start, cfg.steps); record("gradient", K, pl, ev.solves - s0, time.time() - t0)
        for r in range(cfg.restarts):
            st = [ev.random_position(rng) for _ in range(K)]
            s0, t0 = ev.solves, time.time(); pl = gradient(ev, st, cfg.steps); record(f"gradient_rand{r}", K, pl, ev.solves - s0, time.time() - t0)
    results["total_solves"] = ev.solves; results["total_solve_seconds"] = ev.solve_seconds
    os.makedirs(os.path.dirname(os.path.abspath(cfg.out)), exist_ok=True)
    json.dump(results, open(cfg.out, "w"), indent=1)
    print(f"{ev.solves} solves, {ev.solve_seconds:.0f} s in the solver -> {cfg.out}")


if __name__ == "__main__":
    main()
