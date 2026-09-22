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
import gc
import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scene_common import LOBBY_X, LOBBY_Y, RX_HEIGHT, clearance, enable_reverse_mode, indoor_mask, load_radio_scene, make_solver, rx_grid  # noqa: E402


class Evaluator:
    """One scene, all receivers resident; K transmitters swapped by position. Counts solves.

    The solve counter is the fairness mechanism of the whole benchmark: every
    method is charged the same way, so "gradient beat random" is a claim about
    equal ray-tracer budgets rather than equal wall-clock or equal code effort.
    """

    def __init__(self, cfg):
        import mitsuba as mi
        mi.set_variant(cfg.variant); enable_reverse_mode()
        from sionna.rt import Receiver, Transmitter
        self.cfg = cfg; self.Transmitter = Transmitter
        self.scene = load_radio_scene(cfg.scene_xml, scattering=cfg.scattering)
        self.solver = make_solver(reverse_mode=True)
        rx = rx_grid(cfg.x_range, cfg.y_range, cfg.rx_z, cfg.rx_step)
        self.rx = rx[indoor_mask(self.scene, rx)]
        # Receivers are added once and never moved; only transmitters change.
        for i, p in enumerate(self.rx):
            self.scene.add(Receiver(name=f"rx{i}", position=[float(v) for v in p]))
        self.txs = []; self.solves = 0; self.solve_seconds = 0.0
        self.noise_lin = 10.0 ** (cfg.noise_floor_db / 10.0)
        # Thermal noise in dBm over the bandwidth, plus the receiver noise figure.
        n_dbm = -174.0 + 10.0 * math.log10(cfg.bandwidth_hz) + cfg.noise_figure_db
        self.snr_offset_db = cfg.tx_power_dbm - n_dbm          # SNR_dB = gain_dB + offset

    def set_txs(self, positions):
        """Replace the transmitter set. Removing and re-adding is required
        because Sionna keys transmitters by name, not by index."""
        for t in self.txs:
            self.scene.remove(t.name)
        self.txs = []
        for k, p in enumerate(positions):
            t = self.Transmitter(name=f"tx{k}", position=[float(v) for v in p]); self.scene.add(t); self.txs.append(t)

    def solve(self, samples, seed=42):
        """One ray-tracer launch, counted and timed. The single place `solves` grows."""
        import drjit as dr
        t0 = time.time()
        paths = self.solver(scene=self.scene, max_depth=self.cfg.max_depth, max_num_paths_per_src=10_000_000, samples_per_src=samples,
                            los=True, specular_reflection=True, diffuse_reflection=True, refraction=False, synthetic_array=True, seed=seed)
        self.solves += 1; self.solve_seconds += time.time() - t0
        # Dr.Jit's allocator holds on to freed blocks; without this the benchmark
        # runs out of GPU memory partway through a long exhaustive sweep.
        if self.solves % 25 == 0:
            gc.collect(); dr.flush_malloc_cache()
        if self.solves % 100 == 0:
            # A periodic memory report, so a slow leak is visible in the log.
            import subprocess
            try:
                mem = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader"], capture_output=True, text=True, timeout=10).stdout.strip()
            except Exception:
                mem = "?"
            print(f"    [solve {self.solves}: GPU memory used {mem}]", flush=True)
        return paths

    def gains_db(self, positions, samples):
        """[n_rx, K] gain in dB (numpy) for a placement."""
        import drjit as dr
        self.set_txs(positions)
        paths = self.solve(samples)
        a_re, a_im = paths.a
        if a_re.shape[-1] == 0:
            return np.full((len(self.rx), len(positions)), -300.0)   # nothing reached
        p = dr.square(a_re) + dr.square(a_im)                    # [rx, rx_ant, tx, tx_ant, paths]
        # Sum over paths (4), tx antennas (3) and rx antennas (1), KEEPING the
        # transmitter axis: which transmitter serves which receiver is the point.
        p = dr.sum(dr.sum(dr.sum(p, axis=4), axis=3), axis=1)      # [rx, tx]
        g = np.asarray(p).reshape(len(self.rx), len(positions))
        return 10.0 * np.log10(g + self.noise_lin)

    def kpis(self, gains_db):
        """Coverage and rate statistics from a [n_rx, K] gain table.

        max over transmitters = each receiver is served by its best transmitter,
        the standard assumption in the placement literature (no joint transmission).
        """
        best = gains_db.max(axis=1)
        snr = 10.0 ** ((best + self.snr_offset_db) / 10.0)
        rate = np.log2(1.0 + snr)                                 # Shannon, per Hz
        # The 5th percentile is the "edge" user: a placement can raise the mean
        # while leaving the worst-served receivers exactly where they were.
        return {"coverage": float((best > self.cfg.threshold_db).mean()), "mean_rate": float(rate.mean()),
                "edge_rate_p5": float(np.percentile(rate, 5)), "rate_p25": float(np.percentile(rate, 25)), "rate_p50": float(np.percentile(rate, 50)),
                "mean_gain_db": float(best.mean())}

    def admissible(self, p):
        """Inside the building and at least --margin from any surface."""
        return bool(indoor_mask(self.scene, np.asarray(p, float)[None])[0]) and clearance(self.scene, p) >= self.cfg.margin

    def random_position(self, rng):
        """Rejection-sample one admissible position at the transmitter height."""
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
        # The K=1 special case avoids a dr.max over a length-1 axis, whose
        # gradient is well defined but needlessly indirect.
        best = p[:, 0] if len(positions) == 1 else dr.max(p, axis=1)
        p_db = 10.0 * dr.log(best + self.noise_lin) / math.log(10.0)
        # Sigmoid instead of a hard count, for the same reason as optimize_tx.py:
        # a step function has no usable gradient.
        soft = 1.0 / (1.0 + dr.exp(-(p_db - self.cfg.threshold_db) / self.cfg.width_db))
        obj = dr.mean(soft)
        dr.backward(obj)
        # Note the max means only the serving transmitter receives gradient at
        # each receiver, which is what makes the K transmitters specialise.
        grads = [np.asarray(dr.grad(q)).reshape(-1).astype(float) for q in ps]
        return float(np.asarray(obj).reshape(-1)[0]), np.asarray(p_db).reshape(-1), grads


def candidates(ev, step):
    """The admissible cells of the candidate grid, at transmitter height."""
    grid = rx_grid(ev.cfg.x_range, ev.cfg.y_range, ev.cfg.tx_z, step)
    return np.array([p for p in grid if ev.admissible(p)])


def exhaustive(ev, cand, samples):
    """Per-candidate gains [n_cand, n_rx] (one solve each) and the greedy placements for every K.

    The greedy pass costs no solves at all: the per-candidate gain rows G are
    already in memory, and adding a transmitter is an elementwise maximum
    against the combined coverage so far. That is what makes this a fair
    baseline -- its whole budget is the one solve per candidate.
    """
    G = np.stack([ev.gains_db([c], samples)[:, 0] for c in cand])
    out = {}
    chosen = []; combined = np.full(len(ev.rx), -300.0)
    for k in range(1, max(ev.cfg.k) + 1):
        best_j, best_val = None, None
        for j in range(len(cand)):
            if j in chosen:
                continue
            kp = ev.kpis(np.maximum(combined, G[j])[:, None])
            # Tuple comparison: coverage first, mean rate as the tie-break.
            key = (kp["coverage"], kp["mean_rate"])
            if best_val is None or key > best_val:
                best_j, best_val = j, key
        chosen.append(best_j); combined = np.maximum(combined, G[best_j])
        # Every prefix is recorded, so one pass answers all requested K.
        out[k] = [cand[j].tolist() for j in chosen]
    return G, out


def gradient(ev, start, steps, lr=0.1):
    """Adam ascent on the soft coverage, with optimize_tx.py's wall handling.

    Returns the BEST position seen, not the last: the objective is stochastic
    and Adam can overshoot, so the final step is not reliably the best one.
    """
    pos = [np.array(p, float) for p in start]
    m = [np.zeros(3) for _ in pos]; v = [np.zeros(3) for _ in pos]; b1, b2, eps = 0.9, 0.999, 1e-8
    best = (None, -1.0)
    for step in range(1, steps + 1):
        obj, p_db, grads = ev.soft_objective_and_grad(pos)
        if obj is None:
            break                                   # transmitter ended up inside geometry
        cov = float((p_db > ev.cfg.threshold_db).mean())
        if obj > best[1]:
            best = ([p.copy() for p in pos], obj)
        for k in range(len(pos)):
            g = grads[k]; m[k] = b1 * m[k] + (1 - b1) * g; v[k] = b2 * v[k] + (1 - b2) * g * g
            upd = lr * (m[k] / (1 - b1 ** step)) / (np.sqrt(v[k] / (1 - b2 ** step)) + eps)     # ascent
            upd[2] = 0.0                            # height is fixed in this benchmark
            # Full step, then slide along each wall, then progressively halve.
            for trial in [upd, upd * [0, 1, 1], upd * [1, 0, 1]] + [upd / 2 ** j for j in (1, 2, 3)]:
                cand = pos[k] + trial; cand[0] = np.clip(cand[0], *ev.cfg.x_range); cand[1] = np.clip(cand[1], *ev.cfg.y_range)
                if np.any(trial != 0) and ev.admissible(cand):
                    pos[k] = cand; break
    return best[0] if best[0] is not None else pos


def nelder_mead(ev, start, budget):
    """Derivative-free baseline on the same soft objective, same solve accounting.

    Optimises only x and y (z is pinned), so the search space matches the
    gradient method's. Every function evaluation costs one solve, which is how
    --budget is spent.
    """
    from scipy.optimize import minimize
    K = len(start)
    def f(x):
        pts = [np.array([x[2 * k], x[2 * k + 1], ev.cfg.tx_z]) for k in range(K)]
        # Inadmissible points return the worst possible value rather than
        # raising, so Nelder-Mead simply contracts away from them.
        if not all(ev.admissible(p) for p in pts):
            return 1.0
        obj, _, _ = ev.soft_objective_and_grad(pts)
        return 1.0 - (obj if obj is not None else 0.0)     # minimise 1 - coverage
    x0 = np.concatenate([[p[0], p[1]] for p in start])
    r = minimize(f, x0, method="Nelder-Mead", options={"maxfev": budget, "xatol": 0.05, "fatol": 1e-4, "initial_simplex": None})
    return [np.array([r.x[2 * k], r.x[2 * k + 1], ev.cfg.tx_z]) for k in range(K)]


def main():
    """Run every method at every K, re-score each placement, and write the table."""
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
    ap.add_argument("--resume", action="store_true", help="keep the methods already in --out and skip them")
    cfg = ap.parse_args()
    rng = np.random.default_rng(cfg.seed)
    ev = Evaluator(cfg)
    print(f"{len(ev.rx)} indoor receivers; SNR = gain + {ev.snr_offset_db:.1f} dB; threshold {cfg.threshold_db} dB")
    results = {"config": vars(cfg), "n_rx": int(len(ev.rx)), "snr_offset_db": ev.snr_offset_db, "methods": {}}
    if cfg.control and not results.get("control"):
        # Scored at the optimiser's sample count, not --eval-samples: this is a
        # sanity reference, not a benchmark row.
        kp = ev.kpis(ev.gains_db([np.array(cfg.control)], cfg.samples)); print(f"control {cfg.control} at {cfg.samples} samples: {kp}")
        results["control"] = {"position": cfg.control, "samples": cfg.samples, **kp}

    if cfg.resume and os.path.exists(cfg.out):
        prev = json.load(open(cfg.out)); results["methods"] = prev.get("methods", {}); results.setdefault("control", prev.get("control"))
        print(f"resuming: {sorted(results['methods'])} already in {cfg.out}")

    def done(name, K):
        """Has this (method, K) already been recorded? Drives --resume."""
        return f"{name}_K{K}" in results["methods"]

    def record(name, K, placement, solves, seconds):
        """Re-score a placement at --eval-samples and append it to the results.

        The re-scoring is the point: each optimiser reports its own noisy
        estimate, so placements are compared here on a common, higher-sample
        evaluation that none of them optimised against. The file is rewritten
        after every row, so a crashed run loses at most one method.
        """
        g = ev.gains_db(placement, cfg.eval_samples); kp = ev.kpis(g)
        results["methods"][f"{name}_K{K}"] = {"placement": [list(map(float, p)) for p in placement], "solves": int(solves), "seconds": float(seconds), **kp}
        os.makedirs(os.path.dirname(os.path.abspath(cfg.out)), exist_ok=True); json.dump(results, open(cfg.out, "w"), indent=1)
        print(f"  {name:12s} K={K}: coverage {kp['coverage']:.3f}  mean rate {kp['mean_rate']:.2f}  edge rate {kp['edge_rate_p5']:.2f} bit/s/Hz  "
              f"({solves} solves, {seconds:.0f} s)  at {np.round(placement, 2).tolist()}")

    # exhaustive + greedy
    cand = candidates(ev, cfg.cand_step); print(f"{len(cand)} admissible candidate cells at {cfg.cand_step} m")
    # The same solve count and elapsed time are charged to every K, because one
    # exhaustive sweep produced all of them.
    s0, t0 = ev.solves, time.time(); G, greedy = exhaustive(ev, cand, cfg.samples); ex_solves, ex_sec = ev.solves - s0, time.time() - t0
    for K in cfg.k:
        if not done("exhaustive" if K == 1 else "greedy", K):
            record("exhaustive" if K == 1 else "greedy", K, [np.array(p) for p in greedy[K]], ex_solves, ex_sec)
    best_cell = np.array(greedy[1][0])
    for K in cfg.k:
        # random search
        s0, t0 = ev.solves, time.time(); best = (None, None)
        # range(0) when already done, so --resume skips the work but the code path
        # stays identical.
        for _ in range(0 if done("random", K) else cfg.budget):
            pl = [ev.random_position(rng) for _ in range(K)]; kp = ev.kpis(ev.gains_db(pl, cfg.samples)); key = (kp["coverage"], kp["mean_rate"])
            if best[1] is None or key > best[1]:
                best = (pl, key)
        if best[0] is not None:
            record("random", K, best[0], ev.solves - s0, time.time() - t0)
        # Nelder-Mead from the exhaustive best cell(s)
        # Both derivative-free and gradient methods start from the same place, so
        # the comparison is about the search rule and not about initialisation.
        start = [np.array(p) for p in greedy[K]]
        if not done("nelder-mead", K):
            s0, t0 = ev.solves, time.time(); pl = nelder_mead(ev, start, cfg.budget); record("nelder-mead", K, pl, ev.solves - s0, time.time() - t0)
        # gradient from the exhaustive best cell(s), then random restarts; report the best and every start
        if not done("gradient", K):
            s0, t0 = ev.solves, time.time(); pl = gradient(ev, start, cfg.steps); record("gradient", K, pl, ev.solves - s0, time.time() - t0)
        for r in range(cfg.restarts):
            st = [ev.random_position(rng) for _ in range(K)]        # drawn even when skipped, so the rng stream is unchanged
            if not done(f"gradient_rand{r}", K):
                s0, t0 = ev.solves, time.time(); pl = gradient(ev, st, cfg.steps); record(f"gradient_rand{r}", K, pl, ev.solves - s0, time.time() - t0)
    results["total_solves"] = ev.solves; results["total_solve_seconds"] = ev.solve_seconds
    os.makedirs(os.path.dirname(os.path.abspath(cfg.out)), exist_ok=True)
    json.dump(results, open(cfg.out, "w"), indent=1)
    print(f"{ev.solves} solves, {ev.solve_seconds:.0f} s in the solver -> {cfg.out}")


if __name__ == "__main__":
    main()
