"""Transmitter placement as DLSS-style guided upsampling of the objective map.

DLSS renders the expensive shading at low resolution and reconstructs the full-resolution frame with help
from buffers that are cheap at full resolution (depth, normals, motion). Real-time denoisers add one more
trick: divide out the high-frequency factor that is known exactly (albedo), filter the smooth remainder,
multiply back ("demodulation"). The placement problem has the same structure:

  expensive "shading"   one full Sionna solve (LoS + specular + diffuse) per candidate transmitter cell
  cheap "G-buffer"      the line-of-sight term alone (a visibility test and free-space loss per Rx) --
                        the part of the gain that flips when a wall cuts the ray, i.e. the high frequency
  demodulation          P_full = P_los + P_mp per (Tx, Rx); P_mp, the multipath remainder, is smooth in the
                        Tx position, so it is solved on a coarse grid and interpolated to the fine one,
                        and the exact fine-grid P_los is added back

Candidates are then ranked on the reconstructed fine map and only the top few are verified with full solves.
Compared, all on the same fine grid and re-scored with the benchmark's common evaluator (200k samples):

  fine_exhaustive   every fine cell solved (the reference, and its cost)
  coarse_1m / 2m    every coarse cell solved, best coarse cell (the benchmark's exhaustive at 1 m)
  guided_{c}        coarse solves + fine LoS + residual interpolation, top-k verified
  naive_{c}         the same interpolation of P_full itself, no LoS guide (plain upsampling), top-k verified
  los_only          the LoS map alone ranks, top-k verified (the guide without any shading)

Solves are charged as in benchmark_placement.py (one per full evaluation of the receiver grid); LoS-only
solves are counted and timed separately, since they are the price of the guide.

    set PYTHONUTF8=1
    cd tx_planning && python guided_placement.py --bench ../output/tx_planning/benchmark_lobby.json --fine-step 0.5
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benchmark_placement import Evaluator, candidates  # noqa: E402


def los_power(ev, p):
    """Linear LoS power [n_rx] for one transmitter: the same solver, reflections and scattering off."""
    import drjit as dr
    ev.set_txs([p])
    t0 = time.time()
    paths = ev.solver(scene=ev.scene, max_depth=0, max_num_paths_per_src=10_000_000, samples_per_src=1000, los=True,
                      specular_reflection=False, diffuse_reflection=False, refraction=False, synthetic_array=True, seed=42)
    a_re, a_im = paths.a
    if a_re.shape[-1] == 0:
        out = np.zeros(len(ev.rx))
    else:
        q = dr.square(a_re) + dr.square(a_im)
        out = np.asarray(dr.sum(dr.sum(dr.sum(q, axis=4), axis=3), axis=1)).reshape(len(ev.rx))
    ev.los_seconds += time.time() - t0; ev.los_solves += 1
    if ev.los_solves % ev.cfg.flush_every == 0:
        import gc
        gc.collect(); dr.flush_malloc_cache()
    return out


def los_power_analytic(ev, tx_points):
    """Linear LoS power [n_tx, n_rx] for every (candidate, receiver) pair in ONE ray-test launch.

    Both ends of the planning scene are single isotropic V-polarised elements (scene_common.load_radio_scene),
    so a visible pair's LoS gain is the Friis free-space factor (lambda / 4 pi d)^2 and an occluded pair's is 0.
    Visibility is Mitsuba's ray_test from the transmitter towards the receiver, stopped just short of it.
    This is the G-buffer: exact, and a single launch for the whole fine grid instead of one solve per cell.
    """
    import drjit as dr
    import mitsuba as mi
    t0 = time.time()
    tx = np.repeat(np.asarray(tx_points, float), len(ev.rx), axis=0)          # [n_tx * n_rx, 3]
    rx = np.tile(np.asarray(ev.rx, float), (len(tx_points), 1))
    v = rx - tx; d = np.linalg.norm(v, axis=1); u = v / d[:, None]
    ray = mi.Ray3f(mi.Point3f(*[mi.Float(tx[:, k].astype(np.float32)) for k in range(3)]),
                   mi.Vector3f(*[mi.Float(u[:, k].astype(np.float32)) for k in range(3)]))
    ray.maxt = mi.Float((d * (1.0 - 1e-4)).astype(np.float32))
    blocked = np.asarray(ev.scene.mi_scene.ray_test(ray)).reshape(-1)
    lam = 299_792_458.0 / float(np.asarray(ev.scene.frequency).reshape(-1)[0])
    p = np.where(blocked, 0.0, (lam / (4.0 * np.pi * d)) ** 2).reshape(len(tx_points), len(ev.rx))
    dr.sync_thread()
    ev.analytic_seconds = getattr(ev, "analytic_seconds", 0.0) + time.time() - t0
    return p


def full_power(ev, p, samples):
    """Linear full power [n_rx] for one transmitter (one charged solve)."""
    g_db = ev.gains_db([p], samples)[:, 0]
    return np.maximum(10.0 ** (g_db / 10.0) - ev.noise_lin, 0.0)


def idw(coarse_xy, fine_xy, values, radius):
    """Inverse-distance-squared interpolation of values [n_c, n_rx] onto fine_xy [n_f, 2].

    Coarse cells within `radius` contribute (the four lattice neighbours at radius 1.5 x step); a fine
    cell that coincides with a coarse one gets that cell's value exactly; a fine cell with no coarse cell
    in range (a corner of the L-shaped admissible set) takes its nearest one.
    """
    out = np.empty((len(fine_xy), values.shape[1]))
    for f, q in enumerate(fine_xy):
        d = np.linalg.norm(coarse_xy - q, axis=1)
        j0 = int(d.argmin())
        if d[j0] < 1e-6:
            out[f] = values[j0]; continue
        near = np.where(d <= radius)[0]
        if len(near) == 0:
            out[f] = values[j0]; continue
        w = 1.0 / d[near] ** 2
        out[f] = (w[:, None] * values[near]).sum(0) / w.sum()
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bench", required=True, help="benchmark_placement.py JSON whose config (scene, grids, KPI) to reuse")
    ap.add_argument("--fine-step", type=float, default=0.5)
    ap.add_argument("--coarse-steps", type=float, nargs="+", default=[1.0, 2.0])
    ap.add_argument("--verify", type=int, default=5, help="top-k of the reconstructed map verified with full solves")
    ap.add_argument("--k", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--guide", choices=["analytic", "sionna"], default="analytic",
                    help="the fine-grid LoS term: one ray-test launch + Friis (analytic), or a LoS-only Sionna solve per cell")
    ap.add_argument("--flush-every", type=int, default=5,
                    help="free Dr.Jit's memory pool every N solves (the corridor's 1020-cell fine sweep ran out of memory at 25)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    bench = json.load(open(a.bench))
    cfg = SimpleNamespace(**bench["config"])
    cfg.flush_every = a.flush_every
    ev = Evaluator(cfg)
    ev.los_seconds, ev.los_solves = 0.0, 0
    thr_lin = 10.0 ** (cfg.threshold_db / 10.0)
    print(f"{os.path.basename(cfg.scene_xml)}: {len(ev.rx)} receivers, threshold {cfg.threshold_db} dB, {cfg.samples} samples/solve")

    fine = candidates(ev, a.fine_step)
    print(f"fine grid {a.fine_step} m: {len(fine)} admissible cells")

    # -- the reference: every fine cell, full solve and LoS-only ------------------------------------------
    s0, t0 = ev.solves, time.time()
    P_full = np.stack([full_power(ev, p, cfg.samples) for p in fine])          # [n_f, n_rx] linear
    fine_solve_seconds = ev.solve_seconds
    t_los0 = ev.los_seconds
    P_los = np.stack([los_power(ev, p) for p in fine])
    los_seconds_fine = ev.los_seconds - t_los0
    per_full_ms = 1000 * fine_solve_seconds / len(fine); per_los_ms = 1000 * los_seconds_fine / len(fine)
    print(f"full solve {per_full_ms:.1f} ms/cell, LoS-only {per_los_ms:.1f} ms/cell (x{per_full_ms / max(per_los_ms, 1e-9):.1f})")
    # the analytic G-buffer against Sionna's own LoS term (the control that licenses using it)
    los_power_analytic(ev, fine[:2])                                           # warm-up: the first launch compiles
    ev.analytic_seconds = 0.0
    P_los_a = los_power_analytic(ev, fine)
    per_analytic_ms = 1000 * ev.analytic_seconds / len(fine)
    vis_s, vis_a = P_los > 0, P_los_a > 0
    both = vis_s & vis_a
    los_db_diff = np.abs(10 * np.log10(P_los_a[both]) - 10 * np.log10(P_los[both])) if both.any() else np.zeros(1)
    los_check = {"visibility_agreement": float((vis_s == vis_a).mean()), "visible_pairs": int(both.sum()),
                 "db_diff_max": float(los_db_diff.max()), "db_diff_median": float(np.median(los_db_diff)),
                 "analytic_ms_per_cell": per_analytic_ms}
    print(f"analytic LoS: {per_analytic_ms:.3f} ms/cell (x{per_full_ms / max(per_analytic_ms, 1e-9):.0f} cheaper than a full solve); "
          f"visibility agrees with Sionna's LoS on {100 * los_check['visibility_agreement']:.3f}% of pairs, "
          f"|dB| max {los_check['db_diff_max']:.4f} median {los_check['db_diff_median']:.5f} on {both.sum()} visible pairs")
    if a.guide == "analytic":
        P_los = P_los_a
    # what the fine-grid guide costs in total: one launch, or one LoS-only solve per cell
    guide_ms = 1000 * (ev.analytic_seconds if a.guide == "analytic" else los_seconds_fine)
    # control: the LoS path is deterministic and the gains sum |a|^2 over paths, so P_full - P_los >= 0 exactly
    resid = P_full - P_los
    neg = resid < -1e-6 * np.maximum(P_full, 1e-30)
    print(f"control P_full >= P_los: {int(neg.sum())} of {neg.size} (Tx, Rx) pairs violate it "
          f"(worst {float((resid / np.maximum(P_full, 1e-30)).min()):.2e} relative)")
    P_mp = np.maximum(resid, 0.0)

    def coverage(P):          # [n, n_rx] linear -> [n]
        return (P > thr_lin).mean(axis=1)

    cov_true = coverage(P_full)
    j_true = int(cov_true.argmax())
    print(f"fine truth: best coverage {cov_true[j_true]:.3f} at {fine[j_true][:2]} (20k-sample estimate)")

    rows = {}

    def rescore(name, placement, solves, extra):
        # free Dr.Jit's pool first: a K = 3 solve at the evaluation budget takes the symbolic loop and most of
        # the card (benchmark round 23b), and the corridor ran out of memory there after the fine sweep
        import gc
        import drjit as dr
        gc.collect(); dr.flush_malloc_cache()
        g = ev.gains_db([np.array(p) for p in placement], cfg.eval_samples)
        kp = ev.kpis(g)
        rows[name] = {"placement": [list(map(float, p)) for p in placement], "solves": int(solves), **extra, **kp}
        print(f"  {name:22s} coverage {kp['coverage']:.3f} mean rate {kp['mean_rate']:.2f} | {solves:4d} solves "
              + " ".join(f"{k} {v}" for k, v in extra.items() if k in ("guide_ms", "map_mae", "rank_corr")))

    def greedy(P, K):
        chosen, comb = [], np.zeros(P.shape[1])
        for _ in range(K):
            best = None
            for j in range(len(P)):
                if j in chosen:
                    continue
                c = np.maximum(comb, P[j]); key = ((c > thr_lin).mean(), np.log2(1 + c.sum()))
                if best is None or key > best[0]:
                    best = (key, j)
            chosen.append(best[1]); comb = np.maximum(comb, P[best[1]])
        return chosen

    def rank_corr(x, y):
        rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
        return float(np.corrcoef(rx, ry)[0, 1])

    for K in a.k:
        rescore(f"fine_exhaustive_K{K}", [fine[j] for j in greedy(P_full, K)], len(fine), {})

    fine_xy = fine[:, :2]
    for step in a.coarse_steps:
        # coarse cells are fine cells (the grids share their origin), so their full solves are rows of P_full:
        # the same numbers a separate coarse sweep would produce (fixed seed), without solving them twice
        def on_lattice(p):
            return all(abs(v / step - round(v / step)) < 1e-6 for v in (p[0] - cfg.x_range[0], p[1] - cfg.y_range[0]))
        idx_c = [j for j, p in enumerate(fine) if on_lattice(p)]
        n_c = len(idx_c)
        C_xy = fine_xy[idx_c]
        tag = f"{step:g}m"
        for K in a.k:
            rescore(f"coarse_{tag}_K{K}", [fine[idx_c[j]] for j in greedy(P_full[idx_c], K)], n_c, {})
        db = lambda P: 10.0 * np.log10(P + ev.noise_lin)                          # noqa: E731
        undb = lambda D: np.maximum(10.0 ** (D / 10.0) - ev.noise_lin, 0.0)       # noqa: E731
        recon = {"guided": P_los + undb(idw(C_xy, fine_xy, db(P_mp[idx_c]), 1.5 * step)),
                 "naive": undb(idw(C_xy, fine_xy, db(P_full[idx_c]), 1.5 * step))}
        # analytic control: at a coarse cell the reconstruction is the measurement itself
        err_at_c = float(np.abs(db(recon["guided"][idx_c]) - db(P_full[idx_c])).max())
        print(f"coarse {tag}: {n_c} cells; guided reconstruction at the coarse cells differs from the solve by {err_at_c:.2e} dB")
        for name, P_hat in recon.items():
            cov_hat = coverage(P_hat)
            off = np.setdiff1d(np.arange(len(fine)), idx_c)                      # cells the method never solved
            extra = {"map_mae": round(float(np.abs(cov_hat[off] - cov_true[off]).mean()), 4),
                     "rank_corr": round(rank_corr(cov_hat[off], cov_true[off]), 3),
                     "gain_db_mae_offgrid": round(float(np.abs(db(P_hat[off]) - db(P_full[off])).mean()), 2),
                     "guide_ms": round(guide_ms, 1) if name == "guided" else 0.0}
            top = [j for j in np.argsort(-cov_hat) if j not in idx_c][:a.verify]
            # verification: the top-k unsolved cells get their full solve (their rows of P_full), the rest keep
            # the reconstruction; the budget is n_c + k
            P_ver = P_hat.copy(); P_ver[idx_c] = P_full[idx_c]; P_ver[top] = P_full[top]
            for K in a.k:
                rescore(f"{name}_{tag}_K{K}", [fine[j] for j in greedy(P_ver, K)], n_c + len(top), extra)
    cov_los = coverage(P_los)
    top = list(np.argsort(-cov_los)[:a.verify])
    P_ver = P_los.copy(); P_ver[top] = P_full[top]
    for K in a.k:
        rescore(f"los_only_K{K}", [fine[j] for j in greedy(P_ver, K)], len(top),
                {"map_mae": round(float(np.abs(cov_los - cov_true).mean()), 4), "rank_corr": round(rank_corr(cov_los, cov_true), 3),
                 "guide_ms": round(guide_ms, 1)})

    out = a.out or os.path.join(os.path.dirname(os.path.abspath(a.bench)),
                                os.path.basename(a.bench).replace("benchmark_", "guided_"))
    json.dump({"bench": a.bench, "fine_step": a.fine_step, "coarse_steps": a.coarse_steps, "verify": a.verify,
               "n_fine": len(fine), "n_rx": len(ev.rx), "full_solve_ms": per_full_ms, "los_solve_ms": per_los_ms,
               "guide": a.guide, "analytic_los_check": los_check,
               "control_negative_pairs": int(neg.sum()), "fine_truth_best_cov_20k": float(cov_true[j_true]),
               "rows": rows}, open(out, "w"), indent=1)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
