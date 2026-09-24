"""P1's first experiment (docs/tech_paths.md): is the MVDR of the angular power alone as good as the truth's?

The dataset's MVDR uses the coherent delay-tap covariance R = sum_l x_l x_l^H, x_l = sum_{p in tap l} a_p (a_p the
per-element response of path p, gain and element pattern included). A radiance field can only render power per
direction; the covariance it could feed an MVDR layer is the incoherent one, R_inc = sum_p a_p a_p^H (every
path's own term, the cross terms inside a tap dropped). This regenerates held-out views of the dataset with its
own configuration and seeds, computes both MVDRs from the same paths, and scores R_inc's spectrum against the
truth with mvdr_peaks' metrics. The coherent re-computation must reproduce the stored truth (sanity check).

Expectation, written before the run (P1 is now judged against the float64 coherent MVDR of the same paths, since
the stored complex64 truth turned out to be rounding noise in its weak directions): if R_inc keeps the main peak within 1 deg in >= 80 % of views and the power
at the true peak within 1 dB (median), an MVDR layer on rendered angular power (P1) can reach the truth's peaks;
if not, the tap coherence carries peak information and P1 would have to render per-tap quantities.

    python sionna_port/t6_incoherent_mvdr.py --truth ../RF-3DGS_dataset/regenerated/3dgs_MVDR_100_gpct --positions 20
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(REPO, "rrf_gsplat"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--truth", required=True); ap.add_argument("--positions", type=int, default=20)
    ap.add_argument("--protocol", default=None, help="the held-out views are the protocol's --eval-set, not test_index.txt")
    ap.add_argument("--eval-set", default="val")
    ap.add_argument("--final-test", action="store_true", help="allow a sealed test set (logged)")
    a = ap.parse_args()
    import generate_dataset as G
    from rf_spectra import ArrayGrid, merge_paths_to_time_grid, mvdr_spectrum
    from mvdr_peaks import pixel_dirs, local_maxima, top_peaks, angle
    from sionna.rt import PathSolver, Receiver, Transmitter
    meta = json.load(open(os.path.join(a.truth, "generation_meta.json")))
    fields = {f.name for f in dataclasses.fields(G.Config)}
    cfg = G.Config(**{k: v for k, v in meta.items() if k in fields})
    cwd = os.getcwd(); os.chdir(HERE)                         # the meta's scene path is relative to sionna_port
    scene = G.build_scene(cfg); os.chdir(cwd)
    solver = PathSolver()
    dev = "cuda"
    grid = ArrayGrid.build(cfg.M, cfg.width, cfg.height, cfg.fov_deg, element_gain_fn=G.element_gain_fn, device=dev)
    scene.add(Transmitter(name="tx", position=list(cfg.tx_loc)))
    groups = G.read_pose_groups(os.path.join(a.truth, "sparse", "0", "images.txt"))
    names = []
    for line in open(os.path.join(a.truth, "sparse", "0", "images.txt")):
        p = line.split()
        if len(p) >= 10 and p[9].lower().endswith(".png"):
            names.append(p[9][:-4])
    if a.protocol:
        import protocol as PR
        PR.check_dataset(a.protocol, a.truth)
        test = set(PR.eval_names(a.protocol, a.eval_set, allow_test=a.final_test))
    else:
        test = set(l.strip() for l in open(os.path.join(a.truth, "test_index.txt")) if l.strip())
    view_pos = []                                             # (position index, yaw, name) in dataset order
    k = 0
    for i, (rx, yaws) in enumerate(groups):
        for yaw in yaws:
            view_pos.append((i, yaw, names[k])); k += 1
    test_pos = sorted({i for i, _, n in view_pos if n in test})
    # spread along the route (the route's order is the index's; both end points included; 0 = every one) ...
    pick = (list(test_pos) if a.positions <= 0 else
            [test_pos[j] for j in np.linspace(0, len(test_pos) - 1, a.positions).round().astype(int)])
    # ... plus the degenerate cases: any held-out position with an empty face (no path reaches it; the truth holds
    # EMPTY_VIEW_DB there) and the held-out position whose strongest face is weakest (the most shadowed one)
    peak = {}
    for i, _, n in view_pos:
        if i in test_pos:
            peak[i] = max(peak.get(i, -1e9), float(np.load(os.path.join(a.truth, "spectra_float", n + ".npy")).max()))
    empty = [i for i in test_pos if peak[i] <= G.EMPTY_VIEW_DB + 1]
    weakest = min(test_pos, key=lambda i: peak[i])
    pick = sorted(set(pick) | set(empty[:2]) | {weakest})
    print(f"{len(pick)} positions: {pick}; empty-face positions among the held-out: {len(empty)}; "
          f"weakest held-out position {weakest} (max {peak[weakest]:.1f} dB)")
    dirs = pixel_dirs(a.truth); flat = dirs.reshape(-1, 3)
    rows, stats, covs = [], {}, {}
    # the float64 recomputation of every processed view, in a run's layout so mvdr_peaks.py / t4_score.py can score
    # the stored truth's rounding noise like any predictor
    tag = ("" if not a.protocol else f"_{a.eval_set}")                  # protocol dumps never overwrite the old ones
    truth64_dir = os.path.join(REPO, "output", "rrf", "t6_truth64" + tag + ("" if a.positions <= 0 else f"_{a.positions}pos"))
    for i in pick:
        rx, yaws = groups[i]
        for yaw in yaws:
            name = next(n for ii, yy, n in view_pos if ii == i and yy == yaw)
            scene.remove("rx") if "rx" in scene.receivers else None
            scene.add(Receiver(name="rx", position=list(rx), orientation=[yaw, 0.0, 0.0]))
            paths = G.solve_paths(solver, scene, cfg, view_index=i)
            amp, tau = paths.cir(normalize_delays=False, out_type="torch")
            A = amp[0, :, 0, 0, :, 0].to(dev)
            tau = (tau[0, 0, 0, 0, :] if tau.dim() == 5 else tau.reshape(-1)).to(dev)
            keep = torch.isfinite(tau) & (tau >= 0); A, tau = A[:, keep], tau[keep]
            truth = np.load(os.path.join(a.truth, "spectra_float", name + ".npy")).astype(np.float64)
            if A.shape[1] == 0:
                # no path: the generator wrote EMPTY_VIEW_DB; there is no covariance to compare
                rows.append({"view": name, "position": i, "paths": 0,
                             "truth_is_empty": bool(truth.max() <= G.EMPTY_VIEW_DB + 1)})
                print(f"{name} pos {i}: no paths (truth empty: {rows[-1]['truth_is_empty']})")
                continue
            # everything in float64: the unloaded tap covariance has cond ~1e11, and the dataset's complex64 MVDR is
            # rounding noise in its weak directions (not reproducible between two solves of the same view)
            A = A.to(torch.complex128)
            tau_ns = tau.double() / 1e-9
            tg = torch.arange(0, max(int(np.ceil(float(tau_ns.max()))), 1), cfg.time_interval_ns, device=dev, dtype=tau_ns.dtype)
            X = merge_paths_to_time_grid(A, tau_ns, tg)
            coh = mvdr_spectrum(X, grid, cfg.diagonal_loading)[1].cpu().numpy().astype(np.float64)
            os.makedirs(os.path.join(truth64_dir, "renders"), exist_ok=True)
            np.save(os.path.join(truth64_dir, "renders", name + ".npy"), coh.astype(np.float32))
            # T4 levels 2 and 3 from the same solve: the channel's large-scale parameters (the estimators
            # t4_baselines.py applies to InH) and the tap covariance, on which a steered beam's gain is evaluated
            stats[name] = channel_stats(A, tau_ns, cfg.time_interval_ns)
            covs[name] = (X @ X.conj().T).cpu().numpy().astype(np.complex64)
            # incoherent: every path its own snapshot, R = A A^H, i.e. mvdr_spectrum on the per-path "response";
            # with fewer paths than elements R is singular and gets the smallest loading that inverts it
            loading = 0.0 if A.shape[1] >= A.shape[0] else 1e-6
            while True:
                try:
                    inc = mvdr_spectrum(A, grid, loading)[1].cpu().numpy().astype(np.float64)
                    break
                except ValueError:
                    loading = 1e-6 if loading == 0.0 else loading * 10
            r = {"view": name, "position": i, "paths": int(A.shape[1]), "incoherent_loading": loading}
            # (1) the label's own noise: the stored (complex64) truth against the float64 recomputation
            r["truth32_vs_coherent64_max_abs_db"] = float(np.abs(coh - truth).max())
            r["truth32_vs_coherent64_median_abs_db"] = float(np.median(np.abs(coh - truth)))
            within20 = coh >= coh.max() - 20
            r["truth32_vs_coherent64_max_abs_db_within_20db"] = float(np.abs(coh - truth)[within20].max())
            r["truth32_peak_angle_deg"] = angle(dirs, int(coh.argmax()), int(truth.argmax()))
            # (2) P1: the incoherent MVDR against the coherent one from the same paths (both float64), and (3) against
            # the stored truth. The incoherent spectrum's absolute level differs (no intra-tap sums): levels are
            # compared after aligning the maxima; the raw offset is kept
            for ref_name, ref in (("coh64", coh), ("truth", truth)):
                kr, ki = int(ref.argmax()), int(inc.argmax())
                kept = top_peaks(ref, dirs, 3); pm = np.flatnonzero(local_maxima(inc))
                det = [np.degrees(np.arccos(np.clip(flat[pm] @ flat[c], -1, 1))).min() <= 1.5 for c in kept] if len(pm) else [False] * len(kept)
                off = float(ref.max() - inc.max())
                r[f"inc_vs_{ref_name}_peak_angle_deg"] = angle(dirs, kr, ki)
                r[f"inc_vs_{ref_name}_offset_db"] = off
                r[f"inc_vs_{ref_name}_at_peak_db_aligned"] = float(inc.flat[kr] + off - ref.flat[kr])
                r[f"inc_vs_{ref_name}_top3_detected"] = float(np.mean(det)) if det else None
                r[f"inc_vs_{ref_name}_rmse_db_aligned"] = float(np.sqrt(np.mean((inc + off - ref) ** 2)))
            rows.append(r)
            print(f"{name} pos {i}: paths {r['paths']}, stored truth vs float64: max |d| {r['truth32_vs_coherent64_max_abs_db']:.2f} "
                  f"(within 20 dB of the peak {r['truth32_vs_coherent64_max_abs_db_within_20db']:.2f}) dB, peak {r['truth32_peak_angle_deg']:.2f} deg | "
                  f"incoherent vs coherent64: peak {r['inc_vs_coh64_peak_angle_deg']:.2f} deg, at peak "
                  f"{r['inc_vs_coh64_at_peak_db_aligned']:+.2f} dB (aligned) | vs stored truth: peak {r['inc_vs_truth_peak_angle_deg']:.2f} deg")
    empty_rows = [r for r in rows if r["paths"] == 0]
    rows_all, rows = rows, [r for r in rows if r["paths"] > 0]
    col = lambda k: np.array([r[k] for r in rows if r[k] is not None], dtype=np.float64)   # noqa: E731
    summ = {"views": len(rows), "views_without_paths": len(empty_rows),
            "views_loaded": sum(r["incoherent_loading"] > 0 for r in rows),
            "label_noise": {"max_abs_db_max": float(col("truth32_vs_coherent64_max_abs_db").max()),
                            "median_abs_db_median": float(np.median(col("truth32_vs_coherent64_median_abs_db"))),
                            "max_abs_db_within_20db_of_peak_max": float(col("truth32_vs_coherent64_max_abs_db_within_20db").max()),
                            "peak_angle_within_1deg": float((col("truth32_peak_angle_deg") <= 1).mean()),
                            "peak_angle_median": float(np.median(col("truth32_peak_angle_deg")))}}
    for ref_name in ("coh64", "truth"):
        ang = col(f"inc_vs_{ref_name}_peak_angle_deg")
        summ[f"incoherent_vs_{ref_name}"] = {"within_1deg": float((ang <= 1).mean()), "angle_median": float(np.median(ang)),
                                            "at_peak_median_db_aligned": float(np.median(col(f"inc_vs_{ref_name}_at_peak_db_aligned"))),
                                            "top3_detected": float(np.mean(col(f"inc_vs_{ref_name}_top3_detected"))),
                                            "rmse_db_aligned_median": float(np.median(col(f"inc_vs_{ref_name}_rmse_db_aligned")))}
    print(json.dumps(summ, indent=1))
    sfx = tag + ("_all" if a.positions <= 0 else f"_{len(pick)}pos")
    rr = os.path.join(REPO, "output", "rrf")
    json.dump({"summary": summ, "views": rows_all}, open(os.path.join(rr, f"t6_incoherent_mvdr{sfx}.json"), "w"), indent=1)
    json.dump(stats, open(os.path.join(rr, f"t4_rt_stats{sfx}.json"), "w"), indent=1)
    names_c = sorted(covs)
    np.savez_compressed(os.path.join(rr, f"t4_rt_cov{sfx}.npz"), names=np.array(names_c), R=np.stack([covs[n] for n in names_c]))


def channel_stats(A, tau_ns, dt):
    """Large-scale parameters of one view's channel from its per-element path coefficients A [M^2, P] and delays
    (ns): path gain (element patterns included), RMS delay spread, and the power of the first delay tap (the
    earliest path + dt) over the rest -- a K-factor proxy that needs no path labels, so that the same estimator
    applies to InH's cluster coefficients."""
    p = (A.abs() ** 2).mean(0).double(); t = tau_ns.double()
    tot = float(p.sum())
    m1 = float((p * t).sum()) / tot; m2 = float((p * t * t).sum()) / tot
    first = t <= float(t.min()) + dt
    pf, pr = float(p[first].sum()), float(p[~first].sum())
    return {"paths": int(p.numel()), "gain_db": 10 * np.log10(max(tot, 1e-300)),
            "ds_ns": float(np.sqrt(max(m2 - m1 * m1, 0.0))),
            "first_tap_ratio_db": 10 * np.log10(max(pf, 1e-300) / max(pr, 1e-300)),
            "first_delay_ns": float(t.min())}


if __name__ == "__main__":
    main()
