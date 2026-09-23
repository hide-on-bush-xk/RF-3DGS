"""The data side of the DLSS question: where a generated view's time goes, and what cheaper views cost.

Three measurements on a stratified set of receiver positions (farthest-point sampling over the dataset's
800 positions, so the lobby's extremes are in) plus one point outside the building (the degenerate
zero/few-path case):

  1. time    Sionna solve, and the spectrum synthesis at 300x200 and at 150x100 -- for the projection
             family (MULTI: equirect splat + four faces) and the beamformed one (MVDR, per face). If the
             synthesis is not per-pixel, rendering labels at low resolution cannot save anything.
  2. SR      the same paths rendered at 150x100 and bilinearly upsampled to 300x200, against the native
             300x200 rendering: per-pixel error of every decoded channel, and for MVDR the main peak --
             its direction and its power. This is the "does a low-resolution picture still hold the
             peak" question asked of the labels themselves, where the answer needs no learned model.
  3. rays    samples_per_src 1M / 300k / 100k / 30k against a 4M-sample reference at the same pose
             (reference seeds disjoint from the low ones): decoded azimuth / zenith / delay / power on
             the pixels the reference reaches, and the change of the reached set itself.

    set PYTHONUTF8=1
    python sionna_port/profile_generation.py --positions 10

Writes output/rrf/profile_generation.json. Timings need the card to itself.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import subprocess
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
from generate_dataset import (Config, VIEW_YAWS, build_scene, element_gain_fn, projection_equirect,  # noqa: E402
                              read_pose_groups)
from rf_spectra import ArrayGrid, compute_angle_matrices, equirect_to_perspective, mvdr_spectrum, paths_to_response  # noqa: E402

SCENE = os.path.join(REPO, "sionna_tutorial/RF-3DGS_Sionna_simulation_tutorial/NIST_lobby_v1.0/NIST_lobby_V1.1_sionna12.xml")
MULTI_DS = os.path.join(REPO, "RF-3DGS_dataset/regenerated/3dgs_MULTI_24ghz_tut_cs")


def gpu_util():
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader"],
                              capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception as exc:                      # noqa: BLE001
        return f"unavailable ({exc})"


def fps(points, k):
    """Farthest-point sampling: index 0 first, then the point farthest from everything chosen."""
    chosen = [0]; d = np.linalg.norm(points - points[0], axis=1)
    while len(chosen) < k:
        j = int(d.argmax()); chosen.append(j); d = np.minimum(d, np.linalg.norm(points - points[j], axis=1))
    return chosen


def timed(fn):
    torch.cuda.synchronize(); t0 = time.perf_counter(); out = fn(); torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0, out


def up2(x, size):
    """Bilinear upsampling of [C, h, w] (or [h, w]) to size (H, W), pixel centres aligned."""
    single = x.dim() == 2
    y = torch.nn.functional.interpolate((x[None, None] if single else x[None]), size=size, mode="bilinear",
                                        align_corners=False)[0]
    return y[0] if single else y


def multi_errors(pred, ref, floor_db=-150.0):
    """Decoded-channel errors of a MULTI face stack [4, 5, H, W] against a reference stack, on the
    pixels the reference reaches. Returns medians / P90s and the reached-set IoU."""
    hit_r = ref[:, 0] > floor_db; hit_p = pred[:, 0] > floor_db
    both = hit_r & hit_p
    out = {"hit_iou": float((hit_r & hit_p).sum() / max(int((hit_r | hit_p).sum()), 1)),
           "ref_hit_pixels": int(hit_r.sum()), "missed_fraction": float((hit_r & ~hit_p).sum() / max(int(hit_r.sum()), 1))}
    if not both.any():
        return out
    az_p = torch.rad2deg(torch.atan2(pred[:, 2], pred[:, 1])); az_r = torch.rad2deg(torch.atan2(ref[:, 2], ref[:, 1]))
    errs = {"power_db": (pred[:, 0] - ref[:, 0]).abs(),
            "az_deg": ((az_p - az_r + 180.0) % 360.0 - 180.0).abs(),
            "zen_deg": (pred[:, 3] - ref[:, 3]).abs() * 180.0,
            "delay_ns": (pred[:, 4] - ref[:, 4]).abs()}
    for k, e in errs.items():
        v = e[both].float()
        out[f"{k}_median"] = float(v.median()); out[f"{k}_p90"] = float(torch.quantile(v[:1_000_000], 0.9))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--positions", type=int, default=10)
    ap.add_argument("--samples", type=int, nargs="+", default=[1_000_000, 300_000, 100_000, 30_000])
    ap.add_argument("--reference-samples", type=int, default=4_000_000)
    ap.add_argument("--outdoor", type=float, nargs=3, default=[20.0, 12.0, -0.088],
                    help="a receiver outside the building: the degenerate zero/few-path case")
    ap.add_argument("--no-mvdr", action="store_true")
    cfg_a = ap.parse_args()
    device = "cuda"
    util_before = gpu_util()

    meta = json.load(open(os.path.join(MULTI_DS, "generation_meta.json")))
    groups = read_pose_groups(os.path.join(MULTI_DS, "sparse", "0", "images.txt"))
    pos = np.array([g[0] for g in groups])
    pick = fps(pos, cfg_a.positions)
    rx_list = [(int(i), pos[i].tolist()) for i in pick] + [(-1, list(cfg_a.outdoor))]
    print(f"{len(rx_list) - 1} positions by farthest-point sampling + 1 outdoor; GPU {util_before}")

    # the MULTI dataset's own settings (2.4 GHz, tutorial materials, depth 1, sigma 3)
    fields = {f.name for f in dataclasses.fields(Config)}
    mcfg = Config(**{k: v for k, v in meta.items() if k in fields})
    mcfg = dataclasses.replace(mcfg, scene_xml=SCENE)
    from sionna.rt import PathSolver, Receiver, Transmitter
    scene = build_scene(mcfg)
    scene.add(Transmitter(name="tx", position=list(mcfg.tx_loc)))
    solver = PathSolver()

    def solve(sc, rx, yaw, samples, seed):
        if "rx" in sc.receivers:
            sc.remove("rx")
        sc.add(Receiver(name="rx", position=list(rx), orientation=[yaw, 0.0, 0.0]))
        return solver(scene=sc, max_depth=1, samples_per_src=samples, los=True, specular_reflection=True,
                      diffuse_reflection=True, refraction=False, diffraction=False, synthetic_array=True, seed=seed)

    def multi_faces(paths, w, h):
        eq = projection_equirect(paths, "MULTI", device, mcfg.splat_sigma, mcfg.power_floor_db)
        return torch.stack([equirect_to_perspective(eq, w, h, 90.0, yaw_rad=y) for y in VIEW_YAWS])

    rows = []
    # generate_dataset.py solves a projection-family position once, with the receiver at the first yaw, and
    # cuts all four faces from that sphere; the same here
    y0 = VIEW_YAWS[0]
    # warm-up: the first solve compiles kernels
    for _ in range(2):
        p = solve(scene, rx_list[0][1], y0, 100_000, 1); multi_faces(p, 300, 200)
    # side check: does the receiver's orientation reach the projection spectra? The rx element pattern is
    # tr38901 (directional), and the one solve per position points the array at the first face only.
    orient = {}
    for yaw in (VIEW_YAWS[0], VIEW_YAWS[2]):
        f = multi_faces(solve(scene, rx_list[0][1], yaw, 1_000_000, 7), 300, 200)
        orient[f"{math.degrees(yaw):+.0f}"] = [float(torch.quantile(f[k, 0][f[k, 0] > -150], 0.99)) if (f[k, 0] > -150).any() else None
                                              for k in range(4)]
    print(f"orientation check, P99 power per face (yaw -90, 0, +90, 180) with the array at -90: {orient['-90']}, at +90: {orient['+90']}")
    for idx, rx in rx_list:
        t_ref_solve, ref_paths = timed(lambda: solve(scene, rx, y0, cfg_a.reference_samples, 100_000 + idx))
        ref = multi_faces(ref_paths, 300, 200)
        row = {"position_index": idx, "rx": rx, "reference_solve_ms": t_ref_solve,
               "reference_hit_pixels": int((ref[:, 0] > -150).sum())}
        for s in cfg_a.samples:
            t_solve, paths = timed(lambda: solve(scene, rx, y0, s, 42 + max(idx, 0)))
            t_300, f300 = timed(lambda: multi_faces(paths, 300, 200))
            t_150, f150 = timed(lambda: multi_faces(paths, 150, 100))
            a, _ = paths.cir(normalize_delays=False, out_type="torch")
            n_paths = int(a.shape[-2]) if a.numel() else 0
            r = {"solve_ms": t_solve, "multi_synth_ms_300x200": t_300, "multi_synth_ms_150x100": t_150, "paths": n_paths,
                 "vs_reference": multi_errors(f300, ref)}
            if s == cfg_a.samples[0]:
                # SR on the labels: the same paths at 150x100, bilinear to 300x200, against native 300x200
                r["sr_150_to_300_vs_native"] = multi_errors(up2(f150.reshape(-1, 100, 150), (200, 300)).reshape(4, 5, 200, 300), f300)
            row[str(s)] = r
        rows.append(row)
        s0 = row[str(cfg_a.samples[0])]
        print(f"pos {idx:4d}: ref hits {row['reference_hit_pixels']:7d} | " + " | ".join(
            f"{s//1000}k solve {row[str(s)]['solve_ms']:5.0f} ms az med {row[str(s)]['vs_reference'].get('az_deg_median', float('nan')):.2f} "
            f"miss {row[str(s)]['vs_reference']['missed_fraction']:.3f}" for s in cfg_a.samples))
        print(f"          synth 300x200 {s0['multi_synth_ms_300x200']:.1f} ms, 150x100 {s0['multi_synth_ms_150x100']:.1f} ms; "
              f"SR az med {s0['sr_150_to_300_vs_native'].get('az_deg_median', float('nan')):.3f} P90 {s0['sr_150_to_300_vs_native'].get('az_deg_p90', float('nan')):.2f}")

    mvdr_rows = []
    if not cfg_a.no_mvdr:
        # the beamformed family at the MVDR_100 dataset's settings (60 GHz, uniform scattering 0.7), per face
        vcfg = dataclasses.replace(Config(scene_xml=SCENE, rx_loc_file="", out_dir=""), frequency=60e9)
        vscene = build_scene(vcfg)
        vscene.add(Transmitter(name="tx", position=list(vcfg.tx_loc)))
        grids = {(300, 200): ArrayGrid.build(10, 300, 200, 90.0, element_gain_fn=element_gain_fn, device=device),
                 (150, 100): ArrayGrid.build(10, 150, 100, 90.0, element_gain_fn=element_gain_fn, device=device)}
        th, ph = compute_angle_matrices(300, 200, 90.0, device=device)
        for _ in range(2):
            p = solve(vscene, rx_list[0][1], 0.0, 100_000, 1); mvdr_spectrum(paths_to_response(p, 0.1, device=device), grids[(300, 200)])
        for idx, rx in rx_list[:-1]:
            for yaw in VIEW_YAWS:
                t_solve, paths = timed(lambda: solve(vscene, rx, yaw, 1_000_000, 42 + idx))
                try:
                    t_resp, resp = timed(lambda: paths_to_response(paths, 0.1, device=device))
                    t_m300, (_, db300) = timed(lambda: mvdr_spectrum(resp, grids[(300, 200)]))
                    t_m150, (_, db150) = timed(lambda: mvdr_spectrum(resp, grids[(150, 100)]))
                except ValueError as exc:
                    mvdr_rows.append({"position_index": idx, "yaw": yaw, "error": str(exc)}); continue
                up = up2(db150, (200, 300))
                k_n = int(db300.argmax()); k_u = int(up.argmax())
                # main-peak direction: the angle between the two argmax pixels' directions
                def unit(k):
                    t, p_ = th.reshape(-1)[k], ph.reshape(-1)[k]
                    return torch.stack([torch.sin(t) * torch.cos(p_), torch.sin(t) * torch.sin(p_), torch.cos(t)])
                ang = float(torch.rad2deg(torch.arccos(torch.clamp((unit(k_n) * unit(k_u)).sum(), -1, 1))))
                mvdr_rows.append({"position_index": idx, "yaw": yaw, "solve_ms": t_solve, "response_ms": t_resp,
                                  "mvdr_ms_300x200": t_m300, "mvdr_ms_150x100": t_m150,
                                  "sr_db_rmse": float(((up - db300) ** 2).mean().sqrt()),
                                  "sr_db_p90_abs": float(torch.quantile((up - db300).abs().reshape(-1), 0.9)),
                                  "sr_peak_angle_err_deg": ang,
                                  "sr_peak_power_err_db": float(up.max() - db300.max()),
                                  "native_peak_db": float(db300.max())})
            m = [r for r in mvdr_rows if r.get("position_index") == idx and "solve_ms" in r]
            if m:
                print(f"MVDR pos {idx:4d}: solve {np.median([r['solve_ms'] for r in m]):5.0f} ms, response "
                      f"{np.median([r['response_ms'] for r in m]):5.0f} ms, mvdr 300x200 {np.median([r['mvdr_ms_300x200'] for r in m]):5.1f} ms "
                      f"/ 150x100 {np.median([r['mvdr_ms_150x100'] for r in m]):5.1f} ms | SR peak angle "
                      f"{np.median([r['sr_peak_angle_err_deg'] for r in m]):.2f} deg, peak power {np.median([r['sr_peak_power_err_db'] for r in m]):+.2f} dB")

    def agg(key, s=None, sub="vs_reference"):
        vals = [r[str(s)][sub].get(key) for r in rows if r["position_index"] >= 0 and r[str(s)][sub].get(key) is not None]
        return float(np.median(vals)) if vals else None
    summary = {"multi": {str(s): {k: agg(k, s) for k in ("az_deg_median", "az_deg_p90", "zen_deg_median", "zen_deg_p90",
                                                             "delay_ns_median", "delay_ns_p90", "power_db_median", "power_db_p90",
                                                             "missed_fraction", "hit_iou")}
                         | {"solve_ms_median": float(np.median([r[str(s)]["solve_ms"] for r in rows if r["position_index"] >= 0]))}
                         for s in cfg_a.samples},
               "multi_sr_150_to_300": {k: agg(k, cfg_a.samples[0], "sr_150_to_300_vs_native")
                                       for k in ("az_deg_median", "az_deg_p90", "zen_deg_median", "zen_deg_p90",
                                                 "delay_ns_median", "delay_ns_p90", "power_db_median", "power_db_p90", "hit_iou")}}
    out = {"gpu": torch.cuda.get_device_name(0), "gpu_util_before": util_before, "gpu_util_after": gpu_util(),
           "positions": rx_list, "reference_samples": cfg_a.reference_samples, "samples": cfg_a.samples,
           "summary": summary, "orientation_check_p99_power_per_face": orient, "multi_rows": rows, "mvdr_rows": mvdr_rows,
           "note": "medians over positions of per-position medians/P90s; the outdoor row is excluded from the summary"}
    path = os.path.join(REPO, "output", "rrf", "profile_generation.json")
    json.dump(out, open(path, "w"), indent=1)
    print(json.dumps(summary, indent=1))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
