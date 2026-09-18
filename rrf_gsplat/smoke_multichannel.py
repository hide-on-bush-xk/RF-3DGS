"""Smoke test for the multi-channel spectrum pipeline (CLAUDE.md: smoke first).

Pass criteria, written before running:

  A. analytic control -- one synthetic path (amp 1, delay 50 ns, AoA
     theta 80 / phi 30 deg, AoD theta 100 / phi -120 deg), sigma 3 px:
     at the AoA pixel the channels decode to |az + 120| <= 0.34 deg (one
     equirect pixel), |zen - 100| <= 0.34 deg, |delay - 50| <= 1e-3 ns,
     power finite; the four pinhole faces resample to [5, 200, 300].
  B. the tail mechanism, reported not judged -- two equal paths with
     opposite azimuth in one pixel: the circular mean is undefined; the
     decoded azimuth may be anything. Smoke only verifies no exception.
  C. degenerate input -- zero paths: [5, 540, 1080] with power == -200 dB
     everywhere and zeros elsewhere, no exception; resampling still works.
  D. structured sample of the real pipeline (needs the scene): the route's
     first, middle and last positions (endpoints included) x all 4 yaws,
     plus one receiver outside the building (12, 0, -0.088). Expected: every
     indoor view [5, 200, 300] with power inside [-200, -20] dB and a hit
     fraction in [0.2, 0.98]; the outdoor receiver yields zero paths and the
     channels are all floor / zero; all five channels present; the whole
     sample under 90 s on the RTX 3060.
  E. known-failure control -- the tutorial's angle x amplitude RGB encoding,
     decoded from the trained AOD3 model's renders against the MULTI truth:
     azimuth median in [20, 35] deg (measured 27 deg on the full run).
     Requires output/rrf/m_aod3_24_tut and the MULTI dataset; skipped with a
     notice if absent (then the smoke is incomplete, not passed).

What this smoke cannot answer: tail behaviour (P90 / RMSE), numerical
stability at scale, memory and throughput.

Record of the first run (2026-09-18, criteria above, unchanged): A, C, E pass;
D FAILS its stated ranges while the pipeline behaves correctly -- the ranges
were mis-specified. (1) power minimum: pixels a kernel tail barely touches
carry 10 log10 of a tiny power, down to -273 dB, below the -200 dB floor
the criterion assumed was the minimum; (2) hit fraction: faces looking at a
wall reach 4-14 %, below the 20 % assumed. Proposed v2 criteria, not
applied here: power in [-300, -20] dB, indoor hit fraction in [0.02, 0.98].
Per CLAUDE.md the v1 result stands as a fail until v2 is agreed and re-run.

    PYTHONUTF8=1 python rrf_gsplat/smoke_multichannel.py
"""

from __future__ import annotations

import json
import math
import os
import sys
import time

import numpy as np
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO, "sionna_port"))
sys.path.insert(0, os.path.join(REPO, "tx_planning"))

PASS, FAIL, SKIP = "PASS", "FAIL", "SKIP"
results = []


def check(name, ok, detail):
    results.append((name, PASS if ok else FAIL, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")


def skip(name, detail):
    results.append((name, SKIP, detail))
    print(f"[SKIP] {name}: {detail}")


def decode(ch):
    """[5,...] channels -> az deg, zen deg, delay ns, power dB."""
    az = torch.rad2deg(torch.atan2(ch[2], ch[1]))
    return az, ch[3] * 180.0, ch[4], ch[0]


def main():
    from rf_spectra import equirect_to_perspective, multichannel_from_arrays
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t_start = time.time()

    # -- A. one path ------------------------------------------------------
    r = lambda d: torch.tensor([math.radians(d)], device=dev)
    ch = multichannel_from_arrays(torch.tensor([1.0], device=dev), torch.tensor([50e-9], device=dev),
                                  r(80), r(30), r(100), r(-120), scale=3, sigma=3.0)
    ti, pi = int(round(80 * 3)), int(round((-30 + 180) * 3))         # equirect_splat's index convention
    az, zen, dl, pw = decode(ch[:, ti, pi])
    ok = (abs(float(az) + 120) <= 0.34 and abs(float(zen) - 100) <= 0.34 and abs(float(dl) - 50) <= 1e-3
          and math.isfinite(float(pw)) and tuple(ch.shape) == (5, 540, 1080))
    check("A single path decodes to itself", ok,
          f"az {float(az):.3f} (want -120), zen {float(zen):.3f} (want 100), delay {float(dl):.4f} ns (want 50), power {float(pw):.1f} dB, shape {tuple(ch.shape)}")
    faces = [equirect_to_perspective(ch, 300, 200, 90.0, yaw_rad=y) for y in (-math.pi / 2, 0.0, math.pi / 2, math.pi)]
    check("A four faces resample", all(tuple(f.shape) == (5, 200, 300) for f in faces), f"{[tuple(f.shape) for f in faces]}")

    # -- B. the tail mechanism (report only) ------------------------------
    ch2 = multichannel_from_arrays(torch.tensor([1.0, 1.0], device=dev), torch.tensor([50e-9, 60e-9], device=dev),
                                   r(80).repeat(2), r(30).repeat(2), r(100).repeat(2),
                                   torch.tensor([math.radians(10), math.radians(-170)], device=dev), scale=3, sigma=3.0)
    az2, _, _, _ = decode(ch2[:, ti, pi])
    print(f"[INFO] B two opposite paths in one pixel decode to az {float(az2):.1f} deg (paths at +10 / -170): "
          f"the circular mean is undefined here; this is the tail the RMSE carries, not judged by the smoke")

    # -- C. zero paths ------------------------------------------------------
    e = torch.zeros(0, device=dev)
    ch0 = multichannel_from_arrays(e, e, e, e, e, e)
    ok = tuple(ch0.shape) == (5, 540, 1080) and bool((ch0[0] == -200.0).all()) and bool((ch0[1:] == 0).all())
    f0 = equirect_to_perspective(ch0, 300, 200, 90.0)
    check("C zero paths", ok and tuple(f0.shape) == (5, 200, 300), f"power all floor: {bool((ch0[0] == -200).all())}, others zero: {bool((ch0[1:] == 0).all())}")

    # -- D. structured sample of the real pipeline -------------------------
    try:
        import mitsuba as mi
        mi.set_variant("cuda_ad_mono_polarized")
        from sionna.rt import Receiver
        from generate_dataset import Config, build_scene, projection_equirect, solve_paths, VIEW_YAWS
        from sionna.rt import PathSolver, Transmitter
        scene_xml = os.path.join(REPO, "sionna_tutorial/RF-3DGS_Sionna_simulation_tutorial/NIST_lobby_v1.0/NIST_lobby_V1.1_sionna12.xml")
        route = os.path.join(REPO, "sionna_tutorial/RF-3DGS_Sionna_simulation_tutorial/NIST_rx_loc.txt")
        cfg = Config(scene_xml=scene_xml, rx_loc_file=route, out_dir="", spectrum="MULTI", frequency=2.4e9,
                     materials="tutorial", samples_per_src=200_000, dashboard=False)
        scene = build_scene(cfg)
        scene.add(Transmitter(name="tx", position=list(cfg.tx_loc)))
        solver = PathSolver()
        pts = [l.split() for l in open(route) if len(l.split()) == 4]
        rows = [pts[0], pts[len(pts) // 2], pts[-1]]                      # endpoints and the middle
        positions = [[float(p[1]) / 1000, float(p[2]) / 1000, 1.625 - 1.713] for p in rows]
        positions.append([12.0, 0.0, -0.088])                              # outside the building
        t0 = time.time(); bad = []
        for k, pos in enumerate(positions):
            if "rx" in scene.receivers:
                scene.remove("rx")
            scene.add(Receiver(name="rx", position=pos))
            paths = solve_paths(solver, scene, cfg, view_index=k)
            eq = projection_equirect(paths, "MULTI", dev, cfg.splat_sigma)
            for yaw in VIEW_YAWS:
                face = equirect_to_perspective(eq, cfg.width, cfg.height, cfg.fov_deg, yaw_rad=yaw)
                hit = float((face[0] > -150).float().mean())
                pmin, pmax = float(face[0].min()), float(face[0].max())
                outdoor = k == len(positions) - 1
                ok = tuple(face.shape) == (5, 200, 300) and (
                    (hit == 0.0 and pmax <= -199.0) if outdoor else (0.2 <= hit <= 0.98 and -200 <= pmin and pmax <= -20))
                if not ok:
                    bad.append((k, round(math.degrees(yaw)), tuple(face.shape), round(hit, 3), round(pmin, 1), round(pmax, 1)))
                print(f"       position {k} yaw {math.degrees(yaw):5.0f}: hit {hit:.3f}, power {pmin:.1f}..{pmax:.1f} dB"
                      + ("  (outdoor)" if outdoor else ""))
        dt = time.time() - t0
        check("D structured sample (3 positions + outdoor, all yaws, all channels)", not bad and dt < 90,
              f"{'no violations' if not bad else bad}; {dt:.0f} s")
    except Exception as exc:
        check("D structured sample", False, f"{type(exc).__name__}: {exc}")

    # -- E. known failure ----------------------------------------------------
    enc = os.path.join(REPO, "output/rrf/encoding_comparison.json")
    if os.path.exists(enc):
        aod3 = json.load(open(enc))["aod3"]["az"]
        ok = 20.0 <= aod3["median"] <= 35.0
        check("E known failure: tutorial RGB encoding decodes badly", ok, f"azimuth median {aod3['median']:.1f} deg (want 20..35)")
    else:
        skip("E known failure", "no output/rrf/encoding_comparison.json; the smoke is incomplete")

    n_fail = sum(1 for _, s, _ in results if s == FAIL); n_skip = sum(1 for _, s, _ in results if s == SKIP)
    print(f"\n{len(results)} checks: {len(results) - n_fail - n_skip} pass, {n_fail} fail, {n_skip} skip; {time.time() - t_start:.0f} s total")
    print("not answered by this smoke: tail behaviour (P90/RMSE), stability at scale, memory, throughput")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
