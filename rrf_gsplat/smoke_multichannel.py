"""Smoke test for the multi-channel spectrum pipeline (CLAUDE.md: smoke first).

Pass criteria, v2 (2026-09-18). A gains one sub-check, D is replaced; B, C, E
are unchanged from v1.

  A. analytic control -- one synthetic path (amp 1, delay 50 ns, AoA
     theta 80 / phi 30 deg, AoD theta 100 / phi -120 deg), sigma 3 px:
     at the AoA pixel the channels decode to |az + 120| <= 0.34 deg (one
     equirect pixel), |zen - 100| <= 0.34 deg, |delay - 50| <= 1e-3 ns,
     power finite; the four pinhole faces resample to [5, 200, 300]; and
     the path's pinhole projection (the formula D1/D2 rely on, written
     independently of the resampler) lands within 1 px of the yaw-0 face's
     power maximum.
  B. the tail mechanism, reported not judged -- two equal paths with
     opposite azimuth in one pixel: the circular mean is undefined; the
     decoded azimuth may be anything. Smoke only verifies no exception.
  C. degenerate input -- zero paths: [5, 540, 1080] with power == -200 dB
     everywhere and zeros elsewhere, no exception; resampling still works.
  D. structured sample of the real pipeline (needs the scene): the route's
     first, middle and last positions (endpoints included) x all 4 yaws,
     plus one receiver outside the building (12, 0, -0.088). Per indoor
     position:
     D1 path conservation -- every valid path is inside exactly one of the
        four pinhole faces (frustum test in the face's camera frame) or
        outside the faces' elevation coverage (|tan el| > (H/2f) cos delta,
        delta = azimuth offset from the nearest face centre), never both and
        never neither: sum over faces + outside == total, zero mismatches.
        The share of paths and of power outside (the polar-cap gap) is
        reported, not judged.
     D2 landing -- every in-face path whose kernel-centre power
        10 log10(|a|^2 k0) lies >= 20 dB above the truncation floor lands on
        a hit pixel of its rendered face (power > power_floor, -150 dB): 100 %.
     D3 relative dynamic range -- the maximum face power minus the minimum
        face power over the hit pixels (> power_floor) that in-face paths
        land on, within [40, 140] dB. (A face pixel resampled across a
        hit / floor edge carries a mixed value below the floor; it is not a
        hit, in the smoke as in the generator's channel ranges.)
     D4 truncation -- on the equirect no pixel has floor < power <
        power_floor (-150 dB), and channels 1-4 are zero wherever the power
        is at the floor; on the faces cos^2 + sin^2 <= 1 + 1e-4, zen in
        [0, 1], delay >= 0.
     D5 the outdoor receiver yields zero paths and all faces at floor / zero.
     D6 the whole sample under 90 s on an exclusive RTX 3060.
  E. known-failure control -- the tutorial's angle x amplitude RGB encoding,
     decoded from the trained AOD3 model's renders against the MULTI truth:
     azimuth median in [20, 35] deg (measured 27 deg on the full run).
     Requires output/rrf/m_aod3_24_tut and the MULTI dataset; skipped with a
     notice if absent (then the smoke is incomplete, not passed).

What this smoke cannot answer: tail behaviour (P90 / RMSE), numerical
stability at scale, memory and throughput, and the bilinear resampling of
decoded channels across a hit / floor boundary (such pixels mix a value
with zero; at the current floor they fall below the hit threshold, which is
not checked here).

Record of the first run (v1 criteria, 2026-09-18): A, C, E pass; D FAILED
its stated ranges while the pipeline behaved correctly -- the ranges were
mis-specified: (1) pixels a kernel tail barely touched carried 10 log10 of
a vanishing power, down to -273 dB, below the -200 dB floor the criterion
assumed was the minimum; (2) faces looking at a wall reached hit fractions
of 4-14 %, below the 20 % assumed. v1's proposed fix (widen the range to
-300 dB) was rejected: a kernel tail is a numerical residue, not a path
loss. v2 instead truncates the splat at a noise floor (power_floor_db,
-150 dB, as the paper truncates its path loss) and replaces the hit
fraction by the path-conservation and relative-dynamic-range criteria above.

Record of v2 (2026-09-18, exclusive RTX 3060): first run 10 of 11 pass; D3
FAILED with 174.5 dB at two positions because "hit" was written as power >
floor + 1 dB on the faces, and a face pixel resampled across a hit / floor
edge carries a mixed value (-199 .. -150 dB) that is below the truncation
floor but above -199. That is the boundary effect listed above as not
checked, surfacing in D3. The hit definition was aligned with the
generator's (power > power_floor) in both the smoke and channel_ranges, and
the second run passes 11 of 11: conservation exact (0 mismatches), strong
paths landing 100 %, relative dynamic range 125.6 / 84.6 / 125.7 dB, no
gap pixels, 0 s. Reported, not judged: 8-12 % of the paths but only 0-5 %
of the power fall outside the four faces' elevation coverage (+-33.7 deg
at a face centre, +-25.2 deg at an edge), and 9.5-26 % of the paths sit
under the -150 dB truncation floor.

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
    """Record and print one criterion. `detail` always carries the measured
    value, so a pass is as auditable as a failure."""
    results.append((name, PASS if ok else FAIL, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")


def skip(name, detail):
    """A criterion that could not be evaluated. Distinct from a pass: a smoke
    with skips is incomplete, not passed (see the module docstring on E)."""
    results.append((name, SKIP, detail))
    print(f"[SKIP] {name}: {detail}")


def decode(ch):
    """[5,...] channels -> az deg, zen deg, delay ns, power dB."""
    az = torch.rad2deg(torch.atan2(ch[2], ch[1]))
    return az, ch[3] * 180.0, ch[4], ch[0]


def wrap(a):
    """An angle difference in radians wrapped into [-pi, pi)."""
    return (a + math.pi) % (2 * math.pi) - math.pi


def project(theta, phi, yaw, width, height, fov_deg):
    """Pinhole projection of arrival directions into the face at `yaw`.

    Written from compute_angle_matrices' definitions, not from the resampler:
    a pixel direction (x, y, z) has phi = atan2(-x, z), theta = pi/2 - asin(y),
    so x = -sin(theta) sin(phi_c), y = cos(theta), z = sin(theta) cos(phi_c)
    with the camera azimuth phi_c = phi - yaw (equirect_to_perspective adds
    yaw to the camera grid). Returns (inside, row, col).
    """
    f = width / (2 * math.tan(math.radians(fov_deg) / 2))
    phc = phi - yaw
    st = torch.sin(theta)
    x, y, z = -st * torch.sin(phc), torch.cos(theta), st * torch.cos(phc)
    zs = z.clamp_min(1e-9)
    inside = (z > 0) & ((x / zs).abs() <= (width / 2) / f) & ((y / zs).abs() <= (height / 2) / f)
    col = (((x / zs) * f + width / 2) * (width - 1) / width).round().long().clamp(0, width - 1)
    row = (((height / 2) - (y / zs) * f) * (height - 1) / height).round().long().clamp(0, height - 1)
    return inside, row, col


def main():
    """Run every criterion in order and print a PASS / FAIL / SKIP line for each.

    A to C need nothing but torch; D needs the scene and a GPU; E needs a
    trained AOD3 run. The later groups are guarded so the earlier ones still
    report on a machine that cannot run them.
    """
    from rf_spectra import equirect_to_perspective, multichannel_from_arrays
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t_start = time.time()
    W, H, FOV = 300, 200, 90.0
    YAWS = (-math.pi / 2, 0.0, math.pi / 2, math.pi)

    # -- A. one path ------------------------------------------------------
    r = lambda d: torch.tensor([math.radians(d)], device=dev)
    ch = multichannel_from_arrays(torch.tensor([1.0], device=dev), torch.tensor([50e-9], device=dev),
                                  r(80), r(30), r(100), r(-120), scale=3, sigma=3.0)
    ti, pi = int(round(80 * 3)), int(round((-30 + 180) * 3))         # equirect_splat's index convention
    # The whole point of A: one path, so the "power-weighted mean" is that
    # path's own angle and the decode must return exactly what went in.
    az, zen, dl, pw = decode(ch[:, ti, pi])
    ok = (abs(float(az) + 120) <= 0.34 and abs(float(zen) - 100) <= 0.34 and abs(float(dl) - 50) <= 1e-3
          and math.isfinite(float(pw)) and tuple(ch.shape) == (5, 540, 1080))
    check("A single path decodes to itself", ok,
          f"az {float(az):.3f} (want -120), zen {float(zen):.3f} (want 100), delay {float(dl):.4f} ns (want 50), power {float(pw):.1f} dB, shape {tuple(ch.shape)}")
    faces = [equirect_to_perspective(ch, W, H, FOV, yaw_rad=y) for y in YAWS]
    check("A four faces resample", all(tuple(f.shape) == (5, H, W) for f in faces), f"{[tuple(f.shape) for f in faces]}")
    # The independent check: project() was written from the angle-grid
    # definitions rather than from the resampler, so agreement between the two
    # is evidence and not a tautology.
    inside, row, col = project(r(80), r(30), 0.0, W, H, FOV)
    am = int(faces[1][0].argmax()); ar, ac = am // W, am % W    # faces[1] is yaw 0
    ok = bool(inside[0]) and abs(ar - int(row[0])) <= 1 and abs(ac - int(col[0])) <= 1
    check("A projection agrees with the resampler", ok,
          f"projected (row {int(row[0])}, col {int(col[0])}), face power maximum at ({ar}, {ac}), inside {bool(inside[0])}")

    # -- B. the tail mechanism (report only) ------------------------------
    ch2 = multichannel_from_arrays(torch.tensor([1.0, 1.0], device=dev), torch.tensor([50e-9, 60e-9], device=dev),
                                   r(80).repeat(2), r(30).repeat(2), r(100).repeat(2),
                                   torch.tensor([math.radians(10), math.radians(-170)], device=dev), scale=3, sigma=3.0)
    az2, _, _, _ = decode(ch2[:, ti, pi])
    print(f"[INFO] B two opposite paths in one pixel decode to az {float(az2):.1f} deg (paths at +10 / -170): "
          f"the circular mean is undefined here; this is the tail the RMSE carries, not judged by the smoke")

    # -- C. zero paths ------------------------------------------------------
    # The degenerate path the project has actually been burned by: an outdoor
    # or fully occluded receiver returns no paths at all.
    e = torch.zeros(0, device=dev)
    ch0 = multichannel_from_arrays(e, e, e, e, e, e)
    ok = tuple(ch0.shape) == (5, 540, 1080) and bool((ch0[0] == -200.0).all()) and bool((ch0[1:] == 0).all())
    f0 = equirect_to_perspective(ch0, W, H, FOV)
    check("C zero paths", ok and tuple(f0.shape) == (5, H, W), f"power all floor: {bool((ch0[0] == -200).all())}, others zero: {bool((ch0[1:] == 0).all())}")

    # -- D. structured sample of the real pipeline -------------------------
    try:
        import mitsuba as mi
        mi.set_variant("cuda_ad_mono_polarized")
        from sionna.rt import PathSolver, Receiver, Transmitter
        from generate_dataset import Config, build_scene, projection_equirect, solve_paths, VIEW_YAWS
        from rf_spectra import _gaussian_kernel, _path_arrays
        scene_xml = os.path.join(REPO, "sionna_tutorial/RF-3DGS_Sionna_simulation_tutorial/NIST_lobby_v1.0/NIST_lobby_V1.1_sionna12.xml")
        route = os.path.join(REPO, "sionna_tutorial/RF-3DGS_Sionna_simulation_tutorial/NIST_rx_loc.txt")
        cfg = Config(scene_xml=scene_xml, rx_loc_file=route, out_dir="", spectrum="MULTI", frequency=2.4e9,
                     materials="tutorial", dashboard=False)
        W, H, FOV = cfg.width, cfg.height, cfg.fov_deg
        f = W / (2 * math.tan(math.radians(FOV) / 2)); lim_y = (H / 2) / f
        k0 = float(_gaussian_kernel(3, cfg.splat_sigma, dev)[0].max())
        pf, fl = cfg.power_floor_db, -200.0
        scene = build_scene(cfg)
        scene.add(Transmitter(name="tx", position=list(cfg.tx_loc)))
        solver = PathSolver()
        # The structured sample required by CLAUDE.md: both route endpoints (the
        # boundary cases) and the middle, never a [::k] stride, plus one
        # deliberately degenerate receiver.
        pts = [l.split() for l in open(route) if len(l.split()) == 4]
        rows = [pts[0], pts[len(pts) // 2], pts[-1]]                      # endpoints and the middle
        positions = [[float(p[1]) / 1000, float(p[2]) / 1000, 1.625 - 1.713] for p in rows]
        positions.append([12.0, 0.0, -0.088])                              # outside the building
        t0 = time.time()
        viol = {"D1": [], "D2": [], "D3": [], "D4": []}; caps, drs = [], []
        print(f"       elevation coverage of the four faces: +-{math.degrees(math.atan(lim_y)):.1f} deg at a face centre, "
              f"+-{math.degrees(math.atan(lim_y * math.cos(math.pi / 4))):.1f} deg at a face edge; truncation floor {pf} dB, "
              f"kernel centre weight {10 * math.log10(k0):.1f} dB")
        for k, pos in enumerate(positions):
            if "rx" in scene.receivers:
                scene.remove("rx")
            scene.add(Receiver(name="rx", position=pos))
            paths = solve_paths(solver, scene, cfg, view_index=k)
            amp, tau, th_r, ph_r, th_t, ph_t = _path_arrays(paths, dev)
            eq = projection_equirect(paths, "MULTI", dev, cfg.splat_sigma, pf)
            faces = [equirect_to_perspective(eq, W, H, FOV, yaw_rad=y) for y in VIEW_YAWS]
            P = int(amp.numel())
            if k == len(positions) - 1:
                ok = P == 0 and all(bool((fc[0] <= fl + 1e-3).all()) and bool((fc[1:] == 0).all()) for fc in faces)
                check("D5 outdoor receiver: zero paths, all floor", ok, f"{P} paths")
                continue
            # D1 conservation
            ins, land = [], []
            for yaw in VIEW_YAWS:
                i_, r_, c_ = project(th_r, ph_r, yaw, W, H, FOV)
                ins.append(i_); land.append((r_, c_))
            ins = torch.stack(ins, 1)                                        # [P, 4]
            n_in = ins.sum(1)
            # The four 90-degree faces tile the azimuth circle but leave polar
            # caps uncovered. A path is legitimately outside when its elevation
            # exceeds the coverage at its azimuth offset from the nearest face
            # centre -- the cos(delta) is the widening towards a face edge.
            delta = torch.stack([wrap(ph_r - y) for y in VIEW_YAWS], 1).abs().min(1).values
            outside = torch.tan(math.pi / 2 - th_r).abs() > lim_y * torch.cos(delta)
            # Two failure modes in one count: a path in neither category or in
            # both (the != test), and a path counted by two faces (n_in > 1).
            mism = int(((n_in == 1) != ~outside).sum()) + int((n_in > 1).sum())
            cap_p = float(outside.float().mean()); cap_w = float((amp[outside] ** 2).sum() / (amp ** 2).sum())
            caps.append((cap_p, cap_w))
            if mism:
                viol["D1"].append((k, mism))
            # D2 landing, D3 dynamic range
            pw_path = 10 * torch.log10(amp ** 2 * k0)
            landed_min, n_strong, n_hit, per_face = [], 0, 0, []
            under = float((pw_path < pf).float().mean())
            for fi, fc in enumerate(faces):
                sel = ins[:, fi]
                hit_frac = float((fc[0] > pf).float().mean())
                if not sel.any():
                    per_face.append(f"yaw {math.degrees(VIEW_YAWS[fi]):4.0f}: 0 paths, hit {hit_frac:.3f}")
                    continue
                val = fc[0][land[fi][0][sel], land[fi][1][sel]]
                # Only paths 20 dB clear of the truncation floor are required to
                # land: one within a few dB of it may legitimately be truncated.
                strong = pw_path[sel] >= pf + 20
                n_strong += int(strong.sum()); n_hit += int((val[strong] > pf).sum())
                hits = val[val > pf]
                if hits.numel():
                    landed_min.append(float(hits.min()))
                per_face.append(f"yaw {math.degrees(VIEW_YAWS[fi]):4.0f}: {int(sel.sum()):,} paths, hit {hit_frac:.3f}, "
                                f"landed {int((val > pf).sum())}/{int(sel.sum())}")
            if n_hit != n_strong:
                viol["D2"].append((k, n_strong - n_hit, n_strong))
            pmax = max(float(fc[0].max()) for fc in faces)
            dr = pmax - min(landed_min) if landed_min else float("nan")
            drs.append(dr)
            if not 40 <= dr <= 140:
                viol["D3"].append((k, round(dr, 1)))
            # D4 truncation
            p = eq[0]
            gap = int(((p > fl + 1e-3) & (p < pf)).sum())
            nz = int((eq[1:][:, p <= fl + 1e-3] != 0).sum())
            bad4 = gap + nz
            for fc in faces:
                bad4 += int(((fc[1] ** 2 + fc[2] ** 2) > 1 + 1e-4).sum()) + int(((fc[3] < 0) | (fc[3] > 1)).sum()) + int((fc[4] < 0).sum())
            if bad4:
                viol["D4"].append((k, gap, nz, bad4))
            print(f"       position {k} {tuple(round(v, 2) for v in pos)}: {P:,} paths; outside the elevation coverage "
                  f"{cap_p:.1%} of paths / {cap_w:.1%} of power; mismatches {mism}; strong paths landing on a hit pixel "
                  f"{n_hit}/{n_strong}; paths under the truncation floor {under:.1%}; relative dynamic range {dr:.1f} dB (max {pmax:.1f}); equirect gap pixels {gap}, "
                  f"non-zero-at-floor {nz}\n         " + "; ".join(per_face))
        dt = time.time() - t0
        check("D1 path conservation (faces + outside == total, no double count)", not viol["D1"],
              f"mismatches {viol['D1'] or 0}; polar-cap gap per position (paths, power): "
              + ", ".join(f"({a:.1%}, {b:.1%})" for a, b in caps))
        check("D2 strong in-face paths land on hit pixels", not viol["D2"], f"missed (position, n, of) {viol['D2'] or 'none'}")
        check("D3 relative dynamic range in [40, 140] dB", not viol["D3"], f"{[round(d, 1) for d in drs]} dB")
        check("D4 truncation at the floor, channel ranges", not viol["D4"], f"violations (position, gap, nonzero, total) {viol['D4'] or 'none'}")
        check("D6 sample time under 90 s", dt < 90, f"{dt:.0f} s for {len(positions)} positions x 4 faces")
    except Exception as exc:
        import traceback; traceback.print_exc()
        check("D structured sample", False, f"{type(exc).__name__}: {exc}")

    # -- E. known failure ----------------------------------------------------
    enc = os.path.join(REPO, "output/rrf/encoding_comparison.json")
    if os.path.exists(enc):
        aod3 = json.load(open(enc))["aod3"]["az"]
        ok = 20.0 <= aod3["median"] <= 35.0
        check("E known failure: tutorial RGB encoding decodes badly", ok, f"azimuth median {aod3['median']:.1f} deg (want 20..35)")
    else:
        skip("E known failure", "no output/rrf/encoding_comparison.json; the smoke is incomplete")

    # Exit code is the result, so this can gate a full run from a shell script.
    # Note skips do not fail the run but are counted and reported: a smoke with
    # skips has not answered everything it claims to.
    n_fail = sum(1 for _, s, _ in results if s == FAIL); n_skip = sum(1 for _, s, _ in results if s == SKIP)
    print(f"\n{len(results)} checks: {len(results) - n_fail - n_skip} pass, {n_fail} fail, {n_skip} skip; {time.time() - t_start:.0f} s total")
    print("not answered by this smoke: tail behaviour (P90/RMSE), stability at scale, memory, throughput, boundary resampling")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
