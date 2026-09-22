"""Sweep solver settings, record channel metrics, and emit JSON for the dashboard.

Each configuration is solved at several receiver positions, and every run records
both what the spectrum looks like and what the channel is actually doing. The
second half is the part RF-3DGS never measures: its evaluation is PSNR and SSIM
between colourmapped images, which cannot see delay spread, coherence bandwidth
or array gain at all.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime

import numpy as np


def build_scene(scene_xml, frequency, M, scattering, variant):
    """A fresh scene at one scattering coefficient.

    Rebuilt per configuration rather than mutated, because changing a material
    on a loaded scene does not reliably invalidate everything the solver caches.
    The transmitter is the NIST measurement position and is fixed here.
    """
    import mitsuba as mi
    # Only set the variant if nothing has yet: changing it after Dr.Jit has
    # initialised raises.
    if mi.variant() is None:
        mi.set_variant(variant)
    from sionna.rt import load_scene, PlanarArray, Transmitter

    scene = load_scene(scene_xml, merge_shapes=True)
    scene.frequency = frequency
    scene.tx_array = PlanarArray(num_rows=1, num_cols=1, pattern="iso",
                                 polarization="V")
    scene.rx_array = PlanarArray(num_rows=M, num_cols=M, vertical_spacing=0.5,
                                 horizontal_spacing=0.5, pattern="iso",
                                 polarization="V")
    scene.add(Transmitter(name="tx", position=[6.905, 0.0, 0.287]))
    for material in scene.radio_materials.values():
        material.scattering_coefficient = scattering
    return scene


def sweep(args):
    """Run every (scattering, depth) pair and write one JSON entry per configuration."""
    import mitsuba as mi
    mi.set_variant(args.variant)
    from sionna.rt import PathSolver, Receiver

    from rf_metrics import channel_metrics, frequency_response
    from rf_spectra import ArrayGrid, cbf_spectrum, mvdr_spectrum, paths_to_response
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    grid = ArrayGrid.build(args.M, args.width, args.height, args.fov, device=device)
    solver = PathSolver()

    # Fixed seed, so every configuration is scored at the SAME receiver
    # positions -- the sweep compares settings, not sampling of the room.
    rng = np.random.default_rng(0)
    rx_positions = [[rng.uniform(-3.0, 6.0), rng.uniform(-8.0, 0.0),
                     rng.uniform(-1.1, 0.2)] for _ in range(args.positions)]

    # Full cross product of the two swept axes.
    configs = []
    for scattering in args.scattering:
        for depth in args.depth:
            configs.append({"scattering_coefficient": scattering,
                            "max_depth": depth})

    runs = []
    for cfg in configs:
        scene = build_scene(args.scene_xml, args.frequency, args.M,
                            cfg["scattering_coefficient"], args.variant)
        label = f"s={cfg['scattering_coefficient']:g}, depth={cfg['max_depth']}"
        print(f"--- {label}")
        per_position, spectra, solve_times, cfr_curve = [], [], [], None

        for i, pos in enumerate(rx_positions):
            # One receiver at a time, swapped by name.
            if "rx" in scene.receivers:
                scene.remove("rx")
            scene.add(Receiver(name="rx", position=pos, orientation=[0.0, 0.0, 0.0]))
            t0 = time.time()
            paths = solver(scene=scene, max_depth=cfg["max_depth"],
                           samples_per_src=args.samples, los=True,
                           specular_reflection=True, diffuse_reflection=True,
                           refraction=False, synthetic_array=True, seed=42)
            n_valid = int(np.asarray(paths.valid).sum())
            # Timed even when the position yields nothing, so the cost figure
            # covers every solve actually performed.
            solve_times.append(time.time() - t0)
            if n_valid == 0:
                continue                 # a position with no coverage at all
            try:
                metrics = channel_metrics(paths, bandwidth_hz=args.bandwidth)
                per_position.append(metrics.as_dict())
            except ValueError:
                continue                 # channel_metrics rejects an empty path set

            response = paths_to_response(paths, args.time_interval, device=device)
            spec_fn = mvdr_spectrum if args.spectrum == "MVDR" else cbf_spectrum
            try:
                # MVDR takes the loading argument; CBF has no such parameter.
                _, spec_db = (spec_fn(response, grid, args.diagonal_loading)
                              if args.spectrum == "MVDR" else spec_fn(response, grid))
                spectra.append(spec_db.cpu().numpy().astype(np.float32))
            except ValueError as exc:
                # A singular covariance at one position is reported and skipped;
                # the channel metrics for it are still kept.
                print(f"    spectrum failed at position {i}: {exc}")

            # Frequency response from the first position only: it is per-position
            # detail, and one curve is enough to show the selectivity.
            if cfr_curve is None and args.cfr:
                try:
                    freqs, mag_db = frequency_response(
                        paths, args.frequency, args.bandwidth, args.subcarriers)
                    cfr_curve = {"frequency_hz": freqs.tolist(),
                                 "magnitude_db": mag_db.tolist()}
                except Exception as exc:
                    print(f"    cfr failed: {type(exc).__name__}: {exc}")

        if not per_position:
            print("    no valid positions")
            continue

        # One rendered panel per configuration, from the same receiver position
        # every time, so the grid varies only in the setting being swept.
        preview_rel = None
        if spectra and args.previews:
            preview_dir = os.path.join(os.path.dirname(args.out) or ".",
                                       "ablation_previews")
            os.makedirs(preview_dir, exist_ok=True)
            import imageio.v2 as imageio
            from matplotlib import colormaps
            panel = spectra[0]
            # Per-panel normalisation: these are for eyeballing the structure,
            # not for comparing absolute levels between configurations.
            lo, hi = float(panel.min()), float(panel.max())
            norm = np.clip((panel - lo) / max(hi - lo, 1e-9), 0.0, 1.0)
            rgb = (colormaps["jet"](norm)[..., :3] * 255).astype(np.uint8)
            name = f"s{cfg['scattering_coefficient']:g}_d{cfg['max_depth']}.png"
            imageio.imwrite(os.path.join(preview_dir, name), rgb)
            # Stored relative, so the dashboard resolves it from its own directory.
            preview_rel = os.path.join("ablation_previews", name)

        # Mean and spread across positions: the spread is what says whether a
        # difference between configurations is bigger than the room's variation.
        keys = per_position[0].keys()
        summary = {k: float(np.mean([p[k] for p in per_position])) for k in keys}
        spread = {k: float(np.std([p[k] for p in per_position])) for k in keys}

        entry = {"label": label, **cfg,
                 "preview": preview_rel,
                 "positions": len(per_position),
                 "mean_solve_seconds": float(np.mean(solve_times)),
                 "metrics_mean": summary,
                 "metrics_std": spread,
                 # Per-position values are kept, so the distribution can be
                 # re-analysed without re-solving.
                 "per_position": per_position}
        if spectra:
            from rf_metrics import spectrum_stats
            entry["spectrum"] = spectrum_stats(np.stack(spectra))
        if cfr_curve:
            entry["cfr"] = cfr_curve
        runs.append(entry)
        print(f"    paths {summary['num_paths']:,.0f}  "
              f"rms delay {summary['rms_delay_spread_ns']:.1f} ns  "
              f"K {summary['k_factor_db']:.1f} dB  "
              f"capacity {summary['capacity_bps_hz']:.1f} bps/Hz")

    # The settings travel with the results, so a dashboard card can state them.
    out = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "scene_xml": args.scene_xml,
        "frequency_hz": args.frequency,
        "bandwidth_hz": args.bandwidth,
        "M": args.M,
        "spectrum": args.spectrum,
        "samples_per_src": args.samples,
        "positions": args.positions,
        "runs": runs,
    }
    with open(args.out, "w", encoding="utf-8") as fid:
        json.dump(out, fid, indent=1)
    print(f"\nwrote {args.out} with {len(runs)} configurations")


def main():
    """Parse arguments and run the sweep."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene-xml", required=True)
    ap.add_argument("--out", default="ablation.json")
    # 0.0 is included as the control: no diffuse scattering at all.
    ap.add_argument("--scattering", type=float, nargs="+",
                    default=[0.0, 0.3, 0.5, 0.7])
    ap.add_argument("--depth", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--positions", type=int, default=8)
    ap.add_argument("--samples", type=int, default=1_000_000)
    ap.add_argument("--M", type=int, default=10)
    ap.add_argument("--width", type=int, default=300)
    ap.add_argument("--height", type=int, default=200)
    ap.add_argument("--fov", type=float, default=90.0)
    ap.add_argument("--frequency", type=float, default=60e9)
    ap.add_argument("--bandwidth", type=float, default=400e6)
    ap.add_argument("--subcarriers", type=int, default=128)
    ap.add_argument("--time-interval", type=float, default=0.1)
    ap.add_argument("--spectrum", default="MVDR", choices=["CBF", "MVDR"])
    # 0.0 means no loading: MVDR then raises on a singular covariance rather
    # than silently returning a spectrum built from numerical noise.
    ap.add_argument("--diagonal-loading", type=float, default=0.0)
    ap.add_argument("--cfr", action="store_true", default=True)
    ap.add_argument("--no-previews", dest="previews", action="store_false",
                    help="skip the rendered panel per configuration")
    ap.add_argument("--variant", default="cuda_ad_mono_polarized")
    sweep(ap.parse_args())


if __name__ == "__main__":
    main()
