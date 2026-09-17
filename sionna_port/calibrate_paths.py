"""Find solver settings that produce a path count comparable to the paper's.

The released dataset reports "more than 300,000 MPCs per Tx-Rx pair" at 60 GHz.
A depth-1 solve with diffuse reflection and 100k samples gives single digits, and
a spectrum built from single-digit path counts is just the array's point spread
function -- a regular sidelobe lattice with no scene structure in it.

0.19's scat_keep_prob has no counterpart in the rewritten solver, so the knobs
that remain are samples_per_src, max_depth and which interaction types are on.
"""

from __future__ import annotations

import argparse
import time

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene-xml", required=True)
    ap.add_argument("--tx", type=float, nargs=3, default=[6.905, 0.0, 0.287])
    ap.add_argument("--rx", type=float, nargs=3, default=[3.0, -2.0, 0.0])
    ap.add_argument("--M", type=int, default=10)
    ap.add_argument("--variant", default="cuda_ad_mono_polarized")
    args = ap.parse_args()

    import mitsuba as mi
    mi.set_variant(args.variant)
    from sionna.rt import PathSolver, PlanarArray, Receiver, Transmitter, load_scene

    scene = load_scene(args.scene_xml, merge_shapes=True)
    scene.frequency = 60e9
    scene.tx_array = PlanarArray(num_rows=1, num_cols=1, pattern="iso",
                                 polarization="V")
    scene.rx_array = PlanarArray(num_rows=args.M, num_cols=args.M,
                                 vertical_spacing=0.5, horizontal_spacing=0.5,
                                 pattern="iso", polarization="V")
    scene.add(Transmitter(name="tx", position=args.tx))
    scene.add(Receiver(name="rx", position=args.rx, orientation=[0.0, 0.0, 0.0]))

    solver = PathSolver()

    # (max_depth, samples_per_src, diffuse, specular, refraction)
    trials = [
        (1, 10**5, True,  True,  False),
        (1, 10**6, True,  True,  False),
        (1, 10**7, True,  True,  False),
        (2, 10**6, True,  True,  False),
        (3, 10**6, True,  True,  False),
        (3, 10**7, True,  True,  False),
        (5, 10**6, True,  True,  False),
        (3, 10**6, True,  True,  True),
        (3, 10**6, False, True,  False),
    ]

    print(f"{'depth':>5} {'samples':>10} {'diff':>5} {'spec':>5} {'refr':>5} "
          f"{'paths':>8} {'delay taps':>11} {'seconds':>8}")
    print("-" * 68)
    for depth, samples, diffuse, specular, refraction in trials:
        t0 = time.time()
        try:
            paths = solver(scene=scene, max_depth=depth, samples_per_src=samples,
                           los=True, specular_reflection=specular,
                           diffuse_reflection=diffuse, refraction=refraction,
                           synthetic_array=True, seed=42)
            n = int(np.asarray(paths.valid).sum())
            tau = np.asarray(paths.tau).reshape(-1)
            tau = tau[np.isfinite(tau) & (tau > 0)]
            taps = len(np.unique(np.floor(tau / 1e-9 / 0.1))) if tau.size else 0
            dt = time.time() - t0
            print(f"{depth:>5} {samples:>10,} {str(diffuse):>5} {str(specular):>5} "
                  f"{str(refraction):>5} {n:>8,} {taps:>11,} {dt:>8.1f}")
        except Exception as exc:
            print(f"{depth:>5} {samples:>10,} -> {type(exc).__name__}: "
                  f"{str(exc)[:60]}")

    print(f"\nMVDR needs at least {args.M**2} independent delay taps for an "
          f"invertible covariance.")


if __name__ == "__main__":
    main()
