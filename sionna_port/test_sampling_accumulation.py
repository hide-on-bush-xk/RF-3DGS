"""Does a cheap view with a moved lattice accumulate towards an expensive one?

The claim behind `per_view_seed`: a low `samples_per_src` leaves sampling
structure in each spectrum, but if the lattice moves between views that
structure is independent from view to view, so a multi-view fit averages it out
and converges towards the high-sample answer. Reusing one seed instead leaves a
residual that is identical everywhere, which no amount of fitting can remove --
it looks like a real feature of the field.

This measures both, against a high-sample reference at the same pose.

The design is a matched pair: moving and fixed lattices use the same pose, the
same budget and the same number of views, so the only difference between the
two columns of output is whether the seed advances.
"""

from __future__ import annotations

import argparse
import math

import numpy as np


def main():
    """Build the reference, then accumulate both variants view by view."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene-xml", required=True)
    ap.add_argument("--rx", type=float, nargs=3, default=[2.728, 0.240, -0.968])
    ap.add_argument("--yaw-deg", type=float, default=-90.0)
    ap.add_argument("--tx", type=float, nargs=3, default=[6.905, 0.0, 0.287])
    # 80x the low budget, so the reference's own sampling error is negligible
    # against the quantity being measured.
    ap.add_argument("--reference-samples", type=int, default=4_000_000)
    ap.add_argument("--low-samples", type=int, default=50_000)
    ap.add_argument("--views", type=int, default=8)
    ap.add_argument("--M", type=int, default=10)
    ap.add_argument("--scattering", type=float, default=0.7)
    ap.add_argument("--variant", default="cuda_ad_mono_polarized")
    args = ap.parse_args()

    import mitsuba as mi
    mi.set_variant(args.variant)
    import torch
    from sionna.rt import (PathSolver, PlanarArray, Receiver, Transmitter,
                           load_scene)
    from rf_spectra import ArrayGrid, mvdr_spectrum, paths_to_response

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 300x200 at 90 degrees: the released dataset's spectrum geometry.
    grid = ArrayGrid.build(args.M, 300, 200, 90.0, device=device)

    scene = load_scene(args.scene_xml, merge_shapes=True)
    scene.frequency = 60e9
    scene.tx_array = PlanarArray(num_rows=1, num_cols=1, pattern="iso",
                                 polarization="V")
    scene.rx_array = PlanarArray(num_rows=args.M, num_cols=args.M,
                                 vertical_spacing=0.5, horizontal_spacing=0.5,
                                 pattern="iso", polarization="V")
    for material in scene.radio_materials.values():
        material.scattering_coefficient = args.scattering
    scene.add(Transmitter(name="tx", position=[float(v) for v in args.tx]))
    scene.add(Receiver(name="rx", position=[float(v) for v in args.rx],
                       orientation=[math.radians(args.yaw_deg), 0.0, 0.0]))
    solver = PathSolver()

    def spectrum(samples, seed):
        """One MVDR spectrum in dB. The seed IS the sampling lattice."""
        paths = solver(scene=scene, max_depth=1, samples_per_src=samples,
                       los=True, specular_reflection=True,
                       diffuse_reflection=True, refraction=False,
                       synthetic_array=True, seed=seed)
        response = paths_to_response(paths, 0.1, device=device)
        _, db = mvdr_spectrum(response, grid)
        # float64 so the running mean over views does not accumulate error.
        return db.detach().cpu().numpy().astype(np.float64)

    print(f"reference: {args.reference_samples:,} samples")
    reference = spectrum(args.reference_samples, 1)

    def rmse(x):
        """Distance to the high-sample reference, in dB."""
        return float(np.sqrt(((x - reference) ** 2).mean()))

    print(f"low budget: {args.low_samples:,} samples, {args.views} views\n")
    moving, fixed = [], []
    for i in range(args.views):
        moving.append(spectrum(args.low_samples, 100 + i))   # lattice moves
        fixed.append(spectrum(args.low_samples, 100))        # lattice reused
        # Running means, printed per view: the shape of the curve is the result.
        # Moving should fall roughly as 1/sqrt(n); fixed should flatten.
        m = np.mean(moving, axis=0)
        f = np.mean(fixed, axis=0)
        print(f"  {i+1:2d} view(s)   moving lattice {rmse(m):7.3f} dB   "
              f"fixed lattice {rmse(f):7.3f} dB")

    m_final, f_final = rmse(np.mean(moving, axis=0)), rmse(np.mean(fixed, axis=0))
    single = rmse(moving[0])
    print(f"\nsingle cheap view          {single:7.3f} dB")
    print(f"{args.views} views, lattice moving  {m_final:7.3f} dB  "
          f"({single / max(m_final, 1e-9):.2f}x better than one)")
    print(f"{args.views} views, lattice fixed   {f_final:7.3f} dB  "
          f"({single / max(f_final, 1e-9):.2f}x)")
    # A 5 % margin, so a marginal difference is not reported as a result.
    if m_final < f_final * 0.95:
        print("\nMoving the lattice per view accumulates; reusing it does not.")
    else:
        print("\nNo accumulation advantage measured at this budget.")


if __name__ == "__main__":
    main()
