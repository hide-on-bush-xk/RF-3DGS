"""End-to-end check of the port against a real scene and a real path solve.

The decisive test is the last one: with a single line-of-sight path, the
beamformer peak must land on the angle of arrival Sionna itself reports. If the
element ordering assumed by `rf_spectra._element_offsets` disagrees with the way
Sionna's PlanarArray lays out its elements, the peak lands somewhere else.

Run inside WSL:
    LD_LIBRARY_PATH=$CONDA_PREFIX/lib python smoke_sionna.py --scene-xml ...
"""

import argparse
import math
import sys
import time

import numpy as np
import torch

from rf_spectra import ArrayGrid, array_manifold_vector, cbf_spectrum, paths_to_response, steering_vector


def select_variant(variant: str) -> str:
    """Pick the Mitsuba variant before sionna.rt gets a chance to.

    sionna.rt sets `cuda_ad_mono_polarized` when CUDA is present, but WSL2 ships
    only a 14 KB OptiX loader stub, so the CUDA variant loads and then fails with
    "Could not initialize OptiX". Setting the variant here wins, because
    sionna.rt only chooses one when none is set yet.
    """
    import mitsuba as mi
    if variant != "auto":
        mi.set_variant(variant)
    return mi.variant() or "auto"


def sphere_grid(n_theta=181, n_phi=361, device=None):
    """Full-sphere angle grid, so the peak is findable in any direction."""
    theta = torch.linspace(0.0, math.pi, n_theta, device=device)
    phi = torch.linspace(-math.pi, math.pi, n_phi, device=device)
    return torch.meshgrid(theta, phi, indexing="ij")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene-xml", required=True)
    ap.add_argument("--tx", type=float, nargs=3, default=[6.905, 0.0, 0.287])
    ap.add_argument("--rx", type=float, nargs=3, default=[3.0, -2.0, 0.0])
    ap.add_argument("--M", type=int, default=10)
    ap.add_argument("--variant", default="llvm_ad_mono_polarized",
                    help="Mitsuba variant; 'auto' lets sionna.rt decide")
    args = ap.parse_args()

    print(f"mitsuba variant: {select_variant(args.variant)}")
    from sionna.rt import PathSolver, PlanarArray, Receiver, Transmitter, load_scene

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"torch device: {device}")

    t0 = time.time()
    scene = load_scene(args.scene_xml, merge_shapes=True)
    print(f"scene load: {time.time() - t0:.1f} s")
    scene.frequency = 60e9
    scene.tx_array = PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")
    scene.rx_array = PlanarArray(num_rows=args.M, num_cols=args.M,
                                 vertical_spacing=0.5, horizontal_spacing=0.5,
                                 pattern="iso", polarization="V")
    scene.add(Transmitter(name="tx", position=args.tx))
    scene.add(Receiver(name="rx", position=args.rx, orientation=[0.0, 0.0, 0.0]))
    print(f"scene loaded: {len(scene.objects)} objects, f = {np.asarray(scene.frequency).item()/1e9:.1f} GHz")

    solver = PathSolver()

    # --- 1. line of sight only, so there is exactly one path ------------------
    paths = solver(scene=scene, max_depth=0, los=True,
                   specular_reflection=False, diffuse_reflection=False,
                   refraction=False, samples_per_src=10_000, seed=42)
    a, tau = paths.cir(normalize_delays=False, out_type="torch")
    print(f"LOS solve: a {tuple(a.shape)}, tau {tuple(tau.shape)}")

    theta_r = np.asarray(paths.theta_r).reshape(-1)
    phi_r = np.asarray(paths.phi_r).reshape(-1)
    valid = np.asarray(paths.valid).reshape(-1)
    n_valid = int(valid.sum())
    print(f"valid paths: {n_valid}")
    if n_valid != 1:
        print("  expected exactly one LOS path; is the receiver behind geometry?")
        sys.exit(1)

    aoa_theta, aoa_phi = float(theta_r[0]), float(phi_r[0])
    print(f"Sionna AoA: theta {math.degrees(aoa_theta):7.2f} deg, "
          f"phi {math.degrees(aoa_phi):7.2f} deg")

    # --- 2. beamform over the whole sphere and find the peak -----------------
    th, ph = sphere_grid(device=device)
    grid = ArrayGrid(M=args.M, theta=th, phi=ph,
                     steering=steering_vector(args.M, th, ph),
                     manifold=array_manifold_vector(args.M, th, ph))
    response = paths_to_response(paths, time_interval_ns=1.0, device=device)
    print(f"response: {tuple(response.shape)} (elements x delay taps)")

    lin, _ = cbf_spectrum(response, grid)
    idx = int(lin.argmax())
    row, col = divmod(idx, lin.shape[1])
    peak_theta = float(th[row, col])
    peak_phi = float(ph[row, col])
    print(f"CBF peak  : theta {math.degrees(peak_theta):7.2f} deg, "
          f"phi {math.degrees(peak_phi):7.2f} deg")

    # The array lies in the y-z plane, so its manifold depends on
    # (sin(theta) sin(phi), cos(theta)) and phi is indistinguishable from
    # 180 - phi: both give the same steering vector to within float noise. The
    # spectrum therefore has two equal peaks and argmax picks one arbitrarily.
    # Accept either, and check the ambiguity itself separately below.
    mirror_phi = math.pi - aoa_phi
    d_theta = abs(math.degrees(peak_theta - aoa_theta))
    d_phi_direct = abs((math.degrees(peak_phi - aoa_phi) + 180) % 360 - 180)
    d_phi_mirror = abs((math.degrees(peak_phi - mirror_phi) + 180) % 360 - 180)
    d_phi = min(d_phi_direct, d_phi_mirror)
    which = "direct" if d_phi_direct <= d_phi_mirror else "mirror (180 - phi)"
    print(f"offset    : dtheta {d_theta:.2f} deg, dphi {d_phi:.2f} deg [{which}]")

    # One degree of grid spacing, plus beam width, so allow a few degrees.
    if d_theta < 3.0 and d_phi < 3.0:
        print("\nELEMENT ORDERING OK: beamformer agrees with Sionna's AoA "
              "up to the array's front/back ambiguity")
    else:
        print("\nELEMENT ORDERING MISMATCH: rf_spectra._element_offsets does not "
              "match Sionna's PlanarArray layout. Fix the offsets before "
              "generating a dataset.")
        sys.exit(2)

    # --- 2b. the ambiguity is a property of the array, not a bug -------------
    pair_theta = torch.tensor([[aoa_theta, aoa_theta]], device=device)
    pair_phi = torch.tensor([[aoa_phi, mirror_phi]], device=device)
    pair = steering_vector(args.M, pair_theta, pair_phi)
    coherence = (pair[:, 0, 0].conj() * pair[:, 0, 1]).sum().abs().item() / args.M ** 2
    print(f"front/back coherence: {coherence:.6f} "
          f"(1.0 means the two directions are indistinguishable)")
    if coherence > 0.999:
        print("  -> a planar array cannot separate phi from 180 - phi. Each "
              "pinhole view spans only +-45 deg of azimuth, so the ghost falls "
              "in a neighbouring view rather than the same image.")

    # --- 3. a realistic solve, for shapes and path counts --------------------
    paths = solver(scene=scene, max_depth=1, los=True,
                   specular_reflection=True, diffuse_reflection=True,
                   refraction=False, samples_per_src=100_000, seed=42)
    a, tau = paths.cir(normalize_delays=False, out_type="torch")
    print(f"\ndepth-1 solve with diffuse: a {tuple(a.shape)}, "
          f"{int(np.asarray(paths.valid).sum())} valid paths")
    resp = paths_to_response(paths, time_interval_ns=0.1, device=device)
    print(f"response: {tuple(resp.shape)}  "
          f"(needs >= {args.M**2} taps for MVDR without loading)")


if __name__ == "__main__":
    main()
