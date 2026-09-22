"""Is Sionna 2.1's path solve differentiable in the two quantities we need?

Tx-side planning rests on two gradients: d(received power)/d(scattering
coefficient), for fitting materials from observations at one transmitter
position, and d(received power)/d(transmitter position), for placing the
transmitter directly by gradient descent. Sionna advertises both. This checks
them on this machine, on the real scene, before anything is built on top.

Each autodiff gradient is printed next to a central finite difference of the
same quantity. The position gradient matched FD within 5 %; the material one
agrees in sign and order only, because the diffuse sampler is stochastic and FD
differences the noise as well as the signal.
"""

from __future__ import annotations

import argparse
import math

import numpy as np


def main():
    """Run both gradient checks and print each against its finite difference."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene-xml", required=True)
    # Defaults are a real Tx/Rx pair from the tutorial campaign, not arbitrary points.
    ap.add_argument("--tx", type=float, nargs=3, default=[6.905, 0.0, 0.287])
    ap.add_argument("--rx", type=float, nargs=3, default=[2.728, 0.240, -0.968])
    ap.add_argument("--scattering", type=float, default=0.7)
    ap.add_argument("--samples", type=int, default=200_000)
    ap.add_argument("--variant", default="cuda_ad_mono_polarized")
    args = ap.parse_args()

    import drjit as dr
    import mitsuba as mi
    mi.set_variant(args.variant)
    # The reverse-mode kernel in evaluated loop mode dies in ptxas with
    # "Smem spilling should not be enabled when functions use abi"; this is
    # the flag that turns that spilling on.
    dr.set_flag(dr.JitFlag.SpillToSharedMemory, False)
    from sionna.rt import PathSolver, PlanarArray, Receiver, Transmitter, load_scene

    # Scene built inline rather than via scene_common, so this probe stays
    # independent of the shared helpers it is meant to justify.
    scene = load_scene(args.scene_xml, merge_shapes=True)
    scene.frequency = 60e9
    scene.tx_array = PlanarArray(num_rows=1, num_cols=1, pattern="iso",
                                 polarization="V")
    scene.rx_array = PlanarArray(num_rows=1, num_cols=1, pattern="iso",
                                 polarization="V")
    for material in scene.radio_materials.values():
        material.scattering_coefficient = args.scattering
    tx = Transmitter(name="tx", position=[float(v) for v in args.tx])
    rx = Receiver(name="rx", position=[float(v) for v in args.rx])
    scene.add(tx)
    scene.add(rx)
    solver = PathSolver()
    # Symbolic Dr.Jit loops need an iteration bound for reverse mode; evaluated
    # mode unrolls them and needs nothing.
    solver.loop_mode = "evaluated"

    def scalar(t):
        """A 0-d Dr.Jit tensor as a Python float."""
        return float(np.asarray(t).reshape(-1)[0])

    def received_power():
        """Total |a|^2 at the receiver, as a differentiable Dr.Jit scalar.

        seed is fixed so that two calls differ only by the parameter that
        changed -- without that, the finite differences below would be
        dominated by Monte-Carlo noise.
        """
        paths = solver(scene=scene, max_depth=1, samples_per_src=args.samples,
                       los=True, specular_reflection=True,
                       diffuse_reflection=True, refraction=False,
                       synthetic_array=True, seed=42)
        a_re, a_im = paths.a           # Dr.Jit tensors, differentiable
        return dr.sum(dr.square(a_re) + dr.square(a_im))

    # --- 1. gradient w.r.t. one material's scattering coefficient ---------
    # Just the first material: this is a capability probe, not a calibration.
    mat = next(iter(scene.radio_materials.values()))
    print(f"material under test: {mat.name}")
    s = mat.scattering_coefficient
    dr.enable_grad(s)                  # mark as a differentiable input
    power = received_power()
    dr.backward(power)                 # reverse pass through the whole solve
    g_s = dr.grad(s)
    p0 = scalar(power)
    print(f"  P_rx = {p0:.4e}   dP/d(scattering) = {np.asarray(g_s)}")

    # finite-difference check on the same knob
    # eps = 0.05 is a compromise: smaller and the sampler noise swamps the
    # difference, larger and the curvature of P(s) biases it.
    eps = 0.05
    mat.scattering_coefficient = args.scattering + eps
    p_plus = scalar(received_power())
    mat.scattering_coefficient = args.scattering - eps
    p_minus = scalar(received_power())
    mat.scattering_coefficient = args.scattering      # restore before part 2
    fd = (p_plus - p_minus) / (2 * eps)
    print(f"  finite difference          = {fd:.4e}")

    # --- 2. gradient w.r.t. transmitter position -------------------------
    pos = tx.position
    dr.enable_grad(pos)
    power = received_power()
    dr.backward(power)
    g_pos = dr.grad(pos)               # [dP/dx, dP/dy, dP/dz]
    print(f"\ntransmitter position gradient dP/d(p_tx) = {np.asarray(g_pos).reshape(-1)}")

    # finite difference along x
    # Only x is differenced: three axes would triple the runtime and one
    # component is enough to confirm the scale is right.
    eps = 0.05
    tx.position = [args.tx[0] + eps, args.tx[1], args.tx[2]]
    p_plus = scalar(received_power())
    tx.position = [args.tx[0] - eps, args.tx[1], args.tx[2]]
    p_minus = scalar(received_power())
    tx.position = [float(v) for v in args.tx]
    print(f"  finite difference along x  = {(p_plus - p_minus) / (2 * eps):.4e}")


if __name__ == "__main__":
    main()
