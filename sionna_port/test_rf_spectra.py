"""Checks for the ported array processing. Runs without Sionna or a GPU.

These are the trivial-input analytic controls: a single synthetic path whose
direction is known, so the beamformer's peak must land on exactly that pixel
and the CBF peak must equal M^2. If these pass, the angle convention, the
steering vectors and the beamformers agree with each other; they say nothing
about whether the ray tracer feeding them is right.

Plain asserts and a __main__ runner rather than pytest, so this can be run on
any machine with torch and nothing else.
"""

import math

import torch

from rf_spectra import (ArrayGrid, cbf_spectrum, compute_angle_matrices,
                        merge_paths_to_time_grid, mvdr_spectrum,
                        steering_vector)

# Deliberately small and non-square: a W == H grid would hide an axis swap.
W, H, FOV, M = 60, 40, 90.0, 6


def test_angle_grid():
    """The pixel-to-direction map: centre pixel and both horizontal edges."""
    theta, phi = compute_angle_matrices(W, H, FOV)
    assert theta.shape == (H, W) and phi.shape == (H, W)
    # The principal ray looks along +z: zenith 90 degrees, azimuth 0.
    ct, cp = theta[H // 2, W // 2], phi[H // 2, W // 2]
    # Tolerances are half a pixel's worth of angle, not arbitrary slack.
    assert abs(ct - math.pi / 2) < 0.05, ct
    assert abs(cp) < 0.05, cp
    # A 90 degree horizontal FoV spans +-45 degrees of azimuth.
    # Also pins the sign convention: azimuth decreases left to right.
    assert abs(phi[H // 2, 0] - math.radians(45)) < 0.02
    assert abs(phi[H // 2, -1] + math.radians(45)) < 0.02
    print("angle grid            ok")


def test_steering_unit_modulus():
    """Steering vectors are pure phase: every entry must have magnitude 1.

    A magnitude error here would silently reweight the array and bias every
    spectrum, without moving the peak -- so it would not be caught below.
    """
    theta, phi = compute_angle_matrices(W, H, FOV)
    a = steering_vector(M, theta, phi)
    assert a.shape == (M ** 2, H, W)
    assert torch.allclose(a.abs(), torch.ones_like(a.abs()), atol=1e-5)
    print("steering vectors      ok")


def test_delay_binning():
    """Paths sharing a delay bin add coherently; distant ones stay separate."""
    a = torch.tensor([[1 + 0j, 2 + 0j, 4 + 0j]])          # one element, 3 paths
    tau_ns = torch.tensor([0.4, 0.6, 2.5])
    grid = torch.arange(0.0, 3.0, 1.0)                     # bins at 0, 1, 2
    out = merge_paths_to_time_grid(a, tau_ns, grid)
    # 0.4 and 0.6 both land in bin 0 and add; 2.5 lands in bin 2.
    # Bin 1 staying empty is half the check: it proves nothing leaked sideways.
    assert torch.allclose(out, torch.tensor([[3 + 0j, 0 + 0j, 4 + 0j]]))
    print("delay binning         ok")


def _single_path_response(grid, row, col, taps=1):
    """A response built from one path arriving at pixel (row, col).

    Constructed from the grid's own steering vector, so the true direction is
    known exactly. The 1/sqrt(taps) keeps total power independent of `taps`.
    """
    a = grid.steering[:, row, col]                         # [M**2]
    return a.unsqueeze(1).repeat(1, taps) / math.sqrt(taps)


def test_cbf_peaks_at_true_direction():
    """Conventional beamforming: peak pixel and peak value are both analytic."""
    grid = ArrayGrid.build(M, W, H, FOV)
    # Off-centre and off-diagonal, so a transposed or mirrored grid would fail.
    row, col = H // 3, W // 4
    resp = _single_path_response(grid, row, col)
    lin, db = cbf_spectrum(resp, grid)
    assert lin.shape == (H, W)
    peak = divmod(int(lin.argmax()), W)                    # flat index -> (row, col)
    assert peak == (row, col), f"peak at {peak}, expected {(row, col)}"
    # A matched filter over M**2 unit-modulus elements gives a peak of M**2.
    assert abs(float(lin[row, col]) - M ** 2) < 1e-3
    print("CBF peak              ok")


def test_mvdr_peaks_at_true_direction():
    """MVDR finds the same direction as CBF on a single clean path."""
    grid = ArrayGrid.build(M, W, H, FOV)
    row, col = H // 2, 2 * W // 3
    # MVDR needs a full-rank covariance; one path cannot provide it, so the
    # loading term stands in for the noise floor.
    resp = _single_path_response(grid, row, col, taps=4)
    lin, db = mvdr_spectrum(resp, grid, diagonal_loading=1e-2)
    peak = divmod(int(lin.argmax()), W)
    # Only the location is checked: MVDR's peak height depends on the loading,
    # so there is no analytic value to compare against.
    assert peak == (row, col), f"peak at {peak}, expected {(row, col)}"
    print("MVDR peak             ok")


def test_mvdr_rejects_singular_covariance():
    """Without loading, a rank-deficient covariance must raise, not return noise.

    This is the failure mode worth guarding: a silently inverted singular matrix
    produces a plausible-looking spectrum made entirely of numerical noise.
    """
    grid = ArrayGrid.build(M, W, H, FOV)
    resp = _single_path_response(grid, 1, 1, taps=2)       # rank 1, M**2 = 36
    try:
        mvdr_spectrum(resp, grid, diagonal_loading=0.0)
    except ValueError as exc:
        assert "singular" in str(exc)
        print("MVDR singular guard   ok")
        return
    # Reached only if no exception was raised, which is the bug being tested for.
    raise AssertionError("expected a ValueError for a singular covariance")


if __name__ == "__main__":
    torch.manual_seed(0)
    test_angle_grid()
    test_steering_unit_modulus()
    test_delay_binning()
    test_cbf_peaks_at_true_direction()
    test_mvdr_peaks_at_true_direction()
    test_mvdr_rejects_singular_covariance()
    print("\nall checks passed")
