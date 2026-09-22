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


def _two_tap(grid, row, col, second_sign):
    """Two equal paths from the same direction, one tap apart, given relative sign."""
    a = grid.steering[:, row, col]
    resp = torch.zeros(a.shape[0], 5, dtype=torch.complex64)
    resp[:, 0] = a
    resp[:, 3] = second_sign * a
    return resp


def test_cbf_variants_agree_on_one_path():
    """Criterion B: a single populated tap is where the two conventions coincide.

    This is what lets "fixed" be introduced without invalidating the analytic
    control: both must still peak at exactly M**2.
    """
    grid = ArrayGrid.build(M, W, H, FOV)
    row, col = H // 3, W // 4
    resp = _single_path_response(grid, row, col)
    for variant in ("asis", "fixed"):
        lin, _ = cbf_spectrum(resp, grid, variant=variant)
        peak = divmod(int(lin.argmax()), W)
        assert peak == (row, col), f"{variant}: peak at {peak}"
        assert abs(float(lin[row, col]) - M ** 2) < 1e-3, f"{variant}: {float(lin[row, col])}"
    print("CBF variants, one path  ok")


def test_cbf_variants_differ_on_two_taps():
    """Criteria C and D: the conventions diverge once energy spans taps.

    Under "asis" the answer depends only on the relative phase of two equal
    paths -- 0 when they oppose, 2 M**2 when they align -- while "fixed" gives
    the incoherent sum sqrt(2) M**2 in both cases.
    """
    grid = ArrayGrid.build(M, W, H, FOV)
    row, col = H // 3, W // 4
    want = math.sqrt(2) * M ** 2
    for sign, asis_want in ((-1.0, 0.0), (+1.0, 2.0 * M ** 2)):
        resp = _two_tap(grid, row, col, sign)
        lin_a, _ = cbf_spectrum(resp, grid, variant="asis")
        lin_f, _ = cbf_spectrum(resp, grid, variant="fixed")
        assert abs(float(lin_a[row, col]) - asis_want) < 1e-2, float(lin_a[row, col])
        assert abs(float(lin_f[row, col]) - want) < 1e-2, float(lin_f[row, col])
    print("CBF variants, two taps  ok")


def test_mvdr_fixed_is_snapshot_invariant():
    """Criterion E: the 1/L is what makes the covariance an average.

    Tiling the CIR along the tap axis leaves the sample covariance unchanged, so
    "fixed" must not move; "asis" scales with the tap count and shifts by
    10 log10(k). This is the per-view drift the fix removes.
    """
    grid = ArrayGrid.build(M, W, H, FOV)
    row, col = H // 2, 2 * W // 3
    resp = _single_path_response(grid, row, col, taps=8)
    # A little noise, so tiling is not a rank-1 degenerate case.
    resp = resp + 0.05 * torch.randn_like(resp)
    for k in (2, 3):
        tiled = resp.repeat(1, k)
        for variant, shift in (("fixed", 0.0), ("asis", 10 * math.log10(k))):
            _, db0 = mvdr_spectrum(resp, grid, diagonal_loading=1e-2, variant=variant)
            _, db1 = mvdr_spectrum(tiled, grid, diagonal_loading=1e-2, variant=variant)
            got = float((db1 - db0).mean())
            assert abs(got - shift) < 1e-2, f"{variant} k={k}: {got:+.4f}, want {shift:+.4f}"
    print("MVDR snapshot invariance ok")


def test_mpc_variant_is_power():
    """Criterion F: the projection family's weight is an amplitude, not a power.

    A unit path hides the difference (log 1 = 0), so the path here carries
    amplitude 2 and the two variants must differ by exactly 10 log10(2).
    """
    from rf_spectra import _path_arrays, equirect_splat, _splat_weight
    amp = torch.tensor([2.0])
    th, ph = torch.tensor([math.radians(80.0)]), torch.tensor([math.radians(30.0)])
    ti, pi = int(round(80 * 3)), int(round((-30 + 180) * 3))
    vals = {}
    for variant in ("asis", "fixed"):
        img = equirect_splat(th, ph, _splat_weight(amp, variant), 3, 3.0)[0]
        vals[variant] = 10 * math.log10(float(img[ti, pi]))
    got = vals["fixed"] - vals["asis"]
    assert abs(got - 10 * math.log10(2.0)) < 1e-2, f"{got:.4f}"
    print("MPC weight is a power   ok")


def test_mpc_fixed_matches_multichannel_power():
    """Criterion G: the decisive consistency check.

    multichannel_from_arrays already splats power. With variant="fixed" the MPC
    spectrum must therefore reproduce its power channel pixel for pixel; with
    the default it must not. If this fails, the two encodings in rf_spectra
    still disagree about what "power" means.
    """
    from rf_spectra import equirect_splat, multichannel_from_arrays, _splat_weight
    g = torch.Generator().manual_seed(3)
    n = 40
    amp = torch.rand(n, generator=g) * 3 + 0.2
    tau = torch.rand(n, generator=g) * 50e-9
    th_r = torch.rand(n, generator=g) * math.pi
    ph_r = (torch.rand(n, generator=g) * 2 - 1) * math.pi
    th_t = torch.rand(n, generator=g) * math.pi
    ph_t = (torch.rand(n, generator=g) * 2 - 1) * math.pi

    multi = multichannel_from_arrays(amp, tau, th_r, ph_r, th_t, ph_t, scale=3, sigma=3.0)
    lit = multi[0] > -150.0
    assert int(lit.sum()) > 100, "the synthetic paths must light enough pixels to compare"

    for variant, should_match in (("fixed", True), ("asis", False)):
        img = equirect_splat(th_r, ph_r, _splat_weight(amp, variant), 3, 3.0)[0]
        db = 10 * torch.log10(img.clamp_min(1e-300))
        gap = float((db[lit] - multi[0][lit]).abs().max())
        if should_match:
            assert gap < 1e-2, f"fixed disagrees with the power channel by {gap:.4f} dB"
        else:
            assert gap > 1.0, f"asis unexpectedly agrees (gap {gap:.4f} dB)"
    print("MPC == multichannel     ok")


if __name__ == "__main__":
    torch.manual_seed(0)
    test_angle_grid()
    test_steering_unit_modulus()
    test_delay_binning()
    test_cbf_peaks_at_true_direction()
    test_mvdr_peaks_at_true_direction()
    test_mvdr_rejects_singular_covariance()
    test_cbf_variants_agree_on_one_path()
    test_cbf_variants_differ_on_two_taps()
    test_mvdr_fixed_is_snapshot_invariant()
    test_mpc_variant_is_power()
    test_mpc_fixed_matches_multichannel_power()
    print("\nall checks passed")
