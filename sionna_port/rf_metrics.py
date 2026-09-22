"""Channel metrics for a solved set of paths, and spectrum statistics.

RF-3DGS uses Sionna for one thing: solving paths and taking their CIR. Its
tutorial imports `cir_to_ofdm_channel`, `OFDMChannel`, `PUSCHConfig`,
`compute_ber`, `KBestDetector` and `StreamManagement` and never calls any of
them -- each name appears exactly once in the notebook, on the import line. The
evaluation that follows is therefore entirely image-domain: PSNR and SSIM between
colourmapped spectra.

This module computes what the channel itself says, so a run can be judged on
delay spread, coherence bandwidth, K-factor, array gain and capacity rather than
on how two PNGs compare. `paths.cfr()` in particular is a Sionna call the
original pipeline never makes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import numpy as np

C0 = 299_792_458.0


@dataclass
class ChannelMetrics:
    """One Tx-Rx pair's channel summary. Units are in the field names."""
    num_paths: int
    total_power_dbm: float
    strongest_path_dbm: float
    mean_excess_delay_ns: float
    rms_delay_spread_ns: float
    coherence_bandwidth_mhz: float
    k_factor_db: float
    aoa_azimuth_spread_deg: float
    aoa_zenith_spread_deg: float
    aod_azimuth_spread_deg: float
    array_gain_db: float
    capacity_bps_hz: float
    snr_db: float

    def as_dict(self):
        return asdict(self)


def _power_weighted_circular_spread(angles_rad: np.ndarray,
                                    weights: np.ndarray) -> float:
    """Circular standard deviation in degrees, weighted by path power.

    Circular, not linear: azimuth wraps, so a plain std would report a huge
    spread for paths clustered around +-180 degrees. The resultant length r of
    the unit vectors maps to a spread through sqrt(-2 ln r), which is 0 for
    perfectly aligned paths and grows without bound as they cancel.
    """
    if weights.sum() <= 0:
        return float("nan")
    w = weights / weights.sum()
    r = np.abs((w * np.exp(1j * angles_rad)).sum())
    # Clamped so log() gets neither 0 nor a value above 1 from rounding.
    r = min(max(r, 1e-12), 1.0)
    return math.degrees(math.sqrt(-2.0 * math.log(r)))


def _power_weighted_spread(values: np.ndarray, weights: np.ndarray) -> float:
    """Ordinary power-weighted standard deviation, for quantities that do not wrap."""
    if weights.sum() <= 0:
        return float("nan")
    w = weights / weights.sum()
    mean = float((w * values).sum())
    # max(..., 0) guards the tiny negative that E[x^2] - E[x]^2 can produce.
    return float(math.sqrt(max((w * (values - mean) ** 2).sum(), 0.0)))


def channel_metrics(paths, noise_figure_db: float = 7.0,
                    bandwidth_hz: float = 400e6,
                    tx_power_dbm: float = 30.0) -> ChannelMetrics:
    """Summarise one Tx-Rx pair from a Sionna Paths object.

    The defaults describe a plausible 60 GHz link: 400 MHz of bandwidth and a
    7 dB noise figure. They only scale SNR and capacity; every other quantity is
    a property of the channel alone.
    """
    a, tau = paths.cir(normalize_delays=False, out_type="numpy")
    # a:   [num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, num_time_steps]
    # One receiver, one transmitter, one time step; the antenna and path axes stay.
    a = np.asarray(a)[0, :, 0, 0, :, 0]                  # [ant, path]
    tau = np.asarray(tau).reshape(-1)                    # [path]

    # Padding slots carry inf or negative delays and must be dropped before any
    # weighted statistic is taken.
    finite = np.isfinite(tau) & (tau >= 0)
    a, tau = a[:, finite], tau[finite]
    if tau.size == 0:
        raise ValueError("no valid paths")

    # Per-path power, summed over array elements.
    path_power = (np.abs(a) ** 2).sum(axis=0)
    total_power = float(path_power.sum())
    tau_ns = tau / 1e-9

    # First and second power-weighted moments of the delay profile.
    mean_delay = float((path_power * tau_ns).sum() / total_power)
    rms_delay = float(math.sqrt(
        max((path_power * (tau_ns - mean_delay) ** 2).sum() / total_power, 0.0)))
    # The usual 0.5-correlation rule of thumb.
    # An impulsive channel (zero spread) gives infinite coherence bandwidth,
    # which is correct rather than an error.
    coherence_bw_mhz = 1.0 / (5.0 * rms_delay * 1e-9) / 1e6 if rms_delay > 0 else float("inf")

    # Rician K: strongest path against everything else.
    strongest = float(path_power.max())
    k_factor = strongest / max(total_power - strongest, 1e-30)

    theta_r = np.asarray(paths.theta_r).reshape(-1)[finite]
    phi_r = np.asarray(paths.phi_r).reshape(-1)[finite]
    phi_t = np.asarray(paths.phi_t).reshape(-1)[finite]

    # How beamformable the channel is: the gain a single fixed weight vector
    # (all-ones, i.e. broadside) achieves over incoherent summation. It reaches
    # 10*log10(M^2) for a plane wave that arrives in phase at every element, and
    # falls towards 0 dB as the field becomes diffuse and the element phases
    # decorrelate. Comparing MRC against a per-element average instead would be
    # a tautology -- that ratio is the element count by construction, whatever
    # the channel does.
    h = a.sum(axis=1)                                    # narrowband per element
    n_ant = h.size
    incoherent = float((np.abs(h) ** 2).sum())           # sum of per-element powers
    coherent = float(abs(h.sum()) ** 2)                  # power of the summed field
    array_gain_db = 10 * math.log10(max(coherent / max(incoherent, 1e-30), 1e-30))

    # Link budget: path gain -> SNR -> SIMO capacity with MRC.
    rx_power_dbm = tx_power_dbm + 10 * math.log10(max(total_power, 1e-30))
    noise_dbm = -174.0 + 10 * math.log10(bandwidth_hz) + noise_figure_db
    snr_db = rx_power_dbm - noise_dbm
    capacity = math.log2(1.0 + 10 ** (snr_db / 10.0))

    return ChannelMetrics(
        num_paths=int(tau.size),
        total_power_dbm=10 * math.log10(max(total_power, 1e-30)),
        strongest_path_dbm=10 * math.log10(max(strongest, 1e-30)),
        mean_excess_delay_ns=mean_delay,
        rms_delay_spread_ns=rms_delay,
        coherence_bandwidth_mhz=coherence_bw_mhz,
        k_factor_db=10 * math.log10(max(k_factor, 1e-30)),
        # Azimuths wrap and use the circular measure; zenith does not.
        aoa_azimuth_spread_deg=_power_weighted_circular_spread(phi_r, path_power),
        aoa_zenith_spread_deg=_power_weighted_spread(np.degrees(theta_r), path_power),
        aod_azimuth_spread_deg=_power_weighted_circular_spread(phi_t, path_power),
        array_gain_db=array_gain_db,
        capacity_bps_hz=capacity,
        snr_db=snr_db,
    )


def frequency_response(paths, centre_hz: float, bandwidth_hz: float,
                       num_subcarriers: int = 128):
    """Per-subcarrier channel via `paths.cfr()`, which the tutorial never calls.

    Returns (frequencies_hz, |H| per subcarrier summed over elements, in dB).
    Frequency selectivity is invisible in an angular power image, so this is the
    axis the image-domain evaluation cannot see at all.
    """
    import drjit as dr
    import mitsuba as mi

    # Offsets centred on zero, so the subcarrier grid straddles the carrier.
    offsets = (np.arange(num_subcarriers) - num_subcarriers // 2) \
        * (bandwidth_hz / num_subcarriers)
    freqs = mi.Float(offsets.astype(np.float32))
    h = paths.cfr(frequencies=freqs, normalize_delays=False, out_type="numpy")
    # The leading axes (rx, rx_ant, tx, tx_ant) and any trailing time axis vary
    # with how the solve was configured; only the subcarrier axis is fixed, so
    # fold everything else into one and sum power over it.
    h = np.asarray(h)
    # Found by length rather than position, which is what makes this robust to
    # the axis layout changing with the solve configuration.
    axis = next(i for i, n in enumerate(h.shape) if n == num_subcarriers)
    h = np.moveaxis(h, axis, -1).reshape(-1, num_subcarriers)
    mag = np.sqrt((np.abs(h) ** 2).sum(axis=0))
    return centre_hz + offsets, 20 * np.log10(np.maximum(mag, 1e-30))


def spectrum_stats(spectra_db: np.ndarray, levels: int = 256) -> dict:
    """What the 8-bit colourmap costs, measured rather than assumed.

    Simulates the global-normalisation encoding the released dataset uses:
    the full dB range maps onto `levels` steps, so a view occupying a small part
    of that range keeps only a fraction of the levels. The per-view span figures
    are what quantify that loss.
    """
    lo, hi = float(spectra_db.min()), float(spectra_db.max())
    span = hi - lo
    step = span / levels
    # Round-trip through the quantiser to measure the error directly rather than
    # assuming it is uniform.
    quantised = np.round((spectra_db - lo) / max(span, 1e-12) * (levels - 1))
    recovered = quantised / (levels - 1) * span + lo
    err = np.abs(recovered - spectra_db)
    # Per-view dynamic range, against the global range the encoding uses.
    per_view_span = spectra_db.reshape(spectra_db.shape[0], -1)
    per_view_span = per_view_span.max(axis=1) - per_view_span.min(axis=1)
    return {
        "num_views": int(spectra_db.shape[0]),
        "global_min_db": lo,
        "global_max_db": hi,
        "global_span_db": span,
        "quantisation_step_db": step,
        "quantisation_rmse_db": float(np.sqrt((err ** 2).mean())),
        "per_view_span_db_min": float(per_view_span.min()),
        "per_view_span_db_max": float(per_view_span.max()),
        "per_view_span_db_mean": float(per_view_span.mean()),
        # The headline number: 0.25 here means the average view uses a quarter
        # of the 256 levels, i.e. effectively 6 bits rather than 8.
        "fraction_of_range_used_mean": float((per_view_span / max(span, 1e-12)).mean()),
    }
