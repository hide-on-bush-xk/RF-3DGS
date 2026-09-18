"""Array signal processing for RF-3DGS spectra, in PyTorch.

Ported from the project's Sionna simulation tutorial, which was written against
Sionna 0.19 and TensorFlow. The maths is unchanged; the tensors are torch and the
per-view work that used to be redone for every receiver pose is hoisted into a
cache, because the angle grid is fixed by the camera model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

__all__ = [
    "compute_angle_matrices",
    "steering_vector",
    "array_manifold_vector",
    "merge_paths_to_time_grid",
    "paths_to_response",
    "cbf_spectrum",
    "mvdr_spectrum",
    "ArrayGrid",
]


def compute_angle_matrices(width: int, height: int, fov_deg: float,
                           device=None, dtype=torch.float32):
    """Zenith/azimuth of every pixel of a pinhole view, following the tutorial.

    Returns (theta, phi), each [height, width], in radians.
    """
    focal = width / (2.0 * math.tan(math.radians(fov_deg) / 2.0))
    x = torch.linspace(-width / 2, width / 2, width, device=device, dtype=dtype)
    y = torch.linspace(height / 2, -height / 2, height, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(y, x, indexing="ij")

    dirs = torch.stack([xx / focal, yy / focal, torch.ones_like(xx)], dim=-1)
    dirs = dirs / dirs.norm(dim=-1, keepdim=True)

    phi = torch.atan2(-dirs[..., 0], dirs[..., 2])
    theta = math.pi / 2 - torch.asin(dirs[..., 1])
    return theta, phi


def _element_offsets(M: int, device=None, dtype=torch.float32):
    """Element positions of an M x M UPA in wavelengths, tutorial ordering."""
    values = 0.25 + 0.5 * np.arange(M // 2)
    y_i = np.concatenate((-values[::-1], values))
    z_i = np.concatenate((values[::-1], -values))
    y_grid, z_grid = np.meshgrid(y_i, z_i)
    y_flat = y_grid.T.reshape(M ** 2)
    z_flat = z_grid.T.reshape(M ** 2)
    return (torch.as_tensor(y_flat, device=device, dtype=dtype),
            torch.as_tensor(z_flat, device=device, dtype=dtype))


def steering_vector(M: int, theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    """Unit-gain steering vectors, [M**2, *theta.shape], complex."""
    y_i, z_i = _element_offsets(M, device=theta.device, dtype=theta.dtype)
    v = torch.sin(theta) * torch.sin(phi)
    w = torch.cos(theta)

    shape = (M ** 2,) + (1,) * v.dim()
    phase = 2 * math.pi * (y_i.reshape(shape) * v.unsqueeze(0)
                           + z_i.reshape(shape) * w.unsqueeze(0))
    return torch.exp(1j * phase.to(torch.complex64 if theta.dtype == torch.float32
                                   else torch.complex128))


def array_manifold_vector(M: int, theta: torch.Tensor, phi: torch.Tensor,
                          element_gain: torch.Tensor | None = None) -> torch.Tensor:
    """Steering vectors weighted by the element pattern, [M**2, *theta.shape].

    `element_gain` is the complex co-polar element response on the same grid
    (c_theta in the tutorial). Pass None for an isotropic element.
    """
    a = steering_vector(M, theta, phi)
    if element_gain is None:
        return a
    return element_gain.unsqueeze(0).to(a.dtype) * a


@dataclass
class ArrayGrid:
    """Everything about a view that does not depend on the receiver pose.

    The tutorial rebuilt these inside the per-view loop; for a fixed camera model
    they are constant across all 3200 views, so building them once is the single
    biggest speedup in this port.
    """

    M: int
    theta: torch.Tensor
    phi: torch.Tensor
    steering: torch.Tensor          # [M**2, H, W]
    manifold: torch.Tensor          # [M**2, H, W]

    @classmethod
    def build(cls, M: int, width: int, height: int, fov_deg: float,
              element_gain_fn=None, device=None) -> "ArrayGrid":
        theta, phi = compute_angle_matrices(width, height, fov_deg, device=device)
        gain = None if element_gain_fn is None else element_gain_fn(theta, phi)
        return cls(M=M, theta=theta, phi=phi,
                   steering=steering_vector(M, theta, phi),
                   manifold=array_manifold_vector(M, theta, phi, gain))


def merge_paths_to_time_grid(a: torch.Tensor, tau_ns: torch.Tensor,
                             time_grid: torch.Tensor) -> torch.Tensor:
    """Bin per-path coefficients onto a delay grid.

    a         [M**2, P] complex, per element and path
    tau_ns    [P] delays in ns, shared across elements
    time_grid [L] ascending bin edges in ns
    returns   [M**2, L] complex
    """
    idx = torch.searchsorted(time_grid, tau_ns, right=False) - 1
    idx = idx.clamp_(0, time_grid.numel() - 1)
    out = torch.zeros(a.shape[0], time_grid.numel(), dtype=a.dtype, device=a.device)
    out.index_add_(1, idx, a)
    return out


def paths_to_response(paths, time_interval_ns: float = 1.0,
                      device=None) -> torch.Tensor:
    """Per-element discrete baseband response, [M**2, L] complex.

    `paths` is a Sionna RT 2.x Paths object. Delays are *not* normalized, matching
    `paths.normalize_delays = False` in the 0.19 tutorial.
    """
    a, tau = paths.cir(normalize_delays=False, out_type="torch")
    # a:   [num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, num_time_steps]
    # tau: [num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths]
    a = a[0, :, 0, 0, :, 0]                      # [M**2, P]
    tau = tau[0, 0, 0, 0, :] if tau.dim() == 5 else tau.reshape(-1)
    if device is not None:
        a, tau = a.to(device), tau.to(device)

    finite = torch.isfinite(tau) & (tau >= 0)
    a, tau = a[:, finite], tau[finite]
    if tau.numel() == 0:
        raise ValueError("no valid paths for this receiver pose")

    tau_ns = tau / 1e-9
    max_spread = math.ceil(float(tau_ns.max().item()))
    time_grid = torch.arange(0, max(max_spread, 1), time_interval_ns,
                             device=a.device, dtype=tau_ns.dtype)
    return merge_paths_to_time_grid(a, tau_ns, time_grid)


def cbf_spectrum(response: torch.Tensor, grid: ArrayGrid):
    """Conventional (Bartlett) beamforming spectrum.

    response [M**2, L] from `paths_to_response`
    returns (linear [H, W], dB [H, W]), both real
    """
    amp = torch.einsum("ml,mhw->hw", response.conj(), grid.steering).abs()
    return amp, 20.0 * torch.log10(amp.clamp_min(torch.finfo(amp.dtype).tiny))


def mvdr_spectrum(response: torch.Tensor, grid: ArrayGrid,
                  diagonal_loading: float = 0.0):
    """MVDR (Capon) spectrum.

    The covariance needs at least M**2 independent delay taps to be invertible;
    the tutorial says as much in a comment and then does not check it. Here a
    rank-deficient covariance raises, and `diagonal_loading` (as a fraction of
    tr(R)/M**2) is the knob for regularising it instead.
    """
    m2, taps = response.shape
    if diagonal_loading <= 0 and taps < m2:
        # torch.linalg.inv does not raise on a rank-deficient matrix, it returns
        # inf/nan, so check the condition the tutorial only mentions in a comment.
        raise ValueError(
            f"covariance is singular: {taps} delay taps for {m2} elements. "
            f"Shorten time_interval_ns or set diagonal_loading."
        )

    R = response @ response.conj().transpose(0, 1)
    if diagonal_loading > 0:
        eps = diagonal_loading * torch.diagonal(R).real.mean()
        R = R + eps * torch.eye(m2, dtype=R.dtype, device=R.device)
    R_inv = torch.linalg.inv(R)
    if not torch.isfinite(R_inv).all():
        raise ValueError(
            "covariance inverse is not finite; the delay taps are linearly "
            "dependent. Set diagonal_loading to regularise it."
        )

    aH_Rinv = torch.einsum("mhw,mn->nhw", grid.manifold.conj(), R_inv)
    quad = torch.einsum("nhw,nhw->hw", aH_Rinv, grid.manifold).abs()
    p = 1.0 / quad.clamp_min(torch.finfo(quad.dtype).tiny)
    return p, 10.0 * torch.log10(p)


# --------------------------------------------------------------------------
# Projection spectra: MPC, Delay and AoD
#
# These are not beamformed. Each path is splatted onto an equirectangular grid
# at its angle of arrival with a Gaussian kernel, and Delay and AoD colour that
# splat by the path's delay or departure angle. The tutorial loops over every
# path in Python, which is minutes for 300k paths; this scatters them in one go.
# --------------------------------------------------------------------------

def _gaussian_kernel(kernel_size: int, sigma: float, device=None):
    size = int(kernel_size * sigma) | 1          # odd, as in the tutorial
    axis = torch.arange(size, device=device, dtype=torch.float32) - size // 2
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    k = torch.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    return k / k.sum(), size


def equirect_splat(theta_rad: torch.Tensor, phi_rad: torch.Tensor,
                   weights: torch.Tensor, scale: int = 3, sigma: float = 3.0,
                   kernel_size: int = 3) -> torch.Tensor:
    """Accumulate weighted Gaussian blobs on a [180*scale, 360*scale] grid.

    `weights` is [P] for a scalar spectrum or [C, P] for one channel per path
    attribute. Returns [C, 180*scale, 360*scale], indexed (theta, phi) with phi
    running +180 to -180 left to right, matching the tutorial's layout.
    """
    device = theta_rad.device
    if weights.dim() == 1:
        weights = weights.unsqueeze(0)
    channels = weights.shape[0]
    h, w = 180 * scale, 360 * scale

    kernel, size = _gaussian_kernel(kernel_size, sigma, device)
    half = size // 2

    theta_idx = (torch.rad2deg(theta_rad) * scale).round().long()
    phi_idx = ((-torch.rad2deg(phi_rad) + 180.0) * scale).round().long()

    offs = torch.arange(-half, half + 1, device=device)
    dy, dx = torch.meshgrid(offs, offs, indexing="ij")
    dy, dx = dy.reshape(-1), dx.reshape(-1)
    kflat = kernel.reshape(-1)

    ys = (theta_idx.unsqueeze(1) + dy.unsqueeze(0)).clamp_(0, h - 1)
    xs = (phi_idx.unsqueeze(1) + dx.unsqueeze(0)).clamp_(0, w - 1)
    flat = (ys * w + xs).reshape(-1)

    out = torch.zeros(channels, h * w, device=device, dtype=weights.dtype)
    for c in range(channels):
        contrib = (weights[c].unsqueeze(1) * kflat.unsqueeze(0)).reshape(-1)
        out[c].index_add_(0, flat, contrib)
    return out.reshape(channels, h, w)


def _path_arrays(paths, device=None):
    """(amplitude, delay, AoA, AoD) for the valid paths, as torch tensors."""
    a, tau = paths.cir(normalize_delays=False, out_type="torch")
    a = a[0, :, 0, 0, :, 0]                       # [ant, path]
    amp = a.abs().sum(dim=0) if a.dim() == 2 else a.abs()
    tau = tau.reshape(-1)

    def to_t(x):
        return torch.as_tensor(np.asarray(x).reshape(-1), dtype=torch.float32,
                               device=amp.device)

    theta_r, phi_r = to_t(paths.theta_r), to_t(paths.phi_r)
    theta_t, phi_t = to_t(paths.theta_t), to_t(paths.phi_t)
    keep = torch.isfinite(tau) & (tau >= 0) & (amp > 0)
    out = (amp[keep], tau[keep], theta_r[keep], phi_r[keep],
           theta_t[keep], phi_t[keep])
    if device is not None:
        out = tuple(t.to(device) for t in out)
    return out


def mpc_spectrum_equirect(paths, scale: int = 3, sigma: float = 3.0):
    """Per-path amplitude splatted at its angle of arrival, in dB."""
    amp, _, theta_r, phi_r, _, _ = _path_arrays(paths)
    img = equirect_splat(theta_r, phi_r, amp, scale, sigma)[0]
    nonzero = img > 0
    out = torch.full_like(img, float("nan"))
    out[nonzero] = 10 * torch.log10(img[nonzero])
    floor = out[nonzero].min() - 10 if nonzero.any() else torch.tensor(-200.0)
    return torch.nan_to_num(out, nan=float(floor))


def delay_spectrum_equirect(paths, scale: int = 3, sigma: float = 3.0):
    """Amplitude in green and blue, normalised delay in red, as the tutorial has it."""
    amp, tau, theta_r, phi_r, _, _ = _path_arrays(paths)
    lo, hi = tau.min(), tau.max()
    r_weight = amp * ((tau - lo) / (hi - lo).clamp_min(1e-30))
    stacked = torch.stack([r_weight, amp, amp])
    img = equirect_splat(theta_r, phi_r, stacked, scale, sigma)
    return _log_rgb(img)


def aod_spectrum_equirect(paths, scale: int = 3, sigma: float = 3.0):
    """Departure zenith in red, departure azimuth in green, amplitude in blue."""
    amp, _, theta_r, phi_r, theta_t, phi_t = _path_arrays(paths)
    r_weight = amp * (torch.rad2deg(theta_t) / 180.0).clamp(0, 1)
    g_weight = amp * (1.0 - (torch.rad2deg(phi_t) + 180.0) / 360.0).clamp(0, 1)
    stacked = torch.stack([r_weight, g_weight, amp])
    img = equirect_splat(theta_r, phi_r, stacked, scale, sigma)
    return _log_rgb(img)


MULTI_CHANNELS = ("power_db", "aod_az_cos", "aod_az_sin", "aod_zen", "delay_ns")


def multichannel_spectrum_equirect(paths, scale: int = 3, sigma: float = 3.0,
                                   floor_db: float = -200.0, power_floor_db: float = -150.0):
    """One physical quantity per channel, splatted at the angle of arrival.

    Returns [5, 180*scale, 360*scale]:
      power_db     10 log10 of the splatted path power (the MPC spectrum)
      aod_az_cos,  power-weighted mean of (cos phi_t, sin phi_t): the circular
      aod_az_sin   mean of the departure azimuth as a vector, so the channel
                   has no seam at +-180 deg and decodes with atan2
      aod_zen      power-weighted mean departure zenith, theta / 180 in [0, 1]
      delay_ns     power-weighted mean delay in ns
    Pixels no path touches get floor_db and zeros. Unlike the tutorial's AoD
    and Delay pictures, no channel is a product of an angle and an amplitude:
    the amplitude has its own channel.
    """
    return multichannel_from_arrays(*_path_arrays(paths), scale=scale, sigma=sigma, floor_db=floor_db,
                                    power_floor_db=power_floor_db)


def multichannel_from_arrays(amp, tau, theta_r, phi_r, theta_t, phi_t,
                             scale: int = 3, sigma: float = 3.0, floor_db: float = -200.0,
                             power_floor_db: float = -150.0):
    """The same, from per-path tensors (amplitude, delay [s], AoA, AoD in rad).

    Split out so a smoke test can feed one synthetic path and check that the
    channels decode to exactly that path. Zero paths is a valid input: every
    pixel gets floor_db and zeros.

    `power_floor_db` truncates the splat: a Gaussian kernel's tail carries
    10 log10 of a vanishing power (exp(-r^2/2 sigma^2) at 40 px is -386 dB),
    which is a numerical residue, not a path-loss value. Pixels whose
    splatted power falls under this floor are treated as untouched (floor_db,
    zero angles and delay), the way the paper truncates path loss at the
    noise floor of its training samples.
    """
    device = amp.device
    h, w_px = 180 * scale, 360 * scale
    out = torch.zeros(5, h, w_px, device=device, dtype=torch.float32)
    out[0] = floor_db
    if amp.numel() == 0:
        return out
    w = amp * amp
    stacked = torch.stack([w, w * torch.cos(phi_t), w * torch.sin(phi_t),
                           w * theta_t, w * tau * 1e9])
    img = equirect_splat(theta_r, phi_r, stacked, scale, sigma)
    power, cos_az, sin_az, zen, delay = img
    hit = power > 10.0 ** (power_floor_db / 10.0)
    p = power.clamp_min(1e-300)
    out[0] = torch.where(hit, 10 * torch.log10(p), torch.full_like(power, floor_db))
    out[1] = torch.where(hit, cos_az / p, torch.zeros_like(cos_az))
    out[2] = torch.where(hit, sin_az / p, torch.zeros_like(sin_az))
    out[3] = torch.where(hit, torch.rad2deg(zen / p) / 180.0, torch.zeros_like(zen))
    out[4] = torch.where(hit, delay / p, torch.zeros_like(delay))
    return out


def decode_azimuth_deg(cos_ch, sin_ch):
    """(cos, sin) channels -> azimuth in degrees, (-180, 180]."""
    return torch.rad2deg(torch.atan2(sin_ch, cos_ch))


def _log_rgb(img: torch.Tensor) -> torch.Tensor:
    """10log10 with the tutorial's +150 dB offset, clipped at zero."""
    out = torch.zeros_like(img)
    nz = img > 0
    out[nz] = 10 * torch.log10(img[nz]) + 150.0
    return out.clamp_min(0.0)


def equirect_to_perspective(equirect: torch.Tensor, width: int, height: int,
                            fov_deg: float, yaw_rad: float = 0.0) -> torch.Tensor:
    """Resample an equirectangular image through the same pinhole model.

    `equirect` is [H, W] or [C, H, W]. The pixel directions come from
    compute_angle_matrices, so the perspective views stay consistent with the
    beamformed spectra, which are evaluated on that same grid.
    """
    single = equirect.dim() == 2
    src = equirect.unsqueeze(0) if single else equirect
    device = src.device
    src_h, src_w = src.shape[-2:]

    theta, phi = compute_angle_matrices(width, height, fov_deg, device=device)
    phi = phi + yaw_rad

    # Same index convention as equirect_splat.
    y = (torch.rad2deg(theta) * (src_h / 180.0)).clamp(0, src_h - 1)
    x = ((-torch.rad2deg(phi) + 180.0) * (src_w / 360.0)) % src_w

    grid_y = (y / (src_h - 1)) * 2 - 1
    grid_x = (x / (src_w - 1)) * 2 - 1
    grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)
    out = torch.nn.functional.grid_sample(
        src.unsqueeze(0), grid, mode="bilinear", padding_mode="border",
        align_corners=True)[0]
    return out[0] if single else out


def tapering_matrix(M: int, tapering_level: float = 1.0, device=None):
    """Hann taper over the array, the window TCBF applies to CBF's weights."""
    x = torch.linspace(0, 1, M, device=device)
    window = 0.5 * (1 - torch.cos(2 * math.pi * x))
    return torch.outer(window, window).reshape(-1) * tapering_level


def tcbf_spectrum(response: torch.Tensor, grid: "ArrayGrid",
                  tapering_level: float = 1.0):
    """CBF with a Hann-tapered weight vector: lower sidelobes, wider main lobe."""
    taper = tapering_matrix(grid.M, tapering_level,
                            device=grid.steering.device).to(grid.steering.dtype)
    weights = grid.steering * taper.reshape(-1, 1, 1)
    amp = torch.einsum("ml,mhw->hw", response.conj(), weights).abs()
    return amp, 20.0 * torch.log10(amp.clamp_min(torch.finfo(amp.dtype).tiny))
