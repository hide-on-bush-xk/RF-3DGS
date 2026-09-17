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
