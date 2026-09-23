"""Image-space residual networks and guide buffers: the DLSS-style pieces of rounds 39+.

Two users:
  label_sr.py     a learned upsampler for low-resolution labels (experiment A1)
  train_rrf.py    a deferred shading head on top of the rasterised field (experiment B): gsplat renders a base
                  image plus guide buffers, one small CNN shared by every view adds a bounded residual

The guide buffers follow DLSS Ray Reconstruction's input list (DLSS-RR Integration Guide, section 3.4: colour,
depth, normals, albedo / roughness, specular hit distance, camera matrices), translated to a radio field:
depth -> hit point; normals -> the alpha-composited shortest axis of the Gaussians; specular direction and hit
distance -> the mirror alignment and the range of the hit point to the transmitter; albedo / roughness (unknown
materials) -> a small learned per-Gaussian feature; camera matrices -> world ray direction and receiver position.

Every network here starts as the identity: the output layer is zero-initialised, so before the first step the
head's image equals the base image exactly (a control the smoke checks).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def pad2d(x, p, mode):
    """Pad [B, C, H, W] by p on every side. 'replicate': each image alone. 'strip': circular left-right (the four
    faces of a position laid side by side in azimuth order close into a ring), replicate top-bottom."""
    if p == 0:
        return x
    if mode == "strip":
        x = F.pad(x, (p, p, 0, 0), mode="circular")
        return F.pad(x, (0, 0, p, p), mode="replicate")
    return F.pad(x, (p, p, p, p), mode="replicate")


class ResidualCNN(nn.Module):
    """[B, c_in, H, W] -> residual [B, c_out, H, W]: two-level encoder-decoder, 3x3 convolutions, GELU.

    max_residual bounds the output softly (m * tanh(o / m)): the head can move a pixel by at most m, so it can
    correct the field but not paint a new one. None leaves it unbounded (the label upsampler, where the target
    is the truth itself).
    """

    def __init__(self, c_in, c_out, width=32, max_residual=None):
        super().__init__()
        self.enc1 = nn.Conv2d(c_in, width, 3)
        self.enc2 = nn.Conv2d(width, 2 * width, 3, stride=2)
        self.mid = nn.Conv2d(2 * width, 2 * width, 3)
        self.dec1 = nn.Conv2d(3 * width, width, 3)
        self.out = nn.Conv2d(width, c_out, 1)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)          # identity at the start
        self.max_residual = max_residual

    def forward(self, x, pad="replicate"):
        h1 = F.gelu(self.enc1(pad2d(x, 1, pad)))
        h2 = F.gelu(self.enc2(pad2d(h1, 1, pad)))
        h2 = F.gelu(self.mid(pad2d(h2, 1, pad)))
        u = F.interpolate(h2, size=h1.shape[-2:], mode="bilinear", align_corners=False)
        h = F.gelu(self.dec1(pad2d(torch.cat([u, h1], 1), 1, pad)))
        o = self.out(h)
        if self.max_residual is not None:
            o = self.max_residual * torch.tanh(o / self.max_residual)
        return o


def pixel_coords(h, w, device):
    """[2, H, W] in [-1, 1]: where a pixel sits in the face. The pinhole's angular pitch changes across it."""
    ys = torch.linspace(-1, 1, h, device=device); xs = torch.linspace(-1, 1, w, device=device)
    return torch.stack(torch.meshgrid(ys, xs, indexing="ij")[::-1])


def world_rays(viewmats, K, h, w):
    """Unit ray direction per pixel in world coordinates, [B, 3, H, W], and the camera centres [B, 3]."""
    dev = viewmats.device
    u = (torch.arange(w, device=dev, dtype=torch.float32) + 0.5 - K[0, 2]) / K[0, 0]
    v = (torch.arange(h, device=dev, dtype=torch.float32) + 0.5 - K[1, 2]) / K[1, 1]
    cam = torch.stack([u[None, :].expand(h, w), v[:, None].expand(h, w), torch.ones(h, w, device=dev)])  # [3,H,W]
    R = viewmats[:, :3, :3]                                            # world -> camera
    d = torch.einsum("bji,jhw->bihw", R, cam)                         # R^T cam
    centres = -torch.einsum("bji,bj->bi", R, viewmats[:, :3, 3])       # -R^T t
    return F.normalize(d, dim=1), centres, cam


def gaussian_normals(quats, log_scales):
    """The shortest axis of each Gaussian, [N, 3] (unit, sign arbitrary)."""
    q = F.normalize(quats, dim=-1)
    w, x, y, z = q.unbind(-1)
    R = torch.stack([
        torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
        torch.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
        torch.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1)], -2)   # [N,3,3]
    k = log_scales.argmin(dim=-1)
    return R[torch.arange(R.shape[0], device=R.device), :, k]           # column k = that axis in world


def geometry_guides(depth, normal, rays, cam_plane, centres, scene_scale):
    """Guide channels that need only the geometry: [B, 10, H, W].

    depth       expected camera-z depth [B, H, W] (metres), as log(1 + z)
    normal      alpha-composited normal [B, 3, H, W], renormalised and turned to face the camera
    rays        world ray directions [B, 3, H, W]
    centres     receiver positions [B, 3], broadcast, divided by scene_scale
    """
    B, _, h, w = rays.shape
    n = F.normalize(normal, dim=1)
    n = torch.where((n * rays).sum(1, keepdim=True) > 0, -n, n)       # facing the receiver
    pos = (centres / scene_scale)[:, :, None, None].expand(B, 3, h, w)
    return torch.cat([torch.log1p(depth.clamp_min(0))[:, None], n, rays, pos], 1)


def physics_guides(depth, normal, rays, cam_plane, centres, viewmats, tx, span_m):
    """Guide channels that also need the transmitter: [B, 5, H, W].

    los        cos of the angle between the pixel's ray and the direction to the transmitter (the direct path)
    mirror     how well the hit point mirrors the transmitter into the pixel: l . reflect(v) in [-1, 1]
    incidence  n . l, the transmitter's incidence at the hit point
    r_tx       log range hit point -> transmitter
    r_path     log of the one-bounce path length transmitter -> hit point -> receiver (the delay-depth range)
    """
    B, _, h, w = rays.shape
    R = viewmats[:, :3, :3]
    p_cam = cam_plane[None] * depth[:, None]                                        # [B,3,H,W] camera frame
    p = torch.einsum("bji,bjhw->bihw", R, p_cam) + centres[:, :, None, None]       # hit point, world
    n = F.normalize(normal, dim=1)
    n = torch.where((n * rays).sum(1, keepdim=True) > 0, -n, n)
    to_tx = tx[None, :, None, None] - p
    r_tx = to_tx.norm(dim=1, keepdim=True).clamp_min(1e-3); l = to_tx / r_tx
    v = -rays                                                                       # hit point -> receiver
    refl = 2 * (n * v).sum(1, keepdim=True) * n - v
    los_dir = F.normalize(tx[None] - centres, dim=-1)[:, :, None, None]
    los = (rays * los_dir).sum(1, keepdim=True)
    mirror = (l * refl).sum(1, keepdim=True)
    incidence = (n * l).sum(1, keepdim=True)
    r_path = r_tx + depth[:, None] * cam_plane.norm(dim=0)[None, None]
    return torch.cat([los, mirror, incidence, torch.log(r_tx / span_m), torch.log(r_path / span_m)], 1)
