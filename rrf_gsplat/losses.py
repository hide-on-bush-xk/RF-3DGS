"""The photometric losses train_rrf.py uses: INRIA's l1_loss and ssim (original_rf3dgs/utils/loss_utils.py),
copied verbatim in their arithmetic so that the current pipeline does not import the original code base.

Note (2026-09-24): with PyTorch's default torch.backends.cudnn.allow_tf32 = True, the SSIM's convolutions run in
TF32 on Ampere GPUs, and its variances (E[x^2] - E[x]^2) lose so much precision that a quarter of our pixels come
out above SSIM 1; train_rrf.py --cudnn-tf32 off computes them in full fp32.

Copyright of the SSIM code: (C) 2023, Inria, GRAPHDECO research group (see original_rf3dgs/LICENSE.md terms).
"""

from __future__ import annotations

from math import exp

import torch
import torch.nn.functional as F


def l1_loss(network_output, gt):
    """Mean absolute error over every element."""
    return torch.abs((network_output - gt)).mean()


def gaussian(window_size, sigma):
    """1-D Gaussian kernel of length window_size, normalised to sum to 1."""
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    """The separable Gaussian window for grouped conv2d: [channel, 1, window_size, window_size]."""
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    return _2D_window.expand(channel, 1, window_size, window_size).contiguous()


def ssim(img1, img2, window_size=11, size_average=True):
    """Structural similarity of CHW / NCHW images in [0, 1] (higher is better)."""
    channel = img1.size(-3)
    window = create_window(window_size, channel)
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    return _ssim(img1, img2, window, window_size, channel, size_average)


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    if size_average:
        return ssim_map.mean()
    return ssim_map.mean(1).mean(1).mean(1)
