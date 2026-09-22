#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp

def l1_loss(network_output, gt):
    """Mean absolute error over every element. Scalar, differentiable.

    This is the main photometric term of the 3DGS loss; train.py combines it with
    (1 - SSIM) using the weight opt.lambda_dssim.
    """
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    """Mean squared error over every element. Provided for experiments; the default
    training objective in train.py uses l1_loss instead."""
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    """1-D Gaussian kernel of length `window_size`, normalised to sum to 1.

    The kernel is centred on window_size // 2, so an odd window_size (11 here)
    puts the peak exactly on the middle tap.
    """
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    # Normalising makes the convolution below a weighted *average*, so mu1/mu2 are
    # local means rather than local sums.
    return gauss / gauss.sum()

def create_window(window_size, channel):
    """Build the separable Gaussian window used by SSIM, shaped for grouped conv2d.

    Returns a [channel, 1, window_size, window_size] tensor: one identical kernel
    per channel, which is what F.conv2d(groups=channel) expects so that channels
    are filtered independently and never mixed.
    """
    # sigma = 1.5 with an 11-tap window is the standard choice from the SSIM paper.
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)                      # [W, 1]
    # Outer product of the 1-D kernel with itself gives the 2-D separable kernel.
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)  # [1, 1, W, W]
    # expand() only broadcasts (no copy); .contiguous() materialises it because
    # conv2d needs a real, densely laid out weight tensor.
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    """Structural similarity between two images, in [0, 1] (higher is better).

    Inputs are CHW or NCHW float tensors with values in [0, 1]. Note the window is
    rebuilt on every call, which costs a small allocation per training iteration.
    """
    channel = img1.size(-3)               # channel count, works for both CHW and NCHW
    window = create_window(window_size, channel)

    # Move the freshly built (CPU) window to img1's device and dtype so conv2d
    # does not fail on a device/dtype mismatch.
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    """Core SSIM computation once the Gaussian window is prepared.

    Every statistic below is *local*: a Gaussian-weighted average over the 11x11
    neighbourhood of each pixel, obtained by convolving with `window`.
    """
    # Local means of each image.
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    # Local variances and covariance via E[x^2] - E[x]^2 and E[xy] - E[x]E[y].
    # This shortcut can go slightly negative from floating-point cancellation;
    # the +C2 terms below keep the result stable anyway.
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    # Stabilising constants from the SSIM paper: C = (K * L)^2 with K1 = 0.01,
    # K2 = 0.03 and dynamic range L = 1. They assume inputs are in [0, 1];
    # unnormalised 0-255 images would need L = 255 and are silently wrong here.
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    # SSIM = (luminance term) * (contrast-structure term), evaluated per pixel.
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()            # one scalar over the whole image
    else:
        # Collapse channel, height and width one at a time, keeping the batch dim:
        # yields one SSIM value per image in the batch.
        return ssim_map.mean(1).mean(1).mean(1)
