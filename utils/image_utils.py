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

def mse(img1, img2):
    """Mean squared error between two images, reported per leading-dimension slice.

    Inputs are the CHW tensors this codebase passes around (C=3), so dim 0 is the
    colour channel and the result is one MSE value per channel, shape [C, 1].
    Callers that want a single number take .mean() on the result.
    """
    # (img1 - img2) ** 2  -> squared error at every pixel, same shape as the inputs.
    # .view(img1.shape[0], -1) -> flatten everything after dim 0, giving [C, H*W].
    # .mean(1, keepdim=True)   -> average over the flattened pixels, giving [C, 1].
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)

def psnr(img1, img2):
    """Peak signal-to-noise ratio in dB, per leading-dimension slice (see mse above).

    Assumes pixel values live in [0, 1]: the peak value MAX_I is hard-coded to 1.0
    in the numerator below. Feeding unnormalised (0-255) images here silently
    produces wrong numbers rather than an error.
    """
    # Same per-channel MSE as mse(); recomputed locally instead of calling it.
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    # PSNR = 20 * log10(MAX_I / sqrt(MSE)), with MAX_I = 1.0 for [0, 1] images.
    # Identical images give MSE = 0 and therefore +inf here, which is expected.
    return 20 * torch.log10(1.0 / torch.sqrt(mse))
