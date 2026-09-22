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
import sys
from datetime import datetime
import numpy as np
import random

def inverse_sigmoid(x):
    """Logit function: the inverse of torch.sigmoid.

    GaussianModel stores opacity in this unconstrained "raw" space and applies a
    sigmoid to read it back, which keeps opacity in (0, 1) without clamping. This
    helper converts a desired opacity into the raw value to store.
    Undefined at x = 0 and x = 1 (gives -inf / +inf).
    """
    return torch.log(x/(1-x))

def PILtoTorch(pil_image, resolution):
    """Resize a PIL image and convert it to a float CHW tensor in [0, 1].

    `resolution` is a (width, height) pair, matching PIL's convention (note this is
    the opposite order from a tensor's shape). Grayscale images are given an
    explicit single channel so the output is always CHW.
    """
    resized_image_PIL = pil_image.resize(resolution)
    # np.array on a PIL image gives HWC (or HW for grayscale); /255 puts it in [0, 1].
    resized_image = torch.from_numpy(np.array(resized_image_PIL)) / 255.0
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)        # HWC -> CHW
    else:
        # Grayscale: HW -> HW1 -> 1HW, so callers can treat every image as CHW.
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)

def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input

    In 3DGS this schedules the *position* learning rate only (see
    GaussianModel.training_setup / update_learning_rate); all other parameter
    groups keep a constant rate.
    """

    def helper(step):
        # A parameter group with both endpoints at zero is treated as frozen.
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            # Warm-up multiplier rising smoothly from lr_delay_mult to 1.0 over the
            # first lr_delay_steps steps, then pinned at 1.0 by the clip.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)          # normalised progress in [0, 1]
        # Linear interpolation in log space == exponential decay in linear space.
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper

def strip_lowerdiag(L):
    """Pack the 6 unique entries of a batch of symmetric 3x3 matrices into [N, 6].

    A 3D covariance is symmetric, so only the upper triangle needs to be stored or
    passed to the rasteriser. Order is (xx, xy, xz, yy, yz, zz).
    """
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty

def strip_symmetric(sym):
    """Alias for strip_lowerdiag, named for the case where the input is known symmetric."""
    return strip_lowerdiag(sym)

def build_rotation(r):
    """Quaternions [N, 4] in (w, x, y, z) order -> rotation matrices [N, 3, 3].

    The quaternions are normalised here rather than being constrained during
    optimisation, so the optimiser is free to move them off the unit sphere.
    """
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    q = r / norm[:, None]    # unit quaternion; [:, None] broadcasts over the 4 components

    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    # Note the naming: `r` is rebound here from the input tensor to the scalar (w)
    # component. Everything below reads from `q`, so this shadowing is harmless.
    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    # Standard quaternion-to-matrix expansion.
    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R

def build_scaling_rotation(s, r):
    """Compose per-Gaussian scale [N, 3] and rotation quaternion [N, 4] into M = R @ S.

    The 3D covariance of a Gaussian is then Sigma = M @ M^T, which is how
    GaussianModel.build_covariance_from_scaling_rotation uses this. Factorising
    the covariance this way keeps it positive semi-definite for any parameter
    values the optimiser produces.
    """
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    # Diagonal scale matrix S: one axis length per Gaussian.
    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = s[:,2]

    L = R @ L     # rotate the scaled axes into world orientation
    return L

def safe_state(silent):
    """Process-wide setup run once at the start of train.py / render.py.

    Does three things: timestamps every printed line, seeds the RNGs, and pins the
    process to GPU 0. Note the seeding makes a run reproducible only up to CUDA's
    own non-determinism; it is not a guarantee of bit-identical results.
    """
    old_f = sys.stdout
    class F:
        """stdout wrapper that appends a [dd/mm HH:MM:SS] stamp to each finished line."""
        def __init__(self, silent):
            self.silent = silent

        def write(self, x):
            if not self.silent:
                # Only stamp when the chunk ends a line, so partial writes (tqdm's
                # in-place progress bar, for instance) are passed through untouched.
                if x.endswith("\n"):
                    old_f.write(x.replace("\n", " [{}]\n".format(str(datetime.now().strftime("%d/%m %H:%M:%S")))))
                else:
                    old_f.write(x)

        def flush(self):
            old_f.flush()

    sys.stdout = F(silent)

    # Fixed seeds for the three RNGs this codebase draws from. train.py's random
    # camera order and the densification noise both depend on these.
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    # Hard-coded to the first GPU: there is no flag to train on a different card,
    # so selecting one means setting CUDA_VISIBLE_DEVICES in the environment.
    torch.cuda.set_device(torch.device("cuda:0"))
