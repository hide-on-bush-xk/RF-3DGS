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

# The forward pass of 3DGS: project every Gaussian into the camera and alpha-blend
# them into an image. All the heavy lifting happens inside the CUDA extension
# `diff_gaussian_rasterization`; this file only marshals tensors into it.

import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!

    Args:
        viewpoint_camera: a Camera (or MiniCam) supplying the pose, FoV and image size.
        pc:               the GaussianModel holding every Gaussian's parameters.
        pipe:             PipelineParams; its flags pick between the CUDA and Python
                          paths for covariance and SH evaluation.
        bg_color:         [3] tensor used where nothing is rendered. During training
                          train.py may pass a random colour instead of a fixed one.
        scaling_modifier: multiplies every Gaussian's scale. 1.0 for training; the
                          viewer uses it to shrink splats and expose the geometry.
        override_color:   [N, 3] to bypass appearance entirely (used for debug views).

    Returns a dict with the image and the three by-products densification needs.
    """

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    #
    # The rasteriser writes the screen-space gradient of each Gaussian's centre into
    # this tensor's .grad. That gradient is the densification signal: large values
    # mean the Gaussian is under-fitting its screen region and should be split or
    # cloned. The trailing `+ 0` is deliberate, not dead code -- it turns the leaf
    # tensor into a non-leaf one so that retain_grad() below actually keeps .grad.
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        # Silently skipped under torch.no_grad() (render.py, evaluation), where
        # there is no graph to retain a gradient on.
        pass

    # Set up rasterization configuration
    # The CUDA kernel wants the tangent of the half-angles, not the FoV itself.
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,   # world -> camera (row-vector)
        projmatrix=viewpoint_camera.full_proj_transform,    # world -> clip (row-vector)
        # Only the bands unlocked so far are evaluated, so early training is
        # view-independent and detail is added as active_sh_degree grows.
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,              # needed for view directions
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz              # [N, 3] world-space centres
    means2D = screenspace_points      # gradient sink described above
    opacity = pc.get_opacity          # [N, 1], already through the sigmoid

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    #
    # Two equivalent routes to the same Sigma = R S S^T R^T. The CUDA route (the
    # default) is faster; the Python route exists for debugging and for the viewer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            # [N, 3, (deg+1)^2]: transpose so the SH coefficients are the last axis,
            # which is the layout eval_sh expects.
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            # Viewing direction per Gaussian: from the camera towards the Gaussian.
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            # +0.5 matches the RGB2SH convention; clamp_min keeps colours non-negative
            # (note it does not clamp the upper end, so values above 1 can pass through).
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            # Default: hand the raw coefficients over and let CUDA evaluate them.
            shs = pc.get_features
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    #
    # This single call does the whole forward pass: frustum culling, projection to
    # 2D covariances, tile binning, depth sorting and front-to-back alpha blending.
    # Exactly one of (scales, rotations) / cov3D_precomp must be set, and exactly
    # one of shs / colors_precomp.
    rendered_image, radii = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,              # [3, H, W] in roughly [0, 1]
            "viewspace_points": screenspace_points,# read .grad after backward()
            "visibility_filter" : radii > 0,       # [N] bool mask of what was drawn
            "radii": radii}                        # [N] on-screen radius in pixels
