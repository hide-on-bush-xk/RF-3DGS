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
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix

class Camera(nn.Module):
    """One training or test view: its pose, intrinsics, and its ground-truth image.

    Subclasses nn.Module only for convenience (device handling); it holds no
    learnable parameters. The image is kept resident on `data_device` for the whole
    run, so the training set's total size is what drives VRAM at start-up.

    All four matrices below are stored transposed relative to how
    utils/graphics_utils.py builds them, i.e. in row-vector form (p @ M), because
    that is what the CUDA rasteriser expects.
    """
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask,
                 image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda"
                 ):
        super(Camera, self).__init__()

        self.uid = uid                # index within this Scene's camera list
        self.colmap_id = colmap_id    # original id from the dataset (COLMAP / Blender)
        self.R = R                    # camera-to-world rotation, 3x3
        self.T = T                    # world-to-camera translation, 3
        self.FoVx = FoVx              # horizontal field of view, radians
        self.FoVy = FoVy              # vertical field of view, radians
        self.image_name = image_name

        # data_device is usually "cuda". Setting it to "cpu" trades speed for VRAM
        # on large datasets, since images then have to be moved per iteration.
        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        # Ground truth, CHW in [0, 1]. The clamp is defensive: anything outside the
        # range would break the PSNR assumption in utils/image_utils.py.
        self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        if gt_alpha_mask is not None:
            # Composite over black: masked-out pixels become 0 and the loss then
            # pushes the Gaussians to leave them empty.
            self.original_image *= gt_alpha_mask.to(self.data_device)
        else:
            # No mask: this multiply by ones is a no-op that still allocates a full
            # H x W tensor per camera. Harmless, but it is pure overhead.
            self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)

        # Hard-coded clipping planes. Anything nearer than 1 cm or further than
        # 100 m is clipped, which matters for scenes larger than that -- worth
        # checking before blaming the model for missing far geometry.
        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans    # optional recentring of the camera rig, see getWorld2View2
        self.scale = scale    # optional rescaling of the camera rig

        # world -> camera, transposed into row-vector form and moved to GPU.
        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        # camera -> clip, same convention.
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        # world -> clip in one matrix. Row-vector order means world_view comes first.
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        # Camera position in world space. Row 3 (not column 3) holds the translation
        # because the matrix is transposed; the rasteriser needs this for SH directions.
        self.camera_center = self.world_view_transform.inverse()[3, :3]

class MiniCam:
    """A camera defined directly by its matrices, with no image attached.

    Used by the interactive viewer path (gaussian_renderer/network_gui.py), where
    poses arrive over a socket rather than from a dataset and there is no ground
    truth to compare against.
    """
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        # Same camera-centre extraction as Camera above, written out inline.
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]
