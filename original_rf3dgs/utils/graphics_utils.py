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

# Camera / projection maths shared by the dataset readers, the Camera objects and
# the rasteriser.
#
# Convention note, because it is the single easiest thing to get wrong here:
# the matrices *built* in this file use the textbook column-vector convention
# (M @ p). scene/cameras.py calls .transpose(0, 1) on them before storing, so the
# matrices that reach the CUDA rasteriser are row-vector (p @ M). That is also why
# Camera.camera_center reads row 3 rather than column 3.

import torch
import math
import numpy as np
from typing import NamedTuple

class BasicPointCloud(NamedTuple):
    """The minimal point cloud the scene readers hand to GaussianModel for init.

    points  : [N, 3] float  XYZ positions in world space
    colors  : [N, 3] float  RGB in [0, 1] (converted to SH DC terms downstream)
    normals : [N, 3] float  unused by 3DGS itself; kept so PLY round-trips cleanly
    """
    points : np.array
    colors : np.array
    normals : np.array

def geom_transform_points(points, transf_matrix):
    """Apply a 4x4 transform to [P, 3] points and return the [P, 3] result.

    Uses the row-vector convention (p @ M), matching the already-transposed
    matrices stored on Camera. Performs the perspective divide, so this is what
    you want for a full projection matrix, not just a rigid transform.
    """
    P, _ = points.shape
    # Promote to homogeneous coordinates: [x, y, z] -> [x, y, z, 1].
    ones = torch.ones(P, 1, dtype=points.dtype, device=points.device)
    points_hom = torch.cat([points, ones], dim=1)
    # unsqueeze(0) makes this a batched matmul of [1, P, 4] by [1, 4, 4].
    points_out = torch.matmul(points_hom, transf_matrix.unsqueeze(0))

    # Perspective divide by w. The epsilon guards points that land exactly on the
    # camera plane (w = 0), which would otherwise produce inf/NaN.
    denom = points_out[..., 3:] + 0.0000001
    return (points_out[..., :3] / denom).squeeze(dim=0)

def getWorld2View(R, t):
    """Build the 4x4 world-to-camera matrix from a COLMAP-style pose.

    R is stored throughout this codebase as the *camera-to-world* rotation, so it
    is transposed here to get world-to-camera; t is already the world-to-camera
    translation. Returned float32 because downstream torch tensors are float32.
    """
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()    # camera-to-world -> world-to-camera
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return np.float32(Rt)

def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    """Same as getWorld2View, plus an optional recentre/rescale of the camera rig.

    `translate` and `scale` act on the *camera centre* in world space, which is why
    the matrix is inverted to C2W, edited, and inverted back. Used to normalise a
    scene into a convenient coordinate range; the defaults make it a no-op.
    """
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    # Invert to camera-to-world so that column 3 is the camera centre in world space.
    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    # Shift then scale the centre. Note the order: translate is applied first and is
    # therefore also multiplied by `scale`.
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    # Back to world-to-camera with the edited centre.
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)

def getProjectionMatrix(znear, zfar, fovX, fovY):
    """Perspective projection matrix from near/far planes and the two field-of-view angles.

    Maps the view frustum into clip space with z in [0, 1] (Direct3D-style depth
    range), not the [-1, 1] that textbook OpenGL uses. Built column-vector style;
    scene/cameras.py transposes it before use.
    """
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    # Frustum extents on the near plane.
    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0    # +1 means the camera looks down +z here (not the OpenGL -z)

    # Scale x and y so the frustum maps to the [-1, 1] NDC square.
    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    # Off-centre terms: zero for the symmetric frustum above, kept for generality.
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    # Row 3 copies z into w, which is what makes the later divide a perspective divide.
    P[3, 2] = z_sign
    # Depth mapping: znear -> 0, zfar -> 1.
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

def fov2focal(fov, pixels):
    """Field of view (radians) -> focal length in pixels, for an image `pixels` wide/tall."""
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    """Focal length in pixels -> field of view in radians. Inverse of fov2focal."""
    return 2*math.atan(pixels/(2*focal))
