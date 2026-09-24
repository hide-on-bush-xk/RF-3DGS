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

# Reads a dataset off disk into the CameraInfo / SceneInfo records that Scene
# consumes. Two formats are supported, dispatched through sceneLoadTypeCallbacks
# at the bottom of this file:
#
#   "Colmap"  -- a sparse/0 reconstruction plus an images folder
#   "Blender" -- NeRF-synthetic transforms_{train,test}.json
#
# The COLMAP reader here is NOT the upstream 3DGS one: it has been modified for
# RF-3DGS to take its train/test split from explicit index files. See the note on
# readColmapSceneInfo.

import os
import sys
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud

class CameraInfo(NamedTuple):
    """One view, before its image has been resized or moved to the GPU.

    The PIL image is held open (not decoded) until utils/camera_utils.loadCam turns
    this into a real Camera. Note this type -- not Camera -- is what
    camera_to_JSON receives, which is why it carries width/height.
    """
    uid: int
    R: np.array          # camera-to-world rotation (stored transposed, see below)
    T: np.array          # world-to-camera translation
    FovY: np.array
    FovX: np.array
    image: np.array      # actually a PIL.Image, despite the annotation
    image_path: str
    image_name: str      # filename without extension; the key the split files match on
    width: int
    height: int

class SceneInfo(NamedTuple):
    """Everything a reader produces: the seed point cloud, both camera lists,
    the scene normalisation, and where the point cloud lives on disk."""
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str

def getNerfppNorm(cam_info):
    """Derive the scene's centre and radius from how the cameras are spread out.

    The radius becomes Scene.cameras_extent, which sets the position learning rate
    and the densification size thresholds. Only *training* cameras are passed in,
    so the split affects this number.
    """
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        # Distance from the centroid to each camera; the furthest one defines the radius.
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        # Invert world-to-camera to read off the camera position in world space.
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1      # 10% margin so the scene is not exactly at the edge

    translate = -center          # would recentre the rig; currently unused downstream

    return {"translate": translate, "radius": radius}

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    """Join COLMAP's per-image extrinsics with per-camera intrinsics into CameraInfo.

    Only undistorted pinhole models are accepted -- run COLMAP's image_undistorter
    first if your reconstruction uses a distortion model.
    """
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        # In-place progress counter on one line.
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        # Several images can share one camera (one rig, many shots).
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        # R is stored transposed throughout this codebase; see the note in
        # readCamerasFromTransforms and utils/graphics_utils.py.
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            # One focal length for both axes; the two FoVs differ only via w/h.
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        # basename() discards any directory structure COLMAP recorded, so all
        # images must sit flat in images_folder.
        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)     # lazy: pixels are not decoded yet

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path):
    """Read a coloured point cloud into the BasicPointCloud that seeds the model."""
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    # Requires nx/ny/nz to be present; a PLY without normals raises here.
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    """Write xyz + rgb as a PLY, with zero normals to complete the standard layout.

    Used to cache COLMAP's points3D as a .ply the first time a scene is opened, and
    to persist the random cloud generated for Blender scenes.
    """
    # Define the dtype for the structured array
    # Positions float32, colours uint8 -- matching what fetchPly expects to read back.
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]

    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

# Kept for reference: the original upstream 3DGS reader, whose train/test split is
# "every llffhold-th image is a test view". The live version below replaces that
# rule with explicit index files.
# def readColmapSceneInfo(path, images, eval, llffhold=8):
#     try:
#         cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
#         cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
#         cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
#         cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
#     except:
#         cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
#         cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
#         cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
#         cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

#     reading_dir = "images" if images == None else images
#     cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir))
#     cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

#     if eval:
#         train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
#         test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
#     else:
#         train_cam_infos = cam_infos
#         test_cam_infos = []

#     nerf_normalization = getNerfppNorm(train_cam_infos)

#     ply_path = os.path.join(path, "sparse/0/points3D.ply")
#     bin_path = os.path.join(path, "sparse/0/points3D.bin")
#     txt_path = os.path.join(path, "sparse/0/points3D.txt")
#     if not os.path.exists(ply_path):
#         print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
#         try:
#             xyz, rgb, _ = read_points3D_binary(bin_path)
#         except:
#             xyz, rgb, _ = read_points3D_text(txt_path)
#         storePly(ply_path, xyz, rgb)
#     try:
#         pcd = fetchPly(ply_path)
#     except:
#         pcd = None

#     scene_info = SceneInfo(point_cloud=pcd,
#                            train_cameras=train_cam_infos,
#                            test_cameras=test_cam_infos,
#                            nerf_normalization=nerf_normalization,
#                            ply_path=ply_path)
#     return scene_info

def readColmapSceneInfo(path, images, eval, llffhold=8):
    """Read a COLMAP scene. RF-3DGS version: the split comes from index files.

    IMPORTANT, and the main divergence from upstream 3DGS:

      * The split is read from <path>/train_index.txt and <path>/test_index.txt,
        one image name per line. Both files are mandatory -- a COLMAP scene without
        them raises FileNotFoundError here.
      * `eval` and `llffhold` are accepted but no longer used on this path. Passing
        --eval does not change the split, and omitting it does not merge the test
        views back into training. The upstream every-8th-image behaviour is the
        commented-out block above.

    Any image present in the reconstruction but named in neither file is silently
    dropped from both lists.
    """
    try:
        # COLMAP writes either binary or text; try binary first and fall back.
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        # Bare except, so a genuine parse error in the .bin also lands here and is
        # reported as though the .txt were the problem.
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir))
    # Sorted by name so the ordering is deterministic across runs and machines.
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    # 加载 train_index.txt 和 test_index.txt
    # (Load the explicit train/test split files.)
    train_index_file = os.path.join(path, "train_index.txt")
    test_index_file = os.path.join(path, "test_index.txt")

    def load_indices(file_path):
        """One image name per line -> a set of extension-free base names."""
        with open(file_path, 'r') as f:
            indices = set()
            for line in f:
                filename = line.strip()
                # Remove the .png extension if it exists
                # split(".")[0] also truncates at the first dot, so a name like
                # "view.01.png" would reduce to "view".
                base_name = os.path.basename(filename).split(".")[0]
                indices.add(base_name)
            return indices

    # Load train and test indices, handling .png extension
    train_indices = load_indices(train_index_file)
    test_indices = load_indices(test_index_file)

    # Membership test against CameraInfo.image_name, which is likewise extension-free.
    train_cam_infos = [c for c in cam_infos if c.image_name in train_indices]
    print("train_cam_length: ", len(train_cam_infos))
    test_cam_infos = [c for c in cam_infos if c.image_name in test_indices]

    # 如果 eval 参数为 True，这部分代码可以忽略，直接使用上面加载的 train_cam_infos 和 test_cam_infos
    # (With the index files in use, the upstream eval-driven split is dead code.)
    #if eval:
    #    train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
    #    test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    #else:
    #    train_cam_infos = cam_infos
    #    test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    # COLMAP's points3D become the initial Gaussians. Converted to .ply once and
    # cached, so the second run of a scene skips this.
    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        # pcd = None only works when training resumes from a checkpoint; a fresh
        # run would then fail in create_from_pcd rather than here.
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


def readCamerasFromTransforms(path, transformsfile, white_background, extension=""):
    """Read one NeRF-synthetic transforms_*.json into CameraInfo records.

    Unlike the COLMAP path, images are decoded and composited here rather than
    later, because the alpha channel has to be flattened against the chosen
    background before the image can be treated as RGB.
    """
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        # One shared horizontal FoV for every frame in the file.
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            # `extension` is appended because file_path in these files is usually
            # extension-free ("./train/r_0").
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            # Alpha-composite onto the background: out = rgb*a + bg*(1-a). This is
            # why --white_background must match how the dataset was rendered.
            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.uint8), "RGB")

            # Vertical FoV is derived from the horizontal one plus the aspect ratio,
            # via the shared focal length. image.size is (width, height).
            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy
            FovX = fovx

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1]))

    return cam_infos

def readNerfSyntheticInfo(path, white_background, eval, extension=""):
    """Read a Blender / NeRF-synthetic scene.

    Here `eval` does still work: without it the test views are folded into the
    training set. Note the asymmetry with readColmapSceneInfo above, where the flag
    is ignored.
    """
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)

    if not eval:
        # No held-out split: train on everything.
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")

        # We create random points inside the bounds of the synthetic Blender scenes
        # Uniform in the cube [-1.3, 1.3]^3, which is where the standard NeRF
        # synthetic objects sit. A scene outside those bounds needs this changed.
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        # Written to disk so later runs reuse the same cloud; note the `pcd` built
        # on the line above is then discarded in favour of the fetchPly below.
        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

# Dispatch table used by Scene.__init__ once it has sniffed the dataset type.
sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo
}
