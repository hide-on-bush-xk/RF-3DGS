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

# Turns the lightweight CameraInfo records produced by the dataset readers into
# real Camera objects with their images loaded onto the GPU.

from scene.cameras import Camera
import numpy as np
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal

WARNED = False     # module-level latch so the downscaling notice prints only once

def loadCam(args, id, cam_info, resolution_scale):
    """Build one Camera from a CameraInfo, resizing its image per the resolution args.

    `args.resolution` is interpreted two ways:
      1, 2, 4, 8  -> divide the original size by that integer factor
      -1          -> keep the original size, except images wider than 1600 px which
                     are capped at 1600 (this is the default, and the reason a
                     high-resolution dataset may train at a size you did not pick)
      any other   -> treat it as the target *width* in pixels
    `resolution_scale` multiplies on top, which is how Scene supports training at
    several resolutions at once.
    """
    orig_w, orig_h = cam_info.image.size      # PIL gives (width, height)

    if args.resolution in [1, 2, 4, 8]:
        # Integer downscale factor, combined with the multi-resolution scale.
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600     # shrink just enough to hit 1600 px wide
            else:
                global_down = 1                 # small enough already: no change
        else:
            global_down = orig_w / args.resolution   # args.resolution is a target width

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    # CHW float tensor in [0, 1]; may carry a 4th alpha channel for RGBA sources.
    resized_image_rgb = PILtoTorch(cam_info.image, resolution)

    gt_image = resized_image_rgb[:3, ...]     # drop any alpha; keep RGB
    loaded_mask = None

    # NOTE (upstream 3DGS quirk, left as-is): PILtoTorch returns CHW, so the channel
    # count is shape[0] and shape[1] is the image height. This test therefore asks
    # whether the image is 4 pixels tall, not whether it has an alpha channel, and
    # is false for any real image. In practice alpha masks are never picked up here;
    # Camera falls back to multiplying by ones. Change to shape[0] if you need them.
    if resized_image_rgb.shape[1] == 4:
        loaded_mask = resized_image_rgb[3:4, ...]

    return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T,
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY,
                  image=gt_image, gt_alpha_mask=loaded_mask,
                  image_name=cam_info.image_name, uid=id, data_device=args.data_device)

def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    """Load every CameraInfo into a Camera, at one resolution scale.

    This is where the whole training set is pulled into GPU memory (Camera holds
    original_image on args.data_device), so it dominates start-up time and VRAM.
    `id` is the list index and becomes Camera.uid; cam_info.uid is kept separately
    as colmap_id.
    """
    camera_list = []

    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, c, resolution_scale))

    return camera_list

def camera_to_JSON(id, camera : Camera):
    """Serialise one camera into the dict written to <model_path>/cameras.json.

    That file is what the SIBR viewer reads to reconstruct the camera set, which is
    why it survives independently of the dataset: the viewer can run from the model
    directory alone, even if the original source_path is gone.

    WARNING: the `camera : Camera` annotation is wrong and Python does not enforce
    it. Scene.__init__ calls this with the CameraInfo records straight from the
    dataset readers, and the .width / .height reads below only work because
    CameraInfo has those fields. A real Camera exposes image_width / image_height
    instead, so passing one here raises AttributeError.
    """
    # Rebuild world-to-camera exactly as getWorld2View does (R stored camera-to-world).
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    # Invert to camera-to-world. Despite the variable name, W2C now holds C2W:
    # column 3 is the camera position in world space and the 3x3 block is its
    # orientation, which is what the viewer wants.
    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]   # ndarray rows -> plain lists
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        # The viewer wants focal lengths in pixels, not FoV angles.
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry
