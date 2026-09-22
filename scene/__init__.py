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

# Owns everything about a scene except the Gaussians' parameters: the camera sets,
# the train/test split, and the input point cloud used to initialise the model.

import os
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON

class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
        """b
        :param path: Path to colmap scene main folder.

        Two modes:
          load_iteration is None  -> fresh training. The GaussianModel is seeded from
                                     the dataset's point cloud, and cameras.json /
                                     input.ply are written into model_path.
          load_iteration is set   -> evaluation. The Gaussians come from a saved .ply
                                     and nothing is written.

        Mutates `gaussians` in place; the object is passed in already constructed so
        that train.py holds a reference to it.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            # -1 is the conventional "latest": resolve it against what is on disk.
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        # Keyed by resolution scale, so the same scene can be held at several sizes.
        self.train_cameras = {}
        self.test_cameras = {}

        # Dataset type is inferred purely from what files are present:
        #   sparse/               -> COLMAP
        #   transforms_train.json -> Blender / NeRF-synthetic
        # Anything else is rejected. The most common cause of the assert below is
        # simply forgetting -s, because ModelParams.extract turns an empty
        # source_path into the current working directory rather than failing.
        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            # Training run: copy the seed point cloud and dump the camera set, so the
            # model directory is self-contained for the viewer afterwards.
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                # Note these are CameraInfo records, not Camera objects -- see the
                # warning on camera_to_JSON.
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling
            # Shuffled once here, before the per-resolution loop below, so every
            # resolution scale ends up with cameras in the same order. render.py
            # passes shuffle=False to keep output indices stable.

        # Radius of the scene, derived from how far apart the cameras are. Used as
        # the length scale for densification thresholds and position learning rate.
        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            # This is the expensive part of start-up: every image is decoded,
            # resized and uploaded to the GPU.
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)

        if self.loaded_iter:
            # Evaluation: restore the trained Gaussians verbatim.
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
        else:
            # Training: one Gaussian per input point, sized from local point spacing.
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

    def save(self, iteration):
        """Write the Gaussians to point_cloud/iteration_<n>/point_cloud.ply.

        Parameters only -- no optimiser state. Resuming training needs the .pth that
        train.py writes for --checkpoint_iterations instead.
        """
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        """Cameras at the given resolution scale. KeyError if that scale was not loaded."""
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        """Held-out cameras. Empty unless the dataset was loaded with --eval."""
        return self.test_cameras[scale]
