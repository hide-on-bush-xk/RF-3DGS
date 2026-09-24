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

# Step 2 of the pipeline: load a trained model and write out one PNG per view,
# alongside the matching ground truth. metrics.py then scores those PNGs.
#
# Only -m is needed. get_combined_args recovers everything else from the cfg_args
# file the training run left in that directory -- which also means the original
# source_path must still exist, since the ground-truth images are read from it.

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel

def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    """Render every view in `views` and save render/ground-truth pairs as PNGs.

    Layout written: <model_path>/<name>/ours_<iteration>/{renders,gt}/00000.png ...
    Files are numbered by position in `views`, not by the original image name, so
    the two directories line up index for index (which is what metrics.py relies on).
    """
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        rendering = render(view, gaussians, pipeline, background)["render"]
        gt = view.original_image[0:3, :, :]      # drop any alpha channel
        # save_image clamps to [0, 1] and quantises to 8-bit. Note that this
        # quantisation happens before metrics.py scores the files, so the reported
        # PSNR/SSIM/LPIPS are measured on 8-bit PNGs, not on the float renders.
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    """Load the trained Gaussians and render the train and/or test splits."""
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        # load_iteration=-1 means "the highest iteration saved"; shuffle=False keeps
        # the camera order deterministic so render indices are reproducible.
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)

        if not skip_test:
             # Empty unless the model was trained with --eval; without it every
             # camera is a training camera and this writes nothing.
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    # sentinel=True makes every ModelParams default None, so get_combined_args can
    # tell which flags were actually typed and let cfg_args supply the rest.
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)   # -1 = latest saved
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)
