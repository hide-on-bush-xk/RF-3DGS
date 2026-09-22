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

# Step 3 of the pipeline: score the PNGs render.py wrote.
#
# Reads <model_path>/test/<method>/{renders,gt}/*.png and reports PSNR, SSIM and
# LPIPS. Note this operates on 8-bit PNGs in the image domain -- for RF-3DGS the
# numbers therefore describe the encoded spectrum images, not physical RF units.

from pathlib import Path
import os
from PIL import Image
import torch
import torchvision.transforms.functional as tf
from utils.loss_utils import ssim
from lpipsPyTorch import lpips
import json
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser

def readImages(renders_dir, gt_dir):
    """Load every render and its same-named ground truth onto the GPU.

    Pairing is by filename, so renders/00042.png is compared with gt/00042.png. A
    file present in renders/ but missing from gt/ raises FileNotFoundError.

    Everything is held in GPU memory at once, which is the practical limit on how
    large a test set this can score in one go.
    """
    renders = []
    gts = []
    image_names = []
    for fname in os.listdir(renders_dir):
        render = Image.open(renders_dir / fname)
        gt = Image.open(gt_dir / fname)
        # to_tensor gives CHW float in [0, 1]; unsqueeze(0) adds the batch dim the
        # ssim/lpips implementations expect; [:, :3] drops any alpha channel.
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        image_names.append(fname)
    return renders, gts, image_names

def evaluate(model_paths):
    """Score every model directory given and write results.json / per_view.json.

    `method` is the ours_<iteration> subdirectory, so a model rendered at several
    iterations gets one entry per iteration.
    """

    full_dict = {}
    per_view_dict = {}
    # Declared and written but never populated with anything different, and never
    # saved: dead state carried over from upstream.
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}
    print("")

    for scene_dir in model_paths:
        # WARNING: this bare except swallows *every* error, including the real
        # cause, and prints only the generic message at the bottom. If you see
        # "Unable to compute metrics", re-run with this try/except removed (or add
        # traceback.print_exc()) to find out what actually failed.
        try:
            print("Scene:", scene_dir)
            full_dict[scene_dir] = {}
            per_view_dict[scene_dir] = {}
            full_dict_polytopeonly[scene_dir] = {}
            per_view_dict_polytopeonly[scene_dir] = {}

            # Only the held-out split is scored; the train/ renders are ignored.
            test_dir = Path(scene_dir) / "test"

            for method in os.listdir(test_dir):
                print("Method:", method)

                full_dict[scene_dir][method] = {}
                per_view_dict[scene_dir][method] = {}
                full_dict_polytopeonly[scene_dir][method] = {}
                per_view_dict_polytopeonly[scene_dir][method] = {}

                method_dir = test_dir / method
                gt_dir = method_dir/ "gt"
                renders_dir = method_dir / "renders"
                renders, gts, image_names = readImages(renders_dir, gt_dir)

                ssims = []
                psnrs = []
                lpipss = []

                for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
                    ssims.append(ssim(renders[idx], gts[idx]))
                    psnrs.append(psnr(renders[idx], gts[idx]))
                    # 'vgg' is the variant reported in the 3DGS paper. The backbone
                    # is rebuilt on every call -- see lpipsPyTorch/__init__.py.
                    lpipss.append(lpips(renders[idx], gts[idx], net_type='vgg'))

                # The trailing ".5" argument does nothing: format() ignores extra
                # positional arguments that no placeholder references.
                print("  SSIM : {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"))
                print("  PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean(), ".5"))
                print("  LPIPS: {:>12.7f}".format(torch.tensor(lpipss).mean(), ".5"))
                print("")

                # Headline numbers: the mean over views of each per-view metric.
                full_dict[scene_dir][method].update({"SSIM": torch.tensor(ssims).mean().item(),
                                                        "PSNR": torch.tensor(psnrs).mean().item(),
                                                        "LPIPS": torch.tensor(lpipss).mean().item()})
                # Same values kept per view, keyed by filename, so a bad view can
                # be traced back to its image.
                per_view_dict[scene_dir][method].update({"SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
                                                            "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)},
                                                            "LPIPS": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)}})

            with open(scene_dir + "/results.json", 'w') as fp:
                json.dump(full_dict[scene_dir], fp, indent=True)
            with open(scene_dir + "/per_view.json", 'w') as fp:
                json.dump(per_view_dict[scene_dir], fp, indent=True)
        except:
            print("Unable to compute metrics for model", scene_dir)

if __name__ == "__main__":
    # Pinned to the first GPU, like safe_state does for train.py and render.py.
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    # nargs="+" so several model directories can be scored in one invocation.
    parser.add_argument('--model_paths', '-m', required=True, nargs="+", type=str, default=[])
    args = parser.parse_args()
    evaluate(args.model_paths)
