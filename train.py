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

# Entry point for training. Two distinct uses, selected by --start_checkpoint:
#
#   stage 1 (no checkpoint): ordinary 3DGS. Optimise positions, scales, rotations,
#           opacities and appearance together to reconstruct the visual scene.
#   stage 2 (with checkpoint): RF-3DGS. Load the stage-1 geometry, freeze it, wipe
#           the appearance, and relearn only appearance + opacity against the radio
#           spatial spectrum. See the "RF_3dgs_retraining" block in training().

import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
# TensorBoard is optional. Without the `tensorboard` package installed (note:
# having torch is not enough) training runs normally, it just logs nothing.
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    """Run the whole optimisation loop.

    Args:
        dataset:  extracted ModelParams -- source_path, model_path, sh_degree, ...
        opt:      extracted OptimizationParams -- learning rates, densification schedule
        pipe:     extracted PipelineParams -- the CUDA-vs-Python rendering switches
        testing_iterations / saving_iterations / checkpoint_iterations:
                  iteration numbers at which to evaluate / write a .ply / write a .pth
        checkpoint:  path to a .pth to resume from, or None to start fresh
        debug_from:  iteration after which to turn on rasteriser debug output (-1 = never)
    """
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)

    # The model holds every Gaussian's parameters; the Scene owns the cameras and
    # seeds the model from the input point cloud.
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    # Builds the Adam optimiser and the position learning-rate schedule. Must come
    # after Scene, which is what creates the Gaussians in the first place.
    gaussians.training_setup(opt)
    if checkpoint:
        # torch>=2.6 defaults torch.load to weights_only=True, which refuses the
        # numpy scalars stored inside the 3DGS checkpoints.
        (model_params, first_iter) = torch.load(checkpoint, weights_only=False)
        gaussians.restore(model_params, opt)

        # RF_3dgs_retraining
        #
        # This is the RF-3DGS stage-2 switch and the main departure from upstream
        # 3DGS. The geometry recovered from the visual scene is taken as given:
        # positions, scales and rotations are frozen, so the Gaussians stay where
        # the optical reconstruction put them. Only opacity stays trainable,
        # because a surface's transparency to radio differs from its transparency
        # to light.
        print("load checkpoints and reset for RF training")
        gaussians._xyz.requires_grad = False
        gaussians._scaling.requires_grad = False
        gaussians._rotation.requires_grad = False
        gaussians._opacity.requires_grad = True

        # Wipe the learned RGB appearance. The SH coefficients are about to be
        # retrained to carry the radio spatial spectrum instead, so keeping the
        # visual colours would only be a bad initialisation. Zeroed in place with
        # .data so the optimiser's parameter objects (and their Adam state) survive.
        gaussians._features_dc.data.fill_(0.0)
        gaussians._features_rest.data.fill_(0.0)

    # Background colour for pixels no Gaussian covers.
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # CUDA events time the GPU work of one iteration without stalling the pipeline
    # the way time.time() around a .item() would.
    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None      # cameras not yet used in the current epoch
    ema_loss_for_log = 0.0      # smoothed loss, for the progress bar only
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    # TRAP: --iterations is an ABSOLUTE iteration number, not "how many more steps
    # to run". Resuming from chkpnt30000 sets first_iter to 30000, so any smaller
    # value makes this range empty: the loop body never executes, nothing is saved
    # or evaluated, and the script still prints "Training complete." and exits 0,
    # with no warning that it did nothing. A resumed run needs --iterations above
    # the checkpoint's iteration (the README uses 40000), and --test_iterations /
    # --save_iterations must be numbers this range actually reaches.
    for iteration in range(first_iter, opt.iterations + 1):
        # ---- interactive viewer service -------------------------------------
        # Serves the SIBR remote viewer on --ip/--port. Entirely optional: with no
        # client connected, try_connect() fails and the while loop never runs.
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                # The viewer can also flip the pipeline flags live, which is why
                # pipe.* is assigned to here.
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    # CHW float -> HWC uint8 bytes for the wire.
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                # Leave the loop and resume training unless the viewer asked us to
                # pause (do_training false) or to hold the process open at the end.
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                # Any socket error simply drops the client; training carries on.
                network_gui.conn = None

        iter_start.record()

        # Position learning rate follows an exponential schedule; everything else
        # is constant. See GaussianModel.update_learning_rate.
        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        # Coarse-to-fine on appearance: start view-independent, unlock one SH band
        # at a time so the optimiser cannot use high-frequency view dependence to
        # paper over bad geometry early on.
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        # Sampling without replacement: the stack is refilled once empty, so every
        # camera is seen once per "epoch" in a fresh random order.
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        # A random background each iteration stops the model explaining empty
        # regions with a background-coloured blob. Off by default.
        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        # The 3DGS objective: mostly L1, with a structural term weighted by
        # lambda_dssim (0.2 by default). SSIM is a similarity, hence 1 - ssim.
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        loss.backward()

        iter_end.record()

        # Everything below is bookkeeping: no gradients should be built here.
        with torch.no_grad():
            # Progress bar
            # Exponential moving average, 0.4 on the new value. Display only --
            # this number never feeds back into the optimisation.
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            if (iteration in saving_iterations):
                # Writes point_cloud/iteration_N/point_cloud.ply -- the artefact
                # render.py and the viewers consume.
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            # Adaptive density control: grow Gaussians where the reconstruction is
            # under-resolved, remove ones that contribute nothing. Runs only during
            # the first densify_until_iter iterations; after that the count is fixed.
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                # Largest on-screen radius ever reached, used by the size-based
                # pruning rule below. Only visible Gaussians are updated.
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                # Accumulate the screen-space position gradient that decides which
                # Gaussians get cloned or split.
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    # Screen-size pruning (drop Gaussians wider than 20 px) is only
                    # enabled after the first opacity reset, so it does not fire
                    # while opacities are still settling.
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    # 0.005 is the minimum opacity: anything fainter is pruned.
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)

                # Periodically push every opacity back down near zero. Gaussians
                # that matter recover within a few hundred iterations; floaters do
                # not, and get pruned. The white-background case also resets once
                # at densify_from_iter, where a bright background otherwise hides
                # spurious geometry.
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            # Skipped on the very last iteration so the saved .ply matches exactly
            # the parameters that produced the final reported metrics.
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                # Full resumable state (parameters + optimiser + iteration), unlike
                # the .ply above which is parameters only. This is the file a later
                # run passes to --start_checkpoint.
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def prepare_output_and_logger(args):
    """Create the output directory, record the run's arguments, open TensorBoard.

    Writing `cfg_args` here is what lets render.py and metrics.py later run with
    only -m: get_combined_args reads this file back to recover source_path, eval,
    white_background and the rest.
    """
    if not args.model_path:
        # No -m given: invent a name. On an OAR cluster use the job id, otherwise a
        # random UUID. Note this means a run launched without -m lands in a
        # meaningless ./output/<random> directory, which is hard to match to the
        # command afterwards -- always pass -m explicitly.
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    # Serialised as a Namespace repr, which get_combined_args later eval()s.
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        # Not an error: this only means the `tensorboard` package is missing from
        # the environment. Training proceeds; the loss curves are simply not logged.
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    """Log scalars every iteration, and run a full evaluation on `testing_iterations`.

    Every tb_writer call is guarded, so with TensorBoard missing this degrades to
    the print() below and nothing else.
    """
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        # Evaluation renders at full resolution on top of the training footprint,
        # so the cache is cleared on both sides of it.
        torch.cuda.empty_cache()
        # Two sets: the whole held-out split, plus 5 training views (indices
        # 5, 10, 15, 20, 25) as a sanity check on fit vs generalisation. The
        # modulo keeps it valid for datasets with fewer than 30 training cameras.
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()},
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    # Clamped to [0, 1] before scoring, which is what psnr() assumes.
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        # Only the first 5 views get images logged, to bound the
                        # event file size. image[None] adds the batch dim add_images wants.
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        # Ground truth never changes, so log it once at the first
                        # evaluation only.
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    # .double() so summing hundreds of views does not lose precision.
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                # NOTE: this is a mean of per-view PSNR, which is not the same as
                # the PSNR of the pooled error. metrics.py averages the same way,
                # so the two agree, but neither matches a global-MSE definition.
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            # Opacity distribution and Gaussian count: the two numbers that show
            # whether densification and pruning are behaving.
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()



if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    # Each of these registers its own flags on the parser and can later extract a
    # plain namespace of just its own fields. See arguments/__init__.py.
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    # Address the interactive viewer connects to.
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    # Autograd anomaly detection: finds the op that produced a NaN, at a large
    # speed cost. Leave off unless chasing one.
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)

    args = parser.parse_args(sys.argv[1:])
    # The final iteration is always saved, whatever --save_iterations says.
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    # Also seeds the RNGs and pins the process to cuda:0.
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    # .extract(args) pulls out each group's own fields; ModelParams.extract also
    # turns source_path into an absolute path -- which is why omitting -s silently
    # resolves to the current working directory rather than failing outright.
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
