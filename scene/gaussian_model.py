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

# The model itself: every Gaussian's parameters, plus the adaptive density control
# that grows and prunes them during training.
#
# Central design point -- parameters are stored in an *unconstrained* space and an
# activation maps them to the valid range on read:
#
#   _scaling  --exp-->        scale > 0
#   _opacity  --sigmoid-->    opacity in (0, 1)
#   _rotation --normalize-->  unit quaternion
#   _xyz                      used directly
#
# That is why the underscored attributes are what the optimiser sees, and the
# get_* properties are what the renderer sees. Assigning to a get_* value does
# nothing useful; write to the underscored tensor instead.

import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation

class GaussianModel:

    def setup_functions(self):
        """Bind the activation functions described in the module comment above."""
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            # Sigma = (R S)(R S)^T. Factorising this way guarantees a valid
            # (positive semi-definite) covariance for any parameter values.
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            # Only the 6 unique entries are passed on, since Sigma is symmetric.
            symm = strip_symmetric(actual_covariance)
            return symm

        # exp/log keep scale strictly positive while the raw parameter roams freely.
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree : int):
        """Create an empty model. The tensors are filled by create_from_pcd,
        load_ply or restore -- nothing is usable until one of those has run."""
        # Starts at 0 and is raised every 1000 iterations by oneupSHdegree, so
        # appearance begins view-independent and gains detail gradually.
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree
        self._xyz = torch.empty(0)            # [N, 3]  centres, world space
        self._features_dc = torch.empty(0)    # [N, 1, 3]  band-0 SH (base colour)
        self._features_rest = torch.empty(0)  # [N, (d+1)^2-1, 3]  higher SH bands
        self._scaling = torch.empty(0)        # [N, 3]  log of the axis lengths
        self._rotation = torch.empty(0)       # [N, 4]  quaternion (w, x, y, z)
        self._opacity = torch.empty(0)        # [N, 1]  logit of the opacity
        # Densification bookkeeping, all [N]-shaped and kept in lockstep with the
        # parameters above by prune_points / densification_postfix.
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0             # scene radius; scales the position lr
        self.setup_functions()

    def capture(self):
        """Everything needed to resume training, as a plain tuple.

        train.py pickles this for --checkpoint_iterations. Unlike save_ply it
        includes the optimiser state and the densification counters, so a resumed
        run continues rather than restarting the schedule.
        """
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )

    def restore(self, model_args, training_args):
        """Inverse of capture(). Order must match capture() exactly.

        training_setup is called in the middle because the optimiser has to exist
        (and point at the restored parameter tensors) before its state dict can be
        loaded back into it.
        """
        (self.active_sh_degree,
        self._xyz,
        self._features_dc,
        self._features_rest,
        self._scaling,
        self._rotation,
        self._opacity,
        self.max_radii2D,
        xyz_gradient_accum,
        denom,
        opt_dict,
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        # Restored after training_setup, which would otherwise have zeroed them.
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        """[N, 3] positive axis lengths."""
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        """[N, 4] unit quaternions."""
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        """[N, 3] centres. No activation: positions are already unconstrained."""
        return self._xyz

    @property
    def get_features(self):
        """All SH coefficients as one [N, (d+1)^2, 3] tensor.

        DC and the higher bands are stored separately because they are optimised at
        different learning rates (f_rest uses feature_lr / 20), but the rasteriser
        wants them contiguous.
        """
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_opacity(self):
        """[N, 1] opacities in (0, 1)."""
        return self.opacity_activation(self._opacity)

    def get_covariance(self, scaling_modifier = 1):
        """[N, 6] packed 3D covariances. Only used when pipe.compute_cov3D_python."""
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        """Unlock one more SH band, up to max_sh_degree. Called every 1000 iterations."""
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        """Initialise one Gaussian per input point.

        spatial_lr_scale is the scene radius; it multiplies the position learning
        rate so that the same hyper-parameters work across scenes of different size.
        """
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        # Point colours become the band-0 SH coefficients.
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color    # band 0 gets the colour
        # No-op left over from upstream: `features` has exactly 3 channels, so the
        # slice 3: is empty and this assigns to nothing. The higher bands are
        # already zero from torch.zeros above.
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        # Initial size = local point spacing: distCUDA2 returns the mean squared
        # distance to each point's nearest neighbours, so sqrt gives a length. The
        # clamp keeps duplicate points (distance 0) from producing log(0) = -inf.
        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        # Stored in log space (the inverse of the exp activation), isotropic to start.
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1                       # identity quaternion (w=1, xyz=0)

        # Start almost transparent (0.1) and let the optimiser raise what it needs.
        # Stored as a logit because of the sigmoid activation.
        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        # transpose(1, 2) puts the SH-coefficient axis before the channel axis,
        # giving [N, 1, 3] for DC and [N, rest, 3] for the higher bands.
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def training_setup(self, training_args):
        """Build the Adam optimiser with one parameter group per attribute.

        The groups are named, and every later piece of optimiser surgery
        (replace/prune/cat) looks them up by that name.
        """
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            # Position lr is scaled by the scene radius and is the only one scheduled.
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            # Higher SH bands learn 20x slower, so view dependence is added cautiously.
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        # lr=0.0 is only the default for groups that do not set their own; every
        # group above does. eps is unusually small because gradients here are tiny.
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step

        Only the "xyz" group is scheduled; the early return leaves every other
        group at its constant rate.
        '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        """Names of the per-vertex PLY fields, in the exact order save_ply writes them.

        Must stay in step with the concatenation in save_ply and with the parsing in
        load_ply; the numeric suffixes are how load_ply re-sorts the SH coefficients.
        """
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        """Write the Gaussians as a PLY point cloud.

        Values are saved in their raw (pre-activation) form, so a viewer reading
        this file must apply exp/sigmoid/normalize itself. Normals are written as
        zeros purely to keep the standard PLY vertex layout.
        """
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        # transpose back to [N, 3, coeffs] then flatten, matching the field order
        # that construct_list_of_attributes generates.
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        # Every field is float32 ('f4'), which is why a 1M-Gaussian model is ~240 MB.
        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        """Clamp every opacity down to at most 0.01.

        Part of adaptive density control: Gaussians that genuinely matter climb back
        within a few hundred iterations, while floaters stay faint and are removed by
        the opacity pruning in densify_and_prune. Goes through the optimiser so Adam's
        moments for this parameter are reset too -- otherwise stale momentum would
        undo the reset immediately.
        """
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        """Read back a model written by save_ply.

        Fields are located by name and sorted by their numeric suffix rather than
        trusting the file's declaration order.
        """
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        # Sort numerically ("f_rest_10" must come after "f_rest_9", not after "f_rest_1").
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        # Fails loudly if the file's SH degree does not match this model's: loading a
        # degree-3 .ply into a model built with a different --sh_degree stops here.
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        # A saved model is fully trained, so all SH bands are active immediately.
        self.active_sh_degree = self.max_sh_degree

    # ---------------------------------------------------------------------------
    # Optimiser surgery.
    #
    # Densification changes the number of Gaussians, but Adam keeps per-element
    # moment buffers (exp_avg, exp_avg_sq) that must stay the same shape as the
    # parameters. The three methods below rebuild the parameter tensors and their
    # moments together. Doing this by hand is unavoidable: torch.optim has no API
    # for resizing a parameter in place.
    # ---------------------------------------------------------------------------

    def replace_tensor_to_optimizer(self, tensor, name):
        """Swap one named parameter for `tensor` and zero its Adam moments.

        Used only by reset_opacity. Note stored_state is dereferenced without a None
        check, so this assumes the optimiser has already taken at least one step.
        """
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                # The state dict is keyed by the Parameter object, so the old key
                # must be deleted before the new Parameter is registered.
                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        """Keep only the rows selected by `mask` in every parameter and its moments."""
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                # Moments are masked alongside the parameter so the surviving
                # Gaussians keep their momentum.
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                # Before the first optimiser step there is no state to carry over.
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        """Delete the Gaussians where `mask` is True.

        `mask` marks points to remove; it is inverted here into a keep-mask. The
        densification counters are masked with the same indices so every per-Gaussian
        array stays aligned.
        """
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        """Append new rows to every parameter, with zeroed Adam moments for them.

        tensors_dict is keyed by parameter-group name, so it must contain an entry
        for all six groups or the lookup raises KeyError.
        """
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1     # one tensor per group, by construction
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                # New Gaussians start with no momentum; existing ones keep theirs.
                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation):
        """Append the given new Gaussians and reset the densification counters.

        The counters are zeroed (not extended), so the gradient statistics restart
        from scratch after every densification round.
        """
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        """Split *large* under-fitting Gaussians into N smaller ones.

        Selection: high accumulated screen-space gradient AND an extent larger than
        percent_dense * scene_extent. The originals are deleted at the end, so the
        count grows by (N-1) per selected Gaussian.
        """
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        # densify_and_clone has already appended new rows, so `grads` is shorter
        # than the current point count; pad it so the mask lines up. The padded
        # entries are zero, which means freshly cloned Gaussians are never split
        # in the same round.
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        # "Large" test -- this is what separates split from clone.
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        # Place the children by sampling from the parent's own Gaussian: standard
        # deviation = the parent's scale, then rotated into the parent's frame.
        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        # Children are shrunk by 1/(0.8*N): the empirical factor from the 3DGS paper
        # that keeps the split from inflating the total volume.
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        # Orientation, appearance and opacity are inherited unchanged.
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

        # Remove the parents. The mask must be padded with False for the N*k
        # children just appended, so that they are kept.
        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        """Duplicate *small* under-fitting Gaussians in place.

        Selection is the mirror image of densify_and_split: same gradient test, but
        an extent at or below percent_dense * scene_extent. The copy is identical,
        including position -- the optimiser separates the pair on later steps.
        """
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)

        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        """One full round of adaptive density control. Called every densification_interval.

        Args:
            max_grad:        screen-space gradient above which to clone or split
            min_opacity:     prune anything fainter (train.py passes 0.005)
            extent:          scene radius
            max_screen_size: prune anything wider on screen, or None to skip that test
        """
        # Mean screen-space gradient per Gaussian since the last round.
        grads = self.xyz_gradient_accum / self.denom
        # Gaussians never rendered have denom 0 and therefore NaN here.
        grads[grads.isnan()] = 0.0

        # Clone first, then split. Order matters: this is why densify_and_split has
        # to pad `grads` to the grown point count.
        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        # Three pruning rules, OR-ed together.
        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size              # too big on screen
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent  # too big in world
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        # All that reallocation leaves the caching allocator fragmented.
        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        """Accumulate the screen-space position gradient for the visible Gaussians.

        `viewspace_point_tensor.grad` is what gaussian_renderer.render set up; only
        its first two components (x, y in screen space) are used. `denom` counts how
        many views contributed, so the ratio in densify_and_prune is a mean.
        """
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1
