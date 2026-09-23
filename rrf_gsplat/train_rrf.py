"""Radio radiance field fine-tuning on gsplat, with the colour function as a knob.

RF-3DGS fine-tunes the colour (SH) and opacity of a frozen visual 3DGS on
jet-colourmapped spectrum PNGs, through the INRIA rasteriser (3 channels,
L1 + SSIM). This does the same on gsplat, which rasterises any number of
channels, so the target no longer has to be an RGB picture:

  --mode rgb     the RF-3DGS baseline: 3-channel jet PNG target, SH colours
  --mode db      1-channel target: the spectrum in dB, normalised to [0, 1]
                 with the dataset's global range; SH gives the dB value
  --mode power   each Gaussian carries a linear power; alpha compositing sums
                 powers, and the loss is taken on 10 log10 of the sum. This is
                 the composition rule that matches what the spectrum is.

Geometry is frozen as in RF-3DGS unless --train-geometry; --densify mcmc adds
gsplat's MCMC strategy (relocate dead Gaussians, add up to --cap-max, inject
noise), which -- unlike the INRIA densifier gated by densify_until_iter --
works in the 30k -> 40k window the RF fine-tune lives in.

Every run is evaluated the same way: predicted dB against the float truth
(when the dataset has spectra_float/), and a paper-comparable PSNR after
mapping the prediction through the jet colormap. Wall time is logged so the
"how fast after the transmitter moves" question has a number.

Runs in WSL where gsplat is built (env rf-gsplat); reads the repo from /mnt/c.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)
from utils.loss_utils import l1_loss, ssim          # noqa: E402  (pure torch)

from jet import jet_rgb, jet_inverse               # noqa: E402


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------
def qvec2rotmat(q):
    """COLMAP quaternion (w, x, y, z) -> 3x3 rotation."""
    w, x, y, z = q
    return np.array([
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
        [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
        [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y]])


def read_colmap_text(sparse_dir):
    """cameras.txt + images.txt -> {name: (viewmat 4x4, K 3x3, w, h)}."""
    cams = {}
    with open(os.path.join(sparse_dir, "cameras.txt")) as fid:
        for line in fid:
            if line.startswith("#") or not line.strip():
                continue
            p = line.split()
            cid, model, w, h = int(p[0]), p[1], int(p[2]), int(p[3])
            if model == "PINHOLE":
                fx, fy, cx, cy = map(float, p[4:8])
            elif model == "SIMPLE_PINHOLE":
                fx = fy = float(p[4]); cx, cy = float(p[5]), float(p[6])
            else:
                raise ValueError(f"camera model {model} not handled")
            cams[cid] = (np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32), w, h)
    views = {}
    with open(os.path.join(sparse_dir, "images.txt")) as fid:
        lines = [l for l in fid if not l.startswith("#") and l.strip()]
    # image lines end with the file name; the POINTS2D line that follows each
    # (possibly empty) is all numbers
    for line in lines:
        p = line.split()
        if len(p) < 10 or not p[9].lower().endswith((".png", ".jpg", ".jpeg")):
            continue
        q = list(map(float, p[1:5])); t = np.array(list(map(float, p[5:8])), dtype=np.float32)
        cid, name = int(p[8]), p[9]
        R = qvec2rotmat(q).astype(np.float32)          # world -> camera
        view = np.eye(4, dtype=np.float32); view[:3, :3] = R; view[:3, 3] = t
        K, w, h = cams[cid]
        views[name] = (view, K, w, h)
    return views


def read_index(path):
    """One image name per line, blanks dropped. The dataset's split file."""
    with open(path) as fid:
        return [l.strip() for l in fid if l.strip()]


def ensure_split(source, test_fraction=0.2, seed=0, views_per_position=4):
    """Write train_index.txt / test_index.txt if the dataset has none.

    Held out by receiver position, as the released split is (160 positions x
    4 yaws): images are numbered position-major, so consecutive groups of
    `views_per_position` names belong to one position.
    """
    tr, te = os.path.join(source, "train_index.txt"), os.path.join(source, "test_index.txt")
    if os.path.exists(tr) and os.path.exists(te):
        return read_index(tr), read_index(te)
    names = sorted(os.path.splitext(f)[0] for f in os.listdir(os.path.join(source, "images"))
                   if f.endswith(".png"))
    # By position, not by view: holding out a single yaw of a position would
    # leave the other three faces of the same receiver in training, which is a
    # leak rather than a held-out sample.
    n_pos = len(names) // views_per_position
    rng = np.random.default_rng(seed)
    test_pos = set(rng.permutation(n_pos)[:int(round(test_fraction * n_pos))].tolist())
    test = [n for i, n in enumerate(names) if (i // views_per_position) in test_pos]
    train = [n for i, n in enumerate(names) if (i // views_per_position) not in test_pos]
    with open(tr, "w") as fid:
        fid.write("\n".join(train) + "\n")
    with open(te, "w") as fid:
        fid.write("\n".join(test) + "\n")
    print(f"wrote split: {len(train)} train / {len(test)} test (seed {seed})")
    return train, test


def check_jet_encoded(rgb, tol=12.0, sample=20000):
    """Raise unless these images were written through the jet colormap.

    rgb is [N, 3, H, W] uint8. Measures the distance from a sample of pixels to
    the nearest of the 256 jet colours; a genuinely jet-encoded dataset scores
    under 4, a dataset that merely happens to be colourful scores in the tens.
    """
    lut = jet_rgb(torch.linspace(0, 1, 256, device=rgb.device)).movedim(0, 1) * 255.0   # [256, 3]
    flat = rgb.float().permute(0, 2, 3, 1).reshape(-1, 3)
    if flat.shape[0] > sample:
        idx = torch.randperm(flat.shape[0], device=flat.device)[:sample]
        flat = flat[idx]
    d = torch.cdist(flat, lut).min(dim=1).values
    mean, within = float(d.mean()), float((d < tol).float().mean())
    print(f"jet check: mean distance to the jet curve {mean:.1f}, {100 * within:.1f}% of pixels within {tol:g}")
    if mean >= tol:
        raise SystemExit(
            f"this dataset was not written through the jet colormap (mean distance {mean:.1f}, "
            f"only {100 * within:.1f}% of pixels within {tol:g}), so jet_inverse would return an "
            f"arbitrary value per pixel. Use --mode rgb, or supply spectra_float/ with the real values."
        )


def load_views(source, names, views, device, want_float):
    """PNG targets as uint8 [n,3,H,W]; float spectra as float16 [n,H,W] if present."""
    from PIL import Image
    rgb, flt, viewmats, Ks = [], [], [], []
    have_float = want_float and os.path.isdir(os.path.join(source, "spectra_float"))
    width = height = None
    for n in names:
        img = np.array(Image.open(os.path.join(source, "images", n + ".png")).convert("RGB"))
        rgb.append(torch.from_numpy(img).permute(2, 0, 1))
        if have_float:
            flt.append(torch.from_numpy(np.load(os.path.join(source, "spectra_float", n + ".npy"))
                                        .astype(np.float16)))
        view, K, w, h = views[n + ".png"]
        ih, iw = img.shape[:2]
        if (iw, ih) != (w, h):
            # the released AoD / Delay pictures are 231 x 154 under a 300 x 200 camera: like INRIA's loader
            # (resolution -1), the image defines the render size and the intrinsics scale with it
            K = K.copy(); K[0, :] *= iw / w; K[1, :] *= ih / h; w, h = iw, ih
        if width is None:
            width, height = w, h
        elif (w, h) != (width, height):
            raise ValueError(f"{n}: image size {w}x{h} differs from the first view's {width}x{height}")
        viewmats.append(torch.from_numpy(view)); Ks.append(torch.from_numpy(np.ascontiguousarray(K)))
    # Everything resident on the GPU: uint8 for the PNGs and float16 for the
    # spectra, which is what makes a whole dataset fit alongside the model.
    out = {"rgb": torch.stack(rgb).to(device), "viewmats": torch.stack(viewmats).to(device),
           "Ks": torch.stack(Ks).to(device), "names": names, "width": width, "height": height}
    if have_float:
        out["float"] = torch.stack(flt).to(device)
    return out


# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------
def sh_basis(deg, dirs):
    """Real SH basis up to degree 3 for unit directions [N,3] -> [N,(deg+1)^2].

    Same ordering and constants as INRIA's utils/sh_utils.eval_sh, so the
    coefficients mean the same thing.
    """
    x, y, z = dirs[:, 0], dirs[:, 1], dirs[:, 2]
    cols = [torch.full_like(x, 0.28209479177387814)]
    if deg >= 1:
        cols += [-0.4886025119029199 * y, 0.4886025119029199 * z, -0.4886025119029199 * x]
    if deg >= 2:
        xx, yy, zz, xy, yz, xz = x * x, y * y, z * z, x * y, y * z, x * z
        cols += [1.0925484305920792 * xy, -1.0925484305920792 * yz,
                 0.31539156525252005 * (2.0 * zz - xx - yy),
                 -1.0925484305920792 * xz, 0.5462742152960396 * (xx - yy)]
    if deg >= 3:
        cols += [-0.5900435899266435 * y * (3 * xx - yy), 2.890611442640554 * xy * z,
                 -0.4570457994644658 * y * (4 * zz - xx - yy),
                 0.3731763325901154 * z * (2 * zz - 3 * xx - 3 * yy),
                 -0.4570457994644658 * x * (4 * zz - xx - yy),
                 1.445305721320277 * z * (xx - yy),
                 -0.5900435899266435 * x * (xx - 3 * yy)]
    return torch.stack(cols, dim=1)


class RRF(torch.nn.Module):
    """Geometry from the visual checkpoint; colours and opacity learn.

    Parameters live in a ParameterDict under gsplat's names (means, scales,
    quats, opacities, sh0, shN) so a densification strategy can replace them;
    everything reads through the properties below.
    """

    def __init__(self, ckpt_path, mode, channels, sh_degree, device,
                 train_opacity=True, train_geometry=False):
        super().__init__()
        (m, it) = torch.load(ckpt_path, weights_only=False, map_location="cpu")
        (_, xyz, f_dc, f_rest, scaling, rotation, opacity, *rest) = m
        self.spatial_lr_scale = float(rest[-1]) if rest else 1.0
        # The checkpoint holds nn.Parameters of the visual training, so every
        # tensor still requires grad; detach, or exp()/normalize() become graph
        # nodes that are freed after the first backward.
        xyz, scaling, rotation, opacity = (t.detach().to(device) for t in (xyz, scaling, rotation, opacity))
        self.mode, self.channels, self.sh_degree = mode, channels, sh_degree
        self.delay_channel, self.delay_span_ns, self.delay_depth_mode = None, None, "D"   # --delay-depth: delay = learned residual + rendered depth / c
        self.delay_range, self._range_factor = "z", None                                  # --delay-range euclid: depth * sec(theta_pixel) = range along the ray
        n, k = xyz.shape[0], (sh_degree + 1) ** 2
        # RF-3DGS zeroes every SH coefficient before RF training; the same here,
        # for any channel count. DC and the rest are separate parameters so
        # they can take INRIA's separate learning rates (Adam ignores gradient
        # scaling, so a single tensor could not emulate that). [N, K, C]
        # (coefficient-major) is the layout gsplat's SH kernel takes.
        self.params = torch.nn.ParameterDict({
            "means": torch.nn.Parameter(xyz.clone(), requires_grad=train_geometry),
            "scales": torch.nn.Parameter(scaling.clone(), requires_grad=train_geometry),    # log
            "quats": torch.nn.Parameter(rotation.clone(), requires_grad=train_geometry),
            "opacities": torch.nn.Parameter(opacity.reshape(-1).clone(), requires_grad=train_opacity),  # logit [N]
            "sh0": torch.nn.Parameter(torch.zeros(n, 1, channels, device=device)),
            "shN": torch.nn.Parameter(torch.zeros(n, k - 1, channels, device=device)),
        })
        self.visual_iteration = it
        self.last_info = None

    # The activated views of the raw parameters. Same convention as INRIA's
    # GaussianModel: parameters are stored unconstrained (log scale, logit
    # opacity, unnormalised quaternion) and these apply the activation.
    @property
    def means(self):
        return self.params["means"]

    @property
    def scales(self):
        return torch.exp(self.params["scales"])

    @property
    def quats(self):
        return F.normalize(self.params["quats"], dim=-1)

    @property
    def opacities(self):
        return torch.sigmoid(self.params["opacities"])

    @property
    def sh(self):
        return torch.cat([self.params["sh0"], self.params["shN"]], dim=1)   # [N, K, C]

    @property
    def n_gaussians(self):
        return int(self.params["means"].shape[0])

    def colours(self, cam_center):
        """View-dependent per-Gaussian value(s), [N, C]; the +0.5 is INRIA's.

        The SH basis depends only on the means and the camera; with frozen
        geometry it is a constant per view and the colour is one linear map of
        the coefficients, so the backward pass is one multiply-add instead of
        autograd through INRIA's eval_sh expression (two thirds of a step for
        1M Gaussians). With trainable geometry the basis carries a gradient.
        """
        if self.sh_degree == 0:
            return self.params["sh0"][:, 0, :] * 0.28209479177387814 + 0.5
        ctx = torch.enable_grad() if self.params["means"].requires_grad else torch.no_grad()
        with ctx:
            dirs = F.normalize(self.means - cam_center[None], dim=-1)
            basis = sh_basis(self.sh_degree, dirs)                  # [N, K]
        return (self.sh * basis[:, :, None]).sum(dim=1) + 0.5

    def render(self, viewmat, K, width, height, span_db):
        """One view, [C, H, W]. Three paths depending on mode, plus the optional
        delay-depth decomposition; span_db is the dataset's dB range, needed only
        by the power mode and the delay term."""
        from gsplat import rasterization
        device = self.means.device
        if self.mode == "rgb" and self.sh_degree > 0:
            # gsplat evaluates 3-channel SH in CUDA (and adds INRIA's 0.5 and
            # clamps at 0 itself); its backward is a third of the torch path's
            img, alpha, info = rasterization(
                self.means, self.quats, self.scales, self.opacities, self.sh,
                viewmat[None], K[None], width, height, sh_degree=self.sh_degree,
                backgrounds=torch.zeros(1, 3, device=device))
            self.last_info = info
            return img[0].permute(2, 0, 1)
        cam_center = torch.linalg.inv(viewmat)[:3, 3]
        col = self.colours(cam_center)
        if self.mode == "rgb":
            col = col.clamp_min(0.0)
        elif self.mode == "power":
            # value in [0,1] is dB above the floor as a fraction of the span;
            # composite linear powers, read back in dB
            col = torch.pow(10.0, col.clamp(0.0, 1.0) * span_db / 10.0)
        img, alpha, info = rasterization(
            self.means, self.quats, self.scales, self.opacities, col,
            viewmat[None], K[None], width, height, sh_degree=None,
            backgrounds=torch.zeros(1, self.channels, device=device),
            render_mode=("RGB+" + self.delay_depth_mode) if self.delay_channel is not None else "RGB")
        self.last_info = info
        img = img[0].permute(2, 0, 1)                      # [C(+1),H,W]
        if self.delay_channel is not None:
            # tau = tau_scatter + |p - mu| / c: the second term is the alpha-composited
            # depth gsplat renders natively (sum_i w_i d_i, metres); the learned channel
            # keeps only the view-independent part. In the channel's normalised units.
            depth_m = img[self.channels]
            if self.delay_range == "euclid":
                # gsplat's depth is the camera z; the path length is z / cos(theta) along the pixel's ray
                # (1.41 at the edge and 1.56 at the corner of a 90-degree face). Measured on the lobby: the
                # gap (range - z) / c is 2.2 ns mean, 4.6 ns P90 (depth_gt.py), a known per-pixel factor.
                key = (float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2]), width, height)
                if self._range_factor is None or self._range_factor[0] != key:
                    u = (torch.arange(width, device=device, dtype=torch.float32) + 0.5 - key[2]) / key[0]
                    v = (torch.arange(height, device=device, dtype=torch.float32) + 0.5 - key[3]) / key[1]
                    self._range_factor = (key, torch.sqrt(1.0 + u[None, :] ** 2 + v[:, None] ** 2))
                depth_m = depth_m * self._range_factor[1]
            img = img[:self.channels].clone()
            img[self.delay_channel] = img[self.delay_channel] + depth_m / (0.299792458 * self.delay_span_ns)
        if self.mode == "power":
            img = torch.log10(img + 1e-12) * 10.0 / span_db
        return img

    def load_state(self, st):
        """rrf_state.pt of another run; accepts the earlier layouts too."""
        p = self.params
        if "sh0" in st:
            p["sh0"].data.copy_(st["sh0"]); p["shN"].data.copy_(st["shN"])
            p["opacities"].data.copy_(st["opacities"].reshape(-1))
            for k in ("means", "scales", "quats"):
                if k in st and st[k].shape == p[k].shape:
                    p[k].data.copy_(st[k])
            return
        for name, key in (("sh0", "sh_dc"), ("shN", "sh_rest")):
            t = st[key]
            if t.shape != p[name].shape:
                t = t.permute(0, 2, 1).contiguous()     # the older [N, C, K] layout
            p[name].data.copy_(t)
        p["opacities"].data.copy_(st["opacity_logit"].reshape(-1))

    def state(self):
        """Everything rrf_state.pt holds: the raw parameters plus the two fields
        needed to reconstruct the model that produced them."""
        return {k: v.data for k, v in self.params.items()} | {"mode": self.mode, "sh_degree": self.sh_degree}


# --------------------------------------------------------------------------
# evaluation
# --------------------------------------------------------------------------
def psnr(a, b):
    """PSNR in dB for tensors in [0, 1]. The epsilon bounds identical inputs at
    about 120 dB instead of returning inf."""
    mse = ((a - b) ** 2).mean()
    return float(20 * torch.log10(1.0 / torch.sqrt(mse + 1e-12)))


@torch.no_grad()
def evaluate(model, data, idx, span, vmin, save_dir=None):
    """dB RMSE against the float truth, and PSNR/SSIM on jet RGB for every mode."""
    tot = {"psnr_rgb": 0.0, "ssim_rgb": 0.0, "rmse_db": 0.0, "mae_db": 0.0,
           "rmse_db_in_range": 0.0, "n": 0}
    have_float = "float" in data
    for i in idx:
        img = model.render(data["viewmats"][i], data["Ks"][i], data["width"], data["height"], span)
        gt_rgb = data["rgb"][i].float() / 255.0
        if model.mode == "rgb":
            pred_rgb = img.clamp(0, 1)
            pred_norm = jet_inverse(pred_rgb)
        else:
            pred_norm = img[0].clamp(0, 1)
            pred_rgb = jet_rgb(pred_norm)
        tot["psnr_rgb"] += psnr(pred_rgb, gt_rgb)
        tot["ssim_rgb"] += float(ssim(pred_rgb[None], gt_rgb[None]))
        if have_float:
            gt_db = data["float"][i].float()
            pred_db = pred_norm * span + vmin
            tot["rmse_db"] += float(torch.sqrt(((pred_db - gt_db) ** 2).mean()))
            tot["mae_db"] += float((pred_db - gt_db).abs().mean())
            # against the truth clipped to the colormap range: what the
            # representation could have got right (pixels under the floor of
            # the range are unreachable for every mode alike)
            gt_c = gt_db.clamp(vmin, vmin + span)
            tot["rmse_db_in_range"] += float(torch.sqrt(((pred_db - gt_c) ** 2).mean()))
        else:
            gt_norm = jet_inverse(gt_rgb)
            e = float(torch.sqrt(((pred_norm - gt_norm) ** 2).mean())) * span
            tot["rmse_db"] += e
            tot["rmse_db_in_range"] += e
            tot["mae_db"] += float((pred_norm - gt_norm).abs().mean()) * span
        tot["n"] += 1
        if save_dir is not None:
            from PIL import Image
            os.makedirs(save_dir, exist_ok=True)
            Image.fromarray((pred_rgb.permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
                            ).save(os.path.join(save_dir, data["names"][i] + ".png"))
            np.save(os.path.join(save_dir, data["names"][i] + ".npy"),
                    (pred_norm * span + vmin).cpu().numpy().astype(np.float32))
    n = max(tot.pop("n"), 1)
    return {k: v / n for k, v in tot.items()}


@torch.no_grad()
def evaluate_multi(model, data, idx, ch_ranges, channel_names, save_dir=None, mask_channel=0):
    """Per-channel errors in native units on pixels the mask channel reaches;
    PSNR/SSIM of the mask channel (the power) after jet mapping."""
    lo, hi = ch_ranges[:, 0, None, None], ch_ranges[:, 1, None, None]
    mc = mask_channel
    span0 = float(hi[mc, 0, 0] - lo[mc, 0, 0])
    tot = {"psnr_rgb": 0.0, "ssim_rgb": 0.0, "rmse_db": 0.0, "mae_db": 0.0, "rmse_db_in_range": 0.0, "n": 0}
    pair = ("aod_az_cos" in channel_names) and ("aod_az_sin" in channel_names)
    for c in range(len(channel_names)):
        if c != mc and channel_names[c] not in ("aod_az_cos", "aod_az_sin"):
            tot[f"rmse_{channel_names[c]}"] = 0.0
    if pair:
        tot["rmse_aod_az"] = 0.0; tot["median_aod_az"] = 0.0
    for i in idx:
        img = model.render(data["viewmats"][i], data["Ks"][i], data["width"], data["height"], span0).clamp(0, 1)
        gt_f = data["float"][i].float()
        gt_n = ((gt_f - lo) / (hi - lo)).clamp(0, 1)
        pred_f = img * (hi - lo) + lo
        gt_rgb = data["rgb"][i].float() / 255.0
        pred_rgb = jet_rgb(img[mc])
        tot["psnr_rgb"] += psnr(pred_rgb, gt_rgb)
        tot["ssim_rgb"] += float(ssim(pred_rgb[None], gt_rgb[None]))
        d0 = pred_f[mc] - gt_f[mc]
        tot["rmse_db"] += float(torch.sqrt((d0 ** 2).mean()))
        tot["mae_db"] += float(d0.abs().mean())
        tot["rmse_db_in_range"] += float(torch.sqrt(((pred_f[mc] - gt_f[mc].clamp(float(lo[mc, 0, 0]), float(hi[mc, 0, 0]))) ** 2).mean()))
        mask = gt_n[mc] > 0.02
        if pair and mask.any():
            ic, is_ = channel_names.index("aod_az_cos"), channel_names.index("aod_az_sin")
            az_p = torch.rad2deg(torch.atan2(pred_f[is_], pred_f[ic])); az_t = torch.rad2deg(torch.atan2(gt_f[is_], gt_f[ic]))
            d = ((az_p - az_t + 180.0) % 360.0 - 180.0)[mask]
            tot["rmse_aod_az"] += float(torch.sqrt((d ** 2).mean())); tot["median_aod_az"] += float(d.abs().median())
        for c in range(len(channel_names)):
            if c == mc or channel_names[c] in ("aod_az_cos", "aod_az_sin"):
                continue
            d = pred_f[c] - gt_f[c]
            name = channel_names[c]
            if name == "aod_az":                  # channel is (phi+180)/360: wrap-aware, in degrees
                d = (d * 360.0 + 180.0) % 360.0 - 180.0
            elif name == "aod_zen":
                d = d * 180.0
            tot[f"rmse_{name}"] += float(torch.sqrt((d[mask] ** 2).mean())) if mask.any() else 0.0
        tot["n"] += 1
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            np.save(os.path.join(save_dir, data["names"][i] + ".npy"), pred_f.cpu().numpy().astype(np.float32))
    n = max(tot.pop("n"), 1)
    return {k: v / n for k, v in tot.items()}


# --------------------------------------------------------------------------
def main():
    """Load the dataset and checkpoint, run the fine-tune, evaluate and save.

    Writes into --out: results.json (config, per-eval history, final metrics and
    wall time), rrf_state.pt (the parameters) and, with --save-renders, the
    held-out predictions in both PNG and .npy form.
    """
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, help="dataset dir: images/, sparse/0, [spectra_float/]")
    ap.add_argument("--checkpoint", default=os.path.join(REPO, "RF-3DGS_dataset/blender_visual_trained/chkpnt30000.pth"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["rgb", "db", "power", "multi"], default="rgb",
                    help="multi: one channel per float channel of the dataset (generation_meta "
                         "channel_ranges), L1 on channels 1.. masked to pixels the power channel "
                         "reaches; SSIM and PSNR on channel 0")
    ap.add_argument("--sh-degree", type=int, default=3)
    ap.add_argument("--iterations", type=int, default=10_000, help="RF-3DGS: 30k -> 40k")
    ap.add_argument("--feature-lr", type=float, default=0.0025)
    ap.add_argument("--opacity-lr", type=float, default=0.05)
    ap.add_argument("--freeze-opacity", action="store_true")
    ap.add_argument("--train-geometry", action="store_true",
                    help="unfreeze means/scales/quats (INRIA lrs; means at 10x its final lr)")
    ap.add_argument("--densify", choices=["none", "mcmc"], default="none",
                    help="mcmc: gsplat's MCMC strategy (implies --train-geometry)")
    ap.add_argument("--cap-max", type=int, default=None, help="MCMC cap; default: the checkpoint's count")
    ap.add_argument("--geometry-fraction", type=float, default=0.11, help="size of the random geometry subset")
    ap.add_argument("--geometry-seed", type=int, default=0)
    ap.add_argument("--geometry-subset", choices=["all", "needles", "discs", "random"], default="all",
                    help="with --train-geometry: let only the needle-like (s_max/s_mid > 3, s_mid/s_min < 3) or "
                         "disc-like (s_mid/s_min > 3, s_max/s_mid < 3) Gaussians of the checkpoint move")
    ap.add_argument("--mask-channel", type=int, default=0,
                    help="multi: the channel whose value above 0.02 marks pixels a path reaches "
                         "(0 for MULTI's power; 2 for AOD3's amplitude)")
    ap.add_argument("--lambda-dssim", type=float, default=0.2)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--eval-subset", type=int, default=64, help="test views for the running eval")
    ap.add_argument("--max-train-views", type=int, default=None, help="data-efficiency ablation")
    ap.add_argument("--subset-mode", choices=["route", "fps"], default="route",
                    help="how --max-train-views picks positions: evenly along the training list (route) or farthest-point sampling in space (fps)")
    ap.add_argument("--delay-depth-mode", choices=["D", "ED"], default="D",
                    help="--delay-depth range term: accumulated depth sum w d (D) or expected depth sum w d / alpha (ED)")
    ap.add_argument("--delay-range", choices=["z", "euclid"], default="z",
                    help="--delay-depth range term as the camera z (z, as the first runs) or the Euclidean range z * sec(theta_pixel) (euclid)")
    ap.add_argument("--init-from", default=None, help="rrf_state.pt of another run (warm start)")
    ap.add_argument("--init-geometry-only", action="store_true",
                    help="with --init-from: take means/scales/quats/opacities from that run but "
                         "start the colours from zero (does an RF-adapted geometry transfer to "
                         "another transmitter?)")
    ap.add_argument("--save-renders", type=int, default=-1,
                    help="test renders to write; -1 (default) = every held-out view, so the decoded metrics (eval_baselines.py) "
                         "always cover the full test set. A run with fewer saved renders cannot support a full-test-set claim.")
    ap.add_argument("--db-range", type=float, nargs=2, default=None,
                    help="min max dB for the colormap; default from generation_meta.json")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--delay-depth", action="store_true",
                    help="multi mode: add the rendered depth / c to the delay channel (analytic range + learned residual)")
    ap.add_argument("--no-eval", action="store_true",
                    help="skip the final test pass (adaptation runs whose only product is the geometry)")
    cfg = ap.parse_args()
    if cfg.densify == "mcmc":
        cfg.train_geometry = True

    torch.manual_seed(cfg.seed)
    device = torch.device("cuda")
    t_start = time.time()

    # -- range of the colormap: what a normalised value means in dB ----------
    meta_path = os.path.join(cfg.source, "generation_meta.json")
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
    if cfg.db_range:
        vmin, vmax = cfg.db_range
    elif meta:
        vmin, vmax = meta["spec_min_db"], meta["spec_max_db"]
    else:
        vmin, vmax = 0.0, 1.0      # released data: no float truth, dB numbers are relative
        print("no generation_meta.json: dB errors are in normalised units x span=1")
    span = vmax - vmin
    channel_names = meta.get("channels")
    ch_ranges = None
    if cfg.mode == "multi":
        if not channel_names:
            raise SystemExit("--mode multi needs a multi-channel dataset (channels in generation_meta.json)")
        ch_ranges = torch.tensor(meta["channel_ranges"], device=device, dtype=torch.float32)  # [C,2]

    views = read_colmap_text(os.path.join(cfg.source, "sparse", "0"))
    train_names, test_names = ensure_split(cfg.source)
    if cfg.max_train_views:
        # Keep whole positions (all four yaws), not every k-th image: images
        # are position-major, so every 4th image would be the same yaw at
        # every position, and the model would never see the other three.
        per_pos, n_pos = 4, len(train_names) // 4
        keep = max(1, cfg.max_train_views // per_pos)
        if cfg.subset_mode == "fps":
            # farthest-point sampling over the positions: the most even spatial coverage a
            # subset of this size can have (the route order interleaves passes of the walk)
            pos = np.array([-(views[train_names[p * per_pos] + ".png"][0][:3, :3].T @ views[train_names[p * per_pos] + ".png"][0][:3, 3]) for p in range(n_pos)])
            chosen = [0]; dmin = np.linalg.norm(pos - pos[0], axis=1)
            while len(chosen) < keep:
                j = int(dmin.argmax()); chosen.append(j); dmin = np.minimum(dmin, np.linalg.norm(pos - pos[j], axis=1))
            pos_idx = np.array(sorted(chosen))
        else:
            pos_idx = np.linspace(0, n_pos - 1, keep).round().astype(int)
        train_names = [train_names[p * per_pos + k] for p in pos_idx for k in range(per_pos)]
    t0 = time.time()
    train = load_views(cfg.source, train_names, views, device, want_float=True)
    test = load_views(cfg.source, test_names, views, device, want_float=True)
    if cfg.mode in ("db", "power") and "float" not in train:
        # the released RF-3DGS data ship only the jet PNGs: invert the colormap (nearest LUT entry) and train on
        # that value in [0, 1]. The dB range is unknown, so every dB number of this run is in colour-range units.
        #
        # That inversion is only meaningful if the PNGs really were WRITTEN through jet, and two of the six
        # released datasets were not: measured against the 256-entry jet curve, the mean distance of a pixel to
        # the nearest jet colour is 0.5-3.1 for CBF/MVDR/TCBF/MPC but 69.5 for Delay and 117.5 for AoD -- 0.0% of
        # their pixels lie within 12 of the curve. Delay has G == B in 22767360 of 22767360 held-out pixels, so it
        # is a two-channel packing rather than a colormap lookup at all, and AoD is near-greyscale. Inverting
        # those returns an arbitrary value per pixel, and the runs that did it scored 6.6 to 10.0 dB BELOW their
        # rgb-mode counterparts while db mode beat rgb by about 1 dB on all four jet datasets. Refuse rather than
        # train against a meaningless target again.
        check_jet_encoded(train["rgb"])
        print("no spectra_float/: the value target is the jet-inverted PNG, in normalised units")
        for d in (train, test):
            d["float"] = torch.stack([jet_inverse(d["rgb"][i].float() / 255.0) for i in range(len(d["names"]))]).half()
    print(f"loaded {len(train_names)} train / {len(test_names)} test views in {time.time()-t0:.0f} s; "
          f"float truth: {'yes' if 'float' in test else 'no'}; range {vmin:.1f}..{vmax:.1f} dB")

    channels = 3 if cfg.mode == "rgb" else (int(ch_ranges.shape[0]) if cfg.mode == "multi" else 1)
    model = RRF(cfg.checkpoint, cfg.mode, channels, cfg.sh_degree, device,
                train_opacity=not cfg.freeze_opacity, train_geometry=cfg.train_geometry)
    if cfg.init_from:
        model.load_state(torch.load(cfg.init_from, map_location=device))
        if cfg.init_geometry_only:
            model.params["sh0"].data.zero_(); model.params["shN"].data.zero_()
            print(f"geometry (and opacity) from {cfg.init_from}, colours from zero")
        else:
            print(f"warm start from {cfg.init_from}")
    print(f"{model.n_gaussians:,} Gaussians from visual iteration {model.visual_iteration}; "
          f"mode {cfg.mode}, {channels} channel(s), SH degree {cfg.sh_degree}; "
          f"geometry {'trained' if cfg.train_geometry else 'frozen'}, densify {cfg.densify}")

    # INRIA's groups: f_dc at feature_lr, f_rest at feature_lr/20, opacity at
    # opacity_lr; geometry (when trained) at INRIA's scaling/rotation lrs and
    # the means at ten times INRIA's final position lr. One Adam per tensor,
    # which is what gsplat's strategies expect.
    lrs = {"sh0": cfg.feature_lr, "shN": cfg.feature_lr / 20.0, "opacities": cfg.opacity_lr,
           "means": 1.6e-5 * model.spatial_lr_scale, "scales": 5e-3, "quats": 1e-3}
    optimizers = {name: torch.optim.Adam([p], lr=lrs[name], eps=1e-15)
                  for name, p in model.params.items() if p.requires_grad}
    geom_mask = None
    if cfg.train_geometry and cfg.geometry_subset != "all":
        sc = torch.sort(torch.exp(model.params["scales"].detach()), dim=1).values      # min, mid, max
        mid_min, max_mid = sc[:, 1] / sc[:, 0], sc[:, 2] / sc[:, 1]
        if cfg.geometry_subset == "needles":
            sel = (max_mid > 3) & (mid_min < 3)
        elif cfg.geometry_subset == "discs":
            sel = (mid_min > 3) & (max_mid < 3)
        else:                                   # random control of a given size: no shape criterion at all
            g = torch.Generator().manual_seed(cfg.geometry_seed)
            sel = (torch.rand(sc.shape[0], generator=g) < cfg.geometry_fraction).to(sc.device)
        geom_mask = sel.float()
        print(f"geometry trains on the {cfg.geometry_subset} only: {int(sel.sum()):,} of {sel.numel():,} Gaussians ({sel.float().mean():.1%})")
    strategy = state = None
    if cfg.densify == "mcmc":
        from gsplat.strategy import MCMCStrategy
        strategy = MCMCStrategy(cap_max=cfg.cap_max or model.n_gaussians, refine_start_iter=500,
                                refine_stop_iter=int(0.8 * cfg.iterations), refine_every=100,
                                min_opacity=0.005, verbose=False)
        strategy.check_sanity(model.params, optimizers)
        state = strategy.initialize_state()
    if cfg.delay_depth:
        if cfg.mode != "multi" or "delay_ns" not in channel_names:
            raise SystemExit("--delay-depth needs multi mode with a delay_ns channel")
        model.delay_channel = channel_names.index("delay_ns")
        model.delay_span_ns = float(ch_ranges[model.delay_channel, 1] - ch_ranges[model.delay_channel, 0])
        model.delay_depth_mode = cfg.delay_depth_mode
        model.delay_range = cfg.delay_range
        print(f"delay channel {model.delay_channel}: rendered depth / c added, span {model.delay_span_ns:.1f} ns")

    def target(i):
        """Training view i as the tensor this mode's loss expects, in [0, 1].

        Every mode normalises to [0, 1] so one set of learning rates works
        across all of them: rgb from the PNG, multi per channel from the
        dataset's channel_ranges, db and power from the single global range.
        """
        if cfg.mode == "rgb":
            return train["rgb"][i].float() / 255.0
        if cfg.mode == "multi":
            f = train["float"][i].float()                                  # [C,H,W]
            lo, hi = ch_ranges[:, 0, None, None], ch_ranges[:, 1, None, None]
            return ((f - lo) / (hi - lo)).clamp(0, 1)
        return ((train["float"][i].float() - vmin) / span).clamp(0, 1)[None]

    def multi_loss(img, gt):
        """L1 on every channel, channels 1.. only where the power channel is above
        the floor (an angle or delay means nothing where no path arrives); SSIM on
        the power channel."""
        mc = cfg.mask_channel
        mask = (gt[mc] > 0.02).float()
        others = [c for c in range(gt.shape[0]) if c != mc]
        l1 = (img[mc] - gt[mc]).abs().mean()
        if others:
            l1 = l1 + ((img[others] - gt[others]).abs() * mask[None]).sum() / (mask.sum() * len(others) + 1)
        return (1 - cfg.lambda_dssim) * l1 + cfg.lambda_dssim * (1 - ssim(img[mc:mc+1][None], gt[mc:mc+1][None]))

    n_train = len(train_names)
    eval_idx = list(range(0, len(test_names), max(1, len(test_names) // cfg.eval_subset)))[:cfg.eval_subset]
    history = []
    torch.cuda.synchronize(); t_train = time.time()
    rng = np.random.default_rng(cfg.seed)
    for it in range(1, cfg.iterations + 1):
        i = int(rng.integers(n_train))
        img = model.render(train["viewmats"][i], train["Ks"][i], train["width"], train["height"], span)
        gt = target(i)
        if cfg.mode == "multi":
            loss = multi_loss(img, gt)
        else:
            loss = (1 - cfg.lambda_dssim) * l1_loss(img, gt) + cfg.lambda_dssim * (1 - ssim(img[None], gt[None]))
        if strategy is not None:
            # gsplat's MCMC regularisers, its defaults
            loss = loss + 0.01 * model.opacities.abs().mean() + 0.01 * model.scales.abs().mean()
            strategy.step_pre_backward(model.params, optimizers, state, it, model.last_info)
        for opt in optimizers.values():
            opt.zero_grad(set_to_none=True)
        loss.backward()
        if geom_mask is not None:
            for name in ("means", "scales", "quats"):
                g = model.params[name].grad
                if g is not None:
                    g.mul_(geom_mask[:, None])
        for opt in optimizers.values():
            opt.step()
        if strategy is not None:
            strategy.step_post_backward(model.params, optimizers, state, it, model.last_info,
                                        lr=lrs["means"])
        if it % cfg.eval_every == 0 or it == cfg.iterations:
            torch.cuda.synchronize()
            m = (evaluate_multi(model, test, eval_idx, ch_ranges, channel_names, mask_channel=cfg.mask_channel)
                 if cfg.mode == "multi"
                 else evaluate(model, test, eval_idx, span, vmin))
            m.update(iteration=it, seconds=time.time() - t_train, loss=float(loss.detach()),
                     gaussians=model.n_gaussians)
            history.append(m)
            print(f"  it {it:6d}  {m['seconds']:6.0f} s  loss {m['loss']:.4f}  "
                  f"PSNR(jet) {m['psnr_rgb']:5.2f}  SSIM {m['ssim_rgb']:.3f}  "
                  f"RMSE {m['rmse_db']:5.2f} dB  (subset {len(eval_idx)}"
                  + (f", {m['gaussians']:,} Gaussians" if strategy is not None else "") + ")")
    torch.cuda.synchronize(); train_seconds = time.time() - t_train

    os.makedirs(cfg.out, exist_ok=True)
    all_idx = list(range(len(test_names)))
    if cfg.no_eval:
        # an adaptation run whose only product is its geometry: skip the test pass
        final = None
    elif cfg.mode == "multi":
        final = evaluate_multi(model, test, all_idx, ch_ranges, channel_names, mask_channel=cfg.mask_channel)
        if cfg.save_renders:
            evaluate_multi(model, test, (all_idx if cfg.save_renders < 0 else all_idx[:cfg.save_renders]), ch_ranges, channel_names,
                           save_dir=os.path.join(cfg.out, "renders"), mask_channel=cfg.mask_channel)
    else:
        final = evaluate(model, test, all_idx, span, vmin)
        if cfg.save_renders:
            evaluate(model, test, (all_idx if cfg.save_renders < 0 else all_idx[:cfg.save_renders]), span, vmin,
                     save_dir=os.path.join(cfg.out, "renders"))
    torch.save(model.state(), os.path.join(cfg.out, "rrf_state.pt"))
    result = {"config": vars(cfg), "db_range": [vmin, vmax], "n_train": n_train,
              "channels": channel_names, "channel_ranges": meta.get("channel_ranges"),
              "n_test": len(test_names), "gaussians": model.n_gaussians,
              "train_seconds": train_seconds, "iters_per_second": cfg.iterations / train_seconds,
              "total_seconds": time.time() - t_start, "history": history, "final": final}
    with open(os.path.join(cfg.out, "results.json"), "w") as fid:
        json.dump(result, fid, indent=1)
    if final is None:
        print(f"\nno final evaluation (--no-eval); {cfg.iterations} iterations in {train_seconds:.0f} s "
              f"({cfg.iterations/train_seconds:.0f} it/s); {model.n_gaussians:,} Gaussians; wrote {cfg.out}")
        return
    print(f"\nfinal on {len(test_names)} test views: PSNR(jet) {final['psnr_rgb']:.2f} dB, "
          f"SSIM {final['ssim_rgb']:.3f}, RMSE {final['rmse_db']:.2f} dB, MAE {final['mae_db']:.2f} dB; "
          f"{cfg.iterations} iterations in {train_seconds:.0f} s "
          f"({cfg.iterations/train_seconds:.0f} it/s); {model.n_gaussians:,} Gaussians; wrote {cfg.out}")


if __name__ == "__main__":
    main()
