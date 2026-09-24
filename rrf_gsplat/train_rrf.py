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
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from losses import l1_loss, ssim                    # noqa: E402  (INRIA's, copied: losses.py)

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
    from concurrent.futures import ThreadPoolExecutor
    from PIL import Image
    rgb, flt, viewmats, Ks = [], [], [], []
    have_float = want_float and os.path.isdir(os.path.join(source, "spectra_float"))
    width = height = None

    def read(n):
        img = np.array(Image.open(os.path.join(source, "images", n + ".png")).convert("RGB"))
        f = np.load(os.path.join(source, "spectra_float", n + ".npy")) if have_float else None
        # single-channel spectra stay float32 (~0.7 GB for 3200 views; float16 quantised dB to 0.125 dB above
        # 128 dB, since 2026-09-24); multichannel ones float16, or 10k five-channel views would not fit (12 GB)
        if f is not None:
            f = f.astype(np.float32 if f.ndim == 2 else np.float16)
        return img, f
    # Read in parallel, in order: from WSL every file crosses the 9p mount, whose per-file latency (not its
    # bandwidth) made a 3200-view load 38 s sequentially; 16 threads read the same files in about a fifth of that.
    with ThreadPoolExecutor(16) as ex:
        loaded = list(ex.map(read, names))
    for n, (img, f) in zip(names, loaded):
        rgb.append(torch.from_numpy(img).permute(2, 0, 1))
        if have_float:
            flt.append(torch.from_numpy(f))
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
    # Everything resident on the GPU: uint8 for the PNGs, float32 for single-channel spectra and float16 for
    # multichannel ones, which is what makes a whole multichannel dataset fit alongside the model.
    out = {"rgb": torch.stack(rgb).to(device), "viewmats": torch.stack(viewmats).to(device),
           "Ks": torch.stack(Ks).to(device), "names": names, "width": width, "height": height}
    if have_float:
        out["float"] = torch.stack(flt).to(device)
    return out


# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------
# channel counts gsplat's rasteriser is compiled for (gsplat/cuda/csrc/Config.h, GSPLAT_NUM_CHANNELS; depth counts)
GSPLAT_CHANNELS = (1, 2, 3, 4, 5, 6, 8, 9, 16, 17, 21, 23, 24, 32, 33, 64, 65, 128, 129, 256, 257, 512, 513)
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
        self.sh_backend = "torch"            # --sh-backend gsplat: the same SH in gsplat's CUDA kernel (frozen geometry only)
        self.head = None                     # --head cnn: an image-space residual network after the rasteriser
        self.head_cfg = None                 # its guide buffers, latent width, strip layout, bound (attach_head)
        self.head_on = True                  # False during --head-warmup: the Gaussians alone
        self._normals = None                # the Gaussians' shortest axes, cached while the geometry is frozen
        self.res_stats = None                # [sum res^2, sum (out - mean)^2, count], channel 0, while evaluating

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
        col = self._sh_colours(cam_center)
        if "lobe_w" in self.params:
            col = col + self.lobe_colours(cam_center)
        if getattr(self, "pcolor_mlp", None) is not None:
            col = col + self.pcolor(cam_center)
        return col

    def pcolor(self, cam_center):
        """P7 (docs/tech_paths.md): a position-conditioned colour residual. A shared MLP maps each Gaussian's latent,
        its direction from the receiver and the receiver position (Fourier features over the scene's extent) to a
        value added to the SH colour -- so a Gaussian's value may depend on where the receiver is, not only on the
        direction it is seen from. The last layer starts at zero: the model starts exactly as the SH one."""
        z = self.params["pc_latent"]                                                          # [N, W]
        d = F.normalize(self.means.detach() - cam_center[None], dim=-1)                       # [N, 3]
        r = (cam_center - self._pc_centre) / self._pc_scale                                   # [3], about [-1, 1]
        f = torch.cat([torch.sin(r[None] * self._pc_freqs[:, None]), torch.cos(r[None] * self._pc_freqs[:, None])], -1).reshape(-1)
        x = torch.cat([z, d, f[None].expand(z.shape[0], -1)], -1)
        return self.pcolor_mlp(x)                                                             # [N, C]

    def add_pcolor(self, width, hidden, n_freqs, receivers):
        dev, n = self.params["means"].device, self.n_gaussians
        g0 = torch.Generator(device="cpu").manual_seed(2)
        self.params["pc_latent"] = torch.nn.Parameter((0.1 * torch.randn(n, width, generator=g0)).to(dev))
        rx = receivers.to(dev)
        self._pc_centre = rx.mean(0); self._pc_scale = (rx - self._pc_centre).abs().max().clamp_min(1e-3)
        self._pc_freqs = (2.0 ** torch.arange(n_freqs, device=dev, dtype=torch.float32)) * math.pi
        n_in = width + 3 + 6 * n_freqs
        last = torch.nn.Linear(hidden, self.channels)
        torch.nn.init.zeros_(last.weight); torch.nn.init.zeros_(last.bias)
        self.pcolor_mlp = torch.nn.Sequential(torch.nn.Linear(n_in, hidden), torch.nn.SiLU(),
                                              torch.nn.Linear(hidden, hidden), torch.nn.SiLU(), last).to(dev)
        self.pcolor_cfg = {"width": width, "hidden": hidden, "n_freqs": n_freqs,
                           "centre": self._pc_centre.tolist(), "scale": float(self._pc_scale)}

    def lobe_colours(self, cam_center):
        """P2 (docs/tech_paths.md): K spherical-Gaussian lobes per Gaussian on top of the SH colour,
        sum_k w_k exp(kappa_k (m_k . d - 1)) with d the same Gaussian-from-camera direction the SH uses. A lobe's
        width is about 1 / sqrt(kappa) rad, so it can be far sharper than SH3's ~45 deg band limit."""
        d = F.normalize(self.means.detach() - cam_center[None], dim=-1)                      # [N, 3]
        m = F.normalize(self.params["lobe_axis"], dim=-1)                                    # [N, K, 3]
        kappa = torch.exp(self.params["lobe_logk"])                                          # [N, K]
        g = torch.exp(kappa * ((m * d[:, None, :]).sum(-1) - 1.0))                           # [N, K]
        return (g[:, :, None] * self.params["lobe_w"]).sum(1)                                # [N, C]

    def add_lobes(self, k, kappa_init, receivers):
        """K lobes per Gaussian, weights zero (the model starts exactly as the SH one), axes pointing from the mean
        receiver position to the Gaussian (where a receiver can see the lobe), kappa = kappa_init."""
        dev, n = self.params["means"].device, self.n_gaussians
        g0 = torch.Generator(device="cpu").manual_seed(1)
        base = F.normalize(self.means.detach() - receivers.mean(0).to(dev)[None], dim=-1)    # [N, 3]
        jitter = 0.1 * torch.randn(n, k, 3, generator=g0).to(dev)
        self.params["lobe_axis"] = torch.nn.Parameter(F.normalize(base[:, None, :] + jitter, dim=-1))
        self.params["lobe_logk"] = torch.nn.Parameter(torch.full((n, k), math.log(kappa_init), device=dev))
        self.params["lobe_w"] = torch.nn.Parameter(torch.zeros(n, k, self.channels, device=dev))

    def _sh_colours(self, cam_center, means=None, sh=None, frozen=None):
        means = self.means if means is None else means
        sh = self.sh if sh is None else sh
        frozen = (not self.params["means"].requires_grad) if frozen is None else frozen
        if self.sh_degree == 0:
            return sh[:, 0, :] * 0.28209479177387814 + 0.5
        if self.sh_backend == "gsplat" and frozen:
            # gsplat's SH kernel takes any channel count and reads the camera position from a view
            # matrix as -R^T t, so an identity rotation with t = -c places the camera at c. Same basis,
            # same coefficients: 6e-8 from the torch path forward, and a third of its time with backward.
            from gsplat import spherical_harmonics
            vm = torch.eye(4, device=cam_center.device); vm[:3, 3] = -cam_center
            return spherical_harmonics(self.sh_degree, means, vm[None], sh)[0] + 0.5
        ctx = torch.no_grad() if frozen else torch.enable_grad()
        with ctx:
            dirs = F.normalize(means - cam_center[None], dim=-1)
            basis = sh_basis(self.sh_degree, dirs)                  # [N, K]
        return (sh * basis[:, :, None]).sum(dim=1) + 0.5

    EM_VMAX = 1.5                  # an emitter's value v = EM_VMAX * sigmoid(SH): up to 1.5 x the span above the floor

    def add_emitters(self, means, scale_m, opacity=0.05, v0=0.02):
        """M3's placement oracle (docs/tech_paths.md; sionna_port/path_emitters.py): extra isotropic Gaussians at
        given points -- the ray tracer's interaction points -- rendered in their OWN pass and added to the field in
        linear power (power mode only). The visual Gaussians do not occlude them, so this is the most generous
        placement oracle: whether a peak can be drawn where the energy really comes from. Geometry fixed; only the
        SH (the emitted power per direction) learns.

        No dead zones (the first full run had them: 97 % of the emitters never lit):
          value     v = EM_VMAX * sigmoid(SH), power 10^(v span / 10) - 1: smooth everywhere, and v -> 0 adds nothing.
                    (With the visual Gaussians' clamp(0, 1) the values were pushed below 0 early, when the whole
                    render is too bright, and their gradient died there.)
          opacity   fixed at `opacity`: above gsplat's 1 / 255 alpha cut-off (a trained opacity decayed below it and
                    the rasteriser skipped those emitters), and small, so the pass is nearly additive (little
                    occlusion among emitters on one line of sight).
        Starts at v0, i.e. about opacity x (10^(v0 span / 10) - 1) of the floor's power: the plain model's render."""
        dev = self.params["means"].device
        m = torch.as_tensor(means, dtype=torch.float32, device=dev)
        n, k = m.shape[0], (self.sh_degree + 1) ** 2
        raw0 = math.log(v0 / self.EM_VMAX / (1 - v0 / self.EM_VMAX))
        self.params["em_means"] = torch.nn.Parameter(m, requires_grad=False)
        self.params["em_scales"] = torch.nn.Parameter(torch.full((n, 3), math.log(scale_m), device=dev), requires_grad=False)
        self.params["em_opacities"] = torch.nn.Parameter(torch.full((n,), math.log(opacity / (1 - opacity)), device=dev),
                                                         requires_grad=False)
        self.params["em_sh0"] = torch.nn.Parameter(torch.full((n, 1, self.channels), raw0 / 0.28209479177387814, device=dev))
        self.params["em_shN"] = torch.nn.Parameter(torch.zeros(n, k - 1, self.channels, device=dev))
        self._em_quats = torch.zeros(n, 4, device=dev); self._em_quats[:, 0] = 1.0

    EM_PC_GAIN = 4.0               # --em-pcolor's output is in logits of the emitter value: x4, so switching an emitter
                                   # on or off (about 4 logits) needs outputs of order 1 from the MLP

    def add_em_pcolor(self, width, hidden, n_freqs, receivers):
        """P7 for the emitters (M3 oracle, round 2): a per-emitter latent and a shared MLP of (latent, direction from
        the receiver, receiver position) added to the emitter's logit, so an emitter can be on for some receiver
        positions and off for others -- what one shared SH3 colour per emitter could not do (round 1: 25.1 -> 27.1 %).
        The last layer starts at zero: the model starts exactly as the emitters alone."""
        p = self.params
        dev, n = p["em_means"].device, p["em_means"].shape[0]
        g0 = torch.Generator(device="cpu").manual_seed(4)
        p["em_pc_latent"] = torch.nn.Parameter((0.1 * torch.randn(n, width, generator=g0)).to(dev))
        rx = receivers.to(dev)
        self._epc_centre = rx.mean(0); self._epc_scale = (rx - self._epc_centre).abs().max().clamp_min(1e-3)
        self._epc_freqs = (2.0 ** torch.arange(n_freqs, device=dev, dtype=torch.float32)) * math.pi
        last = torch.nn.Linear(hidden, self.channels)
        torch.nn.init.zeros_(last.weight); torch.nn.init.zeros_(last.bias)
        self.em_pcolor_mlp = torch.nn.Sequential(torch.nn.Linear(width + 3 + 6 * n_freqs, hidden), torch.nn.SiLU(),
                                                 torch.nn.Linear(hidden, hidden), torch.nn.SiLU(), last).to(dev)
        self.em_pcolor_cfg = {"width": width, "hidden": hidden, "n_freqs": n_freqs,
                              "centre": self._epc_centre.tolist(), "scale": float(self._epc_scale)}

    def em_pcolor(self, cam_center):
        z = self.params["em_pc_latent"]                                                        # [E, W]
        d = F.normalize(self.params["em_means"] - cam_center[None], dim=-1)                   # [E, 3]
        r = (cam_center - self._epc_centre) / self._epc_scale
        f = torch.cat([torch.sin(r[None] * self._epc_freqs[:, None]), torch.cos(r[None] * self._epc_freqs[:, None])], -1).reshape(-1)
        return self.EM_PC_GAIN * self.em_pcolor_mlp(torch.cat([z, d, f[None].expand(z.shape[0], -1)], -1))   # [E, C]

    def _emitter_raw(self, c, sh):
        raw = self._sh_colours(c, self.params["em_means"], sh, frozen=True) - 0.5
        if getattr(self, "em_pcolor_mlp", None) is not None:
            raw = raw + self.em_pcolor(c)
        return raw

    def _emitter_power(self, viewmats, Ks, width, height, span_db, centres):
        """The emitters' own rasterisation, linear power [B, C, H, W] on a zero background."""
        from gsplat import rasterization
        p = self.params
        sh = torch.cat([p["em_sh0"], p["em_shN"]], dim=1)
        if viewmats.shape[0] == 1 or float((centres - centres[:1]).norm(dim=-1).max()) < 1e-4:
            raw = self._emitter_raw(centres[0], sh)
        else:
            raw = torch.stack([self._emitter_raw(c, sh) for c in centres])
        v = self.EM_VMAX * torch.sigmoid(raw)
        col = torch.pow(10.0, v * span_db / 10.0) - 1.0
        img, _, _ = rasterization(p["em_means"], self._em_quats, torch.exp(p["em_scales"]), torch.sigmoid(p["em_opacities"]),
                                  col, viewmats, Ks, width, height, sh_degree=None,
                                  backgrounds=torch.zeros(viewmats.shape[0], col.shape[-1], device=col.device))
        return img.permute(0, 3, 1, 2)

    def render(self, viewmat, K, width, height, span_db):
        """One view, [C, H, W]: render_batch with a batch of one."""
        return self.render_batch(viewmat[None], K[None], width, height, span_db)[0]

    def render_batch(self, viewmats, Ks, width, height, span_db):
        """B views, [B, C, H, W]. Three paths depending on mode, plus the optional
        delay-depth decomposition; span_db is the dataset's dB range, needed only
        by the power mode and the delay term.

        The per-Gaussian value depends on the camera only through its position
        (the SH direction is mean - camera centre), so views that share a
        centre -- the four faces of one receiver position -- share one colour
        evaluation, and gsplat rasterises them in one batched call. That
        evaluation, not the rasteriser, is the per-view cost at 300 x 200
        (profile_resolution.py). Views at different centres get one each.
        """
        from gsplat import rasterization
        device = self.means.device
        B = viewmats.shape[0]
        if self.mode == "rgb" and self.sh_degree > 0:
            # gsplat evaluates 3-channel SH in CUDA (and adds INRIA's 0.5 and
            # clamps at 0 itself); its backward is a third of the torch path's
            img, alpha, info = rasterization(
                self.means, self.quats, self.scales, self.opacities, self.sh,
                viewmats, Ks, width, height, sh_degree=self.sh_degree,
                backgrounds=torch.zeros(B, 3, device=device))
            self.last_info = info
            return img.permute(0, 3, 1, 2)
        centres = torch.linalg.inv(viewmats)[:, :3, 3]                   # [B, 3]
        if B == 1 or float((centres - centres[:1]).norm(dim=-1).max()) < 1e-4:
            col = self.colours(centres[0])                               # [N, C], shared by every view
        else:
            col = torch.stack([self.colours(c) for c in centres])        # [B, N, C]
        if self.mode == "rgb":
            col = col.clamp_min(0.0)
        elif self.mode == "power":
            # value in [0,1] is dB above the floor as a fraction of the span;
            # composite linear powers, read back in dB
            col = torch.pow(10.0, col.clamp(0.0, 1.0) * span_db / 10.0)
        hc = self.head_cfg
        n_col = col.shape[-1]
        if hc is not None and (hc["normals"] or hc["latent"]):
            # the head's per-Gaussian inputs ride along in the same rasterisation: the normal, turned towards the
            # (shared) camera centre so that compositing does not cancel opposite signs, and the learned feature
            extras = []
            if hc["normals"]:
                if self._normals is None or self.params["means"].requires_grad:
                    from neural_shading import gaussian_normals
                    self._normals = gaussian_normals(self.params["quats"], self.params["scales"])
                def side(c):
                    return torch.where(((c[None] - self.means) * self._normals).sum(-1, keepdim=True) < 0, -1.0, 1.0)
                extras.append(self._normals * side(centres[0]) if col.dim() == 2
                              else torch.stack([self._normals * side(c) for c in centres]))
            if hc["latent"]:
                lat = self.params["latent"]
                extras.append(lat if col.dim() == 2 else lat[None].expand(B, -1, -1))
            col = torch.cat([col] + extras, -1)
        n_real = col.shape[-1]                             # colour + the head's extras, before any padding
        want_depth = self.delay_channel is not None or (hc is not None and hc["depth"])
        depth_mode = self.delay_depth_mode if self.delay_channel is not None else "ED"
        # gsplat's kernels are compiled for a fixed set of channel counts (depth included): multi's 5 + normal 3
        # + latent 4 + depth = 13 is not one of them, so the colour vector is padded with zeros up to the next
        # one. Only ever triggered by the head's extras: every earlier configuration already fits.
        total = n_real + (1 if want_depth else 0)
        if total not in GSPLAT_CHANNELS:
            pad = min(c for c in GSPLAT_CHANNELS if c >= total) - total
            col = torch.cat([col, col.new_zeros(col.shape[:-1] + (pad,))], -1)
        img, alpha, info = rasterization(
            self.means, self.quats, self.scales, self.opacities, col,
            viewmats, Ks, width, height, sh_degree=None,
            backgrounds=torch.zeros(B, col.shape[-1], device=device),
            render_mode=("RGB+" + depth_mode) if want_depth else "RGB")
        self.last_info = info
        img = img.permute(0, 3, 1, 2)                      # [B,C(+extras)(+1),H,W]
        head_in = None
        if hc is not None:
            depth_ed = None
            if want_depth:
                depth_ed = img[:, col.shape[-1]]
                if depth_mode == "D":                      # accumulated depth -> expected depth of the surface
                    depth_ed = depth_ed / alpha[..., 0].clamp_min(1e-4)
            head_in = (img[:, n_col:n_real], depth_ed)
        if col.shape[-1] > n_col:
            # drop the head's extras and any padding: the colour channels, then the depth, where the code below
            # expects them
            img = torch.cat([img[:, :n_col], img[:, col.shape[-1]:]], 1)
        if self.delay_channel is not None:
            # tau = tau_scatter + |p - mu| / c: the second term is the alpha-composited
            # depth gsplat renders natively (sum_i w_i d_i, metres); the learned channel
            # keeps only the view-independent part. In the channel's normalised units.
            K = Ks[0]                                      # one camera model per dataset (load_views checks the size)
            depth_m = img[:, self.channels]
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
            img = img[:, :self.channels].clone()
            img[:, self.delay_channel] = img[:, self.delay_channel] + depth_m / (0.299792458 * self.delay_span_ns)
        if "em_means" in self.params:
            img = img + self._emitter_power(viewmats, Ks, width, height, span_db, centres)   # linear power (power mode)
        if self.mode == "power":
            img = torch.log10(img + 1e-12) * 10.0 / span_db
        if hc is not None:
            img = img[:, :self.channels]                   # the depth rendered for the guides is not an output
            if self.head_on:
                img = self._apply_head(img, head_in[0], head_in[1], viewmats, Ks, width, height)
        return img

    def _apply_head(self, base, extra, depth, viewmats, Ks, width, height):
        """base [B, C, H, W] in normalised units -> base + bounded residual, from base + guide buffers + latent."""
        from neural_shading import geometry_guides, physics_guides, world_rays
        hc = self.head_cfg
        B = base.shape[0]
        parts = [base]
        normal = extra[:, :3] if hc["normals"] else None
        if hc["guides"]:
            rays, centres, cam_plane = world_rays(viewmats, Ks[0], height, width)
            if "geo" in hc["guides"]:
                parts.append(geometry_guides(depth, normal, rays, cam_plane, centres, hc["scene_scale"]))
            if "phys" in hc["guides"]:
                parts.append(physics_guides(depth, normal, rays, cam_plane, centres, viewmats, hc["tx"], hc["span_m"]))
        if hc["latent"]:
            parts.append(extra[:, 3 if hc["normals"] else 0:])
        x = torch.cat(parts, 1)
        if hc["strip"] and B == 4:
            # the four faces of one position laid side by side in azimuth order: a ring, padded circularly, so the
            # network sees across the seams the way the field runs across them
            order = hc["order"]
            ring = self.head(torch.cat([x[i] for i in order], -1)[None], pad="strip")[0]
            res = torch.stack([ring[..., order.index(i) * width:(order.index(i) + 1) * width] for i in range(B)])
        else:
            res = self.head(x, pad="replicate")
        out = base + res
        if self.res_stats is not None:
            r0, o0 = res[:, 0].detach(), out[:, 0].detach()
            self.res_stats[0] += float((r0 ** 2).sum())
            self.res_stats[1] += float(((o0 - o0.mean(dim=(1, 2), keepdim=True)) ** 2).sum())
            self.res_stats[2] += r0.numel()
        return out

    def load_state(self, st):
        """rrf_state.pt of another run; accepts the earlier layouts too."""
        p = self.params
        if "head" in st and self.head is not None:
            self.head.load_state_dict(st["head"])
        if "latent" in st and "latent" in p:
            p["latent"].data.copy_(st["latent"])
        if "em_pcolor_mlp" in st and getattr(self, "em_pcolor_mlp", None) is not None:
            self.em_pcolor_mlp.load_state_dict(st["em_pcolor_mlp"])
            c = st["em_pcolor_cfg"]
            self._epc_centre = torch.tensor(c["centre"], device=p["means"].device); self._epc_scale = torch.tensor(c["scale"], device=p["means"].device)
        for k in ("lobe_axis", "lobe_logk", "lobe_w", "pc_latent", "em_opacities", "em_sh0", "em_shN", "em_pc_latent"):
            if k in st and k in p and st[k].shape == p[k].shape:
                p[k].data.copy_(st[k])
        if "pcolor_mlp" in st and getattr(self, "pcolor_mlp", None) is not None:
            self.pcolor_mlp.load_state_dict(st["pcolor_mlp"])
            c = st["pcolor_cfg"]
            self._pc_centre = torch.tensor(c["centre"], device=p["means"].device); self._pc_scale = torch.tensor(c["scale"], device=p["means"].device)
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
        st = {k: v.data for k, v in self.params.items()} | {"mode": self.mode, "sh_degree": self.sh_degree}
        if self.head is not None:
            st["head"] = self.head.state_dict()
            st["head_cfg"] = {k: v for k, v in self.head_cfg.items() if k != "tx"}
        if getattr(self, "pcolor_mlp", None) is not None:
            st["pcolor_mlp"] = self.pcolor_mlp.state_dict(); st["pcolor_cfg"] = self.pcolor_cfg
        if getattr(self, "em_pcolor_mlp", None) is not None:
            st["em_pcolor_mlp"] = self.em_pcolor_mlp.state_dict(); st["em_pcolor_cfg"] = self.em_pcolor_cfg
        return st


def attach_head(model, head="cnn", guides="", latent=0, strip=False, max_db=6.0, width=32, span_db=1.0, tx=None,
                device="cuda", bound_units=False):
    """Give the model a deferred shading head (neural_shading.py): a residual CNN shared by every view.

    guides   comma list of "geo" (depth, normal, world ray, receiver position: 10 channels) and "phys"
             (direct-path alignment, mirror alignment, incidence, ranges to the transmitter: 5 channels)
    latent   a learned per-Gaussian feature of this width, rasterised beside the colour (the materials we do
             not know, in the place DLSS Ray Reconstruction takes albedo and roughness)
    strip    run the head on the four faces of a position side by side, padded circularly
    max_db   the residual's bound, in dB of the value (or power) channel; span_db converts it to value units
    bound_units   the output layer in units of the bound (ResidualCNN): off reproduces rounds 39-42
    """
    from neural_shading import ResidualCNN
    g = [x for x in guides.split(",") if x]
    if "phys" in g and tx is None:
        raise SystemExit("--head-guides phys needs the transmitter position (generation_meta.json tx_loc)")
    c = model.channels
    c_in = c + (10 if "geo" in g else 0) + (5 if "phys" in g else 0) + latent
    model.head = ResidualCNN(c_in, c, width=width, max_residual=max_db / span_db, bound_units=bound_units).to(device)
    model.head_cfg = {"head": head, "guides": g, "latent": latent, "strip": strip, "normals": bool(g), "depth": bool(g),
                      "tx": None if tx is None else torch.tensor(tx, dtype=torch.float32, device=device),
                      "scene_scale": 10.0, "span_m": 10.0, "order": [0, 1, 2, 3], "max_db": max_db, "width": width,
                      "bound_units": bound_units}
    if latent:
        n = model.n_gaussians
        g0 = torch.Generator(device="cpu").manual_seed(0)
        model.params["latent"] = torch.nn.Parameter((0.1 * torch.randn(n, latent, generator=g0)).to(device))
    return model.head


# --------------------------------------------------------------------------
# evaluation
# --------------------------------------------------------------------------
def spread_positions(viewmats, n):
    """About n view indices: whole receiver positions (runs of consecutive views sharing a camera centre, i.e. every
    face = every orientation) spread evenly over the list, first and last included."""
    c = torch.linalg.inv(viewmats)[:, :3, 3].cpu()
    groups, cur = [], [0]
    for i in range(1, c.shape[0]):
        if float((c[i] - c[cur[0]]).norm()) < 1e-4:
            cur.append(i)
        else:
            groups.append(cur); cur = [i]
    groups.append(cur)
    per = max(1, round(float(np.mean([len(g) for g in groups]))))
    k = min(len(groups), max(1, n // per))
    pick = np.linspace(0, len(groups) - 1, k).round().astype(int)
    return [i for j in sorted(set(pick.tolist())) for i in groups[j]][:n]


def psnr(a, b):
    """PSNR in dB for tensors in [0, 1]. The epsilon bounds identical inputs at
    about 120 dB instead of returning inf."""
    mse = ((a - b) ** 2).mean()
    return float(20 * torch.log10(1.0 / torch.sqrt(mse + 1e-12)))


class RenderWriter:
    """Writes the saved renders on background threads, each file serialised in memory and written in one call.

    np.save straight to a file under /mnt/c took about 100 ms per 1.2 MB render inside the evaluation (its
    ndarray.tofile path; the same call in a bare script takes 8 ms, and the cause was not isolated), which made
    writing 640 renders a minute. One write() of the serialised bytes takes 16 ms, and eight threads overlap
    it with the rendering. The file contents are the same bytes np.save would write.
    """

    def __init__(self, threads=8):
        from concurrent.futures import ThreadPoolExecutor
        self.ex, self.futs = ThreadPoolExecutor(threads), []

    @staticmethod
    def _write(path, data):
        with open(path, "wb") as fid:
            fid.write(data)

    def npy(self, path, arr):
        import io
        buf = io.BytesIO(); np.save(buf, np.ascontiguousarray(arr))
        self.futs.append(self.ex.submit(self._write, path, buf.getvalue()))

    def png(self, path, rgb_uint8):
        import io
        from PIL import Image
        buf = io.BytesIO(); Image.fromarray(rgb_uint8).save(buf, format="PNG")
        self.futs.append(self.ex.submit(self._write, path, buf.getvalue()))

    def close(self):
        for f in self.futs:
            f.result()                      # re-raises a failed write here rather than losing it
        self.ex.shutdown()


def render_views(model, data, idx, span, group=False, max_faces=4):
    """(i, [C, H, W]) for every i in idx, in order.

    With group, consecutive indices whose cameras share a centre (the faces of
    one receiver position; images are position-major) render in one batched
    call with one colour evaluation. The images are the same either way; only
    the work differs.
    """
    if not group:
        for i in idx:
            yield i, model.render(data["viewmats"][i], data["Ks"][i], data["width"], data["height"], span)
        return
    if "centres" not in data:
        data["centres"] = torch.linalg.inv(data["viewmats"])[:, :3, 3]
    c = data["centres"]
    run = []
    for i in idx:
        if run and (len(run) == max_faces or float((c[i] - c[run[0]]).norm()) > 1e-4):
            yield from zip(run, model.render_batch(data["viewmats"][run], data["Ks"][run], data["width"], data["height"], span))
            run = []
        run.append(i)
    if run:
        yield from zip(run, model.render_batch(data["viewmats"][run], data["Ks"][run], data["width"], data["height"], span))


@torch.no_grad()
def evaluate(model, data, idx, span, vmin, save_dir=None, group=False):
    """dB RMSE against the float truth, and PSNR/SSIM on jet RGB for every mode."""
    tot = {"psnr_rgb": 0.0, "ssim_rgb": 0.0, "rmse_db": 0.0, "mae_db": 0.0,
           "rmse_db_in_range": 0.0, "n": 0}
    have_float = "float" in data
    if model.head is not None:
        model.res_stats = [0.0, 0.0, 0]
    writer = RenderWriter() if save_dir is not None else None
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
    for i, img in render_views(model, data, idx, span, group):
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
        if writer is not None:
            writer.png(os.path.join(save_dir, data["names"][i] + ".png"),
                       (pred_rgb.permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8))
            writer.npy(os.path.join(save_dir, data["names"][i] + ".npy"),
                       (pred_norm * span + vmin).cpu().numpy().astype(np.float32))
    if writer is not None:
        writer.close()
    n = max(tot.pop("n"), 1)
    out = {k: v / n for k, v in tot.items()}
    return out | head_stats(model, span)


def head_stats(model, span):
    """The shading head's share of the output: RMS of its residual (in dB of channel 0) and its share of the
    output's per-view variance. Read after an evaluation pass; empty without a head."""
    if model.head is None or not model.res_stats or not model.res_stats[2]:
        return {}
    r2, var, n = model.res_stats
    model.res_stats = None
    return {"head_residual_rms_db": math.sqrt(r2 / n) * span, "head_residual_var_share": r2 / max(var, 1e-12)}


@torch.no_grad()
def evaluate_multi(model, data, idx, ch_ranges, channel_names, save_dir=None, mask_channel=0, group=False):
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
    if model.head is not None:
        model.res_stats = [0.0, 0.0, 0]
    writer = RenderWriter() if save_dir is not None else None
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
    for i, img in render_views(model, data, idx, span0, group):
        img = img.clamp(0, 1)
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
        if writer is not None:
            writer.npy(os.path.join(save_dir, data["names"][i] + ".npy"), pred_f.cpu().numpy().astype(np.float32))
    if writer is not None:
        writer.close()
    n = max(tot.pop("n"), 1)
    return {k: v / n for k, v in tot.items()} | head_stats(model, span0)


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
    # speed: the first two change only the work, not the numbers; the third changes the optimisation
    ap.add_argument("--sh-backend", choices=["torch", "gsplat"], default="torch",
                    help="non-rgb modes, frozen geometry: evaluate the SH colour in gsplat's CUDA kernel "
                         "(the same values; a third of the time with backward)")
    ap.add_argument("--eval-group", action="store_true",
                    help="render the faces of one receiver position together at evaluation (one colour "
                         "evaluation per position; identical images)")
    ap.add_argument("--faces-per-step", type=int, default=1,
                    help="each step renders this many faces of ONE receiver position (they share the colour "
                         "evaluation) and takes one Adam step on their mean loss; 1 = one random view per step")
    ap.add_argument("--lr-scale", type=float, default=1.0,
                    help="multiply every learning rate (e.g. sqrt(faces-per-step) for the batched step)")
    ap.add_argument("--protocol", default=None,
                    help="a protocol directory (rrf_gsplat/protocol_v1): train on its training set and evaluate on "
                         "--eval-set, instead of the dataset's own train_index / test_index")
    ap.add_argument("--eval-set", default="val",
                    help="with --protocol: val (default), val_random, val_segment; the sealed test sets (test, "
                         "test_interp, test_region) only together with --final-test")
    ap.add_argument("--final-test", action="store_true",
                    help="allow a sealed test set (stage 4 only; every use is logged in <protocol>/test_access.log)")
    ap.add_argument("--test-source", default=None,
                    help="take the held-out views (poses, targets, render size) from this dataset instead: the same "
                         "positions and image names, e.g. rendered at another resolution or ray budget. The split and "
                         "the value ranges still come from --source.")
    # DLSS-style deferred shading (rounds 39+): every piece is a switch, for the ablation
    ap.add_argument("--head", choices=["none", "cnn"], default="none",
                    help="cnn: a residual CNN after the rasteriser, shared by every view (neural_shading.py)")
    ap.add_argument("--head-guides", default="", help="comma list: geo (depth, normal, ray, receiver), phys (transmitter terms)")
    ap.add_argument("--head-latent", type=int, default=0, help="learned per-Gaussian feature channels for the head")
    ap.add_argument("--head-strip", action="store_true", help="the head sees a position's four faces as one ring")
    ap.add_argument("--head-max-db", type=float, default=6.0, help="bound of the head's residual, dB")
    ap.add_argument("--head-bound-units", action="store_true",
                    help="the head's output layer in units of the bound (m tanh(o), not m tanh(o/m)): the fix for the "
                         "round-41 heads that saturated at the bound and stopped learning")
    ap.add_argument("--head-warmup", type=int, default=0,
                    help="steps with the head switched off (the Gaussians alone) before it starts: the untrained field "
                         "renders above most of the truth, and a head that is on from step 1 takes that global offset "
                         "at its bound, where tanh has no gradient left (round 43)")
    ap.add_argument("--head-width", type=int, default=32)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--peak-loss", type=float, default=0.0,
                    help="extra L1 weight on the pixels within 10 dB of each view's true maximum (value / power channel)")
    # 3DGS-LM (Hoellein et al., ICCV 2025): Levenberg-Marquardt on the SH coefficients after N Adam steps (lm_optim.py);
    # the defaults are 3DGS-LM's, the subset size is its 25 images (6 positions x 4 faces)
    ap.add_argument("--lm-after", type=int, default=0, help="switch from Adam to LM after this many steps (0 = off)")
    ap.add_argument("--lm-iters", type=int, default=5)
    ap.add_argument("--lm-subset-positions", type=int, default=6)
    ap.add_argument("--lm-subsets", type=int, default=4)
    ap.add_argument("--lm-pcg-iters", type=int, default=8)
    ap.add_argument("--lm-pcg-rtol", type=float, default=5e-2)
    ap.add_argument("--lm-radius", type=float, default=1e-3)
    ap.add_argument("--lm-radius-min", type=float, default=1e-4)
    ap.add_argument("--lm-radius-max", type=float, default=1e-2)
    ap.add_argument("--lm-linesearch-frac", type=float, default=0.3)
    ap.add_argument("--cudnn-tf32", choices=["on", "off"], default="off",
                    help="off (the default since round 45): cuDNN convolutions in full fp32. PyTorch's default (on) runs "
                         "the SSIM's convolutions in TF32 on this GPU, and SSIM's variances (E[x^2] - E[x]^2) then lose so "
                         "much that a quarter of our pixels come out above 1 (up to 1.48) and the loss's SSIM term and its "
                         "gradient are wrong in smooth regions; the reported SSIM metric too (found 2026-09-24). Every run "
                         "before round 45 used on. Off costs about 4 %% of the training time and slows the head's CNN.")
    ap.add_argument("--lobes", type=int, default=0,
                    help="P2 (docs/tech_paths.md): K spherical-Gaussian lobes per Gaussian on top of the SH colour")
    ap.add_argument("--lobe-kappa", type=float, default=20.0, help="initial lobe sharpness (width ~ 1/sqrt(kappa) rad)")
    ap.add_argument("--lobe-axis-lr", type=float, default=1e-3)
    ap.add_argument("--lobe-kappa-lr", type=float, default=1e-2)
    ap.add_argument("--pcolor", type=int, default=0,
                    help="P7 (docs/tech_paths.md): per-Gaussian latent of this width + a shared MLP of (latent, direction, "
                         "receiver position) added to the SH colour; 0 = off")
    ap.add_argument("--pcolor-hidden", type=int, default=32)
    ap.add_argument("--pcolor-freqs", type=int, default=4, help="Fourier frequencies of the receiver position")
    ap.add_argument("--pcolor-lr", type=float, default=1e-3, help="the MLP's learning rate (the latent: --feature-lr)")
    ap.add_argument("--emitters", default=None,
                    help="M3's placement oracle: an npz of points (sionna_port/path_emitters.py) that become extra "
                         "isotropic Gaussians rendered in their own pass and added in linear power (--mode power)")
    ap.add_argument("--emitter-scale", type=float, default=0.025, help="the emitters' isotropic scale (m)")
    ap.add_argument("--emitter-opacity", type=float, default=0.05,
                    help="the emitters' FIXED opacity (above gsplat's 1/255 cut-off, small enough to add almost linearly)")
    ap.add_argument("--emitter-lr-mult", type=float, default=10.0,
                    help="the emitters' SH learning rate over --feature-lr: their value is a sigmoid of the SH and starts "
                         "near the floor, so at the visual Gaussians' rate it could not get bright within 3000 steps")
    ap.add_argument("--em-pcolor", type=int, default=0,
                    help="with --emitters: P7 for the emitters -- a per-emitter latent of this width and a shared MLP of "
                         "(latent, direction, receiver position) added to each emitter's logit; 0 = off")
    ap.add_argument("--em-pcolor-hidden", type=int, default=32)
    ap.add_argument("--train-names-file", default=None,
                    help="train on exactly these views (one name per line; each must be in the training set)")
    ap.add_argument("--bwd-no-geom", choices=["auto", "on", "off"], default="auto",
                    help="gsplat_win's GSPLAT_BWD_NO_GEOM: the rasteriser's backward skips the conic / 2D-mean gradients, "
                         "which frozen geometry discards anyway (same colour / opacity gradients; -6.5 %% training time in "
                         "round 45). auto = on whenever the geometry is frozen; the stock gsplat build ignores it")
    cfg = ap.parse_args()
    if cfg.lm_after and cfg.cudnn_tf32 == "on":
        cfg.cudnn_tf32 = "off"
        print("--lm-after: cuDNN TF32 off (the LM residuals need an exact SSIM)")
    torch.backends.cudnn.allow_tf32 = cfg.cudnn_tf32 == "on"
    if cfg.lm_after:
        if cfg.lobes or cfg.pcolor:
            raise SystemExit("--lm-after assumes the render is linear in the SH coefficients alone; not with --lobes / --pcolor")
        if cfg.mode != "db" or cfg.train_geometry or cfg.head != "none" or cfg.densify != "none" or cfg.faces_per_step < 2:
            raise SystemExit("--lm-after: db mode, frozen geometry, no head, no densification, --faces-per-step >= 2 "
                             "(the render must be linear in the SH coefficients; LM works on receiver positions)")
    if cfg.densify == "mcmc":
        cfg.train_geometry = True
    if cfg.bwd_no_geom == "on" and cfg.train_geometry:
        raise SystemExit("--bwd-no-geom on drops the geometry gradients; it needs frozen geometry")
    if cfg.bwd_no_geom == "on" or (cfg.bwd_no_geom == "auto" and not cfg.train_geometry):
        os.environ["GSPLAT_BWD_NO_GEOM"] = "1"          # read by the extension at every backward call
    elif "GSPLAT_BWD_NO_GEOM" in os.environ:
        if cfg.train_geometry:
            raise SystemExit("GSPLAT_BWD_NO_GEOM is set in the environment but the geometry trains")
        del os.environ["GSPLAT_BWD_NO_GEOM"]
    if cfg.head != "none":
        if cfg.mode == "rgb":
            raise SystemExit("--head works on the value modes (db, power, multi), not on jet RGB")
        if cfg.densify != "none":
            raise SystemExit("--head is not wired into a densification strategy")
        if cfg.head_strip and not cfg.eval_group:
            cfg.eval_group = True
            print("--head-strip: evaluation groups each position's faces (--eval-group), or the ring could not form")
    if cfg.faces_per_step > 1 and cfg.densify != "none":
        raise SystemExit("--faces-per-step > 1 is not wired into the densification strategy's per-view statistics")
    if cfg.em_pcolor and not cfg.emitters:
        raise SystemExit("--em-pcolor needs --emitters")
    if cfg.emitters and (cfg.mode != "power" or cfg.head != "none" or cfg.delay_depth or cfg.densify != "none"
                         or cfg.lm_after or cfg.train_geometry):
        raise SystemExit("--emitters: --mode power, frozen geometry, no head, no delay channel, no densification, no LM")

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
    if cfg.protocol:
        # protocol_v1 and later: train on the protocol's training set, evaluate on --eval-set (the validation set
        # unless --final-test), never on the dataset's own train / test index
        import protocol as PR
        PR.check_dataset(cfg.protocol, cfg.source)
        train_names = PR.train_names(cfg.protocol)
        test_names = PR.eval_names(cfg.protocol, cfg.eval_set, allow_test=cfg.final_test)
        print(f"protocol {os.path.basename(os.path.abspath(cfg.protocol))}: {len(train_names)} training views, "
              f"evaluating on {cfg.eval_set} ({len(test_names)} views)")
    else:
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
    if cfg.train_names_file:
        chosen = [l.strip() for l in open(cfg.train_names_file) if l.strip()]
        outside = [n for n in chosen if n not in set(train_names)]
        if outside:
            raise SystemExit(f"--train-names-file: {len(outside)} views are not in the training set (e.g. {outside[:3]})")
        train_names = chosen
    t0 = time.time()
    train = load_views(cfg.source, train_names, views, device, want_float=True)
    if cfg.test_source:
        test = load_views(cfg.test_source, test_names, read_colmap_text(os.path.join(cfg.test_source, "sparse", "0")),
                          device, want_float=True)
        print(f"held-out views from {cfg.test_source}: {test['width']}x{test['height']} (training at {train['width']}x{train['height']})")
    else:
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
    model.sh_backend = cfg.sh_backend
    if cfg.head != "none":
        # the bound is in dB of the value (db / power) or of the power channel (multi)
        span_head = span if cfg.mode != "multi" else float(ch_ranges[cfg.mask_channel, 1] - ch_ranges[cfg.mask_channel, 0])
        head = attach_head(model, cfg.head, cfg.head_guides, cfg.head_latent, cfg.head_strip, cfg.head_max_db,
                           cfg.head_width, span_head, meta.get("tx_loc"), device, cfg.head_bound_units)
        print(f"shading head: {sum(p.numel() for p in head.parameters()):,} parameters, guides "
              f"[{cfg.head_guides or 'none'}], latent {cfg.head_latent}, strip {cfg.head_strip}, bound {cfg.head_max_db} dB"
              f"{' (output in units of the bound)' if cfg.head_bound_units else ''}"
              f"{f', off for the first {cfg.head_warmup} steps' if cfg.head_warmup else ''}")
    if cfg.init_from:
        model.load_state(torch.load(cfg.init_from, map_location=device))
        if cfg.init_geometry_only:
            model.params["sh0"].data.zero_(); model.params["shN"].data.zero_()
            print(f"geometry (and opacity) from {cfg.init_from}, colours from zero")
        else:
            print(f"warm start from {cfg.init_from}")
    if cfg.lobes:
        rx = torch.linalg.inv(train["viewmats"])[:, :3, 3]
        model.add_lobes(cfg.lobes, cfg.lobe_kappa, rx)
        if cfg.init_from:
            model.load_state(torch.load(cfg.init_from, map_location=device))       # its lobes, if it has any
        print(f"lobes: {cfg.lobes} per Gaussian, kappa {cfg.lobe_kappa:g} (width ~{math.degrees(cfg.lobe_kappa ** -0.5):.0f} deg), "
              f"weights from zero")
    if cfg.pcolor:
        rx = torch.linalg.inv(train["viewmats"])[:, :3, 3]
        model.add_pcolor(cfg.pcolor, cfg.pcolor_hidden, cfg.pcolor_freqs, rx)
        if cfg.init_from:
            model.load_state(torch.load(cfg.init_from, map_location=device))
        print(f"position-conditioned colour: latent {cfg.pcolor}, MLP hidden {cfg.pcolor_hidden}, "
              f"{cfg.pcolor_freqs} position frequencies ({sum(p.numel() for p in model.pcolor_mlp.parameters()):,} MLP parameters), "
              f"output from zero")
    if cfg.emitters:
        em = np.load(cfg.emitters)
        model.add_emitters(em["means"], cfg.emitter_scale, cfg.emitter_opacity)
        if cfg.init_from:
            model.load_state(torch.load(cfg.init_from, map_location=device))
        if cfg.em_pcolor:
            model.add_em_pcolor(cfg.em_pcolor, cfg.em_pcolor_hidden, cfg.pcolor_freqs, torch.linalg.inv(train["viewmats"])[:, :3, 3])
            if cfg.init_from:
                model.load_state(torch.load(cfg.init_from, map_location=device))
            print(f"emitter position-conditioned colour: latent {cfg.em_pcolor}, MLP hidden {cfg.em_pcolor_hidden}, "
                  f"output x{model.EM_PC_GAIN:g} logits, from zero")
        print(f"emitters (placement oracle): {len(em['means']):,} from {cfg.emitters}, scale {cfg.emitter_scale} m, "
              f"opacity {cfg.emitter_opacity} (fixed), value from near the floor, SH lr x{cfg.emitter_lr_mult:g}; "
              f"own pass, added in linear power")
    print(f"{model.n_gaussians:,} Gaussians from visual iteration {model.visual_iteration}; "
          f"mode {cfg.mode}, {channels} channel(s), SH degree {cfg.sh_degree}; "
          f"geometry {'trained' if cfg.train_geometry else 'frozen'}, densify {cfg.densify}")

    # INRIA's groups: f_dc at feature_lr, f_rest at feature_lr/20, opacity at
    # opacity_lr; geometry (when trained) at INRIA's scaling/rotation lrs and
    # the means at ten times INRIA's final position lr. One Adam per tensor,
    # which is what gsplat's strategies expect.
    lrs = {"sh0": cfg.feature_lr, "shN": cfg.feature_lr / 20.0, "opacities": cfg.opacity_lr,
           "means": 1.6e-5 * model.spatial_lr_scale, "scales": 5e-3, "quats": 1e-3, "latent": cfg.feature_lr,
           "lobe_w": cfg.feature_lr, "lobe_axis": cfg.lobe_axis_lr, "lobe_logk": cfg.lobe_kappa_lr, "pc_latent": cfg.feature_lr,
           "em_sh0": cfg.feature_lr * cfg.emitter_lr_mult, "em_shN": cfg.feature_lr / 20.0 * cfg.emitter_lr_mult,
           "em_pc_latent": cfg.feature_lr}
    lrs = {k: v * cfg.lr_scale for k, v in lrs.items()}
    optimizers = {name: torch.optim.Adam([p], lr=lrs[name], eps=1e-15)
                  for name, p in model.params.items() if p.requires_grad}
    if model.head is not None:
        optimizers["head"] = torch.optim.Adam(model.head.parameters(), lr=cfg.head_lr)
    if getattr(model, "pcolor_mlp", None) is not None:
        optimizers["pcolor"] = torch.optim.Adam(model.pcolor_mlp.parameters(), lr=cfg.pcolor_lr * cfg.lr_scale)
    if getattr(model, "em_pcolor_mlp", None) is not None:
        optimizers["em_pcolor"] = torch.optim.Adam(model.em_pcolor_mlp.parameters(), lr=cfg.pcolor_lr * cfg.lr_scale)
    n_params = sum(p.numel() for p in model.params.values() if p.requires_grad)
    print(f"optimised per-Gaussian parameters: {n_params:,} (Adam state {2 * 4 * n_params / 2**20:.0f} MiB)")
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

    peak_band = None
    if cfg.peak_loss > 0:
        # 10 dB in the normalised units of the value channel (db / power) or of the power channel (multi)
        peak_band = 10.0 / (span if cfg.mode != "multi" else float(ch_ranges[cfg.mask_channel, 1] - ch_ranges[cfg.mask_channel, 0]))

    def view_loss(img, gt):
        """The loss of one rendered view against its target (+ the peak-aware term when asked for)."""
        if cfg.mode == "multi":
            loss = multi_loss(img, gt)
        else:
            loss = (1 - cfg.lambda_dssim) * l1_loss(img, gt) + cfg.lambda_dssim * (1 - ssim(img[None], gt[None]))
        if peak_band is not None:
            c = cfg.mask_channel if cfg.mode == "multi" else 0
            top = gt[c] >= gt[c].max() - peak_band
            loss = loss + cfg.peak_loss * ((img[c] - gt[c]).abs() * top).sum() / top.sum().clamp_min(1)
        return loss

    n_train = len(train_names)
    groups = None
    if cfg.faces_per_step > 1:
        # receiver positions = runs of consecutive training views whose cameras share a centre (images are
        # position-major and the split holds out whole positions, so a position's faces are adjacent)
        c = torch.linalg.inv(train["viewmats"])[:, :3, 3].cpu()
        groups, cur = [], [0]
        for i in range(1, n_train):
            if float((c[i] - c[cur[0]]).norm()) < 1e-4:
                cur.append(i)
            else:
                groups.append(cur); cur = [i]
        groups.append(cur)
        sizes = np.bincount([len(g) for g in groups])
        print(f"faces per step {cfg.faces_per_step}: {len(groups)} receiver positions, "
              f"faces per position {{{', '.join(f'{s}: {int(n)}' for s, n in enumerate(sizes) if n)}}}")
    strip_check = None
    if model.head is not None and cfg.head_strip:
        # Which way round do the faces join? Measured on the targets rather than assumed: for each candidate
        # order, the mean jump from the right edge column of one face to the left edge column of the next
        # (cyclically), over whole training positions spread along the list.
        if groups is None:
            raise SystemExit("--head-strip needs --faces-per-step 4, so every step holds a whole ring")
        full = [g for g in groups if len(g) == 4]
        pick = [full[k] for k in np.linspace(0, len(full) - 1, min(16, len(full))).round().astype(int)]
        def seam(order):
            e = []
            for g in pick:
                f = [target(i)[0] for i in g]
                e += [float((f[order[k]][:, -1] - f[order[(k + 1) % 4]][:, 0]).abs().mean()) for k in range(4)]
            return float(np.mean(e))
        cands = {"dataset order": [0, 1, 2, 3], "reversed": [3, 2, 1, 0]}
        errs = {k: seam(o) for k, o in cands.items()}
        best = min(errs, key=errs.get)
        model.head_cfg["order"] = cands[best]
        interior = float(np.mean([float((target(g[0])[0][:, 1:] - target(g[0])[0][:, :-1]).abs().mean()) for g in pick]))
        strip_check = {"seam_error": errs, "chosen": best, "adjacent_column_step_inside_a_face": interior}
        print(f"ring order: seam jump {errs} (normalised), chosen {best}; a column step inside a face is {interior:.4f}")
    # the running evaluation's subset: whole held-out positions (every face) spread over the list. A stride over
    # views (range(0, n, n // 64), until round 46) samples only some orientations: the views come in blocks of four
    # faces, and a stride of 10 took faces 0 and 2 only. The final evaluation always covers every held-out view.
    eval_idx = spread_positions(test["viewmats"], cfg.eval_subset)
    history = []
    running_eval_seconds = 0.0
    torch.cuda.synchronize(); t_train = time.time()
    rng = np.random.default_rng(cfg.seed)
    for it in range(1, cfg.iterations + 1):
        if cfg.lm_after and it > cfg.lm_after:
            break                                    # the rest of the budget goes to LM (below)
        model.head_on = it > cfg.head_warmup
        if groups is None:
            i = int(rng.integers(n_train))
            img = model.render(train["viewmats"][i], train["Ks"][i], train["width"], train["height"], span)
            loss = view_loss(img, target(i))
        else:
            # one receiver position per step: its faces share the colour evaluation, and one Adam
            # step takes their mean loss
            g = groups[int(rng.integers(len(groups)))]
            if len(g) > cfg.faces_per_step:
                g = sorted(rng.choice(g, cfg.faces_per_step, replace=False).tolist())
            imgs = model.render_batch(train["viewmats"][g], train["Ks"][g], train["width"], train["height"], span)
            loss = sum(view_loss(imgs[k], target(i)) for k, i in enumerate(g)) / len(g)
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
            torch.cuda.synchronize(); t_ev = time.time()
            m = (evaluate_multi(model, test, eval_idx, ch_ranges, channel_names, mask_channel=cfg.mask_channel,
                                group=cfg.eval_group)
                 if cfg.mode == "multi"
                 else evaluate(model, test, eval_idx, span, vmin, group=cfg.eval_group))
            torch.cuda.synchronize(); running_eval_seconds += time.time() - t_ev
            m.update(iteration=it, seconds=time.time() - t_train, loss=float(loss.detach()),
                     gaussians=model.n_gaussians, views_seen=it * cfg.faces_per_step,
                     seconds_excl_eval=time.time() - t_train - running_eval_seconds)
            history.append(m)
            print(f"  it {it:6d}  {m['seconds']:6.0f} s  loss {m['loss']:.4f}  "
                  f"PSNR(jet) {m['psnr_rgb']:5.2f}  SSIM {m['ssim_rgb']:.3f}  "
                  f"RMSE {m['rmse_db']:5.2f} dB  (subset {len(eval_idx)}"
                  + (f", {m['gaussians']:,} Gaussians" if strategy is not None else "") + ")")
    lm_history = None
    if cfg.lm_after:
        from lm_optim import LM
        lm = LM(model, groups,
                lambda g: (train["viewmats"][g], train["Ks"][g], train["width"], train["height"]),
                lambda g: torch.stack([target(i) for i in g]),
                span, cfg.lambda_dssim, sh_basis,
                {"subset_positions": cfg.lm_subset_positions, "subsets": cfg.lm_subsets, "pcg_iters": cfg.lm_pcg_iters,
                 "pcg_rtol": cfg.lm_pcg_rtol, "radius": cfg.lm_radius, "radius_min": cfg.lm_radius_min,
                 "radius_max": cfg.lm_radius_max, "linesearch_frac": cfg.lm_linesearch_frac, "min_diag": 1.0,
                 "max_diag": 1e6, "max_step": 10.0, "gamma0": 1.0, "gamma_alpha": 0.7, "min_rel_decrease": 1e-5})
        for k in range(1, cfg.lm_iters + 1):
            lm.step(k)
            torch.cuda.synchronize(); t_ev = time.time()
            m = evaluate(model, test, eval_idx, span, vmin, group=cfg.eval_group)
            torch.cuda.synchronize(); running_eval_seconds += time.time() - t_ev
            m.update(iteration=cfg.lm_after, lm_iteration=k, seconds=time.time() - t_train, gaussians=model.n_gaussians,
                     seconds_excl_eval=time.time() - t_train - running_eval_seconds)
            history.append(m)
            print(f"  LM {k:3d}  {m['seconds']:6.0f} s  PSNR(jet) {m['psnr_rgb']:5.2f}  RMSE {m['rmse_db']:5.2f} dB "
                  f"(subset {len(eval_idx)})")
        lm_history = lm.history
        del lm
        torch.cuda.empty_cache()
    torch.cuda.synchronize(); train_seconds = time.time() - t_train
    # On Windows the driver's sysmem fallback lets allocations past the card's memory page to host RAM instead of
    # failing, and everything then runs 10-20x slower (round 46): a run whose peak comes near the card's size has
    # no valid timing
    peak_reserved = torch.cuda.max_memory_reserved() / 2 ** 20
    card_mib = torch.cuda.get_device_properties(0).total_memory / 2 ** 20
    if peak_reserved > 0.9 * card_mib:
        print(f"WARNING: peak reserved CUDA memory {peak_reserved:.0f} MiB of {card_mib:.0f}: the timing may include "
              f"host-memory paging (sysmem fallback)")
    model.head_on = True

    os.makedirs(cfg.out, exist_ok=True)
    all_idx = list(range(len(test_names)))
    # One pass scores every held-out view and, when all of them are to be written, writes them too; a
    # partial --save-renders still gets its own second pass. (Earlier versions always rendered twice.)
    save_dir = os.path.join(cfg.out, "renders") if cfg.save_renders else None
    save_all = cfg.save_renders < 0 or cfg.save_renders >= len(all_idx)
    t_eval = time.time()
    if cfg.no_eval:
        # an adaptation run whose only product is its geometry: skip the test pass
        final = None
    elif cfg.mode == "multi":
        final = evaluate_multi(model, test, all_idx, ch_ranges, channel_names, mask_channel=cfg.mask_channel,
                               save_dir=save_dir if save_all else None, group=cfg.eval_group)
        if save_dir and not save_all:
            evaluate_multi(model, test, all_idx[:cfg.save_renders], ch_ranges, channel_names,
                           save_dir=save_dir, mask_channel=cfg.mask_channel, group=cfg.eval_group)
    else:
        final = evaluate(model, test, all_idx, span, vmin, save_dir=save_dir if save_all else None,
                         group=cfg.eval_group)
        if save_dir and not save_all:
            evaluate(model, test, all_idx[:cfg.save_renders], span, vmin, save_dir=save_dir, group=cfg.eval_group)
    torch.cuda.synchronize(); eval_seconds = time.time() - t_eval
    torch.save(model.state(), os.path.join(cfg.out, "rrf_state.pt"))
    result = {"config": vars(cfg), "db_range": [vmin, vmax], "n_train": n_train,
              "channels": channel_names, "channel_ranges": meta.get("channel_ranges"),
              "n_test": len(test_names), "gaussians": model.n_gaussians,
              "train_seconds": train_seconds, "iters_per_second": cfg.iterations / train_seconds,
              "cuda_peak_reserved_mib": peak_reserved, "cuda_card_mib": card_mib,
              "running_eval_seconds": running_eval_seconds,
              "train_seconds_excl_running_eval": train_seconds - running_eval_seconds,
              "views_seen": cfg.iterations * cfg.faces_per_step,
              "optimised_gaussian_params": n_params, "adam_state_mib": 2 * 4 * n_params / 2**20,
              "head_params": sum(p.numel() for p in model.head.parameters()) if model.head is not None else 0,
              "strip_check": strip_check,
              "load_seconds": t_train - t_start, "eval_seconds": eval_seconds,
              "gpu": torch.cuda.get_device_name(0),
              "total_seconds": time.time() - t_start, "history": history, "final": final,
              "adam_steps": cfg.lm_after or cfg.iterations, "lm_history": lm_history,
              "gsplat_bwd_switches": {k: os.environ.get(k) for k in ("GSPLAT_BWD_NO_GEOM", "GSPLAT_BWD_PERGAUSS")}}
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
