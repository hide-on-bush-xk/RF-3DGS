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
from utils.sh_utils import eval_sh                  # noqa: E402

from jet import jet_rgb, jet_inverse               # noqa: E402


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------
def qvec2rotmat(q):
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


def load_views(source, names, views, device, want_float):
    """PNG targets as uint8 [n,3,H,W]; float spectra as float16 [n,H,W] if present."""
    from PIL import Image
    rgb, flt, viewmats, Ks = [], [], [], []
    have_float = want_float and os.path.isdir(os.path.join(source, "spectra_float"))
    for n in names:
        img = np.array(Image.open(os.path.join(source, "images", n + ".png")).convert("RGB"))
        rgb.append(torch.from_numpy(img).permute(2, 0, 1))
        if have_float:
            flt.append(torch.from_numpy(np.load(os.path.join(source, "spectra_float", n + ".npy"))
                                        .astype(np.float16)))
        view, K, w, h = views[n + ".png"]
        viewmats.append(torch.from_numpy(view)); Ks.append(torch.from_numpy(K))
    out = {"rgb": torch.stack(rgb).to(device), "viewmats": torch.stack(viewmats).to(device),
           "Ks": torch.stack(Ks).to(device), "names": names,
           "width": views[names[0] + ".png"][2], "height": views[names[0] + ".png"][3]}
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
    """Frozen geometry from the visual checkpoint; colours and opacity learn."""

    def __init__(self, ckpt_path, mode, channels, sh_degree, device, train_opacity=True):
        super().__init__()
        (m, it) = torch.load(ckpt_path, weights_only=False, map_location="cpu")
        (_, xyz, f_dc, f_rest, scaling, rotation, opacity, *_) = m
        # The checkpoint holds nn.Parameters of the visual training, so every
        # tensor still requires grad; detach, or exp()/normalize() become graph
        # nodes that are freed after the first backward.
        xyz, scaling, rotation, opacity = (t.detach() for t in (xyz, scaling, rotation, opacity))
        self.register_buffer("means", xyz.to(device))
        self.register_buffer("scales", torch.exp(scaling).to(device))
        self.register_buffer("quats", F.normalize(rotation, dim=-1).to(device))
        self.mode, self.channels, self.sh_degree = mode, channels, sh_degree
        n, k = xyz.shape[0], (sh_degree + 1) ** 2
        # RF-3DGS zeroes every SH coefficient before RF training; the same here,
        # for any channel count. DC and the rest are separate parameters so
        # they can take INRIA's separate learning rates (Adam ignores gradient
        # scaling, so a single tensor could not emulate that).
        # [N, K, C] (coefficient-major), the layout gsplat's SH kernel takes
        self.sh_dc = torch.nn.Parameter(torch.zeros(n, 1, channels, device=device))
        self.sh_rest = torch.nn.Parameter(torch.zeros(n, k - 1, channels, device=device))
        self.opacity_logit = torch.nn.Parameter(opacity.to(device).clone(),
                                                requires_grad=train_opacity)
        self.visual_iteration = it

    @property
    def sh(self):
        return torch.cat([self.sh_dc, self.sh_rest], dim=1)               # [N, K, C]

    def colours(self, cam_center):
        """View-dependent per-Gaussian value(s), [N, C]; the +0.5 is INRIA's.

        The SH basis depends only on the (frozen) means and the camera, so it
        is a constant per view and the colour is one linear map of the
        coefficients. Evaluating it that way costs one multiply-add in the
        backward pass instead of autograd through INRIA's eval_sh expression,
        which was two thirds of a training step for 1M Gaussians.
        """
        if self.sh_degree == 0:
            return self.sh_dc[:, 0, :] * 0.28209479177387814 + 0.5
        with torch.no_grad():
            dirs = F.normalize(self.means - cam_center[None], dim=-1)
            basis = sh_basis(self.sh_degree, dirs)                  # [N, K]
        return (self.sh * basis[:, :, None]).sum(dim=1) + 0.5

    def render(self, viewmat, K, width, height, span_db):
        from gsplat import rasterization
        if self.mode == "rgb" and self.sh_degree > 0:
            # gsplat evaluates 3-channel SH in CUDA (and adds INRIA's 0.5 and
            # clamps at 0 itself); its backward is a third of the torch path's
            img, alpha, _ = rasterization(
                self.means, self.quats, self.scales, torch.sigmoid(self.opacity_logit[:, 0]),
                self.sh, viewmat[None], K[None], width, height,
                sh_degree=self.sh_degree, backgrounds=torch.zeros(1, 3, device=self.means.device))
            return img[0].permute(2, 0, 1)
        cam_center = torch.linalg.inv(viewmat)[:3, 3]
        col = self.colours(cam_center)
        if self.mode == "rgb":
            col = col.clamp_min(0.0)
        elif self.mode == "power":
            # value in [0,1] is dB above the floor as a fraction of the span;
            # composite linear powers, read back in dB
            col = torch.pow(10.0, col.clamp(0.0, 1.0) * span_db / 10.0)
        img, alpha, _ = rasterization(
            self.means, self.quats, self.scales, torch.sigmoid(self.opacity_logit[:, 0]),
            col, viewmat[None], K[None], width, height, sh_degree=None,
            backgrounds=torch.zeros(1, self.channels, device=col.device))
        img = img[0].permute(2, 0, 1)                      # [C,H,W]
        if self.mode == "power":
            img = torch.log10(img + 1e-12) * 10.0 / span_db
        return img


# --------------------------------------------------------------------------
# evaluation
# --------------------------------------------------------------------------
def psnr(a, b):
    mse = ((a - b) ** 2).mean()
    return float(20 * torch.log10(1.0 / torch.sqrt(mse + 1e-12)))


@torch.no_grad()
def evaluate(model, data, idx, span, vmin, save_dir=None, mode_target="rgb"):
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


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, help="dataset dir: images/, sparse/0, [spectra_float/]")
    ap.add_argument("--checkpoint", default=os.path.join(REPO, "RF-3DGS_dataset/blender_visual_trained/chkpnt30000.pth"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["rgb", "db", "power"], default="rgb")
    ap.add_argument("--sh-degree", type=int, default=3)
    ap.add_argument("--iterations", type=int, default=10_000, help="RF-3DGS: 30k -> 40k")
    ap.add_argument("--feature-lr", type=float, default=0.0025)
    ap.add_argument("--opacity-lr", type=float, default=0.05)
    ap.add_argument("--freeze-opacity", action="store_true")
    ap.add_argument("--lambda-dssim", type=float, default=0.2)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--eval-subset", type=int, default=64, help="test views for the running eval")
    ap.add_argument("--max-train-views", type=int, default=None, help="data-efficiency ablation")
    ap.add_argument("--init-from", default=None, help="rrf_state.pt of another run (warm start)")
    ap.add_argument("--save-renders", type=int, default=16, help="test renders to write")
    ap.add_argument("--db-range", type=float, nargs=2, default=None,
                    help="min max dB for the colormap; default from generation_meta.json")
    ap.add_argument("--seed", type=int, default=0)
    cfg = ap.parse_args()

    torch.manual_seed(cfg.seed)
    device = torch.device("cuda")
    t_start = time.time()

    # -- range of the colormap: what a normalised value means in dB ----------
    meta_path = os.path.join(cfg.source, "generation_meta.json")
    if cfg.db_range:
        vmin, vmax = cfg.db_range
    elif os.path.exists(meta_path):
        meta = json.load(open(meta_path))
        vmin, vmax = meta["spec_min_db"], meta["spec_max_db"]
    else:
        vmin, vmax = 0.0, 1.0      # released data: no float truth, dB numbers are relative
        print("no generation_meta.json: dB errors are in normalised units x span=1")
    span = vmax - vmin

    views = read_colmap_text(os.path.join(cfg.source, "sparse", "0"))
    train_names, test_names = ensure_split(cfg.source)
    if cfg.max_train_views:
        # Keep whole positions (all four yaws), not every k-th image: images
        # are position-major, so every 4th image would be the same yaw at
        # every position, and the model would never see the other three.
        per_pos, n_pos = 4, len(train_names) // 4
        keep = max(1, cfg.max_train_views // per_pos)
        pos_idx = np.linspace(0, n_pos - 1, keep).round().astype(int)
        train_names = [train_names[p * per_pos + k] for p in pos_idx for k in range(per_pos)]
    t0 = time.time()
    train = load_views(cfg.source, train_names, views, device, want_float=True)
    test = load_views(cfg.source, test_names, views, device, want_float=True)
    print(f"loaded {len(train_names)} train / {len(test_names)} test views in {time.time()-t0:.0f} s; "
          f"float truth: {'yes' if 'float' in test else 'no'}; range {vmin:.1f}..{vmax:.1f} dB")

    channels = 3 if cfg.mode == "rgb" else 1
    model = RRF(cfg.checkpoint, cfg.mode, channels, cfg.sh_degree, device,
                train_opacity=not cfg.freeze_opacity)
    if cfg.init_from:
        st = torch.load(cfg.init_from, map_location=device)
        for name in ("sh_dc", "sh_rest"):                 # accept the older [N, C, K] layout
            t = st[name]
            if t.shape != getattr(model, name).shape:
                t = t.permute(0, 2, 1).contiguous()
            getattr(model, name).data.copy_(t)
        model.opacity_logit.data.copy_(st["opacity_logit"])
        print(f"warm start from {cfg.init_from}")
    print(f"{model.means.shape[0]:,} Gaussians from visual iteration {model.visual_iteration}; "
          f"mode {cfg.mode}, {channels} channel(s), SH degree {cfg.sh_degree}")

    # INRIA's groups: f_dc at feature_lr, f_rest at feature_lr/20, opacity at opacity_lr.
    groups = [{"params": [model.sh_dc], "lr": cfg.feature_lr},
              {"params": [model.sh_rest], "lr": cfg.feature_lr / 20.0}]
    if not cfg.freeze_opacity:
        groups.append({"params": [model.opacity_logit], "lr": cfg.opacity_lr})
    opt = torch.optim.Adam(groups, lr=0.0, eps=1e-15)

    def target(i):
        if cfg.mode == "rgb":
            return train["rgb"][i].float() / 255.0
        return ((train["float"][i].float() - vmin) / span).clamp(0, 1)[None]

    n_train = len(train_names)
    eval_idx = list(range(0, len(test_names), max(1, len(test_names) // cfg.eval_subset)))[:cfg.eval_subset]
    history = []
    torch.cuda.synchronize(); t_train = time.time()
    rng = np.random.default_rng(cfg.seed)
    for it in range(1, cfg.iterations + 1):
        i = int(rng.integers(n_train))
        img = model.render(train["viewmats"][i], train["Ks"][i], train["width"], train["height"], span)
        gt = target(i)
        loss = (1 - cfg.lambda_dssim) * l1_loss(img, gt) + cfg.lambda_dssim * (1 - ssim(img[None], gt[None]))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if it % cfg.eval_every == 0 or it == cfg.iterations:
            torch.cuda.synchronize()
            m = evaluate(model, test, eval_idx, span, vmin)
            m.update(iteration=it, seconds=time.time() - t_train, loss=float(loss))
            history.append(m)
            print(f"  it {it:6d}  {m['seconds']:6.0f} s  loss {m['loss']:.4f}  "
                  f"PSNR(jet) {m['psnr_rgb']:5.2f}  SSIM {m['ssim_rgb']:.3f}  "
                  f"RMSE {m['rmse_db']:5.2f} dB  (subset {len(eval_idx)})")
    torch.cuda.synchronize(); train_seconds = time.time() - t_train

    os.makedirs(cfg.out, exist_ok=True)
    final = evaluate(model, test, list(range(len(test_names))), span, vmin)
    if cfg.save_renders:
        evaluate(model, test, list(range(min(cfg.save_renders, len(test_names)))), span, vmin,
                 save_dir=os.path.join(cfg.out, "renders"))
    torch.save({"sh_dc": model.sh_dc.data, "sh_rest": model.sh_rest.data,
                "opacity_logit": model.opacity_logit.data,
                "mode": cfg.mode, "sh_degree": cfg.sh_degree},
               os.path.join(cfg.out, "rrf_state.pt"))
    result = {"config": vars(cfg), "db_range": [vmin, vmax], "n_train": n_train,
              "n_test": len(test_names), "gaussians": int(model.means.shape[0]),
              "train_seconds": train_seconds, "iters_per_second": cfg.iterations / train_seconds,
              "total_seconds": time.time() - t_start, "history": history, "final": final}
    with open(os.path.join(cfg.out, "results.json"), "w") as fid:
        json.dump(result, fid, indent=1)
    print(f"\nfinal on {len(test_names)} test views: PSNR(jet) {final['psnr_rgb']:.2f} dB, "
          f"SSIM {final['ssim_rgb']:.3f}, RMSE {final['rmse_db']:.2f} dB, MAE {final['mae_db']:.2f} dB; "
          f"{cfg.iterations} iterations in {train_seconds:.0f} s "
          f"({cfg.iterations/train_seconds:.0f} it/s); wrote {cfg.out}")


if __name__ == "__main__":
    main()
