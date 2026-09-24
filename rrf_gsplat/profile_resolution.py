"""How much of a training step does the render resolution control? (the ceiling for any render-low-res scheme)

DLSS-style super-resolution saves per-pixel work only. Before building an
upsampler, this measures what fraction of one train_rrf.py step is per-pixel
at all: the step is split into the colour evaluation (per Gaussian), the
rasteriser forward + loss + backward (per pixel and per Gaussian), and the
Adam update (per parameter), and timed at several render sizes of the same
view. Everything else in the step is identical to train_rrf.py (same model,
same loss, same optimiser groups).

    /home/ke/miniconda3/envs/rf-gsplat/bin/python rrf_gsplat/profile_resolution.py --mode multi --delay-depth
    /home/ke/miniconda3/envs/rf-gsplat/bin/python rrf_gsplat/profile_resolution.py --mode db --source RF-3DGS_dataset/regenerated/3dgs_MVDR_100_gpct

Writes output/rrf/profile_resolution_<mode>.json. Median of --reps steps after
--warmup excluded, per the reporting contract; the GPU must be exclusive.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_rrf import REPO, RRF, load_views, read_colmap_text, read_index  # noqa: E402
from losses import l1_loss, ssim  # noqa: E402  (INRIA's, copied: losses.py)


def gpu_util():
    """nvidia-smi utilisation, so a run on a busy card is visible in its own record."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10).stdout.strip()
        return out
    except Exception as exc:                      # noqa: BLE001
        return f"unavailable ({exc})"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="RF-3DGS_dataset/regenerated/3dgs_MULTI_24ghz_tut_cs")
    ap.add_argument("--checkpoint", default=os.path.join(REPO, "RF-3DGS_dataset/blender_visual_trained/chkpnt30000.pth"))
    ap.add_argument("--mode", choices=["db", "multi"], default="multi")
    ap.add_argument("--delay-depth", action="store_true")
    ap.add_argument("--scales", type=float, nargs="+", default=[0.25, 0.5, 1.0, 2.0],
                    help="render size as a fraction of the dataset's 300x200")
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--faces", type=int, default=1, choices=[1, 4],
                    help="4: the clean step renders one position's four faces with one colour evaluation and one Adam step "
                         "(train_rrf.py --faces-per-step 4); the per-part split is only timed for 1")
    ap.add_argument("--sh-backend", choices=["torch", "gsplat"], default="torch")
    ap.add_argument("--sh-degree", type=int, default=3)
    ap.add_argument("--head", choices=["none", "cnn"], default="none", help="train_rrf.py's shading head")
    ap.add_argument("--head-guides", default="")
    ap.add_argument("--head-latent", type=int, default=0)
    ap.add_argument("--head-strip", action="store_true")
    ap.add_argument("--head-width", type=int, default=32)
    cfg = ap.parse_args()
    device = torch.device("cuda")
    src = os.path.join(REPO, cfg.source)
    meta = json.load(open(os.path.join(src, "generation_meta.json")))
    views = read_colmap_text(os.path.join(src, "sparse", "0"))
    names = read_index(os.path.join(src, "train_index.txt"))
    # stratified by yaw: images are position-major with 4 yaws each, so take one
    # position every so often and all four of its faces
    pick = [names[p * 4 + k] for p in np.linspace(0, len(names) // 4 - 1, 4).round().astype(int) for k in range(4)]
    data = load_views(src, pick, views, device, want_float=True)
    W0, H0 = data["width"], data["height"]
    if cfg.mode == "multi":
        ch_ranges = torch.tensor(meta["channel_ranges"], device=device, dtype=torch.float32)
        channels = ch_ranges.shape[0]
    else:
        channels = 1
    vmin, vmax = meta["spec_min_db"], meta["spec_max_db"]; span = vmax - vmin
    model = RRF(cfg.checkpoint, cfg.mode, channels, cfg.sh_degree, device)
    model.sh_backend = cfg.sh_backend
    if cfg.head != "none":
        from train_rrf import attach_head
        span_h = (meta["spec_max_db"] - meta["spec_min_db"]) if cfg.mode != "multi" else float(meta["channel_ranges"][0][1] - meta["channel_ranges"][0][0])
        attach_head(model, cfg.head, cfg.head_guides, cfg.head_latent, cfg.head_strip, 6.0, cfg.head_width, span_h, meta.get("tx_loc"), device)
        model.head_cfg["order"] = [3, 2, 1, 0]                     # the ring's order does not change the work
    if cfg.delay_depth:
        names_ch = meta["channels"]
        model.delay_channel = names_ch.index("delay_ns")
        model.delay_span_ns = float(ch_ranges[model.delay_channel, 1] - ch_ranges[model.delay_channel, 0])
        model.delay_depth_mode, model.delay_range = "ED", "euclid"
    lrs = {"sh0": 0.0025, "shN": 0.0025 / 20, "opacities": 0.05, "latent": 0.0025}
    opts = {k: torch.optim.Adam([p], lr=lrs[k], eps=1e-15) for k, p in model.params.items() if p.requires_grad}
    if model.head is not None:
        opts["head"] = torch.optim.Adam(model.head.parameters(), lr=1e-3)

    def target(i, w, h):
        f = data["float"][i].float()
        if cfg.mode == "multi":
            lo, hi = ch_ranges[:, 0, None, None], ch_ranges[:, 1, None, None]
            t = ((f - lo) / (hi - lo)).clamp(0, 1)
        else:
            t = ((f - vmin) / span).clamp(0, 1)[None]
        if (w, h) != (W0, H0):
            t = F.interpolate(t[None], size=(h, w), mode="area" if w < W0 else "bilinear", align_corners=None if w < W0 else False)[0]
        return t

    def loss_fn(img, gt):
        if cfg.mode == "multi":
            mask = (gt[0] > 0.02).float()
            l1 = (img[0] - gt[0]).abs().mean() + ((img[1:] - gt[1:]).abs() * mask[None]).sum() / (mask.sum() * (gt.shape[0] - 1) + 1)
            return 0.8 * l1 + 0.2 * (1 - ssim(img[0:1][None], gt[0:1][None]))
        return 0.8 * l1_loss(img, gt) + 0.2 * (1 - ssim(img[None], gt[None]))

    def sync_ms(fn):
        torch.cuda.synchronize(); t0 = time.perf_counter(); out = fn(); torch.cuda.synchronize()
        return (time.perf_counter() - t0) * 1000, out

    rows = []
    util_before = gpu_util()
    for s in cfg.scales:
        w, h = int(round(W0 * s)), int(round(H0 * s))
        Ks = data["Ks"].clone(); Ks[:, :2, :] *= s
        tg = [target(i, w, h) for i in range(len(pick))]
        parts = {"colours": [], "render_fwd": [], "loss_bwd": [], "adam": [], "step": []}
        for r in range(cfg.warmup + cfg.reps if cfg.faces == 1 else 0):
            i = r % len(pick)
            vm, K = data["viewmats"][i], Ks[i]
            torch.cuda.synchronize(); t_all = time.perf_counter()
            cam = torch.linalg.inv(vm)[:3, 3]
            t_col, _ = sync_ms(lambda: model.colours(cam))          # timed alone, recomputed inside render
            t_fwd, img = sync_ms(lambda: model.render(vm, K, w, h, span))
            for o in opts.values():
                o.zero_grad(set_to_none=True)
            t_bwd, _ = sync_ms(lambda: loss_fn(img, tg[i]).backward())
            t_adam, _ = sync_ms(lambda: [o.step() for o in opts.values()])
            torch.cuda.synchronize(); t_step = (time.perf_counter() - t_all) * 1000 - t_col
            if r >= cfg.warmup:
                for k, v in (("colours", t_col), ("render_fwd", t_fwd), ("loss_bwd", t_bwd), ("adam", t_adam), ("step", t_step)):
                    parts[k].append(v)
        # a clean step, untimed pieces removed: the number train_rrf.py's it/s corresponds to
        clean = []
        for r in range(cfg.warmup + cfg.reps):
            i = r % len(pick)
            torch.cuda.synchronize(); t0 = time.perf_counter()
            if cfg.faces == 1:
                img = model.render(data["viewmats"][i], Ks[i], w, h, span)
                loss = loss_fn(img, tg[i])
            else:
                g = list(range(4 * (r % 4), 4 * (r % 4) + 4))              # the four faces of one position
                imgs = model.render_batch(data["viewmats"][g], Ks[g], w, h, span)
                loss = sum(loss_fn(imgs[k], tg[j]) for k, j in enumerate(g)) / 4
            for o in opts.values():
                o.zero_grad(set_to_none=True)
            loss.backward()
            for o in opts.values():
                o.step()
            torch.cuda.synchronize()
            if r >= cfg.warmup:
                clean.append((time.perf_counter() - t0) * 1000)
        row = {"scale": s, "width": w, "height": h, "pixels": w * h, "faces_per_step": cfg.faces,
               "ms_per_view": float(np.median(clean)) / cfg.faces,
               **{f"{k}_ms_median": (float(np.median(v)) if v else None) for k, v in parts.items()},
               "clean_step_ms_median": float(np.median(clean)), "clean_step_ms_p10": float(np.percentile(clean, 10)),
               "clean_step_ms_p90": float(np.percentile(clean, 90))}
        rows.append(row)
        print(f"{w:4d}x{h:<4d} step {row['clean_step_ms_median']:6.1f} ms (p10 {row['clean_step_ms_p10']:.1f}, p90 {row['clean_step_ms_p90']:.1f}), "
              f"{row['ms_per_view']:5.1f} ms per view" + ("" if cfg.faces > 1 else
              f" | colours {row['colours_ms_median']:5.1f}  render fwd {row['render_fwd_ms_median']:5.1f}  "
              f"loss+bwd {row['loss_bwd_ms_median']:5.1f}  adam {row['adam_ms_median']:5.1f}"))
    out = {"mode": cfg.mode, "delay_depth": cfg.delay_depth, "channels": channels, "gaussians": model.n_gaussians,
           "gpu": torch.cuda.get_device_name(0), "gpu_util_before": util_before, "gpu_util_after": gpu_util(),
           "reps": cfg.reps, "warmup": cfg.warmup, "rows": rows,
           "sh_degree": cfg.sh_degree, "faces": cfg.faces, "sh_backend": cfg.sh_backend, "head": cfg.head,
           "head_guides": cfg.head_guides, "head_latent": cfg.head_latent, "head_strip": cfg.head_strip,
           "optimised_gaussian_params": sum(p.numel() for p in model.params.values() if p.requires_grad),
           "head_params": sum(p.numel() for p in model.head.parameters()) if model.head is not None else 0,
           "note": "render_fwd includes the colour evaluation; 'colours' is that evaluation timed alone. "
                   "clean_step = render + loss + backward + Adam, the train_rrf.py step."}
    tag = "" if (cfg.faces == 1 and cfg.sh_backend == "torch") else f"_f{cfg.faces}_{cfg.sh_backend}"
    if cfg.sh_degree != 3 or cfg.head != "none":
        tag += f"_sh{cfg.sh_degree}" + (f"_head-{cfg.head_guides.replace(',', '+') or 'base'}-l{cfg.head_latent}"
                                          + ("-strip" if cfg.head_strip else "") + (f"-w{cfg.head_width}" if cfg.head_width != 32 else "") if cfg.head != "none" else "")
    path = os.path.join(REPO, "output", "rrf", f"profile_resolution_{cfg.mode}{tag}.json")
    json.dump(out, open(path, "w"), indent=1)
    print(f"GPU before: {util_before}; wrote {path}")


if __name__ == "__main__":
    main()
