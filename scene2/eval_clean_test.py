"""Stage-1 PSNR of the corridor checkpoint against noise-free targets.

The training frames are 64-spp Monte-Carlo renders; two renders of the same
view differ by 25.2 dB PSNR (measured), so a PSNR against them is capped near
that floor and not comparable with the lobby's 39.4 dB (its frames are clean
Blender renders). This re-renders only the held-out test poses at high spp and
scores the trainer's test renders (render.py --skip_train) against both the
noisy and the clean targets.

    PYTHONUTF8=1 python scene2/eval_clean_test.py --dataset scene2/visual_dataset --model output/scene2_visual --iteration 30000 --spp 512
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np


def psnr(a, b):
    a = a.astype(np.float64) / 255.0; b = b.astype(np.float64) / 255.0
    mse = float(np.mean((a - b) ** 2))
    return 10 * np.log10(1.0 / mse) if mse > 0 else float("inf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True); ap.add_argument("--model", required=True)
    ap.add_argument("--scene", default="scene2/corridor"); ap.add_argument("--iteration", type=int, default=30000)
    ap.add_argument("--spp", type=int, default=512); ap.add_argument("--width", type=int, default=800); ap.add_argument("--height", type=int, default=450)
    cfg = ap.parse_args()
    from PIL import Image
    frames = json.load(open(os.path.join(cfg.dataset, "transforms_test.json")))["frames"]
    clean_dir = os.path.join(cfg.dataset, f"test_clean_spp{cfg.spp}")
    os.makedirs(clean_dir, exist_ok=True)
    todo = [f for f in frames if not os.path.exists(os.path.join(clean_dir, os.path.basename(f["file_path"])))]
    if todo:
        import mitsuba as mi
        mi.set_variant("cuda_ad_rgb")
        scene = mi.load_file(os.path.join(cfg.scene, "corridor_visual.xml"), spp=cfg.spp, resx=cfg.width, resy=cfg.height)
        t0 = time.time()
        for k, f in enumerate(todo):
            m = np.array(f["transform_matrix"], dtype=np.float64)
            m[:3, 0] *= -1; m[:3, 2] *= -1          # Blender -> Mitsuba (x left, y up, z forward); same as render_visual.py
            sensor = mi.load_dict({"type": "perspective", "fov": 90.0, "fov_axis": "x", "near_clip": 0.05, "far_clip": 1000.0,
                                   "to_world": mi.ScalarTransform4f(m.tolist()),
                                   "sampler": {"type": "independent", "sample_count": cfg.spp},
                                   "film": {"type": "hdrfilm", "width": cfg.width, "height": cfg.height, "pixel_format": "rgba",
                                            "rfilter": {"type": "gaussian"}}})
            img = mi.render(scene, sensor=sensor, spp=cfg.spp, seed=10_000 + k)
            rgba = np.clip(np.asarray(mi.Bitmap(img).convert(mi.Bitmap.PixelFormat.RGBA, mi.Struct.Type.UInt8, srgb_gamma=True)), 0, 255)
            Image.fromarray(rgba).save(os.path.join(clean_dir, os.path.basename(f["file_path"])))
        print(f"rendered {len(todo)} clean test frames at {cfg.spp} spp in {time.time() - t0:.0f} s -> {clean_dir}")
    render_dir = os.path.join(cfg.model, "test", f"ours_{cfg.iteration}", "renders")
    if not os.path.isdir(render_dir):
        print(f"no trainer renders at {render_dir}; run render.py -m {cfg.model} -s {cfg.dataset} --skip_train first"); return
    noisy, clean, floor = [], [], []
    for i, f in enumerate(frames):
        name = os.path.basename(f["file_path"])
        r = np.asarray(Image.open(os.path.join(render_dir, f"{i:05d}.png")).convert("RGB"))
        g_noisy = np.asarray(Image.open(os.path.join(cfg.dataset, f["file_path"])).convert("RGB"))
        g_clean = np.asarray(Image.open(os.path.join(clean_dir, name)).convert("RGB"))
        noisy.append(psnr(r, g_noisy)); clean.append(psnr(r, g_clean)); floor.append(psnr(g_noisy, g_clean))
    for label, v in (("render vs noisy 64-spp target (the training log's number)", noisy), (f"render vs clean {cfg.spp}-spp target", clean),
                     (f"noisy target vs clean target (the noise floor)", floor)):
        v = np.array(v)
        print(f"{label}: mean {v.mean():.2f} dB, median {np.median(v):.2f}, P10 {np.percentile(v, 10):.2f}, min {v.min():.2f}  (n={len(v)})")
    json.dump({"iteration": cfg.iteration, "spp_clean": cfg.spp, "psnr_vs_noisy": noisy, "psnr_vs_clean": clean, "noise_floor": floor},
              open(os.path.join(cfg.model, f"clean_test_psnr_{cfg.iteration}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
