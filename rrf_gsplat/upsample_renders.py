"""Bilinearly upsample a run's saved renders (native units, [C, h, w] .npy) to another size: the 2D upsampler of
the super-resolution comparison, applied to the renders of a model trained and rendered at low resolution.

Pixel centres are aligned (align_corners=False), which is what a pinhole camera with the focal length and the
principal point scaled by the same factor means. The azimuth is carried as (cos, sin), so interpolating the
channels is interpolating the vector, as eval_baselines' interp2 does.

    python rrf_gsplat/upsample_renders.py output/rrf/r26_T150b output/rrf/r26_T150b_up --size 200 300
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src"); ap.add_argument("dst")
    ap.add_argument("--size", type=int, nargs=2, required=True, metavar=("H", "W"))
    a = ap.parse_args()
    src_r, dst_r = os.path.join(a.src, "renders"), os.path.join(a.dst, "renders")
    os.makedirs(dst_r, exist_ok=True)
    names = sorted(f for f in os.listdir(src_r) if f.endswith(".npy"))
    for f in names:
        x = torch.from_numpy(np.load(os.path.join(src_r, f)).astype(np.float32))
        single = x.dim() == 2                                    # a db render is [h, w], a multi one [C, h, w]
        y = torch.nn.functional.interpolate((x[None] if single else x)[None], size=tuple(a.size), mode="bilinear",
                                            align_corners=False)[0]
        np.save(os.path.join(dst_r, f), (y[0] if single else y).numpy())
    r = json.load(open(os.path.join(a.src, "results.json")))
    r["upsampled_from"] = {"run": a.src, "size": a.size, "method": "bilinear, align_corners=False"}
    json.dump(r, open(os.path.join(a.dst, "results.json"), "w"), indent=1)
    print(f"upsampled {len(names)} renders {tuple(x.shape[1:])} -> {tuple(a.size)} into {dst_r}")


if __name__ == "__main__":
    main()
