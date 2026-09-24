"""Edge cases for the Windows-native gsplat (gsplat_win, switches off) against the WSL build. check_gsplat_port.py
covered the trained scene; this covers the boundaries of the rasteriser's input space.

    WSL:     /home/ke/miniconda3/envs/rf-gsplat/bin/python rrf_gsplat/check_gsplat_port_edge.py --tag wsl
    Windows: C:/Users/Ke/miniconda3/envs/rf-gsplat-win/python.exe rrf_gsplat/check_gsplat_port_edge.py --tag win
    either:  python rrf_gsplat/check_gsplat_port_edge.py --compare wsl win

Every case renders with gsplat.rasterization and takes one backward with fixed random output gradients; the
inputs are generated on the CPU from a fixed seed, so both builds see the same numbers:
  sizes     301 x 199, 17 x 15, 1 x 1, 640 x 480 (tiles cut by the image edge; a single pixel)
  cameras   8 cameras in one call, packed and unpacked
  extremes  Gaussians behind the camera, at the near plane, one covering the whole image, sub-pixel ones,
            needles, opacities 1e-4 and 0.9999 (the alpha clamp)
  sh        SH coefficients evaluated inside rasterization, degrees 0-3
  modes     RGB, D, ED, RGB+D, RGB+ED; 1, 2, 4, 8, 16 and 32 channels
  options   white background, a tile mask, absgrad, antialiased mode
  shfun     gsplat.spherical_harmonics alone (the colour path of train_rrf.py), degrees 0-3, with its backward
Criteria (written before the run): outputs max |diff| <= 1e-6 relative to each output's max magnitude (depth
renders are in metres); gradients relative L2 <= 1e-4; a case that raises must raise in both builds.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.abspath(os.path.join(HERE, "..")), "output", "rrf", "port_check")


def scene(n, g, spread=1.0, depth=(3.0, 6.0), size=(0.05, 0.3)):
    means = torch.cat([(torch.rand(n, 2, generator=g) * 2 - 1) * spread,
                       torch.rand(n, 1, generator=g) * (depth[1] - depth[0]) + depth[0]], 1)
    quats = torch.nn.functional.normalize(torch.randn(n, 4, generator=g), dim=-1)
    scales = torch.rand(n, 3, generator=g) * (size[1] - size[0]) + size[0]
    opac = torch.rand(n, generator=g) * 0.9 + 0.05
    return means, quats, scales, opac


def cams(C, W, H, g, f=None):
    vm = torch.eye(4).repeat(C, 1, 1)
    if C > 1:                                   # small rotations / shifts per camera
        ang = (torch.rand(C, generator=g) - 0.5) * 0.4
        vm[:, 0, 0] = torch.cos(ang); vm[:, 0, 2] = torch.sin(ang); vm[:, 2, 0] = -torch.sin(ang); vm[:, 2, 2] = torch.cos(ang)
        vm[:, :3, 3] = (torch.rand(C, 3, generator=g) - 0.5) * 0.3
    f = f or 0.8 * max(W, H)
    K = torch.tensor([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1.0]]).repeat(C, 1, 1)
    return vm, K


def run(tag):
    from gsplat import rasterization, spherical_harmonics
    dev = torch.device("cuda")
    g = torch.Generator().manual_seed(0)
    res, errs = {}, {}

    def case(name, means, quats, scales, opac, colors, vm, K, W, H, **kw):
        try:
            leaves = {"means": means.clone(), "opac": opac.clone(), "colors": colors.clone(), "scales": scales.clone()}
            t = {k: v.to(dev).requires_grad_(True) for k, v in leaves.items()}
            img, alpha, meta = rasterization(t["means"], quats.to(dev), t["scales"], t["opac"], t["colors"], vm.to(dev),
                                             K.to(dev), W, H, **kw)
            wi = torch.randn(img.shape, generator=g).to(dev); wa = torch.randn(alpha.shape, generator=g).to(dev)
            ((img * wi).sum() + (alpha * wa).sum()).backward()
            res[f"{name}/img"] = img.detach().cpu().numpy(); res[f"{name}/alpha"] = alpha.detach().cpu().numpy()
            for k, v in t.items():
                res[f"{name}/grad_{k}"] = (v.grad if v.grad is not None else torch.zeros_like(v)).cpu().numpy()
            if kw.get("absgrad"):
                res[f"{name}/absgrad"] = meta["means2d"].absgrad.detach().cpu().numpy()
        except Exception as e:                   # noqa: BLE001
            errs[name] = f"{type(e).__name__}: {str(e).splitlines()[0][:200]}"

    base = scene(200, g)
    rgb = lambda n, c=3: torch.rand(n, c, generator=g)                           # noqa: E731
    for W, H in ((301, 199), (17, 15), (1, 1), (640, 480)):
        vm, K = cams(1, W, H, g)
        case(f"size_{W}x{H}", *base, rgb(200), vm, K, W, H)
    vm, K = cams(8, 64, 48, g)
    for packed in (True, False):
        case(f"cams8_packed{packed}", *base, rgb(200), vm, K, 64, 48, packed=packed)
    vm, K = cams(1, 64, 48, g)
    m, q, s, o = scene(60, g)
    m[:10, 2] = -2.0                                                             # behind the camera
    m[10:20, 2] = 0.011                                                          # at the near plane (0.01)
    m[20] = torch.tensor([0.0, 0.0, 4.0]); s[20] = torch.tensor([5.0, 5.0, 5.0])  # covers the whole image
    s[21:30] = 1e-4                                                              # sub-pixel
    s[30:40, 0] = 2.0; s[30:40, 1:] = 0.005                                      # needles
    o[40:45] = 1e-4; o[45:50] = 0.9999                                           # alpha threshold / clamp
    case("extremes", m, q, s, o, rgb(60), vm, K, 64, 48)
    for deg in range(4):
        case(f"sh_deg{deg}", *base, torch.randn(200, (deg + 1) ** 2, 3, generator=g) * 0.3, vm, K, 64, 48, sh_degree=deg)
    for mode in ("RGB", "D", "ED", "RGB+D", "RGB+ED"):
        case(f"mode_{mode}", *base, rgb(200), vm, K, 64, 48, render_mode=mode)
    for c in (1, 2, 4, 8, 16, 32):
        case(f"channels_{c}", *base, rgb(200, c), vm, K, 64, 48)
    case("background_white", *base, rgb(200), vm, K, 64, 48, backgrounds=torch.ones(1, 3).to(dev))
    case("absgrad", *base, rgb(200), vm, K, 64, 48, absgrad=True)
    case("antialiased", *base, rgb(200), vm, K, 64, 48, rasterize_mode="antialiased")
    # a tile mask through the lower-level API
    try:
        from gsplat.cuda._wrapper import rasterize_to_pixels
        _, _, meta = rasterization(*(x.to(dev) for x in base), rgb(200).to(dev), vm.to(dev), K.to(dev), 64, 48)
        mask = (torch.rand(meta["tile_height"], meta["tile_width"], generator=g) > 0.4)[None].to(dev)
        cols = rgb(200).to(dev)[meta["gaussian_ids"]] if meta["gaussian_ids"] is not None else rgb(200).to(dev)[None]
        cols.requires_grad_(True)
        img, alpha = rasterize_to_pixels(meta["means2d"], meta["conics"], cols, meta["opacities"], 64, 48,
                                         meta["tile_size"], meta["isect_offsets"], meta["flatten_ids"], masks=mask,
                                         packed=meta["gaussian_ids"] is not None)
        wi = torch.randn(img.shape, generator=g).to(dev)
        (img * wi).sum().backward()
        res["mask/img"] = img.detach().cpu().numpy(); res["mask/grad_colors"] = cols.grad.cpu().numpy()
    except Exception as e:                       # noqa: BLE001
        errs["mask"] = f"{type(e).__name__}: {str(e).splitlines()[0][:200]}"
    for deg in range(4):
        try:
            coeffs = (torch.randn(500, (deg + 1) ** 2, 3, generator=g) * 0.3).to(dev).requires_grad_(True)
            means = (torch.randn(500, 3, generator=g) * 3).to(dev)
            vmx = torch.eye(4)[None].to(dev); vmx[0, :3, 3] = torch.tensor([0.3, -0.2, 0.1]).to(dev)
            out = spherical_harmonics(deg, means, vmx, coeffs)
            (out * torch.randn(out.shape, generator=g).to(dev)).sum().backward()
            res[f"shfun_deg{deg}/out"] = out.detach().cpu().numpy(); res[f"shfun_deg{deg}/grad_coeffs"] = coeffs.grad.cpu().numpy()
        except Exception as e:                   # noqa: BLE001
            errs[f"shfun_deg{deg}"] = f"{type(e).__name__}: {str(e).splitlines()[0][:200]}"
    os.makedirs(OUT, exist_ok=True)
    np.savez_compressed(os.path.join(OUT, f"edge_{tag}.npz"), **res)
    json.dump(errs, open(os.path.join(OUT, f"edge_{tag}_errors.json"), "w"), indent=1)
    print(f"{tag}: {len(res)} arrays, {len(errs)} cases raised: {errs}")


def compare(a, b):
    A = np.load(os.path.join(OUT, f"edge_{a}.npz")); B = np.load(os.path.join(OUT, f"edge_{b}.npz"))
    ea = json.load(open(os.path.join(OUT, f"edge_{a}_errors.json"))); eb = json.load(open(os.path.join(OUT, f"edge_{b}_errors.json")))
    ok = set(ea) == set(eb) and set(A.files) == set(B.files)
    print(f"cases that raised: {a} {sorted(ea)} | {b} {sorted(eb)} -> {'same' if set(ea) == set(eb) else 'DIFFERENT'}")
    worst_o, worst_g = (0.0, ""), (0.0, "")
    for k in sorted(set(A.files) & set(B.files)):
        x, y = A[k].astype(np.float64), B[k].astype(np.float64)
        if "grad" in k or k.endswith("absgrad"):
            n = np.linalg.norm(x)
            d = float(np.linalg.norm(x - y) / n) if n > 0 else float(np.linalg.norm(y))
            if d > worst_g[0]:
                worst_g = (d, k)
            bad = d > 1e-4
        else:
            scale = max(float(np.abs(x).max()) if x.size else 0.0, 1e-30)
            d = float(np.abs(x - y).max()) / scale if x.size else 0.0
            if d > worst_o[0]:
                worst_o = (d, k)
            bad = d > 1e-6
        if bad:
            print(f"FAIL {k:32s} {d:.3e}")
        ok &= not bad
    print(f"outputs max relative diff {worst_o[0]:.2e} ({worst_o[1]}), gradients max relative L2 {worst_g[0]:.2e} ({worst_g[1]})")
    print("ALL PASS" if ok else "FAIL")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag"); ap.add_argument("--compare", nargs=2)
    a = ap.parse_args()
    compare(*a.compare) if a.compare else run(a.tag)
