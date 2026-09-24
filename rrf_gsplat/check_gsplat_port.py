"""Does the Windows-native gsplat (gsplat_win, env rf-gsplat-win) compute what the WSL build computes?

Run once per environment, then compare:

    WSL:     /home/ke/miniconda3/envs/rf-gsplat/bin/python rrf_gsplat/check_gsplat_port.py --tag wsl
    Windows: C:/Users/Ke/miniconda3/envs/rf-gsplat-win/python.exe rrf_gsplat/check_gsplat_port.py --tag win
    either:  python rrf_gsplat/check_gsplat_port.py --compare wsl win

Each run renders the same inputs and takes one backward with the same random output gradient:
  sh3      a trained SH3 db model (r41_B_B1_sh3): 1 channel
  head     a trained full shading-head model (r43_S5_full): colour + normals + latent + depth = 9 channels
  raw5     the Gaussians with fixed random 5-channel colours + expected depth (6 channels), gsplat.rasterization direct
Views: 4 held-out positions spread over the test set (np.linspace, all 4 faces each, the renderer's batch of 4),
plus one degenerate batch: a camera moved 1e6 m behind itself, which sees no Gaussian (no intersections).
Criteria (written before the first run): renders max |diff| <= 1e-6; each gradient tensor's relative L2 difference
<= 1e-4 (the backward scatters with atomics, so its summation order is not fixed); the degenerate batch renders
exactly zero in both (sh3 and raw5; the head adds its residual even to an empty base).
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import train_rrf as T  # noqa: E402

REPO = T.REPO
MV = os.path.join(REPO, "RF-3DGS_dataset/regenerated/3dgs_MVDR_100_gpct")
CK = os.path.join(REPO, "RF-3DGS_dataset/blender_visual_trained/chkpnt30000.pth")
OUT = os.path.join(REPO, "output", "rrf", "port_check")


def batches(dev):
    meta = json.load(open(os.path.join(MV, "generation_meta.json")))
    views = T.read_colmap_text(os.path.join(MV, "sparse", "0"))
    names = T.read_index(os.path.join(MV, "test_index.txt"))
    pos = np.linspace(0, len(names) // 4 - 1, 4).round().astype(int)
    out = []
    for p in pos:
        d = T.load_views(MV, [names[p * 4 + k] for k in range(4)], views, dev, want_float=False)
        out.append((d["viewmats"], d["Ks"], d["width"], d["height"]))
    vm = out[0][0].clone()
    vm[:, 2, 3] -= 1e6                                         # everything ends up behind the camera
    out.append((vm, out[0][1], out[0][2], out[0][3]))
    return meta, out


def model_from_run(run, meta, dev):
    cfg = json.load(open(os.path.join(REPO, "output", "rrf", run, "results.json")))["config"]
    span = meta["spec_max_db"] - meta["spec_min_db"]
    m = T.RRF(CK, "db", 1, cfg["sh_degree"], dev)
    m.sh_backend = "gsplat"
    if cfg.get("head", "none") != "none":
        T.attach_head(m, cfg["head"], cfg["head_guides"], cfg["head_latent"], cfg["head_strip"], cfg["head_max_db"],
                      cfg["head_width"], span, meta.get("tx_loc"), dev, cfg.get("head_bound_units", False))
    m.load_state(torch.load(os.path.join(REPO, "output", "rrf", run, "rrf_state.pt"), map_location=dev))
    return m, span


def run(tag):
    dev = torch.device("cuda")
    import gsplat
    meta, bs = batches(dev)
    res = {"gsplat": gsplat.__version__, "gsplat_file": gsplat.__file__}
    g = torch.Generator(device="cpu").manual_seed(0)
    for name, run_name in (("sh3", "r41_B_B1_sh3"), ("head", "r43_S5_full")):
        m, span = model_from_run(run_name, meta, dev)
        params = {k: p for k, p in m.params.items() if p.requires_grad}
        if m.head is not None:
            m.head.eval()
        for bi, (vm, K, W, H) in enumerate(bs):
            for p in list(params.values()) + (list(m.head.parameters()) if m.head is not None else []):
                p.grad = None
            out = m.render_batch(vm, K, W, H, span)
            w = torch.randn(out.shape, generator=g).to(dev)
            (out * w).sum().backward()
            res[f"{name}/b{bi}/render"] = out.detach().cpu().numpy()
            for k, p in params.items():
                res[f"{name}/b{bi}/grad_{k}"] = (p.grad if p.grad is not None else torch.zeros_like(p)).cpu().numpy()
        del m
        torch.cuda.empty_cache()
    # raw 5 channels + expected depth through gsplat.rasterization
    from gsplat import rasterization
    m = T.RRF(CK, "db", 1, 1, dev)
    cols = torch.rand(m.n_gaussians, 5, generator=g).to(dev).requires_grad_(True)
    opac = m.opacities.detach().clone().requires_grad_(True)
    for bi, (vm, K, W, H) in enumerate(bs):
        cols.grad = None; opac.grad = None
        img, alpha, _ = rasterization(m.means, m.quats, m.scales, opac, cols, vm, K, W, H, sh_degree=None,
                                      backgrounds=torch.zeros(vm.shape[0], 6, device=dev), render_mode="RGB+ED")
        w = torch.randn(img.shape, generator=g).to(dev)
        wa = torch.randn(alpha.shape, generator=g).to(dev)
        ((img * w).sum() + (alpha * wa).sum()).backward()
        res[f"raw5/b{bi}/render"] = img.detach().cpu().numpy()
        res[f"raw5/b{bi}/alpha"] = alpha.detach().cpu().numpy()
        res[f"raw5/b{bi}/grad_colors"] = cols.grad.cpu().numpy()
        res[f"raw5/b{bi}/grad_opacities"] = opac.grad.cpu().numpy()
    os.makedirs(OUT, exist_ok=True)
    np.savez_compressed(os.path.join(OUT, f"{tag}.npz"), **{k: v for k, v in res.items() if isinstance(v, np.ndarray)})
    json.dump({k: v for k, v in res.items() if not isinstance(v, np.ndarray)}, open(os.path.join(OUT, f"{tag}.json"), "w"))
    print(f"{tag}: gsplat {res['gsplat']} from {res['gsplat_file']}; wrote {len(res) - 2} arrays")


def compare(a, b):
    A = np.load(os.path.join(OUT, f"{a}.npz")); B = np.load(os.path.join(OUT, f"{b}.npz"))
    worst_r, worst_g, ok = 0.0, 0.0, True
    rows = []
    for k in sorted(A.files):
        x, y = A[k].astype(np.float64), B[k].astype(np.float64)
        if "/grad_" in k:
            d = float(np.linalg.norm(x - y) / max(np.linalg.norm(x), 1e-30))
            worst_g = max(worst_g, d); bad = d > 1e-4
        else:
            d = float(np.abs(x - y).max()) if x.size else 0.0
            worst_r = max(worst_r, d); bad = d > 1e-6
        ok &= not bad
        rows.append((k, d, bad))
    for k, d, bad in rows:
        if bad or "/b4/" in k:
            print(f"{'FAIL' if bad else '    '} {k:32s} {d:.3e}")
    # without a head nothing is drawn there (the head adds its residual even to an empty base)
    degenerate = all(float(np.abs(A[k]).max()) == 0.0 and float(np.abs(B[k]).max()) == 0.0
                     for k in A.files if "/b4/render" in k and not k.startswith("head/"))
    print(f"renders max |diff| {worst_r:.3e} (<= 1e-6), gradients max relative L2 {worst_g:.3e} (<= 1e-4), "
          f"degenerate batch renders zero in both: {degenerate}")
    print("ALL PASS" if ok and degenerate else "FAIL")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag")
    ap.add_argument("--compare", nargs=2)
    a = ap.parse_args()
    compare(*a.compare) if a.compare else run(a.tag)
