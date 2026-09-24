"""Smoke for the per-Gaussian backward in the Windows gsplat fork (gsplat_win, branch rf-win; PerGaussianBwd.h).

Each variant runs in its own process with its environment switches, renders the same inputs and takes one
backward with the same random output gradients (colour and alpha outputs):

  stock          the unmodified backward path (no switch)
  stock_dev      GSPLAT_BWD_DEVICE_ATOMICS=1   stock backward, device-scope atomics (the engineering part)
  stock_nogeom   GSPLAT_BWD_NO_GEOM=1          stock backward without the conic / 2D-mean gradients
  pg             GSPLAT_BWD_PERGAUSS=2         the per-Gaussian backward (Taming 3DGS)
  pg_nogeom      GSPLAT_BWD_PERGAUSS=2 GSPLAT_BWD_NO_GEOM=1
  pg_broken      GSPLAT_BWD_PERGAUSS=2 GSPLAT_BWD_PERGAUSS_BREAK=1   checkpoints' T zeroed: must be caught

Inputs (the rules for a smoke: stratified views, a boundary and a degenerate case, every channel count in use):
  one        a single Gaussian at the centre of a 32 x 32 image (analytic-scale case)
  crowd      300 random Gaussians in a 48 x 48 image: > 32 per tile, so buckets chain and the shuffle
             pipeline crosses bucket boundaries; white background and a random alpha-output gradient
  sh3        a trained SH3 db model (r41_B_B1_sh3), 1 channel
  head       the full shading-head model (r43_S5_full): colour + normals + latent + depth = 9 channels
  raw5       the scene's Gaussians, random 5 channels + expected depth (6), background 0.3, alpha gradient
Views: 4 held-out positions spread over the test set, all 4 faces (edges of the field of view included), plus a
batch whose camera sees nothing (no intersections: the per-Gaussian path must stand aside).

Criteria, written before the first run:
  a  forward unchanged: every variant's renders equal stock's bit for bit (max |diff| = 0)
  b  pg and pg_nogeom against stock: relative L2 difference <= 1e-4 for every gradient tensor they compute
     (colours / SH, opacities; means2d and conics too for pg, through the model's parameters where they are
     leaves) -- the same bound as the port check, whose run-to-run floor was 9.8e-7
  c  stock_dev and stock_nogeom against stock: the same bound (their colour / opacity sums are the stock ones)
  d  known failure: pg_broken differs from stock by more than 1e-2 in at least one colour / opacity gradient
  e  the per-Gaussian path actually ran (its one-time report appears on stderr for pg, pg_nogeom, pg_broken)
Diagnostic, not a criterion (added after a first quick run put stock-vs-pg at 4-9e-5 on "crowd", near b's
bound): on the two synthetic cases, a float64 re-implementation of gsplat's blending (its projected means /
conics / opacities and per-tile order, its alpha rules and exclusive stop) gives reference colour and opacity
gradients, and each variant's distance to it is reported -- which of the two float32 kernels is the less exact.

    C:/Users/Ke/miniconda3/envs/rf-gsplat-win/python.exe rrf_gsplat/check_pergauss_bwd.py
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
OUT = os.path.join(REPO, "output", "rrf", "pergauss_check")
VARIANTS = {
    "stock": {},
    "stock_dev": {"GSPLAT_BWD_DEVICE_ATOMICS": "1"},
    "stock_nogeom": {"GSPLAT_BWD_NO_GEOM": "1"},
    "pg": {"GSPLAT_BWD_PERGAUSS": "2"},
    "pg_nogeom": {"GSPLAT_BWD_PERGAUSS": "2", "GSPLAT_BWD_NO_GEOM": "1"},
    "pg_broken": {"GSPLAT_BWD_PERGAUSS": "2", "GSPLAT_BWD_PERGAUSS_BREAK": "1"},
}
SWITCHES = ("GSPLAT_BWD_DEVICE_ATOMICS", "GSPLAT_BWD_NO_GEOM", "GSPLAT_BWD_PERGAUSS", "GSPLAT_BWD_PERGAUSS_BREAK")


def run_variant(name):
    import torch
    sys.path.insert(0, HERE)
    import train_rrf as T
    from gsplat import rasterization
    from check_gsplat_port import batches, model_from_run, CK

    dev = torch.device("cuda")
    res = {}
    g = torch.Generator(device="cpu").manual_seed(0)

    def reference(meta, cols, opac, w, wa, W, H, bg):
        """float64 blending of one image with gsplat's rules; gradients of sum(img w) + sum(alpha wa)."""
        c64 = cols.detach().double().requires_grad_(True); o64 = opac.detach().double().requires_grad_(True)
        gid = meta["gaussian_ids"]
        m2 = meta["means2d"].detach().double().reshape(-1, 2); con = meta["conics"].detach().double().reshape(-1, 3)
        if gid is None:
            gid = torch.arange(m2.shape[0], device=dev)
        cc, oo = c64[gid], o64[gid]
        offs = meta["isect_offsets"].reshape(-1).tolist() + [meta["flatten_ids"].numel()]
        fid = meta["flatten_ids"].long()
        tw = meta["tile_width"]; ts = meta["tile_size"]
        img = torch.zeros(H, W, cc.shape[1], dtype=torch.float64, device=dev)
        alp = torch.zeros(H, W, dtype=torch.float64, device=dev)
        bgv = bg[0].double() if bg is not None else torch.zeros(cc.shape[1], dtype=torch.float64, device=dev)
        for t in range(len(offs) - 1):
            ty, tx = divmod(t, tw)
            ys, xs = torch.meshgrid(torch.arange(ty * ts, min((ty + 1) * ts, H), device=dev),
                                    torch.arange(tx * ts, min((tx + 1) * ts, W), device=dev), indexing="ij")
            ys, xs = ys.reshape(-1), xs.reshape(-1)
            ids = fid[offs[t]:offs[t + 1]]
            if ids.numel() == 0:
                img = img.index_put((ys, xs), bgv.expand(ys.numel(), -1)); continue
            dx = m2[ids, 0][None] - (xs[:, None].double() + 0.5); dy = m2[ids, 1][None] - (ys[:, None].double() + 0.5)
            sig = 0.5 * (con[ids, 0][None] * dx * dx + con[ids, 2][None] * dy * dy) + con[ids, 1][None] * dx * dy
            al = torch.clamp(oo[ids][None] * torch.exp(-sig), max=0.99)
            valid = (sig >= 0) & (al >= 1.0 / 255.0)
            ae = torch.where(valid, al, torch.zeros_like(al))
            cum = torch.cumprod(1 - ae, dim=1)                       # T after each Gaussian
            stop = valid & (cum <= 1e-4)
            first = torch.where(stop.any(1), stop.float().argmax(1), torch.full_like(stop[:, 0], ids.numel(), dtype=torch.long))
            keep = torch.arange(ids.numel(), device=dev)[None] < first[:, None]
            ae = ae * keep
            Tb = torch.cat([torch.ones_like(ae[:, :1]), torch.cumprod(1 - ae, 1)[:, :-1]], 1)   # T before each
            Tf = torch.prod(1 - ae, 1)
            col = (ae * Tb) @ cc[ids] + Tf[:, None] * bgv[None]
            img = img.index_put((ys, xs), col); alp = alp.index_put((ys, xs), 1 - Tf)
        ((img * w[0].double()).sum() + (alp * wa[0, ..., 0].double()).sum()).backward()
        return c64.grad.cpu().numpy(), o64.grad.cpu().numpy()

    def raster_case(tag, means, quats, scales, opac, cols, vm, K, W, H, bg, mode, ref=False):
        cols = cols.clone().requires_grad_(True); opac = opac.clone().requires_grad_(True)
        means = means.clone().requires_grad_(True)
        img, alpha, meta = rasterization(means, quats, scales, opac, cols, vm, K, W, H, sh_degree=None,
                                         backgrounds=bg, render_mode=mode)
        w = torch.randn(img.shape, generator=g).to(dev); wa = torch.randn(alpha.shape, generator=g).to(dev)
        ((img * w).sum() + (alpha * wa).sum()).backward()
        res[f"{tag}/render"] = img.detach().cpu().numpy()
        res[f"{tag}/grad_colors"] = cols.grad.cpu().numpy()
        res[f"{tag}/grad_opacities"] = opac.grad.cpu().numpy()
        res[f"{tag}/grad_means"] = means.grad.cpu().numpy()
        if ref:
            rc, ro = reference(meta, cols, opac, w, wa, W, H, bg)
            res[f"{tag}/ref64_colors"] = rc; res[f"{tag}/ref64_opacities"] = ro

    # synthetic cases: a camera at the origin looking down +z
    def cam(W, H, f):
        return torch.eye(4, device=dev)[None], torch.tensor([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1.0]], device=dev)[None]
    vm, K = cam(32, 32, 40.0)
    raster_case("one", torch.tensor([[0.0, 0.0, 4.0]], device=dev), torch.tensor([[1.0, 0, 0, 0]], device=dev),
                torch.full((1, 3), -1.5, device=dev).exp(), torch.tensor([0.8], device=dev),
                torch.rand(1, 3, generator=g).to(dev), vm, K, 32, 32, None, "RGB", ref=True)
    n = 300
    means = torch.cat([torch.rand(n, 2, generator=g) * 2 - 1, torch.rand(n, 1, generator=g) * 3 + 3], 1).to(dev)
    quats = torch.nn.functional.normalize(torch.randn(n, 4, generator=g), dim=-1).to(dev)
    scales = (torch.rand(n, 3, generator=g) * 0.25 + 0.05).to(dev)
    vm, K = cam(48, 48, 50.0)
    raster_case("crowd", means, quats, scales, (torch.rand(n, generator=g) * 0.9 + 0.05).to(dev),
                torch.rand(n, 3, generator=g).to(dev), vm, K, 48, 48, torch.ones(1, 3, device=dev), "RGB", ref=True)

    # the scene
    meta, bs = batches(dev)
    for tag, run_name in (("sh3", "r41_B_B1_sh3"), ("head", "r43_S5_full")):
        m, span = model_from_run(run_name, meta, dev)
        params = {k: p for k, p in m.params.items() if p.requires_grad}
        for bi, (vmb, Kb, W, H) in enumerate(bs):
            for p in list(params.values()) + (list(m.head.parameters()) if m.head is not None else []):
                p.grad = None
            out = m.render_batch(vmb, Kb, W, H, span)
            w = torch.randn(out.shape, generator=g).to(dev)
            (out * w).sum().backward()
            res[f"{tag}/b{bi}/render"] = out.detach().cpu().numpy()
            for k, p in params.items():
                res[f"{tag}/b{bi}/grad_{k}"] = (p.grad if p.grad is not None else torch.zeros_like(p)).cpu().numpy()
        del m
        torch.cuda.empty_cache()
    m = T.RRF(CK, "db", 1, 1, dev)
    cols = torch.rand(m.n_gaussians, 5, generator=g).to(dev)
    for bi, (vmb, Kb, W, H) in enumerate(bs):
        raster_case(f"raw5/b{bi}", m.means.detach(), m.quats.detach(), m.scales.detach(), m.opacities.detach(), cols,
                    vmb, Kb, W, H, torch.full((vmb.shape[0], 6), 0.3, device=dev), "RGB+ED")
    # diagnostic on the real geometry (run 2): one face of the batch where run 1's largest difference was, 6 random
    # channels, background 0.3, against the float64 blending
    vmb, Kb, W, H = bs[1]
    cols6 = torch.rand(m.n_gaussians, 6, generator=g).to(dev)
    raster_case("real1", m.means.detach(), m.quats.detach(), m.scales.detach(), m.opacities.detach(), cols6,
                vmb[:1], Kb[:1], W, H, torch.full((1, 6), 0.3, device=dev), "RGB", ref=True)
    os.makedirs(OUT, exist_ok=True)
    np.savez_compressed(os.path.join(OUT, f"{name}.npz"), **res)


def compare():
    ref = np.load(os.path.join(OUT, "stock.npz"))
    report, ok = {}, True
    for name in VARIANTS:
        if name == "stock":
            continue
        X = np.load(os.path.join(OUT, f"{name}.npz"))
        worst_r, worst_g, worst_key = 0.0, 0.0, ""
        for k in ref.files:
            a, b = ref[k].astype(np.float64), X[k].astype(np.float64)
            if "/render" in k:
                worst_r = max(worst_r, float(np.abs(a - b).max()) if a.size else 0.0)
                continue
            if "nogeom" in name and "grad_means" in k:
                continue                                   # not computed without the geometry gradients
            na = np.linalg.norm(a)
            d = float(np.linalg.norm(a - b) / na) if na > 0 else float(np.linalg.norm(b))
            if d > worst_g:
                worst_g, worst_key = d, k
        report[name] = {"renders_max_abs_diff": worst_r, "grad_max_rel_l2": worst_g, "worst": worst_key}
        if name == "pg_broken":
            passed = worst_r == 0.0 and worst_g > 1e-2
        else:
            passed = worst_r == 0.0 and worst_g <= 1e-4
        report[name]["pass"] = passed
        ok &= passed
        print(f"{name:13s} renders max |diff| {worst_r:.1e}   gradients max rel L2 {worst_g:.2e} ({worst_key})"
              f"   -> {'PASS' if passed else 'FAIL'}")
    print("diagnostic: distance to the float64 blending (relative L2), colours / opacities")
    for name in VARIANTS:
        X = np.load(os.path.join(OUT, f"{name}.npz"))
        cells = []
        for tag in ("one", "crowd", "real1"):
            for q in ("colors", "opacities"):
                r = X[f"{tag}/ref64_{q}"].astype(np.float64); v = X[f"{tag}/grad_{q}"].astype(np.float64)
                cells.append(f"{tag} {q} {np.linalg.norm(v - r) / max(np.linalg.norm(r), 1e-30):.2e}")
        report.setdefault(name, {})["vs_float64"] = cells
        print(f"  {name:13s} " + "   ".join(cells))
    ran = json.load(open(os.path.join(OUT, "ran.json")))
    for name in ("pg", "pg_nogeom", "pg_broken"):
        print(f"e {name}: per-Gaussian path reported on stderr: {ran.get(name)}")
        ok &= bool(ran.get(name))
    json.dump(report, open(os.path.join(OUT, "report.json"), "w"), indent=1)
    print("ALL PASS" if ok else "FAIL")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant")
    a = ap.parse_args()
    if a.variant:
        run_variant(a.variant)
    else:
        ran = {}
        for name, env in VARIANTS.items():
            e = {k: v for k, v in os.environ.items() if k not in SWITCHES} | env | {"PYTHONUTF8": "1"}
            p = subprocess.run([sys.executable, __file__, "--variant", name], env=e, capture_output=True, text=True)
            ran[name] = "per-Gaussian backward:" in p.stderr
            if p.returncode != 0:
                print(p.stderr[-3000:]); raise SystemExit(f"{name} failed")
            print(f"ran {name}")
        json.dump(ran, open(os.path.join(OUT, "ran.json"), "w"))
        compare()
