"""Smoke: the delay range term from gsplat's native along-ray hit distance (--delay-range hit) against ours (euclid).

The delay decomposition adds a range term |p - mu| / c to a learned residual. "euclid" (the method's default since
round 19) takes gsplat's composited camera z of the Gaussian centres (render mode "ED") times sec(theta_pixel): the
distance along the pixel's ray to the plane through the centre parallel to the image. gsplat's eval3d rasteriser
reports the hit distance itself ("Ed"): the distance along the pixel's own ray to the point of the Gaussian's
maximum response, t* = d' S^-1 mu / d' S^-1 d. The two agree on the ray through the centre only; for a flat
Gaussian t* is the ray's intersection with the Gaussian's plane.

S-A  analytic control, one camera and one Gaussian at a time (isotropic, and flat tilted 60 deg):
       native "Ed" against t* per pixel, and "ED" x sec(theta) against mu_z / d_z per pixel.
S-B  the lobby's visual checkpoint at the 640 held-out views of 3dgs_MULTI_24ghz_tut_cs, against the mesh range
     (depth_gs.py / depth_gt.py caches: the mesh's camera z per pixel, cast through Mitsuba). Pixels: mesh hit and
     alpha > 0.5, as depth_gt.py. Three range terms: z (ED, the first runs), euclid (ED x sec), hit (native Ed).
S-C  (rounds/win_hit_smoke.sh) a short delay-channel training, euclid vs hit, same seed.

Pass criteria, written before the first run (out of range = reported, never adjusted):
  S-A  native vs t*: median |e| <= 1 mm and P99 <= 5 mm, both Gaussians; euclid vs mu_z / d_z: median <= 1 mm
  S-B  control (known number): mean (mesh range - mesh z) / c on the mask reproduces depth_vs_mesh_lobby.json
       (2.18 ns) within 0.01 ns; re-rendered ED equals the cached depth, median |d| <= 1 mm
       known-bad: the z term's median |e| against the mesh range exceeds both euclid's and hit's
       hypothesis, stated before: hit's median |e| and RMSE against the mesh range <= euclid's
       divergence, expected range: median |hit - euclid| 0.05 - 1.0 ns, P90 0.3 - 5 ns
       degenerate: pixels with 0.1 < alpha <= 0.5 reported separately, no criterion
       timing: the extra eval3d depth pass per 4-face batch, reported; void when the GPU is shared

    PYTHONUTF8=1 python rrf_gsplat/smoke_hit_distance.py
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_rrf as T          # noqa: E402

C = 0.299792458                # m / ns


def quat_about_y(deg):
    h = math.radians(deg) / 2
    return [math.cos(h), 0.0, math.sin(h), 0.0]          # (w, x, y, z), gsplat's order


def s_a(dev):
    from gsplat import rasterization
    w, h, f = 300, 200, 150.0
    K = torch.tensor([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], device=dev)[None]
    vm = torch.eye(4, device=dev)[None]
    u = (torch.arange(w, device=dev, dtype=torch.float64) + 0.5 - w / 2) / f
    v = (torch.arange(h, device=dev, dtype=torch.float64) + 0.5 - h / 2) / f
    d = torch.stack([u[None, :].expand(h, w), v[:, None].expand(h, w), torch.ones(h, w, dtype=torch.float64, device=dev)], -1)
    sec = d.norm(dim=-1); d = d / sec[..., None]
    out, ok = {}, True
    for name, mu, sc, q in (("isotropic", [0.8, 0.3, 4.0], [0.05, 0.05, 0.05], [1.0, 0, 0, 0]),
                            ("flat, tilted 60 deg", [0.0, 0.0, 4.0], [0.3, 0.3, 0.001], quat_about_y(60.0))):
        means = torch.tensor([mu], device=dev); scales = torch.tensor([sc], device=dev); quats = torch.tensor([q], device=dev)
        op = torch.tensor([0.9], device=dev); col = torch.zeros(1, 1, device=dev)
        ed, a_cl, _ = rasterization(means, quats, scales, op, col, vm, K, w, h, sh_degree=None, render_mode="ED")
        hit, a_hit, _ = rasterization(means, quats, scales, op, col, vm, K, w, h, sh_degree=None, render_mode="Ed",
                                      with_ut=True, with_eval3d=True, packed=False)
        # analytic: Sigma = R S^2 R'; t* = d' S^-1 mu / d' S^-1 d; euclid = mu_z / d_z
        qq = torch.tensor(q, dtype=torch.float64, device=dev); qw, qx, qy, qz = qq / qq.norm()
        R = torch.tensor([[1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
                          [2 * (qx * qy + qw * qz), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qw * qx)],
                          [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx * qx + qy * qy)]], dtype=torch.float64, device=dev)
        Si = R @ torch.diag(1.0 / torch.tensor(sc, dtype=torch.float64, device=dev) ** 2) @ R.T
        m64 = torch.tensor(mu, dtype=torch.float64, device=dev)
        t_star = (d @ Si @ m64) / torch.einsum("hwi,ij,hwj->hw", d, Si, d)
        t_euc = m64[2] / d[..., 2]
        m = (a_hit[0, ..., 0] > 0.05) & (a_cl[0, ..., 0] > 0.05)
        e_hit = (hit[0, ..., 0].double() - t_star).abs()[m]
        e_euc = (ed[0, ..., 0].double() * sec - t_euc).abs()[m]
        div = (t_star - t_euc)[m]
        r = {"pixels": int(m.sum()), "native_vs_tstar_median_mm": float(e_hit.median()) * 1e3, "native_vs_tstar_p99_mm": float(e_hit.quantile(0.99)) * 1e3,
             "euclid_vs_analytic_median_mm": float(e_euc.median()) * 1e3,
             "analytic_hit_minus_euclid_ns": {"median_abs": float(div.abs().median()) / C, "max_abs": float(div.abs().max()) / C}}
        good = r["native_vs_tstar_median_mm"] <= 1 and r["native_vs_tstar_p99_mm"] <= 5 and r["euclid_vs_analytic_median_mm"] <= 1
        ok &= good
        out[name] = r
        print(f"  S-A {name:20s} {r['pixels']:6d} px: native vs t* median {r['native_vs_tstar_median_mm']:.3f} mm, P99 {r['native_vs_tstar_p99_mm']:.3f} mm; "
              f"euclid vs mu_z/d_z median {r['euclid_vs_analytic_median_mm']:.3f} mm; the two conventions differ by "
              f"{r['analytic_hit_minus_euclid_ns']['median_abs']:.3f} ns median, {r['analytic_hit_minus_euclid_ns']['max_abs']:.3f} max -> {'PASS' if good else 'FAIL'}")
    return out, ok


def stats(e):
    a = np.abs(e)
    return {"median_abs_m": float(np.median(a)), "p90_abs_m": float(np.percentile(a, 90)), "rmse_m": float(np.sqrt((e ** 2).mean())),
            "mean_signed_m": float(e.mean()), "median_abs_ns": float(np.median(a)) / C, "rmse_ns": float(np.sqrt((e ** 2).mean())) / C,
            "mean_signed_ns": float(e.mean()) / C}


def s_b(dev, ckpt, out_dir):
    from gsplat import rasterization
    gs = np.load(os.path.join(out_dir, "depth_gs_lobby.npz")); gt = np.load(os.path.join(out_dir, "depth_gt_lobby.npz"))
    ref = json.load(open(os.path.join(out_dir, "depth_vs_mesh_lobby.json")))
    assert (gs["names"] == gt["names"]).all()
    model = T.RRF(ckpt, "db", 1, 0, dev)
    col = torch.zeros(model.means.shape[0], 1, device=dev)
    n, h, w = gs["depth"].shape
    acc = {k: [] for k in ("z", "euclid", "hit", "div", "gap", "rerender")}
    low = {k: [] for k in ("euclid", "hit")}
    t_cl, t_hit = [], []
    with torch.no_grad():
        for s0 in range(0, n, 4):
            idx = list(range(s0, min(s0 + 4, n)))
            vm = torch.as_tensor(gs["viewmat"][idx], device=dev, dtype=torch.float32)
            Ks = torch.as_tensor(gs["K"][idx], device=dev, dtype=torch.float32)
            torch.cuda.synchronize(); t0 = time.perf_counter()
            ed, _, _ = rasterization(model.means, model.quats, model.scales, model.opacities, col, vm, Ks, w, h, sh_degree=None, render_mode="ED")
            torch.cuda.synchronize(); t1 = time.perf_counter()
            hit, _, _ = rasterization(model.means, model.quats, model.scales, model.opacities, col, vm, Ks, w, h, sh_degree=None, render_mode="Ed",
                                      with_ut=True, with_eval3d=True, packed=False)
            torch.cuda.synchronize(); t2 = time.perf_counter()
            if s0 >= 12:                                   # the first three batches are warm-up
                t_cl.append(t1 - t0); t_hit.append(t2 - t1)
            ed = ed[..., 0].cpu().numpy().astype(np.float64); hit = hit[..., 0].cpu().numpy().astype(np.float64)
            for j, i in enumerate(idx):
                K = gs["K"][i]
                uu, vv = np.meshgrid((np.arange(w) + 0.5 - K[0, 2]) / K[0, 0], (np.arange(h) + 0.5 - K[1, 2]) / K[1, 1])
                sec = np.sqrt(1 + uu ** 2 + vv ** 2)
                z_gt = gt["z_gt"][i].astype(np.float64); al = gs["alpha"][i]
                hitm = np.isfinite(z_gt)
                m = hitm & (al > 0.5); ml = hitm & (al > 0.1) & (al <= 0.5)
                r_mesh = z_gt * sec
                acc["z"].append(ed[j][m] - r_mesh[m]); acc["euclid"].append(ed[j][m] * sec[m] - r_mesh[m]); acc["hit"].append(hit[j][m] - r_mesh[m])
                acc["div"].append((hit[j][m] - ed[j][m] * sec[m]) / C)
                acc["gap"].append((r_mesh[m] - z_gt[m]) / C)
                acc["rerender"].append(ed[j][m] - gs["depth"][i][m])
                low["euclid"].append(ed[j][ml] * sec[ml] - r_mesh[ml]); low["hit"].append(hit[j][ml] - r_mesh[ml])
    cat = {k: np.concatenate(v) for k, v in acc.items()}
    res = {"views": n, "pixels": int(cat["z"].size),
           "vs_mesh_range": {k: stats(cat[k]) for k in ("z", "euclid", "hit")},
           "low_alpha_vs_mesh_range": {k: stats(np.concatenate(v)) for k, v in low.items()},
           "low_alpha_pixels": int(np.concatenate(low["hit"]).size),
           "hit_minus_euclid_ns": {"median_abs": float(np.median(np.abs(cat["div"]))), "p90_abs": float(np.percentile(np.abs(cat["div"]), 90)),
                                   "max_abs": float(np.abs(cat["div"]).max()), "mean_signed": float(cat["div"].mean()),
                                   "share_above_1ns": float((np.abs(cat["div"]) > 1).mean())},
           "control_gap_mean_ns": float(cat["gap"].mean()), "control_gap_ref_ns": ref["range_term_gap_ns"]["mean"],
           "rerender_median_abs_mm": float(np.median(np.abs(cat["rerender"]))) * 1e3,
           "timing_ms_per_4_faces": {"classic_ED": float(np.median(t_cl)) * 1e3, "eval3d_Ed": float(np.median(t_hit)) * 1e3}}
    v = res["vs_mesh_range"]; dv = res["hit_minus_euclid_ns"]
    ok = {"control": abs(res["control_gap_mean_ns"] - res["control_gap_ref_ns"]) <= 0.01,
          "rerender": res["rerender_median_abs_mm"] <= 1.0,
          "known_bad": v["z"]["median_abs_m"] > max(v["euclid"]["median_abs_m"], v["hit"]["median_abs_m"])}
    hyp = v["hit"]["median_abs_m"] <= v["euclid"]["median_abs_m"] and v["hit"]["rmse_m"] <= v["euclid"]["rmse_m"]
    div_in = 0.05 <= dv["median_abs"] <= 1.0 and 0.3 <= dv["p90_abs"] <= 5.0
    print(f"  S-B {n} views, {res['pixels']:,} pixels (mesh hit, alpha > 0.5)")
    print(f"    control: gap mean {res['control_gap_mean_ns']:.3f} ns vs {res['control_gap_ref_ns']:.3f} cached -> {'PASS' if ok['control'] else 'FAIL'}; "
          f"re-rendered ED vs cache median {res['rerender_median_abs_mm']:.4f} mm -> {'PASS' if ok['rerender'] else 'FAIL'}")
    for k in ("z", "euclid", "hit"):
        s = v[k]
        print(f"    {k:6s} vs mesh range: median |e| {s['median_abs_m']:.3f} m ({s['median_abs_ns']:.2f} ns), P90 {s['p90_abs_m']:.3f} m, "
              f"RMSE {s['rmse_m']:.3f} m ({s['rmse_ns']:.2f} ns), signed mean {s['mean_signed_m']:+.3f} m ({s['mean_signed_ns']:+.2f} ns)")
    print(f"    known-bad (z worst on median) -> {'PASS' if ok['known_bad'] else 'FAIL'}; hypothesis (hit <= euclid on median and RMSE) -> "
          f"{'HOLDS' if hyp else 'DOES NOT HOLD'}")
    print(f"    hit - euclid: median |d| {dv['median_abs']:.3f} ns, P90 {dv['p90_abs']:.3f}, max {dv['max_abs']:.2f}, signed mean {dv['mean_signed']:+.3f}, "
          f"> 1 ns on {dv['share_above_1ns']:.1%} -> {'in the expected range' if div_in else 'OUT OF the expected range'}")
    lo = res["low_alpha_vs_mesh_range"]
    print(f"    degenerate (0.1 < alpha <= 0.5, {res['low_alpha_pixels']:,} px): euclid median |e| {lo['euclid']['median_abs_m']:.3f} m, "
          f"hit {lo['hit']['median_abs_m']:.3f} m")
    tm = res["timing_ms_per_4_faces"]
    print(f"    timing per 4 faces at 300 x 200 (median over {len(t_cl)} batches after 3 warm-up): classic ED {tm['classic_ED']:.2f} ms, "
          f"eval3d Ed {tm['eval3d_Ed']:.2f} ms")
    res["pass"] = ok; res["hypothesis_holds"] = hyp; res["divergence_in_expected_range"] = div_in
    return res, all(ok.values())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default=os.path.join(T.REPO, "RF-3DGS_dataset/blender_visual_trained/chkpnt30000.pth"))
    ap.add_argument("--cache", default="output/rrf"); ap.add_argument("--out", default="output/rrf/smoke_hit_distance.json")
    a = ap.parse_args()
    dev = "cuda"
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    apps = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip().splitlines()
    others = [l for l in apps if "python" in l.lower() and str(os.getpid()) not in l]
    print(f"GPU: {gpu}; other python processes on it: {others or 'none'}")
    ra, ok_a = s_a(dev)
    rb, ok_b = s_b(dev, a.checkpoint, a.cache)
    res = {"gpu": gpu, "gpu_other_processes": others, "timing_valid": not others, "S-A": ra, "S-A_pass": ok_a, "S-B": rb, "S-B_pass": ok_b}
    json.dump(res, open(a.out, "w"), indent=1)
    print(f"-> {a.out}; S-A {'PASS' if ok_a else 'FAIL'}, S-B {'PASS' if ok_b else 'FAIL'}" + ("" if not others else "; timing VOID (GPU shared)"))


if __name__ == "__main__":
    main()
