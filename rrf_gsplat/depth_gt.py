"""Depth of the reconstructed geometry against the scene mesh (Windows, Sionna env).

Reads depth_gs_<tag>.npz (rrf_gsplat/depth_gs.py), casts the same pixel rays
through the Sionna/Mitsuba scene (first hit) and reports, on pixels where the
Gaussians are opaque (alpha > 0.5) and the mesh is hit:

  depth error      expected depth of the Gaussians minus the mesh's camera-z
                   (RMSE, median |e|, signed mean), in metres and in ns / c
  range-term gap   the delay decomposition adds the *camera z* over c; the
                   path length is the Euclidean range z / cos(theta_pixel).
                   Reported as the mean and P90 of (range - z) / c in ns on the
                   same pixels: the part of the delay residual that is a known
                   per-pixel geometric factor, not a geometry error.

    PYTHONUTF8=1 python rrf_gsplat/depth_gt.py --scene-xml <sionna xml> --tag lobby
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

C = 0.299792458   # m / ns


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene-xml", required=True); ap.add_argument("--tag", required=True); ap.add_argument("--out", default="output/rrf")
    cfg = ap.parse_args()
    import mitsuba as mi
    from sionna.rt import load_scene
    scene = load_scene(cfg.scene_xml)
    ms = scene.mi_scene
    z = np.load(os.path.join(cfg.out, f"depth_gs_{cfg.tag}.npz"))
    depth, alpha, vms, ks = z["depth"], z["alpha"], z["viewmat"], z["K"]
    n, h, w = depth.shape
    errs, gaps, gts, gss = [], [], [], []
    per_view = []
    for i in range(n):
        K = ks[i]; view = vms[i].astype(np.float64)
        u, v = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
        d_cam = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u)], -1)
        d_cam /= np.linalg.norm(d_cam, axis=-1, keepdims=True)                 # unit, camera frame (x right, y down, z forward)
        c2w = np.linalg.inv(view); o = c2w[:3, 3]; d_w = d_cam.reshape(-1, 3) @ c2w[:3, :3].T
        ray = mi.Ray3f(o=mi.Point3f(*[mi.Float(np.full(h * w, o[k])) for k in range(3)]),
                       d=mi.Vector3f(mi.Float(d_w[:, 0].astype(np.float32)), mi.Float(d_w[:, 1].astype(np.float32)), mi.Float(d_w[:, 2].astype(np.float32))))
        si = ms.ray_intersect(ray)
        t = np.array(si.t).reshape(h, w)
        hit = np.isfinite(t) & (t < 1e3)
        z_gt = np.where(hit, t * d_cam[..., 2], np.nan)
        m = hit & (alpha[i] > 0.5)
        e = depth[i][m] - z_gt[m]
        gap = z_gt[m] * (1.0 / d_cam[..., 2][m] - 1.0) / C                   # (range - z) / c, ns
        errs.append(e); gaps.append(gap); gts.append(z_gt[m]); gss.append(depth[i][m])
        per_view.append({"view": int(i), "pixels": int(m.sum()), "rmse_m": float(np.sqrt((e ** 2).mean())) if m.any() else None})
    e = np.concatenate(errs); gap = np.concatenate(gaps); gt = np.concatenate(gts)
    res = {"tag": cfg.tag, "views": n, "pixels": int(e.size), "share_pixels_used": float(e.size / (n * h * w)),
           "gt_depth_median_m": float(np.median(gt)),
           "depth_error": {"rmse_m": float(np.sqrt((e ** 2).mean())), "median_abs_m": float(np.median(np.abs(e))), "p90_abs_m": float(np.percentile(np.abs(e), 90)),
                           "mean_signed_m": float(e.mean()), "rmse_ns": float(np.sqrt((e ** 2).mean()) / C), "median_abs_ns": float(np.median(np.abs(e)) / C)},
           "range_term_gap_ns": {"mean": float(gap.mean()), "median": float(np.median(gap)), "p90": float(np.percentile(gap, 90)), "max": float(gap.max())},
           "per_view": per_view}
    json.dump(res, open(os.path.join(cfg.out, f"depth_vs_mesh_{cfg.tag}.json"), "w"), indent=1)
    d = res["depth_error"]; g = res["range_term_gap_ns"]
    print(f"{cfg.tag}: {n} views, {e.size:,} pixels ({res['share_pixels_used']:.1%}), mesh depth median {res['gt_depth_median_m']:.2f} m")
    print(f"  Gaussian expected depth - mesh: RMSE {d['rmse_m']:.3f} m ({d['rmse_ns']:.2f} ns), median |e| {d['median_abs_m']:.3f} m ({d['median_abs_ns']:.2f} ns), "
          f"P90 |e| {d['p90_abs_m']:.3f} m, signed mean {d['mean_signed_m']:+.3f} m")
    print(f"  range-term gap (range - z)/c: mean {g['mean']:.2f} ns, median {g['median']:.2f}, P90 {g['p90']:.2f}, max {g['max']:.2f} ns")


if __name__ == "__main__":
    main()
