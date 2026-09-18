"""Where did the Gaussians go when the geometry was allowed to train on the spectra?

Compares a run's means with the visual checkpoint's: the displacement
distribution, and whether the Gaussians that moved most ended up on surfaces
(real radio scatterers the visual geometry did not place them on) or in free
space (absorbing the render model's mismatch). Surface distance is the
nearest hit along six axis rays in the Mitsuba scene, before and after.

    PYTHONUTF8=1 python rrf_gsplat/diag_displacement.py --run output/rrf/a_mvdr_db_geom
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO, "tx_planning"))


def surface_distance(scene, pts):
    """Nearest surface along +-x, +-y, +-z from each point, [N]."""
    import mitsuba as mi
    o = mi.Point3f(pts[:, 0].tolist(), pts[:, 1].tolist(), pts[:, 2].tolist())
    best = np.full(len(pts), np.inf)
    for d in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
        si = scene.mi_scene.ray_intersect(mi.Ray3f(o, mi.Vector3f(*map(float, d))))
        best = np.minimum(best, np.asarray(si.t).reshape(-1))
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--checkpoint", default=os.path.join(REPO, "RF-3DGS_dataset/blender_visual_trained/chkpnt30000.pth"))
    ap.add_argument("--scene-xml", default=os.path.join(REPO, "sionna_tutorial/RF-3DGS_Sionna_simulation_tutorial/NIST_lobby_v1.0/NIST_lobby_V1.1_sionna12.xml"))
    ap.add_argument("--subset", type=int, default=20000)
    cfg = ap.parse_args()

    (m, _) = torch.load(cfg.checkpoint, weights_only=False, map_location="cpu")
    means0 = m[1].detach().numpy(); scales0 = np.exp(m[4].detach().numpy()); op0 = 1 / (1 + np.exp(-m[6].detach().numpy().reshape(-1)))
    st = torch.load(os.path.join(cfg.run, "rrf_state.pt"), map_location="cpu")
    means1 = st["means"].numpy(); scales1 = np.exp(st["scales"].numpy()); op1 = 1 / (1 + np.exp(-st["opacities"].numpy().reshape(-1)))
    d = np.linalg.norm(means1 - means0, axis=1)
    print(f"{len(d):,} Gaussians; displacement: median {np.median(d):.4f} m, P90 {np.percentile(d, 90):.3f}, "
          f"P99 {np.percentile(d, 99):.3f}, max {d.max():.2f} m; moved > 0.1 m: {(d > 0.1).mean():.1%}, > 0.5 m: {(d > 0.5).mean():.2%}")
    print(f"scale (mean of 3 axes): before median {np.median(scales0.mean(1)):.4f}, after {np.median(scales1.mean(1)):.4f}; "
          f"opacity mean before {op0.mean():.3f}, after {op1.mean():.3f}; opacity < 0.01: before {(op0 < 0.01).mean():.1%}, after {(op1 < 0.01).mean():.1%}")

    import mitsuba as mi
    mi.set_variant("cuda_ad_mono_polarized")
    from scene_common import load_radio_scene
    scene = load_radio_scene(cfg.scene_xml)
    rng = np.random.default_rng(0)
    top = np.argsort(-d)[:cfg.subset]
    rand = rng.choice(len(d), cfg.subset, replace=False)
    out = {"n": int(len(d)), "displacement_median": float(np.median(d)), "displacement_p90": float(np.percentile(d, 90)),
           "displacement_p99": float(np.percentile(d, 99)), "frac_moved_0p1": float((d > 0.1).mean())}
    for label, idx in (("top-displacement", top), ("random", rand)):
        s0 = surface_distance(scene, means0[idx]); s1 = surface_distance(scene, means1[idx])
        fin = np.isfinite(s0) & np.isfinite(s1)
        closer = (s1 < s0)[fin].mean()
        print(f"{label:17s}: displacement median {np.median(d[idx]):.3f} m | surface distance median before {np.median(s0[fin]):.3f} m, "
              f"after {np.median(s1[fin]):.3f} m | moved closer to a surface: {closer:.1%} | within 0.1 m of a surface: "
              f"before {(s0[fin] < 0.1).mean():.1%}, after {(s1[fin] < 0.1).mean():.1%}")
        out[label] = {"disp_median": float(np.median(d[idx])), "surf_before_median": float(np.median(s0[fin])),
                      "surf_after_median": float(np.median(s1[fin])), "closer_frac": float(closer),
                      "near_before": float((s0[fin] < 0.1).mean()), "near_after": float((s1[fin] < 0.1).mean())}
    json.dump(out, open(os.path.join(cfg.run, "displacement.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
