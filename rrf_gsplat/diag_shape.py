"""What changed in the Gaussians' shape when the geometry trained on Tx-A's spectra?

diag_displacement.py showed the positions barely moved; the gain came from
scales and rotations. Two readings of that, with opposite consequences:

  (a) geometric correction: Gaussians flatten with the short axis along the
      surface normal (towards 2DGS), independent of the transmitter;
  (b) appearance compensation: ellipsoids stretch or turn in a way that
      knows where the transmitter is, to shape their projected footprint.

Three measurements on the checkpoint against a run's rrf_state.pt:
  1. anisotropy s_max / s_min, before and after;
  2. the angle between the shortest axis and the surface normal of the
     nearest surface (six axis rays into the Mitsuba scene), before and after;
  3. whether the change in extent along the direction to Tx-A is special:
     compared with the same quantity along the direction to Tx-B and along
     random directions, and the alignment of the rotation axis with those
     directions. A pure geometric correction cannot know where Tx-A is.

    PYTHONUTF8=1 python rrf_gsplat/diag_shape.py --run output/rrf/a_mvdr_db_geom
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


def quat_to_R(q):
    """[N,4] (w,x,y,z), normalised -> [N,3,3] with columns = the Gaussian's axes."""
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q.T
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
        np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
        np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1)], -2)


def extent_along(R, s, u):
    """sqrt(u^T Sigma u) with Sigma = R diag(s^2) R^T; u [N,3] unit."""
    a = np.einsum("nij,nj->ni", np.transpose(R, (0, 2, 1)), u)     # u in the Gaussian frame
    return np.sqrt(((a * s) ** 2).sum(1))


def nearest_normal(scene, pts):
    import mitsuba as mi
    o = mi.Point3f(pts[:, 0].tolist(), pts[:, 1].tolist(), pts[:, 2].tolist())
    best_t = np.full(len(pts), np.inf); best_n = np.zeros((len(pts), 3))
    for d in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
        si = scene.mi_scene.ray_intersect(mi.Ray3f(o, mi.Vector3f(*map(float, d))))
        t = np.asarray(si.t).reshape(-1)
        n = np.stack([np.asarray(si.n.x).reshape(-1), np.asarray(si.n.y).reshape(-1), np.asarray(si.n.z).reshape(-1)], 1)
        closer = t < best_t
        best_t[closer] = t[closer]; best_n[closer] = n[closer]
    return best_t, best_n


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--checkpoint", default=os.path.join(REPO, "RF-3DGS_dataset/blender_visual_trained/chkpnt30000.pth"))
    ap.add_argument("--scene-xml", default=os.path.join(REPO, "sionna_tutorial/RF-3DGS_Sionna_simulation_tutorial/NIST_lobby_v1.0/NIST_lobby_V1.1_sionna12.xml"))
    ap.add_argument("--tx-a", type=float, nargs=3, default=[6.905, 0.0, 0.287])
    ap.add_argument("--tx-b", type=float, nargs=3, default=[8.2, -5.4, 2.0])
    ap.add_argument("--subset", type=int, default=60000)
    cfg = ap.parse_args()
    rng = np.random.default_rng(0)

    (m, _) = torch.load(cfg.checkpoint, weights_only=False, map_location="cpu")
    mu0, ls0, q0 = m[1].detach().numpy(), m[4].detach().numpy(), m[5].detach().numpy()
    st = torch.load(os.path.join(cfg.run, "rrf_state.pt"), map_location="cpu")
    mu1, ls1, q1 = st["means"].numpy(), st["scales"].numpy(), st["quats"].numpy()
    op1 = 1 / (1 + np.exp(-st["opacities"].numpy().reshape(-1)))
    alive = op1 > 0.01                           # Gaussians the trained field still uses
    idx = rng.choice(np.flatnonzero(alive), min(cfg.subset, int(alive.sum())), replace=False)
    s0, s1 = np.exp(ls0[idx]), np.exp(ls1[idx]); R0, R1 = quat_to_R(q0[idx]), quat_to_R(q1[idx]); mu = mu1[idx]
    out = {"n": int(len(idx)), "alive_frac": float(alive.mean())}
    print(f"{len(idx):,} Gaussians sampled from the {alive.mean():.0%} with opacity > 0.01")

    # 1. anisotropy -- and whether these are discs or needles. s_max/s_min alone
    # cannot tell: a disc has s_max ~ s_mid >> s_min (its short axis is a real
    # direction, the normal); a needle has s_max >> s_mid ~ s_min (its short
    # axis is degenerate in the plane normal to the long axis, so its direction
    # is numerical noise). s_mid/s_min separates them.
    an0, an1 = s0.max(1) / s0.min(1), s1.max(1) / s1.min(1)
    srt0, srt1 = np.sort(s0, 1), np.sort(s1, 1)               # ascending: min, mid, max
    mid0, mid1 = srt0[:, 1] / srt0[:, 0], srt1[:, 1] / srt1[:, 0]
    top0, top1 = srt0[:, 2] / srt0[:, 1], srt1[:, 2] / srt1[:, 1]
    disc0, disc1 = (mid0 > 3) & (top0 < 3), (mid1 > 3) & (top1 < 3)
    needle0, needle1 = (top0 > 3) & (mid0 < 3), (top1 > 3) & (mid1 < 3)
    print(f"1. anisotropy s_max/s_min: before median {np.median(an0):.2f} (P90 {np.percentile(an0, 90):.1f}), "
          f"after median {np.median(an1):.2f} (P90 {np.percentile(an1, 90):.1f}); ratio after/before median {np.median(an1/an0):.3f}, "
          f"flattened (ratio > 1.2): {(an1/an0 > 1.2).mean():.1%}, rounded (< 0.83): {(an1/an0 < 0.83).mean():.1%}")
    print(f"   s_mid/s_min: before median {np.median(mid0):.2f} (P10 {np.percentile(mid0, 10):.2f}, P90 {np.percentile(mid0, 90):.1f}), after {np.median(mid1):.2f}; "
          f"s_max/s_mid: before median {np.median(top0):.2f}, after {np.median(top1):.2f}")
    print(f"   discs (s_mid/s_min > 3, s_max/s_mid < 3): before {disc0.mean():.1%}, after {disc1.mean():.1%} | "
          f"needles (s_max/s_mid > 3, s_mid/s_min < 3): before {needle0.mean():.1%}, after {needle1.mean():.1%}")
    out["anisotropy"] = {"before_median": float(np.median(an0)), "after_median": float(np.median(an1)),
                         "flattened_frac": float((an1 / an0 > 1.2).mean()), "rounded_frac": float((an1 / an0 < 0.83).mean()),
                         "mid_over_min_before": float(np.median(mid0)), "mid_over_min_after": float(np.median(mid1)),
                         "disc_frac_before": float(disc0.mean()), "disc_frac_after": float(disc1.mean()),
                         "needle_frac_before": float(needle0.mean()), "needle_frac_after": float(needle1.mean())}

    # 2. shortest axis against the surface normal
    import mitsuba as mi
    mi.set_variant("cuda_ad_mono_polarized")
    from scene_common import load_radio_scene
    scene = load_radio_scene(cfg.scene_xml)
    t, n = nearest_normal(scene, mu)
    near = np.isfinite(t) & (t < 0.2)
    def short_axis_angle(R, s):
        k = s.argmin(1)
        ax = R[np.arange(len(R)), :, k]                       # column k
        c = np.abs((ax * n).sum(1)) / (np.linalg.norm(ax, axis=1) * np.linalg.norm(n, axis=1) + 1e-12)
        return np.degrees(np.arccos(np.clip(c, 0, 1)))
    a0, a1 = short_axis_angle(R0, s0)[near], short_axis_angle(R1, s1)[near]
    print(f"2. shortest axis vs surface normal ({near.sum():,} Gaussians within 0.2 m of a surface): "
          f"before median {np.median(a0):.1f} deg (within 20 deg: {(a0 < 20).mean():.1%}), after median {np.median(a1):.1f} deg "
          f"(within 20 deg: {(a1 < 20).mean():.1%}); random axes would give median 60 deg")
    # the test that is well defined for a needle: does the long axis lie in the
    # tangent plane (90 deg from the normal)?
    def long_axis_angle(R, s):
        k = s.argmax(1)
        ax = R[np.arange(len(R)), :, k]
        c = np.abs((ax * n).sum(1)) / (np.linalg.norm(ax, axis=1) * np.linalg.norm(n, axis=1) + 1e-12)
        return np.degrees(np.arccos(np.clip(c, 0, 1)))
    l0, l1 = long_axis_angle(R0, s0)[near], long_axis_angle(R1, s1)[near]
    dn, dn1 = disc0[near], disc1[near]
    nd, nd1 = needle0[near], needle1[near]
    print(f"   longest axis vs normal: before median {np.median(l0):.1f} deg (within 20 deg of the tangent plane, i.e. > 70: {(l0 > 70).mean():.1%}), "
          f"after median {np.median(l1):.1f} deg (> 70: {(l1 > 70).mean():.1%})")
    if dn.sum() > 100:
        print(f"   discs only ({dn.sum():,}): short axis vs normal before median {np.median(a0[dn]):.1f}, after (same Gaussians) {np.median(a1[dn]):.1f} deg")
    if nd.sum() > 100:
        print(f"   needles only ({nd.sum():,}): long axis vs normal before median {np.median(l0[nd]):.1f}, after {np.median(l1[nd]):.1f} deg")
    out["normal_alignment"] = {"before_median_deg": float(np.median(a0)), "after_median_deg": float(np.median(a1)),
                               "before_within20": float((a0 < 20).mean()), "after_within20": float((a1 < 20).mean()),
                               "long_axis_before_median": float(np.median(l0)), "long_axis_after_median": float(np.median(l1)),
                               "disc_short_before": float(np.median(a0[dn])) if dn.sum() else None,
                               "disc_short_after": float(np.median(a1[dn])) if dn.sum() else None,
                               "needle_long_before": float(np.median(l0[nd])) if nd.sum() else None,
                               "needle_long_after": float(np.median(l1[nd])) if nd.sum() else None}

    # 3. does the change know where Tx-A is?
    dA = np.array(cfg.tx_a) - mu; dA /= np.linalg.norm(dA, axis=1, keepdims=True)
    dB = np.array(cfg.tx_b) - mu; dB /= np.linalg.norm(dB, axis=1, keepdims=True)
    dR = rng.normal(size=mu.shape); dR /= np.linalg.norm(dR, axis=1, keepdims=True)
    rows = {}
    for name, u in (("to Tx-A", dA), ("to Tx-B", dB), ("random", dR)):
        e0, e1 = extent_along(R0, s0, u), extent_along(R1, s1, u)
        lr = np.log(e1 / e0)
        rows[name] = {"mean_log_ratio": float(lr.mean()), "std": float(lr.std()), "frac_shrunk": float((lr < -0.1).mean()),
                      "frac_grown": float((lr > 0.1).mean())}
        print(f"3. extent along {name:7s}: mean log(after/before) {lr.mean():+.4f} (std {lr.std():.3f}); "
              f"shrunk > 10%: {(lr < -0.1).mean():.1%}, grown > 10%: {(lr > 0.1).mean():.1%}")
    lrA = np.log(extent_along(R1, s1, dA) / extent_along(R0, s0, dA))
    lrB = np.log(extent_along(R1, s1, dB) / extent_along(R0, s0, dB))
    lrR = np.log(extent_along(R1, s1, dR) / extent_along(R0, s0, dR))
    print(f"   correlation of per-Gaussian extent change: along Tx-A vs Tx-B {np.corrcoef(lrA, lrB)[0,1]:.3f}, "
          f"Tx-A vs random {np.corrcoef(lrA, lrR)[0,1]:.3f}")
    # rotation axis of the change against the directions
    dRot = np.einsum("nij,nkj->nik", R1, R0)                 # R1 R0^T
    tr = np.clip((np.trace(dRot, axis1=1, axis2=2) - 1) / 2, -1, 1)
    ang = np.degrees(np.arccos(tr))
    axis = np.stack([dRot[:, 2, 1] - dRot[:, 1, 2], dRot[:, 0, 2] - dRot[:, 2, 0], dRot[:, 1, 0] - dRot[:, 0, 1]], 1)
    turned = ang > 5
    axis = axis[turned] / (np.linalg.norm(axis[turned], axis=1, keepdims=True) + 1e-12)
    cosA, cosB, cosR = (np.abs((axis * d[turned]).sum(1)) for d in (dA, dB, dR))
    print(f"   rotation: median {np.median(ang):.2f} deg, turned > 5 deg: {turned.mean():.1%}; among those, |cos(axis, dir)| mean "
          f"to Tx-A {cosA.mean():.3f}, to Tx-B {cosB.mean():.3f}, random {cosR.mean():.3f} (isotropic: 0.500)")
    out["tx_direction"] = {"extent": rows, "corr_A_B": float(np.corrcoef(lrA, lrB)[0, 1]), "corr_A_random": float(np.corrcoef(lrA, lrR)[0, 1]),
                           "rotation_median_deg": float(np.median(ang)), "turned_frac": float(turned.mean()),
                           "axis_cos_A": float(cosA.mean()), "axis_cos_B": float(cosB.mean()), "axis_cos_random": float(cosR.mean())}
    json.dump(out, open(os.path.join(cfg.run, "shape.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
