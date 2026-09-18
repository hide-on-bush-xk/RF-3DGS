"""How much of the decoded-angle error is in the target itself?

The MULTI target encodes, per pixel, the power-weighted mean departure angle
of every path splatted into that pixel. That is a convex combination: where
several paths share a pixel the encoded angle is one none of them has. This
measures that loss with no RRF involved, straight from the path table:

  encoded (the dataset's convention)  vs  the strongest path in the pixel
  the tutorial's angle x amplitude RGB, decoded  vs  the same strongest path

for a few held-out positions, on the equirectangular grid (before the pinhole
resampling) and after it, for several splat-kernel widths.

    PYTHONUTF8=1 python rrf_gsplat/diag_encoding_truth.py --positions 6
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO, "sionna_port"))
sys.path.insert(0, os.path.join(REPO, "tx_planning"))


def dominant_maps(theta_r, phi_r, w, theta_t, phi_t, tau, scale=3):
    """Per equirect pixel: the strongest path's AoD and delay (no kernel)."""
    h, wd = 180 * scale, 360 * scale
    ti = (torch.rad2deg(theta_r) * scale).round().long().clamp(0, h - 1)
    pi = ((-torch.rad2deg(phi_r) + 180.0) * scale).round().long().clamp(0, wd - 1)
    flat = ti * wd + pi
    best = torch.full((h * wd,), -1.0, device=w.device)
    best = best.scatter_reduce(0, flat, w, reduce="amax", include_self=True)
    winner = (w >= best[flat] - 1e-30) & (best[flat] > 0)
    az = torch.zeros(h * wd, device=w.device); zen = torch.zeros_like(az); dl = torch.zeros_like(az)
    az[flat[winner]] = torch.rad2deg(phi_t[winner]); zen[flat[winner]] = torch.rad2deg(theta_t[winner])
    dl[flat[winner]] = tau[winner] * 1e9
    hit = best > 0
    return az.reshape(h, wd), zen.reshape(h, wd), dl.reshape(h, wd), hit.reshape(h, wd)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene-xml", default=os.path.join(REPO, "sionna_tutorial/RF-3DGS_Sionna_simulation_tutorial/NIST_lobby_v1.0/NIST_lobby_V1.1_sionna12.xml"))
    ap.add_argument("--poses", default=os.path.join(REPO, "RF-3DGS_dataset/regenerated/3dgs_MULTI_24ghz_tut/sparse/0/images.txt"))
    ap.add_argument("--test-index", default=os.path.join(REPO, "RF-3DGS_dataset/regenerated/3dgs_MULTI_24ghz_tut/test_index.txt"))
    ap.add_argument("--positions", type=int, default=6)
    ap.add_argument("--frequency", type=float, default=2.4e9)
    ap.add_argument("--sigmas", type=float, nargs="+", default=[1.0, 3.0, 6.0])
    ap.add_argument("--out", default=os.path.join(REPO, "output/rrf/diag_encoding_truth.json"))
    cfg = ap.parse_args()

    import mitsuba as mi
    mi.set_variant("cuda_ad_mono_polarized")
    from sionna.rt import PlanarArray, Receiver, Transmitter, load_scene, PathSolver
    import tutorial_materials
    from generate_dataset import read_pose_groups, VIEW_YAWS
    from rf_spectra import (_path_arrays, aod_spectrum_equirect, equirect_splat,
                            equirect_to_perspective, multichannel_spectrum_equirect)

    scene = load_scene(cfg.scene_xml, merge_shapes=True)
    scene.frequency = cfg.frequency
    scene.tx_array = PlanarArray(num_rows=1, num_cols=1, pattern="tr38901", polarization="V")
    scene.rx_array = PlanarArray(num_rows=10, num_cols=10, vertical_spacing=0.5, horizontal_spacing=0.5,
                                 pattern="tr38901", polarization="V")
    tutorial_materials.apply(scene, cfg.frequency, "fixed", verbose=False)
    scene.add(Transmitter(name="tx", position=[6.905, 0.0, 2 - 1.713]))
    solver = PathSolver()

    # held-out positions: first N groups whose images are in the test split
    test = set(l.strip() for l in open(cfg.test_index) if l.strip())
    groups = read_pose_groups(cfg.poses)
    chosen = [(gi, g) for gi, g in enumerate(groups) if f"{gi*4+1:05d}" in test][:cfg.positions]

    rows = {s: {"az_enc": [], "zen_enc": [], "dl_enc": [], "az_aod3": [], "zen_aod3": [],
                "az_enc_persp": [], "zen_enc_persp": []} for s in cfg.sigmas}
    n_hit = 0
    for gi, (rx_loc, yaws) in chosen:
        if "rx" in scene.receivers:
            scene.remove("rx")
        scene.add(Receiver(name="rx", position=list(rx_loc)))
        paths = solver(scene=scene, max_depth=1, samples_per_src=1_000_000, los=True, specular_reflection=True,
                       diffuse_reflection=True, refraction=False, synthetic_array=True, seed=42 + gi)
        amp, tau, theta_r, phi_r, theta_t, phi_t = _path_arrays(paths)
        w = amp * amp
        az_d, zen_d, dl_d, hit = dominant_maps(theta_r, phi_r, w, theta_t, phi_t, tau)
        n_hit += int(hit.sum())
        for s in cfg.sigmas:
            enc = multichannel_spectrum_equirect(paths, scale=3, sigma=s)
            az_e, zen_e, dl_e = enc[1] * 360 - 180, enc[2] * 180, enc[3]
            d_az = ((az_e - az_d + 180) % 360 - 180)[hit]
            rows[s]["az_enc"].append(d_az ** 2); rows[s]["zen_enc"].append(((zen_e - zen_d)[hit]) ** 2)
            rows[s]["dl_enc"].append(((dl_e - dl_d)[hit]) ** 2)
            a3 = aod_spectrum_equirect(paths, scale=3, sigma=s)          # 10log10(amp*x)+150, clipped at 0
            zen3 = torch.clamp(10 ** ((a3[0] - a3[2]) / 10), 0, 1) * 180
            az3 = (1 - torch.clamp(10 ** ((a3[1] - a3[2]) / 10), 0, 1)) * 360 - 180
            rows[s]["az_aod3"].append((((az3 - az_d + 180) % 360 - 180)[hit]) ** 2)
            rows[s]["zen_aod3"].append(((zen3 - zen_d)[hit]) ** 2)
            # after the pinhole resampling of the encoded channels (what the RRF is trained on)
            for yaw in yaws:
                pe = equirect_to_perspective(enc, 300, 200, 90.0, yaw_rad=yaw)
                pd = equirect_to_perspective(torch.stack([az_d, zen_d, hit.float()]), 300, 200, 90.0, yaw_rad=yaw)
                m = pd[2] > 0.999                                       # pixels wholly inside hit area
                az_pe, zen_pe = pe[1] * 360 - 180, pe[2] * 180
                rows[s]["az_enc_persp"].append((((az_pe - pd[0] + 180) % 360 - 180)[m]) ** 2)
                rows[s]["zen_enc_persp"].append(((zen_pe - pd[1])[m]) ** 2)
        print(f"  position {gi}: {int(amp.numel()):,} paths, {int(hit.sum()):,} hit pixels")

    def rmse(lst):
        return float(torch.sqrt(torch.cat(lst).mean()))

    def dist(lst):
        # squared errors -> |error|: median, P90, and the share above 90 deg (a wrap would put mass near 180)
        e = torch.sqrt(torch.cat(lst)).cpu().numpy()
        import numpy as _np
        hist, edges = _np.histogram(e, bins=[0, 2, 5, 10, 20, 45, 90, 135, 181])
        return {"median": float(_np.median(e)), "p90": float(_np.percentile(e, 90)),
                "frac_gt_90": float((e > 90).mean()), "hist": hist.tolist(), "edges": edges.tolist()}

    result = {"positions": [gi for gi, _ in chosen], "hit_pixels": n_hit, "sigmas": {}}
    print(f"\n{len(chosen)} positions, {n_hit:,} equirect pixels with a path")
    print("sigma | encoded mean vs strongest path: az / zen / delay | after pinhole resampling: az / zen | AOD3 decoded: az / zen")
    for s in cfg.sigmas:
        r = {k: rmse(v) for k, v in rows[s].items()}
        result["sigmas"][str(s)] = r
        for key in ("az_enc", "az_enc_persp", "az_aod3"):
            d = dist(rows[s][key]); r[key + "_dist"] = d
            print(f"        {key:13s} |err| median {d['median']:5.2f} deg, P90 {d['p90']:6.2f}, >90 deg: {d['frac_gt_90']*100:.2f}%  "
                  f"hist{d['edges'][:-1]} = {d['hist']}")
        print(f"{s:5.1f} | {r['az_enc']:6.2f} deg / {r['zen_enc']:6.2f} deg / {r['dl_enc']:6.2f} ns | "
              f"{r['az_enc_persp']:6.2f} / {r['zen_enc_persp']:6.2f} deg | {r['az_aod3']:6.2f} / {r['zen_aod3']:6.2f} deg")
    json.dump(result, open(cfg.out, "w"), indent=1)


if __name__ == "__main__":
    main()
