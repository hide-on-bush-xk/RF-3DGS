"""Decoded angle-of-departure and delay error: per-quantity channels against the
tutorial's angle x amplitude RGB encoding.

Both models were trained on the same poses; the truth is the MULTI dataset's
float channels (power_db, aod_az, aod_zen, delay_ns). The AOD3 model's renders
are in the tutorial's encoding, v = 10 log10(amp * x) + 150 per channel with
x = zenith/180 (R), 1 - (azimuth+180)/360 (G) and 1 (B), so an angle decodes
as 10^((v_c - v_B)/10). Errors are taken on pixels the truth's power reaches.

    python rrf_gsplat/eval_encoding.py --multi output/rrf/m_multi_24_tut --aod3 output/rrf/m_aod3_24_tut \
        --truth RF-3DGS_dataset/regenerated/3dgs_MULTI_24ghz_tut
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--multi", required=True, help="run dir of the MULTI model (renders/*.npy = [4,H,W] in native units)")
    ap.add_argument("--aod3", required=True, help="run dir of the AOD3 model (renders/*.npy = [3,H,W] encoded values)")
    ap.add_argument("--truth", required=True, help="the MULTI dataset (spectra_float/*.npy, generation_meta.json)")
    ap.add_argument("--out", default=None)
    cfg = ap.parse_args()

    meta = json.load(open(os.path.join(cfg.truth, "generation_meta.json")))
    lo0, hi0 = meta["channel_ranges"][0]
    names_ch = meta.get("channels", ["power_db", "aod_az", "aod_zen", "delay_ns"])
    pair = "aod_az_cos" in names_ch
    iz, idl = names_ch.index("aod_zen"), names_ch.index("delay_ns")

    def azimuth(arr):
        if pair:
            return np.degrees(np.arctan2(arr[names_ch.index("aod_az_sin")], arr[names_ch.index("aod_az_cos")]))
        return arr[names_ch.index("aod_az")] * 360 - 180
    names = sorted(f[:-4] for f in os.listdir(os.path.join(cfg.multi, "renders")) if f.endswith(".npy"))
    names = [n for n in names if os.path.exists(os.path.join(cfg.aod3, "renders", n + ".npy"))]
    acc = {"multi": {"az": [], "zen": [], "delay": []}, "aod3": {"az": [], "zen": []}}
    n_pix = 0
    for n in names:
        truth = np.load(os.path.join(cfg.truth, "spectra_float", n + ".npy")).astype(np.float64)
        mask = (truth[0] - lo0) / (hi0 - lo0) > 0.02
        if not mask.any():
            continue
        n_pix += int(mask.sum())
        t_az, t_zen, t_delay = azimuth(truth), truth[iz] * 180, truth[idl]

        m = np.load(os.path.join(cfg.multi, "renders", n + ".npy")).astype(np.float64)
        p_az, p_zen, p_delay = azimuth(m), m[iz] * 180, m[idl]
        d_az = (p_az - t_az + 180) % 360 - 180
        acc["multi"]["az"].append(d_az[mask] ** 2)
        acc["multi"]["zen"].append((p_zen - t_zen)[mask] ** 2)
        acc["multi"]["delay"].append((p_delay - t_delay)[mask] ** 2)

        a = np.load(os.path.join(cfg.aod3, "renders", n + ".npy")).astype(np.float64)
        # channels: zen x amp, az x amp, amp, each as 10 log10(.) + 150 (0 where empty)
        with np.errstate(divide="ignore", invalid="ignore"):
            zen_frac = np.clip(10 ** ((a[0] - a[2]) / 10), 0, 1)
            az_frac = np.clip(10 ** ((a[1] - a[2]) / 10), 0, 1)
        q_zen = zen_frac * 180
        q_az = (1 - az_frac) * 360 - 180
        d_az = (q_az - t_az + 180) % 360 - 180
        acc["aod3"]["az"].append(d_az[mask] ** 2)
        acc["aod3"]["zen"].append((q_zen - t_zen)[mask] ** 2)

    def summary(lst):
        e = np.sqrt(np.concatenate(lst))
        return {"rmse": float(np.sqrt((e ** 2).mean())), "median": float(np.median(e)), "p90": float(np.percentile(e, 90)),
                "frac_gt_45": float((e > 45).mean())}

    result = {"views": len(names), "pixels": n_pix,
              "multi": {k: summary(v) for k, v in acc["multi"].items()},
              "aod3": {k: summary(v) for k, v in acc["aod3"].items()}}
    print(f"{len(names)} views, {n_pix:,} pixels with a path  (RMSE / median / P90; share > 45 deg)")
    for q, unit in (("az", "deg"), ("zen", "deg"), ("delay", "ns")):
        mm = result["multi"][q]; line = f"  {q:6s} MULTI {mm['rmse']:6.2f} / {mm['median']:5.2f} / {mm['p90']:6.2f} {unit}"
        if q != "delay":
            line += f"  (>45: {mm['frac_gt_45']*100:.1f}%)"
            aa = result["aod3"][q]; line += f" | AOD3 decoded {aa['rmse']:6.2f} / {aa['median']:5.2f} / {aa['p90']:6.2f} {unit} (>45: {aa['frac_gt_45']*100:.1f}%)"
        print(line)
    out = cfg.out or os.path.join(os.path.dirname(cfg.multi.rstrip("/")), "encoding_comparison.json")
    json.dump(result, open(out, "w"), indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()
