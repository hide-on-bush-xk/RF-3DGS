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
        t_az, t_zen, t_delay = truth[1] * 360 - 180, truth[2] * 180, truth[3]

        m = np.load(os.path.join(cfg.multi, "renders", n + ".npy")).astype(np.float64)
        p_az, p_zen, p_delay = m[1] * 360 - 180, m[2] * 180, m[3]
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

    def rmse(lst):
        return float(np.sqrt(np.concatenate(lst).mean())) if lst else float("nan")

    result = {"views": len(names), "pixels": n_pix,
              "multi": {k: rmse(v) for k, v in acc["multi"].items()},
              "aod3": {k: rmse(v) for k, v in acc["aod3"].items()}}
    print(f"{len(names)} views, {n_pix:,} pixels with a path")
    print(f"  AoD azimuth RMSE : MULTI {result['multi']['az']:6.2f} deg | AOD3 (angle x amp, decoded) {result['aod3']['az']:6.2f} deg")
    print(f"  AoD zenith  RMSE : MULTI {result['multi']['zen']:6.2f} deg | AOD3 {result['aod3']['zen']:6.2f} deg")
    print(f"  delay RMSE       : MULTI {result['multi']['delay']:6.2f} ns  | AOD3 has no delay channel")
    out = cfg.out or os.path.join(os.path.dirname(cfg.multi.rstrip("/")), "encoding_comparison.json")
    json.dump(result, open(out, "w"), indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()
