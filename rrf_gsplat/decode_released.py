"""Track D: our decoded-angle metric applied to the released RF-3DGS AoD / Delay models.

The released AoD picture encodes, per pixel, B = 10 log10(sum amp), R = 10 log10(sum amp * theta_t/180),
G = 10 log10(sum amp * (1 - (phi_t + 180)/360)) (tutorial plot_AoD_spatial_spectrum; uint8 dB values), so the
power-weighted mean departure angles decode as theta = 180 * 10^((R - B)/10) and phi = (1 - 10^((G - B)/10)) * 360 - 180.
The Delay picture has R = 10 log10(sum amp * delay_norm) + 150, G = B = 10 log10(sum amp) + 150, so the per-view
normalised delay decodes as 10^((R - B)/10). Both the released model's renders and the released ground truth are
decoded with the same rule on the pixels where the ground truth has power (B > 0), and the difference is the
model's angle / delay fidelity to its own target, in degrees and in normalised delay units. The same fidelity of
our multi-channel field to its target is Table 6's row (0.59 / 0.94 deg, 2.42 ns): the coordinate is shared, the
data are not identical (released data at 2.4 GHz with the tutorial's materials and per-view sampling).

Note what is and is not being measured: this is the released model against its
OWN target, not against ours. It says how faithfully the model reproduces what
it was trained on, which is the only comparison the two datasets support.

    PYTHONUTF8=1 python rrf_gsplat/decode_released.py
"""

from __future__ import annotations

import json
import os

import numpy as np
from PIL import Image

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ROOT = os.path.join(REPO, "RF-3DGS_dataset", "RF-3DGS_trained_RRF")


def decode_aod(img):
    """RGB -> (departure zenith deg, departure azimuth deg, lit mask).

    Inverts the ratio encoding: R - B and G - B cancel the amplitude, leaving
    the normalised angle. The clips guard pixels where quantisation pushed the
    ratio outside [0, 1]. B > 0 is the lit mask: an unlit pixel encodes 0 dB.
    """
    r, g, b = (img[..., c].astype(np.float64) for c in range(3))
    theta = 180.0 * np.clip(10 ** ((r - b) / 10.0), 0, 1)
    phi = (1.0 - np.clip(10 ** ((g - b) / 10.0), 0, 1)) * 360.0 - 180.0
    return theta, phi, b > 0


def decode_delay(img):
    """RGB -> (normalised delay in [0, 1], lit mask). Same ratio trick as above.

    The unit is the view's own delay range, not nanoseconds: the encoding
    normalised per view, so the absolute scale is not recoverable.
    """
    r, b = img[..., 0].astype(np.float64), img[..., 2].astype(np.float64)
    return np.clip(10 ** ((r - b) / 10.0), 0, 1), b > 0


def stats(e):
    """Median, P90 and RMSE of an error array, plus its pixel count."""
    return {"median": float(np.median(e)), "p90": float(np.percentile(e, 90)), "rmse": float(np.sqrt((e ** 2).mean())), "n": int(e.size)}


def main():
    """Decode both released models against their own ground truth and report."""
    out = {}
    for kind in ("AoD", "Delay"):
        d = os.path.join(ROOT, f"3dgs_{kind}_100", "test", "ours_40000")
        names = sorted(f for f in os.listdir(os.path.join(d, "gt")) if f.endswith(".png"))
        err = {"theta": [], "phi": [], "delay": []}
        for n in names:
            gt = np.array(Image.open(os.path.join(d, "gt", n)).convert("RGB")); pr = np.array(Image.open(os.path.join(d, "renders", n)).convert("RGB"))
            if kind == "AoD":
                # The ground truth's mask is used for both, so the model is
                # scored only where there is something to predict.
                t_gt, p_gt, m = decode_aod(gt); t_pr, p_pr, _ = decode_aod(pr)
                # Azimuth wraps: the modulo turns a 359-degree difference into 1.
                err["theta"].append(np.abs(t_pr - t_gt)[m]); err["phi"].append(np.abs((p_pr - p_gt + 180) % 360 - 180)[m])
            else:
                d_gt, m = decode_delay(gt); d_pr, _ = decode_delay(pr)
                err["delay"].append(np.abs(d_pr - d_gt)[m])
        res = {k: stats(np.concatenate(v)) for k, v in err.items() if v}
        res["views"] = len(names); out[kind] = res
        for k, s in res.items():
            if isinstance(s, dict):          # skips the "views" count
                unit = "deg" if k != "delay" else "of the per-view delay range"
                print(f"released {kind} model vs its ground truth, {k}: median {s['median']:.2f}  P90 {s['p90']:.2f}  RMSE {s['rmse']:.2f} {unit}  ({s['n']:,} lit pixels, {len(names)} views)")
    # the encoding's own resolution: one uint8 dB step in R - B moves theta by 26 % of its value
    # -- so an error below that is unresolvable and should not be read as accuracy.
    json.dump(out, open(os.path.join(REPO, "output", "rrf", "decode_released.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
