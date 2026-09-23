"""How much of a shading-head model's output is the head? Measured where there is signal, not on the floor.

train_rrf.py's head_residual_rms_db pools every pixel, and on the floor (targets clamped to the bottom of the
range) the loss lets the head push down freely, so that number sits at the bound and says little. This compares
a run's saved renders (base + head) with the same Gaussians rendered without the head (a 0-step re-render,
--head none) on the pixels whose truth is inside the dataset's range, and separately on each view's peak region
(truth within 10 dB of its maximum):

  rms_db      RMS of (with head - without head), dB
  var_share   that residual's variance over the variance of the output, same pixels
  err_*       the output's and the base's mean |error| against the truth on those pixels
  at_bound    share of those pixels where the residual is within 1 % of the bound (--bound-db): about 1 for a
              head that has saturated and stopped learning (round 41: 0.999 dead, 0.07-0.27 alive)

    python rrf_gsplat/head_share.py --run output/rrf/<head run> --base output/rrf/<its --head none re-render> --truth <dataset>
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True); ap.add_argument("--base", required=True); ap.add_argument("--truth", required=True)
    ap.add_argument("--bound-db", type=float, default=6.0, help="the head's residual bound (train_rrf --head-max-db)")
    a = ap.parse_args()
    meta = json.load(open(os.path.join(a.truth, "generation_meta.json")))
    lo, hi = meta["spec_min_db"], meta["spec_max_db"]
    test = [l.strip() for l in open(os.path.join(a.truth, "test_index.txt")) if l.strip()]
    acc = {k: [] for k in ("res_sig", "out_sig", "eo_sig", "eb_sig", "res_pk", "out_pk", "eo_pk", "eb_pk")}
    for n in test:
        t = np.load(os.path.join(a.truth, "spectra_float", n + ".npy")).astype(np.float64)
        o = np.load(os.path.join(a.run, "renders", n + ".npy")).astype(np.float64)
        b = np.load(os.path.join(a.base, "renders", n + ".npy")).astype(np.float64)
        sig = (t >= lo) & (t <= hi)
        pk = t >= t.max() - 10.0
        for m, tag in ((sig, "sig"), (pk, "pk")):
            if not m.any():
                continue
            acc[f"res_{tag}"].append(o[m] - b[m]); acc[f"out_{tag}"].append(o[m] - o[m].mean())
            acc[f"eo_{tag}"].append(np.abs(o[m] - t[m])); acc[f"eb_{tag}"].append(np.abs(b[m] - t[m]))
    out = {}
    for tag in ("sig", "pk"):
        r = np.concatenate(acc[f"res_{tag}"]); v = np.concatenate(acc[f"out_{tag}"])
        out[tag] = {"pixels": int(r.size), "rms_db": float(np.sqrt((r ** 2).mean())),
                    "var_share": float((r ** 2).sum() / max((v ** 2).sum(), 1e-12)),
                    "at_bound": float((np.abs(r) > 0.99 * a.bound_db).mean()),
                    "err_with_head_db": float(np.concatenate(acc[f"eo_{tag}"]).mean()),
                    "err_without_head_db": float(np.concatenate(acc[f"eb_{tag}"]).mean())}
    name = os.path.basename(os.path.abspath(a.run))
    json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(a.run)), f"headshare_{name}.json"), "w"), indent=1)
    s, p = out["sig"], out["pk"]
    print(f"{name}: in-range pixels: head residual RMS {s['rms_db']:.2f} dB, {100 * s['var_share']:.0f}% of the output variance, "
          f"{100 * s['at_bound']:.1f}% at the bound, "
          f"|error| {s['err_without_head_db']:.2f} -> {s['err_with_head_db']:.2f} dB | peak regions: RMS {p['rms_db']:.2f} dB, "
          f"|error| {p['err_without_head_db']:.2f} -> {p['err_with_head_db']:.2f} dB")


if __name__ == "__main__":
    main()
