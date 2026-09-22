"""Derive the same-channel-count control for the azimuth A/B.

From a (cos, sin) MULTI dataset, write one with five channels where the
azimuth is the seamed single channel (phi + 180) / 360 twice: the same
capacity, the same data, the only difference being the discontinuity at
+-180 deg. PNGs and sparse/ are hard-linked or copied; the meta carries
the new channel names so evaluate_multi scores the azimuth wrap-aware.

Duplicating the seamed channel is what makes this a fair control: dropping to
four channels would confound the seam with a change in model capacity.

    python rrf_gsplat/make_lin5.py <cs dataset> <out dataset>
"""

from __future__ import annotations

import json
import os
import shutil
import sys

import numpy as np


def main():
    """Rewrite every spectra_float array; share everything else with the source."""
    src, dst = sys.argv[1], sys.argv[2]
    meta = json.load(open(os.path.join(src, "generation_meta.json")))
    names = meta["channels"]
    ic, is_ = names.index("aod_az_cos"), names.index("aod_az_sin")
    os.makedirs(os.path.join(dst, "spectra_float"), exist_ok=True)
    os.makedirs(os.path.join(dst, "images"), exist_ok=True)
    n = 0
    for f in sorted(os.listdir(os.path.join(src, "spectra_float"))):
        if not f.endswith(".npy"):
            continue
        a = np.load(os.path.join(src, "spectra_float", f))
        # Decode the vector azimuth, then re-encode it as a single linear
        # channel -- which reintroduces exactly the seam the (cos, sin) pair
        # was designed to avoid. That seam is the independent variable.
        az = (np.degrees(np.arctan2(a[is_], a[ic])) + 180.0) / 360.0     # [0, 1), seam at 0/1
        hit = a[0] > -150.0
        # Unhit pixels get 0, matching how the source encodes them.
        az = np.where(hit, az, 0.0).astype(np.float32)
        out = np.stack([a[0], az, az, a[names.index("aod_zen")], a[names.index("delay_ns")]])
        np.save(os.path.join(dst, "spectra_float", f), out)
        png = f[:-4] + ".png"
        if not os.path.exists(os.path.join(dst, "images", png)):
            # Hard link where the filesystem allows it, so the derived dataset
            # costs nothing extra; fall back to a copy across volumes.
            try:
                os.link(os.path.join(src, "images", png), os.path.join(dst, "images", png))
            except OSError:
                shutil.copy(os.path.join(src, "images", png), os.path.join(dst, "images", png))
        n += 1
    # Poses and the train/test split are shared verbatim: the control must be
    # scored on exactly the same views as the original.
    if not os.path.exists(os.path.join(dst, "sparse")):
        shutil.copytree(os.path.join(src, "sparse"), os.path.join(dst, "sparse"))
    for idx in ("train_index.txt", "test_index.txt"):
        shutil.copy(os.path.join(src, idx), os.path.join(dst, idx))
    r = meta["channel_ranges"]
    # The channel names are what tell the evaluator to score this azimuth
    # wrap-aware rather than as a plain linear quantity.
    meta["channels"] = ["power_db", "aod_az", "aod_az_dup", "aod_zen", "delay_ns"]
    meta["channel_ranges"] = [r[0], [0.0, 1.0], [0.0, 1.0], r[names.index("aod_zen")], r[names.index("delay_ns")]]
    # Provenance, so a derived dataset can always be traced back.
    meta["derived_from"] = os.path.abspath(src)
    json.dump(meta, open(os.path.join(dst, "generation_meta.json"), "w"), indent=1)
    print(f"wrote {n} views to {dst}: five channels, seamed azimuth twice")


if __name__ == "__main__":
    main()
