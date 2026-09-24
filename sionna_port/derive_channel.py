"""One channel of a multichannel (MULTI) dataset as a single-channel dataset, so the single-channel pipeline (train_rrf
db mode, mvdr_peaks, t4_score, diag_train_fit) runs on it unchanged. Channel 0 of MULTI is the power splat: every
path's power at its angle of arrival (additive over paths -- what a radiance field models, unlike an MVDR spectrum).

Writes spectra_float/<view>.npy = that channel, sparse/ and the split files copied, generation_meta.json with
spectrum = "<SPECTRUM>_ch<k>"; images/ are then made by rrf_gsplat/renormalize.py (e.g. --norm global-pct).

    python sionna_port/derive_channel.py SRC DST --channel 0
"""

from __future__ import annotations

import argparse
import json
import os
import shutil

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src"); ap.add_argument("dst"); ap.add_argument("--channel", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(os.path.join(a.dst, "spectra_float"), exist_ok=True)
    shutil.copytree(os.path.join(a.src, "sparse"), os.path.join(a.dst, "sparse"), dirs_exist_ok=True)
    for f in ("train_index.txt", "test_index.txt"):
        if os.path.exists(os.path.join(a.src, f)):
            shutil.copy(os.path.join(a.src, f), os.path.join(a.dst, f))
    lo, hi = np.inf, -np.inf
    for f in sorted(os.listdir(os.path.join(a.src, "spectra_float"))):
        x = np.load(os.path.join(a.src, "spectra_float", f))[a.channel].astype(np.float32)
        np.save(os.path.join(a.dst, "spectra_float", f), x)
        lo, hi = min(lo, float(x.min())), max(hi, float(x.max()))
    meta = json.load(open(os.path.join(a.src, "generation_meta.json")))
    meta.update(spectrum=f"{meta.get('spectrum')}_ch{a.channel}", derived_from=os.path.abspath(a.src), channel=a.channel,
                spec_min_db=lo, spec_max_db=hi)
    meta.pop("channel_ranges", None)
    json.dump(meta, open(os.path.join(a.dst, "generation_meta.json"), "w"), indent=1)
    print(f"{a.dst}: channel {a.channel}, range {lo:.1f} .. {hi:.1f}")


if __name__ == "__main__":
    main()
