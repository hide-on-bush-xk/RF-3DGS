"""The fixed validation subset of the live tab's communication curves: its true-channel beam-gain maps, and the
reference lines (the look-ups and the truth itself) computed on exactly those views.

Subset: 15 val_random + 15 val_segment positions spread along each list (both ends included), all four faces:
120 views. Maps: B = a^H R a / |a|^2 in dB per pixel (t4_score.bartlett_maps), R the view's tap covariance from
t6_incoherent_mvdr.py (output/rrf/t4_rt_cov_val_all.npz). References: rrf_gsplat/live_comm.py on the same views,
for every look-up run given (their renders), and for the truth itself.

    python sionna_port/beam_maps.py --truth RF-3DGS_dataset/regenerated/3dgs_APS_60_gp100 \
        --refs NN=output/rrf/s2_aps_nn IDW-2=output/rrf/s2_aps_idw2 IDW-8=output/rrf/s2_aps_idw8 \
        --out output/rrf/beam_maps_val_subset
writes <out>.npz (names, maps) and <out>.json (the subset, the references).
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(REPO, "rrf_gsplat"))


def subset(protocol, per_set=15):
    import protocol as PR
    names = []
    for s in ("val_random", "val_segment"):
        members = PR.read_set(protocol, s)
        pos = sorted({(int(n) - 1) // 4 for n in members})
        pick = [pos[j] for j in np.linspace(0, len(pos) - 1, per_set).round().astype(int)]
        names += [f"{4 * p + k + 1:05d}" for p in pick for k in range(4)]
    return names


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--truth", required=True)
    ap.add_argument("--protocol", default=os.path.join(REPO, "rrf_gsplat", "protocol_v1"))
    ap.add_argument("--cov", default=os.path.join(REPO, "output", "rrf", "t4_rt_cov_val_all.npz"))
    ap.add_argument("--refs", nargs="*", default=[], metavar="LABEL=RUN_DIR")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import live_comm as LC
    from mvdr_peaks import pixel_dirs
    from t4_score import bartlett_maps
    names = subset(a.protocol)
    maps = bartlett_maps(a.cov, a.truth)
    missing = [n for n in names if n not in maps]
    if missing:
        raise SystemExit(f"{len(missing)} subset views have no tap covariance in {a.cov} (e.g. {missing[:3]})")
    dirs = pixel_dirs(a.truth)
    truth = {n: np.load(os.path.join(a.truth, "spectra_float", n + ".npy")) for n in names}
    refs = {"truth itself": LC.summarise([LC.view_metrics(truth[n], truth[n], maps[n], dirs) for n in names])}
    for spec in a.refs:
        label, run = spec.split("=", 1)
        rows = [LC.view_metrics(np.load(os.path.join(run, "renders", n + ".npy")), truth[n], maps[n], dirs) for n in names]
        refs[label] = LC.summarise(rows)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.savez_compressed(a.out + ".npz", names=np.array(names), maps=np.stack([maps[n] for n in names]).astype(np.float32))
    json.dump({"truth": a.truth, "cov": a.cov, "names": names, "views": len(names), "references": refs},
              open(a.out + ".json", "w"), indent=1)
    for k, v in refs.items():
        print(f"{k:14s} " + ("MISSING" if v is None else
              f"angle median {v['angle_median']:.2f} deg, <= 1 deg {100 * v['within_1deg']:.1f} % "
              f"(distinct {100 * v['distinct_within_1deg']:.1f} % of {v['distinct_views']}), beam loss "
              f"{v['beam_loss_median']:.2f} dB (P90 {v['beam_loss_p90']:.2f}), top-3 {100 * v['top3_detected']:.1f} %"))


if __name__ == "__main__":
    main()
