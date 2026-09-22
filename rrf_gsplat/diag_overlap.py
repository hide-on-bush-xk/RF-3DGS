"""Illumination overlap between a source transmitter and Tx-B.

The transfer curve is not a clean function of distance (Tx-E at 3.2 m
carries less than Tx-N at 4.2 m or Tx-A at 5.8 m). A second variable is
whether the two transmitters light the same surfaces: geometry adapted
where the source's paths land can only help where Tx-B's paths land too.
Every dataset shares the route's 800 positions x 4 yaws, so the raw dB
spectra (spectra_float) of a source and of Tx-B can be compared pixel by
pixel on the 640 held-out views:

  in-range IoU   pixels above Tx-A's p1 (-175.75 dB, the transfer's in-range
                 threshold) in both, over pixels above it in either;
  strong IoU     pixels within 30 dB of the view's maximum, same ratio;
  dB correlation Pearson r of the two spectra on the union of in-range pixels.

Joined with the transfer benefit when transfer_curve.json exists.

Comparing the raw float spectra, not the PNGs, is what makes this possible:
the two datasets have different normalisation ranges, so their images are not
pixel-comparable even though the underlying dB values are.

    PYTHONUTF8=1 python rrf_gsplat/diag_overlap.py
"""

from __future__ import annotations

import json
import os

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REG = os.path.join(REPO, "RF-3DGS_dataset/regenerated")
OUT = os.path.join(REPO, "output/rrf")
SOURCES = {"A": "3dgs_MVDR_100_gpct", "C": "3dgs_MVDR_txC_gpct"}
SOURCES.update({n: f"3dgs_MVDR_tx{n}_gpct" for n in "DENLOHMK"})
TARGET = "3dgs_MVDR_txB_gpct"
# Tx-A's 1st percentile, reused verbatim so "in range" means the same thing
# here as it does in the transfer runs being explained.
IN_RANGE_DB = -175.75


def main():
    """Score every source against Tx-B and correlate the result with the benefit."""
    names = [l.strip() for l in open(os.path.join(REG, TARGET, "test_index.txt")) if l.strip()]
    tgt = np.stack([np.load(os.path.join(REG, TARGET, "spectra_float", n + ".npy")) for n in names]).astype(np.float32)
    t_in = tgt > IN_RANGE_DB
    # "strong" is per view (relative to that view's own peak); "in range" is a
    # fixed absolute threshold. The two answer different questions and can disagree.
    t_strong = tgt > (tgt.reshape(len(names), -1).max(1)[:, None, None] - 30.0)
    curve = {}
    # Both transfer budgets if present, so the correlation can be checked at each.
    for tag in ("", "_2k"):
        p = os.path.join(OUT, f"transfer_curve{tag}.json")
        if os.path.exists(p):
            for r in json.load(open(p))["rows"]:
                curve.setdefault(r["source"], {})[tag or "_10k"] = r["benefit_db"]
    rows = []
    for name, ds in SOURCES.items():
        d = os.path.join(REG, ds, "spectra_float")
        # Sources whose dataset is absent, or which do not cover these exact
        # views, are skipped rather than partially scored.
        if not os.path.isdir(d) or not os.path.exists(os.path.join(d, names[0] + ".npy")):
            continue
        src = np.stack([np.load(os.path.join(d, n + ".npy")) for n in names]).astype(np.float32)
        s_in = src > IN_RANGE_DB
        s_strong = src > (src.reshape(len(names), -1).max(1)[:, None, None] - 30.0)
        # max(..., 1) guards an empty union.
        iou_in = float((s_in & t_in).sum() / max((s_in | t_in).sum(), 1))
        iou_strong = float((s_strong & t_strong).sum() / max((s_strong | t_strong).sum(), 1))
        u = s_in | t_in
        r = float(np.corrcoef(src[u], tgt[u])[0, 1]) if u.sum() > 10 else float("nan")
        # The transmitter position comes from the un-normalised dataset's meta:
        # the _gpct sibling is a renormalisation and carries the same tx_loc.
        meta = json.load(open(os.path.join(REG, ds.replace("_gpct", ""), "generation_meta.json")))
        tx = np.array(meta["tx_loc"], float)
        rows.append({"source": name, "tx": tx.round(3).tolist(), "distance_m": float(np.linalg.norm(tx - np.array([8.2, -5.4, 2.0]))),
                     "iou_in_range": iou_in, "iou_strong_30db": iou_strong, "db_corr": r,
                     "share_in_range_src": float(s_in.mean()), "share_in_range_B": float(t_in.mean()),
                     "benefit_10k": curve.get(name, {}).get("_10k"), "benefit_2k": curve.get(name, {}).get("_2k")})
    rows.sort(key=lambda r: r["distance_m"])      # nearest first, so the anomaly is visible
    print(f"{'src':>3} {'d to B':>7} {'IoU in-range':>12} {'IoU strong':>10} {'dB corr':>8} {'benefit 10k':>11} {'benefit 2k':>10}")
    for r in rows:
        b10 = f"{r['benefit_10k']:+.2f}" if r["benefit_10k"] is not None else "--"
        b2 = f"{r['benefit_2k']:+.2f}" if r["benefit_2k"] is not None else "--"
        print(f"{r['source']:>3} {r['distance_m']:7.2f} {r['iou_in_range']:12.3f} {r['iou_strong_30db']:10.3f} {r['db_corr']:8.3f} {b10:>11} {b2:>10}")
    have = [r for r in rows if r["benefit_10k"] is not None]
    if len(have) >= 4:
        # The test the script exists for: does any overlap measure predict the
        # benefit better than raw distance does? distance_m is included as the
        # incumbent explanation to beat.
        for key in ("distance_m", "iou_in_range", "iou_strong_30db", "db_corr"):
            x = np.array([r[key] for r in have]); y = np.array([r["benefit_10k"] for r in have])
            print(f"corr(benefit_10k, {key}) = {np.corrcoef(x, y)[0, 1]:+.2f} over {len(have)} sources")
    json.dump(rows, open(os.path.join(OUT, "diag_overlap.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
