"""Signed delay error of a multi-channel run on its saved held-out renders.

Reproduces the signed columns of the delay-decomposition table: mean and
median signed error (prediction - truth, ns), power-weighted mean, share of
pixels with |e| > 5 ns and the share of those that are negative (predicted too
short).

The sign is the whole point: an unsigned |error| cannot distinguish a model
that is symmetrically noisy from one that is systematically early, and the
latter is what a geometry sitting in front of the true surface produces.

    PYTHONUTF8=1 python rrf_gsplat/delay_signed.py --run output/rrf/<run> --truth RF-3DGS_dataset/regenerated/<ds>
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    """Pool the signed per-pixel delay error over every held-out view."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True); ap.add_argument("--truth", required=True); ap.add_argument("--tail-ns", type=float, default=5.0)
    cfg = ap.parse_args()
    from eval_baselines import decode
    meta = json.load(open(os.path.join(cfg.truth, "generation_meta.json")))
    # Channel order comes from the dataset's own meta, not from a constant here.
    ch = {n: i for i, n in enumerate(meta["channels"])}
    lo0, hi0 = meta["channel_ranges"][0]
    test = [l.strip() for l in open(os.path.join(cfg.truth, "test_index.txt")) if l.strip()]
    # Views the run did not save a render for are skipped rather than failing.
    test = [n for n in test if os.path.exists(os.path.join(cfg.run, "renders", n + ".npy"))]
    e_all, w_all = [], []
    for n in test:
        truth = np.load(os.path.join(cfg.truth, "spectra_float", n + ".npy")).astype(np.float64)
        # Only pixels above 2 % of the normalised power range count: delay is
        # meaningless where no path arrived, and including those pixels would
        # swamp the statistics with the floor value.
        mask = (truth[0] - lo0) / (hi0 - lo0) > 0.02
        if not mask.any():
            continue                     # an empty view
        pred = np.load(os.path.join(cfg.run, "renders", n + ".npy")).astype(np.float64)
        _, _, t_dl = decode(truth, ch); _, _, p_dl = decode(pred, ch)
        # Weights are linear power, so the power-weighted mean below is
        # dominated by the pixels that carry the signal.
        e_all.append((p_dl - t_dl)[mask]); w_all.append(10.0 ** (truth[0][mask] / 10.0))
    e = np.concatenate(e_all); w = np.concatenate(w_all); w = w / w.sum()
    tail = np.abs(e) > cfg.tail_ns
    # share_negative_in_tail is the decisive number: near 0.5 means a symmetric
    # tail, well above it means a systematic bias towards predicting too short.
    res = {"run": cfg.run, "views": len(test), "pixels": int(e.size), "mean_ns": float(e.mean()), "median_ns": float(np.median(e)),
           "mean_pw_ns": float((w * e).sum()), "share_positive": float((e > 0).mean()),
           f"share_abs_gt_{cfg.tail_ns:g}ns": float(tail.mean()), "share_negative_in_tail": float((e[tail] < 0).mean()) if tail.any() else None,
           "abs_median_ns": float(np.median(np.abs(e))), "abs_p90_ns": float(np.percentile(np.abs(e), 90)), "rmse_ns": float(np.sqrt((e ** 2).mean()))}
    json.dump(res, open(os.path.join(cfg.run, "delay_signed.json"), "w"), indent=1)
    print(f"{os.path.basename(cfg.run)}: {len(test)} views, {e.size:,} pixels; signed mean {res['mean_ns']:+.2f} ns, median {res['median_ns']:+.2f}, "
          f"power-weighted mean {res['mean_pw_ns']:+.2f}; |e| > {cfg.tail_ns:g} ns on {res[f'share_abs_gt_{cfg.tail_ns:g}ns']:.1%} of pixels "
          f"({res['share_negative_in_tail']:.0%} negative); |e| median {res['abs_median_ns']:.2f}, P90 {res['abs_p90_ns']:.2f}, RMSE {res['rmse_ns']:.2f} ns")


if __name__ == "__main__":
    main()
