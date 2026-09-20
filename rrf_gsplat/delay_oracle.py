"""Oracle-depth re-decode of a delay-decomposition run: the ceiling set by the geometry.

The delay channel of a --delay-depth run is (learned residual) + (rendered
range term) in the channel's normalised units. This swaps the rendered range
term for the same term computed from the scene mesh (depth_gt.py) at
evaluation time, pixel by pixel, and re-decodes the delay:

    delay_oracle = delay_pred - f * depth_gs / c + f * z_mesh / c        (the saved renders carry the delay in ns)

with depth_gs the run's own range depth (accumulated "D" or expected "ED")
and f = 1 (camera z) or sec(theta_pixel) (Euclidean range), matching the run.
Pixels the mesh ray misses keep the model's value. The learned residual was
trained against the rendered depth, so this is a partial oracle: what the
geometry alone costs at test time, not what a model trained on true depth
would reach.

    PYTHONUTF8=1 python rrf_gsplat/delay_oracle.py --run output/rrf/m_multi_24_tut_cs_depth_ed_euclid --truth RF-3DGS_dataset/regenerated/3dgs_MULTI_24ghz_tut_cs \
        --depth-gs output/rrf/depth_gs_lobby.npz --depth-gt output/rrf/depth_gt_lobby.npz --depth-mode ED --range euclid
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
C = 0.299792458   # m / ns


def stats(e, w):
    a = np.abs(e); order = np.argsort(a); cw = np.cumsum(w[order])
    return {"median": float(np.median(a)), "p90": float(np.percentile(a, 90)), "rmse": float(np.sqrt((e ** 2).mean())),
            "median_pw": float(a[order][np.searchsorted(cw, 0.5)]), "signed_mean": float(e.mean()), "share_abs_gt_5ns": float((a > 5).mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True); ap.add_argument("--truth", required=True)
    ap.add_argument("--depth-gs", required=True); ap.add_argument("--depth-gt", required=True)
    ap.add_argument("--depth-mode", choices=["D", "ED"], required=True); ap.add_argument("--range", choices=["z", "euclid"], required=True)
    ap.add_argument("--layout", default=None, help="scene layout.json: also per space class of the receiver")
    cfg = ap.parse_args()
    from eval_baselines import decode, read_poses
    meta = json.load(open(os.path.join(cfg.truth, "generation_meta.json")))
    ch = {n: i for i, n in enumerate(meta["channels"])}
    lo0, hi0 = meta["channel_ranges"][0]
    lo_d, hi_d = meta["channel_ranges"][ch["delay_ns"]]; span = hi_d - lo_d
    gs = np.load(cfg.depth_gs); gt = np.load(cfg.depth_gt)
    idx = {str(n): i for i, n in enumerate(gs["names"])}
    assert list(gs["names"]) == list(gt["names"]), "depth_gs and depth_gt must cover the same views"
    depth_key = "depth" if cfg.depth_mode == "ED" else "depth_d"
    K = gs["K"][0]; h, w = gs["depth"].shape[1:]
    if cfg.range == "euclid":
        u = (np.arange(w) + 0.5 - K[0, 2]) / K[0, 0]; v = (np.arange(h) + 0.5 - K[1, 2]) / K[1, 1]
        f = np.sqrt(1.0 + u[None, :] ** 2 + v[:, None] ** 2)
    else:
        f = np.ones((h, w))
    poses = read_poses(os.path.join(cfg.truth, "sparse", "0", "images.txt"))
    spaces = json.load(open(cfg.layout))["spaces"] if cfg.layout else None

    def space_of(rx):
        for s in spaces:
            if s["x0"] - 0.3 <= rx[0] <= s["x1"] + 0.3 and s["y0"] - 0.3 <= rx[1] <= s["y1"] + 0.3:
                return "room" if s["name"].startswith("room") else s["name"]
        c = min(((rx[0] - (s["x0"] + s["x1"]) / 2) ** 2 + (rx[1] - (s["y0"] + s["y1"]) / 2) ** 2, s["name"]) for s in spaces)[1]
        return "room" if c.startswith("room") else c

    test = [l.strip() for l in open(os.path.join(cfg.truth, "test_index.txt")) if l.strip()]
    test = [n for n in test if os.path.exists(os.path.join(cfg.run, "renders", n + ".npy")) and n in idx]
    err = {"model": [], "oracle": []}; weights = []; cls = []; gap_used = []
    for n in test:
        truth = np.load(os.path.join(cfg.truth, "spectra_float", n + ".npy")).astype(np.float64)
        mask = (truth[0] - lo0) / (hi0 - lo0) > 0.02
        if not mask.any():
            continue
        pred = np.load(os.path.join(cfg.run, "renders", n + ".npy")).astype(np.float64)
        i = idx[n]; d_gs = gs[depth_key][i].astype(np.float64); z = gt["z_gt"][i].astype(np.float64)
        # the saved renders are de-normalised: the delay channel is in ns (decode() reads it as such), so the range terms are in ns
        term_gs = f * d_gs / C; term_gt = f * z / C
        swap = np.isfinite(z)
        orc = pred.copy()
        orc[ch["delay_ns"]] = np.where(swap, pred[ch["delay_ns"]] - term_gs + term_gt, pred[ch["delay_ns"]])
        _, _, t_dl = decode(truth, ch); _, _, p_dl = decode(pred, ch); _, _, o_dl = decode(orc, ch)
        err["model"].append((p_dl - t_dl)[mask]); err["oracle"].append((o_dl - t_dl)[mask])
        weights.append(10.0 ** (truth[0][mask] / 10.0)); gap_used.append(float(swap[mask].mean()))
        cls.append(space_of(poses[n][0]) if spaces else "all")
    w = np.concatenate(weights); w = w / w.sum()
    out = {"run": cfg.run, "depth_mode": cfg.depth_mode, "range": cfg.range, "views": len(cls), "pixels": int(w.size),
           "share_pixels_with_mesh_hit": float(np.mean(gap_used)), "all": {k: stats(np.concatenate(v), w) for k, v in err.items()}, "by_space": {}}
    for c in sorted(set(cls)):
        ii = [j for j, k in enumerate(cls) if k == c]
        wc = np.concatenate([weights[j] for j in ii]); wc = wc / wc.sum()
        out["by_space"][c] = {"views": len(ii), **{k: stats(np.concatenate([err[k][j] for j in ii]), wc) for k in err}}
    json.dump(out, open(os.path.join(cfg.run, f"delay_oracle_{cfg.depth_mode}_{cfg.range}.json"), "w"), indent=1)
    def line(tag, s):
        return f"{tag:8s} median {s['median']:.2f}  P90 {s['p90']:.2f}  RMSE {s['rmse']:.2f} ns  pw-median {s['median_pw']:.2f}  signed {s['signed_mean']:+.2f}  |e|>5ns {s['share_abs_gt_5ns']:.1%}"
    print(f"{os.path.basename(cfg.run)} ({cfg.depth_mode}, {cfg.range}): {len(cls)} views, {w.size:,} pixels, mesh hit on {out['share_pixels_with_mesh_hit']:.1%}")
    for k in ("model", "oracle"):
        print("  " + line(k, out["all"][k]))
    for c, g in out["by_space"].items():
        if c != "all":
            print(f"  [{c}] {g['views']} views"); [print("    " + line(k, g[k])) for k in ("model", "oracle")]


if __name__ == "__main__":
    main()
