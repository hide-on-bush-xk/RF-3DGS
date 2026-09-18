"""Collect every output/rrf/*/results.json into one table (markdown + JSON).

    python rrf_gsplat/summarize.py [--out output/rrf/summary.json]
"""

from __future__ import annotations

import argparse
import glob
import json
import os


def time_to(history, key, threshold, higher=True):
    """Seconds until the running eval first crosses `threshold`, or None."""
    for h in history:
        v = h.get(key)
        if v is None:
            continue
        if (v >= threshold) if higher else (v <= threshold):
            return h["seconds"], h["iteration"]
    return None, None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "output", "rrf"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--psnr-target", type=float, default=15.0)
    cfg = ap.parse_args()
    out = cfg.out or os.path.join(cfg.root, "summary.json")

    rows = []
    for path in sorted(glob.glob(os.path.join(cfg.root, "*", "results.json"))):
        r = json.load(open(path))
        c, f = r["config"], r["final"]
        t_psnr, it_psnr = time_to(r["history"], "psnr_rgb", cfg.psnr_target)
        rows.append({
            "run": os.path.basename(os.path.dirname(path)),
            "source": os.path.basename(c["source"].rstrip("/")),
            "mode": c["mode"], "sh_degree": c["sh_degree"],
            "iterations": c["iterations"], "n_train": r["n_train"],
            "opacity": "frozen" if c.get("freeze_opacity") else "trained",
            "warm": os.path.basename(os.path.dirname(c["init_from"])) if c.get("init_from") else "",
            "psnr_rgb": f["psnr_rgb"], "ssim_rgb": f["ssim_rgb"],
            "rmse_db": f["rmse_db"], "mae_db": f["mae_db"],
            "rmse_db_in_range": f.get("rmse_db_in_range"),
            "extra": {k: v for k, v in f.items() if k.startswith("rmse_") and k not in ("rmse_db", "rmse_db_in_range")},
            "geometry": "trained" if c.get("train_geometry") else "frozen",
            "densify": c.get("densify", "none"), "gaussians": r.get("gaussians"),
            "span_db": r["db_range"][1] - r["db_range"][0],
            "it_per_s": r["iters_per_second"], "train_s": r["train_seconds"],
            f"s_to_psnr{cfg.psnr_target:g}": t_psnr, f"it_to_psnr{cfg.psnr_target:g}": it_psnr,
            "history": r["history"],
        })
    json.dump(rows, open(out, "w"), indent=1)

    hdr = ["run", "source", "mode", "sh", "views", "opacity", "warm", "PSNR(jet)", "SSIM",
           "RMSE dB", "RMSE in-range", "MAE dB", "it/s", "train s", f"s→PSNR{cfg.psnr_target:g}"]
    print("| " + " | ".join(hdr) + " |")
    print("|" + "---|" * len(hdr))
    for r in rows:
        t = r[f"s_to_psnr{cfg.psnr_target:g}"]
        ir = r["rmse_db_in_range"]
        print(f"| {r['run']} | {r['source']} | {r['mode']} | {r['sh_degree']} | {r['n_train']} | "
              f"{r['opacity']} | {r['warm']} | {r['psnr_rgb']:.2f} | {r['ssim_rgb']:.3f} | "
              f"{r['rmse_db']:.2f} | {'-' if ir is None else f'{ir:.2f}'} | {r['mae_db']:.2f} | "
              f"{r['it_per_s']:.1f} | {r['train_s']:.0f} | {'-' if t is None else f'{t:.0f}'} |")
    print(f"\n{len(rows)} runs -> {out}")


if __name__ == "__main__":
    main()
