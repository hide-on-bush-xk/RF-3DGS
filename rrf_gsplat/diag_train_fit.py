"""Does a trained radio radiance field reproduce its own training views at the peaks? (underfit vs generalisation)

T4 (round 47) found the nearest training position's truth beats the trained field on held-out main-peak direction
(1.3 vs 5.6 deg median, <= 1 deg 40 vs 20 %). Two readings: the field fits its training views well and fails to
interpolate between them (generalisation), or it does not fit them at the peaks in the first place (underfit --
capacity, schedule, or label noise it cannot represent). This renders a run on a stratified set of its training
views (whole positions, every face, spread along the list) and scores them exactly like the held-out views.

    python rrf_gsplat/diag_train_fit.py --run output/rrf/r45_base --positions 160
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import train_rrf as T  # noqa: E402
from mvdr_peaks import angle, local_maxima, pixel_dirs, top_peaks  # noqa: E402


def peak_metrics(names, pred_dir, truth, dirs):
    flat = dirs.reshape(-1, 3)
    ang, at, det, distinct = [], [], [], []
    for n in names:
        t = np.load(os.path.join(truth, "spectra_float", n + ".npy")).astype(np.float64)
        if t.max() < -250 or t.max() - t.min() < 1e-3:   # no paths (MVDR: EMPTY_VIEW_DB; APS: flat at N0)
            continue
        p = np.load(os.path.join(pred_dir, n + ".npy")).astype(np.float64)
        kt, kp = int(t.argmax()), int(p.argmax())
        ang.append(angle(dirs, kt, kp)); at.append(p.flat[kt] - t.flat[kt])
        pm = np.flatnonzero(local_maxima(p))
        kept = top_peaks(t, dirs, 3)
        for c in kept:
            det.append(bool(len(pm)) and np.degrees(np.arccos(np.clip(flat[pm] @ flat[c], -1, 1))).min() <= 1.5)
        distinct.append(len(kept) < 2 or t.flat[kept[0]] - t.flat[kept[1]] >= 1.0)   # as mvdr_peaks.py
    ang = np.array(ang); ang_d = ang[np.array(distinct, dtype=bool)]
    return {"views": len(ang), "angle_median": float(np.median(ang)), "within_1deg": float((ang <= 1).mean()),
            "at_true_median": float(np.median(at)), "top3_detected": float(np.mean(det)),
            "distinct_views": int(len(ang_d)),
            "distinct_angle_median": float(np.median(ang_d)) if len(ang_d) else None,
            "distinct_within_1deg": float((ang_d <= 1).mean()) if len(ang_d) else None}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True); ap.add_argument("--positions", type=int, default=160)
    ap.add_argument("--train-subset", type=int, default=None,
                    help="the run used --max-train-views N (route mode): score exactly those training views")
    a = ap.parse_args()
    res = json.load(open(os.path.join(a.run, "results.json"))); cfg = res["config"]
    if cfg.get("head", "none") != "none":
        raise SystemExit("head models are not wired into this diagnostic")
    source = cfg["source"] if os.path.isabs(cfg["source"]) else os.path.join(T.REPO, cfg["source"])
    vmin, vmax = res["db_range"]; span = vmax - vmin
    if cfg.get("protocol"):
        import protocol as PR          # the run's own protocol: its training views are the protocol's training set
        train_names = PR.train_names(cfg["protocol"])
    else:
        train_names, test_names = T.ensure_split(source)
    views = T.read_colmap_text(os.path.join(source, "sparse", "0"))
    # stratified: whole training positions (every face), spread over the list
    vm = torch.tensor(np.stack([views[n + ".png"][0] if n + ".png" in views else views[n][0] for n in train_names]))
    idx = T.spread_positions(vm, 4 * a.positions)
    if a.train_subset:
        # train_rrf's route subset: whole positions evenly along the training list
        c = torch.linalg.inv(vm)[:, :3, 3]
        starts = [0] + [i for i in range(1, len(c)) if float((c[i] - c[i - 1]).norm()) > 1e-4]
        groups = [list(range(s0, s1)) for s0, s1 in zip(starts, starts[1:] + [len(c)])]
        keep = max(1, a.train_subset // 4)
        idx = [i for j in np.linspace(0, len(groups) - 1, keep).round().astype(int) for i in groups[j]]
    names = [train_names[i] for i in idx]
    if cfg.get("train_names_file"):
        # the run trained on an explicit list (train_rrf.py --train-names-file): score exactly those
        names = [l.strip() for l in open(cfg["train_names_file"]) if l.strip()]
    dev = "cuda"
    data = T.load_views(source, names, views, dev, want_float=True)
    ck = cfg["checkpoint"] if os.path.isabs(cfg["checkpoint"]) else os.path.join(T.REPO, cfg["checkpoint"])
    model = T.RRF(ck, cfg["mode"], 1, cfg["sh_degree"], dev, train_opacity=not cfg.get("freeze_opacity", False),
                  train_geometry=cfg.get("train_geometry", False))
    model.sh_backend = cfg.get("sh_backend", "torch")
    st = torch.load(os.path.join(a.run, "rrf_state.pt"), map_location=dev)
    if "lobe_w" in st:                    # --lobes: create the parameters, then load them
        model.add_lobes(st["lobe_w"].shape[1], 20.0, torch.zeros(1, 3))
    if "pcolor_mlp" in st:                # --pcolor: the same, the receiver normalisation comes from the state
        c = st["pcolor_cfg"]
        model.add_pcolor(c["width"], c["hidden"], c["n_freqs"], torch.zeros(2, 3))
    if "em_means" in st:                  # --emitters: the points and their scale come from the state
        model.add_emitters(st["em_means"].cpu().numpy(), float(torch.exp(st["em_scales"][0, 0])))
    if "em_pcolor_mlp" in st:             # --em-pcolor: its normalisation comes from the state too
        c = st["em_pcolor_cfg"]
        model.add_em_pcolor(c["width"], c["hidden"], c["n_freqs"], torch.zeros(2, 3))
    model.load_state(st)
    out = os.path.join(a.run + "_trainfit", "renders")
    with torch.no_grad():
        m = T.evaluate(model, data, list(range(len(names))), span, vmin, save_dir=out, group=cfg.get("eval_group", False))
    dirs = pixel_dirs(source)
    pk_train = peak_metrics(names, out, source, dirs)
    pk_path = os.path.join(os.path.dirname(a.run), f"peaks_{os.path.basename(a.run)}.json")
    pk_test = json.load(open(pk_path)) if os.path.exists(pk_path) else None
    rep = {"run": a.run, "train_views": len(names), "train_positions": len(names) // 4,
           "train": {"psnr_rgb": m["psnr_rgb"], "rmse_db": m["rmse_db"], **pk_train},
           "test": None if pk_test is None else {
               "psnr_rgb": res["final"]["psnr_rgb"], "rmse_db": res["final"]["rmse_db"],
               "angle_median": pk_test["main_peak_angle_deg"]["median"],
               "within_1deg": pk_test["main_peak_angle_deg"]["within_1deg"],
               "at_true_median": pk_test["power_at_true_peak_err_db"]["median"],
               "top3_detected": pk_test["top3_peaks"]["detected_within_1p5deg"]}}
    for k in ("train", "test") if pk_test is not None else ("train",):
        r = rep[k]
        print(f"{k:5s}: PSNR(jet) {r['psnr_rgb']:.2f}  RMSE {r['rmse_db']:.2f} dB | main peak median {r['angle_median']:.2f} deg, "
              f"<= 1 deg {100 * r['within_1deg']:.1f} %, at true peak {r['at_true_median']:+.2f} dB, top-3 {100 * r['top3_detected']:.1f} %")
    d = rep["train"]
    if d["distinct_views"]:
        print(f"train, distinct main peak only ({d['distinct_views']} of {d['views']} views): median {d['distinct_angle_median']:.2f} deg, "
              f"<= 1 deg {100 * d['distinct_within_1deg']:.1f} %")
    json.dump(rep, open(a.run + "_trainfit.json", "w"), indent=1)


if __name__ == "__main__":
    main()
