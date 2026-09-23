"""Round 24 table: time split (load / training / running eval / final eval) and quality of every r24_* run.

Quality is the decoded-channel error on all 640 held-out views (eval_baselines.py, rrf row: median / P90) for
the multi-channel runs, and PSNR(jet) / RMSE dB for the db runs. Time is what train_rrf.py recorded on the
exclusive card; 'train' excludes the running evaluations so runs with different --eval-every compare.

    python rrf_gsplat/dlss_table.py [--prefix r24_]
"""

from __future__ import annotations

import argparse
import glob
import json
import os

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="r24_")
    ap.add_argument("--out", default=os.path.join(REPO, "output", "rrf", "dlss_table.json"))
    a = ap.parse_args()
    rows = []
    for d in sorted(glob.glob(os.path.join(REPO, "output", "rrf", a.prefix + "*"))):
        f = os.path.join(d, "results.json")
        if not os.path.isfile(f):
            continue
        r = json.load(open(f)); c = r["config"]; name = os.path.basename(d)
        row = {"run": name, "mode": c["mode"], "faces": c.get("faces_per_step", 1), "lr_scale": c.get("lr_scale", 1.0),
               "sh": c.get("sh_backend", "torch"), "steps": c["iterations"], "views_seen": r.get("views_seen", c["iterations"]),
               "load_s": r.get("load_seconds"), "train_s": r.get("train_seconds_excl_running_eval", r["train_seconds"]),
               "running_eval_s": r.get("running_eval_seconds"), "final_eval_s": r.get("eval_seconds"),
               "ms_per_step": 1000 * r.get("train_seconds_excl_running_eval", r["train_seconds"]) / c["iterations"],
               "psnr_jet": r["final"]["psnr_rgb"], "rmse_db": r["final"]["rmse_db"]}
        b = os.path.join(REPO, "output", "rrf", f"baselines_{name}.json")
        if os.path.isfile(b):
            e = json.load(open(b))["errors"]["rrf"]
            for ch in ("az", "zen", "delay"):
                row[f"{ch}_med"] = e[ch]["median"]; row[f"{ch}_p90"] = e[ch]["p90"]; row[f"{ch}_rmse"] = e[ch]["rmse"]
        rows.append(row)
    json.dump(rows, open(a.out, "w"), indent=1)
    hdr = f"{'run':30s} {'faces':>5s} {'lr':>4s} {'steps':>6s} {'views':>6s} {'ms/st':>6s} {'train':>6s} {'fev':>5s} {'PSNR':>6s} {'az med/P90':>13s} {'zen med/P90':>13s} {'dly med/P90':>13s}"
    print(hdr)
    for w in rows:
        q = lambda k: (f"{w[k + '_med']:.3f}/{w[k + '_p90']:.2f}" if k + "_med" in w else "MISSING")  # noqa: E731
        print(f"{w['run']:30s} {w['faces']:5d} {w['lr_scale']:4.1f} {w['steps']:6d} {w['views_seen']:6d} {w['ms_per_step']:6.1f} "
              f"{w['train_s']:6.0f} {(w['final_eval_s'] or 0):5.0f} {w['psnr_jet']:6.2f} {q('az'):>13s} {q('zen'):>13s} {q('delay'):>13s}")


if __name__ == "__main__":
    main()
