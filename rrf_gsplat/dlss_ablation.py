"""Round 41 tables: every added feature's gain, from the runs' results.json and peaks_*.json (mvdr_peaks.py).

  B ladder     S0 (SH1) -> + head -> + geo guides -> + phys guides -> + latent -> + ring = S5; each row's
               difference from the row above is that feature's gain
  B LOO        S5 minus one feature at a time: the loss when it is taken out of the full model
  B 2 x 2      SH3 x (head, peak loss): B1, B4, B6, B5
  A1           the label upsampler: bilinear -> cnn -> + coords -> + peak loss (MVDR: peaks; MULTI: decoded)
Seeds: a run named <name>_s1 / _s2 is another seed of <name>; means with the min..max across seeds are reported.

    python rrf_gsplat/dlss_ablation.py
"""

from __future__ import annotations

import json
import os
import re

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RRF = os.path.join(REPO, "output", "rrf")

METRICS = [  # key, label, source, better
    ("psnr", "PSNR(jet)", "res", "high"), ("rmse", "RMSE dB", "res", "low"),
    ("ang_med", "peak dir med", "pk", "low"), ("ang_1deg", "<=1 deg %", "pk", "high"),
    ("at_true", "at true peak dB", "pk", "high"), ("pk_pow", "peak power dB", "pk", "high"),
    ("top3", "top-3 det %", "pk", "high"), ("false", "false peaks %", "pk", "low"),
]


def one(name):
    """Every number of one run, or None when it has not been run."""
    r = os.path.join(RRF, name, "results.json"); p = os.path.join(RRF, f"peaks_{name}.json")
    if not os.path.isfile(r):
        return None
    res = json.load(open(r)); out = {}
    f = res.get("final") or {}
    out["psnr"] = f.get("psnr_rgb"); out["rmse"] = f.get("rmse_db")
    out["res_rms_db"] = f.get("head_residual_rms_db"); out["res_share"] = f.get("head_residual_var_share")
    it = (res.get("config") or {}).get("iterations")
    ts = res.get("train_seconds_excl_running_eval", res.get("train_seconds"))
    out["ms_step"] = 1000 * ts / it if ts and it else None
    out["params"] = res.get("optimised_gaussian_params"); out["adam_mib"] = res.get("adam_state_mib")
    out["head_params"] = res.get("head_params")
    if os.path.isfile(p):
        k = json.load(open(p))
        out["ang_med"] = k["main_peak_angle_deg"]["median"]; out["ang_1deg"] = 100 * k["main_peak_angle_deg"]["within_1deg"]
        out["at_true"] = k["power_at_true_peak_err_db"]["median"]; out["pk_pow"] = k["main_peak_power_err_db"]["median"]
        out["top3"] = 100 * k["top3_peaks"]["detected_within_1p5deg"]
        fr = (k.get("false_peaks") or {}).get("false_rate")
        out["false"] = None if fr is None else 100 * fr
    return out


def seeds(name):
    """All seeds of a run: name itself (seed 0) and name_s1, name_s2, ..."""
    runs = [one(name)] + [one(f"{name}_s{s}") for s in range(1, 10)]
    return [r for r in runs if r is not None]


def agg(runs, key):
    v = [r[key] for r in runs if r.get(key) is not None]
    if not v:
        return None, None, None
    return float(np.mean(v)), float(np.min(v)), float(np.max(v))


def fmt(m, lo, hi, n):
    if m is None:
        return "MISSING"
    return f"{m:.2f}" if n <= 1 or hi - lo < 1e-9 else f"{m:.2f} [{lo:.2f}..{hi:.2f}]"


def table(title, rows, ref=None):
    """rows: [(label, run name)]; ref: the label of the row the deltas are taken against (None = previous row)."""
    print(f"\n### {title}\n")
    head = "| row | runs | " + " | ".join(m[1] for m in METRICS) + " | ms/step | Adam MiB | head residual RMS dB |"
    print(head); print("|" + " --- |" * (head.count("|") - 1))
    prev, out = None, []
    means = {}
    for label, name in rows:
        rs = seeds(name)
        cells = []
        for key, _, _, _ in METRICS:
            m, lo, hi = agg(rs, key)
            means[(label, key)] = m
            base = means.get(((ref or prev), key)) if (ref or prev) else None
            d = f" ({m - base:+.2f})" if (m is not None and base is not None and label != (ref or prev)) else ""
            cells.append(fmt(m, lo, hi, len(rs)) + d)
        ms = agg(rs, "ms_step")[0]; adam = agg(rs, "adam_mib")[0]; rr = agg(rs, "res_rms_db")[0]
        print(f"| {label} | {len(rs)} | " + " | ".join(cells) +
              f" | {'MISSING' if ms is None else f'{ms:.1f}'} | {'MISSING' if adam is None else f'{adam:.0f}'} | "
              f"{'—' if rr is None else f'{rr:.2f}'} |")
        out.append({"label": label, "run": name, "seeds": len(rs), **{k: agg(rs, k)[0] for k, *_ in METRICS},
                    "ms_step": ms, "adam_mib": adam, "head_residual_rms_db": rr})
        prev = label
    return out


def main():
    report = {}
    report["B_ladder"] = table("B ladder (SH1): each row adds one feature; (delta) against the row above", [
        ("S0 SH1", "r41_B_S0_sh1"), ("+ head", "r41_B_S1_head"), ("+ geo guides", "r41_B_S2_geo"),
        ("+ phys guides", "r41_B_S3_geo_phys"), ("+ latent", "r41_B_S4_geo_phys_lat"), ("+ ring = S5", "r41_B_S5_full")])
    report["B_loo"] = table("B leave-one-out: (delta) against the full S5", [
        ("S5 full", "r41_B_S5_full"), ("- geo", "r41_B_S5_minus_geo"), ("- phys", "r41_B_S5_minus_phys"),
        ("- latent", "r41_B_S5_minus_lat"), ("- ring", "r41_B_S5_minus_strip"), ("width 8", "r41_B_S5_w8")], ref="S5 full")
    report["B_peak_sh1"] = table("B peak loss on SH1: (delta) against the row above", [
        ("S0 SH1", "r41_B_S0_sh1"), ("S0 + peak", "r41_B_S0_peak"), ("S5", "r41_B_S5_full"), ("S5 + peak", "r41_B_S5_peak")])
    report["B_2x2"] = table("B quality branch, SH3 x (head, peak loss): (delta) against B1", [
        ("B1 SH3", "r41_B_B1_sh3"), ("B4 + peak", "r41_B_B4_sh3_peak"), ("B6 + head", "r41_B_B6_sh3_head"),
        ("B5 + head + peak", "r41_B_B5_sh3_head_peak")], ref="B1 SH3")
    report["A1_mvdr"] = table("A1 label upsampler, MVDR (live pair): (delta) against the row above", [
        ("bilinear", "r41_A1_mvdr_bilinear"), ("cnn", "r41_A1_mvdr_cnn_s0"), ("+ coords", "r41_A1_mvdr_cnn_coords_s0"),
        ("+ peak loss", "r41_A1_mvdr_cnn_coords_peak_s0")])
    # MULTI arms: decoded channels from eval_baselines
    print("\n### A1 label upsampler, MULTI (decoded against the 300x200 labels; median / P90, pw = power-weighted median)\n")
    print("| arm | az | az pw | zen | zen pw | delay ns | delay pw |"); print("|" + " --- |" * 7)
    for tag in ("bilinear", "cnn", "cnn_coords", "cnn_coords_peak"):
        b = os.path.join(RRF, f"baselines_r41_A1_multi_{tag}.json")
        if not os.path.isfile(b):
            print(f"| {tag} | MISSING |" + " |" * 5); continue
        e = json.load(open(b))["errors"]["rrf"]
        print(f"| {tag} | " + " | ".join(f"{e[c]['median']:.3f} / {e[c]['p90']:.2f} | {e[c]['median_pw']:.3f}" for c in ("az", "zen", "delay")) + " |")
    json.dump(report, open(os.path.join(RRF, "dlss_ablation.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
