"""Scene 2 against scene 1 on the three replicated claims.

  A. density crossover   copy vs field per training density, both scenes
  B. zone structure      transfer benefit on the target (room_S2) by source
                         topology: same room / adjacent room / opposite room /
                         corridor / hall; separation and exponential fit
  C. consistency         per-view normalisation and per-view seed, both scenes

    PYTHONUTF8=1 python scene2/analyze_scene2.py
"""

from __future__ import annotations

import json
import os

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(REPO, "output/rrf")


def load(name):
    p = os.path.join(OUT, name)
    return json.load(open(p)) if os.path.exists(p) else None


def final(run):
    d = load(os.path.join(run, "results.json"))
    return d["final"] if d else None


def density():
    print("A. density crossover (median errors; copy = nearest training position; field with --delay-depth)")
    rows = [("scene 1 (lobby)", "baselines_%s.json", [("m_multi_24_tut_cs_depth", 800), ("m_multi_24_tut_cs_depth_v640", 160), ("m_multi_24_tut_cs_depth_v320", 80),
                                                        ("m_multi_24_tut_cs_depth_v160", 40), ("m_multi_24_tut_cs_depth_v80", 20)]),
            # 564 route positions at 0.2 m, 20 % held out: 451 training positions, then 225 / 113 / 56 / 28 (views = 4 x positions)
            ("scene 2 (corridor)", "baselines_%s.json", [("s2_multi_corrM_depth", 451), ("s2_multi_corrM_depth_v900", 225), ("s2_multi_corrM_depth_v452", 113),
                                                          ("s2_multi_corrM_depth_v224", 56), ("s2_multi_corrM_depth_v112", 28)])]
    for scene, pat, runs in rows:
        print(f"  {scene}")
        print(f"  {'positions':>9} {'spacing':>8} {'az copy/field':>16} {'zen copy/field':>16} {'delay copy/field':>18}")
        for run, npos in runs:
            b = load(pat % run)
            if not b:
                print(f"  {npos:>9}  (missing {run})"); continue
            c, f = b["errors"]["nearest"], b["errors"]["rrf"]; sp = b["nearest_train_distance_m"]["median"]
            print(f"  {npos:>9} {sp:8.2f} {c['az']['median']:7.2f}/{f['az']['median']:<7.2f} {c['zen']['median']:7.2f}/{f['zen']['median']:<7.2f} {c['delay']['median']:8.2f}/{f['delay']['median']:<8.2f}")


def zones():
    print("\nB. zone structure on room_S2 (in-range RMSE, 2k-step transfer; benefit = cold - transferred)")
    cold, own = final("s2_t_roomS2_cold_2k"), final("s2_t_roomS2_geom_2k")
    if not cold:
        print("  (no scene-2 target runs yet)"); return
    tx = json.load(open(os.path.join(REPO, "scene2/corridor/tx_positions.json")))
    tgt = np.array(tx["room_S2"])
    group = {"room_S2b": "same room", "room_S1": "adjacent room", "room_S1b": "adjacent room", "room_S3": "adjacent room",
             "room_N1": "opposite room", "room_N2": "opposite room", "room_N2b": "opposite room", "room_N3": "opposite room",
             "corridor_W": "corridor", "corridor_M": "corridor", "corridor_E": "corridor", "hall": "hall"}
    rows = []
    for name, pos in tx.items():
        if name == "room_S2":
            continue
        f = final(f"s2_t_roomS2_geom{name}_2k")
        if not f:
            continue
        rows.append((name, group[name], float(np.linalg.norm(np.array(pos) - tgt)), f["rmse_db_in_range"], cold["rmse_db_in_range"] - f["rmse_db_in_range"]))
    print(f"  cold (visual geometry) {cold['rmse_db_in_range']:.2f} dB, own unfrozen {own['rmse_db_in_range'] if own else float('nan'):.2f} dB")
    print(f"  {'source':>11} {'group':>14} {'dist':>6} {'in-range':>9} {'benefit':>8}")
    for name, g, d, r, b in sorted(rows, key=lambda r: r[2]):
        print(f"  {name:>11} {g:>14} {d:6.2f} {r:9.2f} {b:+8.2f}")
    by = {}
    for _, g, _, _, b in rows:
        by.setdefault(g, []).append(b)
    print("  by group: " + "; ".join(f"{g}: mean {np.mean(v):+.2f} (n={len(v)})" for g, v in by.items()))
    if len(rows) >= 4:
        d = np.array([r[2] for r in rows]); b = np.array([r[4] for r in rows])
        best = None
        for dc in np.linspace(0.25, 40, 800):
            e = np.exp(-d / dc); A = np.stack([e, np.ones_like(e)], 1); coef, *_ = np.linalg.lstsq(A, b, rcond=None)
            sse = float(((b - A @ coef) ** 2).sum())
            if best is None or sse < best[0]:
                best = (sse, dc, coef)
        ss = float(((b - b.mean()) ** 2).sum()) or 1.0
        print(f"  exponential + plateau fit: d_c {best[1]:.1f} m, b0 {best[2][0]:.2f}, c {best[2][1]:+.2f} dB, R^2 {1 - best[0] / ss:.2f}; "
              f"corr(benefit, distance) {np.corrcoef(d, b)[0, 1]:+.2f}")


def consistency():
    print("\nC. cross-view consistency")
    for scene, pairs in (("scene 1", [("per-view normalisation (MVDR rgb)", "e2_mvdr_rgb", "e1_mvdr_perview", "psnr_rgb"),
                                      ("per-view seed (MULTI power in-range RMSE)", "m_multi_24_tut_cs", "m_multi_24_tut_cs_yawseed", "rmse_db_in_range")]),
                         ("scene 2", [("per-view normalisation (MVDR rgb)", "s2_e1_corrM_gpct_rgb", "s2_e1_corrM_perview_rgb", "psnr_rgb"),
                                      ("per-view seed (MULTI power in-range RMSE)", "s2_multi_corrM_plain", "s2_multi_corrM_yawseed", "rmse_db_in_range")])):
        for label, a, b, key in pairs:
            fa, fb = final(a), final(b)
            if fa and fb:
                print(f"  {scene:8s} {label:44s} {fa[key]:7.2f} -> {fb[key]:7.2f}")
            else:
                print(f"  {scene:8s} {label:44s} (missing)")


if __name__ == "__main__":
    density(); zones(); consistency()
