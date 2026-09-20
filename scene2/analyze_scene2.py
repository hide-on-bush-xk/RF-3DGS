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
            ("scene 2 (corridor)", "baselines_%s.json", [("s2_multi_corrM_depth_full", 451), ("s2_multi_corrM_depth_v900_full", 225), ("s2_multi_corrM_depth_v452_full", 113),
                                                          ("s2_multi_corrM_depth_v224_full", 56), ("s2_multi_corrM_depth_v112_full", 28)])]
    for scene, pat, runs in rows:
        print(f"  {scene}")
        print(f"  {'positions':>9} {'spacing':>8} {'az copy/field':>16} {'zen copy/field':>16} {'delay copy/field':>18}")
        for run, npos in runs:
            b = load(pat % run)
            if not b:
                print(f"  {npos:>9}  (missing {run})"); continue
            c, f = b["errors"]["nearest"], b["errors"]["rrf"]; sp = b["nearest_train_distance_m"]["median"]
            print(f"  {npos:>9} {sp:8.2f} {c['az']['median']:7.2f}/{f['az']['median']:<7.2f} {c['zen']['median']:7.2f}/{f['zen']['median']:<7.2f} {c['delay']['median']:8.2f}/{f['delay']['median']:<8.2f}")


def density_by_space():
    """Scene 2 density sweep scored on all 452 held-out views (the *_full re-renders), overall and per space class of the
    held-out receiver, with the copy/field crossover per class (linear interpolation of copy - field on the measured spacing)."""
    runs = [("s2_multi_corrM_depth_full", 451), ("s2_multi_corrM_depth_v900_full", 225), ("s2_multi_corrM_depth_v452_full", 113),
            ("s2_multi_corrM_depth_v224_full", 56), ("s2_multi_corrM_depth_v112_full", 28)]
    rows = [(n, load(f"baselines_{r}.json")) for r, n in runs]
    rows = [(n, b) for n, b in rows if b]
    if not rows:
        print("\nA'. (no full-set baselines yet)"); return
    classes = ["all"] + sorted({c for _, b in rows for c in b.get("by_space", {})})
    print(f"\nA'. scene 2 density on all {rows[0][1]['views']} held-out views, per space class (median copy / field; spacing = nearest training position, median)")
    out = {}
    for c in classes:
        print(f"  [{c}]")
        print(f"  {'positions':>9} {'views':>5} {'spacing':>8} {'az copy/field':>16} {'zen copy/field':>16} {'delay copy/field':>18}")
        pts = []
        for n, b in rows:
            g = b if c == "all" else b["by_space"].get(c)
            if not g:
                continue
            e = g["errors"]; sp = g["nearest_train_distance_m"]["median"]
            pts.append({"positions": n, "views": g["views"], "spacing": sp, "copy": {q: e["nearest"][q]["median"] for q in ("az", "zen", "delay")},
                        "field": {q: e["rrf"][q]["median"] for q in ("az", "zen", "delay")}})
            print(f"  {n:>9} {g['views']:>5} {sp:8.2f} {e['nearest']['az']['median']:7.2f}/{e['rrf']['az']['median']:<7.2f} "
                  f"{e['nearest']['zen']['median']:7.2f}/{e['rrf']['zen']['median']:<7.2f} {e['nearest']['delay']['median']:8.2f}/{e['rrf']['delay']['median']:<8.2f}")
        pts.sort(key=lambda p: p["spacing"])
        def cross(q):
            d = [(p["spacing"], p["copy"][q] - p["field"][q]) for p in pts]
            if d[0][1] > 0:
                return f"below {d[0][0]:.2f} m"
            for (x0, y0), (x1, y1) in zip(d[:-1], d[1:]):
                if y0 <= 0 < y1:
                    return f"{x0 + (x1 - x0) * (-y0) / (y1 - y0):.2f} m"
            return f"beyond {d[-1][0]:.2f} m"
        print(f"  crossover: azimuth {cross('az')}, zenith {cross('zen')}, delay {cross('delay')}")
        out[c] = {"points": pts, "crossover": {q: cross(q) for q in ("az", "zen", "delay")}}
    json.dump(out, open(os.path.join(OUT, "density_by_space_s2.json"), "w"), indent=1)


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
    ov = overlap(rows)
    fit_note = None
    if len(rows) >= 4:
        d = np.array([r[2] for r in rows]); b = np.array([r[4] for r in rows])
        best = None
        for dc in np.linspace(0.25, 40, 800):
            e = np.exp(-d / dc); A = np.stack([e, np.ones_like(e)], 1); coef, *_ = np.linalg.lstsq(A, b, rcond=None)
            sse = float(((b - A @ coef) ** 2).sum())
            if best is None or sse < best[0]:
                best = (sse, dc, coef)
        ss = float(((b - b.mean()) ** 2).sum()) or 1.0
        fit_note = (f"exponential + plateau fit: d_c {best[1]:.1f} m, b0 {best[2][0]:.2f}, c {best[2][1]:+.2f} dB, R² {1 - best[0] / ss:.2f}; "
                    f"corr(benefit, distance) {np.corrcoef(d, b)[0, 1]:+.2f}")
        print("  " + fit_note)
    # the dashboard's transfer card reads this (same shape as transfer_curve_2k.json)
    facing = [r for r in rows if r[1] == "same room" or r[0] in ("room_N2", "room_N2b")]
    other = [r for r in rows if r not in facing]
    out = {"scene": "scene 2 (corridor), target room_S2", "transfer_budget": "2k", "tx_b": tgt.tolist(),
           "cold": {"rmse_db_in_range": cold["rmse_db_in_range"]}, "own_unfrozen": {"rmse_db_in_range": own["rmse_db_in_range"] if own else None},
           "rows": [{"source": n, "run": f"s2_t_roomS2_geom{n}_2k", "tx": tx[n], "distance_m": d, "rmse_in_range": r, "benefit_db": b, "group": g,
                     "lit_iou": ov.get(n, (None, None))[0], "db_corr": ov.get(n, (None, None))[1]} for n, g, d, r, b in sorted(rows, key=lambda r: r[2])],
           "fit": None, "fit_note": fit_note,
           "zones": {"label_in": "same room or the facing room across the corridor", "label_out": "adjacent, diagonal, corridor, hall",
                     "east_sources": [r[0] for r in facing], "east_benefit": [round(r[4], 3) for r in facing], "other_benefit": [round(r[4], 3) for r in other],
                     "east_mean": float(np.mean([r[4] for r in facing])) if facing else None, "east_min": float(min(r[4] for r in facing)) if facing else None,
                     "other_mean": float(np.mean([r[4] for r in other])) if other else None, "other_max": float(max(r[4] for r in other)) if other else None}}
    json.dump(out, open(os.path.join(OUT, "transfer_curve_s2.json"), "w"), indent=1)


def overlap(rows):
    """Lit-surface overlap of each source with the target (room_S2) on the held-out views, as diag_overlap.py
    does for the lobby: strong IoU = pixels within 30 dB of the view's maximum in both over either (empty views,
    written at -300 dB, are skipped); lit IoU = pixels within 60 dB of the dataset's maximum, same ratio; and the
    Pearson r of the two spectra on the union of lit pixels. Prediction 1 said the IoU between rooms is near zero."""
    reg = os.path.join(REPO, "RF-3DGS_dataset/regenerated")
    tgt_dir = os.path.join(reg, "s2_MVDR_txroom_S2_gpct")           # the trainer writes the split there; spectra_float is hard-linked
    if not os.path.isdir(tgt_dir):
        return {}
    names = [l.strip() for l in open(os.path.join(tgt_dir, "test_index.txt")) if l.strip()]
    tgt = np.stack([np.load(os.path.join(tgt_dir, "spectra_float", n + ".npy")) for n in names]).astype(np.float32)
    t_empty = tgt.reshape(len(names), -1).max(1) <= -299
    t_strong = tgt > (tgt.reshape(len(names), -1).max(1)[:, None, None] - 30.0)
    t_lit = tgt > (tgt.max() - 60.0)
    print(f"  overlap with room_S2 on {len(names)} held-out views ({int(t_empty.sum())} of them empty for the target):")
    print(f"  {'source':>11} {'strong IoU':>10} {'lit IoU':>8} {'dB corr':>8} {'benefit':>8}")
    ious = []
    for name, g, d, r, b in sorted(rows, key=lambda r: r[2]):
        src_dir = os.path.join(reg, f"s2_MVDR_tx{name}_gpct")
        src = np.stack([np.load(os.path.join(src_dir, "spectra_float", n + ".npy")) for n in names]).astype(np.float32)
        s_empty = src.reshape(len(names), -1).max(1) <= -299
        keep = ~(t_empty | s_empty)
        s_strong = src > (src.reshape(len(names), -1).max(1)[:, None, None] - 30.0)
        s_lit = src > (src.max() - 60.0)
        iou_s = float((s_strong & t_strong)[keep].sum() / max((s_strong | t_strong)[keep].sum(), 1)) if keep.any() else 0.0
        iou_l = float((s_lit & t_lit).sum() / max((s_lit | t_lit).sum(), 1))
        u = (s_lit | t_lit)
        corr = float(np.corrcoef(src[u], tgt[u])[0, 1]) if u.sum() > 10 else float("nan")
        ious.append((name, iou_s, iou_l, corr, b))
        print(f"  {name:>11} {iou_s:10.3f} {iou_l:8.3f} {corr:8.3f} {b:+8.2f}")
    if len(ious) >= 4:
        b = np.array([x[4] for x in ious])
        for j, lab in ((1, "strong IoU"), (2, "lit IoU"), (3, "dB corr")):
            x = np.array([x[j] for x in ious])
            print(f"  corr(benefit, {lab}) = {np.corrcoef(x, b)[0, 1]:+.2f} over {len(ious)} sources")
    return {x[0]: (x[2], x[3]) for x in ious}


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
    density(); density_by_space(); zones(); consistency()
