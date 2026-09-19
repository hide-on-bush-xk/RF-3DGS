"""Transfer curve: how far does a geometry adapted on one transmitter carry?

x = distance from the source transmitter to Tx-B, y = in-range RMSE on Tx-B
after the source-adapted geometry is frozen there with colours reset
(t_txB_geom<X>_frozen). References: the visual geometry frozen on Tx-B
(t_txB_cold) and Tx-B's own unfrozen fit (t_txB_geom). The benefit
b(d) = RMSE_cold - RMSE(d) is fitted with b0 exp(-d / d_c) (grid search on
d_c, least squares on b0), and d_c is the correlation length of the
adaptation. A negative benefit (worse than the visual geometry) is kept as
is; the fit treats it as data.

    PYTHONUTF8=1 python rrf_gsplat/transfer_curve.py
"""

from __future__ import annotations

import json
import os

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REG = os.path.join(REPO, "RF-3DGS_dataset/regenerated")
OUT = os.path.join(REPO, "output/rrf")
TX_B = np.array([8.2, -5.4, 2.0])
# source name -> (transfer run, source dataset whose generation_meta holds tx_loc)
SOURCES = {"A": ("t_txB_geomA_frozen", "3dgs_MVDR_100"), "C": ("t_txB_geomC_frozen", "3dgs_MVDR_txC")}
SOURCES.update({n: (f"t_txB_geom{n}_frozen", f"3dgs_MVDR_tx{n}") for n in "DENLOHMK"})


def _final(run):
    p = os.path.join(OUT, run, "results.json")
    if not os.path.exists(p):
        return None
    return json.load(open(p))["final"]


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="", help="transfer budget suffix of the runs, e.g. 2k (t_txB_geomX_frozen_2k, t_txB_cold_2k)")
    cfg = ap.parse_args()
    suf = f"_{cfg.tag}" if cfg.tag else ""
    cold, own = _final("t_txB_cold" + suf), _final("t_txB_geom" + suf)
    if own is None:
        own = _final("t_txB_geom")                    # the 10k own-unfrozen reference, if the tagged one is absent
    rows = []
    for name, (run, ds) in SOURCES.items():
        f = _final(run + suf)
        meta = os.path.join(REG, ds, "generation_meta.json")
        if f is None or not os.path.exists(meta):
            continue
        tx = np.array(json.load(open(meta))["tx_loc"], dtype=float)
        rows.append({"source": name, "run": run, "tx": tx.round(3).tolist(), "distance_m": float(np.linalg.norm(tx - TX_B)),
                     "rmse_in_range": f["rmse_db_in_range"], "rmse_db": f["rmse_db"], "psnr": f["psnr_rgb"], "ssim": f["ssim_rgb"],
                     "benefit_db": (cold["rmse_db_in_range"] - f["rmse_db_in_range"]) if cold else None})
    rows.sort(key=lambda r: r["distance_m"])
    out = {"tx_b": TX_B.tolist(), "transfer_budget": cfg.tag or "10k", "cold": cold, "own_unfrozen": own, "rows": rows}
    if cold is None or not rows:
        print(f"no rows for tag '{cfg.tag}'"); return
    print(f"Tx-B references: visual geometry frozen {cold['rmse_db_in_range']:.2f} dB in range; own geometry unfrozen {own['rmse_db_in_range']:.2f} dB")
    print(f"{'src':>3} {'tx':>22} {'d to B':>7} {'in-range':>9} {'benefit':>8} {'PSNR':>6}")
    for r in rows:
        print(f"{r['source']:>3} {str(r['tx']):>22} {r['distance_m']:7.2f} {r['rmse_in_range']:9.2f} {r['benefit_db']:+8.2f} {r['psnr']:6.2f}")
    if len(rows) >= 4 and cold:
        d = np.array([r["distance_m"] for r in rows]); b = np.array([r["benefit_db"] for r in rows])
        best = None
        for dc in np.linspace(0.25, 40.0, 1600):
            e = np.exp(-d / dc)
            b0 = float((b * e).sum() / (e * e).sum())
            sse = float(((b - b0 * e) ** 2).sum())
            if best is None or sse < best[2]:
                best = (dc, b0, sse)
        dc, b0, sse = best
        # also a constant-offset variant: b(d) = b0 exp(-d/dc) + c (c = the far-field benefit, may be negative)
        best2 = None
        for dc2 in np.linspace(0.25, 40.0, 1600):
            e = np.exp(-d / dc2); A = np.stack([e, np.ones_like(e)], 1)
            coef, *_ = np.linalg.lstsq(A, b, rcond=None)
            sse2 = float(((b - A @ coef) ** 2).sum())
            if best2 is None or sse2 < best2[3]:
                best2 = (dc2, float(coef[0]), float(coef[1]), sse2)
        ss_tot = float(((b - b.mean()) ** 2).sum()) or 1.0
        out["fit"] = {"model": "benefit = b0 exp(-d/d_c)", "d_c_m": dc, "b0_db": b0, "r2": 1 - sse / ss_tot,
                      "with_offset": {"model": "benefit = b0 exp(-d/d_c) + c", "d_c_m": best2[0], "b0_db": best2[1], "c_db": best2[2], "r2": 1 - best2[3] / ss_tot},
                      "n_points": len(rows)}
        print(f"fit b(d) = b0 exp(-d/d_c): d_c = {dc:.2f} m, b0 = {b0:.2f} dB, R^2 = {out['fit']['r2']:.2f}; "
              f"with offset: d_c = {best2[0]:.2f} m, b0 = {best2[1]:.2f}, c = {best2[2]:+.2f} dB, R^2 = {out['fit']['with_offset']['r2']:.2f}")
    else:
        print(f"{len(rows)} points so far; the fit needs 4")
    json.dump(out, open(os.path.join(OUT, f"transfer_curve{suf}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
