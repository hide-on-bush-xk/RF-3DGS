"""Information ladder, step 1: measurements without training (docs/cluster_log.md §4; readings written before the run).

    nice -n 10 ~/envs/rf-gsplat/bin/python rrf_gsplat/cluster/ladder_step1.py

Views: the 160-position training route subset (640 views, as train_rrf.py --max-train-views 640), its distinct views
selected exactly as diag_train_fit.py / live_comm.py. Faces split into the ring-max face (the face holding the
position's maximum) and the weak faces (the other three).
  M1  the 15 capacity runs, leave-one-out nearest neighbour (pool: the other 159 route positions / the other 466
      training positions) and the mirror-image predictor, converted to bits and split by face
  M2  per peak surface point S: observers, their angular spread, identifiable SH3 coefficients
  M3  required spherical-harmonic degree from the peak observer's local contrast (Bernstein bound)
  M4  single-point SH3 fits of S's demands: L1, L2, peak-weighted L1
Direction information b = log2(Omega_ring / Omega_cap(max(eps, 1 deg))), Omega_ring = the 4 faces' pixel solid angles.
Outputs: output/cluster/ladder/step1/{viewinfo.json, m1.json, points.json, summary.json}.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time

import numpy as np
import torch
from scipy.spatial import cKDTree

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "rrf_gsplat"))
import protocol as PR                                      # noqa: E402
from mvdr_peaks import angle, pixel_dirs, top_peaks        # noqa: E402

SRC = os.path.join(REPO, "RF-3DGS_dataset", "regenerated", "3dgs_APS_60_gp100")
PROTO = os.path.join(REPO, "rrf_gsplat", "protocol_v1")
CKPT = os.path.join(REPO, "RF-3DGS_dataset", "blender_visual_trained", "chkpnt30000.pth")
RUNS = os.path.join(REPO, "output", "rrf", "dirloss_cap_falcon")
OUT = os.path.join(REPO, "output", "cluster", "ladder", "step1")
ARMS, SEEDS = ("base", "eg", "eg3", "ce", "ce3"), (0, 1, 2)
SURF_OPACITY, CONE_DEG, BLOCK_R, NEAR_S, STEP = 0.5, 0.5, 0.05, 0.15, 0.025
torch.set_num_threads(4)

p = open(os.path.join(SRC, "sparse", "0", "cameras.txt")).read().split()
W, H, FX, FY, CX, CY = int(p[2]), int(p[3]), *map(float, p[4:8])
meta = json.load(open(os.path.join(SRC, "generation_meta.json")))
VMIN, VMAX = meta["spec_min_db"], meta["spec_max_db"]
SPAN = VMAX - VMIN
TX = np.array(meta["tx_loc"], dtype=np.float64)
DIRS = pixel_dirs(SRC)
FLAT = DIRS.reshape(-1, 3)
_u = (np.arange(W) + 0.5 - CX) / FX
_v = (np.arange(H) + 0.5 - CY) / FY
PIX_SR = (1.0 / (FX * FY)) / (1 + _u[None, :] ** 2 + _v[:, None] ** 2) ** 1.5      # solid angle per pixel [H, W]
OMEGA_RING = 4 * float(PIX_SR.sum())
DB_PER_DEG_PER_L = 0.5 * SPAN * math.pi / 180                                        # Bernstein: |grad| <= L * 0.5 per rad


def bits(eps_deg):
    e = np.radians(np.maximum(np.asarray(eps_deg, dtype=np.float64), 1.0))
    return np.log2(OMEGA_RING / (2 * np.pi * (1 - np.cos(e))))


def qrot(w, x, y, z):
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


POSES = {}
for line in open(os.path.join(SRC, "sparse", "0", "images.txt")):
    q = line.split()
    if len(q) >= 10 and q[9].endswith(".png"):
        R = qrot(*map(float, q[1:5]))
        t = np.array(list(map(float, q[5:8])))
        POSES[q[9][:-4]] = (R, t, -R.T @ t)
TRAIN = PR.train_names(PROTO)
GROUPS = [TRAIN[4 * k:4 * k + 4] for k in range(len(TRAIN) // 4)]
assert all(np.allclose(POSES[g[0]][2], POSES[n][2]) for g in GROUPS for n in g)
ROUTE = np.linspace(0, len(GROUPS) - 1, 160).round().astype(int)


def yawkey(n):
    return tuple(np.round(POSES[n][0][2], 2))


def truth(n):
    return np.load(os.path.join(SRC, "spectra_float", f"{n}.npy")).astype(np.float64)


def view_info():
    path = os.path.join(OUT, "viewinfo.json")
    if os.path.exists(path):
        return json.load(open(path))
    info = {}
    for n in TRAIN:
        t = truth(n)
        if t.max() < -250 or t.max() - t.min() < 1e-3:        # as diag_train_fit.peak_metrics
            info[n] = {"flat": True, "kt": int(t.argmax()), "max": float(t.max())}
            continue
        kept = top_peaks(t, DIRS, 3)
        info[n] = {"flat": False, "kt": int(t.argmax()), "max": float(t.max()),
                   "distinct": bool(len(kept) < 2 or t.flat[kept[0]] - t.flat[kept[1]] >= 1.0)}
    json.dump(info, open(path, "w"))
    return info


def distinct_views(info):
    """(name, route slot, is_ringmax) for every distinct view of the 160 route positions."""
    out = []
    for slot, j in enumerate(ROUTE):
        g = GROUPS[j]
        ringmax = max(g, key=lambda n: info[n]["max"])
        out += [(n, slot, n == ringmax) for n in g if not info[n]["flat"] and info[n]["distinct"]]
    return out


def classes(views):
    ring = np.array([r for _, _, r in views])
    return {"all": np.ones(len(views), bool), "ringmax": ring, "weak": ~ring}


def summarise(eps, views):
    eps = np.asarray(eps, dtype=np.float64)
    res = {}
    for c, m in classes(views).items():
        e = eps[m]
        res[c] = {"n": int(m.sum()), "hit": float((e <= 1.0).mean()), "eps_median": float(np.median(e)),
                  "bits_mean": float(bits(e).mean())}
    return res


# ---- mirror-image predictor (as verified 2026-09-25: walls and prior fitted on the route views only) ----------
def pix_of(n, X):
    R, t, _ = POSES[n]
    xc = R @ X + t
    if xc[2] <= 0:
        return None
    u, v = FX * xc[0] / xc[2] + CX, FY * xc[1] / xc[2] + CY
    return int(v) * W + int(u) if (0 <= u < W and 0 <= v < H) else None


def fit_images(info, views):
    names = [n for n, _, _ in views]
    Cc = np.array([POSES[n][2] for n in names])
    D = np.array([POSES[n][0].T @ FLAT[info[n]["kt"]] for n in names])

    def cover(X):
        u = X[None] - Cc
        u /= np.linalg.norm(u, axis=1, keepdims=True)
        return np.degrees(np.arccos(np.clip((u * D).sum(1), -1, 1))) <= 1.0
    cands = {}
    for xw in np.arange(-8, 12, 0.05):
        cands[("x", round(xw, 2))] = np.array([2 * xw - TX[0], TX[1], TX[2]])
    for yw in np.arange(-14, 5, 0.05):
        cands[("y", round(yw, 2))] = np.array([TX[0], 2 * yw - TX[1], TX[2]])
    for zw in np.arange(-2.5, 3, 0.05):
        cands[("z", round(zw, 2))] = np.array([TX[0], TX[1], 2 * zw - TX[2]])
    cands[("los", 0)] = TX.copy()
    A = {k: cover(X) for k, X in cands.items()}
    chosen, covered = [], np.zeros(len(names), bool)
    for _ in range(6):
        best = max(A, key=lambda k: (A[k] & ~covered).sum())
        if (A[best] & ~covered).sum() < 5:
            break
        chosen.append(best)
        covered |= A[best]
    IM = [cands[k] for k in chosen if k[0] != "los"] + [TX.copy()]
    labels = [f"{k[0]}={float(k[1]):.2f}" for k in chosen if k[0] != "los"] + ["LOS"]
    cin, cok = np.zeros(len(IM)), np.zeros(len(IM))
    for n, c, d in zip(names, Cc, D):
        for k, X in enumerate(IM):
            if pix_of(n, X) is None:
                continue
            u = (X - c) / np.linalg.norm(X - c)
            cin[k] += 1
            cok[k] += np.degrees(np.arccos(np.clip(u @ d, -1, 1))) <= 1
    return {"IM": IM, "labels": labels, "prior": cok / np.maximum(cin, 1)}


def image_explains(info, n, images):
    """True if the true main-peak direction is within 1 deg of the direction to any image point."""
    _, _, c = POSES[n]
    d = POSES[n][0].T @ FLAT[info[n]["kt"]]
    for X in images["IM"]:
        u = (X - c) / np.linalg.norm(X - c)
        if np.degrees(np.arccos(np.clip(u @ d, -1, 1))) <= 1.0:
            return True
    return False


# ---- M1 ------------------------------------------------------------------------------------------------------
def m1(info, views, images):
    rows = {}
    kts = [info[n]["kt"] for n, _, _ in views]
    # zero point: a uniformly random predicted pixel in the same face
    zero = [float(bits(np.degrees(np.arccos(np.clip(FLAT @ FLAT[kt], -1, 1)))).mean()) for kt in kts]
    zm = {c: float(np.mean(np.array(zero)[m])) for c, m in classes(views).items()}
    for arm in ARMS:
        for s in SEEDS:
            run = f"{arm}_s{s}"
            eps = []
            for n, _, _ in views:
                pr = np.load(os.path.join(RUNS, f"{run}_trainfit", "renders", f"{n}.npy"))
                assert pr.shape == (H, W), pr.shape
                eps.append(angle(DIRS, info[n]["kt"], int(pr.argmax())))
            rows[run] = summarise(eps, views)
            rows[run]["official_check"] = json.load(open(os.path.join(RUNS, f"{run}_trainfit.json")))["train"]["distinct_within_1deg"]
    for label, pool in (("nn_loo_pool159", ROUTE), ("nn_loo_pool466", np.arange(len(GROUPS)))):
        bykey = {}
        for j in pool:
            for n in GROUPS[j]:
                bykey.setdefault(yawkey(n), []).append(n)
        arr = {k: (v, np.array([POSES[m][2] for m in v])) for k, v in bykey.items()}
        eps = []
        for n, _, _ in views:
            v, P = arr[yawkey(n)]
            dd = np.linalg.norm(P - POSES[n][2], axis=1)
            dd[dd < 1e-9] = np.inf
            eps.append(angle(DIRS, info[n]["kt"], info[v[int(dd.argmin())]]["kt"]))
        rows[label] = summarise(eps, views)
    eps = []
    for n, _, _ in views:
        inside = [(k, pix_of(n, X)) for k, X in enumerate(images["IM"])]
        inside = [(k, pp) for k, pp in inside if pp is not None]
        if not inside:
            eps.append(180.0)
            continue
        k, pp = max(inside, key=lambda kp: images["prior"][kp[0]])
        eps.append(angle(DIRS, info[n]["kt"], pp))
    rows["mirror_predictor"] = summarise(eps, views)
    return {"zero_point_bits": zm, "omega_ring_sr": OMEGA_RING, "bits_max": float(bits(0.0)), "rows": rows,
            "images": {"labels": images["labels"], "prior": images["prior"].tolist()}}


# ---- M2-M4 ---------------------------------------------------------------------------------------------------
def sh3(d):
    x, y, z = d[:, 0], d[:, 1], d[:, 2]
    xx, yy, zz, xy, yz, xz = x * x, y * y, z * z, x * y, y * z, x * z
    return np.stack([np.full_like(x, 0.28209479177387814), -0.4886025119029199 * y, 0.4886025119029199 * z,
                     -0.4886025119029199 * x, 1.0925484305920792 * xy, -1.0925484305920792 * yz,
                     0.31539156525252005 * (2 * zz - xx - yy), -1.0925484305920792 * xz, 0.5462742152960396 * (xx - yy),
                     -0.5900435899266435 * y * (3 * xx - yy), 2.890611442640554 * xy * z,
                     -0.4570457994644658 * y * (4 * zz - xx - yy), 0.3731763325901154 * z * (2 * zz - 3 * xx - 3 * yy),
                     -0.4570457994644658 * x * (4 * zz - xx - yy), 1.445305721320277 * z * (xx - yy),
                     -0.5900435899266435 * x * (xx - 3 * yy)], 1)


def wl1(B, y, w, iters=200):
    """Weighted L1 regression by IRLS, started from the weighted L2 solution."""
    sw = np.sqrt(w)
    c = np.linalg.lstsq(B * sw[:, None], y * sw, rcond=None)[0]
    for _ in range(iters):
        r = np.abs(B @ c - y)
        ww = np.sqrt(w / np.maximum(r, 1e-6))
        c_new = np.linalg.lstsq(B * ww[:, None], y * ww, rcond=None)[0]
        if np.max(np.abs(c_new - c)) < 1e-9:
            c = c_new
            break
        c = c_new
    return c


def points(info, views, images):
    ck, _ = torch.load(CKPT, weights_only=False, map_location="cpu")
    xyz = ck[1].detach().double().numpy()
    op = torch.sigmoid(ck[6].detach().double()).reshape(-1).numpy()
    Ms = xyz[op >= SURF_OPACITY]
    tree = cKDTree(Ms)
    names160 = [n for j in ROUTE for n in GROUPS[j]]
    T = {n: truth(n) for n in names160}
    Rall = np.stack([POSES[n][0] for n in names160])
    tall = np.stack([POSES[n][1] for n in names160])
    slot_of = np.repeat(np.arange(160), 4)
    centres = np.stack([POSES[GROUPS[j][0]][2] for j in ROUTE])
    tan = math.tan(math.radians(CONE_DEG))
    recs = []
    t0 = time.time()
    for i, (n, slot, ring) in enumerate(views):
        c = centres[slot]
        d = POSES[n][0].T @ FLAT[info[n]["kt"]]
        v = Ms - c
        s = v @ d
        perp = np.linalg.norm(v - s[:, None] * d, axis=1)
        ok = (s > 0.2) & (perp < tan * s)
        rec = {"view": n, "slot": int(slot), "ringmax": bool(ring), "image": image_explains(info, n, images)}
        if not ok.any():
            rec["surface"] = False
            recs.append(rec)
            continue
        k = np.flatnonzero(ok)[np.argmin(s[ok])]
        X = Ms[k]
        rec.update(surface=True, depth=float(s[k]))
        # which route positions see X inside a face, and at which pixel
        xc = np.einsum("nij,j->ni", Rall, X) + tall
        with np.errstate(divide="ignore", invalid="ignore"):
            uu = FX * xc[:, 0] / xc[:, 2] + CX
            vv = FY * xc[:, 1] / xc[:, 2] + CY
        inside = (xc[:, 2] > 0) & (uu >= 0) & (uu < W) & (vv >= 0) & (vv < H)
        obs = {}
        for f in np.flatnonzero(inside):
            obs.setdefault(int(slot_of[f]), (names160[f], int(vv[f]), int(uu[f])))
        fa = names160.index(n)                  # the peak observer's demand comes from the peak's own face
        if inside[fa]:
            obs[slot] = (n, int(vv[fa]), int(uu[fa]))
        # occlusion: a surface Gaussian within BLOCK_R of the segment, away from X (samples stop NEAR_S before X)
        keep = {}
        segs, owner = [], []
        for sl, pix in obs.items():
            L = np.linalg.norm(X - centres[sl])
            ss = np.arange(0.2, L - NEAR_S, STEP)
            if len(ss):
                u = (X - centres[sl]) / L
                segs.append(centres[sl][None] + ss[:, None] * u[None])
                owner.append(np.full(len(ss), sl))
            keep[sl] = pix
        if segs:
            pts, own = np.concatenate(segs), np.concatenate(owner)
            dist, _ = tree.query(pts, k=1, distance_upper_bound=BLOCK_R, workers=4)
            for sl in np.unique(own[np.isfinite(dist)]):
                if sl != slot:
                    keep.pop(int(sl), None)
        if slot not in keep:                    # the peak's own observer always counts (S is on its peak ray)
            f = [q for q in range(4) if slot_of[slot * 4 + q] == slot and inside[slot * 4 + q]]
            keep[slot] = (names160[slot * 4 + f[0]], int(vv[slot * 4 + f[0]]), int(uu[slot * 4 + f[0]])) if f else None
        if keep.get(slot) is None:
            rec["surface"] = False
            recs.append(rec)
            continue
        sls = sorted(keep)
        dem = np.array([T[keep[sl][0]][keep[sl][1], keep[sl][2]] for sl in sls])          # dB
        U = np.array([(X - centres[sl]) / np.linalg.norm(X - centres[sl]) for sl in sls])  # receiver -> S
        ia = sls.index(slot)
        th = np.degrees(np.arccos(np.clip(U @ U[ia], -1, 1)))
        B = sh3(U)
        sv = np.linalg.svd(B, compute_uv=False)
        rec.update(n_obs=len(sls), spread_max_deg=float(th.max()), spread_median_deg=float(np.median(np.delete(th, ia))) if len(sls) > 1 else 0.0,
                   sh3_identifiable=int((sv >= 0.01 * sv[0]).sum()), d_peak_db=float(dem[ia]))
        # M3: local contrast demanded of S toward its angularly nearest other observers
        oth = [q for q in range(len(sls)) if q != ia and th[q] >= 0.5]
        if oth:
            near = sorted(oth, key=lambda q: th[q])[:5]
            r = float(np.median([max(dem[ia] - dem[q], 0.0) / th[q] for q in near]))
            rec.update(slope_db_per_deg=r, L_need=int(math.ceil(r / DB_PER_DEG_PER_L)), n_near=len(near))
        # M4: single-point SH3 fits of the demands (normalised dB above the floor, as the model's value)
        if len(sls) >= 20:
            y = (dem - VMIN) / SPAN
            fits = {}
            for name, w in (("l1", np.ones(len(y))), ("l2", None),
                            ("peak_l1", np.where(np.arange(len(y)) == ia, len(y) - 1.0, 1.0))):
                cc = np.linalg.lstsq(B, y, rcond=None)[0] if w is None else wl1(B, y, w)
                pred = np.clip(B @ cc, 0, 1)
                others = np.delete(np.abs(pred - y), ia) * SPAN
                fits[name] = {"deficit_db": float((y[ia] - pred[ia]) * SPAN), "others_median_err_db": float(np.median(others))}
            rec["m4"] = fits
            # post-hoc, exploratory (not in the pre-written readings): the cost of the peak-weighted fit on the 5
            # angularly nearest observers (where the parallax conflict sits, invisible in a median over ~100), and
            # whether the restored S would rival those observers' own face maximum (a false main peak there)
            if oth:
                pk = np.clip(B @ wl1(B, y, np.where(np.arange(len(y)) == ia, len(y) - 1.0, 1.0)), 0, 1)
                l1 = np.clip(B @ wl1(B, y, np.ones(len(y))), 0, 1)
                fmax = np.array([(T[keep[sls[q]][0]].max() - VMIN) / SPAN for q in near])
                # S is a FALSE peak for q only if q's own face maximum lies more than 1 deg from S's pixel there
                away = np.array([angle(DIRS, info[keep[sls[q]][0]]["kt"], keep[sls[q]][1] * W + keep[sls[q]][2]) > 1.0
                                 for q in near])
                rec["x_near"] = {"cost_db": float(np.median((np.abs(pk[near] - y[near]) - np.abs(l1[near] - y[near])) * SPAN)),
                                 "false_peak_share": float(np.mean((pk[near] >= fmax - 1.0 / SPAN) & away)),
                                 "false_peak_share_l1": float(np.mean((l1[near] >= fmax - 1.0 / SPAN) & away)),
                                 "near_own_peak_at_S": float(np.mean(~away))}
        recs.append(rec)
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(views)} peak points, {time.time() - t0:.0f} s", flush=True)
    return recs


def shares(vals, edges):
    vals = np.asarray(vals)
    return [float(((vals >= lo) & (vals <= hi)).mean()) if len(vals) else float("nan") for lo, hi in edges]


def readings(recs):
    out = {}
    groups = {"ringmax": [r for r in recs if r["ringmax"]], "weak": [r for r in recs if not r["ringmax"]],
              "weak_image": [r for r in recs if not r["ringmax"] and r["image"]],
              "weak_other": [r for r in recs if not r["ringmax"] and not r["image"]]}
    for g, rs in groups.items():
        s = [r for r in rs if r.get("surface")]
        L = [r["L_need"] for r in s if "L_need" in r]
        m4 = [r["m4"] for r in s if "m4" in r]
        o = {"views": len(rs), "with_surface": len(s), "no_surface": len(rs) - len(s),
             "n_obs_median": float(np.median([r["n_obs"] for r in s])) if s else None,
             "identifiable_median": float(np.median([r["sh3_identifiable"] for r in s])) if s else None,
             "spread_max_median_deg": float(np.median([r["spread_max_deg"] for r in s])) if s else None,
             "depth_median_m": float(np.median([r["depth"] for r in s])) if s else None,
             "with_L_need": len(L), "L_need_median": float(np.median(L)) if L else None,
             "L_need_bins_le3_4to6_7to10_gt10": shares(L, [(0, 3), (4, 6), (7, 10), (11, 10 ** 9)]),
             "slope_median_db_per_deg": float(np.median([r["slope_db_per_deg"] for r in s if "slope_db_per_deg" in r])) if L else None,
             "m4_points": len(m4)}
        if m4:
            dl1 = np.array([m["l1"]["deficit_db"] for m in m4])
            dl2 = np.array([m["l2"]["deficit_db"] for m in m4])
            dpk = np.array([m["peak_l1"]["deficit_db"] for m in m4])
            cost = np.array([m["peak_l1"]["others_median_err_db"] - m["l1"]["others_median_err_db"] for m in m4])
            o.update(m4_deficit_median_db={"l1": float(np.median(dl1)), "l2": float(np.median(dl2)), "peak_l1": float(np.median(dpk))},
                     m4_cost_median_db=float(np.median(cost)),
                     m4_share_objective_fixable=float(((dpk <= 3) & (cost <= 3)).mean()),
                     m4_share_repr_limited_original=float((dpk >= 10).mean()),
                     m4_share_repr_limited_amended=float((cost > 3).mean()))
        xn = [r["x_near"] for r in s if "x_near" in r]
        if xn:
            o["x_posthoc_near5"] = {"points": len(xn), "cost_median_db": float(np.median([x["cost_db"] for x in xn])),
                                    "cost_share_gt3db": float(np.mean([x["cost_db"] > 3 for x in xn])),
                                    "points_with_a_false_peak": float(np.mean([x["false_peak_share"] > 0 for x in xn])),
                                    "points_with_a_false_peak_l1": float(np.mean([x["false_peak_share_l1"] > 0 for x in xn])),
                                    "near_neighbours_whose_own_peak_is_at_S": float(np.mean([x["near_own_peak_at_S"] for x in xn]))}
        out[g] = o
    w = out["weak"]
    verdict = {}
    if w["n_obs_median"] is not None:
        verdict["M2"] = ("data sufficient at training positions" if w["n_obs_median"] >= 16 and w["identifiable_median"] >= 12
                         else "under-determined" if w["n_obs_median"] < 16 or w["identifiable_median"] < 8 else "between")
    le3 = out["ringmax"]["L_need_bins_le3_4to6_7to10_gt10"][0]
    verdict["M3_ringmax"] = f"{100 * le3:.0f} % need L <= 3 ({'in' if le3 >= 0.70 else 'OUTSIDE'} the expected >= 70 %)"
    b = w["L_need_bins_le3_4to6_7to10_gt10"]
    verdict["M3_weak"] = ("parameters lever (>= 50 % need L <= 6)" if b[0] + b[1] >= 0.5
                          else "representation (>= 50 % need L > 10)" if b[3] >= 0.5 else "between: test both")
    if w.get("m4_points"):
        verdict["M4_weak"] = {"objective_fixable": w["m4_share_objective_fixable"] >= 0.5,
                              "repr_limited_original": w["m4_share_repr_limited_original"] >= 0.5,
                              "repr_limited_amended": w["m4_share_repr_limited_amended"] >= 0.5}
    return out, verdict


def main():
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    info = view_info()
    views = distinct_views(info)
    nr = sum(r for _, _, r in views)
    print(f"distinct views {len(views)} (ring-max {nr}, weak {len(views) - nr}); Omega_ring {OMEGA_RING:.3f} sr, "
          f"max {float(bits(0.0)):.2f} bits; view info {time.time() - t0:.0f} s", flush=True)
    images = fit_images(info, views)
    print("mirror images:", dict(zip(images["labels"], np.round(images["prior"], 2).tolist())), flush=True)
    r1 = m1(info, views, images)
    json.dump(r1, open(os.path.join(OUT, "m1.json"), "w"), indent=1)
    worst = max(abs(v["all"]["hit"] - v["official_check"]) for k, v in r1["rows"].items() if "official_check" in v)
    print(f"M1 done, {time.time() - t0:.0f} s; largest |hit - diag_train_fit| over the 15 runs: {100 * worst:.3f} points", flush=True)
    for k, v in r1["rows"].items():
        print(f"  {k:16s} all {100 * v['all']['hit']:5.1f} % {v['all']['bits_mean']:5.2f} b | ring-max {100 * v['ringmax']['hit']:5.1f} % "
              f"{v['ringmax']['bits_mean']:5.2f} b | weak {100 * v['weak']['hit']:5.1f} % {v['weak']['bits_mean']:5.2f} b", flush=True)
    print("  zero point (random pixel in the face), bits:", {c: round(z, 2) for c, z in r1["zero_point_bits"].items()}, flush=True)
    recs = points(info, views, images)
    json.dump(recs, open(os.path.join(OUT, "points.json"), "w"), indent=1)
    rd, verdict = readings(recs)
    json.dump({"m1": r1, "points": rd, "verdict": verdict, "seconds": time.time() - t0},
              open(os.path.join(OUT, "summary.json"), "w"), indent=1)
    print(json.dumps({"points": rd, "verdict": verdict}, indent=1))
    print(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
