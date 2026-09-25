"""M3's placement oracle, part 1: where does the radio energy come from? (docs/tech_paths.md, M3 / P4)

For each chosen receiver position, re-solve the dataset's own paths (same scene, configuration and lattice seed as
its first solve), and take every path's interaction point (depth 1: the reflection or scattering point; the direct
path: the transmitter) with its power -- the same |a|^2 the angular power spectrum (APS) splats. Points are pooled
in voxels of --voxel metres; per position and face sector the strongest voxels that carry --keep-share of that sector's power are kept (at most
--cap per position), and the union over positions is written as emitter centres (power-weighted centroids) for
train_rrf.py --emitters.

Checks, written before the first run (2026-09-24, stage "M3 smoke"):
  index     vertex i belongs to path i: the direction from the receiver to a path's interaction point is its angle
            of arrival (depth 1; the direct path: the transmitter) -- >= 99 % of paths within 0.01 deg
  frame     the emitters and the visual Gaussians share a frame: median distance from an emitter to the nearest
            visual Gaussian <= 5 cm (interaction points lie on the mesh surfaces the visual Gaussians cover)
  peaks     the emitter set carries the APS peaks: its own APS (every kept voxel's power splatted at its direction
            from the receiver, the dataset's 1 deg kernel, cut to the four faces) against the dataset's truth --
            main peak <= 1 deg on >= 85 % of the distinct views (two independent draws of the scene: 98 %; the voxels,
            the kept share and one solve instead of four cost some of that)
  reported  emitters, kept voxels per position, captured power share, direct-path share, seconds

    python sionna_port/path_emitters.py --truth <APS dataset> --protocol rrf_gsplat/protocol_v1 --capacity 160 --out <npz>
    python sionna_port/path_emitters.py ... --positions 23 90 460 ... --out <npz>        (an explicit list: the smoke)
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(REPO, "rrf_gsplat"))


def capacity_positions(protocol, n):
    """The capacity benchmark's positions: train_rrf.py --max-train-views 4n (route subset) on the protocol's
    training list -- n whole positions evenly along it, both ends included. Dataset position = (name - 1) // 4."""
    import protocol as PR
    tr = PR.train_names(protocol)
    keep = np.linspace(0, len(tr) // 4 - 1, n).round().astype(int)
    return [(int(tr[4 * p]) - 1) // 4 for p in keep]


def position_paths(solver, scene, cfg, G, rx, i):
    """(points [P, 3], power [P], is_direct [P]) of position i's valid paths, from the dataset's first solve."""
    from sionna.rt import Receiver
    import drjit as dr
    scene.remove("rx") if "rx" in scene.receivers else None
    scene.add(Receiver(name="rx", position=list(rx), orientation=[G.VIEW_YAWS[0], 0.0, 0.0]))
    paths = G.solve_paths(solver, scene, cfg, view_index=i)
    a, tau = paths.cir(normalize_delays=False, out_type="torch")
    a = a[0, :, 0, 0, :, 0]
    amp = (a.abs().sum(dim=0) if a.dim() == 2 else a.abs()).double()      # as rf_spectra._path_arrays
    tau = tau.reshape(-1)
    keep = (torch.isfinite(tau) & (tau >= 0) & (amp > 0)).cpu().numpy()
    v = np.asarray(paths.vertices)                                         # [depth, (rx, tx,) paths, 3]
    it = np.asarray(paths.interactions)
    v = v.reshape(v.shape[0], -1, 3)[0]                                     # first interaction of each path
    it = it.reshape(it.shape[0], -1)[0]
    direct = it == 0                                                       # InteractionType.NONE: the LOS path
    pw = (amp ** 2).cpu().numpy()
    pts = np.where(direct[:, None], np.asarray(cfg.tx_loc, dtype=np.float64)[None], v.astype(np.float64))
    th, ph = np.asarray(paths.theta_r).reshape(-1), np.asarray(paths.phi_r).reshape(-1)
    aoa = np.stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)], 1)
    del paths
    dr.flush_malloc_cache(); torch.cuda.empty_cache()
    return pts[keep], pw[keep], direct[keep], aoa[keep]


def voxels(pts, pw, size):
    """Voxel key -> (power, power-weighted point sum)."""
    key = np.floor(pts / size).astype(np.int64)
    uniq, inv = np.unique(key, axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    p = np.bincount(inv, weights=pw, minlength=len(uniq))
    s = np.stack([np.bincount(inv, weights=pw * pts[:, k], minlength=len(uniq)) for k in range(3)], 1)
    return uniq, p, s


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--truth", required=True, help="the APS dataset whose configuration and seeds to re-solve")
    ap.add_argument("--protocol", default=None)
    ap.add_argument("--capacity", type=int, default=0, help="the capacity benchmark's n positions")
    ap.add_argument("--positions", type=int, nargs="*", default=None, help="an explicit list of dataset positions")
    ap.add_argument("--sets", nargs="*", default=None,
                    help="protocol sets whose positions to add (train, val, val_random, val_segment); the sealed test sets "
                         "are refused (protocol.eval_names). G2 of the generalisation diagnostics: --sets train val")
    ap.add_argument("--voxel", type=float, default=0.05)
    ap.add_argument("--keep-share", type=float, default=0.9)
    ap.add_argument("--cap", type=int, default=2000, help="most voxels kept per position")
    ap.add_argument("--checkpoint", default=os.path.join(REPO, "RF-3DGS_dataset", "blender_visual_trained", "chkpnt30000.pth"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache", default=None,
                    help="a directory of per-position voxel aggregates: read when present, else solved and written. The "
                         "solve is the expensive, GPU-bound part (467 positions: ~6 min); a cached build is CPU only")
    a = ap.parse_args()
    t0 = time.time()
    import generate_dataset as G
    meta = json.load(open(os.path.join(a.truth, "generation_meta.json")))
    if meta.get("spectrum", "").upper() != "APS":
        raise SystemExit("--truth must be an APS dataset (its paths and powers are what the emitters stand for)")
    fields = {f.name for f in dataclasses.fields(G.Config)}
    cfg = G.Config(**{k: v for k, v in meta.items() if k in fields})
    rt = {}                                         # the scene and solver, built only if some position is not cached

    def solver_scene():
        if not rt:
            from sionna.rt import PathSolver, Transmitter
            cwd = os.getcwd(); os.chdir(HERE)
            rt["scene"] = G.build_scene(cfg); os.chdir(cwd)
            rt["scene"].add(Transmitter(name="tx", position=list(cfg.tx_loc)))
            rt["solver"] = PathSolver()
        return rt["solver"], rt["scene"]
    groups = G.read_pose_groups(os.path.join(a.truth, "sparse", "0", "images.txt"))
    pos = list(a.positions or []) + (capacity_positions(a.protocol, a.capacity) if a.capacity else [])
    for set_name in a.sets or []:
        import protocol as PR
        names_s = PR.train_names(a.protocol) if set_name == "train" else PR.eval_names(a.protocol, set_name)
        pos += [(int(n) - 1) // 4 for n in names_s]              # dataset position = (1-based name - 1) // 4
    pos = sorted(set(pos))
    per_pos, pooled = [], {}
    kept_pos = {}                                   # position -> kept voxel keys with their power (for the peak check)
    # vertex i belongs to path i: rx -> its interaction point = its AoA. Kept as a histogram (0.001 deg bins up to
    # 2 deg, then an overflow bin) so that a cached build can still report it
    edges = np.append(np.arange(0, 2.0005, 0.001), np.inf)
    idx_hist = np.zeros(len(edges) - 1, dtype=np.int64)
    n_cached = 0
    for i in pos:
        cf = os.path.join(a.cache, f"pos_{i:04d}_v{int(round(a.voxel * 1000))}mm.npz") if a.cache else None
        if cf and os.path.isfile(cf):
            z = np.load(cf)
            key, p, s = z["key"].astype(np.int64), z["p"], z["s"]
            n_paths, direct_share, h = int(z["n_paths"]), float(z["direct_share"]), z["idx_hist"]
            n_cached += 1
        else:
            pts, pw, direct, aoa = position_paths(*solver_scene(), cfg, G, groups[i][0], i)
            n_paths = len(pw)
            if n_paths:
                dv = pts - np.asarray(groups[i][0], dtype=np.float64)[None]
                dv /= np.linalg.norm(dv, axis=1, keepdims=True).clip(1e-9)
                h = np.histogram(np.degrees(np.arccos(np.clip((dv * aoa).sum(1), -1, 1))), bins=edges)[0]
                key, p, s = voxels(pts, pw, a.voxel)
                direct_share = float(pw[direct].sum() / pw.sum())
            else:
                h = np.zeros(len(edges) - 1, dtype=np.int64)
                key, p, s = np.zeros((0, 3), np.int64), np.zeros(0), np.zeros((0, 3))
                direct_share = 0.0
            if cf:
                os.makedirs(a.cache, exist_ok=True)
                np.savez(cf, key=key.astype(np.int32), p=p, s=s, n_paths=n_paths, direct_share=direct_share,
                         idx_hist=h)
        idx_hist += h
        if n_paths == 0:
            per_pos.append({"position": i, "paths": 0}); print(f"position {i}: no paths"); continue
        # kept per face sector (azimuth within 45 deg of a face's yaw), not per position: a position's strongest
        # 90 % all sit on the faces towards the transmitter, and the back faces' peaks were dropped (smoke, first
        # run: 73.5 % of distinct views <= 1 deg; every voxel kept: 94.1 %)
        c = s / p[:, None] - np.asarray(groups[i][0], dtype=np.float64)[None]
        az = np.arctan2(c[:, 1], c[:, 0])
        yaws = np.asarray(G.VIEW_YAWS)
        sector = np.argmin(np.abs(np.angle(np.exp(1j * (az[:, None] - yaws[None])))), axis=1)
        sel, cum_s = [], []
        for k in range(len(yaws)):
            idx = np.flatnonzero(sector == k)
            if len(idx) == 0:
                continue
            o = idx[np.argsort(-p[idx])]
            cum = np.cumsum(p[o]) / p[o].sum()
            n_k = min(int(np.searchsorted(cum, a.keep_share)) + 1, a.cap // len(yaws), len(o))
            sel.append(o[:n_k]); cum_s.append(float(cum[n_k - 1]))
        sel = np.concatenate(sel); n_keep = len(sel)
        tot = float(p.sum())
        for k, pk, sk in zip(map(tuple, key[sel]), p[sel], s[sel]):
            w = pk / tot                                # each position's kept voxels weigh by their share of it
            q = pooled.setdefault(k, [0.0, np.zeros(3)])
            q[0] += w; q[1] += w * sk / pk               # centroid pooled across positions by share
        kept_pos[i] = (key[sel], p[sel], s[sel] / p[sel, None])
        per_pos.append({"position": i, "paths": n_paths, "voxels": int(len(p)), "kept": int(n_keep),
                        "captured_share": float(p[sel].sum() / tot), "sector_share_min": float(min(cum_s)),
                        "direct_share": direct_share})
    keys = list(pooled)
    means = np.stack([pooled[k][1] / pooled[k][0] for k in keys]).astype(np.float32)
    weight = np.array([pooled[k][0] for k in keys], dtype=np.float32)
    # frame: distance from each emitter to the nearest visual Gaussian
    from scipy.spatial import cKDTree
    (m, _it) = torch.load(a.checkpoint, weights_only=False, map_location="cpu")
    vis = m[1].detach().cpu().numpy().astype(np.float64)
    tree = cKDTree(vis)
    dist = tree.query(means.astype(np.float64))[0]
    # a sharper frame diagnostic than the distance itself (large wall Gaussians sit centimetres off the surface):
    # shift the emitters and see whether the median distance is smallest at no shift
    shift_sweep = {}
    for ax in range(3):
        for sh in (-0.2, -0.1, -0.05, 0.05, 0.1, 0.2):
            dlt = np.zeros(3); dlt[ax] = sh
            shift_sweep[f"{'xyz'[ax]}{sh:+.2f}"] = float(np.median(tree.query(means.astype(np.float64) + dlt)[0]))
    # peaks: the emitter set's own APS against the truth, per face
    from rf_spectra import aps_splat
    from mvdr_peaks import angle, pixel_dirs, top_peaks
    dirs = pixel_dirs(a.truth)
    names = [l.split()[9][:-4] for l in open(os.path.join(a.truth, "sparse", "0", "images.txt")) if len(l.split()) >= 10]
    face_cfg = dataclasses.replace(cfg)
    ang, dist_flags = [], []
    for i in pos:
        if i not in kept_pos:
            continue
        _, pk, ck = kept_pos[i]
        d = ck - np.asarray(groups[i][0], dtype=np.float64)[None]
        d /= np.linalg.norm(d, axis=1, keepdims=True).clip(1e-9)
        th, ph = np.arccos(np.clip(d[:, 2], -1, 1)), np.arctan2(d[:, 1], d[:, 0])
        eq = aps_splat(torch.tensor(th), torch.tensor(ph), torch.tensor(pk, dtype=torch.float64), sigma=cfg.splat_sigma)
        for j, yaw in enumerate(groups[i][1]):
            n = names[4 * i + j]
            t = np.load(os.path.join(a.truth, "spectra_float", n + ".npy")).astype(np.float64)
            if t.max() - t.min() < 1e-3:
                continue
            face = G.cut_face(eq, face_cfg, yaw).numpy()
            kept = top_peaks(t, dirs, 3)
            dist_flags.append(len(kept) < 2 or t.flat[kept[0]] - t.flat[kept[1]] >= 1.0)
            ang.append(angle(dirs, int(t.argmax()), int(face.argmax())))
    ang, dist_flags = np.array(ang), np.array(dist_flags, dtype=bool)
    rep = {"truth": a.truth, "positions": pos, "voxel_m": a.voxel, "keep_share": a.keep_share, "cap": a.cap,
           "emitters": int(len(means)), "seconds": time.time() - t0,
           "kept_per_position_median": float(np.median([r["kept"] for r in per_pos if r.get("kept")])),
           "captured_share_median": float(np.median([r["captured_share"] for r in per_pos if r.get("kept")])),
           "direct_share_median": float(np.median([r["direct_share"] for r in per_pos if r.get("kept")])),
           "frame_nearest_visual_m": {"median": float(np.median(dist)), "p90": float(np.percentile(dist, 90)),
                                      "median_under_shift": shift_sweep},
           "index_vertex_vs_aoa_deg": {"within_0p01": float(idx_hist[:10].sum() / max(idx_hist.sum(), 1)),
                                       "p99": float(edges[1:][np.searchsorted(np.cumsum(idx_hist),
                                                                               0.99 * idx_hist.sum())])},
           "positions_from_cache": n_cached,
           "peaks": {"views": int(len(ang)), "within_1deg": float((ang <= 1).mean()),
                     "distinct_views": int(dist_flags.sum()),
                     "distinct_within_1deg": float((ang[dist_flags] <= 1).mean()) if dist_flags.any() else None,
                     "angle_median": float(np.median(ang))},
           "per_position": per_pos}
    rep["pass"] = {"index": rep["index_vertex_vs_aoa_deg"]["within_0p01"] >= 0.99,
                   "frame": rep["frame_nearest_visual_m"]["median"] <= 0.05,
                   "peaks": rep["peaks"]["distinct_within_1deg"] is not None and rep["peaks"]["distinct_within_1deg"] >= 0.85}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.savez(a.out, means=means, weight=weight, voxel=a.voxel, positions=np.array(pos))
    json.dump(rep, open(os.path.splitext(a.out)[0] + ".json", "w"), indent=1)
    print(json.dumps({k: v for k, v in rep.items() if k != "per_position"}, indent=1))
    print("ALL PASS" if all(rep["pass"].values()) else "FAIL: " + ", ".join(k for k, v in rep["pass"].items() if not v))


if __name__ == "__main__":
    main()
