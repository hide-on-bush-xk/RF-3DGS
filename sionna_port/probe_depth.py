"""Probe: what does raising max_depth (reflection order) do to the APS target? (Ke: "你把反射次数增大会如何?")

The 12 stratified smoke positions of the M3 oracle, the APS dataset's configuration (iso pattern, scattering 0.7,
60 GHz), 1M samples per solve (the same for every depth, so the depths compare with each other, not with the
dataset's 4 x 4M), max_depth 1 / 2 / 3. Per depth: path count (and whether it approaches the solver cap -- a silent
truncation bit us before), solve seconds, the power share of each reflection order, and per face the main peak of
the APS against depth 1's (distinct views by depth 1's own peaks).
Expectations written before the run (reported, not judged):
  orders >= 2 carry < 25 % of the power; >= 80 % of distinct views keep their main peak within 1 deg of depth 1's;
  depth 2 solves 3-10 x slower than depth 1.
"""
import dataclasses, json, os, sys, time
import numpy as np
import torch
REPO = r"C:\Users\Ke\Documents\GitHub\RF-3DGS"
sys.path.insert(0, os.path.join(REPO, "sionna_port")); sys.path.insert(0, os.path.join(REPO, "rrf_gsplat"))
os.chdir(os.path.join(REPO, "sionna_port"))
import generate_dataset as G
from sionna.rt import PathSolver, Receiver, Transmitter
import drjit as dr
from rf_spectra import aps_splat
from mvdr_peaks import angle, pixel_dirs, top_peaks

AP = os.path.join(REPO, "RF-3DGS_dataset", "regenerated", "3dgs_APS_60")
meta = json.load(open(os.path.join(AP, "generation_meta.json")))
fields = {f.name for f in dataclasses.fields(G.Config)}
cfg0 = G.Config(**{k: v for k, v in meta.items() if k in fields})
cfg0 = dataclasses.replace(cfg0, samples_per_src=1_000_000)
scene = G.build_scene(cfg0)
scene.add(Transmitter(name="tx", position=list(cfg0.tx_loc)))
solver = PathSolver()
groups = G.read_pose_groups(os.path.join(AP, "sparse", "0", "images.txt"))
dirs = pixel_dirs(AP)
positions = [23, 95, 230, 293, 382, 439, 460, 463, 577, 636, 728, 799]
n0 = 10 ** (meta["aps_floor_db"] / 10)

faces, rep = {}, {}
for depth in (1, 2, 3):
    cfg = dataclasses.replace(cfg0, max_depth=depth)
    t_solve, n_paths, pw_by_order, fcs = 0.0, [], np.zeros(depth + 1), {}
    try:
        for i in positions:
            scene.remove("rx") if "rx" in scene.receivers else None
            scene.add(Receiver(name="rx", position=list(groups[i][0]), orientation=[G.VIEW_YAWS[0], 0.0, 0.0]))
            t0 = time.perf_counter()
            paths = G.solve_paths(solver, scene, cfg, view_index=i)
            dr.sync_thread(); t_solve += time.perf_counter() - t0
            a, tau = paths.cir(normalize_delays=False, out_type="torch")
            a = a[0, :, 0, 0, :, 0]
            amp = (a.abs().sum(dim=0) if a.dim() == 2 else a.abs()).double()
            tau = tau.reshape(-1)
            keep = (torch.isfinite(tau) & (tau >= 0) & (amp > 0)).cpu().numpy()
            it = np.asarray(paths.interactions); it = it.reshape(it.shape[0], -1)
            order = (it != 0).sum(0)[keep]
            pw = (amp ** 2).cpu().numpy()[keep]
            th = np.asarray(paths.theta_r).reshape(-1)[keep]; ph = np.asarray(paths.phi_r).reshape(-1)[keep]
            n_paths.append(int(keep.sum()))
            for o in range(depth + 1):
                pw_by_order[o] += pw[order == o].sum()
            eq = aps_splat(torch.tensor(th, dtype=torch.float64), torch.tensor(ph, dtype=torch.float64),
                           torch.tensor(pw, dtype=torch.float64), sigma=cfg.splat_sigma)
            for j, yaw in enumerate(groups[i][1]):
                f = G.cut_face(eq, cfg, yaw).double().numpy()
                fcs[(i, j)] = 10 * np.log10(10 ** (f / 10) + n0)
            del paths
            dr.flush_malloc_cache(); torch.cuda.empty_cache()
    except Exception as exc:
        rep[depth] = {"error": f"{type(exc).__name__}: {exc}"[:300]}
        print(f"depth {depth}: FAILED {rep[depth]['error']}", flush=True)
        dr.flush_malloc_cache(); torch.cuda.empty_cache()
        continue
    faces[depth] = fcs
    share = pw_by_order / pw_by_order.sum()
    rep[depth] = {"paths_median": int(np.median(n_paths)), "paths_max": int(max(n_paths)),
                  "near_cap": max(n_paths) > 0.9 * cfg.max_num_paths_per_src, "solve_s": round(t_solve, 1),
                  "power_share_by_order": [round(float(x), 4) for x in share]}
    print(f"depth {depth}: {rep[depth]}", flush=True)

ref = faces.get(1, {})
for depth in (2, 3):
    if depth not in faces:
        continue
    ang, dist, flat = [], [], 0
    for k, f1 in ref.items():
        if f1.max() - f1.min() < 1e-3:
            flat += 1; continue
        kept = top_peaks(f1, dirs, 3)
        dist.append(len(kept) < 2 or f1.flat[kept[0]] - f1.flat[kept[1]] >= 1.0)
        ang.append(angle(dirs, int(f1.argmax()), int(faces[depth][k].argmax())))
    ang, dist = np.array(ang), np.array(dist, dtype=bool)
    rep[depth]["main_peak_vs_depth1"] = {"views": int(len(ang)), "within_1deg": round(float((ang <= 1).mean()), 3),
                                         "distinct_views": int(dist.sum()),
                                         "distinct_within_1deg": round(float((ang[dist] <= 1).mean()), 3),
                                         "flat_at_depth1": flat}
    print(f"depth {depth} vs 1: {rep[depth]['main_peak_vs_depth1']}", flush=True)
json.dump(rep, open(os.path.join(REPO, "output", "rrf", "m3", "depth_probe.json"), "w"), indent=1)
