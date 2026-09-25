"""Smoke: gsplat's `lidar` camera model as a 360-degree panorama of a trained field, against our four pinhole faces.

Why: docs/stage2_notes.md ("gsplat 相机模型") said gsplat has no angle-domain camera, so the four-face stitch (seams,
four renders per position) looked unavoidable. gsplat_win has had camera_model="lidar" since 30ed0d85 (2026-01-28)
and 4b641b96 (2026-02-10): a structured spinning-lidar model whose image axes are azimuth and elevation, rendered by
the 3DGUT path (with_ut + with_eval3d). This checks that it renders the same field as the faces before anything is
built on it.

Three renders of the same trained field at the same receiver centre:
  A  pinhole, classic rasteriser: the training path today (4 faces batched, 300 x 200 each)
  B  pinhole, eval3d (with_ut, with_eval3d, packed=False, global_z_order=False): the lidar's rasteriser family,
     on the pinhole camera
  C  lidar panorama, eval3d, global_z_order=False: columns placed exactly at the 1200 face-column azimuths (a
     pinhole column is a constant azimuth when the faces are level; checked), rows every 0.1 deg over +-34 deg
C is sampled at each face pixel's direction (exact column, linear in elevation). C - B isolates the camera model;
C - A is what switching the training path to a panorama would change; B - A is eval3d against the classic rasteriser.

Pass criteria, written before the first run (docs/stage2_notes.md records the run; out of range = reported, never
adjusted):
  shapes        panorama [1, 681, 1200, 1]; faces [4, 200, 300, 1]
  frame         faces level: azimuth spread along every face column < 1e-4 rad
  control 1     one isotropic Gaussian (5 cm, opacity 0.9) 5 m away at azimuth 30, elevation 10 deg: the alpha-
                weighted mean direction in C and in B within 0.05 deg of the truth
  control 2     known failure: C sampled mirrored (azimuth -> -azimuth) against B: median |dB| >= 1 dB (the check
                can see a wrong frame)
  C vs B        opaque pixels (alpha > 0.5 in both): median |dB| <= 0.1, P90 <= 0.5; alpha median |d| <= 0.01;
                seam columns (3 either side of a face edge) median |dB| <= 2 x interior + 0.05
  peaks         every position: the panorama's strongest direction within 0.35 deg of the faces' (B), one pixel
  C vs A        reported with an expected range, not a gate: median |dB| 0.05 - 1.0
  degenerate    the validation position with the smallest opaque share is always included; its C vs B median |dB|
                reported beside the others, and its alpha agreement held to the same 0.01
  timing        reported only: median of 20 after 3 warm-up, A at two face resolutions, C at two panorama grids;
                void under the project's contract when the GPU is shared (nvidia-smi is read at start)

    PYTHONUTF8=1 python rrf_gsplat/smoke_lidar_panorama.py --run output/rrf/m3/rr/plain_s0
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_rrf as T          # noqa: E402
import protocol as PR          # noqa: E402

EL_TOP, EL_STEP = 34.0, 0.1                    # panorama rows, degrees
# gsplat picks the eval3d tile size from the image size (8 below 1080 px), but compute_tiling() packs up to 16 x 16
# rays into one lidar tile: at tile size 8 the kernel rasterises only 64 of each tile's rays and leaves the rest of
# the output uninitialised (NaN, 1e36, different on every call; found by this smoke's first run, 2026-09-24).
# The lidar renders therefore pass tile_size=16 explicitly.
LIDAR_TILE = 16


def load_model(run, dev):
    """The trained field of a finished run (plain / pcolor / lobes), as diag_train_fit.py rebuilds it."""
    res = json.load(open(os.path.join(run, "results.json"))); cfg = res["config"]
    ck = cfg["checkpoint"] if os.path.isabs(cfg["checkpoint"]) else os.path.join(T.REPO, cfg["checkpoint"])
    model = T.RRF(ck, cfg["mode"], 1, cfg["sh_degree"], dev)
    model.sh_backend = cfg.get("sh_backend", "torch")
    st = torch.load(os.path.join(run, "rrf_state.pt"), map_location=dev)
    if "lobe_w" in st:
        model.add_lobes(st["lobe_w"].shape[1], 20.0, torch.zeros(1, 3))
    if "pcolor_mlp" in st:
        c = st["pcolor_cfg"]; model.add_pcolor(c["width"], c["hidden"], c["n_freqs"], torch.zeros(2, 3))
    if "em_means" in st:
        raise SystemExit("emitter runs have a second rasterisation pass; not covered by this smoke")
    model.load_state(st)
    vmin, vmax = res["db_range"]
    return model, cfg, vmax - vmin


def lidar_coeffs(col_az, el_deg, dev):
    from gsplat.cuda._lidar import (RowOffsetStructuredSpinningLidarModelParameters as P, SpinningDirection,
                                    compute_angles_to_columns_map, compute_tiling)
    from gsplat.cuda._wrapper import RowOffsetStructuredSpinningLidarModelParametersExt as PExt
    el = torch.tensor(np.radians(el_deg), dtype=torch.float32, device=dev)
    az = torch.tensor(col_az, dtype=torch.float32, device=dev)
    p = P(row_elevations_rad=el, column_azimuths_rad=az, row_azimuth_offsets_rad=torch.zeros_like(el),
          spinning_frequency_hz=10.0, spinning_direction=SpinningDirection.COUNTER_CLOCKWISE)
    return PExt(p, compute_angles_to_columns_map(p).to(dev), compute_tiling(p))


class Rig:
    """One receiver position: its four faces, the lidar frame (x = face 0 forward, z = up) and the face-pixel
    directions in it."""

    def __init__(self, vms, K, w, h, dev):
        self.vms, self.K, self.w, self.h = vms, K, w, h
        c2w = torch.linalg.inv(vms.double())
        self.centre = c2w[0, :3, 3]
        fwd, down = c2w[0, :3, 2], c2w[0, :3, 1]
        z = -down; x = fwd - (fwd @ z) * z; x = x / x.norm(); z = z / z.norm(); y = torch.linalg.cross(z, x)
        self.R = torch.stack([x, y, z])                                  # world -> lidar rows
        vm = torch.eye(4, dtype=torch.float64, device=dev); vm[:3, :3] = self.R; vm[:3, 3] = -self.R @ self.centre
        self.vm_lidar = vm.float()[None]
        u = (torch.arange(w, device=dev, dtype=torch.float64) + 0.5 - K[0, 2]) / K[0, 0]
        v = (torch.arange(h, device=dev, dtype=torch.float64) + 0.5 - K[1, 2]) / K[1, 1]
        d = torch.stack([u[None, :].expand(h, w), v[:, None].expand(h, w), torch.ones(h, w, dtype=torch.float64, device=dev)], -1)
        d = d / d.norm(dim=-1, keepdim=True)
        dw = torch.einsum("fij,hwj->fhwi", c2w[:, :3, :3], d)           # [4, h, w, 3] world
        dl = torch.einsum("ij,fhwj->fhwi", self.R, dw)
        self.dirs_l = dl
        self.az = torch.atan2(dl[..., 1], dl[..., 0])                    # [4, h, w]
        self.el = torch.rad2deg(torch.asin(dl[..., 2].clamp(-1, 1)))
        self.col_spread = float((self.az - self.az.mean(1, keepdim=True)).abs().max())
        col_az = self.az.mean(1).reshape(-1)                             # [4 * w], one azimuth per face column
        order = torch.argsort(col_az)
        self.col_az = col_az[order].cpu().numpy()
        self.col_index = torch.empty_like(order); self.col_index[order] = torch.arange(order.numel(), device=dev)
        self.col_index = self.col_index.reshape(4, w)                    # face, column -> panorama column


def sample_panorama(pan, rig, el_rows, mirror=False):
    """Panorama [rows, cols] at every face pixel: the face column's own panorama column, linear in elevation."""
    cols = rig.col_index[:, None, :].expand(4, rig.h, rig.w)
    if mirror:
        # azimuth -> -azimuth: the nearest panorama column to the mirrored direction
        tgt = torch.as_tensor(rig.col_az, device=pan.device)
        cols = torch.searchsorted(tgt, (-rig.az).float().contiguous()).clamp(0, tgt.numel() - 1)
    r = (EL_TOP - rig.el) / EL_STEP
    r0 = r.floor().long().clamp(0, len(el_rows) - 2); t = (r - r0).clamp(0, 1).float()
    return pan[r0, cols] * (1 - t) + pan[r0 + 1, cols] * t


def render_all(model, rig, col, span, ext, el_rows, time_it=False):
    from gsplat import rasterization
    m = model
    B = rig.vms.shape[0]
    Ks = rig.K[None].expand(B, 3, 3).float()
    bg = lambda n: torch.zeros(n, col.shape[-1], device=col.device)
    fa = lambda: rasterization(m.means, m.quats, m.scales, m.opacities, col, rig.vms, Ks, rig.w, rig.h, sh_degree=None,
                               backgrounds=bg(B))
    fb = lambda: rasterization(m.means, m.quats, m.scales, m.opacities, col, rig.vms, Ks, rig.w, rig.h, sh_degree=None,
                               backgrounds=bg(B), with_ut=True, with_eval3d=True, packed=False, global_z_order=False)
    fc = lambda: rasterization(m.means, m.quats, m.scales, m.opacities, col, rig.vm_lidar, torch.eye(3, device=col.device)[None],
                               ext.n_columns, ext.n_rows, sh_degree=None, backgrounds=bg(1), camera_model="lidar",
                               lidar_coeffs=ext, with_ut=True, with_eval3d=True, packed=False, global_z_order=False,
                               tile_size=LIDAR_TILE)
    out = {}
    for k, f in (("A", fa), ("B", fb), ("C", fc)):
        img, alpha, _ = f()
        out[k] = (10.0 * torch.log10(img[..., 0] + 1e-12), alpha[..., 0])
    return out


def timed(f, n=20, warm=3):
    for _ in range(warm):
        f()
    torch.cuda.synchronize(); ts = []
    for _ in range(n):
        t0 = time.perf_counter(); f(); torch.cuda.synchronize(); ts.append(time.perf_counter() - t0)
    return float(np.median(ts)) * 1e3


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="output/rrf/m3/rr/plain_s0")
    ap.add_argument("--positions", type=int, default=6, help="validation positions, spread; plus the degenerate one")
    ap.add_argument("--out", default="output/rrf/smoke_lidar_panorama.json")
    a = ap.parse_args()
    dev = "cuda"
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip()
    apps = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"],
                          capture_output=True, text=True).stdout.strip().splitlines()
    others = [l for l in apps if "python" in l.lower() and str(os.getpid()) not in l]
    print(f"GPU: {gpu}; other python processes on it: {others or 'none'}")
    torch.manual_seed(0)
    model, cfg, span = load_model(a.run, dev)
    source = cfg["source"] if os.path.isabs(cfg["source"]) else os.path.join(T.REPO, cfg["source"])
    views = T.read_colmap_text(os.path.join(source, "sparse", "0"))
    names = PR.eval_names(cfg["protocol"], "val")
    vm_all = torch.tensor(np.stack([views[n + ".png"][0] if n + ".png" in views else views[n][0] for n in names]), device=dev)
    K = torch.tensor(views[names[0] + ".png"][1] if names[0] + ".png" in views else views[names[0]][1], device=dev, dtype=torch.float64)
    w, h = (views[names[0] + ".png"] if names[0] + ".png" in views else views[names[0]])[2:4]
    c = torch.linalg.inv(vm_all)[:, :3, 3]
    starts = [0] + [i for i in range(1, len(c)) if float((c[i] - c[i - 1]).norm()) > 1e-4]
    groups = [list(range(s0, s1)) for s0, s1 in zip(starts, starts[1:] + [len(c)])]
    groups = [g for g in groups if len(g) == 4]
    # the degenerate position: the smallest opaque share over all validation positions (classic alpha, cheap)
    from gsplat import rasterization
    shares = []
    with torch.no_grad():
        for g in groups:
            _, al, _ = rasterization(model.means, model.quats, model.scales, model.opacities,
                                     torch.zeros(model.means.shape[0], 1, device=dev), vm_all[g].float(),
                                     K[None].expand(4, 3, 3).float(), w, h, sh_degree=None)
            shares.append(float((al > 0.5).float().mean()))
    worst = int(np.argmin(shares))
    pick = sorted(set(np.linspace(0, len(groups) - 1, a.positions).round().astype(int).tolist()) | {worst})
    print(f"{len(groups)} validation positions; opaque share min {min(shares):.3f} (position {worst}), median {np.median(shares):.3f}; "
          f"picked {pick} (first and last of the sorted list are the val set's extremes)")
    el_rows = np.arange(EL_TOP, -EL_TOP - 1e-9, -EL_STEP)
    res = {"gpu": gpu, "gpu_other_processes": others, "run": a.run, "positions": [], "criteria": {}}
    rig0 = Rig(vm_all[groups[pick[0]]], K, w, h, dev)
    ext = lidar_coeffs(rig0.col_az, el_rows, dev)            # the column azimuths are the same at every level position
    ok = {}

    # control 1: one Gaussian at a known direction
    with torch.no_grad():
        az0, el0 = math.radians(30.0), math.radians(10.0)
        d0 = torch.tensor([math.cos(az0) * math.cos(el0), math.sin(az0) * math.cos(el0), math.sin(el0)], device=dev, dtype=torch.float64)
        mu = (rig0.centre + 5.0 * (rig0.R.T @ d0)).float()[None]
        one = type("G", (), {})()
        one.means, one.quats = mu, torch.tensor([[1.0, 0, 0, 0]], device=dev)
        one.scales, one.opacities = torch.full((1, 3), 0.05, device=dev), torch.tensor([0.9], device=dev)
        r1 = render_all(one, rig0, torch.ones(1, 1, device=dev), span, ext, el_rows)
        errs = {}
        # panorama: directions of its grid; faces: rig0.dirs_l
        az_g = torch.as_tensor(rig0.col_az, device=dev, dtype=torch.float64)[None, :]
        el_g = torch.as_tensor(np.radians(el_rows), device=dev, dtype=torch.float64)[:, None]
        dir_pan = torch.stack([torch.cos(az_g) * torch.cos(el_g), torch.sin(az_g) * torch.cos(el_g), torch.sin(el_g).expand_as(az_g * el_g)], -1)
        for k, (dirs, al) in {"C": (dir_pan, r1["C"][1][0].double()), "B": (rig0.dirs_l, r1["B"][1].double()),
                              "A": (rig0.dirs_l, r1["A"][1].double())}.items():
            mvec = (dirs * al[..., None]).reshape(-1, 3).sum(0); mvec = mvec / mvec.norm()
            errs[k] = math.degrees(math.acos(float((mvec @ d0).clamp(-1, 1))))
    ok["control1"] = errs["C"] <= 0.05 and errs["B"] <= 0.05
    res["criteria"]["control1_deg"] = errs
    print(f"control 1 (one Gaussian at az 30, el 10): direction error C {errs['C']:.4f} deg, B {errs['B']:.4f}, A {errs['A']:.4f} "
          f"-> {'PASS' if ok['control1'] else 'FAIL'} (<= 0.05 for C and B)")

    rows = []
    with torch.no_grad():
        for p in pick:
            rig = Rig(vm_all[groups[p]], K, w, h, dev)
            col = torch.pow(10.0, model.colours(rig.centre.float()).clamp(0.0, 1.0) * span / 10.0)
            r = render_all(model, rig, col, span, ext, el_rows)
            (dA, aA), (dB, aB), (dC, aC) = r["A"], r["B"], r["C"]
            sC = sample_panorama(dC[0], rig, el_rows); saC = sample_panorama(aC[0], rig, el_rows)
            sCm = sample_panorama(dC[0], rig, el_rows, mirror=True)
            op = (aB > 0.5) & (saC > 0.5)
            seam = torch.zeros(w, dtype=torch.bool, device=dev); seam[:3] = True; seam[-3:] = True
            seam = seam[None, None, :].expand_as(op)
            dcb = (sC - dB).abs(); dca = (sC - dA).abs(); dba = (dB - dA).abs()
            # the strongest direction: panorama argmax vs faces (B) argmax, over the face pixels' extent
            ip = int(torch.argmax(torch.where(aC[0] > 0.5, dC[0], torch.full_like(dC[0], -1e9))))
            pr, pc = divmod(ip, dC.shape[-1])
            dpan = torch.tensor([math.cos(rig.col_az[pc]) * math.cos(math.radians(el_rows[pr])),
                                 math.sin(rig.col_az[pc]) * math.cos(math.radians(el_rows[pr])), math.sin(math.radians(el_rows[pr]))],
                                device=dev, dtype=torch.float64)
            ib = int(torch.argmax(torch.where(aB > 0.5, dB, torch.full_like(dB, -1e9))))
            dfb = rig.dirs_l.reshape(-1, 3)[ib]
            peak = math.degrees(math.acos(float((dpan @ dfb).clamp(-1, 1))))
            row = {"position": p, "degenerate": p == worst, "opaque_share": float(op.float().mean()),
                   "col_spread_rad": rig.col_spread,
                   "CB_median_db": float(dcb[op].median()), "CB_p90_db": float(dcb[op].quantile(0.9)),
                   "CB_seam_median_db": float(dcb[op & seam].median()), "CB_interior_median_db": float(dcb[op & ~seam].median()),
                   "CB_alpha_median": float((saC - aB).abs().median()),
                   "CB_mirror_median_db": float((sCm - dB).abs()[op].median()),
                   "CA_median_db": float(dca[op].median()), "CA_p90_db": float(dca[op].quantile(0.9)),
                   "BA_median_db": float(dba[op].median()), "BA_p90_db": float(dba[op].quantile(0.9)),
                   "peak_C_vs_B_deg": peak}
            rows.append(row)
            print(f"  pos {p:3d}{' (degenerate)' if p == worst else '':13s} opaque {row['opaque_share']:.3f}  C-B median {row['CB_median_db']:.3f} "
                  f"P90 {row['CB_p90_db']:.3f} dB (seam {row['CB_seam_median_db']:.3f} / interior {row['CB_interior_median_db']:.3f}), "
                  f"alpha {row['CB_alpha_median']:.4f}, mirrored {row['CB_mirror_median_db']:.2f}; C-A {row['CA_median_db']:.3f} / "
                  f"{row['CA_p90_db']:.3f}; B-A {row['BA_median_db']:.3f} / {row['BA_p90_db']:.3f}; peak {peak:.3f} deg; col spread {rig.col_spread:.1e}")
        res["positions"] = rows
        # shapes, from the last position
        shp = {"panorama": list(dC.shape), "faces": list(dA.shape)}
    g = lambda k: [r[k] for r in rows]
    ok["shapes"] = shp["panorama"] == [1, len(el_rows), 4 * w] and shp["faces"] == [4, h, w]
    ok["frame"] = max(g("col_spread_rad")) < 1e-4
    ok["control2"] = min(g("CB_mirror_median_db")) >= 1.0
    ok["CB"] = max(g("CB_median_db")) <= 0.1 and max(g("CB_p90_db")) <= 0.5 and max(g("CB_alpha_median")) <= 0.01
    ok["seam"] = all(r["CB_seam_median_db"] <= 2 * r["CB_interior_median_db"] + 0.05 for r in rows)
    ok["peaks"] = max(g("peak_C_vs_B_deg")) <= 0.35
    ca = float(np.median(g("CA_median_db")))
    res["CA_in_expected_range"] = 0.05 <= ca <= 1.0
    print(f"shapes {shp} -> {'PASS' if ok['shapes'] else 'FAIL'}; faces level (max col spread {max(g('col_spread_rad')):.1e}) -> {'PASS' if ok['frame'] else 'FAIL'}")
    print(f"control 2 (mirrored): min median {min(g('CB_mirror_median_db')):.2f} dB -> {'PASS' if ok['control2'] else 'FAIL'} (>= 1)")
    print(f"C vs B: max median {max(g('CB_median_db')):.3f} dB (<= 0.1), max P90 {max(g('CB_p90_db')):.3f} (<= 0.5), "
          f"max alpha {max(g('CB_alpha_median')):.4f} (<= 0.01) -> {'PASS' if ok['CB'] else 'FAIL'}; seams -> {'PASS' if ok['seam'] else 'FAIL'}")
    print(f"peaks: max {max(g('peak_C_vs_B_deg')):.3f} deg (<= 0.35) -> {'PASS' if ok['peaks'] else 'FAIL'}")
    print(f"C vs A (reported): median over positions {ca:.3f} dB, expected 0.05-1.0 -> {'in range' if res['CA_in_expected_range'] else 'OUT OF RANGE'}")

    # timing: one position, A at two face sizes, C at two panorama grids
    rig = Rig(vm_all[groups[pick[0]]], K, w, h, dev)
    col = torch.pow(10.0, model.colours(rig.centre.float()).clamp(0.0, 1.0) * span / 10.0).detach()
    from gsplat import rasterization
    m = model
    tm = {}
    with torch.no_grad():
        for s in (1, 2):
            # the face at s times the resolution: focal length and principal point scale, the field of view does not
            Ks_s = K * torch.tensor([[s, 1, s], [1, s, s], [1, 1, 1]], device=dev, dtype=K.dtype)
            Ks = Ks_s[None].expand(4, 3, 3).float()
            tm[f"A_4faces_{w * s}x{h * s}_ms"] = timed(lambda: rasterization(m.means, m.quats, m.scales, m.opacities, col, rig.vms, Ks,
                                                                             w * s, h * s, sh_degree=None))
            # the panorama at the same angular sampling: one column per face column, rows at the face's pixel pitch
            # at its centre (90 deg / w)
            el_t = np.arange(EL_TOP, -EL_TOP - 1e-9, -90.0 / (w * s))
            ext_t = lidar_coeffs(Rig(rig.vms, Ks_s, w * s, h * s, dev).col_az, el_t, dev)
            tm[f"C_panorama_{ext_t.n_columns}x{ext_t.n_rows}_ms"] = timed(lambda: rasterization(
                m.means, m.quats, m.scales, m.opacities, col, rig.vm_lidar, torch.eye(3, device=dev)[None], ext_t.n_columns, ext_t.n_rows,
                sh_degree=None, camera_model="lidar", lidar_coeffs=ext_t, with_ut=True, with_eval3d=True, packed=False, global_z_order=False,
                tile_size=LIDAR_TILE))
    res["timing_ms"] = tm; res["timing_valid"] = not others
    print(f"timing (median of 20 after 3 warm-up; {model.means.shape[0]:,} Gaussians; rasterisation only, colours precomputed): "
          + ", ".join(f"{k} {v:.2f}" for k, v in tm.items())
          + ("" if not others else "  -- VOID: the GPU was shared"))
    res["pass"] = ok
    json.dump(res, open(a.out, "w"), indent=1)
    print(f"-> {a.out}; {'ALL PASS' if all(ok.values()) else 'FAILED: ' + ', '.join(k for k, v in ok.items() if not v)}")


if __name__ == "__main__":
    main()
