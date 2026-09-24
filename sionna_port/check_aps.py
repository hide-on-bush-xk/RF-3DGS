"""Stage 2 smoke: is the instrument-free angular power spectrum (APS) clean enough to be a target?

Two generations of the same positions that differ only in the solver's sampling lattice (--seed 42 / 43 with the
same --poses-from, iso receive pattern, 4M samples) are compared view by view. The positions are a stratified subset
of protocol_v1's poses (make_smoke_poses): spread along the route with both ends, all four faces each, the most
shadowed position of the MVDR truth, and at least one position of every protocol set.

Criteria, written before the first run (docs: the stage-2 plan agreed on 2026-09-24):
  a  seed-to-seed RMSE on the lit pixels (> N0 + 3 dB in either seed), median over views <= 1 dB
  b  main-peak direction difference <= 1 deg in >= 90 % of views
  c  no instrument left: the columns where adjacent faces meet (+-45 deg) agree, median |difference| <= 1 dB
     (with a tr38901 receive pattern the faces cut from one solve differ by up to ~26 dB)
  d  reported, not judged: the share of pixels within 1 dB of the soft floor, and the generation time
The smoke says nothing about how well any model fits this target.

    python sionna_port/check_aps.py --a <dataset seed 42> --b <dataset seed 43>
    python sionna_port/check_aps.py --make-poses <reference dataset> --protocol rrf_gsplat/protocol_v1 --out <dir>
    python sionna_port/check_aps.py --control <APS dataset>        (one known path must come back at its direction)

The noise-floor row of stage 2 (a second draw of the validation positions, scored like a predictor):
    python sionna_port/check_aps.py --make-poses <APS dataset> --protocol rrf_gsplat/protocol_v1 --sets val_random val_segment --out <dir>
    (generate_dataset.py --poses-from <dir>/sparse/0/images.txt --seed 43 --aps-floor-fixed-db <the reference's N0>)
    python sionna_port/check_aps.py --draw-as-run <draw> <dir>/names.txt <run>
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(REPO, "rrf_gsplat"))


def make_smoke_poses(reference, protocol, out, n_spread=10):
    """A sparse/0/images.txt holding a stratified subset of the reference's positions (see the module docstring)."""
    import protocol as PR
    lines = [l for l in open(os.path.join(reference, "sparse", "0", "images.txt")) if l.split() and l.split()[-1].endswith(".png")]
    names = [l.split()[9][:-4] for l in lines]
    n_pos = len(names) // 4
    pick = set(np.linspace(0, n_pos - 1, n_spread).round().astype(int).tolist())
    peak = [max(float(np.load(os.path.join(reference, "spectra_float", names[4 * p + k] + ".npy")).max()) for k in range(4))
            for p in range(n_pos)]
    pick.add(int(np.argmin(peak)))
    for s in ("train", "val_random", "val_segment", "test_interp", "test_region"):
        members = set(PR.read_set(protocol, s))
        if not any(names[4 * p] in members for p in pick):
            pick.add(next(p for p in range(n_pos) if names[4 * p] in members))
    os.makedirs(os.path.join(out, "sparse", "0"), exist_ok=True)
    with open(os.path.join(out, "sparse", "0", "images.txt"), "w") as f:
        for p in sorted(pick):
            for k in range(4):
                f.write(lines[4 * p + k].rstrip("\n") + "\n\n")
    print(f"{len(pick)} positions -> {out}/sparse/0/images.txt: {sorted(pick)} (weakest {int(np.argmin(peak))})")


def single_path_control(dataset, trials=25):
    """The analytic control: one path from a known world direction -> aps_splat -> cut_face -> the argmax pixel, taken
    back to the world through the dataset's own COLMAP pose and pixel_dirs, must be that direction. Directions are
    random pixel centres >= 10 px inside each of the first position's four faces. Criterion (written before the
    first run): max error <= 0.5 deg (a face pixel is 0.3 deg, an equirect pixel 1/3 deg), and the opposite face
    stays > 20 dB below. Runs on the CPU."""
    import types
    import torch
    from generate_dataset import VIEW_YAWS, cut_face
    from mvdr_peaks import pixel_dirs
    from rf_spectra import aps_splat
    dirs = pixel_dirs(dataset); H, W = dirs.shape[:2]
    cfg = types.SimpleNamespace(spectrum="APS", width=W, height=H, fov_deg=90.0)

    def rotmat(q):
        w, x, y, z = q
        return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                         [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                         [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    lines = [l.split() for l in open(os.path.join(dataset, "sparse", "0", "images.txt")) if len(l.split()) >= 10]
    rng = np.random.default_rng(0)
    errs, leaks = [], 0
    for f, yaw in zip(lines[:4], VIEW_YAWS):                  # the generator writes a position's faces in VIEW_YAWS order
        R = rotmat([float(v) for v in f[1:5]])                  # world -> camera
        for _ in range(trials):
            i, j = rng.integers(10, H - 10), rng.integers(10, W - 10)
            d = R.T @ dirs[i, j]
            th, ph = np.arccos(np.clip(d[2], -1, 1)), np.arctan2(d[1], d[0])
            eq = aps_splat(torch.tensor([th], dtype=torch.float64), torch.tensor([ph], dtype=torch.float64),
                           torch.tensor([1.0], dtype=torch.float64))
            face = cut_face(eq, cfg, yaw).numpy()
            leaks += int(cut_face(eq, cfg, yaw + np.pi).numpy().max() > face.max() - 20)
            d_hat = R.T @ dirs.reshape(-1, 3)[int(face.argmax())]
            errs.append(float(np.degrees(np.arccos(np.clip(d_hat @ d, -1, 1)))))
    errs = np.array(errs)
    ok = errs.max() <= 0.5 and leaks == 0
    print(f"single-path control, {len(errs)} paths: error median {np.median(errs):.3f} max {errs.max():.3f} deg, "
          f"opposite face within 20 dB: {leaks} -> {'PASS' if ok else 'FAIL'}")
    return ok


def make_set_poses(reference, protocol, sets, out):
    """The poses of the named protocol sets only (whole positions, dataset order) for a second draw, plus names.txt:
    line k = the reference's name of the draw's view k (the generator numbers its views from 0)."""
    import protocol as PR
    members = set(n for s in sets for n in PR.read_set(protocol, s))
    lines = [l for l in open(os.path.join(reference, "sparse", "0", "images.txt")) if l.split() and l.split()[-1].endswith(".png")]
    keep = [l for l in lines if l.split()[9][:-4] in members]
    os.makedirs(os.path.join(out, "sparse", "0"), exist_ok=True)
    with open(os.path.join(out, "sparse", "0", "images.txt"), "w") as f:
        f.writelines(l.rstrip("\n") + "\n\n" for l in keep)
    with open(os.path.join(out, "names.txt"), "w") as f:
        f.writelines(l.split()[9][:-4] + "\n" for l in keep)
    print(f"{len(keep)} views of {'+'.join(sets)} -> {out}")


def draw_as_run(draw, names_txt, run):
    """A second draw's spectra under the reference's view names, in a run's layout (<run>/renders/<name>.npy), so
    mvdr_peaks.py / t4_score.py score it like any predictor: the noise floor of the target."""
    names = [l.strip() for l in open(names_txt) if l.strip()]
    # the generator numbers its views from 1 (COLMAP image ids); check every view's pose against the poses file the
    # draw was made from, so a numbering slip cannot pair a view with another position's truth
    pose = lambda f: {l.split()[9][:-4]: np.array(l.split()[1:8], dtype=np.float64)          # noqa: E731
                      for l in open(f) if len(l.split()) >= 10}
    drawn = pose(os.path.join(draw, "sparse", "0", "images.txt"))
    wanted = pose(os.path.join(os.path.dirname(names_txt), "sparse", "0", "images.txt"))
    os.makedirs(os.path.join(run, "renders"), exist_ok=True)
    for k, n in enumerate(names):
        if np.abs(drawn[f"{k + 1:05d}"] - wanted[n]).max() > 1e-6:
            raise SystemExit(f"draw view {k + 1:05d} does not have the pose of {n}")
        x = np.load(os.path.join(draw, "spectra_float", f"{k + 1:05d}.npy")).astype(np.float32)
        np.save(os.path.join(run, "renders", n + ".npy"), x)
    print(f"{len(names)} views of {draw} -> {run}/renders")


def compare(a, b):
    from mvdr_peaks import angle, pixel_dirs
    ma, mb = (json.load(open(os.path.join(d, "generation_meta.json"))) for d in (a, b))
    n0 = max(ma["aps_floor_db"], mb["aps_floor_db"])
    dirs = pixel_dirs(a)
    names = sorted(f[:-4] for f in os.listdir(os.path.join(a, "spectra_float")))
    rmse, ang, seam, floor = [], [], [], []
    faces = {}
    for n in names:
        x = np.load(os.path.join(a, "spectra_float", n + ".npy")).astype(np.float64)
        y = np.load(os.path.join(b, "spectra_float", n + ".npy")).astype(np.float64)
        lit = (x > n0 + 3) | (y > n0 + 3)
        rmse.append(float(np.sqrt(np.mean((x[lit] - y[lit]) ** 2))) if lit.any() else 0.0)
        ang.append(angle(dirs, int(x.argmax()), int(y.argmax())))
        floor.append(float((x < ma["aps_floor_db"] + 1).mean()))
        faces[n] = x
    # adjacent faces of one position: face k's right edge meets face k+1's left edge (yaws -90, 0, 90, 180)
    for i in range(0, len(names) - 3, 4):
        f = [faces[names[i + k]] for k in range(4)]
        for k in range(4):
            l, r = f[k], f[(k + 1) % 4]
            seam.append(float(np.median(np.abs(l[:, 0] - r[:, -1]))))    # phi runs +45 -> -45 left to right
    rmse, ang, seam = np.array(rmse), np.array(ang), np.array(seam)
    res = {"views": len(names), "n0_db": [ma["aps_floor_db"], mb["aps_floor_db"]],
           "a_rmse_lit_median_db": float(np.median(rmse)), "a_rmse_lit_p90_db": float(np.percentile(rmse, 90)),
           "b_peak_within_1deg": float((ang <= 1).mean()), "b_peak_angle_median": float(np.median(ang)),
           "c_seam_median_db": float(np.median(seam)), "c_seam_p90_db": float(np.percentile(seam, 90)),
           "d_floor_share_median": float(np.median(floor)), "d_seconds": [ma.get("seconds"), mb.get("seconds")]}
    res["pass"] = {"a": res["a_rmse_lit_median_db"] <= 1.0, "b": res["b_peak_within_1deg"] >= 0.90, "c": res["c_seam_median_db"] <= 1.0}
    print(json.dumps(res, indent=1))
    print("ALL PASS" if all(res["pass"].values()) else "FAIL: " + ", ".join(k for k, v in res["pass"].items() if not v))
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a"); ap.add_argument("--b")
    ap.add_argument("--make-poses"); ap.add_argument("--protocol"); ap.add_argument("--out")
    ap.add_argument("--sets", nargs="+", default=None, help="with --make-poses: these protocol sets' poses only")
    ap.add_argument("--draw-as-run", nargs=3, metavar=("DRAW", "NAMES_TXT", "RUN"))
    ap.add_argument("--control", metavar="DATASET", help="the single-path analytic control on this dataset's poses")
    a = ap.parse_args()
    if a.control:
        sys.exit(0 if single_path_control(a.control) else 1)
    elif a.draw_as_run:
        draw_as_run(*a.draw_as_run)
    elif a.make_poses and a.sets:
        make_set_poses(a.make_poses, a.protocol, a.sets, a.out)
    elif a.make_poses:
        make_smoke_poses(a.make_poses, a.protocol, a.out)
    else:
        res = compare(a.a, a.b)
        json.dump(res, open(os.path.join(REPO, "output", "rrf", "check_aps.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
