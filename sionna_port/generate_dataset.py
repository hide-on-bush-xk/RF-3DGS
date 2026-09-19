"""Generate an RF-3DGS spectrum dataset with Sionna RT 2.x.

Ported from `RF-3DGS-tutorial.ipynb`, which targets Sionna 0.19 and TensorFlow.
Two things changed beyond the API translation, both flagged in README.md:

* every spectrum type is normalised the same way, and
* the float spectrum is written next to the PNG, so the quantisation is no
  longer the only surviving copy of the data.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
import dataclasses
from dataclasses import dataclass, asdict
from datetime import datetime

import numpy as np
import torch

from rf_metrics import channel_metrics, frequency_response, spectrum_stats
from rf_spectra import ArrayGrid, cbf_spectrum, mvdr_spectrum, paths_to_response

# Sionna RT 2.x. The 0.19 imports (sionna.rt.antenna, sionna.channel) are gone;
# see README.md for the full mapping.
from sionna.rt import (load_scene, PathSolver, PlanarArray, Receiver,
                       Transmitter)
from sionna.rt.antenna_pattern import v_tr38901_pattern
from scipy.spatial.transform import Rotation

# The four yaw angles the tutorial samples at each receiver position; together
# with a 90 degree FoV they tile the azimuth circle.
VIEW_YAWS = (-math.pi / 2, 0.0, math.pi / 2, math.pi)


@dataclass
class Config:
    scene_xml: str
    rx_loc_file: str
    out_dir: str
    spectrum: str = "MVDR"          # CBF | MVDR
    tx_loc: tuple = (6.905, 0.0, 2 - 1.713)
    num_positions: int = 800
    M: int = 10                     # M x M UPA
    width: int = 300
    height: int = 200
    fov_deg: float = 90.0
    frequency: float = 60e9
    time_interval_ns: float = 0.1
    max_depth: int = 1
    samples_per_src: int = 1_000_000
    specular_reflection: bool = True
    diffuse_reflection: bool = True
    refraction: bool = False
    diffraction: bool = False
    seed: int = 42
    save_float: bool = True
    diagonal_loading: float = 0.0
    # ITU materials load with scattering_coefficient = 0, so diffuse_reflection
    # produces nothing at all and a depth-1 solve yields single-digit path
    # counts. 0.7 reproduces the "more than 300,000 MPCs per Tx-Rx pair" the
    # paper reports at 60 GHz: it gives 312,683 at depth 1. See
    # calibrate_paths.py for the sweep this came from.
    scattering_coefficient: float = 0.7
    materials: str = "uniform"      # uniform | tutorial | tutorial-asis
    poses_from: str = None          # images.txt of a dataset whose exact poses to reuse
    splat_sigma: float = 3.0        # projection family: splat kernel width in equirect pixels
    power_floor_db: float = -150.0  # projection family: splat power under this is a kernel tail, not a path
    # Cheap views, decorrelated between views, and let the fit do the averaging.
    #
    # Sionna's solver shoots samples_per_src rays from a fixed lattice, so a low
    # budget leaves visible sampling structure -- and reusing one seed makes that
    # structure identical in every view, which is exactly the kind of error a
    # multi-view fit cannot average away: it looks like a consistent feature of
    # the field. Advancing the seed per view moves the lattice instead, so the
    # residual is independent between views and the radiance field accumulates
    # towards the high-sample answer while each view stays cheap.
    per_view_seed: bool = True
    seed_per_view: bool = False     # pre-fix behaviour: every view (not position) gets its own lattice and solve; for A/B only
    bandwidth_hz: float = 400e6     # link budget only; scales SNR and capacity
    dashboard: bool = True


# --------------------------------------------------------------------------
# COLMAP export, unchanged from the tutorial apart from being importable
# --------------------------------------------------------------------------

def euler_to_quaternion(euler):
    """Sionna receiver orientation -> COLMAP camera-to-world quaternion."""
    r_posz2posx = Rotation.from_euler("ZYX", [-np.pi / 2, 0.0, -np.pi / 2])
    yaw, pitch, roll = euler
    r_posx2array = Rotation.from_euler("ZYX", [yaw, pitch, roll])
    r_w2c = r_posx2array * r_posz2posx
    r_c2w = r_w2c.inv()
    q = r_c2w.as_quat()
    return r_c2w, [q[3], q[0], q[1], q[2]]      # COLMAP wants scalar first


def write_cameras_txt(path, camera_id, width, height, fx, fy, cx, cy):
    with open(path, "w") as fid:
        fid.write(f"{camera_id} PINHOLE {width} {height} {fx} {fy} {cx} {cy}\n")


def write_images_txt(path, images):
    with open(path, "w") as fid:
        for img_id, (qvec, tvec, camera_id, name) in sorted(images.items()):
            fid.write(f"{img_id} {' '.join(map(str, qvec))} "
                      f"{' '.join(map(str, tvec))} {camera_id} {name}\n\n")


def read_pose_groups(images_txt):
    """Receiver positions and yaws from a dataset's COLMAP images.txt.

    The file stores the camera-to-world rotation and t = -R_c2w(rx), as the
    tutorial writes it, so rx = -R^T t; the yaw is whichever of VIEW_YAWS
    reproduces the rotation. Consecutive views at one position form a group.
    """
    def rotmat(q):
        w, x, y, z = q
        return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                         [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                         [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    yaw_R = {yaw: euler_to_quaternion([yaw, 0.0, 0.0])[0].as_matrix() for yaw in VIEW_YAWS}
    groups = []
    with open(images_txt) as fid:
        for line in fid:
            p = line.split()
            if len(p) < 10 or not p[9].lower().endswith(".png"):
                continue
            R = rotmat([float(v) for v in p[1:5]])
            t = np.array([float(v) for v in p[5:8]])
            rx = (-R.T @ t).tolist()
            yaw = min(VIEW_YAWS, key=lambda y: np.abs(yaw_R[y] - R).sum())
            if groups and np.allclose(groups[-1][0], rx, atol=1e-6):
                groups[-1][1].append(yaw)
            else:
                groups.append((rx, [yaw]))
    return groups


def load_rx_locations(path, rng):
    """Positions along the NIST sampling route, jittered as in the tutorial."""
    locs = []
    with open(path) as fid:
        for line in fid:
            parts = line.split()
            if len(parts) != 4:
                continue
            x_pos, y_pos = float(parts[1]), float(parts[2])
            locs.append([x_pos / 1000 + rng.uniform(-0.5, 0.5),
                         y_pos / 1000 + rng.uniform(-0.5, 0.5),
                         1.625 - 1.713 + rng.uniform(-1.0, 0.3)])
    return locs


# --------------------------------------------------------------------------
# Sionna 2.x scene and solver
# --------------------------------------------------------------------------

def build_scene(cfg: Config):
    scene = load_scene(cfg.scene_xml, merge_shapes=True)
    scene.frequency = cfg.frequency
    scene.tx_array = PlanarArray(num_rows=1, num_cols=1,
                                 pattern="tr38901", polarization="V")
    # The beamforming maths in rf_spectra assumes this element order and a
    # half-wavelength UPA; keep the two in step if you change either.
    scene.rx_array = PlanarArray(num_rows=cfg.M, num_cols=cfg.M,
                                 vertical_spacing=0.5, horizontal_spacing=0.5,
                                 pattern="tr38901", polarization="V")

    if cfg.materials != "uniform":
        # the notebook's per-material definitions replace the ITU placeholders
        import tutorial_materials
        tutorial_materials.apply(scene, cfg.frequency,
                                 variant="asis" if cfg.materials == "tutorial-asis" else "fixed")
    elif cfg.scattering_coefficient > 0:
        for material in scene.radio_materials.values():
            material.scattering_coefficient = cfg.scattering_coefficient
    return scene


def solve_paths(solver, scene, cfg: Config, view_index: int = 0):
    """0.19's scene.compute_paths, expressed with the 2.x PathSolver.

    reflection -> specular_reflection, scattering -> diffuse_reflection,
    num_samples -> samples_per_src. scat_keep_prob and scat_random_phases have
    no 2.x equivalent: the rewritten solver samples diffuse paths itself.
    """
    return solver(scene=scene,
                  max_depth=cfg.max_depth,
                  samples_per_src=cfg.samples_per_src,
                  los=True,
                  specular_reflection=cfg.specular_reflection,
                  diffuse_reflection=cfg.diffuse_reflection,
                  refraction=cfg.refraction,
                  diffraction=cfg.diffraction,
                  synthetic_array=True,
                  seed=cfg.seed + (view_index if cfg.per_view_seed else 0))


def element_gain_fn(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    """Co-polar TR 38.901 element response on the pixel grid.

    0.19's sionna.rt.antenna.tr38901_pattern returned a (c_theta, c_phi) pair and
    the tutorial kept c_theta. The 1.2+ replacement returns a single Complex2f,
    which is already that co-polar component -- unpacking it as a pair silently
    yields the real and imaginary parts instead.
    """
    import drjit as dr
    import mitsuba as mi
    t = mi.Float(theta.reshape(-1).cpu().numpy())
    p = mi.Float(phi.reshape(-1).cpu().numpy())
    c_theta = v_tr38901_pattern(t, p)
    vals = np.asarray(dr.real(c_theta)) + 1j * np.asarray(dr.imag(c_theta))
    return torch.as_tensor(vals, dtype=torch.complex64,
                           device=theta.device).reshape(theta.shape)


PROJECTION_KINDS = ("MULTI", "AOD3")
# A receiver pose the ray tracer cannot reach at all (a closed room, a
# transmitter behind two walls at depth 1) gives no paths. The projection
# targets already render such a view as all-floor; the beamformed spectra
# raised and killed the run (scene 2, room transmitters). They now get a
# constant spectrum at this level, below the MVDR numerical floor (about
# -280 dB), and the count is recorded in generation_meta.json.
EMPTY_VIEW_DB = -300.0


def projection_equirect(paths, kind: str, device, sigma: float = 3.0, power_floor_db: float = -150.0):
    """The projection family's whole-sphere image for one receiver position.

    Splatted at the angle of arrival in the world frame, so it does not
    depend on the receiver's yaw: one solve per position, and the four
    pinhole faces are cut from this one image. `sigma` is the splat kernel
    in equirect pixels (1/3 degree each): it sets how many paths share a
    pixel, i.e. how much the encoded mean angle departs from any real path.
    """
    from rf_spectra import aod_spectrum_equirect, multichannel_spectrum_equirect
    eq = (multichannel_spectrum_equirect(paths, sigma=sigma, power_floor_db=power_floor_db) if kind == "MULTI"
          else aod_spectrum_equirect(paths, sigma=sigma))
    return eq.to(device)


def spectrum_for_paths(paths, grid: ArrayGrid, cfg: Config, yaw: float = 0.0):
    kind = cfg.spectrum.upper()
    if kind in PROJECTION_KINDS:
        from rf_spectra import equirect_to_perspective
        eq = projection_equirect(paths, kind, grid.theta.device, cfg.splat_sigma, cfg.power_floor_db)
        persp = equirect_to_perspective(eq, cfg.width, cfg.height, cfg.fov_deg, yaw_rad=yaw)
        return None, persp                                   # [C, H, W]
    response = paths_to_response(paths, cfg.time_interval_ns, device=grid.theta.device)
    if kind == "CBF":
        return cbf_spectrum(response, grid)
    if kind == "MVDR":
        return mvdr_spectrum(response, grid, cfg.diagonal_loading)
    raise ValueError(f"unknown spectrum type {cfg.spectrum!r}")


def channel_ranges(all_db: np.ndarray, kind: str, power_floor_db: float = -150.0):
    """Per-channel (min, max) for a multi-channel dataset, [C, 2]."""
    if kind == "MULTI":
        power = all_db[:, 0]
        # Pixels a path reaches. The pinhole resampling blends the -200 dB
        # floor into edge pixels, so "above the floor" is not enough; nothing
        # physical sits below -150 dB here.
        hit = power > power_floor_db     # the encoder truncated there; a face pixel resampled across the hit/floor edge is not a hit
        rng = [[float(np.percentile(power[hit], 1)), float(np.percentile(power[hit], 99.99))],
               [-1.0, 1.0], [-1.0, 1.0], [0.0, 1.0],
               [float(np.percentile(all_db[:, 4][hit], 0.1)), float(np.percentile(all_db[:, 4][hit], 99.9))]]
        return rng
    lo, hi = float(all_db.min()), float(all_db.max())          # AOD3: one range, it is a picture
    return [[lo, hi]] * all_db.shape[1]


# --------------------------------------------------------------------------
# Dataset generation
# --------------------------------------------------------------------------

def generate(cfg: Config, shared=None):
    """One dataset. `shared` = (scene, solver, grid) built once by the caller
    when several transmitters are generated in one process (--tx-list): the
    scene load, the OptiX acceleration structure and the array grid are the
    same for every transmitter, only the Tx object moves."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(cfg.seed)

    os.makedirs(os.path.join(cfg.out_dir, "images"), exist_ok=True)
    if cfg.save_float:
        os.makedirs(os.path.join(cfg.out_dir, "spectra_float"), exist_ok=True)
    os.makedirs(os.path.join(cfg.out_dir, "sparse", "0"), exist_ok=True)

    if shared is None:
        scene = build_scene(cfg)
        solver = PathSolver()
        grid = ArrayGrid.build(cfg.M, cfg.width, cfg.height, cfg.fov_deg,
                               element_gain_fn=element_gain_fn, device=device)
    else:
        scene, solver, grid = shared
    if "tx" in scene.transmitters:
        # move the existing transmitter (as the planner does) rather than
        # remove / add: editing the scene graph invalidates the solver's
        # cached structures and every later solve pays for it
        scene.get("tx").position = [float(v) for v in cfg.tx_loc]
    else:
        scene.add(Transmitter(name="tx", position=list(cfg.tx_loc)))

    if cfg.poses_from:
        # exact poses of an existing dataset (e.g. the released one), grouped
        # by position so the loop below is unchanged
        groups = read_pose_groups(cfg.poses_from)[:cfg.num_positions]
        print(f"poses from {cfg.poses_from}: {sum(len(y) for _, y in groups)} views")
    else:
        rx_locs = load_rx_locations(cfg.rx_loc_file, rng)[:cfg.num_positions]
        groups = [(rx_loc, list(VIEW_YAWS)) for rx_loc in rx_locs]

    # Pass 1: collect every spectrum in dB, so the normalisation range comes from
    # the data instead of two hand-picked probe positions.
    print(f"pass 1/2: {len(groups)} positions x {len(VIEW_YAWS)} views")
    specs, poses, per_view, cfr_curve = [], [], [], None
    t_start = time.time()
    projection = cfg.spectrum.upper() in PROJECTION_KINDS
    empty_views = 0
    for i, (rx_loc, yaws) in enumerate(groups):
        if i % 50 == 0:
            print(f"  {i}/{len(groups)}")
        eq = None
        for j, yaw in enumerate(yaws):
            if projection and eq is not None and not cfg.seed_per_view:
                # same position, same paths: cut the next face from the sphere
                from rf_spectra import equirect_to_perspective
                spec_db = equirect_to_perspective(eq, cfg.width, cfg.height, cfg.fov_deg, yaw_rad=yaw)
            else:
                scene.remove("rx") if "rx" in scene.receivers else None
                scene.add(Receiver(name="rx", position=list(rx_loc),
                                   orientation=[yaw, 0.0, 0.0]))
                # The sampling lattice is seeded per *position*: the four faces
                # of one sphere must come from one path set, or a path near a
                # face boundary can exist in one face and be missing in the
                # next. Randomising the lattice across positions still gives the
                # multi-view fit independent residuals to average.
                paths = solve_paths(solver, scene, cfg, view_index=(i * len(yaws) + j) if cfg.seed_per_view else i)
                if projection:
                    eq = projection_equirect(paths, cfg.spectrum.upper(), grid.theta.device, cfg.splat_sigma, cfg.power_floor_db)
                    from rf_spectra import equirect_to_perspective
                    spec_db = equirect_to_perspective(eq, cfg.width, cfg.height, cfg.fov_deg, yaw_rad=yaw)
                else:
                    try:
                        _, spec_db = spectrum_for_paths(paths, grid, cfg, yaw=yaw)
                    except ValueError as exc:
                        if "no valid paths" not in str(exc):
                            raise
                        spec_db = torch.full(tuple(grid.theta.shape[-2:]), EMPTY_VIEW_DB, dtype=torch.float32)
                        empty_views += 1
            specs.append(spec_db.cpu().numpy().astype(np.float32))
            poses.append((rx_loc, yaw))

            if cfg.dashboard:
                try:
                    per_view.append(channel_metrics(
                        paths, bandwidth_hz=cfg.bandwidth_hz).as_dict())
                except ValueError:
                    pass
                if cfr_curve is None:
                    try:
                        freqs, mag_db = frequency_response(
                            paths, cfg.frequency, cfg.bandwidth_hz)
                        cfr_curve = {"frequency_hz": freqs.tolist(),
                                     "magnitude_db": np.asarray(mag_db).reshape(-1).tolist()}
                    except Exception as exc:
                        print(f"  cfr unavailable: {type(exc).__name__}: {exc}")
            if "rx" in scene.receivers:
                scene.remove("rx")

    all_db = np.stack(specs)
    multi = all_db.ndim == 4                                  # [n, C, H, W]
    if multi:
        ranges = channel_ranges(all_db, cfg.spectrum.upper(), cfg.power_floor_db)
        spec_min, spec_max = ranges[0]
        print(f"channel ranges: {[(round(a, 2), round(b, 2)) for a, b in ranges]}")
    else:
        ranges = None
        spec_min, spec_max = float(all_db.min()), float(all_db.max())
        print(f"global dB range: [{spec_min:.2f}, {spec_max:.2f}]")

    # Pass 2: write PNGs against that one range, plus the float arrays.
    import imageio.v2 as imageio
    from matplotlib import colormaps

    jet = colormaps["jet"]
    focal = cfg.width / (2 * math.tan(math.radians(cfg.fov_deg) / 2))
    images = {}
    for n, (spec_db, (rx_loc, yaw)) in enumerate(zip(specs, poses), start=1):
        preview = spec_db[0] if multi else spec_db           # the PNG shows channel 0
        norm = np.clip((preview - spec_min) / (spec_max - spec_min), 0.0, 1.0)
        rgb = (jet(norm)[..., :3] * 255).astype(np.uint8)
        imageio.imwrite(os.path.join(cfg.out_dir, "images", f"{n:05d}.png"), rgb)
        if cfg.save_float:
            np.save(os.path.join(cfg.out_dir, "spectra_float", f"{n:05d}.npy"), spec_db)

        r_c2w, qvec = euler_to_quaternion([yaw, 0.0, 0.0])
        images[n] = (qvec, (-r_c2w.apply(rx_loc)).tolist(), 1, f"{n:05d}.png")

    write_cameras_txt(os.path.join(cfg.out_dir, "sparse", "0", "cameras.txt"),
                      1, cfg.width, cfg.height, focal, focal,
                      cfg.width / 2, cfg.height / 2)
    write_images_txt(os.path.join(cfg.out_dir, "sparse", "0", "images.txt"), images)

    elapsed = time.time() - t_start
    meta = asdict(cfg) | {"spec_min_db": spec_min, "spec_max_db": spec_max,
                          "num_images": len(images), "colormap": "jet",
                          "normalization": "global",
                          "seconds": elapsed,
                          "views_per_second": len(images) / max(elapsed, 1e-9),
                          "empty_views": empty_views, "empty_view_db": EMPTY_VIEW_DB}
    if empty_views:
        print(f"empty views (no paths at this receiver pose, written at {EMPTY_VIEW_DB:.0f} dB): {empty_views} of {len(images)}")
    if multi:
        from rf_spectra import MULTI_CHANNELS
        meta["channels"] = (list(MULTI_CHANNELS) if cfg.spectrum.upper() == "MULTI"
                            else ["aod3_zen_x_amp", "aod3_az_x_amp", "aod3_amp"])
        meta["channel_ranges"] = ranges
    with open(os.path.join(cfg.out_dir, "generation_meta.json"), "w") as fid:
        json.dump(meta, fid, indent=1)
    print(f"wrote {len(images)} views to {cfg.out_dir} "
          f"({len(images)/max(elapsed,1e-9):.1f} views/s)")

    if cfg.dashboard and per_view:
        report = build_report(cfg, meta, per_view, all_db, cfr_curve)
        path = os.path.join(cfg.out_dir, "run_report.json")
        with open(path, "w", encoding="utf-8") as fid:
            json.dump(report, fid, indent=1)
        try:
            import dashboard
            html = dashboard.render(report,
                                    preview_dir=os.path.join(cfg.out_dir, "images"))
            out_html = os.path.join(cfg.out_dir, "dashboard.html")
            with open(out_html, "w", encoding="utf-8", newline="\n") as fid:
                fid.write(html)
            print(f"dashboard: {out_html}")
        except Exception as exc:
            print(f"dashboard skipped: {type(exc).__name__}: {exc}")


def build_report(cfg, meta, per_view, all_db, cfr_curve):
    """One generation run in the same shape the ablation dashboard consumes."""
    keys = per_view[0].keys()
    run = {
        "label": f"{cfg.spectrum}, s={cfg.scattering_coefficient:g}, "
                 f"depth={cfg.max_depth}",
        "scattering_coefficient": cfg.scattering_coefficient,
        "max_depth": cfg.max_depth,
        "positions": len(per_view),
        "mean_solve_seconds": meta["seconds"] / max(len(per_view), 1),
        "metrics_mean": {k: float(np.mean([p[k] for p in per_view])) for k in keys},
        "metrics_std": {k: float(np.std([p[k] for p in per_view])) for k in keys},
        "per_position": per_view,
        "spectrum": spectrum_stats(all_db),
    }
    if cfr_curve:
        run["cfr"] = cfr_curve
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "scene_xml": cfg.scene_xml,
        "frequency_hz": cfg.frequency,
        "bandwidth_hz": cfg.bandwidth_hz,
        "M": cfg.M,
        "spectrum": cfg.spectrum,
        "samples_per_src": cfg.samples_per_src,
        "positions": len(per_view),
        "throughput_views_per_second": meta["views_per_second"],
        "runs": [run],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scene-xml", required=True)
    ap.add_argument("--rx-loc-file", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--spectrum", default="MVDR", choices=["CBF", "MVDR", "MULTI", "AOD3"],
                    help="MULTI: one channel each for path power (dB), AoD azimuth, AoD zenith "
                         "and delay; AOD3: the tutorial's angle-times-amplitude RGB encoding")
    ap.add_argument("--tx", dest="tx_loc", type=float, nargs=3, default=(6.905, 0.0, 2 - 1.713),
                    help="transmitter position; the default is the NIST measurement Tx")
    ap.add_argument("--tx-list", nargs="+", default=None, metavar="NAME:x,y,z",
                    help="several transmitters in one process; each dataset goes to <out-dir><NAME>")
    ap.add_argument("--frequency", type=float, default=60e9,
                    help="carrier in Hz; the tutorial's dataset cells use 2.4e9")
    ap.add_argument("--power-floor-db", type=float, default=-150.0,
                    help="MULTI: splat power under this is a kernel tail and is dropped (paper: noise-floor truncation)")
    ap.add_argument("--splat-sigma", type=float, default=3.0,
                    help="MULTI/AOD3: splat kernel width in equirect pixels (1/3 deg); 3 is the tutorial's")
    ap.add_argument("--poses-from", default=None,
                    help="a dataset's sparse/0/images.txt: generate at exactly those poses "
                         "(the released data's, for a like-for-like comparison)")
    ap.add_argument("--materials", choices=["uniform", "tutorial", "tutorial-asis"], default="uniform",
                    help="uniform: ITU materials with one scattering coefficient; tutorial: the "
                         "notebook's per-material definitions with the conductivity formulas' "
                         "frequency unit fixed; tutorial-asis: exactly the notebook (near-PEC walls)")
    ap.add_argument("--diagonal-loading", dest="diagonal_loading", type=float, default=0.0,
                    help="MVDR: regularise the covariance by this fraction of tr(R)/M^2 (needed when delay taps < M^2, e.g. M = 64)")
    ap.add_argument("--M", type=int, default=10, help="M x M UPA (the tutorial's 10; the paper headlines 64)")
    ap.add_argument("--num-positions", type=int, default=800)
    ap.add_argument("--max-depth", type=int, default=1)
    ap.add_argument("--diffraction", action="store_true")
    ap.add_argument("--scattering-coefficient", type=float, default=0.7,
                    help="applied to every material; 0 reproduces Sionna's "
                         "default, which yields almost no diffuse paths")
    ap.add_argument("--no-save-float", dest="save_float", action="store_false")
    ap.add_argument("--no-dashboard", dest="dashboard", action="store_false")
    ap.add_argument("--samples-per-src", type=int, default=1_000_000,
                    dest="samples_per_src")
    ap.add_argument("--seed-per-view", dest="seed_per_view", action="store_true",
                    help="pre-fix behaviour for A/B tests: one lattice and solve per view instead of per position")
    ap.add_argument("--fixed-seed", dest="per_view_seed", action="store_false",
                    help="reuse one sampling lattice for every view, which "
                         "correlates the sampling residual across views")
    args = ap.parse_args()
    tx_list = vars(args).pop("tx_list")
    if not tx_list:
        generate(Config(**vars(args)))
        return
    # several transmitters, one process: scene / OptiX / grid built once
    base = Config(**vars(args))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    scene = build_scene(base)
    solver = PathSolver()
    grid = ArrayGrid.build(base.M, base.width, base.height, base.fov_deg,
                           element_gain_fn=element_gain_fn, device=device)
    print(f"scene + solver + grid built once in {time.time() - t0:.1f} s for {len(tx_list)} transmitters")
    for entry in tx_list:
        name, xyz = entry.split(":")
        cfg = dataclasses.replace(base, tx_loc=tuple(float(v) for v in xyz.split(",")),
                                  out_dir=base.out_dir + name)
        t1 = time.time()
        generate(cfg, shared=(scene, solver, grid))
        print(f"[tx-list] {name} {cfg.tx_loc} -> {cfg.out_dir} in {time.time() - t1:.0f} s")


if __name__ == "__main__":
    main()
