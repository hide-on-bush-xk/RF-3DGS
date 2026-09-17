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
from dataclasses import dataclass, asdict

import numpy as np
import torch

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
    return scene


def solve_paths(solver, scene, cfg: Config):
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
                  seed=cfg.seed)


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


def spectrum_for_paths(paths, grid: ArrayGrid, cfg: Config):
    response = paths_to_response(paths, cfg.time_interval_ns, device=grid.theta.device)
    if cfg.spectrum.upper() == "CBF":
        return cbf_spectrum(response, grid)
    if cfg.spectrum.upper() == "MVDR":
        return mvdr_spectrum(response, grid, cfg.diagonal_loading)
    raise ValueError(f"unknown spectrum type {cfg.spectrum!r}")


# --------------------------------------------------------------------------
# Dataset generation
# --------------------------------------------------------------------------

def generate(cfg: Config):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(cfg.seed)

    os.makedirs(os.path.join(cfg.out_dir, "images"), exist_ok=True)
    if cfg.save_float:
        os.makedirs(os.path.join(cfg.out_dir, "spectra_float"), exist_ok=True)
    os.makedirs(os.path.join(cfg.out_dir, "sparse", "0"), exist_ok=True)

    scene = build_scene(cfg)
    scene.add(Transmitter(name="tx", position=list(cfg.tx_loc)))
    solver = PathSolver()

    grid = ArrayGrid.build(cfg.M, cfg.width, cfg.height, cfg.fov_deg,
                           element_gain_fn=element_gain_fn, device=device)

    rx_locs = load_rx_locations(cfg.rx_loc_file, rng)[:cfg.num_positions]

    # Pass 1: collect every spectrum in dB, so the normalisation range comes from
    # the data instead of two hand-picked probe positions.
    print(f"pass 1/2: {len(rx_locs)} positions x {len(VIEW_YAWS)} views")
    specs, poses = [], []
    for i, rx_loc in enumerate(rx_locs):
        if i % 50 == 0:
            print(f"  {i}/{len(rx_locs)}")
        for yaw in VIEW_YAWS:
            scene.remove("rx") if "rx" in scene.receivers else None
            scene.add(Receiver(name="rx", position=list(rx_loc),
                               orientation=[yaw, 0.0, 0.0]))
            paths = solve_paths(solver, scene, cfg)
            _, spec_db = spectrum_for_paths(paths, grid, cfg)
            specs.append(spec_db.cpu().numpy().astype(np.float32))
            poses.append((rx_loc, yaw))
            scene.remove("rx")

    all_db = np.stack(specs)
    spec_min, spec_max = float(all_db.min()), float(all_db.max())
    print(f"global dB range: [{spec_min:.2f}, {spec_max:.2f}]")

    # Pass 2: write PNGs against that one range, plus the float arrays.
    import imageio.v2 as imageio
    from matplotlib import colormaps

    jet = colormaps["jet"]
    focal = cfg.width / (2 * math.tan(math.radians(cfg.fov_deg) / 2))
    images = {}
    for n, (spec_db, (rx_loc, yaw)) in enumerate(zip(specs, poses), start=1):
        norm = np.clip((spec_db - spec_min) / (spec_max - spec_min), 0.0, 1.0)
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

    meta = asdict(cfg) | {"spec_min_db": spec_min, "spec_max_db": spec_max,
                          "num_images": len(images), "colormap": "jet",
                          "normalization": "global"}
    with open(os.path.join(cfg.out_dir, "generation_meta.json"), "w") as fid:
        json.dump(meta, fid, indent=1)
    print(f"wrote {len(images)} views to {cfg.out_dir}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scene-xml", required=True)
    ap.add_argument("--rx-loc-file", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--spectrum", default="MVDR", choices=["CBF", "MVDR"])
    ap.add_argument("--num-positions", type=int, default=800)
    ap.add_argument("--max-depth", type=int, default=1)
    ap.add_argument("--diffraction", action="store_true")
    ap.add_argument("--no-save-float", dest="save_float", action="store_false")
    args = ap.parse_args()
    generate(Config(**vars(args)))


if __name__ == "__main__":
    main()
