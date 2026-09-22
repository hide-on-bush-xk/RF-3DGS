"""Time the original RF-3DGS dataset pipeline: Sionna 0.19 + TensorFlow, as the tutorial runs it.

Executes the tutorial notebook's own cells (imports, scene + custom materials,
array setup, spectrum functions) from `sionna_nb_src.py`, then times exactly
what `generate_3dgs_MVDR_spectrum` does per receiver position: four
`compute_paths` calls (one per yaw) and four `MVDR_spectrum` calls, with the
tutorial's settings (max_depth 1, scattering on, scat_keep_prob 0.1,
num_samples 1e6, M = 10, time_interval 0.1).

Runs on the CPU (LLVM) in WSL: TensorFlow finds no GPU there and OptiX is
unavailable, so this is the pipeline as a laptop user would run it, not the
authors' GPU box. The per-view split into path solving and spectrum
computation is reported separately so the comparison stays honest.

Running their code rather than a reimplementation is the point: a timing of
something written here would measure this code, not theirs. The hardware caveat
above must travel with any number this produces.

    LD_LIBRARY_PATH=/home/ke/miniconda3/envs/rf-gsplat/lib \
    /home/ke/venvs/rf-sionna019/bin/python rrf_gsplat/time_tutorial_019.py --positions 2
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

TUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "sionna_tutorial",
                   "RF-3DGS_Sionna_simulation_tutorial")


def load_cells(path):
    """Split the notebook dump into cells, or return it whole if unmarked."""
    src = open(path, encoding="utf-8").read()
    # the dump separates cells with "# %%" or similar markers; fall back to the whole file
    parts = re.split(r"^# %%.*$", src, flags=re.M)
    return parts if len(parts) > 1 else [src]


def extract_functions(src, names):
    """Top-level `def name(...)` blocks by indentation; last definition wins.

    Text extraction rather than importing the dump: running the whole file
    would execute every notebook cell, including the ones that generate data.
    A function's body is everything indented (or blank) after its def line.
    """
    lines = src.split("\n")
    out = {}
    for i, line in enumerate(lines):
        m = re.match(r"def (\w+)\(", line)
        if m and m.group(1) in names:
            j = i + 1
            while j < len(lines) and (lines[j].startswith((" ", "\t")) or not lines[j].strip()):
                j += 1
            out[m.group(1)] = "\n".join(lines[i:j])
    return out


def main():
    """Reconstruct the notebook's state, then time its per-view work."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--positions", type=int, default=2)
    ap.add_argument("--num-samples", type=float, default=1e6)
    ap.add_argument("--frequency", type=float, default=2.4e9,
                    help="the tutorial's dataset cells set 2.4e9 (60e9 for MPC)")
    ap.add_argument("--scene-xml", default="./NIST_lobby_v1.0/NIST_lobby_V1.0_material_assigned_ascii.xml",
                    help="the tutorial's scene with mesh names transliterated to ASCII "
                         "(the meshes were renamed for Windows Mitsuba)")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                  "..", "output", "rrf", "tutorial_019_timing.json"))
    cfg = ap.parse_args()

    # The notebook uses relative paths throughout, so the working directory has
    # to be its own.
    os.chdir(TUT)
    src = open("sionna_nb_src.py", encoding="utf-8").read()
    # Strip the IPython magics (%matplotlib, !pip), which are not Python.
    src = "\n".join(l for l in src.split("\n") if not l.lstrip().startswith(("%", "!")))
    src = src.replace('load_scene("./NIST_lobby_v1.0/NIST_lobby_V1.0_material_assigned.xml")',
                      f'load_scene("{cfg.scene_xml}")')

    g = {"__name__": "__tutorial__"}
    # 1. imports: every top-level import line in the dump
    # notebook-only modules (cv2, ipywidgets, ...) are not needed for the timing
    # Each import is executed individually so one missing package cannot stop
    # the rest; the failures are printed rather than swallowed.
    for l in src.split("\n"):
        if re.match(r"(import |from \S+ import )", l):
            try:
                exec(l, g)
            except ImportError as exc:
                print(f"skipping: {l.strip()} ({exc})")
    import numpy as np, tensorflow as tf
    tf.get_logger().setLevel("ERROR")
    # 2. the scene cell: from load_scene to the end of the material replacement loop
    # Sliced by its literal text: fragile, but it keeps the executed code
    # byte-identical to the notebook's.
    i0 = src.index(f'scene = load_scene("{cfg.scene_xml}")')
    i1 = src.index("# Verify material assignments", i0)
    t0 = time.time()
    exec(src[i0:i1], g)
    scene = g["scene"]
    # Scene load is timed separately: it is a one-off, not a per-view cost.
    t_scene = time.time() - t0
    print(f"scene + materials in {t_scene:.1f} s; {len(scene.objects)} objects")
    # 3. the spectrum functions
    funcs = extract_functions(src, {"array_manifold_vector", "steering_vector", "compute_angle_matrices",
                                    "paths_to_response", "merge_paths_to_time_grid", "MVDR_spectrum",
                                    "CBF_spectrum", "paths_to_response_sinc"})
    # Defined in dependency order, since each exec sees only what is already in g.
    for name in ("array_manifold_vector", "steering_vector", "compute_angle_matrices",
                 "merge_paths_to_time_grid", "paths_to_response", "paths_to_response_sinc",
                 "CBF_spectrum", "MVDR_spectrum"):
        exec(funcs[name], g)
    # 4. the dataset cell's array setup (cell 93): M = 10, tr38901, synthetic array
    PlanarArray, Receiver, Transmitter = g["PlanarArray"], g["Receiver"], g["Transmitter"]
    M = 10
    scene.frequency = cfg.frequency
    scene.tx_array = PlanarArray(num_rows=1, num_cols=1, vertical_spacing=0.5, horizontal_spacing=0.5,
                                 pattern="iso", polarization="V")
    scene.rx_array = PlanarArray(num_rows=M, num_cols=M, vertical_spacing=0.5, horizontal_spacing=0.5,
                                 pattern="tr38901", polarization="V")
    scene.synthetic_array = True
    # cell 85: "id x_mm y_mm z" -> metres, at the tutorial's receiver height
    # No jitter here, unlike the generator: this is a timing run, and the exact
    # positions do not matter as long as they are real ones.
    rx_locs = []
    with open("NIST_rx_loc.txt") as fid:
        for line in fid:
            p = line.split()
            if len(p) == 4:
                rx_locs.append([float(p[1]) / 1000, float(p[2]) / 1000, 1.625 - 1.713])
    rx_locs = np.array(rx_locs)
    tx_loc = [6.905, 0, 2 - 1.713]
    for n in ("tx", "rx"):
        try:
            scene.remove(n)
        except Exception:
            pass                           # not present yet, which is fine
    scene.add(Transmitter(name="tx", position=tx_loc))
    theta_grid, phi_grid = g["compute_angle_matrices"](width=300, height=200, fov=90)

    rows = []
    for i in range(min(cfg.positions, len(rx_locs))):
        rx_loc = [float(v) for v in rx_locs[i][:3]]
        for angle in [-np.pi / 2, 0, np.pi / 2, np.pi]:
            orientation = tf.Variable([angle, 0, 0], dtype=tf.float32)
            rx = Receiver(name="rx", position=rx_loc, look_at=None, orientation=orientation)
            scene.add(rx)
            # The two timed halves. Note 0.19 re-solves per yaw, which is the
            # cost the 2.x port avoids by solving once per position.
            t0 = time.time()
            paths = scene.compute_paths(max_depth=1, reflection=True, diffraction=False, scattering=True,
                                        scat_keep_prob=0.1, scat_random_phases=False,
                                        num_samples=cfg.num_samples)
            t_paths = time.time() - t0
            n_paths = int(paths.a.shape[-2])
            t0 = time.time()
            _abs, spec_db = g["MVDR_spectrum"](paths, M=M, theta_grid=theta_grid, phi_grid=phi_grid,
                                               time_interval=0.1)
            t_spec = time.time() - t0
            scene.remove("rx")
            rows.append({"position": i, "yaw": float(angle), "paths": n_paths,
                         "compute_paths_s": t_paths, "mvdr_spectrum_s": t_spec})
            print(f"  pos {i} yaw {np.degrees(angle):5.0f}: {n_paths:7d} paths, "
                  f"compute_paths {t_paths:6.1f} s, MVDR_spectrum {t_spec:6.1f} s")
    per_view = {k: float(np.mean([r[k] for r in rows])) for k in ("compute_paths_s", "mvdr_spectrum_s")}
    per_view["total_s"] = per_view["compute_paths_s"] + per_view["mvdr_spectrum_s"]
    # The backend string travels with the numbers, because on a GPU box they
    # would be different and quoting them without it would misrepresent the work.
    result = {"config": vars(cfg), "scene_seconds": t_scene, "rows": rows, "per_view_mean": per_view,
              "backend": "CPU (LLVM), TensorFlow without GPU", "sionna": g["sionna"].__version__,
              "views_per_position": 4,
              # 800 positions x 4 yaws, extrapolated from the sampled views.
              "estimated_800_positions_hours": per_view["total_s"] * 3200 / 3600}
    os.makedirs(os.path.dirname(os.path.abspath(cfg.out)), exist_ok=True)
    json.dump(result, open(cfg.out, "w"), indent=1)
    print(f"per view: paths {per_view['compute_paths_s']:.1f} s + spectrum {per_view['mvdr_spectrum_s']:.1f} s; "
          f"800 positions x 4 views = {result['estimated_800_positions_hours']:.1f} h")


if __name__ == "__main__":
    main()
