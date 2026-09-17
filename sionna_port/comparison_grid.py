"""Build the comparison table: one row per source or setting, one column per spectrum.

Reading down a column compares the released ground truth against every setting
we sweep, for one array-processing algorithm. Reading across a row compares the
algorithms at one setting. Cells this port cannot fill yet are marked as such
rather than left silently blank -- the gaps are the remaining work.
"""

from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np

# Every spectrum the released dataset ships, in the order the table shows them.
# `ported` says whether generate_dataset.py can produce it: AoD, Delay, MPC and
# TCBF are projections or tapers the port has not implemented.
SPECTRA = [
    ("AoD", "Angle of departure", False),
    ("CBF", "Conventional beamforming", True),
    ("Delay", "Propagation delay", False),
    ("MPC", "Multipath components", False),
    ("MVDR", "MVDR / Capon", True),
    ("TCBF", "Tapered beamforming", False),
]

PER_VIEW_NORMALISED = {"CBF", "TCBF"}

# The optical view is not a spectrum, but it belongs in the table: it is what the
# receiver is looking at while the rest of the row is measured.
OPTICAL = ("optical", "scene from this pose")


def recover_pose(root: str, spectrum: str = "MVDR", test_view_index: int = 0):
    """Position and yaw of a released test view, from its COLMAP entry.

    The released spectra and the Blender renders come from two unrelated
    sampling campaigns, so no optical image matches a spectrum. Recovering the
    pose lets one be rendered instead of hunting for a near miss. Inverts what
    the tutorial's euler_to_quaternion wrote out.
    """
    import numpy as np
    from scipy.spatial.transform import Rotation

    base = os.path.join(root, "RF-3DGS_dataset", "training-rf-spectrum",
                        f"3dgs_{spectrum}_100")
    index = os.path.join(base, "test_index.txt")
    images = os.path.join(base, "sparse", "0", "images.txt")
    if not (os.path.isfile(index) and os.path.isfile(images)):
        return None

    with open(index) as fid:
        names = sorted(os.path.splitext(l.strip())[0] for l in fid if l.strip())
    if test_view_index >= len(names):
        return None
    want = names[test_view_index]

    with open(images) as fid:
        for line in fid:
            parts = line.split()
            if len(parts) >= 10 and os.path.splitext(parts[9])[0] == want:
                qvec = np.array([float(x) for x in parts[1:5]])
                tvec = np.array([float(x) for x in parts[5:8]])
                break
        else:
            return None

    r_c2w = Rotation.from_quat([qvec[1], qvec[2], qvec[3], qvec[0]])
    position = -r_c2w.as_matrix().T @ tvec
    r_posz2posx = Rotation.from_euler("ZYX", [-np.pi / 2, 0.0, -np.pi / 2])
    yaw = float((r_c2w.inv() * r_posz2posx.inv()).as_euler("ZYX")[0])
    return [float(v) for v in position], yaw


def _published_psnr(root: str, name: str) -> str | None:
    path = os.path.join(root, "RF-3DGS_dataset", "RF-3DGS_trained_RRF",
                        f"3dgs_{name}_100", "results.json")
    if not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as fid:
            return f"{json.load(fid)['ours_40000']['PSNR']:.2f} dB"
    except Exception:
        return None


def released_rows(root: str, view: str, test_view: str = "00000.png",
                  optical=None) -> list:
    """Two rows: the training target, and what the trained model renders.

    Both come from test/ours_40000, where gt/ and renders/ share an index, so
    the two rows show the same receiver pose. TCBF falls back to the training
    set for its target and has no model output at all -- the authors released
    trained checkpoints for five of the six spectra.
    """
    gt_cells, model_cells = {}, {}
    for name, _, _ in SPECTRA:
        base = os.path.join(root, "RF-3DGS_dataset", "RF-3DGS_trained_RRF",
                            f"3dgs_{name}_100", "test", "ours_40000")
        gt_img = os.path.join(base, "gt", test_view)
        render_img = os.path.join(base, "renders", test_view)
        norm = ("per image" if name in PER_VIEW_NORMALISED
                else "global, 2 probes")

        if os.path.isfile(gt_img):
            gt_cells[name] = {"image": os.path.relpath(gt_img, root),
                              "params": {"normalisation": norm}}
        else:
            fallback = os.path.join(root, "RF-3DGS_dataset",
                                    "training-rf-spectrum", f"3dgs_{name}_100",
                                    "images", view)
            if os.path.isfile(fallback):
                gt_cells[name] = {"image": os.path.relpath(fallback, root),
                                  "params": {"normalisation": norm,
                                             "note": "training set, other pose"}}

        if os.path.isfile(render_img):
            params = {"normalisation": norm}
            psnr = _published_psnr(root, name)
            if psnr:
                params["PSNR vs gt"] = psnr
            model_cells[name] = {"image": os.path.relpath(render_img, root),
                                 "params": params}
        else:
            model_cells[name] = {"missing": "no trained model"}

    if optical:
        gt_cells[OPTICAL[0]] = optical
        model_cells[OPTICAL[0]] = optical

    return [
        {"label": "RF-3DGS ground truth", "base": "root",
         "params": {"source": "Sionna 0.19 ray tracing",
                    "role": "training target",
                    "views": "3200"},
         "cells": gt_cells},
        {"label": "RF-3DGS model output", "base": "root",
         "params": {"source": "released RRF checkpoint",
                    "role": "what the model predicts",
                    "iterations": "40,000"},
         "cells": model_cells},
    ]


def render_optical_cell(scene_xml, position, yaw, args, tag):
    """Render the scene from one pose and return a table cell for it."""
    try:
        import imageio.v2 as imageio
        import render_optical
    except ImportError:
        return None
    out_dir = os.path.join(os.path.dirname(args.out) or ".", "comparison_previews")
    os.makedirs(out_dir, exist_ok=True)
    try:
        scene = render_optical.load_radio_scene(scene_xml, args.frequency, 0.7)
        img = render_optical.render_pose(scene, position, yaw, args.width,
                                         args.height, args.fov, num_samples=96)
    except Exception as exc:
        print(f"  optical render failed ({tag}): {type(exc).__name__}: {exc}")
        return None
    name = f"optical_{tag}.png"
    imageio.imwrite(os.path.join(out_dir, name), img)
    return {"image": os.path.join("comparison_previews", name),
            "base": "out",
            "params": {"pose": f"{position[0]:.1f}, {position[1]:.1f}, "
                               f"{position[2]:.1f}",
                       "yaw": f"{math.degrees(yaw):+.0f} deg",
                       "shading": "by radio material"}}


def generate_rows(args):
    import mitsuba as mi
    mi.set_variant(args.variant)
    from sionna.rt import PathSolver, PlanarArray, Receiver, Transmitter, load_scene

    import torch
    import imageio.v2 as imageio
    from matplotlib import colormaps

    from rf_metrics import channel_metrics
    from rf_spectra import ArrayGrid, cbf_spectrum, mvdr_spectrum, paths_to_response

    device = "cuda" if torch.cuda.is_available() else "cpu"
    grid = ArrayGrid.build(args.M, args.width, args.height, args.fov, device=device)
    solver = PathSolver()
    out_dir = os.path.join(os.path.dirname(args.out) or ".", "comparison_previews")
    os.makedirs(out_dir, exist_ok=True)

    optical_cell = render_optical_cell(args.scene_xml, args.rx, 0.0, args, "ours")

    rows = []
    for scattering, depth in [(s, d) for s in args.scattering
                              for d in args.depth]:
        scene = load_scene(args.scene_xml, merge_shapes=True)
        scene.frequency = args.frequency
        scene.tx_array = PlanarArray(num_rows=1, num_cols=1, pattern="iso",
                                     polarization="V")
        scene.rx_array = PlanarArray(num_rows=args.M, num_cols=args.M,
                                     vertical_spacing=0.5, horizontal_spacing=0.5,
                                     pattern="iso", polarization="V")
        scene.add(Transmitter(name="tx", position=args.tx))
        for material in scene.radio_materials.values():
            material.scattering_coefficient = scattering
        scene.add(Receiver(name="rx", position=args.rx, orientation=[0.0, 0.0, 0.0]))

        paths = solver(scene=scene, max_depth=depth,
                       samples_per_src=args.samples, los=True,
                       specular_reflection=True, diffuse_reflection=True,
                       refraction=False, synthetic_array=True, seed=42)
        n_paths = int(np.asarray(paths.valid).sum())
        metrics = channel_metrics(paths) if n_paths else None
        response = paths_to_response(paths, args.time_interval, device=device)

        cells = {}
        for name, _, ported in SPECTRA:
            if not ported:
                cells[name] = {"missing": "not ported"}
                continue
            try:
                if name == "MVDR":
                    _, spec_db = mvdr_spectrum(response, grid, args.diagonal_loading)
                else:
                    _, spec_db = cbf_spectrum(response, grid)
            except ValueError as exc:
                cells[name] = {"missing": str(exc)[:40]}
                continue
            arr = spec_db.cpu().numpy()
            lo, hi = float(arr.min()), float(arr.max())
            norm = np.clip((arr - lo) / max(hi - lo, 1e-9), 0.0, 1.0)
            rgb = (colormaps["jet"](norm)[..., :3] * 255).astype(np.uint8)
            fname = f"{name}_s{scattering:g}_d{depth}.png"
            imageio.imwrite(os.path.join(out_dir, fname), rgb)
            cells[name] = {
                "image": os.path.join("comparison_previews", fname),
                "params": {"normalisation": "global, from data",
                           "dB range": f"{lo:.0f} .. {hi:.0f}"}}

        if optical_cell:
            cells[OPTICAL[0]] = optical_cell
        params = {"scattering": f"{scattering:g}", "max_depth": f"{depth}",
                  "paths": f"{n_paths:,}", "solver": "Sionna 2.1, GPU"}
        if metrics:
            params["K-factor"] = f"{metrics.k_factor_db:+.1f} dB"
            params["RMS delay"] = f"{metrics.rms_delay_spread_ns:.1f} ns"
        rows.append({"label": f"ours, s={scattering:g}, depth={depth}",
                     "base": "out", "params": params, "cells": cells})
        print(f"  s={scattering:g} depth={depth}: {n_paths:,} paths")
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene-xml", required=True)
    ap.add_argument("--root", default="..", help="repo root, for the released data")
    ap.add_argument("--out", default="../output/comparison.json")
    ap.add_argument("--scattering", type=float, nargs="+", default=[0.0, 0.3, 0.7])
    ap.add_argument("--depth", type=int, nargs="+", default=[1])
    ap.add_argument("--view", default="00001.png")
    ap.add_argument("--tx", type=float, nargs=3, default=[6.905, 0.0, 0.287])
    ap.add_argument("--rx", type=float, nargs=3, default=[3.0, -2.0, 0.0])
    ap.add_argument("--M", type=int, default=10)
    ap.add_argument("--width", type=int, default=300)
    ap.add_argument("--height", type=int, default=200)
    ap.add_argument("--fov", type=float, default=90.0)
    ap.add_argument("--frequency", type=float, default=60e9)
    ap.add_argument("--samples", type=int, default=1_000_000)
    ap.add_argument("--time-interval", type=float, default=0.1)
    ap.add_argument("--diagonal-loading", type=float, default=0.0)
    ap.add_argument("--variant", default="cuda_ad_mono_polarized")
    args = ap.parse_args()

    import mitsuba as mi
    if mi.variant() is None:
        mi.set_variant(args.variant)

    released_optical = None
    pose = recover_pose(args.root)
    if pose:
        position, yaw = pose
        print(f"released test view pose: {position}, yaw {math.degrees(yaw):+.0f} deg")
        released_optical = render_optical_cell(args.scene_xml, position, yaw,
                                               args, "released")

    rows = released_rows(args.root, args.view, optical=released_optical)         + generate_rows(args)
    payload = {"columns": [{"name": OPTICAL[0], "algorithm": OPTICAL[1],
                            "ported": True}]
                          + [{"name": n, "algorithm": a, "ported": p}
                             for n, a, p in SPECTRA],
               "rows": rows,
               "root": os.path.abspath(args.root),
               "out_base": os.path.abspath(os.path.dirname(args.out) or ".")}
    with open(args.out, "w", encoding="utf-8") as fid:
        json.dump(payload, fid, indent=1)
    print(f"wrote {args.out}: {len(rows)} rows x {len(SPECTRA)} columns")


if __name__ == "__main__":
    main()
