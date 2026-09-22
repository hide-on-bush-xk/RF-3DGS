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
    ("AoD", "Angle of departure", True),
    ("CBF", "Conventional beamforming", True),
    ("Delay", "Propagation delay", True),
    ("MPC", "Multipath components", True),
    ("MVDR", "MVDR / Capon", True),
    ("TCBF", "Tapered beamforming", True),
]

# The two the authors rescaled per image. That is the finding the table is built
# to show: they are also the two whose reported PSNR collapses.
PER_VIEW_NORMALISED = {"CBF", "TCBF"}

# Every row reports the same fields in the same order, so the column can be read
# down. A row that genuinely does not have a value prints a dash rather than
# omitting the line, which would shift everything below it out of alignment.
ROW_FIELDS = [
    ("source", "source"),
    ("role", "role"),
    ("max depth", "max_depth"),
    ("diffuse", "diffuse"),
    ("paths / pose", "paths"),
    ("K-factor", "k_factor"),
    ("RMS delay", "rms_delay"),
    ("normalisation", "normalisation"),
]


def row_params(**values) -> dict:
    """Fill the fixed field list, leaving a dash wherever a row has no value."""
    # Raises on a typo rather than silently dropping the value, which would
    # leave a dash that looks like a legitimate "no data".
    unknown = set(values) - {key for _, key in ROW_FIELDS}
    if unknown:
        raise TypeError(f"unknown row fields: {sorted(unknown)}")
    return {label: values.get(key, "--") for label, key in ROW_FIELDS}

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

    # scipy wants the scalar last; COLMAP stores it first.
    r_c2w = Rotation.from_quat([qvec[1], qvec[2], qvec[3], qvec[0]])
    position = -r_c2w.as_matrix().T @ tvec
    # Undo the fixed camera/array frame change, leaving only the yaw.
    r_posz2posx = Rotation.from_euler("ZYX", [-np.pi / 2, 0.0, -np.pi / 2])
    yaw = float((r_c2w.inv() * r_posz2posx.inv()).as_euler("ZYX")[0])
    return [float(v) for v in position], yaw


def _published_psnr(root: str, name: str) -> str | None:
    """The authors' own reported PSNR for one spectrum, or None if unavailable."""
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
            # No test render for this spectrum: fall back to a training image,
            # which is a DIFFERENT pose. The note below says so in the cell, so
            # the row is not read as a like-for-like comparison.
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

    # The tutorial's own solver settings, for the row that produced the targets.
    # scat_keep_prob is not a scattering coefficient -- it thinned diffuse paths
    # by a factor of ten -- so it is named rather than translated.
    return [
        {"label": "RF-3DGS ground truth", "base": "root",
         "params": row_params(source="Sionna 0.19, TensorFlow",
                              role="training target",
                              max_depth="1",
                              diffuse="scat_keep_prob 0.1",
                              paths="&gt; 300,000 (paper)",
                              normalisation="per type, 2 probes"),
         "cells": gt_cells},
        {"label": "RF-3DGS model output", "base": "root",
         "params": row_params(source="released RRF checkpoint",
                              role="model prediction, 40k iterations",
                              normalisation="inherited from target"),
         "cells": model_cells},
    ]


def spectrum_panel(name, paths, response, grid, args):
    """One spectrum as an 8-bit RGB panel, whichever family it belongs to.

    CBF, TCBF and MVDR beamform on the pinhole grid directly. MPC, Delay and AoD
    splat paths onto an equirectangular grid and are resampled through the same
    pinhole model, so every panel in a row shares one camera.
    """
    import numpy as np
    from matplotlib import colormaps
    from rf_spectra import (aod_spectrum_equirect, cbf_spectrum,
                            delay_spectrum_equirect, equirect_to_perspective,
                            mpc_spectrum_equirect, mvdr_spectrum, tcbf_spectrum)

    if name in ("CBF", "TCBF", "MVDR"):
        if name == "MVDR":
            _, spec_db = mvdr_spectrum(response, grid, args.diagonal_loading)
        elif name == "TCBF":
            _, spec_db = tcbf_spectrum(response, grid)
        else:
            _, spec_db = cbf_spectrum(response, grid)
        arr = spec_db.detach().cpu().numpy()
        # Per-panel normalisation: these panels are for comparing structure
        # across settings, not absolute levels. The generated datasets use one
        # global range instead.
        lo, hi = float(arr.min()), float(arr.max())
        norm = np.clip((arr - lo) / max(hi - lo, 1e-9), 0.0, 1.0)
        return (colormaps["jet"](norm)[..., :3] * 255).astype(np.uint8)

    if name == "MPC":
        equirect = mpc_spectrum_equirect(paths, args.equirect_scale)
        view = equirect_to_perspective(equirect, args.width, args.height,
                                       args.fov, args.yaw)
        arr = view.detach().cpu().numpy()
        lo, hi = float(arr.min()), float(arr.max())
        norm = np.clip((arr - lo) / max(hi - lo, 1e-9), 0.0, 1.0)
        return (colormaps["jet"](norm)[..., :3] * 255).astype(np.uint8)

    # Delay and AoD are already RGB (angle and amplitude in separate channels),
    # so they get no colourmap -- only a transpose to HWC and a scale.
    equirect = (delay_spectrum_equirect(paths, args.equirect_scale)
                if name == "Delay"
                else aod_spectrum_equirect(paths, args.equirect_scale))
    view = equirect_to_perspective(equirect, args.width, args.height,
                                   args.fov, args.yaw)
    arr = view.detach().cpu().numpy().transpose(1, 2, 0)
    # One shared maximum across channels, so the relative channel weights survive.
    hi = max(float(arr.max()), 1e-9)
    return np.clip(arr / hi * 255.0, 0, 255).astype(np.uint8)


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
    """One row per (scattering, depth), each with every spectrum as a cell.

    All rows share the receiver pose, the transmitter and the seed, so a column
    varies only in the setting named in its row label.
    """
    import mitsuba as mi
    mi.set_variant(args.variant)
    from sionna.rt import PathSolver, PlanarArray, Receiver, Transmitter, load_scene

    import torch
    import imageio.v2 as imageio
    from matplotlib import colormaps

    from rf_metrics import channel_metrics
    from rf_spectra import (ArrayGrid, aod_spectrum_equirect, cbf_spectrum,
                            delay_spectrum_equirect, equirect_to_perspective,
                            mpc_spectrum_equirect, mvdr_spectrum,
                            paths_to_response, tcbf_spectrum)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    grid = ArrayGrid.build(args.M, args.width, args.height, args.fov, device=device)
    solver = PathSolver()
    out_dir = os.path.join(os.path.dirname(args.out) or ".", "comparison_previews")
    os.makedirs(out_dir, exist_ok=True)

    optical_cell = render_optical_cell(args.scene_xml, args.rx, args.yaw,
                                       args, "ours")

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
        scene.add(Receiver(name="rx", position=[float(v) for v in args.rx],
                           orientation=[float(args.yaw), 0.0, 0.0]))

        paths = solver(scene=scene, max_depth=depth,
                       samples_per_src=args.samples, los=True,
                       specular_reflection=True, diffuse_reflection=True,
                       refraction=False, synthetic_array=True, seed=args.seed)
        n_paths = int(np.asarray(paths.valid).sum())
        metrics = channel_metrics(paths) if n_paths else None
        response = paths_to_response(paths, args.time_interval, device=device)

        cells = {}
        for name, _, ported in SPECTRA:
            try:
                rgb = spectrum_panel(name, paths, response, grid, args)
            except ValueError as exc:
                # A singular MVDR covariance is the usual cause; the reason goes
                # into the cell so the gap is explained rather than blank.
                cells[name] = {"missing": str(exc)[:40]}
                continue
            fname = f"{name}_s{scattering:g}_d{depth}.png"
            imageio.imwrite(os.path.join(out_dir, fname), rgb)
            cells[name] = {
                "image": os.path.join("comparison_previews", fname),
                "params": {"normalisation": "global, from data"}}

        if optical_cell:
            cells[OPTICAL[0]] = optical_cell
        params = row_params(
            source="Sionna 2.1, GPU",
            role="regenerated here",
            max_depth=f"{depth}",
            diffuse=f"coefficient {scattering:g}",
            paths=f"{n_paths:,}",
            k_factor=(f"{metrics.k_factor_db:+.1f} dB" if metrics else "--"),
            rms_delay=(f"{metrics.rms_delay_spread_ns:.1f} ns"
                       if metrics else "--"),
            normalisation="global, from data")
        rows.append({"label": f"ours, s={scattering:g}, depth={depth}",
                     "base": "out", "params": params, "cells": cells})
        print(f"  s={scattering:g} depth={depth}: {n_paths:,} paths")
    return rows


def main():
    """Recover the released pose, generate our rows at it, and write the manifest.

    The output is a JSON manifest, not HTML: comparison_table.py renders it.
    """
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene-xml", required=True)
    ap.add_argument("--root", default="..", help="repo root, for the released data")
    ap.add_argument("--out", default="../output/comparison.json")
    ap.add_argument("--scattering", type=float, nargs="+", default=[0.0, 0.3, 0.7])
    ap.add_argument("--depth", type=int, nargs="+", default=[1])
    ap.add_argument("--view", default="00001.png")
    ap.add_argument("--tx", type=float, nargs=3, default=[6.905, 0.0, 0.287])
    ap.add_argument("--rx", type=float, nargs=3, default=None,
                    help="receiver position; defaults to the released test "
                         "view's own pose so every row shares one camera")
    ap.add_argument("--yaw", type=float, default=None,
                    help="receiver yaw in radians; defaults with --rx")
    ap.add_argument("--equirect-scale", type=int, default=3)
    ap.add_argument("--M", type=int, default=10)
    ap.add_argument("--width", type=int, default=300)
    ap.add_argument("--height", type=int, default=200)
    ap.add_argument("--fov", type=float, default=90.0)
    ap.add_argument("--frequency", type=float, default=60e9)
    ap.add_argument("--samples", type=int, default=1_000_000)
    ap.add_argument("--time-interval", type=float, default=0.1)
    ap.add_argument("--diagonal-loading", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--variant", default="cuda_ad_mono_polarized")
    args = ap.parse_args()

    import mitsuba as mi
    if mi.variant() is None:
        mi.set_variant(args.variant)

    released_optical = None
    pose = recover_pose(args.root)
    if pose:
        position, yaw = pose
        print(f"released test view pose: {position}, "
              f"yaw {math.degrees(yaw):+.0f} deg")
        # Our rows default to that same pose. Comparing a column only means
        # something if every cell in it looks the same way.
        if args.rx is None:
            args.rx = position
        if args.yaw is None:
            args.yaw = yaw
        released_optical = render_optical_cell(args.scene_xml, position, yaw,
                                               args, "released")
    # Fallback pose, used only when the released dataset is not present.
    if args.rx is None:
        args.rx = [3.0, -2.0, 0.0]
    if args.yaw is None:
        args.yaw = 0.0

    rows = released_rows(args.root, args.view, optical=released_optical)         + generate_rows(args)
    # The optical view leads the columns; the two absolute roots let the
    # renderer resolve cells against the repo or against our own output.
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
