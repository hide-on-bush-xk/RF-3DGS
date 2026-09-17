"""Collect the three kinds of image this project deals with, with their parameters.

A spectrum is hard to judge on its own. Putting the optical render of the same
room next to the released ground truth and next to what this port regenerates
makes the differences legible: what the room is, what the authors' pipeline
produced, and what ours does.
"""

from __future__ import annotations

import base64
import io
import json
import os

MAX_WIDTH = 520


def embed_image(path: str, max_width: int = MAX_WIDTH) -> str | None:
    """Base64 PNG data URI, converting and downscaling whatever is on disk."""
    try:
        from PIL import Image
    except ImportError:
        return None
    if not os.path.isfile(path):
        return None
    with Image.open(path) as im:
        im = im.convert("RGB")
        if im.width > max_width:
            im = im.resize((max_width, round(im.height * max_width / im.width)))
        buf = io.BytesIO()
        im.save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


# How each released spectrum type was generated. The normalisation column is the
# finding that matters: CBF and TCBF are the only two rescaled per image, and
# they are also the two the paper reports collapsing to about 5 dB PSNR.
RELEASED_TYPES = [
    ("AoD", "Angle of departure", "global (2 probe positions)"),
    ("CBF", "Conventional beamforming", "per image"),
    ("Delay", "Propagation delay", "global (2 probe positions)"),
    ("MPC", "Multipath components", "global (2 probe positions)"),
    ("MVDR", "MVDR / Capon", "global (2 probe positions)"),
    ("TCBF", "Tapered beamforming", "per image"),
]


def _published_metrics(root: str, spectrum: str) -> dict:
    path = os.path.join(root, "RF-3DGS_dataset", "RF-3DGS_trained_RRF",
                        f"3dgs_{spectrum}_100", "results.json")
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, encoding="utf-8") as fid:
            return json.load(fid).get("ours_40000", {})
    except Exception:
        return {}


def collect(root: str, regen_dir: str | None = None, view: str = "00001.png"):
    """Panels grouped by origin. Missing files are skipped, not faked."""
    groups = []

    # 1. the room, as a camera sees it
    blender = os.path.join(root, "RF-3DGS_dataset", "blender_visual_dataset", "train")
    if os.path.isdir(blender):
        names = sorted(f for f in os.listdir(blender)
                       if f.lower().endswith((".tif", ".png", ".jpg")))
        picks = [names[i] for i in (0, len(names) // 3, 2 * len(names) // 3)
                 if i < len(names)]
        panels = [{"src": embed_image(os.path.join(blender, n)),
                   "title": n,
                   "params": {"resolution": "1600 x 900",
                              "images": f"{len(names)}",
                              "renderer": "Blender + Blender-NeRF",
                              "trains": "geometry stage (30k iterations)"}}
                  for n in picks]
        groups.append({
            "title": "The room, optically",
            "note": "What a camera sees. The visual 3DGS stage reconstructs "
                    "geometry from these, and the RF stage then freezes that "
                    "geometry and refits only the radiance.",
            "panels": [p for p in panels if p["src"]]})

    # 2. the released spectra, one per ASP algorithm
    panels = []
    for spectrum, description, normalisation in RELEASED_TYPES:
        img = os.path.join(root, "RF-3DGS_dataset", "training-rf-spectrum",
                           f"3dgs_{spectrum}_100", "images", view)
        src = embed_image(img, max_width=300)
        if not src:
            continue
        pub = _published_metrics(root, spectrum)
        params = {"algorithm": description,
                  "resolution": "300 x 200, 90 deg FoV",
                  "views": "3200 (800 positions x 4 yaw)",
                  "normalisation": normalisation}
        if pub:
            params["published PSNR"] = f"{pub.get('PSNR', float('nan')):.2f} dB"
            params["published LPIPS"] = f"{pub.get('LPIPS', float('nan')):.4f}"
        panels.append({"src": src, "title": spectrum, "params": params})
    if panels:
        groups.append({
            "title": "Released ground truth",
            "note": "The same receiver pose under six array-processing "
                    "algorithms, as shipped with the paper. Note the "
                    "normalisation row: CBF and TCBF are the only two rescaled "
                    "per image, and they are also the two the paper reports "
                    "collapsing to about 5 dB PSNR.",
            "panels": panels})

    # 3. what this port regenerates
    if regen_dir and os.path.isdir(os.path.join(regen_dir, "images")):
        meta_path = os.path.join(regen_dir, "generation_meta.json")
        meta = {}
        if os.path.isfile(meta_path):
            with open(meta_path, encoding="utf-8") as fid:
                meta = json.load(fid)
        report_path = os.path.join(regen_dir, "run_report.json")
        paths_mean = None
        if os.path.isfile(report_path):
            with open(report_path, encoding="utf-8") as fid:
                run = json.load(fid)["runs"][0]
                paths_mean = run["metrics_mean"]["num_paths"]
        names = sorted(os.listdir(os.path.join(regen_dir, "images")))[:3]
        panels = []
        for n in names:
            params = {
                "algorithm": meta.get("spectrum", "?"),
                "scattering": f"{meta.get('scattering_coefficient', '?')}",
                "max_depth": f"{meta.get('max_depth', '?')}",
                "normalisation": "global (from the data)",
                "dB range": (f"{meta.get('spec_min_db', float('nan')):.1f} .. "
                             f"{meta.get('spec_max_db', float('nan')):.1f}"),
            }
            if paths_mean:
                params["paths per pose"] = f"{paths_mean:,.0f}"
            panels.append({"src": embed_image(
                os.path.join(regen_dir, "images", n), max_width=300),
                "title": n, "params": params})
        panels = [p for p in panels if p["src"]]
        if panels:
            groups.append({
                "title": "Regenerated here",
                "note": f"Sionna {meta.get('frequency', 60e9)/1e9:.0f} GHz on the "
                        f"GPU, with the scattering coefficient calibrated to the "
                        f"paper's path count. The normalisation range comes from "
                        f"the data rather than two hand-picked probe positions.",
                "panels": panels})

    return groups


def render(groups) -> str:
    """HTML for the collected groups."""
    if not groups:
        return ""
    out = []
    for g in groups:
        cells = []
        for p in g["panels"]:
            rows = "".join(
                f'<tr><th>{k}</th><td>{v}</td></tr>' for k, v in p["params"].items())
            cells.append(
                f'<figure class="panel"><img src="{p["src"]}" alt="{p["title"]}"/>'
                f'<figcaption>{p["title"]}</figcaption>'
                f'<table class="params">{rows}</table></figure>')
        out.append(
            f'<section><div class="sec-head"><h2>{g["title"]}</h2>'
            f'<p>{g["note"]}</p></div>'
            f'<div class="panels">{"".join(cells)}</div></section>')
    return "".join(out)


CSS = """
.panels { display:grid; grid-template-columns:repeat(auto-fit,minmax(260px,1fr));
  gap:14px; }
.panel { background:var(--surface); border:1px solid var(--line); border-radius:4px;
  padding:10px; margin:0; }
.panel img { width:100%; border-radius:3px; display:block; }
.panel figcaption { font-family:"IBM Plex Mono",monospace; font-size:11px;
  color:var(--ink); margin:7px 0 5px; font-weight:500; }
table.params { width:100%; border-collapse:collapse; min-width:0; }
table.params th, table.params td { border:0; padding:1px 0; font-size:10.5px;
  font-family:"IBM Plex Mono",monospace; }
table.params th { text-align:left; color:var(--muted); font-weight:400;
  padding-right:8px; white-space:nowrap; }
table.params td { text-align:right; color:var(--ink); }
"""
