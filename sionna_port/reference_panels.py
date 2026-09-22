"""The optical view of the scene, as context for the spectra.

The spectra themselves are cross-tabulated in comparison_table.py; showing them
here as well only duplicated that. What this contributes is the one thing the
table cannot: what the room actually looks like.

embed_image is also the shared image helper the other panel modules import.
"""

from __future__ import annotations

import base64
import io
import json
import os

MAX_WIDTH = 520


def embed_image(path: str, max_width: int = MAX_WIDTH) -> str | None:
    """Base64 PNG data URI, converting and downscaling whatever is on disk.

    Returns None rather than raising for a missing file or a missing PIL, so a
    caller can render a "file missing" placeholder instead of failing the whole
    dashboard. Downscaling matters: the dashboard inlines every image, so
    full-resolution PNGs would make the HTML hundreds of megabytes.
    """
    try:
        from PIL import Image
    except ImportError:
        return None
    if not os.path.isfile(path):
        return None
    with Image.open(path) as im:
        im = im.convert("RGB")           # handles TIFF, palette PNG, RGBA alike
        if im.width > max_width:
            # Height follows the width, so the aspect ratio is preserved.
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
    """The authors' own results.json for one spectrum type, or {} if absent.

    Every failure path returns {}: a missing or malformed metrics file should
    leave a panel without numbers, not stop the dashboard being built.
    """
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
    """The optical renders. Missing files are skipped, not faked."""
    groups = []

    # 1. the room, as a camera sees it
    blender = os.path.join(root, "RF-3DGS_dataset", "blender_visual_dataset", "train")
    if os.path.isdir(blender):
        names = sorted(f for f in os.listdir(blender)
                       if f.lower().endswith((".tif", ".png", ".jpg")))
        # Three views spread evenly through the set rather than the first three,
        # which would all be neighbouring poses showing the same corner.
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
            # Panels whose image failed to load are dropped here, so render()
            # never emits an <img> with a None src.
            "panels": [p for p in panels if p["src"]]})

    return groups


def render(groups) -> str:
    """HTML for the collected groups."""
    if not groups:
        return ""
    out = []
    for g in groups:
        cells = []
        for p in g["panels"]:
            # Each panel carries its own provenance table, so a reader can tell
            # what produced the image without leaving the page.
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


# auto-fit with a 240px minimum: the panel count per row follows the window
# width instead of being fixed.
CSS = """
.panels { display:grid; grid-template-columns:repeat(auto-fit,minmax(240px,1fr));
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
