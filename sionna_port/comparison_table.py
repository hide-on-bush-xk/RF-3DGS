"""Render the comparison manifest as a table of images.

Rows are sources and settings, columns are array-processing algorithms. A cell
this port cannot produce says so, because an empty cell reads as a rendering
failure rather than as work not yet done.
"""

from __future__ import annotations

import os

from reference_panels import embed_image

CSS = """
.cmp-scroll { overflow-x: auto; border:1px solid var(--line); border-radius:4px;
  background:var(--surface); }
table.cmp { border-collapse:separate; border-spacing:0; width:100%;
  min-width:920px; }
table.cmp th, table.cmp td { border:0; border-bottom:1px solid var(--line);
  padding:9px 8px; vertical-align:top; }
table.cmp thead th { position:sticky; top:0; background:var(--sunk);
  text-align:center; font-family:"IBM Plex Mono",monospace; font-weight:500;
  font-size:11px; color:var(--ink); white-space:nowrap; }
table.cmp thead th small { display:block; font-weight:400; font-size:9.5px;
  color:var(--muted); letter-spacing:0; margin-top:2px; }
table.cmp tbody th { text-align:left; width:150px; min-width:150px;
  font-family:"IBM Plex Mono",monospace; font-size:11.5px; font-weight:500;
  color:var(--ink); background:var(--sunk); }
table.cmp tbody th dl { margin:6px 0 0; }
table.cmp tbody th dl div { display:flex; justify-content:space-between; gap:6px;
  font-size:10px; font-weight:400; }
table.cmp tbody th dt { color:var(--muted); }
table.cmp tbody th dd { margin:0; color:var(--ink); font-variant-numeric:tabular-nums; }
table.cmp td img { width:100%; min-width:120px; border-radius:3px; display:block; }
table.cmp td .cellparams { font-family:"IBM Plex Mono",monospace; font-size:9px;
  color:var(--muted); margin-top:4px; line-height:1.35; }
table.cmp td.gap { text-align:center; color:var(--muted);
  font-family:"IBM Plex Mono",monospace; font-size:10.5px; }
table.cmp td.gap span { display:inline-block; border:1px dashed var(--line);
  border-radius:3px; padding:22px 8px; width:100%; }
table.cmp tbody tr:last-child th, table.cmp tbody tr:last-child td { border-bottom:0; }
"""


def render(payload, max_width: int = 260) -> str:
    cols = payload["columns"]
    rows = payload["rows"]
    root = payload.get("root", ".")
    out_base = payload.get("out_base", ".")

    head = ['<th style="width:150px"></th>']
    for c in cols:
        ported = "" if c["ported"] else "<small>not ported</small>"
        head.append(f'<th>{c["name"]}<small>{c["algorithm"]}</small>{ported}</th>')

    body = []
    for row in rows:
        meta = "".join(f"<div><dt>{k}</dt><dd>{v}</dd></div>"
                       for k, v in row.get("params", {}).items())
        cells = [f'<th>{row["label"]}<dl>{meta}</dl></th>']
        for c in cols:
            cell = row.get("cells", {}).get(c["name"])
            if not cell:
                cells.append('<td class="gap"><span>&mdash;</span></td>')
                continue
            if "missing" in cell:
                cells.append(f'<td class="gap"><span>{cell["missing"]}</span></td>')
                continue
            # Released images are relative to the repo; generated ones to the
            # output directory. A cell may override its row: the optical view of
            # a released pose is rendered here, so it lives with our output even
            # though the rest of that row does not.
            which = cell.get("base", row.get("base"))
            base = root if which == "root" else out_base
            src = embed_image(os.path.join(base, cell["image"]), max_width)
            if not src:
                cells.append('<td class="gap"><span>file missing</span></td>')
                continue
            params = "<br>".join(f"{k}: {v}"
                                 for k, v in cell.get("params", {}).items())
            cells.append(f'<td><img src="{src}" alt="{row["label"]} {c["name"]}"/>'
                         f'<div class="cellparams">{params}</div></td>')
        body.append(f"<tr>{''.join(cells)}</tr>")

    return (
        '<section><div class="sec-head"><h2>Every spectrum, every setting</h2>'
        '<p>Down a column: the released ground truth against each scattering '
        'setting, for one array-processing algorithm. Across a row: the '
        'algorithms at one setting. The top row is what the authors shipped; the '
        'rows below are this port. Four columns are still gaps -- AoD, Delay and '
        'MPC project paths onto the sphere rather than beamforming, and TCBF '
        'needs the Hann taper, none of which are ported yet.</p></div>'
        f'<div class="cmp-scroll"><table class="cmp">'
        f'<thead><tr>{"".join(head)}</tr></thead>'
        f'<tbody>{"".join(body)}</tbody></table></div></section>')
