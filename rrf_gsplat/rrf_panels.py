"""Dashboard panels for the stage-2 experiments: the RRF on gsplat.

Reads output/rrf/summary.json (from summarize.py) plus the optional timing
files in the same directory, and renders one section: colour-function and
normalisation comparisons, ablations, the time-to-quality curves for a moved
transmitter (cold against warm start), and the per-transmitter pipeline cost
of the original tutorial against this port. Inline SVG on the dashboard's
tokens, like planning_panels.py.
"""

from __future__ import annotations

import json
import os

CSS = """
.rrf-note { color:var(--muted); font-size:13px; margin:2px 0 10px; }
.rrf-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(420px,1fr)); gap:14px; }
.rrf-grid .wide { grid-column:1 / -1; }
.rrf-grid .tablewrap table { min-width:0; }
"""

GROUPS = {
    "colour": ("Colour function", "e2_"),
    "norm": ("Normalisation", "e1_"),
    "ablate": ("Ablations (dB mode)", "a_"),
    "txmove": ("Transmitter moved", "t_"),
    "baseline": ("Baseline on the released data", "c0_"),
}


def collect(rrf_dir):
    found = {}
    for key, name in (("summary", "summary.json"), ("tut019", "tutorial_019_timing.json"),
                      ("inria", "inria_timing.json"), ("gen", "generation_timing.json")):
        p = os.path.join(rrf_dir, name)
        if os.path.exists(p):
            with open(p, encoding="utf-8") as fid:
                found[key] = json.load(fid)
    return found


def _label(r):
    bits = [r["mode"]]
    if r["sh_degree"] != 3:
        bits.append(f"SH{r['sh_degree']}")
    if r["opacity"] == "frozen":
        bits.append("opacity frozen")
    if r["n_train"] < 2000:
        bits.append(f"{r['n_train']} views")
    if r["iterations"] != 10000:
        bits.append(f"{r['iterations']//1000}k it")
    if r["warm"]:
        bits.append("warm")
    src = r["source"].replace("3dgs_", "").replace("_100", "")
    return f"{src} · " + ", ".join(bits)


def hbars(rows, key, unit, fmt="{:.2f}", lower_better=False, width=560):
    """One horizontal bar per run, value printed at the end."""
    if not rows:
        return ""
    row_h, pad_l, pad_t = 24, 230, 10
    height = pad_t + row_h * len(rows) + 8
    vals = [r[key] for r in rows]
    vmax = max(vals) or 1.0
    pw = width - pad_l - 70
    best = (min if lower_better else max)(vals)
    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{key}" style="width:100%;height:auto">']
    for i, r in enumerate(rows):
        y = pad_t + i * row_h
        w = max(pw * r[key] / vmax, 1.0)
        fill = "var(--s2)" if r[key] == best else "var(--s1)"
        parts.append(f'<text x="{pad_l-8}" y="{y+15}" text-anchor="end" font-size="11" '
                     f'fill="var(--ink)">{_label(r)}</text>')
        parts.append(f'<rect x="{pad_l}" y="{y+5}" width="{w:.1f}" height="13" rx="3" fill="{fill}">'
                     f'<title>{r["run"]}: {fmt.format(r[key])} {unit}</title></rect>')
        parts.append(f'<text x="{pad_l+w+6:.1f}" y="{y+15}" font-size="11" fill="var(--muted)">'
                     f'{fmt.format(r[key])}</text>')
    parts.append("</svg>")
    return "".join(parts)


def curves(rows, key="psnr_rgb", unit="dB", width=560, height=240):
    """Running eval against wall time for up to four runs."""
    rows = [r for r in rows if r.get("history")][:4]
    if not rows:
        return ""
    pad_l, pad_r, pad_t, pad_b = 48, 12, 14, 30
    pw, ph = width - pad_l - pad_r, height - pad_t - pad_b
    xs = [h["seconds"] for r in rows for h in r["history"]]
    ys = [h[key] for r in rows for h in r["history"]]
    xmax, ymin, ymax = max(xs) or 1.0, min(ys), max(ys)
    if ymax - ymin < 1e-9:
        ymax = ymin + 1.0

    def px(v):
        return pad_l + v / xmax * pw

    def py(v):
        return pad_t + ph - (v - ymin) / (ymax - ymin) * ph

    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{key} over time" '
             f'style="width:100%;height:auto">']
    for i in range(4):
        frac = i / 3
        yy = pad_t + ph - frac * ph
        parts.append(f'<line x1="{pad_l}" x2="{width-pad_r}" y1="{yy:.1f}" y2="{yy:.1f}" stroke="var(--line)"/>')
        parts.append(f'<text x="{pad_l-6}" y="{yy+3.5:.1f}" text-anchor="end" font-size="10" '
                     f'fill="var(--muted)">{ymin + frac*(ymax-ymin):.1f}</text>')
    for k, r in enumerate(rows):
        pts = " ".join(f"{px(h['seconds']):.1f},{py(h[key]):.1f}" for h in r["history"])
        parts.append(f'<polyline points="{pts}" fill="none" stroke="var(--s{k+1})" stroke-width="2"/>')
        last = r["history"][-1]
        parts.append(f'<text x="{px(last["seconds"])-4:.1f}" y="{py(last[key])-6:.1f}" text-anchor="end" '
                     f'font-size="10" fill="var(--s{k+1})">{_label(r)}</text>')
    parts.append(f'<text x="{pad_l}" y="{height-8}" font-size="10" fill="var(--muted)">0 s</text>')
    parts.append(f'<text x="{width-pad_r}" y="{height-8}" text-anchor="end" font-size="10" '
                 f'fill="var(--muted)">{xmax:.0f} s of training</text>')
    parts.append(f'<text x="{width-pad_r}" y="{pad_t-3}" text-anchor="end" font-size="10" '
                 f'fill="var(--muted)">{unit}</text>')
    parts.append("</svg>")
    return "".join(parts)


def _meter(value, label, unit=""):
    unit_html = f"<i> {unit}</i>" if unit else ""
    return f'<div class="meter"><b>{value}{unit_html}</b><span>{label}</span></div>'


def _table(head, body_rows):
    h = "".join(f"<th>{c}</th>" for c in head)
    b = "".join("<tr>" + "".join((f"<th>{c}</th>" if i == 0 else f"<td>{c}</td>")
                                 for i, c in enumerate(row)) + "</tr>" for row in body_rows)
    return f'<div class="tablewrap"><table><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table></div>'


def render(found):
    runs = found.get("summary") or []
    if not runs and not found.get("tut019"):
        return ""
    by = {g: [r for r in runs if r["run"].startswith(pre)] for g, (_, pre) in GROUPS.items()}
    meters, cards = [], []

    c0 = next((r for r in by["baseline"]), None)
    if c0:
        meters.append(_meter(f"{c0['psnr_rgb']:.2f}", "gsplat RRF on the released MVDR data, PSNR (published 16.02)", "dB"))
    if by["colour"]:
        best = min(by["colour"], key=lambda r: r["rmse_db"])
        rgb = next((r for r in by["colour"] if r["mode"] == "rgb"), None)
        if rgb:
            meters.append(_meter(f"{rgb['rmse_db']:.2f} &rarr; {best['rmse_db']:.2f}",
                                 f"dB RMSE, jet RGB &rarr; {best['mode']} colour", "dB"))
        cards.append(
            f'<figure class="card"><h2>Colour function</h2>'
            f'<p class="rrf-note">Same regenerated MVDR data, geometry and iterations; only what a Gaussian '
            f'carries changes: a jet RGB triple (RF-3DGS), the dB value, or a linear power that composites '
            f'and is read back in dB. Scored in dB against the float truth; orange = best.</p>'
            f'{hbars(by["colour"], "rmse_db", "dB", lower_better=True)}'
            f'<figcaption>RMSE in dB, lower is better &middot; PSNR after jet mapping: '
            + ", ".join(f"{r['mode']} {r['psnr_rgb']:.2f}" for r in by["colour"]) + '</figcaption></figure>')
    if by["norm"]:
        cards.append(
            f'<figure class="card"><h2>Normalisation</h2>'
            f'<p class="rrf-note">The same float spectra mapped once with one global range and once per image '
            f'by its own min/max (what the tutorial does for CBF and TCBF). The per-view model is scored with '
            f'the oracle range of each test image, the best it could ever do; without it the absolute level '
            f'is simply not in the picture.</p>'
            f'{hbars(by["norm"], "rmse_db", "dB", lower_better=True)}'
            f'<figcaption>RMSE in dB against the float truth</figcaption></figure>')
    if by["ablate"]:
        ref = next((r for r in by["colour"] if r["mode"] == "db"), None)
        rows = ([ref] if ref else []) + by["ablate"]
        body = [[_label(r), f"{r['psnr_rgb']:.2f}", f"{r['ssim_rgb']:.3f}",
                 f"{r['rmse_db']:.2f}", f"{r['train_s']:.0f}"] for r in rows]
        cards.append(
            '<figure class="card"><h2>Ablations on the dB model</h2>'
            '<p class="rrf-note">SH degree, frozen opacity, fewer training views, fewer iterations.</p>'
            + _table(["run", "PSNR (jet)", "SSIM", "RMSE dB", "train s"], body)
            + '</figure>')
    if by["txmove"]:
        cold = [r for r in by["txmove"] if not r["warm"]]
        warm = [r for r in by["txmove"] if r["warm"]]
        if cold and warm:
            c, w = cold[0], warm[0]
            k = next(k for k in c if k.startswith("s_to_psnr"))
            if c.get(k) and w.get(k):
                meters.append(_meter(f"{c[k]:.0f} &rarr; {w[k]:.0f}", f"seconds to PSNR {k[9:]} after the Tx moved, cold &rarr; warm", "s"))
        cards.append(
            f'<figure class="card"><h2>Transmitter moved: cold against warm start</h2>'
            f'<p class="rrf-note">A new dataset for a second transmitter position; the RRF is fitted from '
            f'zeroed colours (as RF-3DGS must) and from the first transmitter\'s colours. Running eval on 64 '
            f'held-out views against wall time.</p>'
            f'{curves(cold[:2] + warm[:2])}<figcaption>PSNR after jet mapping</figcaption></figure>')

    # pipeline cost per transmitter position
    tut, gen, inria = found.get("tut019"), found.get("gen"), found.get("inria")
    if tut or gen:
        rows = []
        if tut:
            pv = tut["per_view_mean"]
            rows.append(["RF-3DGS tutorial (Sionna 0.19, TensorFlow, CPU)",
                         f"{pv['compute_paths_s']:.1f} s + {pv['mvdr_spectrum_s']:.1f} s",
                         f"{tut['estimated_800_positions_hours']:.1f} h",
                         f"{inria['seconds_10k']/60:.1f} min" if inria else "&ndash;"])
        if gen:
            for g in gen:
                rows.append([g["label"], f"{g['seconds_per_view']:.3f} s", f"{g['seconds_3200']/60:.1f} min",
                             g.get("train_note", "&ndash;")])
        cards.append(
            f'<figure class="card wide"><h2>Cost of moving the transmitter</h2>'
            f'<p class="rrf-note">Everything the pipeline has to redo when the transmitter moves: the dataset '
            f'(3200 views) and the RRF fine-tune. The tutorial timing is its own notebook code run unchanged '
            f'on this machine\'s CPU (no OptiX under WSL, no GPU for TensorFlow 2.15 on Windows); the port '
            f'runs on the same GPU the training uses.</p>'
            f'{_table(["pipeline", "per view (paths + spectrum)", "3200 views", "RRF fine-tune 10k it"], rows)}'
            f'</figure>')

    return f"""
<section>
  <div class="sec-head"><h2>Stage 2: the radiance field on gsplat</h2>
  <p>RF-3DGS fine-tunes colour and opacity of a frozen visual 3DGS on jet PNGs through an RGB
  rasteriser. On gsplat the target can be the spectrum itself, so the colour function and the
  normalisation become experiments, every model is scored in dB against float truth, and the
  cost of a moved transmitter has a number. Source: <code>rrf_gsplat/</code>.</p></div>
  <div class="meters">{''.join(meters)}</div>
  <div class="rrf-grid" style="margin-top:14px">{''.join(cards)}</div>
</section>
"""
