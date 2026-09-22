"""Dashboard panels for transmitter-side planning.

Reads what the tx_planning scripts write to output/tx_planning and renders one
section: the coverage sweep as a floor-plan heat map with the gradient
optimiser's path drawn over it, the optimiser's progress, the material fit,
and the next-measurement ranking. Inline SVG on the dashboard's CSS tokens,
like the rest of sionna_port/dashboard.py.

Every panel is optional: collect() returns only what exists on disk and
render() emits only the cards it has data for, so a partial set of runs still
produces a valid section.
"""

from __future__ import annotations

import json
import math
import os

import numpy as np

# Injected into the dashboard's stylesheet; var(--...) tokens are defined there.
CSS = """
.plan-map svg { width:auto; max-width:100%; max-height:640px; height:auto; display:block; }
.plan-note { color:var(--muted); font-size:13px; margin:2px 0 10px; }
.plan-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(420px,1fr)); gap:14px; }
.plan-grid .wide { grid-column:1 / -1; }
.plan-grid .plan-map { grid-row:span 2; }
.plan-table .tablewrap table { min-width:0; }
.plan-table .tablewrap { align-self:start; }
"""

# Sequential ramp for the heat map, light to dark, one hue.
# One hue on a monotone lightness scale, so the map survives greyscale printing
# and does not imply categories the data does not have.
RAMP = [(0xe3, 0xef, 0xfa), (0x9c, 0xc3, 0xe8), (0x4a, 0x8f, 0xd3), (0x1f, 0x5c, 0xa8), (0x0b, 0x2f, 0x66)]

# Accepted filenames per panel, in preference order: the plain name first, then
# the variants later runs produced.
FILES = {
    "sweep": ["tx_sweep.npz", "tx_sweep_batched.npz"],
    "opt": ["optimize_tx.json", "optimize_tx_refine.json"],
    "fit": ["fit_materials.json", "fit_materials_pdp.json"],
    "active": ["active_measurement.json"],
}


def collect(directory):
    """Whatever planning outputs exist in `directory`, first name wins."""
    found = {}
    for key, names in FILES.items():
        for name in names:
            path = os.path.join(directory, name)
            if os.path.exists(path):
                if name.endswith(".npz"):
                    found[key] = dict(np.load(path, allow_pickle=True))
                else:
                    with open(path, encoding="utf-8") as fid:
                        found[key] = json.load(fid)
                # The filename is kept so each card can cite its source.
                found[key + "_file"] = name
                break
    return found


def _ramp(t):
    """Interpolate RAMP at t in [0, 1] and return a #rrggbb string."""
    t = min(max(t, 0.0), 1.0) * (len(RAMP) - 1)
    # Clamped to len-2 so i+1 is always a valid index, including at t = 1.
    i = min(int(t), len(RAMP) - 2)
    f = t - i
    c = [round(RAMP[i][k] + (RAMP[i + 1][k] - RAMP[i][k]) * f) for k in range(3)]
    return "#%02x%02x%02x" % tuple(c)


def coverage_map(sweep, opt=None, threshold=-85.0, width=640):
    """Floor plan: received power from the sweep's best Tx, candidates, and
    the optimiser's path."""
    rx = np.asarray(sweep["rx_positions"], dtype=float)
    tx = np.asarray(sweep["tx_positions"], dtype=float)
    gain = np.asarray(sweep["gain_db"], dtype=float)
    # NaN means no path, which must count as uncovered.
    covered = np.nan_to_num(gain, nan=-999.0) > threshold
    cov = covered.mean(axis=1)
    best = int(np.argmax(cov))       # the candidate whose row the map shows
    g = gain[best]

    # The grid step is recovered from the data rather than passed in, so the map
    # works for any sweep resolution.
    xs = np.unique(rx[:, 0])
    step = float(np.min(np.diff(xs))) if len(xs) > 1 else 1.0
    # Extend by half a cell so the drawn squares are centred on their points.
    x0, x1 = rx[:, 0].min() - step / 2, rx[:, 0].max() + step / 2
    y0, y1 = rx[:, 1].min() - step / 2, rx[:, 1].max() + step / 2
    pad = 28
    scale = (width - 2 * pad) / (x1 - x0)      # uniform: aspect ratio is preserved
    height = (y1 - y0) * scale + 2 * pad + 30  # the +30 is the legend strip

    def px(x):
        return pad + (x - x0) * scale

    def py(y):
        # y1 - y, because SVG's y axis points down and the world's points up.
        return pad + (y1 - y) * scale

    # 2nd-98th percentile rather than min/max: a single very weak point would
    # otherwise compress the whole colour range.
    finite = g[np.isfinite(g)]
    vmin, vmax = (float(np.percentile(finite, 2)), float(np.percentile(finite, 98))) \
        if finite.size else (-100.0, -60.0)
    parts = [f'<svg viewBox="0 0 {width} {height:.0f}" role="img" '
             f'aria-label="coverage map from the best swept transmitter">']
    parts.append(f'<rect x="0" y="0" width="{width}" height="{height:.0f}" fill="var(--surface)"/>')
    s = step * scale
    for (x, y, _), v in zip(rx, g):
        if np.isfinite(v):
            # `or 1.0` guards a constant map, where vmax == vmin.
            fill = _ramp((v - vmin) / ((vmax - vmin) or 1.0))
            title = f"({x:.1f}, {y:.1f}) {v:.1f} dB"
        else:
            # Unreached points are drawn in the sunk colour, not at the bottom of
            # the ramp: "no path" is categorically different from "weak".
            fill, title = "var(--sunk)", f"({x:.1f}, {y:.1f}) no path"
        # s-1 leaves a hairline gap so the cells read as a grid.
        parts.append(f'<rect x="{px(x)-s/2:.1f}" y="{py(y)-s/2:.1f}" width="{s-1:.1f}" '
                     f'height="{s-1:.1f}" fill="{fill}"><title>{title}</title></rect>')
    # candidate transmitters, best one filled
    for i, (x, y, _) in enumerate(tx):
        if i == best:
            continue                  # drawn separately below, larger and filled
        parts.append(f'<circle cx="{px(x):.1f}" cy="{py(y):.1f}" r="4" fill="var(--surface)" '
                     f'stroke="var(--muted)" stroke-width="1.5"><title>candidate Tx '
                     f'({x:.1f}, {y:.1f}) coverage {cov[i]:.0%}</title></circle>')
    bx, by = tx[best][:2]
    parts.append(f'<circle cx="{px(bx):.1f}" cy="{py(by):.1f}" r="6" fill="var(--s2)" '
                 f'stroke="var(--surface)" stroke-width="2"><title>best swept Tx '
                 f'({bx:.1f}, {by:.1f}) coverage {cov[best]:.0%}</title></circle>')
    # optimiser path
    if opt and opt.get("history"):
        pts = [(h["position"][0], h["position"][1]) for h in opt["history"]]
        poly = " ".join(f"{px(x):.1f},{py(y):.1f}" for x, y in pts)
        parts.append(f'<polyline points="{poly}" fill="none" stroke="var(--s3)" '
                     f'stroke-width="2.5" stroke-linejoin="round"/>')
        # Hollow marker at the start, filled at the end, so direction is readable.
        parts.append(f'<circle cx="{px(pts[0][0]):.1f}" cy="{py(pts[0][1]):.1f}" r="4" '
                     f'fill="var(--surface)" stroke="var(--s3)" stroke-width="2"/>')
        parts.append(f'<circle cx="{px(pts[-1][0]):.1f}" cy="{py(pts[-1][1]):.1f}" r="5" '
                     f'fill="var(--s3)" stroke="var(--surface)" stroke-width="2"/>')
    # legend
    # 40 slices of the ramp, drawn as a continuous bar with its two end labels.
    ly = height - 22
    for i in range(40):
        parts.append(f'<rect x="{pad + i*3}" y="{ly}" width="3" height="8" '
                     f'fill="{_ramp(i/39)}"/>')
    parts.append(f'<text x="{pad}" y="{ly+22}" font-size="13" fill="var(--muted)">{vmin:.0f} dB</text>')
    parts.append(f'<text x="{pad+120}" y="{ly+22}" text-anchor="end" font-size="13" '
                 f'fill="var(--muted)">{vmax:.0f} dB</text>')
    parts.append(f'<text x="{width-pad}" y="{ly+22}" text-anchor="end" font-size="13" '
                 f'fill="var(--muted)">x {x0+step/2:.0f}..{x1-step/2:.0f} m, y {y0+step/2:.0f}..{y1-step/2:.0f} m; '
                 f'grey = no path</text>')
    parts.append("</svg>")
    return "".join(parts), best, float(cov[best]), len(tx), len(rx)


def steps_chart(history, key, unit, scale=1.0, width=520, height=200):
    """One series against optimiser step."""
    y = [h[key] * scale for h in history]
    x = list(range(1, len(y) + 1))
    pad_l, pad_r, pad_t, pad_b = 56, 12, 14, 30
    pw, ph = width - pad_l - pad_r, height - pad_t - pad_b
    ymin, ymax = min(y), max(y)
    # A flat series would divide by zero below; give it an arbitrary unit range.
    if ymax - ymin < 1e-9:
        ymax = ymin + 1.0

    def px(v):
        # max(len-1, 1) keeps a single-point series from dividing by zero.
        return pad_l + (v - 1) / max(len(y) - 1, 1) * pw

    def py(v):
        return pad_t + ph - (v - ymin) / (ymax - ymin) * ph      # inverted axis

    # Decimal places chosen from the range, so a series spanning 0.001 and one
    # spanning 1000 both get readable tick labels.
    digits = max(0, 2 - int(math.floor(math.log10(ymax - ymin))))
    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{key} per step" '
             f'style="width:100%;height:auto">']
    for i in range(4):          # four gridlines, including both ends
        frac = i / 3
        yy = pad_t + ph - frac * ph
        parts.append(f'<line x1="{pad_l}" x2="{width-pad_r}" y1="{yy:.1f}" y2="{yy:.1f}" '
                     f'stroke="var(--line)"/>')
        parts.append(f'<text x="{pad_l-8}" y="{yy+3.5:.1f}" text-anchor="end" font-size="10" '
                     f'fill="var(--muted)">{ymin + frac*(ymax-ymin):,.{digits}f}</text>')
    pts = " ".join(f"{px(a):.1f},{py(b):.1f}" for a, b in zip(x, y))
    parts.append(f'<polyline points="{pts}" fill="none" stroke="var(--s3)" stroke-width="2"/>')
    # A dot per step, each carrying its exact value as a tooltip.
    for a, b in zip(x, y):
        parts.append(f'<circle cx="{px(a):.1f}" cy="{py(b):.1f}" r="2.5" fill="var(--s3)">'
                     f'<title>step {a}: {b:.2f} {unit}</title></circle>')
    parts.append(f'<text x="{pad_l}" y="{height-8}" font-size="10" fill="var(--muted)">step 1</text>')
    parts.append(f'<text x="{width-pad_r}" y="{height-8}" text-anchor="end" font-size="10" '
                 f'fill="var(--muted)">step {len(y)}</text>')
    parts.append(f'<text x="{width-pad_r}" y="{pad_t-3}" text-anchor="end" font-size="10" '
                 f'fill="var(--muted)">{unit}</text>')
    parts.append("</svg>")
    return "".join(parts)


def material_bars(fit, width=520):
    """Truth against fitted, one row per identifiable material, plus the prior."""
    # Only the identifiable materials, when that list exists: a fitted value for
    # a material the data never constrained would be meaningless to plot.
    names = fit.get("identifiable") or fit["materials"]
    init = fit["config"]["init"]
    row_h, pad_l, pad_t = 26, 150, 14
    height = pad_t + row_h * len(names) + 26
    pw = width - pad_l - 16

    def px(v):
        return pad_l + v * pw        # the axis is a scattering coefficient, 0..1

    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="material fit" '
             f'style="width:100%;height:auto">']
    for t in (0.0, 0.25, 0.5, 0.75, 1.0):
        parts.append(f'<line x1="{px(t):.1f}" x2="{px(t):.1f}" y1="{pad_t}" '
                     f'y2="{height-22}" stroke="var(--line)"/>')
        parts.append(f'<text x="{px(t):.1f}" y="{height-8}" text-anchor="middle" font-size="10" '
                     f'fill="var(--muted)">{t:g}</text>')
    # Dashed line at the prior: any bar that stayed there did not move.
    parts.append(f'<line x1="{px(init):.1f}" x2="{px(init):.1f}" y1="{pad_t}" y2="{height-22}" '
                 f'stroke="var(--muted)" stroke-dasharray="3 3"/>')
    for i, n in enumerate(names):
        y = pad_t + i * row_h
        truth, fitted = fit["truth"][n], fit["fitted"][n]
        parts.append(f'<text x="{pad_l-8}" y="{y+16}" text-anchor="end" font-size="11" '
                     f'fill="var(--ink)">{n[:20]}</text>')
        # Two stacked bars per material: truth above, fitted below.
        parts.append(f'<rect x="{pad_l}" y="{y+3}" width="{(px(truth)-pad_l):.1f}" height="8" rx="2" '
                     f'fill="var(--s1)"><title>{n} truth {truth:.3f}</title></rect>')
        parts.append(f'<rect x="{pad_l}" y="{y+13}" width="{(px(fitted)-pad_l):.1f}" height="8" rx="2" '
                     f'fill="var(--s2)"><title>{n} fitted {fitted:.3f}</title></rect>')
    parts.append("</svg>")
    return "".join(parts)


def _meter(value, label, unit=""):
    """One headline number for the row of meters above the cards."""
    unit_html = f"<i> {unit}</i>" if unit else ""
    return f'<div class="meter"><b>{value}{unit_html}</b><span>{label}</span></div>'


def render(found):
    """Assemble the whole section from whatever collect() found. "" if nothing."""
    if not found:
        return ""
    meters, cards = [], []

    sweep, opt = found.get("sweep"), found.get("opt")
    if sweep:
        # The optimiser's own threshold wins, so the map and the path agree.
        thr = -85.0
        if opt:
            thr = float(opt["config"].get("threshold_db", thr))
        svg, best, cov, n_tx, n_rx = coverage_map(sweep, opt, thr)
        meta = json.loads(str(sweep["meta"])) if "meta" in sweep else {}
        pos = np.asarray(sweep["tx_positions"])[best]
        meters.append(_meter(f"{cov:.0%}", f"best swept Tx coverage &gt; {thr:.0f} dB"))
        # The note states the sampling settings, so a reader can tell how much
        # Monte-Carlo noise is behind the picture.
        note = (f"Received power at {n_rx} indoor receiver points from the best of {n_tx} "
                f"candidate transmitter positions (filled marker; hollow markers are the "
                f"other candidates, hover for their coverage). Each candidate is one "
                f"solve with every receiver in the scene, {meta.get('samples', '?'):,} "
                f"samples, depth {meta.get('max_depth', '?')}.")
        if opt:
            h0, h1 = opt["history"][0], opt["history"][-1]
            # The wall-sliding behaviour is explained here because it is visible
            # in the drawn path and would otherwise look like a bug.
            note += (f" The green path is the gradient optimiser: from "
                     f"({h0['position'][0]:.1f}, {h0['position'][1]:.1f}) to "
                     f"({h1['position'][0]:.1f}, {h1['position'][1]:.1f}) in "
                     f"{len(opt['history'])} steps, coverage {h0['coverage_frac']:.0%} "
                     f"&rarr; {h1['coverage_frac']:.0%} on its own {opt['config']['rx_step']:g} m "
                     f"grid. It slides along the wall because a step through it is "
                     f"rejected and its wall-normal component dropped.")
        cards.append(f'<figure class="card plan-map"><h2>Coverage sweep and gradient placement</h2>'
                     f'<p class="plan-note">{note}</p>{svg}'
                     f'<figcaption>{found.get("sweep_file")}'
                     + (f' &middot; {found.get("opt_file")}' if opt else "") + '</figcaption></figure>')
    if opt:
        hist = opt["history"]
        meters.append(_meter(f"{hist[0]['coverage_frac']:.0%} &rarr; {hist[-1]['coverage_frac']:.0%}",
                             "gradient placement, coverage"))
        # Two charts side by side on purpose: the hard coverage is what matters
        # and the soft objective is what the gradient actually descends.
        cards.append(f'<figure class="card"><h2>Optimiser: hard coverage per step</h2>'
                     f'<p class="plan-note">Fraction of grid points above the threshold. It moves in '
                     f'jumps because a 0.1 m step at 60 GHz flips line-of-sight for whole '
                     f'points; the smooth objective below is what the gradient sees.</p>'
                     f'{steps_chart(hist, "coverage_frac", "%", 100.0)}</figure>')
        cards.append(f'<figure class="card"><h2>Optimiser: soft-coverage objective</h2>'
                     f'<p class="plan-note">Mean sigmoid of (P &minus; threshold) / '
                     f'{opt["config"].get("width_db", 5):g} dB over the grid, the differentiable '
                     f'stand-in for coverage. Monte-Carlo noise on it is &plusmn;0.001; the '
                     f'wobble is the landscape.</p>'
                     f'{steps_chart(hist, "objective", "soft coverage")}</figure>')

    fit = found.get("fit")
    if fit:
        r = fit["rmse_db"]
        ident = fit.get("identifiable", [])
        meters.append(_meter(f"{r['baseline_b']:.2f} &rarr; {r['fitted_b']:.2f}",
                             "held-out Tx RMSE after material fit", "dB"))
        # Recomputed here rather than read from the file, so the numbers in the
        # caption always match the bars actually drawn.
        err0 = np.mean([abs(fit["config"]["init"] - fit["truth"][n]) for n in ident]) if ident else float("nan")
        err1 = np.mean([abs(fit["fitted"][n] - fit["truth"][n]) for n in ident]) if ident else float("nan")
        cards.append(f'<figure class="card"><h2>Material fit from one transmitter</h2>'
                     f'<p class="plan-note">Scattering coefficient, truth (blue) against fitted '
                     f'(orange), for the {len(ident)} of {len(fit["materials"])} materials whose '
                     f'gradient at Tx A is at least 2% of the largest; dashed line is the prior '
                     f'every material started from. Mean error on these: {err0:.3f} &rarr; '
                     f'{err1:.3f}. The other {len(fit["materials"])-len(ident)} receive no usable '
                     f'gradient from this transmitter and stay at the prior. Fitted on '
                     f'{fit["config"]["pdp_bins"]} &times; {fit["config"]["pdp_bin_ns"]:g} ns PDP '
                     f'bins with {fit["config"]["obs_noise_db"]:g} dB observation noise.</p>'
                     f'<div class="legend"><span><i style="background:var(--s1)"></i>truth</span>'
                     f'<span><i style="background:var(--s2)"></i>fitted</span></div>'
                     f'{material_bars(fit)}<figcaption>{found.get("fit_file")}</figcaption></figure>')

    active = found.get("active")
    if active:
        arms = active["arms"]
        a_only = arms.get("A only", {}).get("rmse_b")
        chosen = arms.get("A + chosen C", {}).get("rmse_b")
        if a_only is not None and chosen is not None:
            meters.append(_meter(f"{a_only:.2f} &rarr; {chosen:.2f}",
                                 "held-out RMSE, adding the chosen 2nd Tx", "dB"))
        cands = sorted(active["candidates"], key=lambda c: -c["score"])     # best first
        rows = "".join(
            f"<tr><th>({c['pos'][0]:.0f}, {c['pos'][1]:.0f})</th><td>{c['score']:.2f}</td>"
            f"<td>{len(c['newly_seen'])}</td></tr>" for c in cands)
        arm_rows = "".join(
            f"<tr><th>{label}</th><td>{v['rmse_b']:.2f}</td><td>{v['err_ident']:.3f}</td>"
            f"<td>{v['n_ident']}</td></tr>" for label, v in arms.items())
        # The prior is prepended as the row every arm has to beat.
        arm_rows = (f"<tr><th>prior</th><td>{active['prior_rmse_b']:.2f}</td><td>&ndash;</td>"
                    f"<td>0</td></tr>") + arm_rows
        n_b = len(active["config"]["tx_b"])
        cards.append(
            f'<figure class="card wide plan-table"><h2>Where to measure next</h2>'
            f'<p class="plan-note">Candidate second-transmitter positions ranked by how much '
            f'their PDPs would reveal about the materials Tx A left unconstrained '
            f'(sensitivity from {active["config"]["probes"]} random projections through '
            f'the gradient, computed at the prior, before any measurement). Then the twin '
            f'is fitted from A plus the top candidate versus A plus the median one, and '
            f'scored at {n_b} held-out transmitters.</p>'
            f'<div class="plan-grid">'
            f'<div class="tablewrap"><table><thead><tr><th>candidate (x, y)</th><th>score</th>'
            f'<th>newly constrained materials</th></tr></thead><tbody>{rows}</tbody></table></div>'
            f'<div class="tablewrap"><table><thead><tr><th>fitted from</th>'
            f'<th>mean RMSE at held-out Tx (dB)</th><th>|err| identifiable</th>'
            f'<th>identifiable</th></tr></thead><tbody>{arm_rows}</tbody></table></div>'
            f'</div>'
            f'<figcaption>{found.get("active_file")}</figcaption></figure>')

    return f"""
<section>
  <div class="sec-head"><h2>Transmitter-side planning</h2>
  <p>RF-3DGS renders the channel from one fixed transmitter. These panels ask the
  reverse question on the same Sionna scene: where should the transmitter go, and
  what does the twin need to learn before it can say. Everything here comes from
  <code>tx_planning/</code>: a brute-force sweep, a placement driven by Sionna's
  own gradient, a material fit from one transmitter scored at another, and a
  choice of where to measure next.</p></div>
  <div class="meters">{''.join(meters)}</div>
  <div class="plan-grid" style="margin-top:14px">{''.join(cards)}</div>
</section>
"""
