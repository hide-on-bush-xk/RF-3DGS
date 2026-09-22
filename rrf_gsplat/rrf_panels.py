"""Dashboard panels for the stage-2 experiments: the RRF on gsplat.

Reads output/rrf/summary.json (from summarize.py) plus the optional timing
files in the same directory, and renders one section: colour-function and
normalisation comparisons, ablations, the time-to-quality curves for a moved
transmitter (cold against warm start), and the per-transmitter pipeline cost
of the original tutorial against this port. Inline SVG on the dashboard's
tokens, like planning_panels.py.

Every card is optional: collect() returns only the files that exist, and
render() emits a card only when its data is present, so a partially finished
experiment set still produces a valid section.
"""

from __future__ import annotations

import json
import math
import os

CSS = """
.rrf-note { color:var(--muted); font-size:13px; margin:2px 0 10px; }
.rrf-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(420px,1fr)); gap:14px; }
.rrf-grid .wide { grid-column:1 / -1; }
.rrf-grid .tablewrap table { min-width:0; }
"""

# Card title -> run-name prefix. Runs are grouped by the naming convention
# alone, so adding an experiment means naming it with the right prefix.
GROUPS = {
    "colour": ("Colour function", "e2_"),
    "norm": ("Normalisation", "e1"),          # e1_ (60 GHz) and e1b_ (2.4 GHz, the released setting)
    "ablate": ("Ablations (dB mode)", "a_"),
    "txmove": ("Transmitter moved", "t_"),
    "baseline": ("Baseline on the released data", "c0_"),
    "multi": ("Multi-channel targets", "m_"),
}


DENSITY_RUNS = {
    # (run name, training positions); the spacing axis is the measured nearest-training distance from the baselines file
    "scene 1 (lobby)": [("m_multi_24_tut_cs_depth", 640), ("m_multi_24_tut_cs_depth_v640", 160), ("m_multi_24_tut_cs_depth_v320", 80),
                        ("m_multi_24_tut_cs_depth_v160", 40), ("m_multi_24_tut_cs_depth_v80", 20)],
    "scene 2 (corridor)": [("s2_multi_corrM_depth_full", 451), ("s2_multi_corrM_depth_v900_full", 225), ("s2_multi_corrM_depth_v452_full", 113),
                           ("s2_multi_corrM_depth_v224_full", 56), ("s2_multi_corrM_depth_v112_full", 28)]     # *_full: all 452 held-out views scored,
}


def crossover(rows, ch):
    """Spacing at which copy - field changes sign (linear interpolation); '< first' / '> last' when it does not."""
    d = [(r["spacing"], r["copy"][ch] - r["field"][ch]) for r in rows]
    if d[0][1] > 0:
        return f"below {d[0][0]:.2f} m"
    for (x0, y0), (x1, y1) in zip(d[:-1], d[1:]):
        if y0 <= 0 < y1:
            return f"{x0 + (x1 - x0) * (-y0) / (y1 - y0):.2f} m"
    return f"beyond {d[-1][0]:.2f} m"


def collect(rrf_dir):
    """Whatever the experiment directory holds. Missing files are simply absent."""
    found = {}
    for key, name in (("summary", "summary.json"), ("tut019", "tutorial_019_timing.json"),
                      ("inria", "inria_timing.json"), ("gen", "generation_timing.json")):
        p = os.path.join(rrf_dir, name)
        if os.path.exists(p):
            with open(p, encoding="utf-8") as fid:
                found[key] = json.load(fid)
    enc = os.path.join(rrf_dir, "encoding_comparison.json")
    if os.path.exists(enc):
        with open(enc, encoding="utf-8") as fid:
            found["encoding"] = json.load(fid)
    for key, name in (("transfer", "transfer_curve.json"), ("transfer_2k", "transfer_curve_2k.json"), ("transfer_s2", "transfer_curve_s2.json")):
        tc = os.path.join(rrf_dir, name)
        if os.path.exists(tc):
            with open(tc, encoding="utf-8") as fid:
                found[key] = json.load(fid)
    # density sweeps: copy (nearest training position) against the field, per scene, on the measured spacing axis
    density = {}
    for scene, runs in DENSITY_RUNS.items():
        rows = []
        for run, npos in runs:
            p = os.path.join(rrf_dir, f"baselines_{run}.json")
            if not os.path.exists(p):
                continue
            with open(p, encoding="utf-8") as fid:
                b = json.load(fid)
            e = b["errors"]
            rows.append({"positions": npos, "spacing": b["nearest_train_distance_m"]["median"],
                         "copy": {c: e["nearest"][c]["median"] for c in ("az", "zen", "delay")},
                         "field": {c: e["rrf"][c]["median"] for c in ("az", "zen", "delay")},
                         "copy_p90": {c: e["nearest"][c]["p90"] for c in ("az", "zen", "delay")},
                         "field_p90": {c: e["rrf"][c]["p90"] for c in ("az", "zen", "delay")}})
        if len(rows) >= 2:
            density[scene] = sorted(rows, key=lambda r: r["spacing"])
    if density:
        found["density"] = density
    sota = os.path.join(rrf_dir, "sota_table.json")
    if os.path.exists(sota):
        with open(sota, encoding="utf-8") as fid:
            found["sota"] = json.load(fid)
    strip = os.path.join(rrf_dir, "compare_colour_modes.png")
    if os.path.exists(strip):
        import base64
        with open(strip, "rb") as fid:
            found["strip"] = "data:image/png;base64," + base64.b64encode(fid.read()).decode()
    return found


def _label(r):
    """A short human label for a run: only the settings that differ from the
    defaults are listed, so two runs' labels differ exactly where the runs do."""
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
    if r.get("densify", "none") != "none":
        bits.append(f"MCMC{'' if not r.get('gaussians') else ' ' + format(r['gaussians'], ',')}")
    elif r.get("geometry") == "trained":
        bits.append("geometry trained")
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
    # The best run is highlighted, and which direction is "best" depends on the
    # metric -- PSNR up, RMSE down.
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


def curves(rows, key="psnr_rgb", unit="dB", width=560, height=240, x="iteration"):
    """Running eval against iterations (or wall time) for up to four runs.

    Iterations by default: wall time depends on what else shared the GPU.
    """
    # Four at most: the CSS token set has four series colours, and more lines
    # than that stop being readable anyway.
    rows = [r for r in rows if r.get("history")][:4]
    if not rows:
        return ""
    pad_l, pad_r, pad_t, pad_b = 48, 12, 14, 30
    pw, ph = width - pad_l - pad_r, height - pad_t - pad_b
    xs = [h[x] for r in rows for h in r["history"]]
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
        pts = " ".join(f"{px(h[x]):.1f},{py(h[key]):.1f}" for h in r["history"])
        parts.append(f'<polyline points="{pts}" fill="none" stroke="var(--s{k+1})" stroke-width="2"/>')
    parts.append(f'<text x="{pad_l}" y="{height-8}" font-size="10" fill="var(--muted)">0</text>')
    parts.append(f'<text x="{width-pad_r}" y="{height-8}" text-anchor="end" font-size="10" '
                 f'fill="var(--muted)">{xmax:,.0f} {"iterations" if x == "iteration" else "s"}</text>')
    parts.append(f'<text x="{width-pad_r}" y="{pad_t-3}" text-anchor="end" font-size="10" '
                 f'fill="var(--muted)">{unit}</text>')
    parts.append("</svg>")
    legend = "".join(f'<span><i style="background:var(--s{k+1})"></i>{_label(r)}</span>'
                     for k, r in enumerate(rows))
    return f'<div class="legend">{legend}</div>' + "".join(parts)


def _meter(value, label, unit=""):
    """One headline number for the row of meters above the cards."""
    unit_html = f"<i> {unit}</i>" if unit else ""
    return f'<div class="meter"><b>{value}{unit_html}</b><span>{label}</span></div>'


def _table(head, body_rows):
    """A table where the first cell of each row is a header cell (the row label)."""
    h = "".join(f"<th>{c}</th>" for c in head)
    b = "".join("<tr>" + "".join((f"<th>{c}</th>" if i == 0 else f"<td>{c}</td>")
                                 for i, c in enumerate(row)) + "</tr>" for row in body_rows)
    return f'<div class="tablewrap"><table><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table></div>'


def transfer_card(tr, width=560, height=260):
    """In-range RMSE on Tx-B against the source transmitter's distance to Tx-B."""
    rows = tr["rows"]
    # The two reference lines drawn across the plot: the floor a transfer must
    # beat and the ceiling it cannot exceed.
    cold, own = tr["cold"]["rmse_db_in_range"], tr["own_unfrozen"]["rmse_db_in_range"]
    pad_l, pad_r, pad_t, pad_b = 48, 12, 14, 30
    pw, ph = width - pad_l - pad_r, height - pad_t - pad_b
    xmax = max(r["distance_m"] for r in rows) * 1.08
    ys = [r["rmse_in_range"] for r in rows] + [cold, own]
    ymin, ymax = min(ys) - 0.15, max(ys) + 0.15

    def px(v):
        return pad_l + v / xmax * pw

    def py(v):
        return pad_t + ph - (v - ymin) / (ymax - ymin) * ph

    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="transfer curve" style="width:100%;height:auto">']
    for i in range(4):
        frac = i / 3
        yy = pad_t + ph - frac * ph
        parts.append(f'<line x1="{pad_l}" x2="{width-pad_r}" y1="{yy:.1f}" y2="{yy:.1f}" stroke="var(--line)"/>')
        parts.append(f'<text x="{pad_l-6}" y="{yy+3.5:.1f}" text-anchor="end" font-size="10" fill="var(--muted)">{ymin + frac*(ymax-ymin):.2f}</text>')
    for v, lab, k in ((cold, "visual geometry, frozen", 2), (own, "Tx-B's own geometry, unfrozen", 3)):
        parts.append(f'<line x1="{pad_l}" x2="{width-pad_r}" y1="{py(v):.1f}" y2="{py(v):.1f}" stroke="var(--s{k})" stroke-width="1.5" stroke-dasharray="5 4"/>')
        parts.append(f'<text x="{width-pad_r}" y="{py(v)-4:.1f}" text-anchor="end" font-size="10" fill="var(--muted)">{lab} {v:.2f}</text>')
    fit = tr.get("fit")
    if fit and fit.get("with_offset"):
        fit = dict(fit["with_offset"])                      # the plateau model: b0 exp(-d/d_c) + c
    if fit:
        c_off = fit.get("c_db", 0.0)
        xs = [xmax * i / 60 for i in range(61)]
        pts = " ".join(f"{px(x):.1f},{py(min(max(cold - (fit['b0_db'] * math.exp(-x / fit['d_c_m']) + c_off), ymin), ymax)):.1f}" for x in xs)
        parts.append(f'<polyline points="{pts}" fill="none" stroke="var(--s1)" stroke-width="1.5" opacity="0.7"/>')
    for r in rows:
        parts.append(f'<circle cx="{px(r["distance_m"]):.1f}" cy="{py(r["rmse_in_range"]):.1f}" r="5" fill="var(--s1)" stroke="var(--bg)" stroke-width="2">'
                     f'<title>Tx-{r["source"]} at {r["tx"]}: {r["distance_m"]:.2f} m from Tx-B, in-range RMSE {r["rmse_in_range"]:.2f} dB</title></circle>')
        parts.append(f'<text x="{px(r["distance_m"]):.1f}" y="{py(r["rmse_in_range"])-8:.1f}" text-anchor="middle" font-size="10" fill="var(--muted)">{r["source"]}</text>')
    parts.append(f'<text x="{pad_l}" y="{height-8}" font-size="10" fill="var(--muted)">0</text>')
    parts.append(f'<text x="{width-pad_r}" y="{height-8}" text-anchor="end" font-size="10" fill="var(--muted)">{xmax:.0f} m from Tx-B</text>')
    parts.append(f'<text x="{width-pad_r}" y="{pad_t-3}" text-anchor="end" font-size="10" fill="var(--muted)">in-range RMSE, dB</text>')
    parts.append("</svg>")
    budget = tr.get("transfer_budget", "10k")
    z = tr.get("zones")
    label_in = (z or {}).get("label_in", "Same side of the lobby as Tx-B")
    label_out = (z or {}).get("label_out", "elsewhere")
    zone_txt = (f" {label_in} ({', '.join(z['east_sources'])}): mean {z['east_mean']:+.2f} dB, worst {z['east_min']:+.2f}; "
                f"{label_out}: mean {z['other_mean']:+.2f} dB, best {z['other_max']:+.2f}." if z and z.get("east_sources") else "")
    bs = (fit or {}).get("bootstrap")
    bs_txt = (f" Bootstrap 90 %: d_c {bs['d_c_p5']:.1f}&ndash;{bs['d_c_p95']:.1f} m ({bs['share_d_c_at_grid_max']:.0%} of resamples resolve no decay), "
              f"plateau {bs['c_p5']:+.2f}&ndash;{bs['c_p95']:+.2f} dB." if bs else "")
    cap = (f"{len(rows)} source transmitters, transfer budget {budget} steps.{zone_txt} Fit b(d) = b0 exp(&minus;d/d_c) + c: d_c = {fit['d_c_m']:.1f} m, "
           f"b0 = {fit['b0_db']:.2f} dB, plateau c = {fit.get('c_db', 0.0):+.2f} dB, R&sup2; {fit['r2']:.2f}.{bs_txt}"
           if fit else f"{len(rows)} source transmitters, transfer budget {budget} steps.{zone_txt} {tr.get('fit_note') or 'The fit needs 4 points.'}")
    scene = tr.get("scene")
    title = f"How far an adapted geometry carries ({budget}-step transfer{', ' + scene if scene else ''})"
    return (f'<figure class="card"><h2>{title}</h2>'
            f'<p class="rrf-note">Geometry unfrozen on a source transmitter (10k steps), then frozen on Tx-B with colours reset '
            f'and refit for {budget} steps (the 2k budget keeps the sign and order of the 10k points; seed noise 0.03 dB). '
            f'The benefit over the visual geometry falls with the source\'s distance to Tx-B, but not only with it: '
            f'it tracks how much of Tx-B\'s lit surface the source also lights.</p>'
            + "".join(parts) + f'<figcaption>{cap}</figcaption></figure>')


def density_card(density, width=560, height=200):
    """Median decoded error of copy and field against the measured spacing, one panel per channel, both scenes."""
    chans = (("az", "azimuth, deg"), ("zen", "zenith, deg"), ("delay", "delay, ns"))
    scenes = list(density)
    pad_l, pad_r, pad_t, pad_b = 40, 12, 14, 26
    pw, ph = width - pad_l - pad_r, height - pad_t - pad_b
    xmax = max(r["spacing"] for rows in density.values() for r in rows) * 1.06
    svgs = []
    for ch, unit in chans:
        ymax = max(max(r["copy"][ch], r["field"][ch]) for rows in density.values() for r in rows) * 1.08

        def px(v):
            return pad_l + v / xmax * pw

        def py(v):
            return pad_t + ph - v / ymax * ph

        parts = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{unit} against spacing" style="width:100%;height:auto">']
        for i in range(4):
            frac = i / 3
            yy = pad_t + ph - frac * ph
            parts.append(f'<line x1="{pad_l}" x2="{width-pad_r}" y1="{yy:.1f}" y2="{yy:.1f}" stroke="var(--line)"/>')
            parts.append(f'<text x="{pad_l-6}" y="{yy+3.5:.1f}" text-anchor="end" font-size="10" fill="var(--muted)">{frac*ymax:.1f}</text>')
        for si, scene in enumerate(scenes):
            dash = "" if si == 0 else ' stroke-dasharray="6 4"'
            for key, k in (("copy", 2), ("field", 1)):
                pts = " ".join(f"{px(r['spacing']):.1f},{py(r[key][ch]):.1f}" for r in density[scene])
                parts.append(f'<polyline points="{pts}" fill="none" stroke="var(--s{k})" stroke-width="2"{dash}/>')
                for r in density[scene]:
                    parts.append(f'<circle cx="{px(r["spacing"]):.1f}" cy="{py(r[key][ch]):.1f}" r="3.5" fill="var(--s{k})" stroke="var(--bg)" stroke-width="1.5">'
                                 f'<title>{scene}, {r["positions"]} training positions, spacing {r["spacing"]:.2f} m: {key} median {r[key][ch]:.2f}, P90 {r[key + "_p90"][ch]:.1f}</title></circle>')
        parts.append(f'<text x="{pad_l}" y="{height-8}" font-size="10" fill="var(--muted)">0</text>')
        parts.append(f'<text x="{width-pad_r}" y="{height-8}" text-anchor="end" font-size="10" fill="var(--muted)">{xmax:.1f} m to the nearest training position (median)</text>')
        parts.append(f'<text x="{width-pad_r}" y="{pad_t-3}" text-anchor="end" font-size="10" fill="var(--muted)">median error, {unit}</text>')
        parts.append("</svg>")
        svgs.append("".join(parts))
    legend = (f'<div class="legend"><span><i style="background:var(--s2)"></i>copy the nearest training position</span>'
              f'<span><i style="background:var(--s1)"></i>field (delay = residual + depth / c)</span>'
              + "".join(f'<span>{"solid" if si == 0 else "dashed"}: {scene}</span>' for si, scene in enumerate(scenes)) + "</div>")
    cap = " ".join(f"{scene}: crossover azimuth {crossover(rows, 'az')}, zenith {crossover(rows, 'zen')}, delay {crossover(rows, 'delay')} "
                   f"({rows[0]['positions']} &rarr; {rows[-1]['positions']} training positions, spacing {rows[0]['spacing']:.2f} &rarr; {rows[-1]['spacing']:.2f} m)."
                   for scene, rows in density.items())
    return (f'<figure class="card wide"><h2>Where the field beats a lookup table: training density, both scenes</h2>'
            f'<p class="rrf-note">Median decoded error on the same held-out images as the training set is thinned; the copy baseline uses the '
            f'same training subset. The spacing axis is measured (nearest training position, median): the corridor route has a 0.2 m step but '
            f'the generator applies the tutorial\'s &plusmn;0.5 m position jitter. Hover a point for its P90.</p>'
            + legend + f'<div class="rrf-grid">{"".join(svgs)}</div><figcaption>{cap}</figcaption></figure>')


SOTA_ROWS = [("released", "RF-3DGS as published (released checkpoint, 40k)"), ("inria", "RF-3DGS retrained here (fork's train.py)"),
             ("rgb", "ours: jet RGB target, frozen geometry, 10k"), ("db", "ours: value target, frozen, 10k"),
             ("rgb_geom", "ours: jet RGB, unfrozen geometry, 10k"), ("db_geom", "ours: value target, unfrozen geometry, 10k"),
             ("rgb_40k", "ours: jet RGB, frozen, 40k"), ("db_geom_40k", "ours: value target, unfrozen, 40k"), ("nerf2", "NeRF2 (their model, pinhole rays), 30k")]
SOTA_SPECTRA = ["MVDR", "CBF", "TCBF", "AoD", "Delay", "MPC"]


def sota_card(sota):
    """The released-benchmark table: released data, released split, the fork's metrics.py; best cell per column bold."""
    def best(S, key, lower):
        """Which row holds the best cell of this column, or None if the column
        is empty. `lower` because LPIPS is better when smaller."""
        vals = [(sota[k][S][key], k) for k, _ in SOTA_ROWS if S in sota.get(k, {})]
        return (min if lower else max)(vals)[1] if vals else None
    head = ["row"] + SOTA_SPECTRA
    body = []
    for kind, label in SOTA_ROWS:
        cells = []
        for S in SOTA_SPECTRA:
            d = sota.get(kind, {}).get(S)
            if not d:
                # Same distinction sota_table.py makes: N/A means the cell can
                # never exist, a dash means it has not been produced yet.
                cells.append("N/A" if (kind in ("db", "db_geom", "nerf2") and S in ("AoD", "Delay")) else "&ndash;"); continue
            parts = []
            for key, lower in (("PSNR", False), ("SSIM", False), ("LPIPS", True)):
                v = f"{d[key]:.2f}" if key == "PSNR" else f"{d[key]:.3f}"
                parts.append(f"<b>{v}</b>" if best(S, key, lower) == kind else v)
            cells.append(" / ".join(parts))
        body.append([label] + cells)
    return (f'<figure class="card wide"><h2>The RF-3DGS released benchmark, on its own coordinates</h2>'
            f'<p class="rrf-note">Released spectra, the released 2560 / 640 split, and the fork\'s metrics.py (PSNR / SSIM / VGG-LPIPS over the 640 '
            f'held-out views) for every row. The retrained-fork row reproduces the released checkpoints to the third decimal, so the table compares '
            f'methods, not runs. N/A: a value target has no meaning for the three-channel AoD / Delay encodings, and NeRF2\'s scalar head cannot '
            f'represent them. Best cell per column in bold.</p>' + _table(head, body)
            + '<figcaption>Every spectrum has a row that beats the release on all three metrics: MVDR +3.4 dB at 10k and +4.3 dB at 40k, CBF +0.9, '
              'TCBF +1.3 against the retrained fork (no released model), AoD +0.8, Delay +1.0, MPC +1.1 dB. RF-PGS (20.61 dB) is on other data and '
              'is not in this table.</figcaption></figure>')


def render(found):
    """The whole stage-2 section from whatever collect() found. "" if nothing."""
    runs = found.get("summary") or []
    if not runs and not found.get("tut019"):
        return ""
    # Runs bucketed by the GROUPS prefixes; a run matching no prefix simply
    # appears in no card.
    by = {g: [r for r in runs if r["run"].startswith(pre)] for g, (_, pre) in GROUPS.items()}
    meters, cards = [], []

    enc = found.get("encoding")
    if enc:
        # the physical metric first: what the field decodes to, as the median
        # (the RMSE is carried by a few percent of multi-path pixels)
        m = enc["multi"]
        meters.append(_meter(f"{m['az']['median']:.1f}&deg; / {m['zen']['median']:.1f}&deg; / {m['delay']['median']:.1f} ns",
                             f"decoded AoD azimuth / zenith / delay, median per pixel (P90 {m['az']['p90']:.0f}&deg; / {m['zen']['p90']:.0f}&deg; / {m['delay']['p90']:.0f} ns)"))
    c0 = next((r for r in by["baseline"]), None)
    if c0:
        meters.append(_meter(f"{c0['psnr_rgb']:.2f}", "gsplat RRF on the released MVDR data, PSNR after jet mapping (published 16.02)", "dB"))
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
        if found.get("strip"):
            cards.append(
                f'<figure class="card wide"><h2>Held-out views: truth, rgb, db, power</h2>'
                f'<p class="rrf-note">Two held-out receiver poses, jet-mapped. The RGB model composites '
                f'colours, so it can land off the jet curve (the dark and yellow smears); the dB and '
                f'power models cannot, because they composite the value itself.</p>'
                f'<img src="{found["strip"]}" alt="truth against the three colour modes" '
                f'style="width:100%;max-width:1200px;height:auto;display:block;border:1px solid var(--line);border-radius:4px">'
                f'<figcaption>output/rrf/compare_colour_modes.png</figcaption></figure>')
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
            '<p class="rrf-note">SH degree, frozen opacity, fewer training views, fewer iterations, '
            'geometry unfrozen, and gsplat\'s MCMC densification (which, unlike the INRIA densifier '
            'gated by densify_until_iter, runs inside the 30k&rarr;40k fine-tune).</p>'
            + _table(["run", "PSNR (jet)", "SSIM", "RMSE dB", "train s"], body)
            + '</figure>')
    if by["multi"]:
        rows = by["multi"]
        keys = sorted({k for r in rows for k in r.get("extra", {})})
        body = [[_label(r), f"{r['psnr_rgb']:.2f}", f"{r['rmse_db']:.2f}"]
                + [f"{r['extra'][k]:.2f}" if k in r.get("extra", {}) else "&ndash;" for k in keys] for r in rows]
        enc = found.get("encoding")
        enc_html = ""
        if enc:
            def cell(q):
                return f"{q['median']:.2f} / {q['p90']:.1f} / {q['rmse']:.1f}"
            mm, aa = enc["multi"], enc["aod3"]
            enc_html = _table(["decoded from", "AoD azimuth (deg): median / P90 / RMSE", "AoD zenith (deg)", "delay (ns)", "pixels > 45&deg; (az)"],
                              [["one channel per quantity (MULTI)", cell(mm["az"]), cell(mm["zen"]), cell(mm["delay"]), f"{mm['az']['frac_gt_45']*100:.1f}%"],
                               ["angle &times; amplitude RGB (the tutorial's AoD encoding)", cell(aa["az"]), cell(aa["zen"]), "&ndash;", f"{aa['az']['frac_gt_45']*100:.1f}%"]])
        cards.append(
            '<figure class="card wide"><h2>Multi-channel targets</h2>'
            '<p class="rrf-note">gsplat rasterises any number of channels, so path power, departure '
            'azimuth, departure zenith and delay each get their own channel instead of the tutorial\'s '
            'angle &times; amplitude RGB pictures. Per-channel errors on pixels a path reaches; the second '
            'table decodes both representations to angles on the same held-out views.</p>'
            + _table(["run", "PSNR (jet)", "RMSE dB (mask ch.)"] + [k.replace("rmse_", "RMSE ") for k in keys], body)
            + enc_html + '</figure>')
    if by["txmove"]:
        cold = [r for r in by["txmove"] if not r["warm"]]
        warm = [r for r in by["txmove"] if r["warm"]]
        if cold and warm:
            c, w = cold[0], warm[0]
            k = next(k for k in c if k.startswith("it_to_psnr"))
            if c.get(k) and w.get(k):
                meters.append(_meter(f"{c[k]:,} &rarr; {w[k]:,}", f"iterations to PSNR {k[10:]} after the Tx moved, cold &rarr; warm ({c['mode']})"))
        final = "; ".join(f"{_label(r)}: {r['psnr_rgb']:.2f} dB, RMSE {r['rmse_db']:.2f} dB"
                          + (f" ({r['rmse_db_in_range']:.2f} in range)" if r.get("rmse_db_in_range") else "")
                          for r in cold[:2] + warm[:2])
        cards.append(
            f'<figure class="card"><h2>Transmitter moved: cold against warm start</h2>'
            f'<p class="rrf-note">A new dataset for a second transmitter position, mapped with the first '
            f'transmitter\'s dB range; the RRF is fitted from zeroed colours (as RF-3DGS must) and from the '
            f'first transmitter\'s colours. Running eval on 64 held-out views against wall time.</p>'
            f'{curves(cold[:2] + warm[:2])}<figcaption>PSNR after jet mapping &middot; final on 640 views: '
            f'{final}</figcaption></figure>')

    for key in ("transfer_2k", "transfer", "transfer_s2"):
        tr = found.get(key)
        if tr and tr.get("rows"):
            cards.append(transfer_card(tr))
    if found.get("sota"):
        cards.insert(0, sota_card(found["sota"]))
    if found.get("density"):
        cards.append(density_card(found["density"]))

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
  cost of a moved transmitter has a number. The physical metrics come first: dB error against the
  float spectrum and, for the multi-channel model, the angle and delay the field decodes to. PSNR
  is kept as a compatibility number &mdash; every model's output is mapped through jet and scored
  against the PNG the way the paper does &mdash; not as anything that was optimised.
  Source: <code>rrf_gsplat/</code>.</p></div>
  <div class="meters">{''.join(meters)}</div>
  <div class="rrf-grid" style="margin-top:14px">{''.join(cards)}</div>
</section>
"""
