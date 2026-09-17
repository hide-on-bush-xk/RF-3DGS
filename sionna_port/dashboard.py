"""Render an ablation run as a self-contained HTML dashboard.

Charts are inline SVG against CSS custom properties, so the file opens from disk
with no network and no CDN, and both themes come from the same tokens.
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import os

# Categorical slots from the data-viz reference palette, assigned in fixed order.
SERIES_LIGHT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
SERIES_DARK = ["#3987e5", "#d95926", "#199e70", "#c98500"]

CSS = """
:root {
  --bg:#f4f5f3; --surface:#ffffff; --sunk:#eceee9; --ink:#16181a;
  --muted:#5b6166; --line:#dfe3de; --accent:#0b6e6b; --ok:#1c6b4a;
  --s1:#2a78d6; --s2:#eb6834; --s3:#1baf7a; --s4:#eda100;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg:#0e1215; --surface:#171c20; --sunk:#12171a; --ink:#e7ecea;
    --muted:#93a0a6; --line:#242e33; --accent:#58c5be; --ok:#6fd3a0;
    --s1:#3987e5; --s2:#d95926; --s3:#199e70; --s4:#c98500;
  }
}
:root[data-theme="dark"] {
  --bg:#0e1215; --surface:#171c20; --sunk:#12171a; --ink:#e7ecea;
  --muted:#93a0a6; --line:#242e33; --accent:#58c5be; --ok:#6fd3a0;
  --s1:#3987e5; --s2:#d95926; --s3:#199e70; --s4:#c98500;
}
* { box-sizing: border-box; }
body { margin:0; background:var(--bg); color:var(--ink); line-height:1.5;
  font-family:"IBM Plex Sans",ui-sans-serif,system-ui,sans-serif; }
.wrap { max-width:1140px; margin:0 auto; padding-inline:20px; padding-block:36px 56px; }
h1 { font-family:"IBM Plex Sans Condensed","IBM Plex Sans",sans-serif;
  font-size:clamp(26px,4.4vw,38px); line-height:1.1; margin:0; letter-spacing:-.01em; }
h2 { font-family:"IBM Plex Sans Condensed","IBM Plex Sans",sans-serif;
  font-size:20px; margin:0 0 4px; }
p { margin:0; }
.eyebrow { font-family:"IBM Plex Mono",ui-monospace,monospace; font-size:11px;
  letter-spacing:.13em; text-transform:uppercase; color:var(--accent); margin-bottom:12px; }
.lede { margin-top:12px; max-width:66ch; color:var(--muted); font-size:15.5px; }
.rig { display:flex; flex-wrap:wrap; gap:6px; margin-top:18px;
  font-family:"IBM Plex Mono",monospace; font-size:11.5px; }
.rig span { border:1px solid var(--line); border-radius:3px; padding:3px 8px;
  color:var(--muted); background:var(--surface); }
section { margin-top:34px; }
.sec-head { margin-bottom:14px; }
.sec-head p { color:var(--muted); font-size:13.5px; margin-top:3px; max-width:70ch; }
.meters { display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr));
  gap:1px; background:var(--line); border:1px solid var(--line); border-radius:4px;
  overflow:hidden; }
.meter { background:var(--surface); padding:14px 16px; }
.meter b { display:block; font-family:"IBM Plex Sans Condensed",sans-serif;
  font-size:25px; font-weight:700; font-variant-numeric:tabular-nums; letter-spacing:-.02em; }
.meter b i { font-style:normal; font-size:14px; color:var(--muted); font-weight:600; }
.meter span { font-family:"IBM Plex Mono",monospace; font-size:10.5px;
  text-transform:uppercase; letter-spacing:.08em; color:var(--muted); }
.card { background:var(--surface); border:1px solid var(--line); border-radius:4px;
  padding:16px 18px; }
.grid2 { display:grid; grid-template-columns:repeat(auto-fit,minmax(330px,1fr)); gap:14px; }
figure { margin:0; }
figcaption { font-family:"IBM Plex Mono",monospace; font-size:10.5px; color:var(--muted);
  margin-top:6px; }
.legend { display:flex; flex-wrap:wrap; gap:12px; font-family:"IBM Plex Mono",monospace;
  font-size:11px; color:var(--muted); margin-bottom:8px; }
.legend i { width:9px; height:9px; border-radius:2px; display:inline-block;
  margin-right:5px; vertical-align:-1px; }
.tablewrap { overflow-x:auto; border:1px solid var(--line); border-radius:4px;
  background:var(--surface); }
table { border-collapse:collapse; width:100%; min-width:720px; }
th,td { text-align:right; padding:9px 13px; border-bottom:1px solid var(--line);
  font-variant-numeric:tabular-nums; }
thead th { font-family:"IBM Plex Mono",monospace; font-size:10px; font-weight:500;
  letter-spacing:.07em; text-transform:uppercase; color:var(--muted);
  background:var(--sunk); white-space:nowrap; }
tbody th { text-align:left; font-family:"IBM Plex Mono",monospace; font-size:12px;
  font-weight:500; white-space:nowrap; }
tbody td { font-family:"IBM Plex Mono",monospace; font-size:12.5px; }
tbody tr:last-child th, tbody tr:last-child td { border-bottom:0; }
svg text { font-family:"IBM Plex Mono",monospace; }
footer { margin-top:38px; padding-top:14px; border-top:1px solid var(--line);
  font-family:"IBM Plex Mono",monospace; font-size:11px; color:var(--muted);
  display:flex; flex-wrap:wrap; gap:6px 18px; }
"""


def _fmt(v, digits=1):
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "n/a"
    return f"{v:,.{digits}f}"


def grouped_bars(runs, key, title, unit, log=False, width=520, height=250):
    """Grouped bars: one group per scattering value, one bar per depth."""
    scatters = sorted({r["scattering_coefficient"] for r in runs})
    depths = sorted({r["max_depth"] for r in runs})
    vals = {(r["scattering_coefficient"], r["max_depth"]): r["metrics_mean"].get(key)
            for r in runs}

    pad_l, pad_r, pad_t, pad_b = 62, 12, 16, 38
    pw, ph = width - pad_l - pad_r, height - pad_t - pad_b
    finite = [v for v in vals.values() if v is not None and math.isfinite(v)]
    if not finite:
        return ""
    vmax = max(finite)
    vmin = min(0.0, min(finite))
    if log:
        vmax = math.log10(max(vmax, 1.0)) or 1.0
        vmin = 0.0

    def y_of(v):
        if v is None or not math.isfinite(v):
            return None
        t = math.log10(max(v, 1.0)) if log else v
        span = (vmax - vmin) or 1.0
        return pad_t + ph - (t - vmin) / span * ph

    gw = pw / max(len(scatters), 1)
    bw = min(26.0, (gw - 12) / max(len(depths), 1))

    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" '
             f'aria-label="{title}" style="width:100%;height:auto">']
    # gridlines and axis labels
    for i in range(5):
        frac = i / 4
        y = pad_t + ph - frac * ph
        t = vmin + frac * ((vmax - vmin) or 1.0)
        label = f"{10 ** t:,.0f}" if log else f"{t:,.0f}"
        parts.append(f'<line x1="{pad_l}" x2="{width-pad_r}" y1="{y:.1f}" y2="{y:.1f}" '
                     f'stroke="var(--line)" stroke-width="1"/>')
        parts.append(f'<text x="{pad_l-8}" y="{y+3.5:.1f}" text-anchor="end" '
                     f'font-size="10" fill="var(--muted)">{label}</text>')

    for gi, s in enumerate(scatters):
        gx = pad_l + gi * gw
        for di, d in enumerate(depths):
            v = vals.get((s, d))
            y = y_of(v)
            if y is None:
                continue
            x = gx + (gw - bw * len(depths)) / 2 + di * bw
            h = pad_t + ph - y
            parts.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw-2:.1f}" height="{max(h,0.5):.1f}" '
                f'rx="3" fill="var(--s{di+1})"/>')
        parts.append(f'<text x="{gx+gw/2:.1f}" y="{height-14}" text-anchor="middle" '
                     f'font-size="10.5" fill="var(--muted)">s={s:g}</text>')

    parts.append(f'<text x="{pad_l}" y="{height-2}" font-size="10" '
                 f'fill="var(--muted)">scattering coefficient</text>')
    parts.append(f'<text x="{pad_l-52}" y="{pad_t+8}" font-size="10" '
                 f'fill="var(--muted)">{unit}</text>')
    parts.append("</svg>")
    return "".join(parts)


def _flatten(seq):
    """Tolerate a nested single-row array from an older metrics run."""
    while seq and isinstance(seq[0], list):
        seq = seq[0]
    return seq


def line_chart(x, y, title, x_unit, y_unit, width=520, height=230):
    x, y = _flatten(list(x)), _flatten(list(y))
    pad_l, pad_r, pad_t, pad_b = 62, 14, 16, 36
    pw, ph = width - pad_l - pad_r, height - pad_t - pad_b
    xmin, xmax = min(x), max(x)
    ymin, ymax = min(y), max(y)
    if ymax - ymin < 1e-9:
        ymax = ymin + 1.0

    def px(v):
        return pad_l + (v - xmin) / ((xmax - xmin) or 1.0) * pw

    def py(v):
        return pad_t + ph - (v - ymin) / (ymax - ymin) * ph

    pts = " ".join(f"{px(a):.1f},{py(b):.1f}" for a, b in zip(x, y))
    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" '
             f'aria-label="{title}" style="width:100%;height:auto">']
    for i in range(5):
        frac = i / 4
        yy = pad_t + ph - frac * ph
        val = ymin + frac * (ymax - ymin)
        parts.append(f'<line x1="{pad_l}" x2="{width-pad_r}" y1="{yy:.1f}" y2="{yy:.1f}" '
                     f'stroke="var(--line)" stroke-width="1"/>')
        parts.append(f'<text x="{pad_l-8}" y="{yy+3.5:.1f}" text-anchor="end" '
                     f'font-size="10" fill="var(--muted)">{val:,.0f}</text>')
    parts.append(f'<polyline points="{pts}" fill="none" stroke="var(--s1)" '
                 f'stroke-width="2" stroke-linejoin="round"/>')
    for v, anchor in ((xmin, "start"), (xmax, "end")):
        parts.append(f'<text x="{px(v):.1f}" y="{height-14}" text-anchor="{anchor}" '
                     f'font-size="10.5" fill="var(--muted)">{v/1e9:,.3f} GHz</text>')
    parts.append(f'<text x="{pad_l-52}" y="{pad_t+8}" font-size="10" '
                 f'fill="var(--muted)">{y_unit}</text>')
    parts.append("</svg>")
    return "".join(parts)


def per_view_bars(values, title, unit, width=520, height=230):
    """One bar per view, for a single generation run."""
    pad_l, pad_r, pad_t, pad_b = 62, 12, 16, 34
    pw, ph = width - pad_l - pad_r, height - pad_t - pad_b
    vals = [v for v in values if v is not None and math.isfinite(v)]
    if not vals:
        return ""
    vmax, vmin = max(vals), min(0.0, min(vals))
    span = (vmax - vmin) or 1.0
    bw = max(pw / max(len(vals), 1) - 2, 1.0)

    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" '
             f'aria-label="{title}" style="width:100%;height:auto">']
    for i in range(5):
        frac = i / 4
        y = pad_t + ph - frac * ph
        parts.append(f'<line x1="{pad_l}" x2="{width-pad_r}" y1="{y:.1f}" y2="{y:.1f}" '
                     f'stroke="var(--line)" stroke-width="1"/>')
        parts.append(f'<text x="{pad_l-8}" y="{y+3.5:.1f}" text-anchor="end" '
                     f'font-size="10" fill="var(--muted)">'
                     f'{vmin + frac * span:,.1f}</text>')
    for i, v in enumerate(vals):
        y = pad_t + ph - (v - vmin) / span * ph
        x = pad_l + i * (pw / len(vals))
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw:.1f}" '
                     f'height="{max(pad_t + ph - y, 0.5):.1f}" rx="2" fill="var(--s1)"/>')
    parts.append(f'<text x="{pad_l}" y="{height-12}" font-size="10" '
                 f'fill="var(--muted)">view 1</text>')
    parts.append(f'<text x="{width-pad_r}" y="{height-12}" text-anchor="end" '
                 f'font-size="10" fill="var(--muted)">view {len(vals)}</text>')
    parts.append(f'<text x="{pad_l-52}" y="{pad_t+8}" font-size="10" '
                 f'fill="var(--muted)">{unit}</text>')
    parts.append("</svg>")
    return "".join(parts)


def embed(path):
    with open(path, "rb") as fid:
        return "data:image/png;base64," + base64.b64encode(fid.read()).decode()


METRIC_ROWS = [
    ("num_paths", "paths", 0),
    ("rms_delay_spread_ns", "RMS delay (ns)", 2),
    ("coherence_bandwidth_mhz", "coherence BW (MHz)", 1),
    ("k_factor_db", "K-factor (dB)", 2),
    ("aoa_azimuth_spread_deg", "AoA az spread (deg)", 1),
    ("aod_azimuth_spread_deg", "AoD az spread (deg)", 1),
    ("array_gain_db", "coherent gain (dB)", 2),
    ("total_power_dbm", "path gain (dB)", 1),
    ("snr_db", "SNR (dB)", 1),
    ("capacity_bps_hz", "capacity (bps/Hz)", 2),
]


def build(data, preview_dir=None):
    runs = data["runs"]
    depths = sorted({r["max_depth"] for r in runs})
    series_names = [f"depth {d}" for d in depths]

    legend = "".join(
        f'<span><i style="background:var(--s{i+1})"></i>{n}</span>'
        for i, n in enumerate(series_names))

    charts = [
        ("Multipath components", "num_paths", "paths (log)", True,
         "Scattering is the only knob that matters. samples_per_src changes this "
         "by nothing at all."),
        ("K-factor", "k_factor_db", "dB", False,
         "Ratio of the strongest path to everything else. It falls towards 0 dB as "
         "scattering fills the channel in, which is the transition from a "
         "LOS-dominated link to a rich one."),
        ("RMS delay spread", "rms_delay_spread_ns", "ns", False,
         "Power-weighted spread of path delays, and the quantity that sets "
         "coherence bandwidth."),
        ("Coherent gain", "array_gain_db", "dB", False,
         "Gain of a fixed broadside weight over incoherent summation. It falls "
         "towards 0 dB as the field becomes diffuse and the element phases "
         "decorrelate."),
    ]

    single = len(runs) == 1
    if single:
        pv = runs[0].get("per_position", [])
        singles = [
            ("Multipath components per view", "num_paths", "paths",
             "How many resolvable paths each receiver pose sees."),
            ("K-factor per view", "k_factor_db", "dB",
             "Below 0 dB the strongest path carries less power than everything "
             "else combined -- a rich channel rather than a LOS one."),
            ("RMS delay spread per view", "rms_delay_spread_ns", "ns",
             "Sets the coherence bandwidth, and therefore how frequency "
             "selective the link is."),
            ("Coherent gain per view", "array_gain_db", "dB",
             "How beamformable each pose is: 0 dB means the field is diffuse "
             "enough that a fixed broadside weight buys nothing."),
        ]
        chart_html = [
            f'<figure class="card"><h2>{t}</h2>'
            f'<p style="color:var(--muted);font-size:13px;margin:2px 0 10px">{note}</p>'
            f'{per_view_bars([p.get(k) for p in pv], t, unit)}</figure>'
            for t, k, unit, note in singles]
    else:
        chart_html = []

    for title, key, unit, log, note in ([] if single else charts):
        chart_html.append(
            f'<figure class="card"><h2>{title}</h2>'
            f'<p style="color:var(--muted);font-size:13px;margin:2px 0 10px">{note}</p>'
            f'<div class="legend">{legend}</div>'
            f'{grouped_bars(runs, key, title, unit, log)}</figure>')

    # Table
    head = "".join(f"<th>{r['label']}</th>" for r in runs)
    body = []
    for key, label, digits in METRIC_ROWS:
        cells = "".join(f"<td>{_fmt(r['metrics_mean'].get(key), digits)}</td>"
                        for r in runs)
        body.append(f"<tr><th>{label}</th>{cells}</tr>")
    spec_rows = []
    if any("spectrum" in r for r in runs):
        for key, label, digits in [("global_span_db", "spectrum span (dB)", 1),
                                   ("quantisation_step_db", "8-bit step (dB)", 3),
                                   ("quantisation_rmse_db", "quantisation RMSE (dB)", 3),
                                   ("fraction_of_range_used_mean", "range used per view", 3)]:
            cells = "".join(f"<td>{_fmt(r.get('spectrum', {}).get(key), digits)}</td>"
                            for r in runs)
            spec_rows.append(f"<tr><th>{label}</th>{cells}</tr>")

    cfr_html = ""
    cfr_run = next((r for r in runs if "cfr" in r), None)
    if cfr_run:
        c = cfr_run["cfr"]
        cfr_html = (
            f'<figure class="card"><h2>Frequency response</h2>'
            f'<p style="color:var(--muted);font-size:13px;margin:2px 0 10px">'
            f'{len(c["frequency_hz"])} subcarriers across '
            f'{data["bandwidth_hz"]/1e6:,.0f} MHz, from <code>paths.cfr()</code> at '
            f'{cfr_run["label"]}. Frequency selectivity is invisible in an angular '
            f'power image, so the image-domain evaluation cannot see this axis at '
            f'all.</p>'
            f'{line_chart(c["frequency_hz"], c["magnitude_db"], "channel frequency response", "GHz", "dB")}'
            f'</figure>')

    previews = ""
    if preview_dir and os.path.isdir(preview_dir):
        files = sorted(f for f in os.listdir(preview_dir) if f.endswith(".png"))[:4]
        if files:
            cells = "".join(
                f'<figure><img src="{embed(os.path.join(preview_dir, f))}" '
                f'alt="spectrum {f}" style="width:100%;border-radius:3px"/>'
                f'<figcaption>{f}</figcaption></figure>' for f in files)
            previews = (f'<section><div class="sec-head"><h2>Rendered spectra</h2>'
                        f'<p>Sample views from the most recent generation run.</p></div>'
                        f'<div class="grid2">{cells}</div></section>')

    best = max(runs, key=lambda r: r["metrics_mean"]["num_paths"])
    fastest = min(runs, key=lambda r: r["mean_solve_seconds"])

    return f"""<title>RF-3DGS Channel Ablation</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans+Condensed:wght@600;700&family=IBM+Plex+Sans:wght@400;500&display=swap">
<style>{CSS}</style>
<div class="wrap">
<header>
  <p class="eyebrow">{data['generated_at']} &middot; NIST lobby &middot; {data['frequency_hz']/1e9:,.0f} GHz</p>
  <h1>What the channel says, not what the picture looks like</h1>
  <p class="lede">RF-3DGS is evaluated on PSNR and SSIM between colourmapped
  spectra. Those metrics cannot see delay spread, coherence bandwidth, K-factor
  or array gain, all of which are properties of the channel the model is supposed
  to represent. Every figure here comes from the solved paths, over
  {data['positions']} receiver positions per configuration.</p>
  <div class="rig">
    <span>{len(runs)} configurations</span><span>{data['M']}x{data['M']} UPA</span>
    <span>{data['spectrum']} spectrum</span>
    <span>{data['bandwidth_hz']/1e6:,.0f} MHz</span>
    <span>samples/src {data['samples_per_src']:,}</span>
  </div>
</header>

<section>
  <div class="meters">
    <div class="meter"><b>{best['metrics_mean']['num_paths']:,.0f}</b><span>most paths ({best['label']})</span></div>
    <div class="meter"><b>{best['metrics_mean']['k_factor_db']:.1f}<i> dB</i></b><span>K-factor there</span></div>
    <div class="meter"><b>{best['metrics_mean']['rms_delay_spread_ns']:.1f}<i> ns</i></b><span>RMS delay spread</span></div>
    <div class="meter"><b>{fastest['mean_solve_seconds']*1000:,.0f}<i> ms</i></b><span>fastest solve</span></div>
  </div>
</section>

<section>
  <div class="sec-head"><h2>Ablation</h2>
  <p>Scattering coefficient against maximum interaction depth. Sionna's ITU
  materials load with a scattering coefficient of zero, so the leftmost group is
  what the pipeline does by default.</p></div>
  <div class="grid2">{''.join(chart_html)}</div>
</section>

<section>
  <div class="sec-head"><h2>Channel and spectrum metrics</h2>
  <p>Means across receiver positions. The lower block measures what the 8-bit
  colourmap costs.</p></div>
  <div class="tablewrap"><table>
    <thead><tr><th>metric</th>{head}</tr></thead>
    <tbody>{''.join(body)}{''.join(spec_rows)}</tbody>
  </table></div>
</section>

<section><div class="grid2">{cfr_html}</div></section>
{previews}

<footer>
  <span>RF-3DGS &middot; Sun Lab, University of Georgia</span>
  <span>metrics from sionna_port/rf_metrics.py</span>
  <span>{os.path.basename(data.get('scene_xml',''))}</span>
</footer>
</div>
"""


def render(data, preview_dir=None):
    """Full standalone HTML document for a report or an ablation sweep."""
    head = (
        "<!doctype html>\n"
        '<html lang="en">\n'
        "<head>\n"
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
    )
    return head + build(data, preview_dir) + "\n</body>\n</html>\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("ablation_json")
    ap.add_argument("--out", default="dashboard.html")
    ap.add_argument("--preview-dir", default=None,
                    help="directory of generated spectrum PNGs to embed")
    args = ap.parse_args()

    with open(args.ablation_json, encoding="utf-8") as fid:
        data = json.load(fid)
    html = build(data, args.preview_dir)
    head = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1">\n')
    with open(args.out, "w", encoding="utf-8", newline="\n") as fid:
        fid.write(head + html + "\n</body>\n</html>\n")
    print(f"wrote {args.out} ({os.path.getsize(args.out)/1e6:.2f} MB)")


if __name__ == "__main__":
    main()
