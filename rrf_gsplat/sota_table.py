"""The SOTA table on the RF-3DGS released benchmark: every row scored by the fork's metrics.py.

Rows: A0 released checkpoints (their results.json), A1 the fork's train.py retrained here (output/sota/inria_<S>),
A2-A5 our trainer (output/rrf/sota_<S>_{rgb,db,rgb_geom,db_geom}/inria_metrics.json), B NeRF2 (output/rrf/nerf2_<S>).
Columns: the six spectra. A cell that has not been produced prints MISSING; an invalid one (jet-inverted target on a
three-channel AoD / Delay picture) prints N/A. Writes output/rrf/sota_table.md and sota_table.json.

MISSING and N/A are deliberately different words: the first is work not done,
the second is a combination that cannot be scored at all. Neither cell is ever
silently dropped, because a table with rows removed reads as a complete
comparison when it is not.

    PYTHONUTF8=1 python rrf_gsplat/sota_table.py
"""

from __future__ import annotations

import json
import os

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SPECTRA = ["MVDR", "CBF", "TCBF", "AoD", "Delay", "MPC"]
# Combinations that are not merely unmeasured but meaningless: a scalar target
# cannot represent a three-channel encoding, so these print N/A forever.
INVALID = {("db", "AoD"), ("db", "Delay"), ("db_geom", "AoD"), ("db_geom", "Delay"),      # jet inverse of a 3-channel encoding
           ("nerf2", "AoD"), ("nerf2", "Delay")}                                              # NeRF2's scalar head
# Row order is the argument: A0 is the published baseline, A1 the same code
# retrained here (which separates a code difference from a method difference),
# A2-A5 our trainer's four variants, B the competing method.
ROWS = [("A0 RF-3DGS as published (released checkpoint, 40k)", "released"),
        ("A1 RF-3DGS retrained here (fork's train.py, 30k->40k)", "inria"),
        ("A2 ours, jet RGB target, frozen geometry, 10k", "rgb"),
        ("A3 ours, jet-inverted value target, frozen, 10k", "db"),
        ("A4 ours, jet RGB, unfrozen geometry, 10k", "rgb_geom"),
        ("A5 ours, value target, unfrozen geometry, 10k", "db_geom"),
        ("A2' ours, jet RGB, frozen, 40k (MVDR)", "rgb_40k"),
        ("A5' ours, value target, unfrozen, 40k (MVDR)", "db_geom_40k"),
        ("B NeRF2 (their model, pinhole rays), 30k", "nerf2")]


def cell(kind, S):
    """(metrics dict, None) for a scored cell, or (None, reason) for an empty one.

    Every empty cell carries its reason, so the table distinguishes "not run"
    from "cannot be run" from "run but incomplete".
    """
    if (kind, S) in INVALID:
        return None, "N/A"
    if kind == "released":
        p = os.path.join(REPO, "RF-3DGS_dataset", "RF-3DGS_trained_RRF", f"3dgs_{S}_100", "results.json")
        if not os.path.exists(p):
            return None, "no released model"
        d = json.load(open(p))["ours_40000"]; return d, None
    if kind == "inria":
        p = os.path.join(REPO, "output", "sota", f"inria_{S}", "results.json")
        if not os.path.exists(p):
            return None, "MISSING"
        # metrics.py keys by "ours_<iteration>"; take the highest iteration present.
        d = json.load(open(p)); d = d[sorted(d)[-1]]; return d, None
    name = f"nerf2_{S}" if kind == "nerf2" else f"sota_{S}_{kind}"
    p = os.path.join(REPO, "output", "rrf", name, "inria_metrics.json")
    if not os.path.exists(p):
        return None, "MISSING"
    d = json.load(open(p))
    # A partial render set is reported as such rather than scored: a number over
    # a subset is not comparable with the other rows.
    if d.get("partial"):
        return None, f"partial ({d['views']} views)"
    return d, None


def main():
    """Build the markdown table and the machine-readable copy beside it."""
    table = {}
    lines = ["| row | " + " | ".join(SPECTRA) + " |", "| --- | " + " | ".join("---" for _ in SPECTRA) + " |"]
    for label, kind in ROWS:
        cells = []
        for S in SPECTRA:
            # The 40k rows were only run for MVDR; an em dash marks "not part of
            # this row", distinct from both MISSING and N/A.
            if kind.endswith("_40k") and S != "MVDR":
                cells.append("—"); continue
            d, note = cell(kind, S)
            if d is None:
                cells.append(note)
            else:
                table.setdefault(kind, {})[S] = {k: d[k] for k in ("PSNR", "SSIM", "LPIPS")}
                cells.append(f"{d['PSNR']:.2f} / {d['SSIM']:.3f} / {d['LPIPS']:.3f}")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    # deltas against the released checkpoint where both exist
    # Only computed where both cells are real, so a delta never implies a
    # comparison that was not actually made.
    deltas = []
    for kind in ("rgb", "db", "rgb_geom", "db_geom", "nerf2", "inria"):
        for S in SPECTRA:
            a, b = table.get("released", {}).get(S), table.get(kind, {}).get(S)
            if a and b:
                deltas.append(f"{kind} on {S}: PSNR {b['PSNR'] - a['PSNR']:+.2f} dB, SSIM {b['SSIM'] - a['SSIM']:+.3f}, LPIPS {b['LPIPS'] - a['LPIPS']:+.3f}")
    # The heading states the data, the split and the metric implementation --
    # the three things that have to match for the rows to be comparable.
    md = ("# RF-3DGS released benchmark: released data, released 2560 / 640 split, the fork's metrics.py (PSNR / SSIM / VGG-LPIPS, 640 held-out views)\n\n"
          + "\n".join(lines) + "\n\nAgainst the released checkpoint (A0):\n\n" + "\n".join(f"- {d}" for d in deltas) + "\n")
    open(os.path.join(REPO, "output", "rrf", "sota_table.md"), "w", encoding="utf-8").write(md)
    json.dump(table, open(os.path.join(REPO, "output", "rrf", "sota_table.json"), "w"), indent=1)
    print(md)


if __name__ == "__main__":
    main()
