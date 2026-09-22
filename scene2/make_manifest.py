"""Manifest of a rendered visual dataset: what produced it and a checksum per file.

The PNGs are derived data (scene XML + render_visual.py + fixed seeds) and are
not tracked; this manifest is, so a re-render can be checked against it.

    PYTHONUTF8=1 python scene2/make_manifest.py --dataset scene2/visual_dataset --out scene2/visual_dataset_manifest.json \
        --command "python scene2/render_visual.py --scene scene2/corridor --out scene2/visual_dataset --width 800 --height 450 --spp 64 --extra 300 --route-every 4"
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess


def main():
    """Write a manifest, or with --check verify a dataset against an existing one.

    Exit code in --check mode is the result: 0 when every tracked file is present
    and matches, 1 otherwise, so this can gate a CI step or a re-render.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--command", default=None, help="the exact render command (writing a manifest)")
    ap.add_argument("--check", action="store_true", help="verify the dataset against the manifest at --out instead of writing it")
    cfg = ap.parse_args()
    files = {}
    # Hash every file under --dataset. sorted(names) keeps the walk order stable,
    # though the result is a dict keyed by path so order does not affect the output.
    for root, _, names in os.walk(cfg.dataset):
        for n in sorted(names):
            p = os.path.join(root, n)
            h = hashlib.sha256()
            with open(p, "rb") as fid:
                # Streamed in 1 MiB chunks so a large PNG set never loads at once.
                for chunk in iter(lambda: fid.read(1 << 20), b""):
                    h.update(chunk)
            # Keys are dataset-relative with forward slashes, so a manifest written
            # on Windows verifies on Linux and vice versa.
            files[os.path.relpath(p, cfg.dataset).replace(os.sep, "/")] = {"sha256": h.hexdigest(), "bytes": os.path.getsize(p)}
    if cfg.check:
        ref = json.load(open(cfg.out))["entries"]
        # Three distinct failure kinds, reported separately: content changed,
        # file gone, file appeared. Only the first two are treated as failures --
        # extra files do not invalidate the dataset.
        missing = sorted(set(ref) - set(files)); extra = sorted(set(files) - set(ref))
        bad = sorted(k for k in set(ref) & set(files) if ref[k]["sha256"] != files[k]["sha256"])
        print(f"{len(ref)} entries in the manifest, {len(files)} files on disk: {len(bad)} mismatched, {len(missing)} missing, {len(extra)} not in the manifest")
        # Only the first 10 offenders are listed; the counts above are the summary.
        for k in (bad + missing)[:10]:
            print("  ", k, "MISMATCH" if k in bad else "MISSING")
        raise SystemExit(1 if (bad or missing) else 0)
    if not cfg.command:
        raise SystemExit("--command is required when writing a manifest")
    # The renderer's commit is recorded alongside the command, so a mismatch can be
    # traced to a code change rather than only to a data change. Best-effort:
    # outside a git checkout this stays None rather than failing the run.
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        commit = None
    man = {"dataset": cfg.dataset, "command": cfg.command, "renderer_commit": commit, "files": len(files),
           "bytes": sum(f["bytes"] for f in files.values()), "entries": files}
    json.dump(man, open(cfg.out, "w"), indent=1)
    print(f"{len(files)} files, {man['bytes'] / 1048576:.0f} MB, commit {commit} -> {cfg.out}")


if __name__ == "__main__":
    main()
