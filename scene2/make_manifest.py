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
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--command", default=None, help="the exact render command (writing a manifest)")
    ap.add_argument("--check", action="store_true", help="verify the dataset against the manifest at --out instead of writing it")
    cfg = ap.parse_args()
    files = {}
    for root, _, names in os.walk(cfg.dataset):
        for n in sorted(names):
            p = os.path.join(root, n)
            h = hashlib.sha256()
            with open(p, "rb") as fid:
                for chunk in iter(lambda: fid.read(1 << 20), b""):
                    h.update(chunk)
            files[os.path.relpath(p, cfg.dataset).replace(os.sep, "/")] = {"sha256": h.hexdigest(), "bytes": os.path.getsize(p)}
    if cfg.check:
        ref = json.load(open(cfg.out))["entries"]
        missing = sorted(set(ref) - set(files)); extra = sorted(set(files) - set(ref))
        bad = sorted(k for k in set(ref) & set(files) if ref[k]["sha256"] != files[k]["sha256"])
        print(f"{len(ref)} entries in the manifest, {len(files)} files on disk: {len(bad)} mismatched, {len(missing)} missing, {len(extra)} not in the manifest")
        for k in (bad + missing)[:10]:
            print("  ", k, "MISMATCH" if k in bad else "MISSING")
        raise SystemExit(1 if (bad or missing) else 0)
    if not cfg.command:
        raise SystemExit("--command is required when writing a manifest")
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
