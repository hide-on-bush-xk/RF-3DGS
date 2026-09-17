"""Rename the scene's mesh files to ASCII and update the XML that references them.

The NIST lobby was authored in a Chinese-locale Blender, so every mesh carries a
name like `立方体_005.ply`. Mitsuba's FileStream converts paths to the active
Windows code page before opening them, and cp1252 cannot represent those
characters, so a scene that loads on Linux fails on Windows with

    [FileStream] "...\\meshes\\立方体_005.ply": No such file or directory

even though the file is right there. (The same names already caused trouble in
the archive, which stores them as UTF-8 without setting the UTF-8 flag.)

Renames are recorded in a JSON sidecar so the originals can be restored.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import unicodedata

# Blender's default primitive names in a Chinese UI, plus the scene's own nouns.
TRANSLIT = {
    "立方体": "cube",
    "球体": "sphere",
    "柱体": "cylinder",
    "平面": "plane",
    "房屋轮廓": "house_outline",
    "灯": "lamp",
    "背景墙": "backdrop",
    "踢脚线": "baseboard",
    "圆环": "torus",
    "锥体": "cone",
    "网格": "mesh",
}


def to_ascii(name: str, fallback_index: int) -> str:
    """ASCII filename for a mesh, keeping any numeric suffix and material tag."""
    stem, ext = os.path.splitext(name)
    out = stem
    for zh, en in TRANSLIT.items():
        out = out.replace(zh, en)
    out = unicodedata.normalize("NFKD", out).encode("ascii", "ignore").decode()
    out = re.sub(r"[^A-Za-z0-9._-]+", "_", out).strip("_")
    if not out or out.lstrip("._-") == "":
        out = f"mesh_{fallback_index:04d}"
    return out + ext


def rename_meshes(mesh_dir: str, dry_run: bool = False) -> dict[str, str]:
    mapping: dict[str, str] = {}
    taken: set[str] = set()
    for i, name in enumerate(sorted(os.listdir(mesh_dir))):
        if not name.lower().endswith(".ply"):
            continue
        if name.isascii():
            taken.add(name)
            continue
        new = to_ascii(name, i)
        base, ext = os.path.splitext(new)
        n = 1
        while new in taken:                      # collisions after transliteration
            new = f"{base}__{n}{ext}"
            n += 1
        taken.add(new)
        mapping[name] = new
        if not dry_run:
            os.rename(os.path.join(mesh_dir, name), os.path.join(mesh_dir, new))
    return mapping


def rewrite_xml(xml_path: str, out_path: str, mapping: dict[str, str]) -> int:
    with open(xml_path, encoding="utf-8") as fid:
        text = fid.read()
    n = 0
    for old, new in mapping.items():
        if old in text:
            text = text.replace(old, new)
            n += 1
    with open(out_path, "w", encoding="utf-8", newline="\n") as fid:
        fid.write(text)
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("xml", help="scene XML to rewrite")
    ap.add_argument("--mesh-dir", default=None,
                    help="defaults to <xml dir>/meshes")
    ap.add_argument("--out", default=None, help="defaults to rewriting in place")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    mesh_dir = args.mesh_dir or os.path.join(os.path.dirname(args.xml), "meshes")
    out = args.out or args.xml

    mapping = rename_meshes(mesh_dir, args.dry_run)
    print(f"{len(mapping)} meshes renamed to ASCII")
    for old, new in list(mapping.items())[:5]:
        print(f"  {old}  ->  {new}")
    if len(mapping) > 5:
        print(f"  ... and {len(mapping) - 5} more")

    if not args.dry_run:
        n = rewrite_xml(args.xml, out, mapping)
        print(f"{n} references updated in {out}")
        sidecar = os.path.join(mesh_dir, "ascii_rename_map.json")
        with open(sidecar, "w", encoding="utf-8") as fid:
            json.dump(mapping, fid, ensure_ascii=False, indent=1)
        print(f"mapping saved to {sidecar}")


if __name__ == "__main__":
    main()
