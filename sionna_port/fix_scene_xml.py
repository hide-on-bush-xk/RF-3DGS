"""Make the NIST lobby scene loadable by Sionna 1.2+.

The scene exported from Blender names its materials `mat-itu_plasterboard.001`,
`mat-custom_plastic.007` and so on. Sionna derives an ITU material type by
stripping `mat-` and `itu_` from the id, so Blender's duplicate suffix turns
`plasterboard` into `plasterboard.001`, which is not a material it knows:

    ValueError: Invalid ITU material type "plasterboard.001"

Materials that are not named `itu_*` are left alone by Sionna and then rejected
at load time for not being radio materials at all. The 0.19 pipeline assigned
their electromagnetic properties through a separate semantic descriptor that was
not published with the tutorial.

This rewrites every material into an explicit `itu-radio-material` plugin with
the type spelled out. Ids are preserved, so `<ref id="...">` in the shapes keeps
resolving.
"""

from __future__ import annotations

import argparse
import re
import xml.etree.ElementTree as ET
from collections import Counter

# Valid frequency bands in GHz per material, from Sionna 1.2's
# sionna.rt.radio_materials.itu.ITU_MATERIALS_PROPERTIES (ITU-R P.2040). Outside
# these bands Sionna raises "Properties of ITU material ... are not defined for
# this frequency" when the scene frequency is set, which is easy to mistake for a
# problem with the scene.
# Checking here turns that late, confusing failure into an early, specific one.
ITU_BANDS_GHZ = {
    "brick": [(1.0, 40.0)],
    "ceiling_board": [(1.0, 100.0), (220.0, 450.0)],
    "chipboard": [(1.0, 100.0)],
    "concrete": [(1.0, 100.0)],
    "floorboard": [(50.0, 100.0)],
    "glass": [(0.1, 100.0), (220.0, 450.0)],
    "marble": [(1.0, 60.0)],
    "medium_dry_ground": [(1.0, 10.0)],
    "metal": [(1.0, 100.0)],
    "plasterboard": [(1.0, 100.0)],
    "plywood": [(1.0, 40.0)],
    "very_dry_ground": [(1.0, 10.0)],
    "wet_ground": [(1.0, 10.0)],
    "wood": [(0.001, 100.0)],
}
ITU_TYPES = set(ITU_BANDS_GHZ)

# The NIST scene's non-ITU materials. These are placeholders ordered by rough
# density -- NOT the authors' values, whose assignment came from a semantic
# descriptor that was never released. Override with --map if you have better
# numbers.
DEFAULT_CUSTOM_MAP = {
    "custom_plastic": "chipboard",      # eps_r 2.58
    "custom_leather": "wood",           # eps_r 1.99
    "custom_cloth": "ceiling_board",    # eps_r 1.48, light and porous
}


def valid_at(itu_type: str, freq_ghz: float) -> bool:
    """Is this material's ITU data defined at that frequency? Bands may be disjoint."""
    return any(lo <= freq_ghz <= hi for lo, hi in ITU_BANDS_GHZ[itu_type])

# Blender's duplicate suffix: ".001", ".016" and so on, only at the end.
SUFFIX = re.compile(r"\.\d+$")


def resolve_type(mat_id: str, custom_map: dict[str, str]) -> str | None:
    """ITU material type for a Blender material id, or None if unmappable."""
    name = mat_id[4:] if mat_id.startswith("mat-") else mat_id
    base = SUFFIX.sub("", name)
    # Both spellings appear in exports from different Blender plugin versions.
    if base.startswith("itu_") or base.startswith("itu-"):
        itu_type = base[4:]
        # An itu_ prefix on an unknown type is NOT silently passed through:
        # returning None routes it to the "unmapped" report instead.
        return itu_type if itu_type in ITU_TYPES else None
    return custom_map.get(base)


def fix(in_path: str, out_path: str, custom_map: dict[str, str],
        freq_ghz: float = 60.0, verbose: bool = True) -> Counter:
    """Rewrite every <bsdf> into an itu-radio-material. Returns a type histogram."""
    tree = ET.parse(in_path)
    root = tree.getroot()

    # Validate the mapping table before touching the scene, so a bad --map fails
    # immediately rather than after a partial rewrite.
    bad = {t: custom_map[t] for t in custom_map if not valid_at(custom_map[t], freq_ghz)}
    if bad:
        options = sorted(t for t in ITU_TYPES if valid_at(t, freq_ghz))
        raise SystemExit(
            f"these mappings are undefined at {freq_ghz} GHz: {bad}; "
            f"pick from: {options}")

    stats = Counter()
    unmapped = set()
    # Materials appear both at the top level and nested inside shapes.
    for bsdf in root.findall("./bsdf") + root.findall(".//shape/bsdf"):
        mat_id = bsdf.attrib.get("id")
        if not mat_id:
            continue                     # anonymous bsdf; nothing can reference it
        itu_type = resolve_type(mat_id, custom_map)
        if itu_type is not None and not valid_at(itu_type, freq_ghz):
            raise SystemExit(
                f"{mat_id} resolves to ITU material {itu_type!r}, which is not "
                f"defined at {freq_ghz} GHz. Remap it with --map.")
        if itu_type is None:
            # Left as it is, and reported: rewriting it to an arbitrary material
            # would produce a scene that loads but is quietly wrong.
            unmapped.add(mat_id)
            stats["unmapped"] += 1
            continue

        # clear() drops the original plugin's children and attributes; the id is
        # then put back, because the shapes' <ref id="..."> must keep resolving.
        bsdf.clear()
        bsdf.attrib["id"] = mat_id
        bsdf.attrib["name"] = mat_id
        bsdf.attrib["type"] = "itu-radio-material"
        bsdf.append(ET.Element("string", {"name": "type", "value": itu_type}))
        stats[itu_type] += 1

    ET.indent(root, space="\t")          # keep the file diffable
    # No XML declaration: matches how the original was written.
    tree.write(out_path, encoding="utf-8", xml_declaration=False)

    if verbose:
        for itu_type, n in sorted(stats.items(), key=lambda kv: -kv[1]):
            print(f"  {itu_type:20s} {n}")
        if unmapped:
            print(f"\n  {len(unmapped)} materials left untouched, which will fail "
                  f"to load: {sorted(unmapped)[:5]}")
        print(f"\nwrote {out_path}")
    return stats


def main():
    """Parse arguments, apply any --map overrides, and rewrite the scene."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--frequency-ghz", type=float, default=60.0,
                    help="scene frequency, used to reject materials whose ITU "
                         "properties are undefined there (default: 60)")
    ap.add_argument("--map", action="append", default=[], metavar="NAME=ITU_TYPE",
                    help="override a custom material mapping, repeatable")
    args = ap.parse_args()

    custom_map = dict(DEFAULT_CUSTOM_MAP)
    for item in args.map:
        name, _, itu_type = item.partition("=")
        # Rejected here rather than at load time, with the valid list in the message.
        if itu_type not in ITU_TYPES:
            ap.error(f"{itu_type!r} is not an ITU material; pick from "
                     f"{sorted(ITU_TYPES)}")
        custom_map[name] = itu_type

    fix(args.input, args.output, custom_map, args.frequency_ghz)


if __name__ == "__main__":
    main()
