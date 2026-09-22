"""Scene 2: a corridor with closed rooms, built procedurally.

Only the topology differs from scene 1 (the open L-shaped NIST lobby):
materials, frequency, array, kernel, channels, sampling protocol and
training budget stay the same. Floor plan (metres, z up; the lobby's
conventions are kept: floor at z = -1.713, receivers at -0.088, i.e. 1.625 m
above the floor, transmitters at 0.287, i.e. 2 m):

  corridor        x in [0, 28], y in [-1.2, 1.2], 3 m high
  rooms           three on each side, 5 x 4.5 m, walls 0.2 m, door openings
                  0.9 x 2.1 m in the corridor wall (openings only, no door leaf)
  hall            x in [28, 38], y in [-4, 4], the open-space control

Outputs, under --out:
  meshes/*.ply            one mesh per wall / floor / ceiling piece, with uv
  corridor_sionna.xml     Sionna scene (itu-radio-material per surface class)
  corridor_visual.xml     Mitsuba scene for rendering the visual dataset:
                          the same meshes with textured diffuse BSDFs (every
                          surface needs photometric gradients or 3DGS puts no
                          geometry there) and area lights under the ceiling
  textures/*.png          noise textures, one per surface class
  rx_route.txt            receiver route in the NIST file format (idx x y z, mm)
  tx_positions.json       ten transmitters: corridor ends and middle, one per
                          room, one in the hall
  layout.json             the rectangles, for indoor tests and plots

    PYTHONUTF8=1 python scene2/make_corridor.py --out scene2/corridor
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

# Heights, in metres, inherited from the NIST lobby so the two scenes share a
# vertical frame. FLOOR is the floor surface; the ceiling sits 3 m above it.
FLOOR, CEIL = -1.713, -1.713 + 3.0
RX_Z, TX_Z = -0.088, 0.287       # receiver 1.625 m and transmitter 2.0 m above the floor
WALL = 0.2                       # wall thickness
CORR = dict(x0=0.0, x1=28.0, y0=-1.2, y1=1.2)
HALL = dict(x0=28.0, x1=38.0, y0=-4.0, y1=4.0)
# Room extents along x. Each successive room is offset by the partition thickness
# so that neighbouring rooms share a wall rather than overlapping.
ROOM_X = [(1.0, 6.0), (6.0 + WALL, 11.0 + WALL), (11.0 + 2 * WALL, 16.0 + 2 * WALL)]
ROOM_D = 4.5                     # room depth, measured away from the corridor
DOOR_W, DOOR_H = 0.9, 2.1
# Surface class -> ITU material name, used for the Sionna (radio) scene only.
CLASSES = {"wall": "plasterboard", "floor": "concrete", "ceiling": "ceiling_board"}


def rooms():
    """The six room rectangles, three on each side of the corridor.

    `side` is +1 for the north row and -1 for the south; y0/y1 are normalised so
    y0 < y1 regardless of which side the room is on.
    """
    out = []
    for side in (+1, -1):
        for k, (xa, xb) in enumerate(ROOM_X):
            y_in = side * (CORR["y1"] + WALL)                     # inner face of the corridor wall
            y_out = y_in + side * ROOM_D
            out.append(dict(name=f"room_{'N' if side > 0 else 'S'}{k+1}", x0=xa, x1=xb, y0=min(y_in, y_out), y1=max(y_in, y_out), side=side))
    return out


def box(x0, x1, y0, y1, z0, z1):
    """Six quads (outward normals), each with planar uv (0.5 m per texture repeat).

    Returns (verts, uvs, tris). Vertices are duplicated per face rather than
    shared across the box, so each face carries its own uv without seams.
    """
    P = lambda x, y, z: (x, y, z)
    # Each quad is wound so its normal points away from the box interior.
    faces = {
        "-x": [P(x0, y0, z0), P(x0, y0, z1), P(x0, y1, z1), P(x0, y1, z0)], "+x": [P(x1, y0, z0), P(x1, y1, z0), P(x1, y1, z1), P(x1, y0, z1)],
        "-y": [P(x0, y0, z0), P(x1, y0, z0), P(x1, y0, z1), P(x0, y0, z1)], "+y": [P(x0, y1, z0), P(x0, y1, z1), P(x1, y1, z1), P(x1, y1, z0)],
        "-z": [P(x0, y0, z0), P(x0, y1, z0), P(x1, y1, z0), P(x1, y0, z0)], "+z": [P(x0, y0, z1), P(x1, y0, z1), P(x1, y1, z1), P(x0, y1, z1)],
    }
    verts, uvs, tris = [], [], []
    for axis, quad in faces.items():
        base = len(verts)
        for (x, y, z) in quad:
            verts.append((x, y, z))
            # Planar projection: drop the axis the face is perpendicular to, then
            # divide by 0.5 m so the texture repeats every half metre.
            u, v = {"x": (y, z), "y": (x, z), "z": (x, y)}[axis[1]]
            uvs.append((u / 0.5, v / 0.5))
        # Fan-triangulate the quad into two triangles sharing the 0-2 diagonal.
        tris += [(base, base + 1, base + 2), (base, base + 2, base + 3)]
    return verts, uvs, tris


def write_ply(path, verts, uvs, tris):
    """Write an ASCII PLY with per-vertex uv, the form both Mitsuba and Sionna read."""
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(verts)}\nproperty float x\nproperty float y\nproperty float z\nproperty float u\nproperty float v\n")
        f.write(f"element face {len(tris)}\nproperty list uchar int vertex_indices\nend_header\n")
        for (x, y, z), (u, v) in zip(verts, uvs):
            f.write(f"{x:.4f} {y:.4f} {z:.4f} {u:.4f} {v:.4f}\n")
        for a, b, c in tris:
            f.write(f"3 {a} {b} {c}\n")          # leading 3 = vertices in this face


def build(out):
    """Generate every output listed in the module docstring into `out`."""
    os.makedirs(os.path.join(out, "meshes"), exist_ok=True); os.makedirs(os.path.join(out, "textures"), exist_ok=True)
    pieces = []                                                    # (name, class, box)
    def add(name, cls, x0, x1, y0, y1, z0=FLOOR, z1=CEIL):
        """Record one axis-aligned box; defaults span floor to ceiling."""
        pieces.append((name, cls, (x0, x1, y0, y1, z0, z1)))
    spaces = [dict(name="corridor", **CORR), dict(name="hall", **HALL)] + rooms()
    # floors and ceilings, one slab per space (slightly below / above so they do not z-fight with walls)
    for s in spaces:
        add(f"floor_{s['name']}", "floor", s["x0"], s["x1"], s["y0"], s["y1"], FLOOR - 0.1, FLOOR)
        add(f"ceiling_{s['name']}", "ceiling", s["x0"], s["x1"], s["y0"], s["y1"], CEIL, CEIL + 0.1)
    # corridor walls, with door openings where a room sits behind
    for side in (+1, -1):
        # The wall slab occupies the 0.2 m between the corridor and the rooms.
        y_a, y_b = (CORR["y1"], CORR["y1"] + WALL) if side > 0 else (CORR["y0"] - WALL, CORR["y0"])
        x = CORR["x0"]
        # Walk along the corridor leaving a gap at each room's centre. `x` tracks
        # where the previous segment ended, so the pieces tile without overlap.
        for k, (xa, xb) in enumerate(ROOM_X):
            dx = (xa + xb) / 2                                   # door centre
            add(f"corrwall_{side}_{k}a", "wall", x, dx - DOOR_W / 2, y_a, y_b)
            # Above the opening: a lintel from door height up to the ceiling.
            add(f"corrwall_{side}_{k}lintel", "wall", dx - DOOR_W / 2, dx + DOOR_W / 2, y_a, y_b, FLOOR + DOOR_H, CEIL)
            x = dx + DOOR_W / 2
        add(f"corrwall_{side}_end", "wall", x, CORR["x1"], y_a, y_b)
    add("corr_west_end", "wall", CORR["x0"] - WALL, CORR["x0"], CORR["y0"] - WALL, CORR["y1"] + WALL)
    # room partitions and outer walls
    for side in (+1, -1):
        rs = [r for r in rooms() if r["side"] == side]
        # The far wall, on whichever side of the room row is away from the corridor.
        y_far0, y_far1 = (rs[0]["y1"], rs[0]["y1"] + WALL) if side > 0 else (rs[0]["y0"] - WALL, rs[0]["y0"])
        add(f"outer_{side}", "wall", rs[0]["x0"] - WALL, rs[-1]["x1"] + WALL, y_far0, y_far1)
        add(f"roomend_{side}_w", "wall", rs[0]["x0"] - WALL, rs[0]["x0"], rs[0]["y0"], rs[0]["y1"])
        add(f"roomend_{side}_e", "wall", rs[-1]["x1"], rs[-1]["x1"] + WALL, rs[0]["y0"], rs[0]["y1"])
        # Partitions fill the ROOM_X gaps between consecutive rooms.
        for k in range(len(rs) - 1):
            add(f"partition_{side}_{k}", "wall", rs[k]["x1"], rs[k + 1]["x0"], rs[0]["y0"], rs[0]["y1"])
    # hall walls (open to the corridor through the corridor's east end)
    add("hall_east", "wall", HALL["x1"], HALL["x1"] + WALL, HALL["y0"] - WALL, HALL["y1"] + WALL)
    add("hall_north", "wall", HALL["x0"], HALL["x1"], HALL["y1"], HALL["y1"] + WALL)
    add("hall_south", "wall", HALL["x0"], HALL["x1"], HALL["y0"] - WALL, HALL["y0"])
    # West side of the hall is split in two so the corridor mouth stays open.
    add("hall_west_n", "wall", HALL["x0"] - WALL, HALL["x0"], CORR["y1"] + WALL, HALL["y1"] + WALL)
    add("hall_west_s", "wall", HALL["x0"] - WALL, HALL["x0"], HALL["y0"] - WALL, CORR["y0"] - WALL)

    # meshes
    for name, cls, b in pieces:
        write_ply(os.path.join(out, "meshes", f"{name}.ply"), *box(*b))
    # textures: coloured noise, one per class (photometric gradients everywhere)
    from PIL import Image
    rng = np.random.default_rng(7)       # fixed seed: textures are reproducible
    tint = {"wall": (0.75, 0.72, 0.65), "floor": (0.45, 0.42, 0.40), "ceiling": (0.85, 0.85, 0.82)}
    for cls, t in tint.items():
        # multi-octave blotches (8 / 32 / 128 cells per 512 px), no per-pixel noise: features that
        # survive the renderer's pixel filter and the trainer's downsampling, so 3DGS can fit them
        # (per-pixel noise aliased into an unfittable residual: stage 1 stalled at 62k Gaussians, 17.8 dB)
        img = np.zeros((512, 512, 1))
        for cells, amp in ((8, 0.45), (32, 0.35), (128, 0.20)):
            # np.kron upsamples a coarse random grid into blocks, giving blotches
            # of a controlled size rather than per-pixel noise.
            img += amp * np.kron(rng.random((cells, cells, 1)), np.ones((512 // cells, 512 // cells, 1)))
        # 0.35 + 0.65 * img keeps the texture from ever going fully black, so no
        # surface is left without photometric signal.
        img = np.clip(255 * np.array(t)[None, None] * (0.35 + 0.65 * img), 0, 255).astype(np.uint8)
        Image.fromarray(img).save(os.path.join(out, "textures", f"{cls}.png"))

    def xml(visual):
        """Emit the scene XML. visual=True gives the Mitsuba render scene
        (textured diffuse BSDFs + area lights), visual=False the Sionna radio
        scene (ITU materials, no lights). Geometry is identical in both."""
        L = ['<scene version="2.1.0">', '\t<default name="spp" value="256" />', '\t<default name="resx" value="1600" />', '\t<default name="resy" value="900" />',
             # max_depth 8 bounces: enough for indirect light in closed rooms.
             '\t<integrator type="path" id="elm__0" name="elm__0">', '\t\t<integer name="max_depth" value="8" />', '\t</integrator>',
             '\t<sensor type="perspective" id="Camera" name="Camera">', '\t\t<string name="fov_axis" value="x" />', '\t\t<float name="fov" value="90.0" />',
             '\t\t<float name="near_clip" value="0.05" />', '\t\t<float name="far_clip" value="1000.0" />',
             # A default camera in the corridor; render_visual.py overrides it per frame.
             '\t\t<transform name="to_world"><lookat origin="14 0 -0.088" target="15 0 -0.088" up="0 0 1" /></transform>',
             '\t\t<sampler type="independent" name="sampler"><integer name="sample_count" value="$spp" /></sampler>',
             '\t\t<film type="hdrfilm" name="film"><integer name="width" value="$resx" /><integer name="height" value="$resy" />'
             '<string name="pixel_format" value="rgba" /><rfilter type="gaussian" name="rfilter" /></film>', '\t</sensor>']
        for cls, itu in CLASSES.items():
            if visual:
                # to_uv scale 0.25 on top of the 0.5 m repeat baked into the mesh uvs.
                L += [f'\t<bsdf type="diffuse" id="mat-{cls}" name="mat-{cls}">',
                      f'\t\t<texture type="bitmap" name="reflectance"><string name="filename" value="textures/{cls}.png" />'
                      f'<transform name="to_uv"><scale value="0.25" /></transform></texture>', '\t</bsdf>']
            else:
                L += [f'\t<bsdf id="mat-itu_{itu}" name="mat-itu_{itu}" type="itu-radio-material">', f'\t\t<string name="type" value="{itu}" />', '\t</bsdf>']
        for name, cls, _ in pieces:
            ref = f"mat-{cls}" if visual else f"mat-itu_{CLASSES[cls]}"
            L += [f'\t<shape type="ply" id="mesh-{name}" name="mesh-{name}">', f'\t\t<string name="filename" value="meshes/{name}.ply" />',
                  f'\t\t<ref id="{ref}" name="bsdf" />', '\t</shape>']
        if visual:
            # area lights just under the ceiling: corridor every 4 m, one per room, four in the hall
            lights = [(x, 0.0) for x in np.arange(2.0, 28.0, 4.0)] + [((r["x0"] + r["x1"]) / 2, (r["y0"] + r["y1"]) / 2) for r in rooms()]
            lights += [(30.5, -2.0), (30.5, 2.0), (35.5, -2.0), (35.5, 2.0)]
            for i, (x, y) in enumerate(lights):
                # rotate 180 about x turns the rectangle's emitting face downward;
                # 0.02 m below the ceiling avoids coplanar z-fighting.
                L += [f'\t<shape type="rectangle" id="light_{i}" name="light_{i}">',
                      f'\t\t<transform name="to_world"><scale x="0.5" y="0.5" /><rotate x="1" angle="180" /><translate x="{x}" y="{y}" z="{CEIL - 0.02}" /></transform>',
                      '\t\t<emitter type="area"><rgb name="radiance" value="40" /></emitter>', '\t</shape>']
        L.append('</scene>')
        return "\n".join(L) + "\n"
    open(os.path.join(out, "corridor_sionna.xml"), "w").write(xml(False))
    open(os.path.join(out, "corridor_visual.xml"), "w").write(xml(True))

    # receiver route: corridor centreline, a U in every room (0.9 m inset), three lanes in the hall
    def polyline(pts, step):
        """Sample a polyline at roughly `step` metres, without duplicating joints.

        endpoint=False on each segment is what prevents a doubled sample where two
        segments meet; the final point is appended once at the end.
        """
        pts = np.array(pts, float); out_ = []
        for a, b in zip(pts[:-1], pts[1:]):
            n = max(1, int(np.ceil(np.linalg.norm(b - a) / step)))
            out_ += [a + (b - a) * t for t in np.linspace(0, 1, n, endpoint=False)]
        return out_ + [pts[-1]]
    step = 0.2
    route = {"corridor": polyline([(0.6, 0.0), (27.4, 0.0)], step)}
    for r in rooms():
        # A U-shaped path inset 0.9 m from the walls, so receivers stay clear of them.
        i = 0.9; xa, xb, ya, yb = r["x0"] + i, r["x1"] - i, r["y0"] + i, r["y1"] - i
        route[r["name"]] = polyline([(xa, ya), (xb, ya), (xb, yb), (xa, yb)], step)
    route["hall"] = polyline([(29.0, -2.5), (37.0, -2.5), (37.0, 0.0), (29.0, 0.0), (29.0, 2.5), (37.0, 2.5)], step)
    # interleave the spaces so any prefix of the file is spread over the plan
    # (round-robin across spaces). This is what makes a truncated route -- a smoke
    # run, say -- still cover every space rather than only the corridor.
    counts = {k: len(v) for k, v in route.items()}
    order, lists = [], list(route.values())
    while any(lists):
        for l in lists:
            if l:
                order.append(l.pop(0))
    with open(os.path.join(out, "rx_route.txt"), "w") as f:
        for i, (x, y) in enumerate(order):
            f.write(f"{i} {x*1000:.0f} {y*1000:.0f} {1625}\n")      # NIST format: mm; the loader jitters and sets the height itself
    # Transmitters: three along the corridor, one in the hall, one per room.
    tx = {"corridor_W": (1.5, 0.0, TX_Z), "corridor_M": (14.0, 0.0, TX_Z), "corridor_E": (27.0, 0.0, TX_Z), "hall": (33.0, 0.0, TX_Z)}
    for r in rooms():
        tx[r["name"]] = ((r["x0"] + r["x1"]) / 2, (r["y0"] + r["y1"]) / 2, TX_Z)
    # same-room pairs: a second transmitter 1.5 m along x in three rooms (the target S2 and two controls)
    for r in rooms():
        if r["name"] in ("room_S2", "room_N2", "room_S1"):
            tx[r["name"] + "b"] = ((r["x0"] + r["x1"]) / 2 + 1.5, (r["y0"] + r["y1"]) / 2, TX_Z)
    json.dump(tx, open(os.path.join(out, "tx_positions.json"), "w"), indent=1)
    json.dump({"floor_z": FLOOR, "ceil_z": CEIL, "rx_z": RX_Z, "tx_z": TX_Z, "spaces": spaces, "pieces": [(n, c, b) for n, c, b in pieces]},
              open(os.path.join(out, "layout.json"), "w"), indent=1)
    print(f"{len(pieces)} meshes, {len(order)} route positions ({', '.join(f'{k} {v}' for k, v in counts.items())}), {len(tx)} transmitters -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "corridor"))
    build(ap.parse_args().out)
