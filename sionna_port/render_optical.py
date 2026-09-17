"""Render the scene from a receiver pose, so the view and the spectrum match.

The published Blender renders and the RF spectra come from two unrelated
sampling campaigns -- 809 camera poses at 1600x900 for the geometry stage,
3200 receiver poses at 300x200 and 90 degrees for the spectra -- so no optical
image corresponds to any given spectrum. Rather than hunt for a near match, this
renders the scene through a camera at the receiver's own position and boresight,
at the same field of view.

Mitsuba renders the *visual* scene almost black: its only emitter is a constant
environment light of radiance 0.08, and a constant emitter is infinitely far
away, so inside a closed room no ray ever reaches it. Blender lit these renders
with its own area lights, which the Mitsuba export did not carry over.

Sionna's renderer is used instead. It shades by radio material, which is more
useful here than a photograph would be: a bright patch in the spectrum can be
read against the surface that produced it, metal or glass or plasterboard.
"""

from __future__ import annotations

import argparse
import math
import os

import numpy as np


def boresight(position, yaw_rad: float):
    """A point one metre along the array boresight.

    Sionna's receiver orientation is (yaw, pitch, roll) on an array whose
    initial direction is +x -- the frame the tutorial's euler_to_quaternion
    maps onto COLMAP's.
    """
    direction = np.array([math.cos(yaw_rad), math.sin(yaw_rad), 0.0])
    # Plain floats: mi.Point3f rejects numpy scalars.
    return [float(v) for v in np.asarray(position, dtype=float) + direction]


def render_pose(scene, position, yaw_rad: float, width: int = 300,
                height: int = 200, fov_deg: float = 90.0,
                num_samples: int = 128, paths=None, lighting_scale: float = 1.0,
                clip_above: float | None = 1.0):
    """One view. `clip_above` is a height, not a preference.

    Sionna lights the scene from outside, so a camera standing inside a closed
    building renders black -- measured mean 0.0000 against 0.128 for the same
    scene seen from above. Clipping the geometry a metre over the receiver lets
    the light in while leaving everything the receiver can see intact.
    """
    from sionna.rt import Camera

    camera = Camera(position=[float(v) for v in position],
                    look_at=boresight(position, yaw_rad))
    extra = {} if clip_above is None else {
        "clip_at": float(position[2]) + clip_above}
    bitmap = scene.render(camera=camera, fov=fov_deg,
                          resolution=(width, height),
                          num_samples=num_samples,
                          lighting_scale=lighting_scale,
                          paths=paths, show_devices=False,
                          return_bitmap=True, **extra)
    arr = np.array(bitmap.convert(srgb_gamma=True, component_format=1))  # UInt8
    return arr[..., :3]


def load_radio_scene(scene_xml: str, frequency: float = 60e9,
                     scattering: float = 0.7):
    from sionna.rt import load_scene
    scene = load_scene(scene_xml, merge_shapes=True)
    scene.frequency = frequency
    for material in scene.radio_materials.values():
        material.scattering_coefficient = scattering
    return scene


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene-xml", required=True)
    ap.add_argument("--position", type=float, nargs=3, default=[3.0, -2.0, 0.0])
    ap.add_argument("--yaw-deg", type=float, nargs="+",
                    default=[-90.0, 0.0, 90.0, 180.0])
    ap.add_argument("--out-dir", default="../output/optical")
    ap.add_argument("--width", type=int, default=300)
    ap.add_argument("--height", type=int, default=200)
    ap.add_argument("--fov", type=float, default=90.0)
    ap.add_argument("--samples", type=int, default=128)
    ap.add_argument("--lighting-scale", type=float, default=1.0)
    ap.add_argument("--clip-above", type=float, default=1.0,
                    help="clip the scene this far above the receiver so light "
                         "reaches the interior; None to disable")
    ap.add_argument("--variant", default="cuda_ad_mono_polarized")
    args = ap.parse_args()

    import mitsuba as mi
    mi.set_variant(args.variant)
    import imageio.v2 as imageio

    scene = load_radio_scene(args.scene_xml)
    os.makedirs(args.out_dir, exist_ok=True)
    for yaw in args.yaw_deg:
        img = render_pose(scene, args.position, math.radians(yaw), args.width,
                          args.height, args.fov, args.samples,
                          lighting_scale=args.lighting_scale,
                          clip_above=args.clip_above)
        name = f"optical_yaw{yaw:+.0f}.png"
        imageio.imwrite(os.path.join(args.out_dir, name), img)
        print(f"  {name}  mean {img.mean():.1f}")


if __name__ == "__main__":
    main()
