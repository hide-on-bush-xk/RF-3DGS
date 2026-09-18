"""Scene setup shared by the transmitter-planning scripts.

Everything here is Tx-independent by construction: geometry and materials come
from the scene file, and the transmitter is added by the caller. That split is
the whole point of Tx-side planning -- the field being optimised over is the one
thing that does not change when the transmitter moves.
"""

from __future__ import annotations

import numpy as np


def enable_reverse_mode():
    """The two settings without which Sionna 2.1 cannot differentiate here.

    The solver's default symbolic Dr.Jit loop refuses reverse mode without a
    max_iterations hint that Sionna never passes; evaluated mode needs none.
    The evaluated reverse kernel then dies in ptxas with "Smem spilling should
    not be enabled when functions use abi", which this flag turns off.
    """
    import drjit as dr
    dr.set_flag(dr.JitFlag.SpillToSharedMemory, False)


def load_radio_scene(scene_xml: str, frequency_hz: float = 60e9,
                     scattering: float = 0.7, rx_elements: int = 1):
    """NIST lobby with ITU materials and a calibrated scattering coefficient.

    Materials load with scattering_coefficient = 0, which yields single-digit
    path counts; 0.7 reproduces the paper's ">300,000 MPCs" at depth 1.
    """
    from sionna.rt import PlanarArray, load_scene

    scene = load_scene(scene_xml, merge_shapes=True)
    scene.frequency = frequency_hz
    scene.tx_array = PlanarArray(num_rows=1, num_cols=1, pattern="iso",
                                 polarization="V")
    scene.rx_array = PlanarArray(num_rows=rx_elements, num_cols=rx_elements,
                                 vertical_spacing=0.5, horizontal_spacing=0.5,
                                 pattern="iso", polarization="V")
    for material in scene.radio_materials.values():
        material.scattering_coefficient = scattering
    return scene


def make_solver(reverse_mode: bool = False):
    from sionna.rt import PathSolver
    solver = PathSolver()
    if reverse_mode:
        solver.loop_mode = "evaluated"
    return solver


def rx_grid(x_range, y_range, z: float, step: float) -> np.ndarray:
    """Candidate receiver positions on a horizontal grid, [N, 3]."""
    xs = np.arange(x_range[0], x_range[1] + 1e-9, step)
    ys = np.arange(y_range[0], y_range[1] + 1e-9, step)
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    return np.stack([gx.reshape(-1), gy.reshape(-1),
                     np.full(gx.size, z)], axis=1)


def indoor_mask(scene, points: np.ndarray) -> np.ndarray:
    """True where a point has a ceiling above it and a floor below it.

    The bounding-box grid covers an L-shaped lobby, so half of it lies outside
    the building, where the solver returns no paths at all. On the 4 m smoke
    grid this test agreed with "at least one path" at every point.
    """
    import mitsuba as mi
    pts = np.asarray(points, dtype=float)
    origin = mi.Point3f(pts[:, 0].tolist(), pts[:, 1].tolist(), pts[:, 2].tolist())
    hits = []
    for dz in (1.0, -1.0):
        si = scene.mi_scene.ray_intersect(mi.Ray3f(origin, mi.Vector3f(0.0, 0.0, dz)))
        hits.append(np.asarray(si.is_valid()).reshape(-1))
    return hits[0] & hits[1]


def clearance(scene, point) -> float:
    """Distance to the nearest surface along the six axis directions."""
    import mitsuba as mi
    origin = mi.Point3f(*[float(v) for v in point])
    best = np.inf
    for d in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
        si = scene.mi_scene.ray_intersect(mi.Ray3f(origin, mi.Vector3f(*map(float, d))))
        best = min(best, float(np.asarray(si.t).reshape(-1)[0]))
    return best


# The NIST lobby's floor footprint, from the scene bounding box measured in
# render_optical.py: x in [-5.9, 16.0], y in [-21.1, 8.3]. The receiver height
# matches the tutorial's sampling campaign (1.625 - 1.713 = -0.088 m).
LOBBY_X = (-4.0, 14.0)
LOBBY_Y = (-19.0, 6.0)
RX_HEIGHT = -0.088
