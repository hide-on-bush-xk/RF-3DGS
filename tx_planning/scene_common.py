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

    Only half the fix lives here: the loop_mode switch is in make_solver below,
    because it is a property of the solver object rather than a global flag.
    """
    import drjit as dr
    dr.set_flag(dr.JitFlag.SpillToSharedMemory, False)


def load_radio_scene(scene_xml: str, frequency_hz: float = 60e9,
                     scattering: float = 0.7, rx_elements: int = 1):
    """NIST lobby with ITU materials and a calibrated scattering coefficient.

    Materials load with scattering_coefficient = 0, which yields single-digit
    path counts; 0.7 reproduces the paper's ">300,000 MPCs" at depth 1.

    rx_elements is the side of a square array, so 1 means a single isotropic
    element and 4 means 16. merge_shapes=True collapses the scene's many mesh
    pieces into one, which the solver traverses considerably faster.
    """
    from sionna.rt import PlanarArray, load_scene

    scene = load_scene(scene_xml, merge_shapes=True)
    scene.frequency = frequency_hz
    # Transmitter is always a single isotropic element: planning moves it, and
    # a pattern would confound position with orientation.
    scene.tx_array = PlanarArray(num_rows=1, num_cols=1, pattern="iso",
                                 polarization="V")
    # Half-wavelength spacing, the standard choice for a beamforming array.
    scene.rx_array = PlanarArray(num_rows=rx_elements, num_cols=rx_elements,
                                 vertical_spacing=0.5, horizontal_spacing=0.5,
                                 pattern="iso", polarization="V")
    # Applied uniformly: no attempt to give different surfaces different
    # scattering, since the single value is what was calibrated.
    for material in scene.radio_materials.values():
        material.scattering_coefficient = scattering
    return scene


def make_solver(reverse_mode: bool = False):
    """A PathSolver, switched to evaluated-loop mode when gradients are needed.

    See enable_reverse_mode: symbolic mode (the default) cannot be
    differentiated through here, so anything taking a gradient must pass True.
    Evaluated mode is slower, which is why it is not simply always on.
    """
    from sionna.rt import PathSolver
    solver = PathSolver()
    if reverse_mode:
        solver.loop_mode = "evaluated"
    return solver


def rx_grid(x_range, y_range, z: float, step: float) -> np.ndarray:
    """Candidate receiver positions on a horizontal grid, [N, 3]."""
    # The 1e-9 makes arange inclusive of the upper bound despite float rounding.
    xs = np.arange(x_range[0], x_range[1] + 1e-9, step)
    ys = np.arange(y_range[0], y_range[1] + 1e-9, step)
    # indexing="ij" so the flattened order is x-major, which keeps the grid
    # reshapeable back to (len(xs), len(ys)) for plotting.
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    return np.stack([gx.reshape(-1), gy.reshape(-1),
                     np.full(gx.size, z)], axis=1)


def indoor_mask(scene, points: np.ndarray) -> np.ndarray:
    """True where a point has a ceiling above it and a floor below it.

    The bounding-box grid covers an L-shaped lobby, so half of it lies outside
    the building, where the solver returns no paths at all. On the 4 m smoke
    grid this test agreed with "at least one path" at every point.

    Cheap by design: two ray casts per point rather than a solve. Both must hit,
    so a point under an overhang but with open floor is correctly excluded.
    """
    import mitsuba as mi
    pts = np.asarray(points, dtype=float)
    origin = mi.Point3f(pts[:, 0].tolist(), pts[:, 1].tolist(), pts[:, 2].tolist())
    hits = []
    for dz in (1.0, -1.0):      # straight up, then straight down
        si = scene.mi_scene.ray_intersect(mi.Ray3f(origin, mi.Vector3f(0.0, 0.0, dz)))
        hits.append(np.asarray(si.is_valid()).reshape(-1))
    return hits[0] & hits[1]


def clearance(scene, point) -> float:
    """Distance to the nearest surface along the six axis directions.

    An axis-aligned approximation of the true distance to the nearest surface:
    it can overestimate when the closest geometry lies on a diagonal. Used to
    keep a candidate transmitter from being placed inside or against a wall.
    Returns inf for a point with open space along every axis.
    """
    import mitsuba as mi
    origin = mi.Point3f(*[float(v) for v in point])
    best = np.inf
    for d in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
        si = scene.mi_scene.ray_intersect(mi.Ray3f(origin, mi.Vector3f(*map(float, d))))
        # si.t is inf where the ray escapes, which propagates correctly through min.
        best = min(best, float(np.asarray(si.t).reshape(-1)[0]))
    return best


# The NIST lobby's floor footprint, from the scene bounding box measured in
# render_optical.py: x in [-5.9, 16.0], y in [-21.1, 8.3]. The receiver height
# matches the tutorial's sampling campaign (1.625 - 1.713 = -0.088 m).
# Note these are inset from the true bounding box, so the grid starts inside the
# building rather than wasting candidates on the surrounding empty space.
LOBBY_X = (-4.0, 14.0)
LOBBY_Y = (-19.0, 6.0)
RX_HEIGHT = -0.088
