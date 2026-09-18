"""The tutorial notebook's material definitions, applied to a Sionna 2.x scene.

Cell 6 of RF-3DGS-tutorial.ipynb defines one RadioMaterial per material class
(permittivity, conductivity, scattering coefficient x a global factor of 4,
a DirectivePattern lobe) and swaps every object's `itu_<x>[.NNN]` material for
`custom_<x>`. This reproduces that on a scene loaded here, by material name.

Two variants:
  asis   exactly the notebook. Its conductivity formulas divide the frequency
         by 1e-9 instead of 1e9, so every ITU-derived conductivity comes out
         around 1e16-1e17 S/m: those walls are effectively perfect conductors.
  fixed  the same with f in GHz, i.e. the ITU-R P.2040 values the formulas
         were copied from.
"""

from __future__ import annotations

GLOBAL_SCATTERING = 4.0

# name: (eps_r, (c, d) for sigma = c * f_GHz**d, or a constant sigma, s x global, alpha_r)
TUTORIAL = {
    "plastic":       (2.3,   0.0,               0.05, 6),
    "leather":       (1.8,   0.0,               0.1,  3),
    "cloth":         (1.8,   0.0,               0.2,  3),
    "plasterboard":  (2.73,  (0.0085, 0.9395),  0.1,  5),
    "wood":          (1.99,  (0.0047, 1.0718),  0.2,  3),
    "concrete":      (5.24,  (0.0462, 1.0718),  0.1,  5),
    "metal":         (1.0,   1e7,               0.025, 10),
    "glass":         (6.31,  (0.0036, 1.3394),  0.025, 10),
    "ceiling_board": (1.48,  (0.0011, 1.0750),  0.1,  5),
    "chipboard":     (2.58,  (0.0217, 0.7800),  0.1,  5),
    "marble":        (7.074, (0.0055, 0.9262),  0.05, 10),
}


def base_name(material_name: str) -> str:
    """'itu_plasterboard.016' / 'custom_plastic.002' / 'mat-itu_metal' -> 'plasterboard'."""
    n = material_name
    for prefix in ("mat-", "itu_", "custom_"):
        if n.startswith(prefix):
            n = n[len(prefix):]
    return n.split(".")[0]


def conductivity(spec, frequency_hz: float, variant: str) -> float:
    if not isinstance(spec, tuple):
        return float(spec)
    c, d = spec
    f = frequency_hz / (1e-9 if variant == "asis" else 1e9)
    return float(c * f ** d)


def apply(scene, frequency_hz: float, variant: str = "asis", verbose: bool = True):
    """Replace each object's material with the tutorial's definition of its class.

    Returns {base name: RadioMaterial} for the classes found.
    """
    from sionna.rt import RadioMaterial

    made = {}
    unknown = set()
    for obj in scene.objects.values():
        base = base_name(obj.radio_material.name)
        if base not in TUTORIAL:
            unknown.add(obj.radio_material.name)
            continue
        if base not in made:
            eps, sigma, s, alpha = TUTORIAL[base]
            # 2.x takes the pattern by registry name, its parameters as kwargs
            made[base] = RadioMaterial(
                name=f"tutorial_{base}", relative_permittivity=eps,
                conductivity=conductivity(sigma, frequency_hz, variant),
                scattering_coefficient=s * GLOBAL_SCATTERING,
                scattering_pattern="directive", alpha_r=alpha)
        obj.radio_material = made[base]
    if verbose:
        print(f"tutorial materials ({variant}): {len(made)} classes on {len(scene.objects)} objects"
              + (f"; unmapped: {sorted(unknown)}" if unknown else ""))
        import numpy as np

        def val(t):
            return float(np.asarray(t).reshape(-1)[0])

        for base, m in sorted(made.items()):
            print(f"  {base:14s} eps_r {val(m.relative_permittivity):6.2f}  sigma {val(m.conductivity):9.3e} S/m"
                  f"  s {val(m.scattering_coefficient):.2f}  alpha_r {TUTORIAL[base][3]}")
    return made
