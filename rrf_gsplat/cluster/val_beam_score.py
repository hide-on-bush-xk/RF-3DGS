"""sionna_port/t4_score.py with the beam-gain loss (--cov), on a machine without Sionna (docs/cluster_log.md §7).

t4_score builds the matched-beam gain map from the receive array's manifold, whose element pattern comes from
sionna_port/generate_dataset.element_gain_fn -> sionna.rt's v_tr38901_pattern (drjit / mitsuba). Here that one
function is supplied in torch from 3GPP TR 38.901 Table 7.3-1 (vertical polarisation): theta_3dB = phi_3dB = 65 deg,
A_V = -min(12 ((theta - 90 deg) / theta_3dB)^2, 30), A_H = -min(12 (phi / phi_3dB)^2, 30), A = -min(-(A_V + A_H), 30)
+ 8 dBi, co-polar field sqrt(10^(A / 10)) with zero phase; phi wrapped to [-pi, pi). Everything else is t4_score.py
unchanged; validated by reproducing the look-ups' beam-gain loss of stage2_notes §16 (the known-answer check).

    python rrf_gsplat/cluster/val_beam_score.py <t4_score.py arguments>
"""

import math
import os
import runpy
import sys
import types

import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def element_gain_fn(theta, phi):
    th, ph = theta.double(), phi.double()
    ph = ph - torch.floor((ph + math.pi) / (2 * math.pi)) * 2 * math.pi
    d3 = math.radians(65.0)
    a_v = -torch.clamp(12 * ((th - math.pi / 2) / d3) ** 2, max=30.0)
    a_h = -torch.clamp(12 * (ph / d3) ** 2, max=30.0)
    a_db = -torch.clamp(-(a_v + a_h), max=30.0) + 8.0
    return torch.sqrt(10 ** (a_db / 10)).to(torch.complex64).to(theta.device).reshape(theta.shape)


fake = types.ModuleType("generate_dataset")
fake.element_gain_fn = element_gain_fn
sys.modules["generate_dataset"] = fake
sys.path.insert(0, os.path.join(REPO, "sionna_port"))
sys.path.insert(0, os.path.join(REPO, "rrf_gsplat"))
sys.argv = [os.path.join(REPO, "sionna_port", "t4_score.py")] + sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")
