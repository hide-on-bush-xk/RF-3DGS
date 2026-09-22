"""Matplotlib's jet colormap and its inverse, in torch, without matplotlib.

The RF-3DGS targets are jet-mapped spectra, so an RGB prediction has to come
back through jet to be scored in dB. jet is a curve in RGB space; the inverse
is a nearest-point lookup on 1024 samples of it.

(The docstring says 1024; the table below is in fact _N = 4096 entries, which
is what the round-trip check at the bottom reports against.)
"""

from __future__ import annotations

import numpy as np
import torch

# matplotlib's _jet_data: (x, y0, y1) breakpoints per channel
# Copied rather than imported, so nothing here depends on matplotlib being
# installed -- this runs inside the training loop.
_SEG = {
    "r": [(0.00, 0.0), (0.35, 0.0), (0.66, 1.0), (0.89, 1.0), (1.00, 0.5)],
    "g": [(0.00, 0.0), (0.125, 0.0), (0.375, 1.0), (0.64, 1.0), (0.91, 0.0), (1.00, 0.0)],
    "b": [(0.00, 0.5), (0.11, 1.0), (0.34, 1.0), (0.65, 0.0), (1.00, 0.0)],
}

# 4096 entries. The table resolution (1/4095) is a lower bound on the round-trip
# error, but NOT the actual one: see the dead band noted on jet_inverse below.
_N = 4096
_x = np.linspace(0.0, 1.0, _N)
_LUT_NP = np.stack([np.interp(_x, [p[0] for p in _SEG[c]], [p[1] for p in _SEG[c]])
                    for c in "rgb"], axis=1).astype(np.float32)        # [N,3]
_LUT = {}


def _lut(device):
    """The lookup table on `device`, built once per device and cached."""
    if device not in _LUT:
        _LUT[device] = torch.from_numpy(_LUT_NP).to(device)
    return _LUT[device]


def jet_rgb(v):
    """v in [0,1], any shape -> RGB in [0,1], shape [3, *v.shape]."""
    lut = _lut(v.device)
    # Nearest table entry; no interpolation, so this is exact only to 1/(_N-1).
    idx = (v.clamp(0, 1) * (_N - 1)).round().long()
    return lut[idx].movedim(-1, 0)


def jet_inverse(rgb):
    """rgb [3, H, W] in [0,1] -> v [H, W] in [0,1] by nearest LUT entry.

    jet is not injective as a map from arbitrary RGB: a predicted colour off
    the jet curve is snapped to the closest point on it, which is the intended
    behaviour when scoring a network's RGB output in dB.

    It is also not injective ALONG the curve. jet has a flat segment where all
    three channels are pinned -- r = 0, g = 0, b = 1 for v in [0.110, 0.125],
    which is 60 consecutive table entries with zero RGB change. A value landing
    there decodes to an arbitrary point in the band, so the worst-case round
    trip is 0.0149, not the 1/4095 the table resolution suggests (measured by
    the __main__ check below; 1.3 % of the range is affected). Over a 60 dB
    span that is about 0.9 dB of irreducible decode error for those values --
    worth knowing before attributing such an error to a model.
    """
    lut = _lut(rgb.device)                                   # [N,3]
    px = rgb.movedim(0, -1).reshape(-1, 3)                   # [P,3]
    # squared distance to every LUT entry, chunked to bound memory
    # A full cdist would be P x 4096 floats at once; 16384-pixel chunks keep
    # that under control for a 300x200 image and larger.
    out = torch.empty(px.shape[0], device=rgb.device)
    step = 16384
    for s in range(0, px.shape[0], step):
        d = torch.cdist(px[s:s + step], lut)
        out[s:s + step] = d.argmin(dim=1).float() / (_N - 1)
    return out.reshape(rgb.shape[1:])


if __name__ == "__main__":
    # Self-check. Note what it reports: the max error is ~0.0149, dominated by
    # the flat band described on jet_inverse, not by the table resolution it is
    # printed against. A much larger number would mean something is actually broken.
    v = torch.rand(200, 300)
    err = (jet_inverse(jet_rgb(v)) - v).abs().max()
    print(f"round trip max error {float(err):.5f} (1/{_N-1} = {1/(_N-1):.5f})")
