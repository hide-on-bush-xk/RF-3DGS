"""Matplotlib's jet colormap and its inverse, in torch, without matplotlib.

The RF-3DGS targets are jet-mapped spectra, so an RGB prediction has to come
back through jet to be scored in dB. jet is a curve in RGB space; the inverse
is a nearest-point lookup on 1024 samples of it.
"""

from __future__ import annotations

import numpy as np
import torch

# matplotlib's _jet_data: (x, y0, y1) breakpoints per channel
_SEG = {
    "r": [(0.00, 0.0), (0.35, 0.0), (0.66, 1.0), (0.89, 1.0), (1.00, 0.5)],
    "g": [(0.00, 0.0), (0.125, 0.0), (0.375, 1.0), (0.64, 1.0), (0.91, 0.0), (1.00, 0.0)],
    "b": [(0.00, 0.5), (0.11, 1.0), (0.34, 1.0), (0.65, 0.0), (1.00, 0.0)],
}

_N = 4096
_x = np.linspace(0.0, 1.0, _N)
_LUT_NP = np.stack([np.interp(_x, [p[0] for p in _SEG[c]], [p[1] for p in _SEG[c]])
                    for c in "rgb"], axis=1).astype(np.float32)        # [N,3]
_LUT = {}


def _lut(device):
    if device not in _LUT:
        _LUT[device] = torch.from_numpy(_LUT_NP).to(device)
    return _LUT[device]


def jet_rgb(v):
    """v in [0,1], any shape -> RGB in [0,1], shape [3, *v.shape]."""
    lut = _lut(v.device)
    idx = (v.clamp(0, 1) * (_N - 1)).round().long()
    return lut[idx].movedim(-1, 0)


def jet_inverse(rgb):
    """rgb [3, H, W] in [0,1] -> v [H, W] in [0,1] by nearest LUT entry."""
    lut = _lut(rgb.device)                                   # [N,3]
    px = rgb.movedim(0, -1).reshape(-1, 3)                   # [P,3]
    # squared distance to every LUT entry, chunked to bound memory
    out = torch.empty(px.shape[0], device=rgb.device)
    step = 16384
    for s in range(0, px.shape[0], step):
        d = torch.cdist(px[s:s + step], lut)
        out[s:s + step] = d.argmin(dim=1).float() / (_N - 1)
    return out.reshape(rgb.shape[1:])


if __name__ == "__main__":
    v = torch.rand(200, 300)
    err = (jet_inverse(jet_rgb(v)) - v).abs().max()
    print(f"round trip max error {float(err):.5f} (1/{_N-1} = {1/(_N-1):.5f})")
