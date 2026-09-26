"""Information ladder, step 3 (docs/cluster_log.md §6): mirror emitters from the visual geometry only, no RF labels.

    nice -n 10 ~/envs/rf-gsplat/bin/python rrf_gsplat/cluster/ladder_step3_prep.py

Large planes: surface Gaussians (opacity >= 0.5) whose shortest axis snaps to a coordinate axis (within 20 deg); along
that axis a 5 cm histogram of their coordinate, smoothed over 3 bins; every peak with >= 3000 Gaussians is a plane.
Emitters = the transmitter + its mirror image in every plane. Known-answer check: the planes include the walls and
floor verified on 2026-09-25 (x 8.48, x -4.70, y -10.79, z -1.713) within 0.1 m.
Writes output/cluster/ladder/step3/prep/{emit_geo.npz, planes.json}.
"""

from __future__ import annotations

import json
import math
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ladder_step1 as L                                          # noqa: E402
sys.path.insert(0, os.path.join(L.REPO, "rrf_gsplat"))
from neural_shading import gaussian_normals                       # noqa: E402

OUT = os.path.join(L.REPO, "output", "cluster", "ladder", "step3", "prep")
MIN_COUNT, BIN, SNAP_DEG = 3000, 0.05, 20.0
VERIFIED = {"x": [8.48, -4.70], "y": [-10.79], "z": [-1.713]}


def main():
    os.makedirs(OUT, exist_ok=True)
    ck, _ = torch.load(L.CKPT, weights_only=False, map_location="cpu")
    xyz = ck[1].detach().double().numpy()
    op = torch.sigmoid(ck[6].detach().double()).reshape(-1).numpy()
    nrm = gaussian_normals(ck[5].detach().double(), ck[4].detach().double()).numpy()
    s = op >= 0.5
    X, N = xyz[s], nrm[s]
    a = np.abs(N); k = a.argmax(1)
    snapped = a[np.arange(len(a)), k] >= math.cos(math.radians(SNAP_DEG))
    planes = []
    for ax in range(3):
        c = X[snapped & (k == ax), ax]
        bins = np.arange(np.floor(c.min() / BIN) * BIN, c.max() + 2 * BIN, BIN)
        h, e = np.histogram(c, bins)
        hs = np.convolve(h, np.ones(3), "same")
        for i in range(1, len(hs) - 1):
            if hs[i] >= hs[i - 1] and hs[i] > hs[i + 1] and hs[i] >= MIN_COUNT:
                planes.append(("xyz"[ax], round(float(0.5 * (e[i] + e[i + 1])), 3), int(hs[i])))
    planes.sort(key=lambda p: -p[2])
    means, labels = [], []
    for ax, w, _ in planes:
        img = L.TX.copy(); j = "xyz".index(ax); img[j] = 2 * w - L.TX[j]
        means.append(img); labels.append(f"{ax}={w:.2f}")
    means.append(L.TX.copy()); labels.append("LOS")
    check = {}
    for ax, ws in VERIFIED.items():
        for w in ws:
            d = min((abs(p[1] - w) for p in planes if p[0] == ax), default=float("inf"))
            check[f"{ax}={w}"] = {"nearest_plane_distance_m": d, "ok": d <= 0.1}
    ok = all(v["ok"] for v in check.values())
    np.savez(os.path.join(OUT, "emit_geo.npz"), means=np.array(means), labels=np.array(labels))
    json.dump({"planes": [{"plane": f"{p[0]}={p[1]:.2f}", "count": p[2]} for p in planes], "emitters": labels,
               "known_answer_check": check, "known_answer_pass": ok}, open(os.path.join(OUT, "planes.json"), "w"), indent=1)
    print("planes:", [(f"{p[0]}={p[1]:.2f}", p[2]) for p in planes])
    print("known-answer check:", json.dumps(check), "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
