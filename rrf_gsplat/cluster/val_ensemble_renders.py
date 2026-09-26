"""§8 combinations of G1 and IDW-2 on the 480 validation views (docs/cluster_log.md §8), fixed equal weights:
ens_lin_G1_s<s>: 10 log10(0.5 P_G1 + 0.5 P_IDW2); ens_db_G1_s<s>: 0.5 (dB_G1 + dB_IDW2). Writes run directories
with renders/<view>.npy (dB) under output/rrf/val_falcon/ for mvdr_peaks.py and val_beam_score.py.

    nice -n 10 ~/envs/rf-gsplat/bin/python rrf_gsplat/cluster/val_ensemble_renders.py
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ladder_step1 as L                                          # noqa: E402

V = os.path.join(L.REPO, "output", "rrf", "val_falcon")


def main():
    names = L.PR.eval_names(os.path.join(L.REPO, "rrf_gsplat", "protocol_v1"), "val")
    for s in (0, 1, 2):
        for tag in ("lin", "db"):
            os.makedirs(os.path.join(V, f"ens_{tag}_G1_s{s}", "renders"), exist_ok=True)
        for n in names:
            g = np.load(os.path.join(V, f"G1_s{s}", "renders", f"{n}.npy")).astype(np.float64)
            i = np.load(os.path.join(V, "lookup_idw2", "renders", f"{n}.npy")).astype(np.float64)
            lin = 10 * np.log10(0.5 * 10 ** (g / 10) + 0.5 * 10 ** (i / 10))
            np.save(os.path.join(V, f"ens_lin_G1_s{s}", "renders", f"{n}.npy"), lin.astype(np.float32))
            np.save(os.path.join(V, f"ens_db_G1_s{s}", "renders", f"{n}.npy"), (0.5 * (g + i)).astype(np.float32))
        print("seed", s, "done", flush=True)


if __name__ == "__main__":
    main()
