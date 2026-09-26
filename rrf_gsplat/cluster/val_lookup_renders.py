"""Look-up predictions on the 480 validation views, as run directories t4_score.py can score (renders/<view>.npy, dB):
lookup_nn (the same face of the nearest of the 467 training positions) and lookup_idw<k> (the k nearest, blended in
linear power with weights 1/d). docs/cluster_log.md §7.

    nice -n 10 ~/envs/rf-gsplat/bin/python rrf_gsplat/cluster/val_lookup_renders.py
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ladder_step1 as L                                          # noqa: E402

OUT = os.path.join(L.REPO, "output", "rrf", "val_falcon")


def main():
    names = L.PR.eval_names(os.path.join(L.REPO, "rrf_gsplat", "protocol_v1"), "val")
    pool = {}
    for g in L.GROUPS:
        for n in g:
            pool.setdefault(tuple(np.round(L.POSES[n][0][2], 2)), []).append(n)
    arr = {k: (v, np.array([L.POSES[m][2] for m in v])) for k, v in pool.items()}
    cache = {}
    lin = lambda m: cache.setdefault(m, 10 ** (L.truth(m) / 10))      # noqa: E731
    for tag, k in (("nn", 1), ("idw2", 2), ("idw8", 8)):
        d = os.path.join(OUT, f"lookup_{tag}", "renders")
        os.makedirs(d, exist_ok=True)
        for n in names:
            v, P = arr[tuple(np.round(L.POSES[n][0][2], 2))]
            dd = np.linalg.norm(P - L.POSES[n][2], axis=1)
            o = np.argsort(dd)[:k]
            if k == 1:
                pred = L.truth(v[o[0]])
            else:
                w = 1.0 / np.maximum(dd[o], 1e-6)
                pred = 10 * np.log10(sum(wi * lin(v[j]) for wi, j in zip(w, o)) / w.sum())
            np.save(os.path.join(d, f"{n}.npy"), pred.astype(np.float32))
        print(tag, len(names), "views ->", d, flush=True)


if __name__ == "__main__":
    main()
