"""Build protocol_v1: the train / validation / test split every later experiment uses (see protocol.py).

Agreed with Ke on 2026-09-24 (numbers from the MVDR datasets' 800 positions, 4 faces each):
  test_region   x > 4 and y < -7 (the south-east corridor end), minus the positions ever evaluated before
  val_segment   y > -1 and 1 < x < 3 (the north side), not adjacent to the region
  buffers       positions outside a region / segment within 0.5 m of it (xy) are dropped, so both extrapolation
                sets sit >= 1.4 m (median) from training
  val_random    half of the old held-out positions (outside region, segment and buffers), drawn at random
  (the other half of the old held-out positions goes back to training: never a leak, they were only evaluated)
  test_interp   10 % of the never-evaluated positions left, drawn at random
  train         the rest
  near-duplicates  positions closer than 5 cm (3D) never straddle training and an evaluation set: the training
                   member is dropped (and of two evaluation sets, the validation member)

    python rrf_gsplat/make_protocol.py --reference RF-3DGS_dataset/regenerated/3dgs_MVDR_100_gpct \
        --out RF-3DGS_dataset/regenerated/protocol_v1
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import protocol as PR  # noqa: E402


def positions(dataset):
    """[(position centre, [view names])] in dataset order: consecutive views sharing a centre."""
    out = []
    for line in open(os.path.join(dataset, "sparse", "0", "images.txt")):
        p = line.split()
        if len(p) < 10 or not p[9].lower().endswith(".png"):
            continue
        w, x, y, z = map(float, p[1:5]); t = np.array(list(map(float, p[5:8])))
        R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                      [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                      [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
        rx = -R.T @ t; name = p[9][:-4]
        if out and np.allclose(out[-1][0], rx, atol=1e-6):
            out[-1][1].append(name)
        else:
            out.append((rx, [name]))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reference", required=True, help="a dataset with the poses and the old train / test split")
    ap.add_argument("--out", required=True)
    ap.add_argument("--buffer", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if os.path.exists(os.path.join(a.out, "test_region.txt")):
        raise SystemExit(f"{a.out} exists: a protocol is written once (its test sets are sealed)")
    pos = positions(a.reference)
    P = np.array([p[0] for p in pos]); n = len(P)
    old_test = set(l.strip() for l in open(os.path.join(a.reference, "test_index.txt")) if l.strip())
    old = np.array([pos[i][1][0] in old_test for i in range(n)])
    xy = P[:, :2]
    D2 = np.sqrt(((xy[:, None] - xy[None]) ** 2).sum(-1)); D3 = np.sqrt(((P[:, None] - P[None]) ** 2).sum(-1))
    region = (P[:, 0] > 4) & (P[:, 1] < -7)
    segment = (P[:, 1] > -1) & (P[:, 0] > 1) & (P[:, 0] < 3)
    near = lambda m: (~m) & (D2[:, m].min(1) < a.buffer)                                  # noqa: E731
    buffer = near(region) | near(segment)
    label = np.array(["train"] * n, dtype=object)
    label[buffer] = "dropped"
    label[region & old] = "dropped"
    label[region & ~old] = "test_region"
    label[segment] = "val_segment"
    rng = np.random.default_rng(a.seed)
    pool = np.flatnonzero(label == "train")
    old_pool = [i for i in pool if old[i]]
    label[rng.permutation(old_pool)[: len(old_pool) // 2]] = "val_random"
    fresh = [i for i in np.flatnonzero(label == "train") if not old[i]]
    label[rng.permutation(fresh)[: int(round(0.10 * len(fresh)))]] = "test_interp"
    # near-duplicates never straddle training and evaluation
    moved = []
    for i, j in zip(*np.nonzero(np.triu(D3 < 0.05, 1))):
        a_, b_ = label[i], label[j]
        if a_ == b_ or "dropped" in (a_, b_):
            continue
        if "train" in (a_, b_):
            k = i if a_ == "train" else j
        else:                                                           # two evaluation sets: drop the validation one
            k = i if str(a_).startswith("val") else j
        moved.append((pos[k][1][0], label[k])); label[k] = "dropped"
    tr = label == "train"
    os.makedirs(a.out, exist_ok=True)
    stats = {}
    for s in PR.SETS:
        idx = np.flatnonzero(label == s)
        names = sorted(nm for i in idx for nm in pos[i][1])
        open(os.path.join(a.out, f"{s}.txt"), "w").write("\n".join(names) + "\n")
        d = D3[idx][:, tr].min(1) if s != "train" and len(idx) else None
        stats[s] = {"positions": int(len(idx)), "views": len(names),
                    "nearest_train_m_median": None if d is None else float(np.median(d)),
                    "nearest_train_m_min": None if d is None else float(d.min())}
    trd = np.sort(D3[tr][:, tr], 1)[:, 1]
    meta = {"reference": os.path.abspath(a.reference), "images_digest": PR.images_digest(a.reference),
            "region": "x > 4 and y < -7", "val_segment": "y > -1 and 1 < x < 3", "buffer_m_xy": a.buffer, "seed": a.seed,
            "near_duplicates_dropped": moved, "train_spacing_m_median": float(np.median(trd)), "sets": stats,
            "sealed": ["test_interp", "test_region"],
            "note": "val = val_random + val_segment; test = test_interp + test_region (read only with --final-test)"}
    json.dump(meta, open(os.path.join(a.out, "meta.json"), "w"), indent=1)
    for s, v in stats.items():
        print(f"{s:<12} {v['positions']:4d} positions / {v['views']:4d} views"
              + ("" if v["nearest_train_m_median"] is None else
                 f", nearest training position median {v['nearest_train_m_median']:.2f} m (min {v['nearest_train_m_min']:.2f})"))
    print(f"near-duplicates dropped: {moved}; training spacing median {meta['train_spacing_m_median']:.2f} m")


if __name__ == "__main__":
    main()
