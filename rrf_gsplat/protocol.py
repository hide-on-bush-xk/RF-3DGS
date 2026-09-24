"""The train / validation / test protocol (protocol_v1, 2026-09-24): one set of split files shared by every dataset
generated on the same poses (MVDR, Bartlett, angular power ...), read by every script that trains or scores.

Sets (view names, four faces per position, a position never split):
  train          what every model may train on, and the NN / IDW baselines may look up
  val_random     half of the positions every earlier round was scored on (interpolation, ~0.23 m from training)
  val_segment    a contiguous stretch of the route's north side, behind a 0.5 m buffer (extrapolation)
  test_interp    positions never evaluated before, drawn at random from the old training set (sealed)
  test_region    the south-east corridor end x > 4, y < -7, behind a 0.5 m buffer, without the positions that were
                 ever evaluated before (sealed)
  dropped        the buffers and the ever-evaluated positions inside the region
"val" = val_random + val_segment; "test" = test_interp + test_region.

The test sets are sealed: eval_names() refuses them unless the caller passes allow_test=True (scripts: --final-test),
and every such read is appended to <protocol>/test_access.log with the time and the command line, so the log is the
record of how often the test set was looked at (the protocol says: once, in stage 4).
"""

from __future__ import annotations

import datetime
import hashlib
import json
import os
import sys

SETS = ("train", "val_random", "val_segment", "test_interp", "test_region", "dropped")
GROUPS = {"val": ("val_random", "val_segment"), "test": ("test_interp", "test_region")}
TEST_SETS = {"test", "test_interp", "test_region"}


def read_set(protocol, name):
    return [l.strip() for l in open(os.path.join(protocol, f"{name}.txt")) if l.strip()]


def eval_names(protocol, eval_set, allow_test=False):
    """The view names of an evaluation set ('val', 'val_random', 'val_segment', 'test', 'test_interp',
    'test_region'); the test ones only with allow_test, and logged."""
    if eval_set in TEST_SETS:
        if not allow_test:
            raise SystemExit(f"--eval-set {eval_set} is a sealed test set: pass --final-test to read it (stage 4 only)")
        with open(os.path.join(protocol, "test_access.log"), "a", encoding="utf-8") as f:
            f.write(f"{datetime.datetime.now().isoformat(timespec='seconds')}\t{eval_set}\t{' '.join(sys.argv)}\n")
    parts = GROUPS.get(eval_set, (eval_set,))
    out = []
    for p in parts:
        out += read_set(protocol, p)
    return sorted(out)


def train_names(protocol):
    return sorted(read_set(protocol, "train"))


def images_digest(dataset):
    """A digest of a dataset's poses -- every view's name, quaternion and receiver position rounded to 1e-6 -- to
    check it shares the protocol's poses. Not the file's bytes: a dataset generated with --poses-from recomputes
    t = -R rx and differs from its source in the last digit (1e-15)."""
    import numpy as np
    rows = []
    for line in open(os.path.join(dataset, "sparse", "0", "images.txt")):
        p = line.split()
        if len(p) < 10 or not p[9].lower().endswith(".png"):
            continue
        w, x, y, z = map(float, p[1:5]); t = np.array(list(map(float, p[5:8])))
        R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                      [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                      [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
        rx = -R.T @ t
        rows.append(f"{p[9]} " + " ".join(f"{v:+.6f}" for v in (w, x, y, z, *rx)).replace("-0.000000", "+0.000000"))
    return hashlib.sha256("\n".join(sorted(rows)).encode()).hexdigest()


def check_dataset(protocol, dataset):
    """Refuse a dataset whose poses are not the ones the protocol was built on."""
    meta = json.load(open(os.path.join(protocol, "meta.json")))
    got = images_digest(dataset)
    if got != meta["images_digest"]:
        raise SystemExit(f"{dataset}: its poses (images.txt digest {got[:12]}) are not the protocol's "
                         f"({meta['images_digest'][:12]}); generate it with --poses-from the protocol's reference")
