"""check_dir_loss.py (the direction loss's unit checks, stage2_notes §17) on Falcon, unchanged.

That script reads the dataset's dB range from a PC run (output/rrf/m3/rr/plain_s0/results.json, "db_range"), which is
not in the data package. Without --db-range, train_rrf.py takes that range from the dataset's generation_meta.json
(spec_min_db, spec_max_db), so the PC run's value is the dataset's; here exactly that one read is answered from
generation_meta.json, and everything else runs as written.

    ~/envs/rf-gsplat/bin/python rrf_gsplat/cluster/check_dir_loss_falcon.py
"""

from __future__ import annotations

import builtins
import io
import json
import os
import runpy

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PC_RUN = os.path.join(REPO, "output", "rrf", "m3", "rr", "plain_s0", "results.json")
META = os.path.join(REPO, "RF-3DGS_dataset", "regenerated", "3dgs_APS_60_gp100", "generation_meta.json")

meta = json.load(open(META))
db_range = [meta["spec_min_db"], meta["spec_max_db"]]
print(f"db_range from generation_meta.json: {db_range}")
_open = builtins.open


def _open_redirect(file, *args, **kwargs):
    if isinstance(file, str) and os.path.abspath(file) == PC_RUN and not os.path.exists(PC_RUN):
        return io.StringIO(json.dumps({"db_range": db_range}))
    return _open(file, *args, **kwargs)


builtins.open = _open_redirect
runpy.run_path(os.path.join(REPO, "rrf_gsplat", "check_dir_loss.py"), run_name="__main__")
