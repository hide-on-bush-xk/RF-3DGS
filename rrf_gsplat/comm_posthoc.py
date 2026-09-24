"""The live communication metrics of a finished run, from its saved held-out renders (for runs trained before
--live-comm-every existed, or with it off). Appends one {"comm": ..., "posthoc": true} line at the run's last
iteration to its live.jsonl and puts the references into its live/status.json, so the viewer's live tab shows the
point beside the look-ups. The same live_comm.py metrics on the same fixed subset (sionna_port/beam_maps.py).

    python rrf_gsplat/comm_posthoc.py --run output/rrf/m3/step1_A_em_pc8
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--maps", default=os.path.join(REPO, "output", "rrf", "beam_maps_val_subset"))
    a = ap.parse_args()
    import live_comm as LC
    from mvdr_peaks import pixel_dirs
    st_path = os.path.join(a.run, "live", "status.json")
    st = json.load(open(st_path))
    src = st["source"] if os.path.isabs(st["source"]) else os.path.join(REPO, st["source"])
    z = np.load(a.maps + ".npz")
    names, maps = [str(n) for n in z["names"]], z["maps"]
    missing = [n for n in names if not os.path.isfile(os.path.join(a.run, "renders", n + ".npy"))]
    if missing:
        raise SystemExit(f"{len(missing)} subset views have no saved render in {a.run} (e.g. {missing[:3]})")
    dirs = pixel_dirs(src)
    rows = [LC.view_metrics(np.load(os.path.join(a.run, "renders", n + ".npy")),
                            np.load(os.path.join(src, "spectra_float", n + ".npy")), maps[k], dirs)
            for k, n in enumerate(names)]
    summary = LC.summarise(rows)
    it = st.get("iteration") or st.get("iterations")
    with open(os.path.join(a.run, "live.jsonl"), "a") as f:
        f.write(json.dumps({"it": it, "t": round(time.time() - st.get("started", time.time()), 2), "comm": summary,
                            "posthoc": True}) + "\n")
    st["comm_refs"] = json.load(open(a.maps + ".json")).get("references")
    st["comm_views"] = len(names)
    tmp = st_path + ".tmp"
    json.dump(st, open(tmp, "w"))
    os.replace(tmp, st_path)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
