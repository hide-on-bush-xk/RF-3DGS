"""End-to-end check of the planner's "Retrain RRF here": start interactive.py, POST /api/retrain the way the page
does (160 positions, 2000 views, db), poll /api/status until the job ends, and record every stage's time.

This is the click-to-render number the page shows, measured without a browser. Use a transmitter position that
has no live_tx_* run yet: the job writes RF-3DGS_dataset/regenerated/live_tx_<pos> and output/rrf/live_tx_<pos>.

    set PYTHONUTF8=1
    python tx_planning/check_interactive_retrain.py --tx 3.0 -5.0 2.0 --faces 4
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def call(url, body=None):
    req = urllib.request.Request(url, data=None if body is None else json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"}, method="GET" if body is None else "POST")
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tx", type=float, nargs=3, required=True)
    ap.add_argument("--faces", type=int, default=4)
    ap.add_argument("--port", type=int, default=8766)
    ap.add_argument("--out", default=os.path.join(REPO, "output", "tx_planning", "interactive_retrain_check.json"))
    a = ap.parse_args()
    env = dict(os.environ, PYTHONUTF8="1", PYTHONUNBUFFERED="1")
    srv = subprocess.Popen([sys.executable, os.path.join(REPO, "tx_planning", "interactive.py"), "--port", str(a.port)],
                           cwd=os.path.join(REPO, "tx_planning"), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        t0 = time.time()
        for line in srv.stdout:                      # wait until the scene is loaded and the port is open
            if "serving on" in line:
                break
            if time.time() - t0 > 300:
                raise SystemExit("server did not come up in 300 s")
        # keep draining the server's output, or a full pipe would block it
        import threading
        threading.Thread(target=lambda: [None for _ in srv.stdout], daemon=True).start()
        base = f"http://127.0.0.1:{a.port}"
        t_click = time.time()
        print("POST /api/retrain:", call(base + "/api/retrain", {"tx": a.tx, "positions": 160, "iterations": 2000,
                                                                "mode": "db", "faces": a.faces}))
        while True:
            time.sleep(2)
            st = call(base + "/api/status")
            if st.get("state") in ("done", "error"):
                break
        wall = time.time() - t_click
        st.get("results", {}).pop("renders", None)            # base64 images: not needed in the record
        rec = {"tx": a.tx, "faces": a.faces, "state": st.get("state"), "click_to_done_s": wall,
               "dataset_seconds": st.get("dataset_seconds"), "train_stage_seconds": st.get("train_seconds"),
               "trainer_train_seconds": st.get("results", {}).get("train_seconds"), "log": st.get("log"),
               "final": st.get("results", {}).get("final"), "error": st.get("error")}
        json.dump(rec, open(a.out, "w"), indent=1)
        print(json.dumps({k: v for k, v in rec.items() if k != "final"}, indent=1))
        if rec["final"]:
            print(f"held-out PSNR(jet) {rec['final']['psnr_rgb']:.2f} dB, RMSE {rec['final']['rmse_db']:.2f} dB")
    finally:
        srv.terminate()
        try:
            srv.wait(timeout=20)
        except subprocess.TimeoutExpired:
            srv.kill()


if __name__ == "__main__":
    main()
