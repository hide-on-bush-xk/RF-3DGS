"""A local page to place the transmitter by hand and see what follows.

The interaction Aerial's digital-twin demo offers -- put a radio unit on the
map, run, look at the coverage -- on this scene, with what this repository
adds: a batched Sionna solve gives the coverage of any transmitter position
in under a second, the gradient optimiser slides it to a better one, and one
click regenerates a 160-position dataset and fine-tunes the radiance field
(about a minute). No Omniverse: a Python HTTP server holding the scene in
memory and one HTML page.

    PYTHONUTF8=1 python tx_planning/interactive.py --scene-xml ... [--port 8765]
    then open http://localhost:8765
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np

from scene_common import (LOBBY_X, LOBBY_Y, RX_HEIGHT, clearance, enable_reverse_mode,
                          indoor_mask, load_radio_scene, make_solver, rx_grid)

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))


class Planner:
    """The scene with the receiver grid in it, ready for any transmitter."""

    def __init__(self, scene_xml, rx_step, samples, materials):
        import drjit as dr
        import mitsuba as mi
        mi.set_variant("cuda_ad_mono_polarized")
        enable_reverse_mode()
        from sionna.rt import Receiver, Transmitter
        self.dr, self.mi = dr, mi
        self.scene_xml, self.samples = scene_xml, samples
        self.scene = load_radio_scene(scene_xml)
        if materials == "tutorial":
            import sys
            sys.path.insert(0, os.path.join(REPO, "sionna_port"))
            import tutorial_materials
            tutorial_materials.apply(self.scene, 60e9, "fixed", verbose=False)
        rx = rx_grid(LOBBY_X, LOBBY_Y, RX_HEIGHT, rx_step)
        self.rx = rx[indoor_mask(self.scene, rx)]
        for i, p in enumerate(self.rx):
            self.scene.add(Receiver(name=f"rx{i}", position=[float(v) for v in p]))
        self.tx = Transmitter(name="tx", position=[6.905, 0.0, 0.287])
        self.scene.add(self.tx)
        self.solver = make_solver()
        self.solver_ad = make_solver(reverse_mode=True)
        self.lock = threading.RLock()      # optimize() calls coverage() while holding it
        self.job = {"state": "idle"}

    def _power_db(self, solver, seed=42, samples=None):
        dr = self.dr
        paths = solver(scene=self.scene, max_depth=1, max_num_paths_per_src=10_000_000, samples_per_src=samples or self.samples, los=True,
                       specular_reflection=True, diffuse_reflection=True, refraction=False,
                       synthetic_array=True, seed=seed)
        a_re, a_im = paths.a
        if a_re.shape[-1] == 0:
            return None
        p = dr.square(a_re) + dr.square(a_im)
        for axis in range(p.ndim - 1, 0, -1):
            p = dr.sum(p, axis=axis)
        return 10.0 * dr.log(p + 1e-13) / np.log(10.0)          # [num_rx], -130 dB floor

    def coverage(self, tx, threshold):
        with self.lock:
            t0 = time.time()
            self.tx.position = [float(v) for v in tx]
            p = self._power_db(self.solver)
            if p is None:
                return {"error": "no paths from there: inside an object or outside the building"}
            gain = np.asarray(p).reshape(-1).astype(float)
            return {"tx": [float(v) for v in tx], "gain_db": gain.round(2).tolist(),
                    "coverage": float((gain > threshold).mean()), "seconds": time.time() - t0}

    def optimize(self, tx, threshold, steps, lr=0.1, width=5.0, margin=0.2):
        """optimize_tx.py's loop: soft coverage, Sionna's gradient, wall constraints."""
        dr = self.dr
        enable_reverse_mode()      # Dr.Jit flags are per thread; this may not be the loading thread
        with self.lock:
            pos = np.array(tx, dtype=float)
            m = np.zeros(3); v = np.zeros(3); b1, b2, eps = 0.9, 0.999, 1e-8
            history = []
            for step in range(1, steps + 1):
                self.tx.position = [float(x) for x in pos]
                p = self.tx.position
                dr.enable_grad(p)
                # the reverse-mode solve is several times the forward one; the
                # objective's Monte-Carlo noise is +-0.001 at 20k samples
                pdb = self._power_db(self.solver_ad, samples=20_000)
                if pdb is None:
                    break
                obj = dr.mean(1.0 / (1.0 + dr.exp(-(pdb - threshold) / width)))
                dr.backward(obj)
                g = np.asarray(dr.grad(p)).reshape(-1).astype(float)
                gain = np.asarray(pdb).reshape(-1).astype(float)
                history.append({"step": step, "tx": pos.round(3).tolist(), "objective": float(np.asarray(obj).reshape(-1)[0]),
                                "coverage": float((gain > threshold).mean())})
                m = b1 * m + (1 - b1) * g; v = b2 * v + (1 - b2) * g * g
                upd = lr * (m / (1 - b1 ** step)) / (np.sqrt(v / (1 - b2 ** step)) + eps)
                upd[2] = 0.0
                for trial in (upd, upd * [0, 1, 1], upd * [1, 0, 1], upd / 2, upd / 4):
                    cand = pos + trial
                    cand[0] = np.clip(cand[0], *LOBBY_X); cand[1] = np.clip(cand[1], *LOBBY_Y)
                    if np.any(trial != 0) and indoor_mask(self.scene, cand[None])[0] and clearance(self.scene, cand) >= margin:
                        pos = cand; break
                else:
                    break
            final = self.coverage(pos, threshold) if history else None
        return {"history": history, "final": final}

    def retrain(self, tx, positions, iterations, mode):
        """Background job: dataset for this transmitter, then the RRF fine-tune in WSL."""
        name = f"live_tx_{tx[0]:+.1f}_{tx[1]:+.1f}_{tx[2]:+.1f}".replace("+", "p").replace("-", "m").replace(".", "_")
        import sys
        reg = os.path.join(REPO, "RF-3DGS_dataset", "regenerated")
        py = sys.executable
        env = dict(os.environ, PYTHONUTF8="1")
        log = []
        try:
            self.job.update(state="generating", name=name, log=log, started=time.time())
            t0 = time.time()
            subprocess.run([py, "generate_dataset.py", "--scene-xml", self.scene_xml,
                            "--rx-loc-file", os.path.join(REPO, "sionna_tutorial/RF-3DGS_Sionna_simulation_tutorial/NIST_rx_loc.txt"),
                            "--out-dir", os.path.join(reg, name), "--spectrum", "MVDR", "--num-positions", str(positions),
                            "--tx", *[str(v) for v in tx], "--no-dashboard"],
                           cwd=os.path.join(REPO, "sionna_port"), env=env, check=True, capture_output=True)
            subprocess.run([py, os.path.join(REPO, "rrf_gsplat", "renormalize.py"), os.path.join(reg, name),
                            os.path.join(reg, name + "_gpct"), "--norm", "global-pct", "--pct", "1", "99.99"],
                           env=env, check=True, capture_output=True)
            log.append(f"dataset: {positions} positions in {time.time()-t0:.0f} s")
            self.job.update(state="training", dataset_seconds=time.time() - t0)
            t1 = time.time()
            wsl = ["wsl.exe", "-d", "Ubuntu-22.04", "-u", "ke", "--", "bash",
                   "/mnt/c/Users/Ke/Documents/GitHub/RF-3DGS/rrf_gsplat/wsl_run.sh", name,
                   "--source", f"RF-3DGS_dataset/regenerated/{name}_gpct", "--mode", mode,
                   "--iterations", str(iterations), "--eval-every", "250", "--save-renders", "4"]
            subprocess.run(wsl, env=dict(env, MSYS_NO_PATHCONV="1"), check=True, capture_output=True)
            out = os.path.join(REPO, "output", "rrf", name)
            res = json.load(open(os.path.join(out, "results.json")))
            renders = []
            rdir = os.path.join(out, "renders")
            src = os.path.join(reg, name + "_gpct", "images")
            for f in sorted(os.listdir(rdir))[:8]:
                if f.endswith(".png"):
                    renders.append({"name": f, "pred": b64(os.path.join(rdir, f)), "truth": b64(os.path.join(src, f))})
            log.append(f"training: {iterations} iterations in {res['train_seconds']:.0f} s")
            self.job.update(state="done", results={"final": res["final"], "history": res["history"],
                                                   "train_seconds": res["train_seconds"], "renders": renders},
                            train_seconds=time.time() - t1, total_seconds=time.time() - t0)
        except subprocess.CalledProcessError as exc:
            self.job.update(state="error", error=(exc.stderr or b"")[-2000:].decode(errors="replace"))
        except Exception as exc:
            self.job.update(state="error", error=f"{type(exc).__name__}: {exc}")


def b64(path):
    with open(path, "rb") as fid:
        return "data:image/png;base64," + base64.b64encode(fid.read()).decode()


def make_handler(planner, sweep):
    page = open(os.path.join(HERE, "interactive.html"), encoding="utf-8").read()

    class H(BaseHTTPRequestHandler):
        def _json(self, obj, code=200):
            body = json.dumps(obj).encode()
            self.send_response(code); self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

        def do_GET(self):
            if self.path.startswith("/api/grid"):
                self._json({"rx": planner.rx.round(3).tolist(), "lobby": [LOBBY_X, LOBBY_Y], "rx_height": RX_HEIGHT,
                            "sweep": sweep, "tx": [float(v) for v in np.asarray(planner.tx.position).reshape(-1)]})
            elif self.path.startswith("/api/status"):
                self._json(planner.job)
            else:
                body = page.encode()
                self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n) or b"{}")
            if self.path == "/api/coverage":
                self._json(planner.coverage(req["tx"], float(req.get("threshold", -85))))
            elif self.path == "/api/optimize":
                self._json(planner.optimize(req["tx"], float(req.get("threshold", -85)), int(req.get("steps", 15))))
            elif self.path == "/api/retrain":
                if planner.job.get("state") in ("generating", "training"):
                    self._json({"error": "a job is running"}, 409); return
                t = threading.Thread(target=planner.retrain, args=(req["tx"], int(req.get("positions", 160)),
                                                                     int(req.get("iterations", 2000)), req.get("mode", "db")), daemon=True)
                t.start(); self._json({"started": True})
            else:
                self._json({"error": "unknown endpoint"}, 404)

        def log_message(self, fmt, *args):
            pass
    return H


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene-xml", default=os.path.join(REPO, "sionna_tutorial/RF-3DGS_Sionna_simulation_tutorial/NIST_lobby_v1.0/NIST_lobby_V1.1_sionna12.xml"))
    ap.add_argument("--rx-step", type=float, default=1.0)
    ap.add_argument("--samples", type=int, default=50_000)
    ap.add_argument("--materials", choices=["uniform", "tutorial"], default="uniform")
    ap.add_argument("--sweep", default=os.path.join(REPO, "output/tx_planning/tx_sweep.npz"),
                    help="a tx_sweep.npz to draw the candidates from, if present")
    ap.add_argument("--port", type=int, default=8765)
    cfg = ap.parse_args()

    planner = Planner(cfg.scene_xml, cfg.rx_step, cfg.samples, cfg.materials)
    sweep = None
    if os.path.exists(cfg.sweep):
        d = np.load(cfg.sweep, allow_pickle=True)
        g = np.nan_to_num(d["gain_db"], nan=-999.0)
        sweep = {"tx": d["tx_positions"].round(3).tolist(), "coverage": (g > -85).mean(1).round(3).tolist()}
    print(f"{len(planner.rx)} indoor receivers; serving on http://localhost:{cfg.port}")
    # single-threaded on purpose: the solves serialise on the GPU anyway, and
    # Dr.Jit keeps its JIT flags per thread (the retrain job runs in its own
    # thread but only shells out)
    HTTPServer(("127.0.0.1", cfg.port), make_handler(planner, sweep)).serve_forever()


if __name__ == "__main__":
    main()
