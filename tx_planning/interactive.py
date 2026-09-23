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
    """The scene with the receiver grid in it, ready for any transmitter.

    Held in memory for the server's lifetime: loading the scene and placing the
    receivers is what costs seconds, so keeping them resident is what makes an
    interactive coverage query sub-second.
    """

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
            # The tutorial's per-surface materials instead of the uniform
            # scattering coefficient load_radio_scene applies.
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
        # Two solvers: the forward one for interactive queries, the reverse-mode
        # one (slower) only for the optimiser.
        self.solver = make_solver()
        self.solver_ad = make_solver(reverse_mode=True)
        self.lock = threading.RLock()      # optimize() calls coverage() while holding it
        self.job = {"state": "idle"}       # the retrain job's status, polled by /api/status

    def _power_db(self, solver, seed=42, samples=None):
        """Received power per receiver in dB, or None when nothing was reached."""
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
        """One forward solve; returns the per-receiver gains and the covered fraction.

        The lock serialises GPU work: two browser requests arriving together
        would otherwise both mutate the shared scene.
        """
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
                # Every step is returned to the browser, which animates the path.
                history.append({"step": step, "tx": pos.round(3).tolist(), "objective": float(np.asarray(obj).reshape(-1)[0]),
                                "coverage": float((gain > threshold).mean())})
                m = b1 * m + (1 - b1) * g; v = b2 * v + (1 - b2) * g * g
                upd = lr * (m / (1 - b1 ** step)) / (np.sqrt(v / (1 - b2 ** step)) + eps)
                upd[2] = 0.0                       # height stays where the user put it
                # Full step, slide along each wall, then halve twice.
                for trial in (upd, upd * [0, 1, 1], upd * [1, 0, 1], upd / 2, upd / 4):
                    cand = pos + trial
                    cand[0] = np.clip(cand[0], *LOBBY_X); cand[1] = np.clip(cand[1], *LOBBY_Y)
                    if np.any(trial != 0) and indoor_mask(self.scene, cand[None])[0] and clearance(self.scene, cand) >= margin:
                        pos = cand; break
                else:
                    break                          # boxed in; stop rather than stall
            # Final coverage at the full sample count, not the optimiser's 20k.
            final = self.coverage(pos, threshold) if history else None
        return {"history": history, "final": final}

    def retrain(self, tx, positions, iterations, mode, faces=4):
        """Background job: dataset for this transmitter, then the RRF fine-tune in WSL.

        Runs in its own thread and reports through self.job, which /api/status
        polls. Every stage is a subprocess, so nothing here touches the GPU
        directly and the Dr.Jit per-thread flag problem does not arise.

        `iterations` counts training VIEWS. With faces > 1 (the default, round 24/27) each step renders the
        four faces of one receiver position with one colour evaluation and one Adam step at twice the
        learning rate, so 2000 views are 500 steps: 14 s of training instead of 42 s at the same held-out
        PSNR / dB RMSE (16.90 / 7.60 against 16.77 / 7.66 dB, round 27). faces=1 is the earlier command.
        """
        # The transmitter position becomes the run name, with the characters a
        # filesystem and a shell would object to replaced.
        name = f"live_tx_{tx[0]:+.1f}_{tx[1]:+.1f}_{tx[2]:+.1f}".replace("+", "p").replace("-", "m").replace(".", "_")
        import sys
        reg = os.path.join(REPO, "RF-3DGS_dataset", "regenerated")
        py = sys.executable
        env = dict(os.environ, PYTHONUTF8="1")     # the scene XML has non-ASCII
        log = []
        try:
            self.job.update(state="generating", name=name, log=log, started=time.time())
            t0 = time.time()
            # Stage 1: ray-trace a fresh spectrum dataset for this transmitter.
            subprocess.run([py, "generate_dataset.py", "--scene-xml", self.scene_xml,
                            "--rx-loc-file", os.path.join(REPO, "sionna_tutorial/RF-3DGS_Sionna_simulation_tutorial/NIST_rx_loc.txt"),
                            "--out-dir", os.path.join(reg, name), "--spectrum", "MVDR", "--num-positions", str(positions),
                            "--tx", *[str(v) for v in tx], "--no-dashboard"],
                           cwd=os.path.join(REPO, "sionna_port"), env=env, check=True, capture_output=True)
            # Stage 2: renormalise to the global-percentile encoding the trainer expects.
            subprocess.run([py, os.path.join(REPO, "rrf_gsplat", "renormalize.py"), os.path.join(reg, name),
                            os.path.join(reg, name + "_gpct"), "--norm", "global-pct", "--pct", "1", "99.99"],
                           env=env, check=True, capture_output=True)
            log.append(f"dataset: {positions} positions in {time.time()-t0:.0f} s")
            self.job.update(state="training", dataset_seconds=time.time() - t0)
            t1 = time.time()
            # Stage 3: the trainer lives in WSL (gsplat does not build under VS 2026),
            # so it is invoked across the boundary with a Linux-side path.
            steps = max(1, iterations // faces)
            fast = (["--faces-per-step", str(faces), "--lr-scale", "2", "--sh-backend", "gsplat", "--eval-group"]
                    if faces > 1 else [])
            wsl = ["wsl.exe", "-d", "Ubuntu-22.04", "-u", "ke", "--", "bash",
                   "/mnt/c/Users/Ke/Documents/GitHub/RF-3DGS/rrf_gsplat/wsl_run.sh", name,
                   "--source", f"RF-3DGS_dataset/regenerated/{name}_gpct", "--mode", mode,
                   "--iterations", str(steps), "--eval-every", str(max(1, 250 // faces)), "--save-renders", "4", *fast]
            # MSYS_NO_PATHCONV stops Git Bash rewriting the /mnt/... path into a Windows one.
            subprocess.run(wsl, env=dict(env, MSYS_NO_PATHCONV="1"), check=True, capture_output=True)
            out = os.path.join(REPO, "output", "rrf", name)
            res = json.load(open(os.path.join(out, "results.json")))
            # First 8 renders, inlined as data URIs so the page needs no static
            # file route to display them.
            renders = []
            rdir = os.path.join(out, "renders")
            src = os.path.join(reg, name + "_gpct", "images")
            for f in sorted(os.listdir(rdir))[:8]:
                if f.endswith(".png"):
                    renders.append({"name": f, "pred": b64(os.path.join(rdir, f)), "truth": b64(os.path.join(src, f))})
            log.append(f"training: {steps} steps x {faces} face(s) = {steps * faces} views in {res['train_seconds']:.0f} s")
            self.job.update(state="done", results={"final": res["final"], "history": res["history"],
                                                   "train_seconds": res["train_seconds"], "renders": renders},
                            train_seconds=time.time() - t1, total_seconds=time.time() - t0)
        except subprocess.CalledProcessError as exc:
            # Only the tail of stderr: a Sionna traceback can be enormous, and the
            # last lines are where the actual error is.
            self.job.update(state="error", error=(exc.stderr or b"")[-2000:].decode(errors="replace"))
        except Exception as exc:
            self.job.update(state="error", error=f"{type(exc).__name__}: {exc}")


def b64(path):
    """A PNG as a data: URI, for embedding straight into the JSON response."""
    with open(path, "rb") as fid:
        return "data:image/png;base64," + base64.b64encode(fid.read()).decode()


def make_handler(planner, sweep):
    """Build the request handler class, closing over the planner and sweep data."""
    # Read once at start-up: editing the HTML requires restarting the server.
    page = open(os.path.join(HERE, "interactive.html"), encoding="utf-8").read()

    class H(BaseHTTPRequestHandler):
        def _json(self, obj, code=200):
            body = json.dumps(obj).encode()
            self.send_response(code); self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

        def do_GET(self):
            if self.path.startswith("/api/grid"):
                # Everything the page needs to draw the floor plan, sent once.
                self._json({"rx": planner.rx.round(3).tolist(), "lobby": [LOBBY_X, LOBBY_Y], "rx_height": RX_HEIGHT,
                            "sweep": sweep, "tx": [float(v) for v in np.asarray(planner.tx.position).reshape(-1)]})
            elif self.path.startswith("/api/status"):
                self._json(planner.job)
            else:
                # Any other path serves the single page.
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
                # One job at a time: 409 rather than queueing, since a second
                # retrain would contend for the same GPU.
                if planner.job.get("state") in ("generating", "training"):
                    self._json({"error": "a job is running"}, 409); return
                t = threading.Thread(target=planner.retrain, args=(req["tx"], int(req.get("positions", 160)),
                                                                     int(req.get("iterations", 2000)), req.get("mode", "db"),
                                                                     int(req.get("faces", 4))), daemon=True)
                t.start(); self._json({"started": True})
            else:
                self._json({"error": "unknown endpoint"}, 404)

        def log_message(self, fmt, *args):
            pass                    # silence the per-request access log

    return H


def main():
    """Load the scene, optionally read a precomputed sweep, and serve the page."""
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
        # Optional background layer: the brute-force sweep's coverage per
        # candidate, so the page can show where the good positions already are.
        d = np.load(cfg.sweep, allow_pickle=True)
        g = np.nan_to_num(d["gain_db"], nan=-999.0)     # NaN = unreached = uncovered
        sweep = {"tx": d["tx_positions"].round(3).tolist(), "coverage": (g > -85).mean(1).round(3).tolist()}
    print(f"{len(planner.rx)} indoor receivers; serving on http://localhost:{cfg.port}")
    # single-threaded on purpose: the solves serialise on the GPU anyway, and
    # Dr.Jit keeps its JIT flags per thread (the retrain job runs in its own
    # thread but only shells out)
    # Bound to 127.0.0.1, so the server is not reachable from the network.
    HTTPServer(("127.0.0.1", cfg.port), make_handler(planner, sweep)).serve_forever()


if __name__ == "__main__":
    main()
