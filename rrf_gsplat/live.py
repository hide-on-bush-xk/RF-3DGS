"""Live progress of a training run, for the viewer's live tab (viewer.py /api/live).

train_rrf.py writes into its --out directory:
  live/status.json   the run's identity, its planned iterations, started / finished, updated at every write
  live.jsonl         one line every --live-every steps: iteration, mean loss since the last line, it/s, elapsed
                     seconds; and one line per evaluation ({"eval": {...}})
  live/render.png    every --live-render-every steps: the prediction above the truth, jet over the run's range,
                     for one fixed training position and one fixed held-out position (all their faces)
Everything is written atomically (a temporary file, then os.replace), so the viewer never reads half a file.

Cost: the loss is accumulated on the GPU and read once per --live-every steps (one synchronisation per 50 steps
by default, none in between); a render is one extra forward pass of two positions.
"""

from __future__ import annotations

import json
import os
import time

import numpy as np
import torch


def _atomic_write(path, data: bytes):
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)


class LiveLog:
    def __init__(self, out, cfg, every=50, render_every=500):
        self.out, self.every, self.render_every = out, every, render_every
        self.dir = os.path.join(out, "live")
        os.makedirs(self.dir, exist_ok=True)
        self.path = os.path.join(out, "live.jsonl")
        open(self.path, "w").close()
        self.t0 = time.time()
        self.t_last, self.it_last = self.t0, 0
        self.acc, self.n_acc = None, 0
        self.status = {"run": os.path.basename(os.path.abspath(out)), "out": out, "iterations": cfg.iterations,
                       "mode": cfg.mode, "source": cfg.source, "seed": cfg.seed,
                       "extras": {k: getattr(cfg, k) for k in ("emitters", "em_pcolor", "pcolor", "max_train_views",
                                                                "train_names_file", "protocol", "eval_set")
                                  if getattr(cfg, k, None)},
                       "started": self.t0, "updated": self.t0, "done": False, "iteration": 0}
        self._write_status()

    def _write_status(self):
        self.status["updated"] = time.time()
        _atomic_write(os.path.join(self.dir, "status.json"), json.dumps(self.status).encode())

    def _append(self, rec):
        with open(self.path, "a") as f:
            f.write(json.dumps(rec) + "\n")

    def step(self, it, loss):
        """Call every step with the (GPU) loss tensor; writes a line every `every` steps."""
        if not self.every:
            return
        l = loss.detach()
        self.acc = l if self.acc is None else self.acc + l
        self.n_acc += 1
        if it % self.every == 0:
            now = time.time()
            rec = {"it": it, "loss": float(self.acc) / self.n_acc, "t": round(now - self.t0, 2),
                   "it_s": (it - self.it_last) / max(now - self.t_last, 1e-9)}
            self._append(rec)
            self.acc, self.n_acc = None, 0
            self.t_last, self.it_last = now, it
            self.status["iteration"] = it
            self._write_status()

    def eval(self, it, metrics):
        if not self.every:
            return
        keep = {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float)) and np.isfinite(v)}
        self._append({"it": it, "t": round(time.time() - self.t0, 2), "eval": keep})

    def want_render(self, it, last):
        return bool(self.every and self.render_every) and (it % self.render_every == 0 or last)

    @torch.no_grad()
    def render(self, it, rows):
        """rows: list of (label, pred [B, H, W] in [0, 1], truth [B, H, W] in [0, 1]) -> live/render.png,
        each row the B faces side by side, the prediction above its truth."""
        from matplotlib import colormaps
        from PIL import Image
        jet = colormaps["jet"]
        strips = []
        for _label, pred, truth in rows:
            for x in (pred, truth):
                a = x.clamp(0, 1).float().cpu().numpy()
                strip = np.concatenate(list(a), axis=1)                           # faces side by side
                strips.append((jet(strip)[..., :3] * 255).round().astype(np.uint8))
                strips.append(np.full((3, strips[-1].shape[1], 3), 255, np.uint8))  # a thin gap
        img = np.concatenate(strips[:-1], axis=0)
        from io import BytesIO
        buf = BytesIO()
        Image.fromarray(img).save(buf, format="PNG")
        _atomic_write(os.path.join(self.dir, "render.png"), buf.getvalue())
        self.status["render_it"] = it
        self.status["render_rows"] = [r[0] for r in rows]
        self._write_status()

    def close(self, results_path=None):
        self.status.update(done=True, finished=time.time(), results=results_path)
        self._write_status()
