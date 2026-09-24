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
import threading
import time

import numpy as np
import torch


def _atomic_write(path, data: bytes):
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)


def _ring(img, y, x, r=7):
    """A white ring with a dark outline around (y, x): the true main peak."""
    yy, xx = np.ogrid[:img.shape[0], :img.shape[1]]
    d = np.sqrt((yy - y) ** 2 + (xx - x) ** 2)
    img[(d >= r - 2.2) & (d < r + 1.2)] = (20, 20, 20)
    img[(d >= r - 1.2) & (d < r + 0.2)] = (255, 255, 255)


def _cross(img, y, x, r=5):
    """A dark X with a white edge at (y, x): the predicted main peak."""
    for k in range(-r, r + 1):
        for dy, dx in ((k, k), (k, -k)):
            for oy, ox, c in ((0, 1, 255), (1, 0, 255), (0, 0, 20)):
                yy, xx = y + dy + oy, x + dx + ox
                if 0 <= yy < img.shape[0] and 0 <= xx < img.shape[1]:
                    img[yy, xx] = (c, c, c)


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
        self._lock = threading.Lock()                # the communication metrics are written from a worker thread
        self.status = {"run": os.path.basename(os.path.abspath(out)), "out": out, "iterations": cfg.iterations,
                       "mode": cfg.mode, "source": cfg.source, "seed": cfg.seed,
                       "extras": {k: getattr(cfg, k) for k in ("emitters", "em_pcolor", "pcolor", "max_train_views",
                                                                "train_names_file", "protocol", "eval_set")
                                  if getattr(cfg, k, None)},
                       "started": self.t0, "updated": self.t0, "done": False, "iteration": 0}
        self._write_status()

    def _write_status(self):
        with self._lock:
            self.status["updated"] = time.time()
            _atomic_write(os.path.join(self.dir, "status.json"), json.dumps(self.status).encode())

    def _append(self, rec):
        with self._lock, open(self.path, "a") as f:
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

    def comm(self, it, summary):
        """The communication metrics of the fixed validation subset (live_comm.py) at this step."""
        if self.every and summary is not None:
            self._append({"it": it, "t": round(time.time() - self.t0, 2), "comm": summary})

    def want_render(self, it, last):
        return bool(self.every and self.render_every) and (it % self.render_every == 0 or last)

    @torch.no_grad()
    def render(self, it, rows):
        """rows: list of (label, pred [B, H, W] in [0, 1], truth [B, H, W] in [0, 1]) -> live/render.png,
        each row the B faces side by side, the prediction above its truth. On every face of both rows the true
        main peak is ringed and the predicted main peak crossed: where a beam would be steered, and where it
        should be."""
        from matplotlib import colormaps
        from PIL import Image
        jet = colormaps["jet"]
        strips = []
        for _label, pred, truth in rows:
            pa, ta = pred.float().cpu().numpy(), truth.float().cpu().numpy()
            w = pa.shape[2]
            peaks = [(np.unravel_index(int(ta[f].argmax()), ta[f].shape), np.unravel_index(int(pa[f].argmax()), pa[f].shape))
                     for f in range(len(pa))]
            for x in (pa, ta):
                strip = np.concatenate(list(np.clip(x, 0, 1)), axis=1)             # faces side by side
                rgb = (jet(strip)[..., :3] * 255).round().astype(np.uint8)
                for f, ((ty, tx), (py, px)) in enumerate(peaks):
                    _ring(rgb, ty, tx + f * w)
                    _cross(rgb, py, px + f * w)
                strips.append(rgb)
                strips.append(np.full((3, rgb.shape[1], 3), 255, np.uint8))        # a thin gap
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
