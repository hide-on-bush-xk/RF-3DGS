"""Experiment A1: can a learned upsampler turn cheap low-resolution LABELS into the high-resolution ones?

The same receiver poses were generated twice, at 150x100 and at 300x200, both computed exactly from the paths.
The network is trained on the training positions' pairs and writes its upsampled labels for the held-out
positions, which are then scored against the true 300x200 labels by the same programs as every model:
mvdr_peaks.py (beamformed, with the false-peak rate) or eval_baselines.py (multi-channel, decoded channels).

Every feature is a switch, so each one's gain can be read off an ablation ladder:
  --arch bilinear     no learning: bilinear upsampling (pixel centres aligned)
  --arch cnn          + a residual CNN on top of the bilinear image (zero-initialised: starts as bilinear)
  --coords            + the pixel's position in the face as two input channels
  --peak-loss W       + extra L1 weight W on the pixels within 10 dB of the view's true maximum (db mode)

    /home/ke/miniconda3/envs/rf-gsplat/bin/python rrf_gsplat/label_sr.py --lo <150x100 set> --hi <300x200 set> \
        --mode db --arch cnn --coords --out output/rrf/r40_A1_mvdr_cnn_coords
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from neural_shading import ResidualCNN, pixel_coords  # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def read_index(p):
    return [l.strip() for l in open(p) if l.strip()]


def load(ds, names):
    with ThreadPoolExecutor(16) as ex:
        arrs = list(ex.map(lambda n: np.load(os.path.join(ds, "spectra_float", n + ".npy")).astype(np.float32), names))
    x = torch.from_numpy(np.stack(arrs))
    return x[:, None] if x.dim() == 3 else x                  # [N, C, h, w]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lo", required=True); ap.add_argument("--hi", required=True)
    ap.add_argument("--mode", choices=["db", "multi"], required=True)
    ap.add_argument("--arch", choices=["bilinear", "cnn"], default="cnn")
    ap.add_argument("--coords", action="store_true")
    ap.add_argument("--peak-loss", type=float, default=0.0)
    ap.add_argument("--steps", type=int, default=3000); ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    dev = torch.device("cuda")
    lo_ds, hi_ds = os.path.join(REPO, a.lo), os.path.join(REPO, a.hi)
    meta = json.load(open(os.path.join(hi_ds, "generation_meta.json")))
    train = read_index(os.path.join(hi_ds, "train_index.txt")); test = read_index(os.path.join(hi_ds, "test_index.txt"))
    assert read_index(os.path.join(lo_ds, "test_index.txt")) == test, "the two resolutions must share the split"
    # value ranges of the high-resolution set: dB range (db) or per-channel ranges (multi). They scale the
    # network's input and its residual only. The upsampled base stays in native units, unclamped: clamping at
    # the dataset's 99.99th percentile flattened the strongest views' main peaks into plateaus, and float16
    # moved values near -100 dB by 0.06 dB -- together they made the bilinear arm disagree with round 27's
    # bilinear labels (first smoke of this script; fixed). The loss clamps the floor only, so the -300 dB of
    # an empty view does not dominate it.
    if a.mode == "db":
        rng_t = torch.tensor([[meta["spec_min_db"], meta["spec_max_db"]]], dtype=torch.float32)
    else:
        rng_t = torch.tensor(meta["channel_ranges"], dtype=torch.float32)
    lo_c, hi_c = rng_t[:, 0].to(dev)[None, :, None, None], rng_t[:, 1].to(dev)[None, :, None, None]
    scale = hi_c - lo_c
    norm_in = lambda x: ((x - lo_c) / scale).clamp(0, 1)                      # network input only  # noqa: E731
    floor = lambda x: (x.clamp_min(lo_c) - lo_c) / scale                     # loss units         # noqa: E731

    t0 = time.time()
    X_tr, Y_tr = load(lo_ds, train), load(hi_ds, train)
    X_te = load(lo_ds, test)
    H, W = Y_tr.shape[-2:]
    C = Y_tr.shape[1]
    X_tr, Y_tr, X_te = X_tr.to(dev), Y_tr.to(dev), X_te.to(dev)               # float32 throughout
    print(f"loaded {len(train)} train / {len(test)} test pairs {tuple(X_tr.shape[-2:])} -> {(H, W)}, {C} channel(s), "
          f"{time.time() - t0:.0f} s")

    up = lambda x: F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False)   # noqa: E731
    coords = pixel_coords(H, W, dev)[None]
    net = None
    if a.arch == "cnn":
        net = ResidualCNN(C + (2 if a.coords else 0), C, width=a.width).to(dev)
        opt = torch.optim.Adam(net.parameters(), lr=a.lr)

    def predict(x):                                   # x: [B, C, h, w] native units -> native [B, C, H, W]
        b = up(x)
        if net is None:
            return b
        inp = norm_in(b)
        if a.coords:
            inp = torch.cat([inp, coords.expand(b.shape[0], -1, -1, -1)], 1)
        return b + scale * net(inp)

    history, t_train = [], time.time()
    if net is not None:
        for step in range(1, a.steps + 1):
            idx = torch.from_numpy(rng.integers(0, X_tr.shape[0], a.batch)).to(dev)
            y = floor(Y_tr[idx]); p = floor(predict(X_tr[idx]))
            if a.mode == "multi":
                # angles and delay mean nothing where no path arrives: masked like train_rrf's multi loss
                mask = (y[:, :1] > 0.02).float()
                loss = (p[:, :1] - y[:, :1]).abs().mean() + ((p[:, 1:] - y[:, 1:]).abs() * mask).sum() / (mask.sum() * (C - 1) + 1)
            else:
                loss = (p - y).abs().mean()
            if a.peak_loss > 0:
                # the pixels within 10 dB of each view's true maximum (normalised units), channel 0
                top = y[:, :1] >= (y[:, :1].amax(dim=(2, 3), keepdim=True) - 10.0 / float(hi_c[0, 0] - lo_c[0, 0]))
                loss = loss + a.peak_loss * ((p[:, :1] - y[:, :1]).abs() * top).sum() / top.sum().clamp_min(1)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            if step % 500 == 0 or step == a.steps:
                history.append({"step": step, "loss": float(loss)}); print(f"  step {step:5d} loss {float(loss):.5f}")
    torch.cuda.synchronize(); train_s = time.time() - t_train

    os.makedirs(os.path.join(a.out, "renders"), exist_ok=True)
    t_inf = time.time()
    with torch.no_grad():
        for k in range(0, len(test), 16):
            y = predict(X_te[k:k + 16]).cpu().numpy()
            for j, n in enumerate(test[k:k + 16]):
                np.save(os.path.join(a.out, "renders", n + ".npy"), y[j, 0] if C == 1 else y[j])
    torch.cuda.synchronize()
    json.dump({"config": vars(a), "train_seconds": train_s, "infer_seconds": time.time() - t_inf,
               "params": sum(p.numel() for p in net.parameters()) if net is not None else 0, "history": history,
               "gpu": torch.cuda.get_device_name(0)}, open(os.path.join(a.out, "results.json"), "w"), indent=1)
    print(f"wrote {len(test)} upsampled held-out labels to {a.out}/renders ({train_s:.0f} s training)")


if __name__ == "__main__":
    main()
