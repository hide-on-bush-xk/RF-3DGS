"""NeRF2 (Zhao et al., MobiCom 2023) on the RF-3DGS released benchmark: a NeRF-family baseline on our coordinates.

Uses NeRF2's own network and spectrum renderer (third_party/NeRF2, MIT) unchanged; what is adapted is the data:
their spectrum is a 90 x 360 hemisphere seen from one fixed gateway with the transmitter moving, RF-3DGS's is a
300 x 200 pinhole face seen from a moving receiver with the transmitter fixed. Rays are therefore the pinhole pixel
rays of each receiver pose (origin = receiver position, direction = K^-1 [u v 1] rotated to the world), the
transmitter input is the fixed transmitter (a constant, so its encoding is a constant), and the per-ray target is
the jet-inverted value of the released PNG in [0, 1] (the scalar spectra only: CBF, TCBF, MVDR, MPC; AoD and Delay
are three-channel encodings NeRF2's scalar head cannot represent). Hyperparameters follow configs/rfid-spectrum.yml
(D 8, W 256, multires 10, 64 samples per ray, lr 8e-4, weight decay 5e-5, batch 8192 rays, cosine schedule) except
near / far, which are set to the room (0.05 / 20 m) instead of their 0 / 5 m. Predictions are written as jet PNGs
in the run's renders/ so inria_metrics.py scores them with the fork's metrics.py exactly like every other row.

Leaving their model untouched and adapting only the data is what makes this a
fair baseline: any difference is the method, not a reimplementation of it. The
one deliberate change, near/far, is stated above because their 5 m far plane
would not reach the far wall of this room.

    python rrf_gsplat/nerf2_pinhole.py --source RF-3DGS_dataset/training-rf-spectrum/3dgs_MVDR_100 --out output/rrf/nerf2_MVDR --iterations 30000
    python rrf_gsplat/nerf2_pinhole.py ... --iterations 300 --overfit-one --eval-views 4      (smoke: one view must be reproduced)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(REPO, "third_party", "NeRF2"))


def main():
    """Train NeRF2 on pinhole rays and write jet PNGs the shared scorer can read."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True); ap.add_argument("--out", required=True)
    # These defaults are NeRF2's own configs/rfid-spectrum.yml, kept verbatim.
    ap.add_argument("--iterations", type=int, default=30000); ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--lr", type=float, default=8e-4); ap.add_argument("--weight-decay", type=float, default=5e-5)
    ap.add_argument("--t-max", type=int, default=10000); ap.add_argument("--eta-min", type=float, default=1e-6)
    # near/far are the one deliberate departure: the room is larger than 5 m.
    ap.add_argument("--n-samples", type=int, default=64); ap.add_argument("--near", type=float, default=0.05); ap.add_argument("--far", type=float, default=20.0)
    ap.add_argument("--tx", type=float, nargs=3, default=[6.9, 0.0, 0.29], help="the fixed transmitter (a constant input)")
    ap.add_argument("--eval-views", type=int, default=-1, help="-1 = every held-out view"); ap.add_argument("--eval-every", type=int, default=5000)
    ap.add_argument("--overfit-one", action="store_true", help="smoke: train on one view's rays and evaluate that view")
    ap.add_argument("--chunk", type=int, default=4096, help="rays per evaluation forward pass")
    ap.add_argument("--seed", type=int, default=0)
    cfg = ap.parse_args()
    torch.manual_seed(cfg.seed); rng = np.random.default_rng(cfg.seed)
    from model import NeRF2
    from renderer import Renderer_spectrum
    from train_rrf import read_colmap_text, read_index, psnr
    from jet import jet_inverse, jet_rgb
    from PIL import Image
    device = torch.device("cuda")
    t0 = time.time()

    views = read_colmap_text(os.path.join(cfg.source, "sparse", "0"))
    train_names = read_index(os.path.join(cfg.source, "train_index.txt")); test_names = read_index(os.path.join(cfg.source, "test_index.txt"))
    if cfg.overfit_one:
        # The smoke control: train and test on the same single view. If the
        # model cannot reproduce that, nothing downstream is worth running.
        train_names, test_names = train_names[:1], train_names[:1]
    if cfg.eval_views > 0:
        test_names = test_names[:: max(1, len(test_names) // cfg.eval_views)][: cfg.eval_views]
    _, K, W, H = views[train_names[0] + ".png"]
    # One ray direction per pixel in the camera frame, shared by every view;
    # only the rotation and origin differ per pose.
    u, v = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    d_cam = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u)], -1).reshape(-1, 3)
    d_cam = torch.as_tensor(d_cam / np.linalg.norm(d_cam, axis=-1, keepdims=True), dtype=torch.float32, device=device)   # [HW, 3]

    def load(names, tag):
        """Poses and jet-inverted targets for a name list, with a disk cache."""
        R, o = [], []
        for n in names:
            view = views[n + ".png"][0].astype(np.float64); c2w = np.linalg.inv(view)
            R.append(c2w[:3, :3]); o.append(c2w[:3, 3])
        # the jet inverse of 3200 pictures is a nearest-LUT search (about 10 min on the GPU); cache it next to the dataset
        # half precision: the targets are in [0, 1] and 3200 x 60000 floats
        # would otherwise be several GB of VRAM.
        cache = os.path.join(cfg.source, f"jet_inverse_{tag}_{len(names)}.npy")
        if os.path.exists(cache):
            lab = torch.from_numpy(np.load(cache)).to(device)
        else:
            lab = torch.stack([jet_inverse(torch.from_numpy(np.array(Image.open(os.path.join(cfg.source, "images", n + ".png")).convert("RGB")))
                                           .permute(2, 0, 1).float().div(255.0).to(device)).reshape(-1).half() for n in names])
            np.save(cache, lab.cpu().numpy())
        return (torch.as_tensor(np.stack(R), dtype=torch.float32, device=device), torch.as_tensor(np.stack(o), dtype=torch.float32, device=device), lab)

    R_tr, o_tr, y_tr = load(train_names, "train"); R_te, o_te, y_te = load(test_names, "test")
    n_tr, n_pix = len(train_names), d_cam.shape[0]
    tx = torch.tensor(cfg.tx, dtype=torch.float32, device=device)
    print(f"{n_tr} training views x {n_pix} rays, {len(test_names)} held-out views; near {cfg.near} far {cfg.far} m, {cfg.n_samples} samples; loaded in {time.time() - t0:.0f} s")

    # NeRF2's network and renderer, constructed exactly as their config does.
    # The tx input is embedded like the others even though it never varies here;
    # changing that would be a modification of their model.
    net = NeRF2(D=8, W=256, skips=[4], input_dims={"pts": 3, "view": 3, "tx": 3}, multires={"pts": 10, "view": 10, "tx": 10},
                is_embeded={"pts": True, "view": True, "tx": True}).to(device)
    renderer = Renderer_spectrum(networks_fn=net, near=cfg.near, far=cfg.far, n_samples=cfg.n_samples)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.999))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.t_max, eta_min=cfg.eta_min)

    def rays(vi, pi):
        """Origins and world-space directions for a batch of (view, pixel) pairs."""
        d = torch.einsum("bij,bj->bi", R_tr[vi], d_cam[pi]); o = o_tr[vi]
        return o, d

    @torch.no_grad()
    def render_view(Rv, ov, chunk=None):
        """A whole view, in ray chunks. Chunked because 60k rays at 64 samples
        each will not fit in one forward pass on a 12 GB card."""
        chunk = chunk or cfg.chunk
        net.eval(); out = []
        for s in range(0, n_pix, chunk):
            d = d_cam[s:s + chunk] @ Rv.T; o = ov[None].expand(d.shape[0], 3)
            out.append(renderer.render_ss(tx[None].expand(d.shape[0], 3), o, d))
        net.train(); return torch.cat(out).reshape(H, W)

    def evaluate(save=False):
        """Mean PSNR over the held-out views, in the jet RGB domain.

        Scored after re-encoding through jet, not on the scalar values: that is
        the domain every other row of the table is scored in.
        """
        ps = []
        os.makedirs(os.path.join(cfg.out, "renders"), exist_ok=True)
        for k, n in enumerate(test_names):
            pred = render_view(R_te[k], o_te[k]).clamp(0, 1); gt = y_te[k].float().reshape(H, W)
            pr, gr = jet_rgb(pred), jet_rgb(gt); ps.append(psnr(pr, gr))
            if save:
                # Both forms: the PNG for inria_metrics.py, the npy for any
                # later analysis that wants the value rather than the colour.
                Image.fromarray((pr.permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)).save(os.path.join(cfg.out, "renders", n + ".png"))
                np.save(os.path.join(cfg.out, "renders", n + ".npy"), pred.cpu().numpy().astype(np.float32))
        return float(np.mean(ps))

    os.makedirs(cfg.out, exist_ok=True)
    history = []; t_train = time.time()
    for it in range(1, cfg.iterations + 1):
        # Rays sampled independently across views and pixels, as NeRF2 does:
        # one batch mixes rays from many views rather than taking one image.
        vi = torch.as_tensor(rng.integers(0, n_tr, cfg.batch), device=device); pi = torch.as_tensor(rng.integers(0, n_pix, cfg.batch), device=device)
        o, d = rays(vi, pi); y = y_tr[vi, pi].float()
        pred = renderer.render_ss(tx[None].expand(cfg.batch, 3), o, d)
        loss = torch.mean((pred - y) ** 2)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
        if it % 100 == 0 or it == 1:
            print(f"  it {it:6d}  {time.time() - t_train:6.0f} s  loss {loss.item():.5f}  lr {opt.param_groups[0]['lr']:.2e}", flush=True)
        if it % cfg.eval_every == 0 and it < cfg.iterations:
            p = evaluate(); history.append({"iteration": it, "psnr_rgb": p, "seconds": time.time() - t_train}); print(f"  eval it {it}: PSNR(jet) {p:.2f} on {len(test_names)} views", flush=True)
    # synchronize before stopping the clock, or the time would exclude queued work.
    torch.cuda.synchronize(); train_seconds = time.time() - t_train
    p = evaluate(save=True)
    torch.save(net.state_dict(), os.path.join(cfg.out, "nerf2_state.pt"))
    # Training time is separated from total, so the jet-inverse load is not
    # charged to the method's training cost.
    result = {"config": vars(cfg), "method": "NeRF2 (third_party/NeRF2 model + Renderer_spectrum, pinhole rays)", "n_train": n_tr, "n_test": len(test_names),
              "train_seconds": train_seconds, "iters_per_second": cfg.iterations / max(train_seconds, 1e-9), "total_seconds": time.time() - t0,
              "history": history, "final": {"psnr_rgb": p}}
    json.dump(result, open(os.path.join(cfg.out, "results.json"), "w"), indent=1)
    print(f"final PSNR(jet) {p:.2f} on {len(test_names)} held-out views; {cfg.iterations} iterations in {train_seconds:.0f} s ({result['iters_per_second']:.1f} it/s); wrote {cfg.out}")


if __name__ == "__main__":
    main()
