"""Unit checks of the main-peak direction loss (dir_loss.py), before it goes into training. No training here.

Criteria, written before the first run (2026-09-25):
  U1  analytic, synthetic ring (4 faces, one true pixel 30 dB above an otherwise flat floor), T = 0.05 dB:
        pred = truth                         -> L <= 0.01 dB
        pred peak on a pixel 30 dB down      -> L = 30 dB +- 0.1 (the loss of steering the beam there)
  U2  real targets (the first training position with a distinct main peak and a rival >= 3 dB down), T = 0.05 dB:
        pred = truth                         -> L <= 0.01 dB
        known failure: faces rotated by one  -> L equals the analytic loss of steering to the rotated argmax within
                                                0.01 dB, and is >= 3 dB
      reported: pred = truth at T = 1 dB (the spread over the main lobe)
  U3  a constant added to pred leaves L unchanged (<= 1e-4 dB)
  U4  solid angle: a flat pred gives pi proportional to cos^3: corner / centre ratio within 1e-4 (relative) of
      (1 + u^2 + v^2)^-1.5, and sum(pi) = 1 within 1e-5
  U5  liveness: a free pred initialised to the rotated truth, Adam (lr 0.01, normalised units), 300 steps of expgain at
      T = 1 dB: the gradient at step 1 is finite and nonzero; the final argmax lies on the true main peak's face within
      3 pixels of it; the final L (T = 0.05 dB) is >= 3 dB below the initial one
  U6  ce: KL <= 1e-6 at pred = truth, > 0.1 at the rotated truth
  U7  degenerate: a flat truth (no paths) gives L = 0 exactly and a finite gradient

First run (2026-09-25): U1, U3, U4, U6 pass; U2, U5, U7 fail as written.
  U2  0.0165 dB at pred = truth (limit 0.01): the real peak is smooth (the 1 deg kernel), so at T = 0.05 dB the pixels
      next to the maximum still share pi. The known-failure half matched the analytic value exactly (74.4608 dB).
  U5  the design flaw: from the rotated truth the prediction at the true peak is ~74 dB under its own maximum, pi
      there is ~e^-74 at T = 1 dB, and the gradient of the expected gain vanishes (|g| 5e-7; L unchanged). The
      expected-reward objective only sees directions the policy already favours: the exploration problem of policy
      gradients, present even though the expectation is exact here.
  U7  L = -6.5e-9, not exactly 0: float rounding in logsumexp; the criterion was written too strictly.
Second round, written after the first run and before the second:
  U5b expgain with T annealed geometrically from 20 dB to 1 dB over the 300 steps passes U5's criteria
  U5c ce at T = 1 dB (gradient pi - q: it raises the true peak whatever pi is) passes U5's criteria (L measured as
      expgain at T = 0.05 dB, as in U5)

    C:/Users/Ke/miniconda3/envs/rf-gsplat-win/python.exe rrf_gsplat/check_dir_loss.py
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import protocol as PR                                        # noqa: E402
from dir_loss import dir_loss, solid_angle_logw              # noqa: E402
from mvdr_peaks import pixel_dirs, top_peaks                 # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TRUTH = os.path.join(REPO, "RF-3DGS_dataset", "regenerated", "3dgs_APS_60_gp100")


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    p = open(os.path.join(TRUTH, "sparse", "0", "cameras.txt")).read().split()
    W, H, fx, fy, cx, cy = int(p[2]), int(p[3]), *map(float, p[4:8])
    K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    logw = solid_angle_logw(K, W, H, dev)
    vmin, vmax = json.load(open(os.path.join(REPO, "output", "rrf", "m3", "rr", "plain_s0", "results.json")))["db_range"]
    span = vmax - vmin
    L = lambda pr, gt, T, kind="expgain": float(dir_loss(pr, gt, logw, span, T, kind)) * (span if kind == "expgain" else 1)
    ok = {}

    # U1: synthetic
    gt = torch.zeros(4, H, W, device=dev); gt[2, 80, 200] = 30.0 / span
    pr = torch.zeros_like(gt); pr[0, 100, 150] = 30.0 / span
    a, b = L(gt, gt, 0.05), L(pr, gt, 0.05)
    ok["U1"] = a <= 0.01 and abs(b - 30.0) <= 0.1
    print(f"U1 synthetic: pred = truth {a:.4f} dB (<= 0.01); peak 30 dB down {b:.3f} dB (30 +- 0.1) -> {'PASS' if ok['U1'] else 'FAIL'}")

    # U2: real targets
    dirs = pixel_dirs(TRUTH)
    tr = PR.train_names(os.path.join(REPO, "rrf_gsplat", "protocol_v1"))
    for k in range(0, len(tr), 4):
        faces = [np.load(os.path.join(TRUTH, "spectra_float", n + ".npy")).astype(np.float32) for n in tr[k:k + 4]]
        ring = np.stack(faces)
        f0 = int(ring.reshape(4, -1).max(1).argmax()); t = ring[f0]
        kept = top_peaks(t, dirs, 3)
        if len(kept) >= 2 and t.flat[kept[0]] - t.flat[kept[1]] >= 3.0 and ring.max() - np.delete(ring, f0, 0).max() >= 3.0:
            break
    gt = torch.tensor((ring - vmin) / span, device=dev)
    rot = torch.roll(gt, 1, 0)
    a, c1 = L(gt, gt, 0.05), L(gt, gt, 1.0)
    g_db = gt * span
    am = int(torch.argmax(rot))
    analytic = float(g_db.max() - g_db.reshape(-1)[am])
    b = L(rot, gt, 0.05)
    ok["U2"] = a <= 0.01 and abs(b - analytic) <= 0.01 and b >= 3.0
    print(f"U2 real ({tr[k]}..{tr[k + 3]}): pred = truth {a:.4f} dB (<= 0.01), at T = 1 dB {c1:.3f} dB (reported); "
          f"faces rotated {b:.4f} dB vs analytic {analytic:.4f} (>= 3) -> {'PASS' if ok['U2'] else 'FAIL'}")

    # U3: invariance
    d = abs(L(rot + 7.0 / span, gt, 1.0) - L(rot, gt, 1.0))
    ok["U3"] = d <= 1e-4
    print(f"U3 constant offset changes L by {d:.2e} dB (<= 1e-4) -> {'PASS' if ok['U3'] else 'FAIL'}")

    # U4: solid angle
    lp = torch.log_softmax(logw.reshape(-1), 0).reshape(H, W)
    ratio = float((lp[0, 0] - lp[H // 2, W // 2]).exp())
    u0, v0 = (0.5 - cx) / fx, (0.5 - cy) / fy; uc, vc = (W // 2 + 0.5 - cx) / fx, (H // 2 + 0.5 - cy) / fy
    expect = ((1 + u0 ** 2 + v0 ** 2) / (1 + uc ** 2 + vc ** 2)) ** -1.5
    s = float(lp.exp().sum())
    ok["U4"] = abs(ratio / expect - 1) <= 1e-4 and abs(s - 1) <= 1e-5
    print(f"U4 corner / centre {ratio:.6f} vs {expect:.6f}; sum pi {s:.7f} -> {'PASS' if ok['U4'] else 'FAIL'}")

    # U5: liveness
    x = rot.clone().requires_grad_(True)
    opt = torch.optim.Adam([x], lr=0.01)
    l0 = L(x.detach(), gt, 0.05)
    for step in range(300):
        opt.zero_grad(); loss = dir_loss(x, gt, logw, span, 1.0); loss.backward()
        if step == 0:
            g1 = float(x.grad.abs().sum()); fin = bool(torch.isfinite(x.grad).all())
        opt.step()
    l1 = L(x.detach(), gt, 0.05)
    pf, pr_, pc = np.unravel_index(int(torch.argmax(x.detach())), x.shape)
    tf, trr, tc = np.unravel_index(int(torch.argmax(gt)), gt.shape)
    near = pf == tf and abs(pr_ - trr) <= 3 and abs(pc - tc) <= 3
    ok["U5"] = fin and g1 > 0 and near and l0 - l1 >= 3.0
    print(f"U5 gradient at step 1 finite {fin}, |g| {g1:.3e}; argmax face {pf} ({pr_}, {pc}) vs true face {tf} ({trr}, {tc}); "
          f"L {l0:.3f} -> {l1:.3f} dB -> {'PASS' if ok['U5'] else 'FAIL'}")

    # U5b / U5c: the two ways round the vanishing gradient
    tf, trr, tc = np.unravel_index(int(torch.argmax(gt)), gt.shape)
    for name, kind, temps in (("U5b", "expgain", np.geomspace(20.0, 1.0, 300)), ("U5c", "ce", np.full(300, 1.0))):
        x = rot.clone().requires_grad_(True)
        opt = torch.optim.Adam([x], lr=0.01)
        for step, T in enumerate(temps):
            opt.zero_grad(); loss = dir_loss(x, gt, logw, span, float(T), kind); loss.backward()
            if step == 0:
                g1 = float(x.grad.abs().sum()); fin = bool(torch.isfinite(x.grad).all())
            opt.step()
        lb = L(x.detach(), gt, 0.05)
        pf, pr_, pc = np.unravel_index(int(torch.argmax(x.detach())), x.shape)
        near = pf == tf and abs(pr_ - trr) <= 3 and abs(pc - tc) <= 3
        ok[name] = fin and g1 > 0 and near and l0 - lb >= 3.0
        print(f"{name} {kind}: gradient at step 1 finite {fin}, |g| {g1:.3e}; argmax face {pf} ({pr_}, {pc}) vs true face {tf} "
              f"({trr}, {tc}); L {l0:.3f} -> {lb:.3f} dB -> {'PASS' if ok[name] else 'FAIL'}")

    # U6: ce
    k0, k1 = L(gt, gt, 1.0, "ce"), L(rot, gt, 1.0, "ce")
    ok["U6"] = k0 <= 1e-6 and k1 > 0.1
    print(f"U6 ce: KL at truth {k0:.2e}, rotated {k1:.3f} -> {'PASS' if ok['U6'] else 'FAIL'}")

    # U7: degenerate
    flat = torch.zeros(4, H, W, device=dev)
    y = torch.randn(4, H, W, device=dev).mul(0.1).requires_grad_(True)
    lf = dir_loss(y, flat, logw, span, 1.0); lf.backward()
    ok["U7"] = float(lf) == 0.0 and bool(torch.isfinite(y.grad).all())
    print(f"U7 flat truth: L {float(lf)!r}, gradient finite {bool(torch.isfinite(y.grad).all())} -> {'PASS' if ok['U7'] else 'FAIL'}")
    print("ALL PASS" if all(ok.values()) else "FAIL: " + ", ".join(k for k, v in ok.items() if not v))


if __name__ == "__main__":
    main()
