"""§9: choose the beam from a predicted spectrum by its implied array covariance, then score the beam-gain loss on the
true channel (docs/cluster_log.md §9). GPU.

    python rrf_gsplat/cluster/val_beam_aware.py

Rule D (v2, 2026-09-26): R_hat = sum_pixels P(pixel) s(pixel) s(pixel)^H (P linear, s = the unit-gain steering vectors:
the spectra and the true covariance are both for isotropic receive elements), beam = argmax_pixel s^H R_hat s / |s|^2.
(v1 built R_hat from the manifold with the TR 38.901 element pattern, weighting the spectrum by |g|^2 twice over.) Rule A (as t4_score): beam = argmax of the spectrum.
Loss = true matched-beam map max - its value at the chosen beam (dB), the true map from t4_rt_cov_val_all.npz.
Controls first: U1 single-path spectra (a 1-deg splat at a known pixel + floor) -> rule D within 1 pixel of it;
U2 the true spectra -> rule D's P90 at least 2 dB under rule A's 10.30 dB and median <= 0.65 dB. Then every run.
Writes output/rrf/val_falcon/beam_aware_v2.json.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
import val_beam_score as VB                                       # noqa: E402  (installs the TR 38.901 element pattern)
sys.path.insert(0, os.path.join(REPO, "sionna_port"))
from rf_spectra import ArrayGrid                                  # noqa: E402
sys.path.insert(0, os.path.join(REPO, "rrf_gsplat"))
import protocol as PR                                             # noqa: E402

SRC = os.path.join(REPO, "RF-3DGS_dataset", "regenerated", "3dgs_APS_60_gp100")
V = os.path.join(REPO, "output", "rrf", "val_falcon")
RUNS = ["truth", "lookup_nn", "lookup_idw2", "lookup_idw8"] + [f"{a}_s{s}" for a in ("G1", "plain", "G1pc") for s in (0, 1, 2)] \
    + [f"ens_{t}_G1_s{s}" for t in ("db", "lin") for s in (0, 1, 2)]


def main():
    dev = "cuda"
    meta = json.load(open(os.path.join(SRC, "generation_meta.json")))
    grid = ArrayGrid.build(meta["M"], meta["width"], meta["height"], meta["fov_deg"], element_gain_fn=VB.element_gain_fn, device=dev)
    man = grid.manifold                                           # [M*M, H, W]
    norm = (man.abs() ** 2).sum(0)
    H, W = norm.shape

    def bart(R):
        return (torch.einsum("mhw,mn,nhw->hw", man.conj(), R, man).real / norm)

    ste = grid.steering                                           # unit-gain steering vectors [M*M, H, W]
    snorm = (ste.abs() ** 2).sum(0)

    def rule_d(p_db):
        P = torch.as_tensor(p_db, device=dev, dtype=torch.float64)
        P = torch.pow(10.0, (P - P.max()) / 10.0).to(torch.float32)
        Rh = torch.einsum("mhw,hw,nhw->mn", ste, P.to(ste.dtype), ste.conj())
        return int(torch.argmax(torch.einsum("mhw,mn,nhw->hw", ste.conj(), Rh, ste).real / snorm))

    res = {}
    # U1: analytic single-path spectra
    yy, xx = np.mgrid[0:H, 0:W]
    u1 = []
    for (py, px) in ((100, 150), (40, 60), (160, 240), (100, 20), (15, 150)):
        spot = np.exp(-((yy - py) ** 2 + (xx - px) ** 2) / (2 * 2.6 ** 2))    # ~1 deg at 0.38 deg / pixel
        p_db = 10 * np.log10(spot + 1e-4)                                      # a floor 40 dB under the path
        k = rule_d(p_db)
        u1.append({"path_pixel": [py, px], "chosen": [k // W, k % W], "pix_err": float(np.hypot(k // W - py, k % W - px))})
    res["U1"] = {"cases": u1, "pass": all(c["pix_err"] <= 1.5 for c in u1)}
    print("U1", json.dumps(res["U1"]), flush=True)
    z = np.load(os.path.join(REPO, "output", "rrf", "t4_rt_cov_val_all.npz"))
    names = PR.eval_names(os.path.join(REPO, "rrf_gsplat", "protocol_v1"), "val")
    Rmap = {str(n): R for n, R in zip(z["names"], z["R"])}
    qtrue = {}
    for n in names:
        q = bart(torch.as_tensor(Rmap[n], device=dev))
        qtrue[n] = (10 * torch.log10(q.clamp_min(1e-300))).cpu().numpy().astype(np.float64)
    for run in RUNS:
        la, ld = [], []
        for n in names:
            p = np.load(os.path.join(SRC, "spectra_float", f"{n}.npy")) if run == "truth" else \
                np.load(os.path.join(V, run, "renders", f"{n}.npy"))
            q = qtrue[n]
            ka, kd = int(np.argmax(p)), rule_d(p)
            la.append(q.max() - q.flat[ka]); ld.append(q.max() - q.flat[kd])
        la, ld = np.array(la), np.array(ld)
        res[run] = {"rule_A_median": float(np.median(la)), "rule_A_p90": float(np.percentile(la, 90)),
                    "rule_D_median": float(np.median(ld)), "rule_D_p90": float(np.percentile(ld, 90)),
                    "rule_D_gt8_pct": float(100 * (ld > 8).mean())}
        print(f"{run:16s} rule A {res[run]['rule_A_median']:5.2f} / {res[run]['rule_A_p90']:5.2f} dB | rule D "
              f"{res[run]['rule_D_median']:5.2f} / {res[run]['rule_D_p90']:5.2f} dB (>8 dB {res[run]['rule_D_gt8_pct']:.1f} %)", flush=True)
        if run == "truth":
            t = res["truth"]
            res["U2"] = {"pass": t["rule_D_p90"] <= t["rule_A_p90"] - 2 and t["rule_D_median"] <= 0.65}
            print("U2", json.dumps(res["U2"]), flush=True)
            if not (res["U1"]["pass"] and res["U2"]["pass"]):
                json.dump(res, open(os.path.join(V, "beam_aware_v2.json"), "w"), indent=1)
                sys.exit("controls failed: other runs not scored")
    json.dump(res, open(os.path.join(V, "beam_aware_v2.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
