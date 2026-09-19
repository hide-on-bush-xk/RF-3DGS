"""Is the transmitter-specific geometry adaptation low-rank?

Ten source transmitters were each adapted (means / scales / quats unfrozen,
10k steps) from the same visual checkpoint. Each gives a deformation
Delta_i = (d log s [N,3], rotation vector [N,3], d mu [N,3]). If the
Tx-specific need is low-rank -- "a few extra degrees of freedom, anywhere"
rather than "these scatterers need these shapes" -- the ten deformations
share a few basis deformations and a moved transmitter only needs new
coefficients. This stacks the ten and reports:

  * the singular-value spectrum of the 10 x 9N matrix, uncentered (energy)
    and centered (variance about the mean deformation, which is the
    transferable part), per block and combined (blocks scaled to unit RMS);
  * pairwise cosine similarity of the deformations, against Tx distance;
  * whether the coefficients are a smooth function of the Tx position:
    leave-one-out, fit PCA on nine, regress coefficients linearly on
    (x, y, z), predict the tenth's deformation, and compare its relative
    error to the nearest source's deformation and to the mean deformation.

Caveat written before the numbers: this is parameter-space rank. The field
is redundant (6-8 Gaussians per ray; a random 1 % of them recovers 73 % of
the gain), so the same functional correction can be realised by different
parameter deltas. A low-rank spectrum supports the hypothesis; a full-rank
one does not kill it, it only says the runs did not converge to shared
parameters.

    PYTHONUTF8=1 python rrf_gsplat/diag_delta_pca.py
"""

from __future__ import annotations

import json
import os

import numpy as np
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(REPO, "output/rrf")
CKPT = os.path.join(REPO, "RF-3DGS_dataset/blender_visual_trained/chkpnt30000.pth")
SOURCES = {"A": "a_mvdr_db_geom", "C": "a_txC_db_geom"}
SOURCES.update({n: f"a_tx{n}_db_geom" for n in "DENLOHMK"})
TX = {"A": (6.905, 0.0, 0.287), "C": (0.0, -3.0, 2.0), "D": (6.9, -5.4, 0.8), "E": (6.9, -2.7, 0.8), "N": (5.0, -3.0, 0.8),
      "L": (3.5, -7.5, 0.8), "O": (2.0, -3.0, 0.8), "H": (1.98, -9.27, 0.8), "M": (-1.0, -7.0, 0.8), "K": (-3.35, 0.0, 0.8)}


def quat_normalize(q):
    return q / q.norm(dim=1, keepdim=True).clamp_min(1e-12)


def quat_mul(a, b):
    """(w, x, y, z) Hamilton product, batched."""
    aw, ax, ay, az = a.unbind(1); bw, bx, by, bz = b.unbind(1)
    return torch.stack([aw * bw - ax * bx - ay * by - az * bz,
                        aw * bx + ax * bw + ay * bz - az * by,
                        aw * by - ax * bz + ay * bw + az * bx,
                        aw * bz + ax * by - ay * bx + az * bw], 1)


def rotvec_between(q_from, q_to):
    """Rotation vector of q_to * conj(q_from): the rotation applied to the Gaussian's axes."""
    q_from, q_to = quat_normalize(q_from), quat_normalize(q_to)
    conj = q_from * torch.tensor([1.0, -1.0, -1.0, -1.0])
    r = quat_mul(q_to, conj)
    r = torch.where(r[:, :1] < 0, -r, r)                              # w >= 0: the short rotation
    v = r[:, 1:]; n = v.norm(dim=1, keepdim=True)
    ang = 2 * torch.atan2(n, r[:, :1])
    return v / n.clamp_min(1e-12) * ang


def main():
    (m, _) = torch.load(CKPT, weights_only=False, map_location="cpu")
    (_, xyz0, _, _, scaling0, rotation0, *_ ) = m
    xyz0, scaling0, rotation0 = (t.detach().float() for t in (xyz0, scaling0, rotation0))
    names, blocks = [], {"dlogs": [], "drot": [], "dmu": []}
    for name, run in SOURCES.items():
        p = os.path.join(OUT, run, "rrf_state.pt")
        if not os.path.exists(p):
            continue
        st = torch.load(p, map_location="cpu")
        names.append(name)
        blocks["dlogs"].append((st["scales"].float() - scaling0).numpy())
        blocks["drot"].append(rotvec_between(rotation0, st["quats"].float()).numpy())
        blocks["dmu"].append((st["means"].float() - xyz0).numpy())
    n_src = len(names)
    pos = np.array([TX[n] for n in names])
    dist = np.linalg.norm(pos[:, None] - pos[None], axis=2)
    out = {"sources": names, "n_gaussians": int(xyz0.shape[0]), "blocks": {}}
    print(f"{n_src} sources, {xyz0.shape[0]:,} Gaussians")

    def spectrum(X, label):
        """X [n_src, D]. Returns uncentered energy shares and centered variance shares."""
        G = X @ X.T
        ev = np.sort(np.linalg.eigvalsh(G))[::-1].clip(min=0)
        Xc = X - X.mean(0, keepdims=True)
        evc = np.sort(np.linalg.eigvalsh(Xc @ Xc.T))[::-1].clip(min=0)
        e, ec = ev / ev.sum(), evc / max(evc.sum(), 1e-30)
        norms = np.sqrt(np.diag(G))
        cos = G / np.outer(norms, norms)
        print(f"\n[{label}] per-source RMS: " + ", ".join(f"{n}:{v / np.sqrt(X.shape[1]):.3g}" for n, v in zip(names, norms)))
        print(f"  uncentered energy share, cumulative: " + " ".join(f"{v:.2f}" for v in np.cumsum(e)))
        print(f"  centered variance share, cumulative: " + " ".join(f"{v:.2f}" for v in np.cumsum(ec)))
        print(f"  mean deformation carries {ev.sum() - evc.sum():.3g} of {ev.sum():.3g} energy ({1 - evc.sum() / ev.sum():.1%})")
        iu = np.triu_indices(n_src, 1)
        r = np.corrcoef(cos[iu], dist[iu])[0, 1]
        print(f"  pairwise cosine: mean {cos[iu].mean():.3f}, min {cos[iu].min():.3f}, max {cos[iu].max():.3f}; corr with Tx distance {r:+.2f}")
        near = sorted(zip(cos[iu], dist[iu], [f"{names[i]}-{names[j]}" for i, j in zip(*iu)]), reverse=True)[:4]
        print(f"  most similar pairs: " + ", ".join(f"{p} ({c:.2f}, {d:.1f} m)" for c, d, p in near))
        return {"energy_cum": np.cumsum(e).round(4).tolist(), "variance_cum": np.cumsum(ec).round(4).tolist(),
                "mean_share": float(1 - evc.sum() / ev.sum()), "cosine": cos.round(4).tolist(), "cos_dist_corr": float(r)}

    Xs = {}
    for key in ("dlogs", "drot", "dmu"):
        X = np.stack([b.reshape(-1) for b in blocks[key]]).astype(np.float64)
        Xs[key] = X
        out["blocks"][key] = spectrum(X, key)
    # combined, each block scaled to unit pooled RMS
    Xall = np.concatenate([Xs[k] / np.sqrt((Xs[k] ** 2).mean()) for k in ("dlogs", "drot", "dmu")], 1)
    out["blocks"]["combined"] = spectrum(Xall, "combined (blocks at unit RMS)")

    # leave-one-out: predict the held-out deformation from the others' coefficients regressed on Tx position
    print("\nleave-one-out prediction of a source's deformation (combined, relative error ||pred - true|| / ||true||):")
    print(f"{'held out':>9} {'k=1':>6} {'k=3':>6} {'k=5':>6} {'mean':>6} {'nearest':>8} {'d nearest':>9}")
    loo = []
    for i in range(n_src):
        keep = [j for j in range(n_src) if j != i]
        Xk = Xall[keep]; mu = Xk.mean(0, keepdims=True); Xc = Xk - mu
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)          # Xc = U S Vt; coefficients = U S
        coef = U * S
        P = np.concatenate([np.ones((len(keep), 1)), pos[keep]], 1)  # linear in (x, y, z)
        row = {"held": names[i]}
        true = Xall[i]; tn = np.linalg.norm(true)
        for k in (1, 3, 5):
            kk = min(k, coef.shape[1])
            beta, *_ = np.linalg.lstsq(P, coef[:, :kk], rcond=None)
            c_pred = np.concatenate([[1.0], pos[i]]) @ beta
            pred = mu[0] + c_pred @ Vt[:kk]
            row[f"k{k}"] = float(np.linalg.norm(pred - true) / tn)
        row["mean"] = float(np.linalg.norm(mu[0] - true) / tn)
        j = min(keep, key=lambda j: dist[i, j])
        row["nearest"] = float(np.linalg.norm(Xall[j] - true) / tn); row["d_nearest"] = float(dist[i, j])
        loo.append(row)
        print(f"{row['held']:>9} {row['k1']:6.2f} {row['k3']:6.2f} {row['k5']:6.2f} {row['mean']:6.2f} {row['nearest']:8.2f} {row['d_nearest']:9.1f}")
    out["loo"] = loo
    for key in ("k1", "k3", "k5", "mean", "nearest"):
        print(f"  median relative error, {key}: {np.median([r[key] for r in loo]):.2f}")
    json.dump(out, open(os.path.join(OUT, "diag_delta_pca.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
