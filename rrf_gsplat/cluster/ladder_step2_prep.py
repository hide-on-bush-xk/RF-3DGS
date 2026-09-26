"""Information ladder, step 2: inputs for the k-position curves (docs/cluster_log.md §5). CPU, run once.

    nice -n 10 ~/envs/rf-gsplat/bin/python rrf_gsplat/cluster/ladder_step2_prep.py

Writes output/cluster/ladder/step2/prep/:
  names_k{k}_r{r}.txt   nested route subsets (slot = index into the 160 route positions), repeats spread along the
                        route, every repeat of a smaller k inside a repeat of the next larger k (k = 1 x6 in k = 2 x4
                        in k = 5 x4 in k = 20_r0 in 160; k = 20_r1 in 160); all 4 faces of each position
  heldout.txt           the 307 non-route training positions (1228 views), scored after training, never trained on
  sel_box.npz / mirror_box.npz   R2-box-snap: surface Gaussians (opacity >= 0.5) whose shortest axis snaps to a room
                        axis (within 20 deg) and whose mirror ray of the transmitter passes through the receiver box
                        (xy box of the 467 training positions x their height band +- 0.1 m) within 0.3-25 m;
                        idx = Gaussian index, axis = 2 (i.n) n - i (receiver->Gaussian direction at the lobe's peak)
  sel_dark.npz          A6 (the owner's literal rule): surface Gaussians lit within 3 dB of the maximum on distinct
                        weak faces of the route and not within 3 dB on any ring-max face (from the adaptive-degree
                        assessment's rays.npz, output/cluster/ladder/explore_adaptive/selection)
  emit_img_k{k}_r{r}.npz  A7: the transmitter and its mirror images fitted on that subset's distinct views
  emit_wall_k160_r0.npz   A8: the same points moved along the line from the route-mean receiver to the wall plane
  checks.json           U1 (mirror axis, analytic) and U3 (SH basis) results
"""

from __future__ import annotations

import json
import math
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ladder_step1 as L                      # noqa: E402
import train_rrf_ladder as T                  # noqa: E402

OUT = os.path.join(L.REPO, "output", "cluster", "ladder", "step2", "prep")
SEL = os.path.join(L.REPO, "output", "cluster", "ladder", "explore_adaptive", "selection")
REPEATS = {1: 6, 2: 4, 5: 4, 20: 2, 160: 1}


def subsets():
    # 2026-09-26 review: the first version built each k independently, so k = 1 and 2 were not inside k = 5 / 20;
    # nested explicitly now (k = 5, 20, 160 unchanged)
    out = {(160, 0): list(range(160)), (20, 0): list(range(0, 160, 8)), (20, 1): list(range(4, 160, 8))}
    for j in range(4):
        out[(5, j)] = [8 * j + 32 * m for m in range(5)]                     # inside k20_r0
    for j, pair in enumerate(([0, 96], [40, 136], [48, 144], [24, 120])):    # pair inside k5_r0 / r1 / r2 / r3
        out[(2, j)] = pair
    for j, s in enumerate((0, 24, 48, 96, 120, 144)):                        # each inside a k = 2 pair
        out[(1, j)] = [s]
    for (k, r), sl in out.items():
        assert len(sl) == k and len(set(sl)) == k
    nest = {1: 2, 2: 5, 5: 20}
    for (k, r), sl in out.items():
        if k in nest:
            assert any(set(sl) <= set(v) for (kk, _), v in out.items() if kk == nest[k]), (k, r)
    return out


def mirror_axis(x, n, tx):
    i = x - tx
    i /= np.linalg.norm(i, axis=-1, keepdims=True)
    m = 2 * (i * n).sum(-1, keepdims=True) * n - i
    return m / np.linalg.norm(m, axis=-1, keepdims=True)


def main():
    os.makedirs(OUT, exist_ok=True)
    checks = {}
    # ---- names ----
    sub = subsets()
    for (k, r), slots in sub.items():
        names = [n for s in slots for n in L.GROUPS[L.ROUTE[s]]]
        open(os.path.join(OUT, f"names_k{k}_r{r}.txt"), "w").write("\n".join(names) + "\n")
    route = set(int(j) for j in L.ROUTE)
    held = [n for j, g in enumerate(L.GROUPS) if j not in route for n in g]
    open(os.path.join(OUT, "heldout.txt"), "w").write("\n".join(held) + "\n")
    print(f"subsets {len(sub)}; held-out views {len(held)} ({len(held) // 4} positions)")
    # ---- geometry selection (R2-box-snap) ----
    ck, _ = torch.load(L.CKPT, weights_only=False, map_location="cpu")
    xyz = ck[1].detach().double().numpy()
    op = torch.sigmoid(ck[6].detach().double()).reshape(-1).numpy()
    nrm = T.__dict__["gaussian_normals"](ck[5].detach().double(), ck[4].detach().double()).numpy() \
        if "gaussian_normals" in T.__dict__ else None
    if nrm is None:
        from neural_shading import gaussian_normals
        nrm = gaussian_normals(ck[5].detach().double(), ck[4].detach().double()).numpy()
    surf = np.flatnonzero(op >= 0.5)
    a = np.abs(nrm[surf]); kax = a.argmax(1)
    snapped = a[np.arange(len(a)), kax] >= math.cos(math.radians(20.0))
    n2 = np.zeros_like(nrm[surf]); n2[np.arange(len(a)), kax] = 1.0
    xs = xyz[surf]
    i = xs - L.TX; i /= np.linalg.norm(i, axis=1, keepdims=True)
    r_out = i - 2 * (i * n2).sum(1, keepdims=True) * n2                     # Gaussian -> receiver on the mirror ray
    call = np.stack([L.POSES[g[0]][2] for g in L.GROUPS])
    lo = call.min(0) - np.array([0.0, 0.0, 0.1]); hi = call.max(0) + np.array([0.0, 0.0, 0.1])
    with np.errstate(divide="ignore", invalid="ignore"):
        t1 = (lo[None] - xs) / r_out; t2 = (hi[None] - xs) / r_out
    tmin = np.maximum(np.nanmax(np.minimum(t1, t2), 1), 0.3); tmax = np.minimum(np.nanmin(np.maximum(t1, t2), 1), 25.0)
    keep = snapped & (tmax >= tmin)
    idx = surf[keep]
    axis = mirror_axis(xs[keep], n2[keep], L.TX)
    np.savez(os.path.join(OUT, "sel_box.npz"), idx=idx)
    np.savez(os.path.join(OUT, "mirror_box.npz"), idx=idx, axis=axis.astype(np.float32))
    checks["selected"] = {"surface": int(len(surf)), "snapped": int(snapped.sum()), "selected": int(len(idx)),
                          "receiver_box_lo": lo.tolist(), "receiver_box_hi": hi.tolist()}
    print("R2-box-snap:", checks["selected"])
    # cross-check the normals against the assessment's surf.npz (same Gaussians, sign-free)
    sp = os.path.join(SEL, "surf.npz")
    if os.path.exists(sp):
        S = np.load(sp)
        assert np.array_equal(S["sel"], surf), "surface set differs from the assessment's"
        checks["normals_vs_assessment_max_1_minus_abs_cos"] = float((1 - np.abs((S["nrm"] * nrm[surf]).sum(1))).max())
    # ---- U1: the mirror axis, analytic ----
    w = 8.45; x0 = np.array([w, -1.0, 0.1]); n0 = np.array([1.0, 0.0, 0.0])
    img = L.TX.copy(); img[0] = 2 * w - L.TX[0]
    p = x0 + 3.0 * (x0 - img) / np.linalg.norm(x0 - img)                  # a receiver on the reflected ray
    d = (x0 - p) / np.linalg.norm(x0 - p)
    m = mirror_axis(x0[None], n0[None], L.TX)[0]
    checks["U1_axis_vs_receiver_on_mirror_ray"] = float(np.abs(d - m).max())
    checks["U1_sign_free"] = float(np.abs(mirror_axis(x0[None], -n0[None], L.TX)[0] - m).max())
    # ---- U3: SH basis ----
    dd = torch.nn.functional.normalize(torch.randn(20000, 3, dtype=torch.float64), dim=-1)
    checks["U3_basis3_vs_sh_basis"] = float((T.sh_basis_any(3, dd) - T.sh_basis(3, dd)).abs().max())
    checks["U3_basis6_shape"] = list(T.sh_basis_any(6, dd).shape)
    # ---- A6 set ----
    rp = os.path.join(SEL, "rays.npz")
    if os.path.exists(rp):
        R = np.load(rp)
        keys = [str(k) for k in R["cat_keys"]]
        weak = R["cat_vals"][keys.index("route|weak|distinct|all")] <= 3.0
        ring = R["cat_vals"][keys.index("route|ring|any|all")] <= 3.0
        dark = surf[weak & ~ring]
        np.savez(os.path.join(OUT, "sel_dark.npz"), idx=dark)
        checks["A6_dark_selected"] = int(len(dark))
        print("A6 dark-face set:", len(dark))
    # ---- emitters (A7 / A8) ----
    info = L.view_info()
    allv = L.distinct_views(info)
    for (k, r) in ((20, 0), (20, 1), (160, 0)):
        slots = set(sub[(k, r)])
        v = [x for x in allv if x[1] in slots]
        im = L.fit_images(info, v)
        np.savez(os.path.join(OUT, f"emit_img_k{k}_r{r}.npz"), means=np.array(im["IM"]), labels=np.array(im["labels"]))
        checks[f"emitters_k{k}_r{r}"] = im["labels"]
        if k == 160:
            cbar = np.stack([L.POSES[L.GROUPS[L.ROUTE[s]][0]][2] for s in range(160)]).mean(0)
            wall = []
            for X, lab in zip(im["IM"], im["labels"]):
                if lab == "LOS":
                    wall.append(X); continue
                ax, wv = lab.split("="); a_ = "xyz".index(ax); wv = float(wv)
                t = (wv - cbar[a_]) / (X[a_] - cbar[a_])
                wall.append(cbar + t * (X - cbar))
            np.savez(os.path.join(OUT, "emit_wall_k160_r0.npz"), means=np.array(wall), labels=np.array(im["labels"]))
            checks["emitters_wall_k160"] = [list(map(float, p_)) for p_ in wall]
    json.dump(checks, open(os.path.join(OUT, "checks.json"), "w"), indent=1)
    print(json.dumps(checks, indent=1))


if __name__ == "__main__":
    main()
