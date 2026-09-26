"""Information ladder step 2, new-code checks on a GPU (docs/cluster_log.md §5).

    python rrf_gsplat/cluster/ladder_step2_checks.py u2
    python rrf_gsplat/cluster/ladder_step2_checks.py u4 <run_dir_mirror_lobes> <run_dir_sh_extra>

U2  at initialisation (lobe weights and extra SH bands zero) the ladder model renders exactly what train_rrf.py's
    model renders (max |diff| <= 1e-6, normalised dB), for the 8 faces of 2 route positions
U4  after a short training run: lobe weights nonzero only on selected Gaussians (outside exactly 0), mirror axes
    unchanged, kappa inside its clamp; extra SH coefficients nonzero somewhere and inside [-1, 1]
"""

import json
import math
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "rrf_gsplat"))
sys.path.insert(0, HERE)
import train_rrf as T0              # noqa: E402
import train_rrf_ladder as T1       # noqa: E402

PREP = os.path.join(REPO, "output", "cluster", "ladder", "step2", "prep")
SRC = os.path.join(REPO, "RF-3DGS_dataset", "regenerated", "3dgs_APS_60_gp100")
CKPT = os.path.join(REPO, "RF-3DGS_dataset", "blender_visual_trained", "chkpnt30000.pth")


def u2():
    dev = "cuda"
    meta = json.load(open(os.path.join(SRC, "generation_meta.json")))
    span = meta["spec_max_db"] - meta["spec_min_db"]
    names = [l.strip() for l in open(os.path.join(PREP, "names_k2_r0.txt")) if l.strip()]
    views = T0.read_colmap_text(os.path.join(SRC, "sparse", "0"))
    data = T0.load_views(SRC, names, views, dev, want_float=True)
    m0 = T0.RRF(CKPT, "power", 1, 3, dev); m0.sh_backend = "gsplat"
    m1 = T1.RRF(CKPT, "power", 1, 3, dev); m1.sh_backend = "gsplat"
    rx = torch.linalg.inv(data["viewmats"])[:, :3, 3]
    m1.add_lobes(1, 300.0, rx)
    ml = np.load(os.path.join(PREP, "mirror_box.npz")); m1.set_mirror_lobes(ml["idx"], ml["axis"])
    m1.add_sh_extra(np.load(os.path.join(PREP, "sel_box.npz"))["idx"], 6)
    worst = 0.0
    with torch.no_grad():
        for g in (list(range(0, 4)), list(range(4, 8))):
            a = m0.render_batch(data["viewmats"][g], data["Ks"][g], data["width"], data["height"], span)
            b = m1.render_batch(data["viewmats"][g], data["Ks"][g], data["width"], data["height"], span)
            a = a[0] if isinstance(a, (tuple, list)) else a
            b = b[0] if isinstance(b, (tuple, list)) else b
            worst = max(worst, float((a - b).abs().max()))
    # the mirror lobe itself: for a selected Gaussian, a receiver on its mirror ray sees g = exp(0) = 1
    j = int(ml["idx"][0]); x = m1.means.detach()[j]
    p = x - 2.0 * torch.as_tensor(ml["axis"][0], device=dev)          # 2 m back along the receiver->Gaussian axis
    with torch.no_grad():
        d = torch.nn.functional.normalize(x - p, dim=-1)
        m = torch.nn.functional.normalize(m1.params["lobe_axis"][j, 0], dim=-1)
        g = float(torch.exp(torch.exp(m1.params["lobe_logk"][j, 0]) * ((m * d).sum() - 1.0)))
    res = {"U2_max_abs_diff": worst, "U2_pass": worst <= 1e-6, "U2_lobe_at_mirror_receiver": g,
           "U2_lobe_pass": abs(g - 1.0) <= 1e-4}
    print(json.dumps(res))
    return res


def u4(run_lobe, run_sh):
    st = torch.load(os.path.join(run_lobe, "rrf_state.pt"), map_location="cpu")
    ml = np.load(os.path.join(PREP, "mirror_box.npz"))
    sel = np.zeros(st["lobe_w"].shape[0], bool); sel[ml["idx"]] = True
    w = st["lobe_w"][:, 0, 0].numpy()
    ax = torch.nn.functional.normalize(st["lobe_axis"][ml["idx"], 0], dim=-1).numpy()
    lk = st["lobe_logk"][ml["idx"], 0].numpy()
    st2 = torch.load(os.path.join(run_sh, "rrf_state.pt"), map_location="cpu")
    h = st2["shH"].numpy()
    res = {"U4_lobe_w_nonzero_selected": int((w[sel] != 0).sum()), "U4_lobe_w_nonzero_outside": int((w[~sel] != 0).sum()),
           "U4_axis_max_change": float(np.abs(ax - ml["axis"]).max()),
           "U4_kappa_range": [float(np.exp(lk.min())), float(np.exp(lk.max()))],
           "U4_shH_nonzero": int((h != 0).sum()), "U4_shH_absmax": float(np.abs(h).max()),
           "U4_shH_count": int(h.shape[0])}
    res["U4_pass"] = (res["U4_lobe_w_nonzero_selected"] > 0 and res["U4_lobe_w_nonzero_outside"] == 0
                      and res["U4_axis_max_change"] <= 1e-6 and 30 * 0.999 <= res["U4_kappa_range"][0]
                      and res["U4_kappa_range"][1] <= 3000 * 1.001 and res["U4_shH_nonzero"] > 0 and res["U4_shH_absmax"] <= 1.0)
    print(json.dumps(res))
    return res


if __name__ == "__main__":
    if sys.argv[1] == "u2":
        r = u2()
        sys.exit(0 if r["U2_pass"] and r["U2_lobe_pass"] else 1)
    r = u4(sys.argv[2], sys.argv[3])
    sys.exit(0 if r["U4_pass"] else 1)
