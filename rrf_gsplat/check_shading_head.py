"""Controls for the shading head (--head cnn), before any of its runs count. Criteria written first:

  1. identity at the start: with every feature on (geo + phys guides, 4 latent channels, the four-face ring),
     the untrained model renders what the model without a head renders, to 1e-6 (the output layer is zero),
     with the output layer in image units and in units of the bound (--head-bound-units)
  2. guide buffers: finite, unit normals, and the direct-path alignment reaches >= 0.99 in a face that looks
     towards the transmitter
  3. gradients: on the first step only the head's output layer receives one (zero init); after one head
     step the latent feature and the SH receive gradients too
(4, render() without a head against the committed file, is check_face_batching.py.)

    /home/ke/miniconda3/envs/rf-gsplat/bin/python rrf_gsplat/check_shading_head.py
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import train_rrf as T  # noqa: E402
from neural_shading import geometry_guides, physics_guides, world_rays  # noqa: E402

REPO = T.REPO


def main():
    dev = torch.device("cuda")
    src = os.path.join(REPO, "RF-3DGS_dataset/regenerated/3dgs_MVDR_100_gpct")
    meta = json.load(open(os.path.join(src, "generation_meta.json")))
    views = T.read_colmap_text(os.path.join(src, "sparse", "0"))
    names = T.read_index(os.path.join(src, "test_index.txt"))
    pos = np.linspace(0, len(names) // 4 - 1, 4).round().astype(int)
    pick = [names[p * 4 + k] for p in pos for k in range(4)]
    data = T.load_views(src, pick, views, dev, want_float=True)
    span = meta["spec_max_db"] - meta["spec_min_db"]
    W, H = data["width"], data["height"]
    ck = os.path.join(REPO, "RF-3DGS_dataset/blender_visual_trained/chkpnt30000.pth")

    def model(with_head, bound_units=False):
        m = T.RRF(ck, "db", 1, 1, dev)
        m.sh_backend = "gsplat"
        g = torch.Generator().manual_seed(0)
        m.params["sh0"].data.copy_((torch.randn(m.params["sh0"].shape, generator=g) * 0.3).to(dev))
        m.params["shN"].data.copy_((torch.randn(m.params["shN"].shape, generator=g) * 0.05).to(dev))
        if with_head:
            T.attach_head(m, "cnn", "geo,phys", 4, True, 6.0, 32, span, meta["tx_loc"], dev, bound_units)
            m.head_cfg["order"] = [3, 2, 1, 0]
        return m

    report, ok = {}, True
    plain, full, full_bu = model(False), model(True), model(True, True)
    with torch.no_grad():
        a = torch.cat([plain.render_batch(data["viewmats"][4 * k:4 * k + 4], data["Ks"][4 * k:4 * k + 4], W, H, span) for k in range(4)])
        b = torch.cat([full.render_batch(data["viewmats"][4 * k:4 * k + 4], data["Ks"][4 * k:4 * k + 4], W, H, span) for k in range(4)])
        b1 = torch.stack([full.render(data["viewmats"][i], data["Ks"][i], W, H, span) for i in range(16)])   # per face
        bu = torch.cat([full_bu.render_batch(data["viewmats"][4 * k:4 * k + 4], data["Ks"][4 * k:4 * k + 4], W, H, span) for k in range(4)])
    report["identity_ring_max_diff"] = float((a - b).abs().max())
    report["identity_single_face_max_diff"] = float((a - b1).abs().max())
    report["identity_bound_units_max_diff"] = float((a - bu).abs().max())
    c1 = (report["identity_ring_max_diff"] < 1e-6 and report["identity_single_face_max_diff"] < 1e-6
          and report["identity_bound_units_max_diff"] < 1e-6)
    del full_bu

    # 2. the guide buffers of one ring, rebuilt the way _apply_head builds them
    with torch.no_grad():
        vm, Ks = data["viewmats"][:4], data["Ks"][:4]
        full.head_cfg_saved = full.head_cfg
        from gsplat import rasterization
        centres = torch.linalg.inv(vm)[:, :3, 3]
        nrm = full._normals
        side = torch.where(((centres[0][None] - full.means) * nrm).sum(-1, keepdim=True) < 0, -1.0, 1.0)
        img, alpha, _ = rasterization(full.means, full.quats, full.scales, full.opacities, nrm * side, vm, Ks, W, H,
                                      sh_degree=None, backgrounds=torch.zeros(4, 3, device=dev), render_mode="RGB+ED")
        img = img.permute(0, 3, 1, 2)
        normal, depth = img[:, :3], img[:, 3]
        rays, cen, cam = world_rays(vm, Ks[0], H, W)
        gg = geometry_guides(depth, normal, rays, cam, cen, 10.0)
        pg = physics_guides(depth, normal, rays, cam, cen, vm, torch.tensor(meta["tx_loc"], device=dev), 10.0)
    covered = (normal.norm(dim=1) > 0.5)
    unit = float((torch.nn.functional.normalize(normal, dim=1).norm(dim=1)[covered] - 1).abs().max())
    los_max = [float(pg[k, 0].max()) for k in range(4)]
    report["guides_finite"] = bool(torch.isfinite(gg).all() and torch.isfinite(pg).all())
    report["normal_unit_err"] = unit
    report["los_alignment_max_per_face"] = los_max
    report["depth_range_m"] = [float(depth[depth > 0].min()), float(depth.max())]
    c2 = report["guides_finite"] and unit < 1e-4 and max(los_max) >= 0.99

    # 3. gradients through the head
    opt_head = torch.optim.Adam(full.head.parameters(), lr=1e-3)
    gt = ((data["float"][:4].float() - meta["spec_min_db"]) / span).clamp(0, 1)[:, None]
    grads = []
    for step in range(2):
        for p in list(full.params.values()) + list(full.head.parameters()):
            p.grad = None
        out = full.render_batch(data["viewmats"][:4], data["Ks"][:4], W, H, span)
        (out - gt).abs().mean().backward()
        grads.append({"head_out": float(full.head.out.weight.grad.abs().sum()),
                      "head_first": float(full.head.enc1.weight.grad.abs().sum()),
                      "latent": float(full.params["latent"].grad.abs().sum()),
                      "sh0": float(full.params["sh0"].grad.abs().sum())})
        opt_head.step()
    report["grads"] = grads
    c3 = (grads[0]["head_out"] > 0 and grads[0]["head_first"] == 0 and grads[0]["latent"] == 0 and grads[0]["sh0"] > 0
          and grads[1]["head_first"] > 0 and grads[1]["latent"] > 0)
    report["pass"] = {"1_identity": c1, "2_guides": c2, "3_gradients": c3}
    ok = c1 and c2 and c3
    print(json.dumps(report, indent=1))
    print("ALL PASS" if ok else "FAIL")
    json.dump(report, open(os.path.join(REPO, "output", "rrf", "check_shading_head.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
