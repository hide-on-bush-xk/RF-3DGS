"""Controls for the speed flags of train_rrf.py: the engineering ones must not change a single number.

  1. render() of this version against the committed one (git HEAD), same random colours
  2. render_batch of a position's four faces against four render() calls
  3. --sh-backend gsplat against the torch SH, rendered
  4. evaluate_multi with --eval-group against without, on 16 held-out views

multi mode with the delay-depth term (the most involved render path), db mode, and rgb mode on a released
dataset (gsplat's SH over several cameras; 231 x 154 pictures under scaled intrinsics).
Pass = every max |difference| below 1e-4 in normalised units (float32 summation-order noise is ~1e-6);
the metrics of check 4 identical to 1e-6.

    /home/ke/miniconda3/envs/rf-gsplat/bin/python rrf_gsplat/check_face_batching.py
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import train_rrf as new  # noqa: E402

REPO = new.REPO
TOL = 1e-4


def load_head():
    """train_rrf.py as committed, imported under another name."""
    src = subprocess.run(["git", "-C", REPO, "show", "HEAD:rrf_gsplat/train_rrf.py"], capture_output=True, text=True,
                         check=True).stdout
    path = os.path.join(tempfile.mkdtemp(), "train_rrf_head.py")
    open(path, "w").write(src)
    spec = importlib.util.spec_from_file_location("train_rrf_head", path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def main():
    device = torch.device("cuda")
    head = load_head()
    ck = os.path.join(REPO, "RF-3DGS_dataset/blender_visual_trained/chkpnt30000.pth")
    report, ok = {}, True
    for mode, source in (("multi", "RF-3DGS_dataset/regenerated/3dgs_MULTI_24ghz_tut_cs"),
                         ("db", "RF-3DGS_dataset/regenerated/3dgs_MVDR_100_gpct"),
                         # rgb on a released set: gsplat's own SH for several cameras, and the 231 x 154 pictures
                         # under a 300 x 200 camera (scaled intrinsics)
                         ("rgb", "RF-3DGS_dataset/training-rf-spectrum/3dgs_AoD_100")):
        src = os.path.join(REPO, source)
        mp = os.path.join(src, "generation_meta.json")
        meta = json.load(open(mp)) if os.path.exists(mp) else {"spec_min_db": 0.0, "spec_max_db": 1.0}
        views = new.read_colmap_text(os.path.join(src, "sparse", "0"))
        test_names = new.read_index(os.path.join(src, "test_index.txt"))
        # four whole positions spread over the list (all four faces each), not a [::k] slice
        pos = np.linspace(0, len(test_names) // 4 - 1, 4).round().astype(int)
        names = [test_names[p * 4 + k] for p in pos for k in range(4)]
        data = new.load_views(src, names, views, device, want_float=True)
        ch = len(meta["channels"]) if mode == "multi" else (3 if mode == "rgb" else 1)
        span = meta["spec_max_db"] - meta["spec_min_db"]
        models = {}
        for tag, mod in (("head", head), ("new", new)):
            m = mod.RRF(ck, mode, ch, 3, device)
            g = torch.Generator(device="cpu").manual_seed(0)
            m.params["sh0"].data.copy_(torch.randn(m.params["sh0"].shape, generator=g).to(device) * 0.3)
            m.params["shN"].data.copy_(torch.randn(m.params["shN"].shape, generator=g).to(device) * 0.05)
            if mode == "multi":
                rng = torch.tensor(meta["channel_ranges"], dtype=torch.float32)
                m.delay_channel = meta["channels"].index("delay_ns")
                m.delay_span_ns = float(rng[m.delay_channel, 1] - rng[m.delay_channel, 0])
                m.delay_depth_mode, m.delay_range = "ED", "euclid"
            models[tag] = m
        with torch.no_grad():
            W, H = data["width"], data["height"]
            singles_head = torch.stack([models["head"].render(data["viewmats"][i], data["Ks"][i], W, H, span) for i in range(16)])
            singles_new = torch.stack([models["new"].render(data["viewmats"][i], data["Ks"][i], W, H, span) for i in range(16)])
            batched = torch.cat([models["new"].render_batch(data["viewmats"][4 * p:4 * p + 4], data["Ks"][4 * p:4 * p + 4], W, H, span)
                                 for p in range(4)])
            models["new"].sh_backend = "gsplat"
            batched_gs = torch.cat([models["new"].render_batch(data["viewmats"][4 * p:4 * p + 4], data["Ks"][4 * p:4 * p + 4], W, H, span)
                                    for p in range(4)])
            models["new"].sh_backend = "torch"
        r = {"head_vs_new_render": float((singles_head - singles_new).abs().max()),
             "batched_vs_single": float((batched - singles_new).abs().max()),
             "gsplat_sh_vs_torch_sh": float((batched_gs - singles_new).abs().max()),
             "value_scale": float(singles_new.abs().mean())}
        if mode == "multi":
            chr_ = torch.tensor(meta["channel_ranges"], device=device, dtype=torch.float32)
            e0 = new.evaluate_multi(models["new"], data, list(range(16)), chr_, meta["channels"])
            e1 = new.evaluate_multi(models["new"], data, list(range(16)), chr_, meta["channels"], group=True)
        else:
            e0 = new.evaluate(models["new"], data, list(range(16)), span, meta["spec_min_db"])
            e1 = new.evaluate(models["new"], data, list(range(16)), span, meta["spec_min_db"], group=True)
        r["eval_group_max_metric_diff"] = max(abs(e0[k] - e1[k]) for k in e0)
        passed = all(r[k] < TOL for k in ("head_vs_new_render", "batched_vs_single", "gsplat_sh_vs_torch_sh")) \
            and r["eval_group_max_metric_diff"] < 1e-6
        r["pass"] = passed; ok &= passed
        report[mode] = r
        print(mode, json.dumps(r))
    print("ALL PASS" if ok else "FAIL")
    json.dump(report, open(os.path.join(REPO, "output", "rrf", "check_face_batching.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
