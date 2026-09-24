"""G2 of the per-Gaussian backward smoke: time of one training step with each backward variant (gsplat_win).

The train_rrf.py step on 3dgs_MVDR_100_gpct: db, SH3, frozen geometry, gsplat SH, one position's 4 faces per step,
Adam on SH + opacity. Each variant in its own process (its switches, see check_pergauss_bwd.py); CUDA events around
the forward (render_batch), the backward (loss.backward) and the whole step (+ Adam); 10 warm-up steps excluded,
median of 30; 300 x 200 and 600 x 400 (the intrinsics scaled, the targets upsampled). Before each variant: two
minutes under 15 % GPU utilisation, and the utilisation before / after is recorded (a busy GPU voids the timing).

Criteria, written before the run (the smoke plan): the per-Gaussian backward is >= 1.2x faster than the stock
backward at both resolutions, its forward (which now writes checkpoints) is <= 10 % slower, and the whole step
is faster. The speed-up is split into engineering (stock -> stock_dev: device-scope atomics) and method
(stock_dev -> pg); no_geom is reported beside it.

    C:/Users/Ke/miniconda3/envs/rf-gsplat-win/python.exe rrf_gsplat/time_bwd_variants.py
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
OUT = os.path.join(REPO, "output", "rrf", "pergauss_check")
VARIANTS = {
    "stock": {},
    "stock_dev": {"GSPLAT_BWD_DEVICE_ATOMICS": "1"},
    "stock_nogeom": {"GSPLAT_BWD_NO_GEOM": "1"},
    "pg": {"GSPLAT_BWD_PERGAUSS": "1"},
    "pg_nogeom": {"GSPLAT_BWD_PERGAUSS": "1", "GSPLAT_BWD_NO_GEOM": "1"},
}
SWITCHES = ("GSPLAT_BWD_DEVICE_ATOMICS", "GSPLAT_BWD_NO_GEOM", "GSPLAT_BWD_PERGAUSS", "GSPLAT_BWD_PERGAUSS_BREAK")


def gpu_util():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True).stdout
        return int(out.strip().splitlines()[0])
    except Exception:
        return -1


def quiet(seconds=120, limit=15):
    ok = 0
    while ok * 5 < seconds:
        ok = ok + 1 if 0 <= gpu_util() < limit else 0
        time.sleep(5)


def run_variant(name):
    import torch
    import torch.nn.functional as F
    sys.path.insert(0, HERE)
    import train_rrf as T
    from check_gsplat_port import MV, CK

    dev = torch.device("cuda")
    meta = json.load(open(os.path.join(MV, "generation_meta.json")))
    views = T.read_colmap_text(os.path.join(MV, "sparse", "0"))
    names = T.read_index(os.path.join(MV, "train_index.txt"))
    span, lo = meta["spec_max_db"] - meta["spec_min_db"], meta["spec_min_db"]
    m = T.RRF(CK, "db", 1, 3, dev)
    m.sh_backend = "gsplat"
    opts = {k: torch.optim.Adam([p], lr=1e-3, eps=1e-15) for k, p in m.params.items() if p.requires_grad}
    pos = np.linspace(0, len(names) // 4 - 1, 8).round().astype(int)
    rows = {}
    for scale in (1, 2):
        batches = []
        for p in pos:
            d = T.load_views(MV, [names[p * 4 + k] for k in range(4)], views, dev, want_float=True)
            K = d["Ks"].clone(); K[:, :2] *= scale
            gt = ((d["float"].float() - lo) / span).clamp(0, 1)[:, None]
            if scale > 1:
                gt = F.interpolate(gt, scale_factor=scale, mode="bilinear")
            batches.append((d["viewmats"], K, d["width"] * scale, d["height"] * scale, gt))
        tf, tb, ts = [], [], []
        for i in range(40):
            vm, K, W, H, gt = batches[i % len(batches)]
            e = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
            torch.cuda.synchronize(); e[0].record()
            out = m.render_batch(vm, K, W, H, span)
            loss = (out - gt).abs().mean()
            e[1].record()
            for o in opts.values():
                o.zero_grad(set_to_none=True)
            loss.backward()
            e[2].record()
            for o in opts.values():
                o.step()
            e[3].record(); torch.cuda.synchronize()
            if i >= 10:
                tf.append(e[0].elapsed_time(e[1])); tb.append(e[1].elapsed_time(e[2])); ts.append(e[0].elapsed_time(e[3]))
        rows[f"{300 * scale}x{200 * scale}"] = {"fwd_ms": float(np.median(tf)), "bwd_ms": float(np.median(tb)),
                                                "step_ms": float(np.median(ts)), "bwd_p10": float(np.percentile(tb, 10)),
                                                "bwd_p90": float(np.percentile(tb, 90))}
    json.dump(rows, open(os.path.join(OUT, f"time_{name}.json"), "w"), indent=1)


def main():
    res = {}
    for name, env in VARIANTS.items():
        quiet()
        before = gpu_util()
        e = {k: v for k, v in os.environ.items() if k not in SWITCHES} | env | {"PYTHONUTF8": "1"}
        p = subprocess.run([sys.executable, __file__, "--variant", name], env=e, capture_output=True, text=True)
        if p.returncode != 0:
            print(p.stderr[-3000:]); raise SystemExit(f"{name} failed")
        res[name] = json.load(open(os.path.join(OUT, f"time_{name}.json"))) | {"gpu_before": before, "gpu_after": gpu_util()}
        print(f"{name:13s} gpu {before}% -> {res[name]['gpu_after']}%  " + "  ".join(
            f"{r}: fwd {v['fwd_ms']:.2f} bwd {v['bwd_ms']:.2f} step {v['step_ms']:.2f} ms"
            for r, v in res[name].items() if isinstance(v, dict)))
    ok = True
    for r in ("300x200", "600x400"):
        s, d, g = res["stock"][r], res["stock_dev"][r], res["pg"][r]
        sp = s["bwd_ms"] / g["bwd_ms"]; fw = g["fwd_ms"] / s["fwd_ms"] - 1; st = g["step_ms"] < s["step_ms"]
        ok &= sp >= 1.2 and fw <= 0.10 and st
        print(f"{r}: backward {s['bwd_ms']:.2f} -> {g['bwd_ms']:.2f} ms = {sp:.2f}x (engineering {s['bwd_ms'] / d['bwd_ms']:.2f}x, "
              f"method {d['bwd_ms'] / g['bwd_ms']:.2f}x); forward {100 * fw:+.1f} %; step {s['step_ms']:.2f} -> {g['step_ms']:.2f} ms; "
              f"no_geom: stock {res['stock_nogeom'][r]['bwd_ms']:.2f}, pg {res['pg_nogeom'][r]['bwd_ms']:.2f} ms")
    json.dump(res, open(os.path.join(OUT, "time_report.json"), "w"), indent=1)
    print("G2 PASS" if ok else "G2 FAIL")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant")
    a = ap.parse_args()
    run_variant(a.variant) if a.variant else main()
