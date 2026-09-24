#!/bin/bash
# Re-measure the timings the reporting contract voided, on an idle GPU, one at a
# time: the rasteriser-vs-resolution benchmark (INRIA on Windows, gsplat in WSL),
# the tutorial's Sionna 0.19 pipeline on the CPU with nothing else running, and
# the interactive planner's endpoints as medians of five calls.
set -u
REPO=/c/Users/Ke/Documents/GitHub/RF-3DGS
PYS=/c/Users/Ke/miniconda3/envs/rf-sionna-win/python.exe
PYT=/c/Users/Ke/miniconda3/envs/rf-3dgs/python.exe
export PYTHONUTF8=1 MSYS_NO_PATHCONV=1
LOG=$REPO/output/rrf/timing_remeasure.log
cd $REPO
echo "== start $(date)" > $LOG
nvidia-smi --query-gpu=name,utilization.gpu,memory.used --format=csv,noheader >> $LOG

echo "== bench_resolution inria (Windows, exclusive)" >> $LOG
$PYT rrf_gsplat/bench_resolution.py --which inria --reps 30 2>&1 | grep -a "ms/step" >> $LOG
echo "== bench_resolution gsplat (WSL, exclusive)" >> $LOG
wsl.exe -d Ubuntu-22.04 -u ke -- /home/ke/miniconda3/envs/rf-gsplat/bin/python /mnt/c/Users/Ke/Documents/GitHub/RF-3DGS/rrf_gsplat/bench_resolution.py --which gsplat --reps 30 2>&1 | grep -a "ms/step" >> $LOG

echo "== tutorial 0.19 pipeline (CPU, nothing else running)" >> $LOG
wsl.exe -d Ubuntu-22.04 -u ke -- bash /mnt/c/Users/Ke/AppData/Local/Temp/claude/c--Users-Ke-Documents-GitHub-RF-3DGS/ed7324c0-b974-4409-b2e5-4c5d23e75259/scratchpad/run_019_timing.sh 2>&1 | grep -a "pos \|per view" >> $LOG

echo "== interactive planner: medians of 5 (exclusive)" >> $LOG
$PYS tx_planning/interactive.py --port 8766 > output/tx_planning/interactive_bench.log 2>&1 &
SRV=$!
$PYS - <<'EOF' >> $LOG 2>&1
import json, time, urllib.request, statistics
base = "http://127.0.0.1:8766"
for _ in range(60):
    try:
        urllib.request.urlopen(base + "/api/grid", timeout=5).read(); break
    except Exception:
        time.sleep(3)
def post(path, body, timeout=900):
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.time(); r = json.loads(urllib.request.urlopen(req, timeout=timeout).read()); return r, time.time() - t0
# five indoor transmitter positions at AP height (the active-measurement candidates)
POS = [[0.0, -5.0, 2.0], [0.0, -7.0, 2.0], [8.0, -7.0, 2.0], [8.0, -11.0, 2.0], [8.0, -3.0, 2.0]]
post("/api/coverage", {"tx": POS[0], "threshold": -85})          # warm-up
cov = [post("/api/coverage", {"tx": p, "threshold": -85})[1] for p in POS]
post("/api/optimize", {"tx": POS[0], "threshold": -85, "steps": 1})   # warm-up (kernel compile)
opt = []
for p in POS:
    r, t = post("/api/optimize", {"tx": p, "threshold": -85, "steps": 5})
    if len(r["history"]) == 5:
        opt.append(t)
    else:
        print(f"  optimise from {p}: only {len(r['history'])} steps (blocked), excluded")
print(f"coverage (239 rx, 50k samples): median {statistics.median(cov):.3f} s over {len(cov)}, all {[round(v,3) for v in cov]}")
print(f"optimise 5 steps (20k samples, reverse mode): median {statistics.median(opt):.2f} s over {len(opt)} valid, all {[round(v,2) for v in opt]}")
EOF
kill $SRV 2>/dev/null; taskkill //F //IM python.exe >/dev/null 2>&1
echo "== done $(date)" >> $LOG
cat $LOG
