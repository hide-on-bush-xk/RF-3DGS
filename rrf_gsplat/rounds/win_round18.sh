#!/bin/bash
# Round 18 (Ke's tenth list, items 2 and 3), before scene 2's RF queue:
#   ED   the delay range term as expected depth (sum w d / alpha) instead of
#        accumulated depth (sum w d): the signed delay error is negative on the
#        tail (-1.10 ns mean), which (1 - alpha) x depth / c would produce.
#   FPS  the 80- and 40-position densities with farthest-point subsampling,
#        the strongest lookup a subset of that size can get; copy baselines
#        from the same fps subset.
# Gate: scene 2's visual checkpoint exists (its training is on the card), no
# game, GPU < 15 % for two minutes.
set -u
REPO=/c/Users/Ke/Documents/GitHub/RF-3DGS
PYS=/c/Users/Ke/miniconda3/envs/rf-sionna-win/python.exe
REG=RF-3DGS_dataset/regenerated
DS=$REG/3dgs_MULTI_24ghz_tut_cs
WSL="wsl.exe -d Ubuntu-22.04 -u ke -- bash /mnt/c/Users/Ke/Documents/GitHub/RF-3DGS/rrf_gsplat/wsl_run.sh"
export PYTHONUTF8=1 MSYS_NO_PATHCONV=1
LOG=$REPO/output/rrf/win_round18.log
cd $REPO
stamp() { echo "== $1 $(date +%H:%M:%S)" >> $LOG; }
gpu_busy() {
  tasklist 2>/dev/null | grep -qi "Against the Storm" && return 0
  u=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr -d ' \r'); [ "${u:-100}" -ge 15 ]
}
stamp "start"
for i in $(seq 1 5760); do [ -f scene2/visual_trained/chkpnt30000.pth ] && break; sleep 5; done
quiet=0; while [ $quiet -lt 24 ]; do if gpu_busy; then quiet=0; else quiet=$((quiet+1)); fi; sleep 5; done
stamp "gpu quiet ($(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader | tr -d '\r'))"
run() { local name=$1; shift; [ -f output/rrf/$name/results.json ] && { echo "skip $name" >> $LOG; return; }
        stamp "$name"; $WSL "$name" "$@" >> $LOG 2>&1 < /dev/null; stamp "done $name"; }
base() { local name=$1; local truth=$2; shift 2; stamp "baselines $name"
         $PYS rrf_gsplat/eval_baselines.py --multi output/rrf/$name --truth $truth --out output/rrf/baselines_$name.json "$@" < /dev/null 2>&1 | grep -v jitc >> $LOG; }

run m_multi_24_tut_cs_depth_ed --source $DS --mode multi --delay-depth --delay-depth-mode ED --save-renders 640
base m_multi_24_tut_cs_depth_ed $DS
for V in 320 160; do
  run m_multi_24_tut_cs_depth_v${V}_fps --source $DS --mode multi --delay-depth --max-train-views $V --subset-mode fps --save-renders 640
  base m_multi_24_tut_cs_depth_v${V}_fps $DS --max-train-views $V --subset-mode fps
done
stamp "all done"
