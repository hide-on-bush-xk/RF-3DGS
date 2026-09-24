#!/bin/bash
# Round 17: the density sweep and the region hold-out again, with the delay
# decomposition (--delay-depth), so every column of the headline table comes
# from one model. Predictions written in docs/stage2_notes.md before this ran.
# Gate: no game process and GPU < 15 % for two minutes.
set -u
REPO=/c/Users/Ke/Documents/GitHub/RF-3DGS
PYS=/c/Users/Ke/miniconda3/envs/rf-sionna-win/python.exe
REG=RF-3DGS_dataset/regenerated
DS=$REG/3dgs_MULTI_24ghz_tut_cs
WSL="wsl.exe -d Ubuntu-22.04 -u ke -- bash /mnt/c/Users/Ke/Documents/GitHub/RF-3DGS/rrf_gsplat/wsl_run.sh"
export PYTHONUTF8=1 MSYS_NO_PATHCONV=1
LOG=$REPO/output/rrf/win_round17.log
cd $REPO
stamp() { echo "== $1 $(date +%H:%M:%S)" >> $LOG; }
gpu_busy() {
  tasklist 2>/dev/null | grep -qi "Against the Storm" && return 0
  u=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr -d ' \r'); [ "${u:-100}" -ge 15 ]
}
stamp "start"
quiet=0; while [ $quiet -lt 24 ]; do if gpu_busy; then quiet=0; else quiet=$((quiet+1)); fi; sleep 5; done
stamp "gpu quiet ($(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader | tr -d '\r'))"
run() { local name=$1; shift; [ -f output/rrf/$name/results.json ] && { echo "skip $name" >> $LOG; return; }
        stamp "$name"; $WSL "$name" "$@" >> $LOG 2>&1 < /dev/null; stamp "done $name"; }
base() { local name=$1; local truth=$2; shift 2; stamp "baselines $name"
         $PYS rrf_gsplat/eval_baselines.py --multi output/rrf/$name --truth $truth --out output/rrf/baselines_$name.json "$@" < /dev/null 2>&1 | grep -v jitc >> $LOG; }

for V in 640 320 160 80; do
  run m_multi_24_tut_cs_depth_v$V --source $DS --mode multi --delay-depth --max-train-views $V --save-renders 640
  base m_multi_24_tut_cs_depth_v$V $DS --max-train-views $V
done
run m_multi_24_tut_cs_depth_region --source ${DS}_region --mode multi --delay-depth --save-renders 492
base m_multi_24_tut_cs_depth_region ${DS}_region
stamp "all done"
