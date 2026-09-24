#!/bin/bash
# Round 16 (Ke's eighth list): the evaluation protocol, on the MULTI cs dataset.
#   0. smoke: --delay-depth for 200 iterations (new render path);
#   1. density sweep: 160 / 80 / 40 / 20 training positions (the 800 run
#      exists), same held-out set, each scored against the copy baselines
#      drawn from the SAME training subset;
#   2. leave-one-region: the south-east corridor end (x > 4, y < -7; 123
#      positions) held out, trained on the rest;
#   3. the delay decomposition (learned residual + rendered depth / c).
# Gate: no game process and GPU < 15 % for two minutes.
set -u
REPO=/c/Users/Ke/Documents/GitHub/RF-3DGS
PYS=/c/Users/Ke/miniconda3/envs/rf-sionna-win/python.exe
REG=RF-3DGS_dataset/regenerated
DS=$REG/3dgs_MULTI_24ghz_tut_cs
WSL="wsl.exe -d Ubuntu-22.04 -u ke -- bash /mnt/c/Users/Ke/Documents/GitHub/RF-3DGS/rrf_gsplat/wsl_run.sh"
export PYTHONUTF8=1 MSYS_NO_PATHCONV=1
LOG=$REPO/output/rrf/win_round16.log
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

# 0. smoke of the new render path
run smoke_delay_depth --source $DS --mode multi --iterations 200 --eval-every 100 --delay-depth --save-renders 0
grep -q "final on" output/rrf/smoke_delay_depth/train.log || { stamp "delay-depth smoke FAILED; stopping"; exit 1; }

# 1. density sweep (views = positions x 4)
for V in 640 320 160 80; do
  run m_multi_24_tut_cs_v$V --source $DS --mode multi --max-train-views $V --save-renders 640
  base m_multi_24_tut_cs_v$V $DS --max-train-views $V
done

# 2. leave-one-region
if [ ! -d ${DS}_region ]; then
  stamp "region split"
  $PYS rrf_gsplat/make_region_split.py --source $DS --out ${DS}_region --xmin 4 --ymax -7 < /dev/null 2>&1 | grep -v jitc >> $LOG
fi
run m_multi_24_tut_cs_region --source ${DS}_region --mode multi --save-renders 492
base m_multi_24_tut_cs_region ${DS}_region

# 3. delay decomposition
run m_multi_24_tut_cs_depth --source $DS --mode multi --delay-depth --save-renders 640
base m_multi_24_tut_cs_depth $DS
stamp "all done"
