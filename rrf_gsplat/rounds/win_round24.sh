#!/bin/bash
# Round 24: DLSS-inspired training speed. Profile first (profile_resolution.py): at 300 x 200 a step does not get
# cheaper below 600 x 400, so super-resolution cannot speed training; the step is the per-Gaussian colour
# evaluation and the per-parameter Adam update. What DLSS frame generation does -- shade once, produce extra frames
# cheaply -- maps onto the four faces of one receiver position, which share the colour evaluation exactly.
#
#   B0   as before: torch SH, one view per step, 10k steps                              (the recorded headline config)
#   E1   engineering only: gsplat CUDA SH + grouped evaluation, identical numbers        (check_face_batching.py)
#   M*   + method: the four faces of one position per step (one colour evaluation, one Adam step), lr scale, steps
# Same seed, same data, exclusive RTX 3060, decoded metrics on all 640 held-out views (eval_baselines.py).
set -u
REPO=/c/Users/Ke/Documents/GitHub/RF-3DGS
PYS=/c/Users/Ke/miniconda3/envs/rf-sionna-win/python.exe
WSL="wsl.exe -d Ubuntu-22.04 -u ke -- bash /mnt/c/Users/Ke/Documents/GitHub/RF-3DGS/rrf_gsplat/wsl_run.sh"
export PYTHONUTF8=1 MSYS_NO_PATHCONV=1
LOG=$REPO/output/rrf/win_round24.log
cd $REPO
stamp() { echo "== $1 $(date +%H:%M:%S) gpu $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader | tr -d '\r')" >> $LOG; }
MUL=RF-3DGS_dataset/regenerated/3dgs_MULTI_24ghz_tut_cs
MV=RF-3DGS_dataset/regenerated/3dgs_MVDR_100_gpct
MC="--source $MUL --mode multi --delay-depth --delay-depth-mode ED --delay-range euclid --save-renders -1"
DB="--source $MV --mode db --save-renders -1"
FAST="--sh-backend gsplat --eval-group"

run() {   # name, args...
  local name=$1; shift
  if [ -f output/rrf/$name/results.json ]; then echo "skip $name (done)" >> $LOG; return; fi
  stamp "start $name"; local T0=$(date +%s)
  $WSL $name "$@" >> $LOG 2>&1 < /dev/null
  echo "wall $name $(( $(date +%s) - T0 )) s" >> $LOG
}
decode() {  # name
  local name=$1
  [ -f output/rrf/$name/results.json ] || return
  [ -f output/rrf/baselines_$name.json ] && return
  $PYS rrf_gsplat/eval_baselines.py --multi output/rrf/$name --truth $MUL --out output/rrf/baselines_$name.json < /dev/null 2>&1 | grep -v jitc | tail -4 >> $LOG
}

stamp "round 24 start"
run r24_B0_multi      $MC --iterations 10000
run r24_E1_multi      $MC $FAST --iterations 10000
run r24_M1_multi_f4_lr1_2500   $MC $FAST --faces-per-step 4 --iterations 2500 --eval-every 250
run r24_M2_multi_f4_lr2_2500   $MC $FAST --faces-per-step 4 --lr-scale 2 --iterations 2500 --eval-every 250
run r24_M3_multi_f4_lr2_5000   $MC $FAST --faces-per-step 4 --lr-scale 2 --iterations 5000 --eval-every 500
run r24_M4_multi_f4_lr1_10000  $MC $FAST --faces-per-step 4 --iterations 10000
run r24_B0_db         $DB --iterations 10000
run r24_E1_db         $DB $FAST --iterations 10000
run r24_M2_db_f4_lr2_2500      $DB $FAST --faces-per-step 4 --lr-scale 2 --iterations 2500 --eval-every 250
stamp "training done; decoding"
for n in r24_B0_multi r24_E1_multi r24_M1_multi_f4_lr1_2500 r24_M2_multi_f4_lr2_2500 r24_M3_multi_f4_lr2_5000 r24_M4_multi_f4_lr1_10000; do
  decode $n
done
stamp "all done"
