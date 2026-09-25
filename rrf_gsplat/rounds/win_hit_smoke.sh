#!/bin/bash
# S-C of the hit-distance smoke (smoke_hit_distance.py has S-A / S-B): a short delay-channel training on the lobby
# MULTI dataset, the range term as euclid (ED x sec theta, the method's default since round 19) against hit (gsplat's
# native along-ray hit distance, --delay-range hit), everything else equal: ED, seed 0, 2500 steps, round 26's fast
# recipe, all 640 held-out views rendered and decoded by eval_baselines.py. All five channels are trained (multi).
# Pass criteria, written before the run (a smoke checks the path; it cannot rank the two: delay P90 / RMSE are tail
# metrics):
#   1. both runs finish; the delay RMSE in the training history is finite at every evaluation
#   2. expected range for euclid at 2500 steps: decoded delay median 0.5 - 1.5 ns (10k steps gave 0.55)
#   3. hit's decoded delay median within +-30 % of euclid's
#   4. angles untouched by the range term: |d azimuth median| and |d zenith median| <= 0.05 deg
#   5. speed reported: hit adds one eval3d rasterisation per step; its it/s against euclid's, void when the GPU is shared
set -u
REPO=/c/Users/Ke/Documents/GitHub/RF-3DGS
PYW=/c/Users/Ke/miniconda3/envs/rf-gsplat-win/python.exe
PYS=/c/Users/Ke/miniconda3/envs/rf-sionna-win/python.exe
MUL=RF-3DGS_dataset/regenerated/3dgs_MULTI_24ghz_tut_cs
export PYTHONUTF8=1
LOG=$REPO/output/rrf/win_hit_smoke.log
cd $REPO
stamp() { echo "== $1 $(date +%H:%M:%S) gpu $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader | tr -d '\r')" >> $LOG; }
M="--source $MUL --mode multi --delay-depth --delay-depth-mode ED --save-renders -1 --sh-backend gsplat --eval-group \
--faces-per-step 4 --lr-scale 2 --iterations 2500 --eval-every 250 --seed 0"
stamp "hit smoke start"
for R in euclid hit; do
  d=output/rrf/hit_smoke/$R
  if [ ! -f $d/results.json ]; then
    mkdir -p $d/live
    echo "时延范围项 $R（S-C 冒烟，2500 步，MULTI 大堂）" > $d/live/desc.txt
    stamp "train $R"
    $PYW rrf_gsplat/train_rrf.py --out $d $M --delay-range $R > $d/train.log 2>&1
  fi
  echo "$R: $(grep -a 'iterations in' $d/train.log | tail -1 | cut -c1-200)" >> $LOG
  [ -f output/rrf/hit_smoke/baselines_$R.json ] || \
    $PYS rrf_gsplat/eval_baselines.py --multi $d --truth $MUL --out output/rrf/hit_smoke/baselines_$R.json < /dev/null 2>&1 | grep -v jitc | tail -8 >> $LOG
done
stamp "all done"
