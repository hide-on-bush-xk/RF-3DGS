#!/bin/bash
# Round 48: is the radio radiance field's peak failure capacity or schedule? (task 6 groundwork; diag_train_fit.py)
# Round 47 (T4): the nearest training position's truth beats the field on held-out main-peak direction, and the field
# is as bad on its own training views as on held-out ones (r45_base: train <= 1 deg 17.8 %, test 18.4 %): underfit.
# Done before this script was written (seed 0, SH3 db, 3000 steps, route subsets; scored on their training views):
#   r48_overfit_4    1 position   train RMSE 0.14 dB, <= 1 deg 50 % (2 of 4 faces; flat-topped peaks), top-3 100 %
#   r48_overfit_64   16 positions train RMSE 2.27 dB, <= 1 deg 59 %, at true peak -4.65 dB, top-3 94.5 %
# The 640-position run sees every view ~4 times (2500 steps x 4 faces / 2560 views), the 16-position one ~190 times.
# This round separates the two:
#   r48_fit_160_3k    160 positions (640 views), 3000 steps  (~19 passes)
#   r48_fit_640_20k   all 640 positions, 20000 steps          (~31 passes)
# Reading, written before the runs: if r48_fit_640_20k's training-view <= 1 deg stays below 30 % (r45: 18 %),
# the field is capacity-limited (per-Gaussian view dependence, M2 in docs/tech_paths.md) and a longer schedule is
# not the fix; if it reaches >= 50 %, the schedule (optimisation) was the limit. r48_fit_160_3k places the knee.
# The time of these runs is not a measurement (no quiet-GPU gate).
set -u
REPO=/c/Users/Ke/Documents/GitHub/RF-3DGS
PYW=/c/Users/Ke/miniconda3/envs/rf-gsplat-win/python.exe
MV=RF-3DGS_dataset/regenerated/3dgs_MVDR_100_gpct
export PYTHONUTF8=1
LOG=$REPO/output/rrf/win_round48.log
cd $REPO
B="--mode db --source $MV --sh-backend gsplat --eval-group --faces-per-step 4 --lr-scale 2 --eval-every 2500 --save-renders -1 --sh-degree 3 --seed 0"
run() {   # name, subset views (0 = all), iterations
  local name=$1 sub=$2 its=$3
  if [ ! -f output/rrf/$name/results.json ]; then
    echo "== $name $(date +%H:%M:%S)" >> $LOG
    mkdir -p output/rrf/$name
    if [ "$sub" = 0 ]; then $PYW rrf_gsplat/train_rrf.py --out output/rrf/$name $B --iterations $its > output/rrf/$name/train.log 2>&1
    else $PYW rrf_gsplat/train_rrf.py --out output/rrf/$name $B --iterations $its --max-train-views $sub > output/rrf/$name/train.log 2>&1; fi
    echo "   $(tail -1 output/rrf/$name/train.log | cut -c1-110)" >> $LOG
  fi
  if [ "$sub" = 0 ]; then $PYW rrf_gsplat/diag_train_fit.py --run output/rrf/$name --positions 160 >> $LOG 2>&1
  else $PYW rrf_gsplat/diag_train_fit.py --run output/rrf/$name --train-subset $sub >> $LOG 2>&1; fi
}
run r48_fit_160_3k 640 3000
run r48_fit_640_20k 0 20000
echo "== all done $(date +%H:%M:%S)" >> $LOG
