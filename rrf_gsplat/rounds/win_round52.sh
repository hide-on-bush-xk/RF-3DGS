#!/bin/bash
# Round 52: P2 -- spherical-Gaussian lobes on top of SH3 -- on round 48's capacity benchmark (160 positions, route
# subset, 3000 steps, scored on the field's own training views by diag_train_fit.py). Smoke: check_lobes.py passed
# (renders bit-identical at initialisation, gradients to every lobe parameter, loss falls, step 1.12 x SH3).
# References on the same benchmark: SH3 19.2 % <= 1 deg (4.95 deg median), SH4 23.9 % (3.74 deg) (round 51).
#   r52_fit_160_l1_k20    SH3 + 1 lobe, initial kappa 20 (width ~13 deg)
#   r52_fit_160_l2_k20    SH3 + 2 lobes
#   r52_fit_160_l1_k100   SH3 + 1 lobe, initial kappa 100 (~6 deg)
# Reading, written before the runs: lobes are worth pursuing if the best of these reaches >= 28.9 % (SH4 + 5
# points) on the training views; at or below SH4's 23.9 % they add nothing beyond one more SH band. One seed each.
set -u
REPO=/c/Users/Ke/Documents/GitHub/RF-3DGS
PYW=/c/Users/Ke/miniconda3/envs/rf-gsplat-win/python.exe
MV=RF-3DGS_dataset/regenerated/3dgs_MVDR_100_gpct
export PYTHONUTF8=1
LOG=$REPO/output/rrf/win_round52.log
cd $REPO
B="--mode db --source $MV --sh-backend gsplat --eval-group --faces-per-step 4 --lr-scale 2 --iterations 3000 --eval-every 3000 --save-renders 0 --seed 0 --max-train-views 640 --sh-degree 3"
echo "== round 52 start $(date +%H:%M:%S)" >> $LOG
run() {   # name, extra args
  local name=$1; shift
  if [ ! -f output/rrf/$name/results.json ]; then
    mkdir -p output/rrf/$name
    $PYW rrf_gsplat/train_rrf.py --out output/rrf/$name $B "$@" > output/rrf/$name/train.log 2>&1
  fi
  echo "$name: $(tail -1 output/rrf/$name/train.log | cut -c1-100)" >> $LOG
  $PYW rrf_gsplat/diag_train_fit.py --run output/rrf/$name --train-subset 640 >> $LOG 2>&1
}
run r52_fit_160_l1_k20 --lobes 1 --lobe-kappa 20
run r52_fit_160_l2_k20 --lobes 2 --lobe-kappa 20
run r52_fit_160_l1_k100 --lobes 1 --lobe-kappa 100
echo "(SH3 r48_fit_160_3k: 19.2 %, 4.95 deg; SH4 r51_fit_160_sh4: 23.9 %, 3.74 deg)" >> $LOG
echo "== all done $(date +%H:%M:%S)" >> $LOG
