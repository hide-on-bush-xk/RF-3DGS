#!/bin/bash
# Round 55: P7 -- a position-conditioned colour -- on round 48's capacity benchmark (160 positions, route subset,
# 3000 steps, scored on the training views by diag_train_fit.py). Smoke: check_pcolor.py passed (renders
# bit-identical at initialisation, gradients to the latent and every MLP tensor, loss falls, step 1.65 x SH3).
# The same benchmark so far: SH3 19.2 %, SH4 23.9 %, SH3 + sharp lobe 25.6 % <= 1 deg (MVDR, stored labels).
#   r55_fit_160_pc8    SH3 + latent 8, MLP hidden 32, 4 position frequencies
#   r55_fit_160_pc16   SH3 + latent 16, MLP hidden 64
# Reading, written before the runs: >= 40 % (about twice SH3) -> conditioning a Gaussian's value on the receiver
# position addresses the capacity limit, and held-out against NN is the next test (it could memorise positions);
# <= 26 % (the SH4 / lobe level) -> no better than more angular bandwidth. One seed each.
set -u
REPO=/c/Users/Ke/Documents/GitHub/RF-3DGS
PYW=/c/Users/Ke/miniconda3/envs/rf-gsplat-win/python.exe
MV=RF-3DGS_dataset/regenerated/3dgs_MVDR_100_gpct
export PYTHONUTF8=1
LOG=$REPO/output/rrf/win_round55.log
cd $REPO
B="--mode db --source $MV --sh-backend gsplat --eval-group --faces-per-step 4 --lr-scale 2 --iterations 3000 --eval-every 3000 --save-renders 0 --seed 0 --max-train-views 640 --sh-degree 3"
echo "== round 55 start $(date +%H:%M:%S)" >> $LOG
run() {
  local name=$1; shift
  if [ ! -f output/rrf/$name/results.json ]; then
    mkdir -p output/rrf/$name
    $PYW rrf_gsplat/train_rrf.py --out output/rrf/$name $B "$@" > output/rrf/$name/train.log 2>&1
  fi
  echo "$name: $(tail -1 output/rrf/$name/train.log | cut -c1-110)" >> $LOG
  $PYW rrf_gsplat/diag_train_fit.py --run output/rrf/$name --train-subset 640 >> $LOG 2>&1
}
run r55_fit_160_pc8 --pcolor 8 --pcolor-hidden 32
run r55_fit_160_pc16 --pcolor 16 --pcolor-hidden 64
echo "(SH3 19.2 % / 4.95 deg; SH4 23.9 % / 3.74 deg; SH3 + lobe kappa 100 25.6 % / 3.32 deg)" >> $LOG
echo "== all done $(date +%H:%M:%S)" >> $LOG
