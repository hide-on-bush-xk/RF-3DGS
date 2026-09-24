#!/bin/bash
# Round 51: does angular bandwidth per Gaussian lift the capacity limit? (M2 / P2 in docs/tech_paths.md)
# Round 48's capacity benchmark: train on 160 positions (route subset, 640 views), 3000 steps, and score the field
# on its OWN training views (diag_train_fit.py). SH3 (r48_fit_160_3k, the current defaults): <= 1 deg 19.2 %,
# main-peak median 4.95 deg. This adds SH degree 1, 2 and 4 (gsplat's CUDA SH goes to 4: 25 coefficients,
# band limit about 36 deg instead of SH3's 45) on the same benchmark, seed 0.
# Reading, written before the runs: if SH4's training-view <= 1 deg share exceeds SH3's by more than 5 points and
# SH1 -> SH4 rises monotonically, angular bandwidth is (part of) the limit and sharper lobes (P2) are worth
# building; if SH1..SH4 stay within +-3 points of each other, bandwidth is not the limiter and the next candidates
# are a position-conditioned colour (P7) or the target (P1).
set -u
REPO=/c/Users/Ke/Documents/GitHub/RF-3DGS
PYW=/c/Users/Ke/miniconda3/envs/rf-gsplat-win/python.exe
MV=RF-3DGS_dataset/regenerated/3dgs_MVDR_100_gpct
export PYTHONUTF8=1
LOG=$REPO/output/rrf/win_round51.log
cd $REPO
B="--mode db --source $MV --sh-backend gsplat --eval-group --faces-per-step 4 --lr-scale 2 --iterations 3000 --eval-every 3000 --save-renders 0 --seed 0 --max-train-views 640"
echo "== round 51 start $(date +%H:%M:%S)" >> $LOG
for d in 1 2 4; do
  name=r51_fit_160_sh$d
  if [ ! -f output/rrf/$name/results.json ]; then
    mkdir -p output/rrf/$name
    $PYW rrf_gsplat/train_rrf.py --out output/rrf/$name $B --sh-degree $d > output/rrf/$name/train.log 2>&1
  fi
  echo "$name: $(tail -1 output/rrf/$name/train.log | cut -c1-100)" >> $LOG
  $PYW rrf_gsplat/diag_train_fit.py --run output/rrf/$name --train-subset 640 >> $LOG 2>&1
done
echo "(SH3 = r48_fit_160_3k: train <= 1 deg 19.2 %, main peak 4.95 deg, PSNR 19.73, RMSE 3.49 dB)" >> $LOG
echo "== all done $(date +%H:%M:%S)" >> $LOG
