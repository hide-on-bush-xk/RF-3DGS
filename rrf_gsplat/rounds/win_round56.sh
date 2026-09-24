#!/bin/bash
# Round 56: the loss or the geometry? (docs/tech_paths.md 0b, "what this implies")
# Every colour model tried (SH1-4, lobes, CNN head, position-conditioned MLP) raises pixel accuracy and leaves the main
# peak where it was (19-26 % <= 1 deg on the capacity benchmark's training views). Two suspects all of them share:
#   the loss      L1 + SSIM over all pixels; the true peak stays 6-8 dB low in every model
#   the geometry  frozen where the visual texture is; a peak appears where some Gaussian projects
# Same benchmark (160 positions, route subset, 3000 steps, scored on the training views), seed 0, MVDR stored labels:
#   r56_pc8_peak        --pcolor 8 --peak-loss 1            the richest colour model + a peak-weighted loss
#   r56_sh3_peak        --peak-loss 1                        the loss alone
#   r56_geom            --train-geometry                     the geometry alone (plain loss)
#   r56_geom_peak       --train-geometry --peak-loss 1       both
# (--peak-loss 1: an extra L1 on each view's pixels within 10 dB of its maximum, as in round 41.)
# Reading, written before the runs: a lever is a run whose training-view <= 1 deg reaches >= 40 % (about twice
# SH3's 19.2 %); if only the geometry runs get there, the peaks are placed by geometry (M3 -- P3 / P4, energy-placed
# emitters); if only the loss runs, by the objective; if neither, by something all four share.
set -u
REPO=/c/Users/Ke/Documents/GitHub/RF-3DGS
PYW=/c/Users/Ke/miniconda3/envs/rf-gsplat-win/python.exe
MV=RF-3DGS_dataset/regenerated/3dgs_MVDR_100_gpct
export PYTHONUTF8=1
LOG=$REPO/output/rrf/win_round56.log
cd $REPO
B="--mode db --source $MV --sh-backend gsplat --eval-group --faces-per-step 4 --lr-scale 2 --iterations 3000 --eval-every 3000 --save-renders 0 --seed 0 --max-train-views 640 --sh-degree 3"
echo "== round 56 start $(date +%H:%M:%S)" >> $LOG
run() {
  local name=$1; shift
  if [ ! -f output/rrf/$name/results.json ]; then
    mkdir -p output/rrf/$name
    $PYW rrf_gsplat/train_rrf.py --out output/rrf/$name $B "$@" > output/rrf/$name/train.log 2>&1
  fi
  echo "$name: $(tail -1 output/rrf/$name/train.log | cut -c1-110)" >> $LOG
  $PYW rrf_gsplat/diag_train_fit.py --run output/rrf/$name --train-subset 640 >> $LOG 2>&1
}
run r56_pc8_peak --pcolor 8 --peak-loss 1
run r56_sh3_peak --peak-loss 1
run r56_geom --train-geometry
run r56_geom_peak --train-geometry --peak-loss 1
echo "(SH3 19.2 % / 4.95 deg; SH4 23.9 %; lobes 25.6 %; P7 22.0 %)" >> $LOG
echo "== all done $(date +%H:%M:%S)" >> $LOG
