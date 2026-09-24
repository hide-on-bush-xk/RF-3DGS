#!/bin/bash
# M3 placement oracle, full (smoke: win_m3_smoke.sh; Ke accepted its two as-written failures, index and frame).
# The capacity benchmark of stage 2 (160 positions = the route subset of protocol train, 3000 steps, power mode,
# scored on those training views), with emitters at the ray tracer's interaction points of the same 160 positions
# (path_emitters.py, 5 cm voxels, 90 % of each face sector's power, <= 8000 per position), added in linear power.
# Reading, written before the run (primary metric: <= 1 deg on the distinct views; without emitters 29.1 %):
#   >= 50 %  placement is the lever -> P4 proper (learn or supervise where energy is placed)
#   <= 30 %  even the true placement does not rescue the peaks with a shared, SH3-coloured representation ->
#            a per-position component (per-position latent, or an explicit path set)
#   between  both matter
# Caveat: the emitters are shared by all positions and coloured by SH3; this tests "true placement + shared SH3
# colour", not placement alone.
set -u
REPO=/c/Users/Ke/Documents/GitHub/RF-3DGS
PYW=/c/Users/Ke/miniconda3/envs/rf-gsplat-win/python.exe
PYS=/c/Users/Ke/miniconda3/envs/rf-sionna-win/python.exe
APG=RF-3DGS_dataset/regenerated/3dgs_APS_60_gpct
EM=output/rrf/m3/emitters_160.npz
export PYTHONUTF8=1
LOG=$REPO/output/rrf/m3/win_m3_oracle.log
cd $REPO
echo "== m3 oracle start $(date +%H:%M:%S)" >> $LOG
if [ ! -f $EM ]; then
  $PYS sionna_port/path_emitters.py --truth RF-3DGS_dataset/regenerated/3dgs_APS_60 --protocol rrf_gsplat/protocol_v1 \
      --capacity 160 --cap 8000 --out $EM > output/rrf/m3/emitters_160.log 2>&1
fi
grep -a -E '"emitters"|kept_per|captured_share_median|"median"|within|distinct|PASS|FAIL|seconds' output/rrf/m3/emitters_160.log >> $LOG
name=m3_fit_160_power_em
if [ ! -f output/rrf/m3/$name/results.json ]; then
  mkdir -p output/rrf/m3/$name
  $PYW rrf_gsplat/train_rrf.py --out output/rrf/m3/$name --mode power --source $APG --protocol rrf_gsplat/protocol_v1 --eval-set val \
      --sh-backend gsplat --eval-group --faces-per-step 4 --lr-scale 2 --sh-degree 3 --seed 0 --iterations 3000 --eval-every 3000 \
      --max-train-views 640 --save-renders 0 --emitters $EM > output/rrf/m3/$name/train.log 2>&1
fi
echo "$name: $(grep -a 'iterations in' output/rrf/m3/$name/train.log | tail -1 | cut -c1-160)" >> $LOG
$PYW rrf_gsplat/diag_train_fit.py --run output/rrf/m3/$name --train-subset 640 >> $LOG 2>&1
echo "(without emitters, s2_aps_fit_160_power: 26.8 % all views, 29.1 % distinct; gate 50 / 30 %)" >> $LOG
echo "== all done $(date +%H:%M:%S)" >> $LOG
