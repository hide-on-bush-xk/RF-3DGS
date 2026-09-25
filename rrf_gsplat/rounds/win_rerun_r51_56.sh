#!/bin/bash
# Re-run of rounds 51-56's key representations under corrected conditions (Ke, 2026-09-24: the order agreed --
# guards, then this, then the generalisation diagnostics G1 / G2). Rounds 51-56 were judged on clipped targets,
# at 3000 steps (~19 visits per view) and mostly one seed; each of the three is now removed:
#   data       3dgs_APS_60_gp100 (the exact range: 0 clipped views; the clip guard would refuse otherwise)
#   budget     at most 250 visits per view (40,000 steps on the 640-view capacity subset), early stopping on the fixed
#              training subset (--early-stop-on train): 10 evaluations (10,000 steps) without distinct <= 1 deg
#              rising 2 points; at least 5,000 steps
#   seeds      0 1 2
#   arms       plain SH3 / SH4 / P7 (--pcolor 8) / one sharp lobe (--lobes 1 --lobe-kappa 100, last, if time allows)
# Capacity benchmark as always: 160 positions (route subset of protocol train), power mode, scored on those training
# views by diag_train_fit.py (distinct <= 1 deg the primary). The validation subset's curves are recorded beside.
# Reading, written before the run: an arm helps if its 3-seed mean beats plain SH3's by more than both arms' seed
# ranges (max - min); otherwise "no difference". If P7 helps, round 55's "P7 fits pixels, not peaks" is overturned
# (it was clipped, 3000 steps, one seed). Reference, not an arm: oracle emitters + switch 61.7 % (40k steps, unclipped).
set -u
REPO=/c/Users/Ke/Documents/GitHub/RF-3DGS
PYW=/c/Users/Ke/miniconda3/envs/rf-gsplat-win/python.exe
APU=RF-3DGS_dataset/regenerated/3dgs_APS_60_gp100
export PYTHONUTF8=1
LOG=$REPO/output/rrf/m3/win_rerun_r51_56.log
cd $REPO
echo "== rerun r51-56 start $(date +%H:%M:%S)" >> $LOG
B="--mode power --source $APU --protocol rrf_gsplat/protocol_v1 --eval-set val --sh-backend gsplat --eval-group \
--faces-per-step 4 --lr-scale 2 --seed-placeholder --max-train-views 640 --save-renders 0 --visits-per-view 250 \
--early-stop-on train --early-stop-patience 10 --early-stop-min-steps 5000 --eval-every 1000000"
run() {   # name, description, extra args...
  local name=$1 desc=$2 seed=$3; shift 3
  local dir=output/rrf/m3/rr/$name
  if [ ! -f $dir/results.json ]; then
    mkdir -p $dir/live
    echo "$desc（种子 $seed）。rounds 51–56 在修正条件下重跑：不截峰、最多每张视图 250 次（早停看训练视图，1 万步无 2 个点提升就停）、容量基准 160 个位置。" > $dir/live/desc.txt
    $PYW rrf_gsplat/train_rrf.py --out $dir ${B/--seed-placeholder/--seed $seed} "$@" > $dir/train.log 2>&1
  fi
  echo "$name: $(grep -a 'iterations in' $dir/train.log | tail -1 | cut -c1-200)" >> $LOG
  $PYW rrf_gsplat/diag_train_fit.py --run $dir --train-subset 640 >> $LOG 2>&1
}
for s in 0 1 2; do
  run plain_s$s "普通场 SH3" $s
  run sh4_s$s "SH4（更高阶的球谐颜色）" $s --sh-degree 4
  run pc8_s$s "P7：随接收位置变化的颜色（pcolor 8）" $s --pcolor 8
done
for s in 0 1 2; do
  run lobe100_s$s "一个锐瓣（kappa 100，约 6 度）" $s --lobes 1 --lobe-kappa 100
done
echo "== all done $(date +%H:%M:%S)" >> $LOG
