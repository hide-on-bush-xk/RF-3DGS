#!/bin/bash
# The visits-per-view control (Ke: "核对一下", 2026-09-24). Every capacity comparison so far ran 3000 steps x 4 faces:
# 640 training views (160 positions) get ~19 visits each, the 12-position smoke's 48 views ~250. "Helps at 12
# positions, not at 160" (M3 rounds 1-2, and rounds 51-56 before them) may be capacity or just too few visits.
# Here: the 160-position capacity benchmark at 40,000 steps (~250 visits per view, as the smoke), seed 0, power mode,
# constant learning rates (train_rrf has no decay with --densify none), everything else unchanged:
#   plain   no emitters (at 3000 steps: 25.1 % distinct <= 1 deg, 3-seed mean; seed 0 29.1 %)
#   A       emitters + --em-pcolor 8 (at 3000 steps: 25.9 %; seed 0 33.4 %)
# Reading, written before the run (primary: distinct <= 1 deg on the 639 training views):
#   A >= 50 %   the training budget, not the representation, was limiting: the "explicit path set" reading of round
#               2 is withdrawn, and the capacity conclusions of rounds 51-56 must be re-read
#   A <= 30 %   the round-2 reading stands: an explicit path set
#   between     more seeds before reading
#   plain       the control: if it also reaches >= 50 %, the night's "capacity" was mostly visits
# One seed detects only a large effect (the seed spread at 3000 steps is 20-33 %); a large one is what is asked.
set -u
REPO=/c/Users/Ke/Documents/GitHub/RF-3DGS
PYW=/c/Users/Ke/miniconda3/envs/rf-gsplat-win/python.exe
APG=RF-3DGS_dataset/regenerated/3dgs_APS_60_gpct
EM=output/rrf/m3/emitters_160.npz
export PYTHONUTF8=1
LOG=$REPO/output/rrf/m3/win_m3_visits.log
cd $REPO
echo "== m3 visits start $(date +%H:%M:%S)" >> $LOG
B="--mode power --source $APG --protocol rrf_gsplat/protocol_v1 --eval-set val --sh-backend gsplat --eval-group --faces-per-step 4 \
--lr-scale 2 --sh-degree 3 --seed 0 --iterations 40000 --eval-every 40000 --max-train-views 640 --save-renders 0"
run() {
  local name=$1; shift
  if [ ! -f output/rrf/m3/$name/results.json ]; then
    mkdir -p output/rrf/m3/$name
    $PYW rrf_gsplat/train_rrf.py --out output/rrf/m3/$name $B "$@" > output/rrf/m3/$name/train.log 2>&1
  fi
  echo "$name: $(grep -a 'iterations in' output/rrf/m3/$name/train.log | tail -1 | cut -c1-120)" >> $LOG
  $PYW rrf_gsplat/diag_train_fit.py --run output/rrf/m3/$name --train-subset 640 >> $LOG 2>&1
}
run m3v_em_pc8_40k --emitters $EM --em-pcolor 8
run m3v_plain_40k
# seed 0 gave A 41.4 %, plain 28.2 %: A in the 30-50 % band -> by the reading above, more seeds before reading
# (plain too, for the paired comparison)
for s in 1 2; do
  B2="${B/--seed 0/--seed $s}"
  for arm in em_pc8 plain; do
    name=m3v_${arm}_40k_s$s
    extra=""; [ $arm = em_pc8 ] && extra="--emitters $EM --em-pcolor 8"
    if [ ! -f output/rrf/m3/$name/results.json ]; then
      mkdir -p output/rrf/m3/$name
      $PYW rrf_gsplat/train_rrf.py --out output/rrf/m3/$name $B2 $extra > output/rrf/m3/$name/train.log 2>&1
    fi
    echo "$name: $(grep -a 'iterations in' output/rrf/m3/$name/train.log | tail -1 | cut -c1-120)" >> $LOG
    $PYW rrf_gsplat/diag_train_fit.py --run output/rrf/m3/$name --train-subset 640 >> $LOG 2>&1
  done
done
echo "== all done $(date +%H:%M:%S)" >> $LOG
