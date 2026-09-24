#!/bin/bash
# M3 placement oracle, round 2 (smoke: win_m3_smoke.sh smoke_em_pc8 + check_emitters.py --em-pcolor 8, all passed;
# Ke: "开始", 2026-09-24). The capacity benchmark (160 positions, 3000 steps, power mode, scored on those training
# views), 3 seeds per arm:
#   A  emitters (output/rrf/m3/emitters_160.npz) + --em-pcolor 8: true placement + a position-conditioned switch
#   B  no emitters, --pcolor 8 on the visual Gaussians: the position-conditioned colour alone
#   (round 1: plain 25.1 %, emitters alone 27.1 %)
# Reading, written before the run (docs/protocol_v1_guide.md 8.8; primary: distinct <= 1 deg, 3-seed mean):
#   A >= 50 % and B < 30 %  -> the recipe is placement + per-position switching; next: learn the placement (P4)
#   A >= 50 % and B >= 50 % -> the position-conditioned colour alone suffices; P7-type models
#   A <= 30 %               -> not enough either; an explicit path set
#   A in between            -> both matter; read against round 1 and B
set -u
REPO=/c/Users/Ke/Documents/GitHub/RF-3DGS
PYW=/c/Users/Ke/miniconda3/envs/rf-gsplat-win/python.exe
APG=RF-3DGS_dataset/regenerated/3dgs_APS_60_gpct
EM=output/rrf/m3/emitters_160.npz
export PYTHONUTF8=1
LOG=$REPO/output/rrf/m3/win_m3_round2.log
cd $REPO
echo "== m3 round 2 start $(date +%H:%M:%S)" >> $LOG
B="--mode power --source $APG --protocol rrf_gsplat/protocol_v1 --eval-set val --sh-backend gsplat --eval-group --faces-per-step 4 \
--lr-scale 2 --sh-degree 3 --iterations 3000 --eval-every 3000 --max-train-views 640 --save-renders 0"
for s in 0 1 2; do
  for arm in A B; do
    if [ $arm = A ]; then name=m3r2_em_pc8_s$s; extra="--emitters $EM --em-pcolor 8"
    else name=m3r2_pc8_s$s; extra="--pcolor 8"; fi
    if [ ! -f output/rrf/m3/$name/results.json ]; then
      mkdir -p output/rrf/m3/$name
      $PYW rrf_gsplat/train_rrf.py --out output/rrf/m3/$name $B --seed $s $extra > output/rrf/m3/$name/train.log 2>&1
    fi
    echo "$name: $(grep -a 'iterations in' output/rrf/m3/$name/train.log | tail -1 | cut -c1-120)" >> $LOG
    $PYW rrf_gsplat/diag_train_fit.py --run output/rrf/m3/$name --train-subset 640 >> $LOG 2>&1
  done
done
echo "== all done $(date +%H:%M:%S)" >> $LOG
