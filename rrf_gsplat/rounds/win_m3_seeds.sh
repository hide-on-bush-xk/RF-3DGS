#!/bin/bash
# M3 placement oracle, seeds 1 and 2 (seed 0: win_m3_oracle.sh and s2_aps_fit_160_power). Seed 0 came out at
# 30.4 % (distinct) with emitters vs 29.1 % without -- 0.4 points into the 30-50 % band of the reading written before
# the run and about one seed's spread from the baseline. The band is not moved; the seeds decide which side the
# mean is on and whether the difference is larger than the seed spread. Same configuration, --seed 1 / 2.
set -u
REPO=/c/Users/Ke/Documents/GitHub/RF-3DGS
PYW=/c/Users/Ke/miniconda3/envs/rf-gsplat-win/python.exe
APG=RF-3DGS_dataset/regenerated/3dgs_APS_60_gpct
EM=output/rrf/m3/emitters_160.npz
export PYTHONUTF8=1
LOG=$REPO/output/rrf/m3/win_m3_seeds.log
cd $REPO
echo "== m3 seeds start $(date +%H:%M:%S)" >> $LOG
B="--mode power --source $APG --protocol rrf_gsplat/protocol_v1 --eval-set val --sh-backend gsplat --eval-group --faces-per-step 4 \
--lr-scale 2 --sh-degree 3 --iterations 3000 --eval-every 3000 --max-train-views 640 --save-renders 0"
for s in 1 2; do
  for arm in plain em; do
    name=m3_fit_160_power_${arm}_s$s
    extra=""; [ $arm = em ] && extra="--emitters $EM"
    if [ ! -f output/rrf/m3/$name/results.json ]; then
      mkdir -p output/rrf/m3/$name
      $PYW rrf_gsplat/train_rrf.py --out output/rrf/m3/$name $B --seed $s $extra > output/rrf/m3/$name/train.log 2>&1
    fi
    echo "$name: $(grep -a 'iterations in' output/rrf/m3/$name/train.log | tail -1 | cut -c1-120)" >> $LOG
    $PYW rrf_gsplat/diag_train_fit.py --run output/rrf/m3/$name --train-subset 640 >> $LOG 2>&1
  done
done
echo "== all done $(date +%H:%M:%S)" >> $LOG
