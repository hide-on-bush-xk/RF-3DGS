#!/bin/bash
# Round 38: the four-face step on the six released datasets, so the model browser's wall (viewer.py, which
# shows only runs on the released data, scored by inria_metrics.py) has the new scheme beside the released
# RF-3DGS model and the round-21 rows. Each spectrum gets its best round-21 configuration (A5 db + unfrozen
# geometry; A4 rgb + unfrozen geometry for AoD / Delay, whose PNGs are not jet-encoded) with 2.5k steps x 4
# faces at lr x2 in place of 10k x 1. Then the run index is rebuilt.
set -u
REPO=/c/Users/Ke/Documents/GitHub/RF-3DGS
PYS=/c/Users/Ke/miniconda3/envs/rf-sionna-win/python.exe
WSL="wsl.exe -d Ubuntu-22.04 -u ke -- bash /mnt/c/Users/Ke/Documents/GitHub/RF-3DGS/rrf_gsplat/wsl_run.sh"
export PYTHONUTF8=1 MSYS_NO_PATHCONV=1
LOG=$REPO/output/rrf/win_round38.log
cd $REPO
stamp() { echo "== $1 $(date +%H:%M:%S) gpu $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader | tr -d '\r')" >> $LOG; }
stamp "round 38 start"
F4="--train-geometry --eval-group --faces-per-step 4 --lr-scale 2 --iterations 2500 --eval-every 250 --save-renders -1"
for spec in MVDR:db CBF:db TCBF:db MPC:db AoD:rgb Delay:rgb; do
  S=${spec%%:*}; mode=${spec##*:}; name=sota_${S}_${mode}_geom_f4; DS=RF-3DGS_dataset/training-rf-spectrum/3dgs_${S}_100
  if [ ! -f output/rrf/$name/results.json ]; then
    stamp "start $name"; T0=$(date +%s)
    $WSL $name --source $DS --mode $mode $F4 >> $LOG 2>&1 < /dev/null
    echo "wall $name $(( $(date +%s) - T0 )) s" >> $LOG
  fi
  [ -f output/rrf/$name/inria_metrics.json ] || $PYS rrf_gsplat/inria_metrics.py --run output/rrf/$name --source $DS < /dev/null 2>&1 | tail -1 >> $LOG
done
$PYS rrf_gsplat/index_db.py >> $LOG 2>&1
stamp "all done"
