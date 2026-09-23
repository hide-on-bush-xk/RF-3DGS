#!/bin/bash
# Round 30 (after round 29): the pieces the round-27 comparison still lacks.
#   E1     the live chain with the engineering changes only (CUDA SH, grouped evaluation, one view per step,
#          2000 steps), so the old -> new speed-up splits into engineering and method
#   peaks  mvdr_peaks.py on the full 800-position db models of round 24 (B0 and M2), to see whether the missing
#          MVDR peaks of the live runs are the 2k-step budget or the model
#   e2e    interactive.py's "Retrain RRF here" clicked through its own HTTP API with the new default (4 faces)
set -u
REPO=/c/Users/Ke/Documents/GitHub/RF-3DGS
PYS=/c/Users/Ke/miniconda3/envs/rf-sionna-win/python.exe
WSL="wsl.exe -d Ubuntu-22.04 -u ke -- bash /mnt/c/Users/Ke/Documents/GitHub/RF-3DGS/rrf_gsplat/wsl_run.sh"
REG=RF-3DGS_dataset/regenerated
export PYTHONUTF8=1 MSYS_NO_PATHCONV=1
LOG=$REPO/output/rrf/win_round30.log
cd $REPO
stamp() { echo "== $1 $(date +%H:%M:%S) gpu $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader | tr -d '\r')" >> $LOG; }
until grep -aq "== all done" output/tx_planning/win_round29.log 2>/dev/null; do sleep 20; done
stamp "round 30 start"
T=$REG/r27_live_300_gpct
if [ ! -f output/rrf/r27_live_E1/results.json ]; then
  stamp "start r27_live_E1"; T0=$(date +%s)
  $WSL r27_live_E1 --source $T --mode db --sh-backend gsplat --eval-group --iterations 2000 --eval-every 250 --save-renders -1 >> $LOG 2>&1 < /dev/null
  echo "wall r27_live_E1 $(( $(date +%s) - T0 )) s" >> $LOG
fi
$PYS rrf_gsplat/mvdr_peaks.py --run output/rrf/r27_live_E1 --truth $T >> $LOG 2>&1
for n in r24_B0_db r24_M2_db_f4_lr2_2500; do
  $PYS rrf_gsplat/mvdr_peaks.py --run output/rrf/$n --truth $REG/3dgs_MVDR_100_gpct >> $LOG 2>&1
done
stamp "e2e retrain"
$PYS tx_planning/check_interactive_retrain.py --tx 3.0 -5.0 2.0 --faces 4 < /dev/null 2>&1 | grep -av "jitc\|WARN" >> $LOG
stamp "all done"
