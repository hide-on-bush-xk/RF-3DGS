#!/bin/bash
# Round 15 (after round 14, same gate: no game process, GPU < 15 % for 2 min):
#   1. depth bias of the planner: re-run the refine optimisation at depth 2
#      from the same start, then score both runs' start and final at depths
#      1 / 2 / 3 with 100k samples (tx_planning/depth_check.py);
#   2. conditional clean-data reruns, only if round 14 found the seed effect
#      on the beamformed family (|RMSE change| >= 0.1 dB or |PSNR change| >=
#      0.3 dB against e2_mvdr_db 18.75 / 3.91):
#        e2_mvdr_rgb_posseed                        (db vs rgb gap on clean data)
#        3dgs_MVDR_txB_posseed (+ gpct, Tx-A posseed range) and
#        t_txB_cold_posseed / t_txB_geomA_frozen_posseed / t_txB_geom_posseed
#                                                   (24/76 on clean data)
set -u
REPO=/c/Users/Ke/Documents/GitHub/RF-3DGS
PYS=/c/Users/Ke/miniconda3/envs/rf-sionna-win/python.exe
SCENE=../sionna_tutorial/RF-3DGS_Sionna_simulation_tutorial/NIST_lobby_v1.0/NIST_lobby_V1.1_sionna12.xml
RXLOC=../sionna_tutorial/RF-3DGS_Sionna_simulation_tutorial/NIST_rx_loc.txt
REG=RF-3DGS_dataset/regenerated
WSL="wsl.exe -d Ubuntu-22.04 -u ke -- bash /mnt/c/Users/Ke/Documents/GitHub/RF-3DGS/rrf_gsplat/wsl_run.sh"
OUT=/mnt/c/Users/Ke/Documents/GitHub/RF-3DGS/output/rrf
export PYTHONUTF8=1 MSYS_NO_PATHCONV=1
LOG=$REPO/output/rrf/win_round15.log
cd $REPO
stamp() { echo "== $1 $(date +%H:%M:%S)" >> $LOG; }
gpu_busy() {
  tasklist 2>/dev/null | grep -qi "Against the Storm" && return 0
  u=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr -d ' \r'); [ "${u:-100}" -ge 15 ]
}
stamp "start"
for i in $(seq 1 5760); do grep -q "all done" output/rrf/win_round14.log 2>/dev/null && break; sleep 5; done
quiet=0; while [ $quiet -lt 24 ]; do if gpu_busy; then quiet=0; else quiet=$((quiet+1)); fi; sleep 5; done
stamp "round 14 finished, gpu quiet ($(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader | tr -d '\r'))"
run() { local name=$1; shift; [ -f output/rrf/$name/results.json ] && { echo "skip $name" >> $LOG; return; }
        stamp "$name"; $WSL "$name" "$@" >> $LOG 2>&1 < /dev/null; stamp "done $name"; }

# 1. planner depth bias
cd $REPO/tx_planning
if [ ! -f ../output/tx_planning/optimize_tx_refine_depth2.json ]; then
  stamp "optimize_tx at depth 2 (refine config)"
  $PYS optimize_tx.py --scene-xml $SCENE --init-from ../output/tx_planning/tx_sweep_batched.npz --tx-height 2.0 --rx-step 2.0 \
      --objective coverage --threshold-db -85 --steps 25 --samples 20000 --max-depth 2 \
      --out ../output/tx_planning/optimize_tx_refine_depth2.json < /dev/null 2>&1 | grep -a "step\|starting\|receivers\|Error\|Traceback" | tail -8 >> $LOG
fi
stamp "depth_check"
$PYS depth_check.py --runs ../output/tx_planning/optimize_tx_refine.json ../output/tx_planning/optimize_tx_refine_depth2.json --samples 100000 \
    < /dev/null 2>&1 | grep -v "jitc_llvm\|WARN" >> $LOG
cd $REPO

# 2. conditional clean-data reruns
NEED=$($PYS -c "
import json,os
try:
    a=json.load(open('output/rrf/e2_mvdr_db/results.json'))['final']; b=json.load(open('output/rrf/e2_mvdr_db_posseed/results.json'))['final']
    print(1 if abs(a['rmse_db']-b['rmse_db'])>=0.1 or abs(a['psnr_rgb']-b['psnr_rgb'])>=0.3 else 0)
except Exception as e: print(0)")
stamp "clean-data reruns needed: $NEED"
if [ "$NEED" = "1" ]; then
  run e2_mvdr_rgb_posseed --source $REG/3dgs_MVDR_100_posseed_gpct --mode rgb --save-renders 0
  DS=3dgs_MVDR_txB_posseed
  if [ ! -d $REG/${DS}_gpct ]; then
    stamp "dataset $DS"
    cd $REPO/sionna_port
    $PYS generate_dataset.py --scene-xml $SCENE --rx-loc-file $RXLOC --out-dir ../$REG/$DS --spectrum MVDR --num-positions 800 --tx 8.2 -5.4 2.0 --no-dashboard \
        < /dev/null 2>&1 | grep -a "views/s\|global dB\|Error\|Traceback" >> $LOG
    cd $REPO
    RANGE=$($PYS -c "import json; m=json.load(open('$REG/3dgs_MVDR_100_posseed_gpct/generation_meta.json')); print(m['spec_min_db'], m['spec_max_db'])")
    $PYS rrf_gsplat/renormalize.py $REG/$DS $REG/${DS}_gpct --norm global-pct --range $RANGE < /dev/null 2>&1 | tail -1 >> $LOG
    cp $REG/3dgs_MVDR_100_gpct/train_index.txt $REG/3dgs_MVDR_100_gpct/test_index.txt $REG/${DS}_gpct/
  fi
  run t_txB_cold_posseed --source $REG/${DS}_gpct --mode db --eval-every 250 --save-renders 0
  run t_txB_geomA_frozen_posseed --source $REG/${DS}_gpct --mode db --eval-every 250 --save-renders 0 \
      --init-from $OUT/a_mvdr_db_geom_posseed/rrf_state.pt --init-geometry-only
  run t_txB_geom_posseed --source $REG/${DS}_gpct --mode db --eval-every 250 --save-renders 0 --train-geometry
fi
stamp "all done"
