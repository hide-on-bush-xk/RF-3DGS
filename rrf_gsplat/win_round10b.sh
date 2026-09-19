#!/bin/bash
# Round 10b: the remaining transfer-curve sources, restructured after Ke's
# stage accounting (a point = dataset + adaptation + transfer, not one run):
#   * all remaining datasets in ONE generator process (--tx-list: scene,
#     OptiX structure and array grid built once), before any training, so
#     the GPU does generation then training instead of alternating;
#   * adaptation runs with --no-eval (their product is the geometry; kept at
#     10k steps for parity with A, C, D, E, N);
#   * transfers at the calibrated budget ITERS (2k if the 2k-vs-10k
#     calibration in round 10a keeps the signs and order of cold/A/C/D,
#     else 10k), seed 0, then transfer_curve.py after each point.
# Every stage is time-stamped; the GPU is shared with nothing once 10a ends.
#   bash rrf_gsplat/win_round10b.sh [ITERS]
set -u
ITERS=${1:-2000}
TAG=$((ITERS / 1000))k
REPO=/c/Users/Ke/Documents/GitHub/RF-3DGS
PYS=/c/Users/Ke/miniconda3/envs/rf-sionna-win/python.exe
SCENE=../sionna_tutorial/RF-3DGS_Sionna_simulation_tutorial/NIST_lobby_v1.0/NIST_lobby_V1.1_sionna12.xml
RXLOC=../sionna_tutorial/RF-3DGS_Sionna_simulation_tutorial/NIST_rx_loc.txt
REG=RF-3DGS_dataset/regenerated
WSL="wsl.exe -d Ubuntu-22.04 -u ke -- bash /mnt/c/Users/Ke/Documents/GitHub/RF-3DGS/rrf_gsplat/wsl_run.sh"
OUT=/mnt/c/Users/Ke/Documents/GitHub/RF-3DGS/output/rrf
export PYTHONUTF8=1 MSYS_NO_PATHCONV=1
LOG=$REPO/output/rrf/win_round10b.log
cd $REPO
stamp() { echo "== $1 $(date +%H:%M:%S)" >> $LOG; }
stamp "start (transfer budget $ITERS)"
for i in $(seq 1 720); do grep -q "all done" output/rrf/win_round10a.log 2>/dev/null && break; sleep 5; done
stamp "10a finished, gpu free"
run() { local name=$1; shift; [ -f output/rrf/$name/results.json ] && { echo "skip $name" >> $LOG; return; }
        stamp "$name"; $WSL "$name" "$@" >> $LOG 2>&1 < /dev/null; stamp "done $name"; }

# 1. datasets, one process. name:x,y,z at 0.8 m; distances to Tx-B: L 5.3, O 6.8, H 7.4, M 9.4, K 12.8 m
NAMES="L O H M K"
LIST="L:3.5,-7.5,0.8 O:2.0,-3.0,0.8 H:1.98,-9.27,0.8 M:-1.0,-7.0,0.8 K:-3.35,0.0,0.8"
MISSING=""; for N in $NAMES; do [ -d $REG/3dgs_MVDR_tx${N}_gpct ] || MISSING="$MISSING $(echo $LIST | tr ' ' '\n' | grep "^$N:")"; done
if [ -n "$MISSING" ]; then
  stamp "datasets (tx-list:$MISSING)"
  cd $REPO/sionna_port
  $PYS generate_dataset.py --scene-xml $SCENE --rx-loc-file $RXLOC --out-dir ../$REG/3dgs_MVDR_tx --spectrum MVDR \
      --num-positions 800 --tx-list $MISSING --no-dashboard < /dev/null 2>&1 | grep -a "built once\|tx-list\|views/s\|Error\|Traceback" >> $LOG
  cd $REPO
  for N in $NAMES; do
    [ -d $REG/3dgs_MVDR_tx${N}_gpct ] && continue
    stamp "renormalize $N"
    $PYS rrf_gsplat/renormalize.py $REG/3dgs_MVDR_tx$N $REG/3dgs_MVDR_tx${N}_gpct --norm global-pct --pct 1 99.99 < /dev/null 2>&1 | tail -1 >> $LOG
    cp $REG/3dgs_MVDR_100_gpct/train_index.txt $REG/3dgs_MVDR_100_gpct/test_index.txt $REG/3dgs_MVDR_tx${N}_gpct/
  done
  stamp "datasets done"
fi

# 2. references at this budget, Tx-N's transfer (its adaptation ran in round 9b), then the five new sources
TR="--source $REG/3dgs_MVDR_txB_gpct --mode db --iterations $ITERS --eval-every 500 --save-renders 0 --seed 0"
run t_txB_geom_${TAG} $TR --train-geometry
run t_txB_geomN_frozen_${TAG} $TR --init-from $OUT/a_txN_db_geom/rrf_state.pt --init-geometry-only
run t_txB_geomE_frozen_${TAG} $TR --init-from $OUT/a_txE_db_geom/rrf_state.pt --init-geometry-only
for N in $NAMES; do
  run a_tx${N}_db_geom --source $REG/3dgs_MVDR_tx${N}_gpct --mode db --train-geometry --no-eval --save-renders 0
  run t_txB_geom${N}_frozen_${TAG} $TR --init-from $OUT/a_tx${N}_db_geom/rrf_state.pt --init-geometry-only
  $PYS rrf_gsplat/transfer_curve.py --tag $TAG < /dev/null >> $LOG 2>&1
done
stamp "all done"
