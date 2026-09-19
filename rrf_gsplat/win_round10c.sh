#!/bin/bash
# Round 10c: fallback for 10b with the datasets generated one process each
# again. The batched generator (--tx-list) ran slower per dataset (L 280 s,
# O 368 s against 222 s for E in its own process) while the card showed
# 12.0 GB in use -- but a game (Against the Storm) was running on the same
# GPU at the time, so whether the long-lived Sionna/Dr.Jit process leaks
# across transmitters or the game took the card cannot be separated. The
# per-process path is the known-good one; the 1 s scene build it repeats
# is nothing. Missing datasets are generated per process, then
# renormalised; existing raw datasets (L, O) are only renormalised.
#   bash rrf_gsplat/win_round10c.sh [ITERS]
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
LOG=$REPO/output/rrf/win_round10c.log
cd $REPO
stamp() { echo "== $1 $(date +%H:%M:%S)" >> $LOG; }
stamp "start (transfer budget $ITERS)"
run() { local name=$1; shift; [ -f output/rrf/$name/results.json ] && { echo "skip $name" >> $LOG; return; }
        stamp "$name"; $WSL "$name" "$@" >> $LOG 2>&1 < /dev/null; stamp "done $name"; }

NAMES="L O H M K"
coords() { case $1 in L) echo "3.5 -7.5 0.8";; O) echo "2.0 -3.0 0.8";; H) echo "1.98 -9.27 0.8";; M) echo "-1.0 -7.0 0.8";; K) echo "-3.35 0.0 0.8";; esac; }
for N in $NAMES; do
  [ -d $REG/3dgs_MVDR_tx${N}_gpct ] && continue
  if [ ! -f $REG/3dgs_MVDR_tx$N/generation_meta.json ]; then
    rm -rf $REG/3dgs_MVDR_tx$N
    stamp "dataset $N ($(coords $N))"
    cd $REPO/sionna_port
    $PYS generate_dataset.py --scene-xml $SCENE --rx-loc-file $RXLOC --out-dir ../$REG/3dgs_MVDR_tx$N --spectrum MVDR \
        --num-positions 800 --tx $(coords $N) --no-dashboard < /dev/null 2>&1 | grep -a "views/s\|Error\|Traceback" >> $LOG
    cd $REPO
  fi
  stamp "renormalize $N"
  $PYS rrf_gsplat/renormalize.py $REG/3dgs_MVDR_tx$N $REG/3dgs_MVDR_tx${N}_gpct --norm global-pct --pct 1 99.99 < /dev/null 2>&1 | tail -1 >> $LOG
  cp $REG/3dgs_MVDR_100_gpct/train_index.txt $REG/3dgs_MVDR_100_gpct/test_index.txt $REG/3dgs_MVDR_tx${N}_gpct/
done
stamp "datasets done"

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
