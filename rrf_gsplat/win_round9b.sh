#!/bin/bash
# Round 9b: the transfer-curve sources round 9 skipped. Round 9 fed its source
# list through a pipe into `while read`, and wsl.exe inside the loop consumed
# the rest of that stdin after the first source (D). Here the list is an array
# and every child gets </dev/null. Waits for round 9's "all done" marker.
set -u
REPO=/c/Users/Ke/Documents/GitHub/RF-3DGS
PYS=/c/Users/Ke/miniconda3/envs/rf-sionna-win/python.exe
SCENE=../sionna_tutorial/RF-3DGS_Sionna_simulation_tutorial/NIST_lobby_v1.0/NIST_lobby_V1.1_sionna12.xml
RXLOC=../sionna_tutorial/RF-3DGS_Sionna_simulation_tutorial/NIST_rx_loc.txt
REG=RF-3DGS_dataset/regenerated
WSL="wsl.exe -d Ubuntu-22.04 -u ke -- bash /mnt/c/Users/Ke/Documents/GitHub/RF-3DGS/rrf_gsplat/wsl_run.sh"
export PYTHONUTF8=1 MSYS_NO_PATHCONV=1
LOG=$REPO/output/rrf/win_round9b.log
cd $REPO
echo "== start $(date)" >> $LOG
for i in $(seq 1 720); do grep -q "all done" output/rrf/win_round9.log 2>/dev/null && break; sleep 5; done
echo "== round 9 finished, gpu free $(date)" >> $LOG
run() { local name=$1; shift; [ -f output/rrf/$name/results.json ] && { echo "skip $name" >> $LOG; return; }
        echo "== $name $(date +%H:%M:%S)" >> $LOG; $WSL "$name" "$@" >> $LOG 2>&1 < /dev/null; }

# name x y z; distances to Tx-B (8.2, -5.4, 2.0): E 3.2, N 4.2, L 5.3, O 6.8, H 7.4, M 9.4, K 12.8 m
SRC=("E 6.9 -2.7 0.8" "N 5.0 -3.0 0.8" "L 3.5 -7.5 0.8" "O 2.0 -3.0 0.8" "H 1.98 -9.27 0.8" "M -1.0 -7.0 0.8" "K -3.35 0.0 0.8")
for entry in "${SRC[@]}"; do
  set -- $entry; NAME=$1; X=$2; Y=$3; Z=$4
  DS=3dgs_MVDR_tx${NAME}
  if [ ! -d $REG/${DS}_gpct ]; then
    echo "== dataset $DS ($X $Y $Z) $(date +%H:%M:%S)" >> $LOG
    cd $REPO/sionna_port
    $PYS generate_dataset.py --scene-xml $SCENE --rx-loc-file $RXLOC --out-dir ../$REG/$DS --spectrum MVDR \
        --num-positions 800 --tx $X $Y $Z --no-dashboard < /dev/null 2>&1 | grep -a "views/s\|global dB\|Error" >> $LOG
    $PYS ../rrf_gsplat/renormalize.py ../$REG/$DS ../$REG/${DS}_gpct --norm global-pct --pct 1 99.99 < /dev/null 2>&1 | tail -1 >> $LOG
    cp ../$REG/3dgs_MVDR_100_gpct/train_index.txt ../$REG/3dgs_MVDR_100_gpct/test_index.txt ../$REG/${DS}_gpct/
    cd $REPO
  fi
  run a_tx${NAME}_db_geom --source $REG/${DS}_gpct --mode db --train-geometry --save-renders 0
  run t_txB_geom${NAME}_frozen --source $REG/3dgs_MVDR_txB_gpct --mode db --eval-every 250 --save-renders 0 \
      --init-from /mnt/c/Users/Ke/Documents/GitHub/RF-3DGS/output/rrf/a_tx${NAME}_db_geom/rrf_state.pt --init-geometry-only
  $PYS rrf_gsplat/transfer_curve.py < /dev/null >> $LOG 2>&1
done
echo "== all done $(date)" >> $LOG
