#!/bin/bash
# Round 12: the third cross-view-consistency number. Every MVDR / CBF
# dataset of stage 2 (3dgs_MVDR_100, _txB, _24ghz_tut, ...) was generated
# on 09-18 before 09:14, i.e. with the lattice seeded per VIEW: the four
# faces of one position come from four different diffuse path sets. The
# projection family measured that at 4.1 dB (12.67 -> 8.57). This
# regenerates the Tx-A MVDR dataset with the per-position seed (today's
# default), renormalises it the same way, and retrains the two headline
# runs (frozen db, unfrozen db) so the beamformed family gets its own number
# and the unfreezing gain is re-measured on consistent data. Also the AoA
# elevation histogram behind the azimuth / zenith asymmetry. Waits for
# round 11's "all done".
set -u
REPO=/c/Users/Ke/Documents/GitHub/RF-3DGS
PYS=/c/Users/Ke/miniconda3/envs/rf-sionna-win/python.exe
SCENE=../sionna_tutorial/RF-3DGS_Sionna_simulation_tutorial/NIST_lobby_v1.0/NIST_lobby_V1.1_sionna12.xml
RXLOC=../sionna_tutorial/RF-3DGS_Sionna_simulation_tutorial/NIST_rx_loc.txt
REG=RF-3DGS_dataset/regenerated
WSL="wsl.exe -d Ubuntu-22.04 -u ke -- bash /mnt/c/Users/Ke/Documents/GitHub/RF-3DGS/rrf_gsplat/wsl_run.sh"
export PYTHONUTF8=1 MSYS_NO_PATHCONV=1
LOG=$REPO/output/rrf/win_round12.log
cd $REPO
stamp() { echo "== $1 $(date +%H:%M:%S)" >> $LOG; }
stamp "start"
for i in $(seq 1 2880); do grep -q "all done" output/rrf/win_round11.log 2>/dev/null && break; sleep 5; done
stamp "round 11 finished ($(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader | tr -d '\r'))"
run() { local name=$1; shift; [ -f output/rrf/$name/results.json ] && { echo "skip $name" >> $LOG; return; }
        stamp "$name"; $WSL "$name" "$@" >> $LOG 2>&1 < /dev/null; stamp "done $name"; }

stamp "aoa elevation"
$PYS rrf_gsplat/diag_aoa_elevation.py < /dev/null 2>&1 | grep -v "jitc_llvm\|WARN\|eps_r\|classes on" >> $LOG

DS=3dgs_MVDR_100_posseed
if [ ! -d $REG/${DS}_gpct ]; then
  stamp "dataset $DS (per-position seed, otherwise as 3dgs_MVDR_100)"
  cd $REPO/sionna_port
  $PYS generate_dataset.py --scene-xml $SCENE --rx-loc-file $RXLOC --out-dir ../$REG/$DS --spectrum MVDR --num-positions 800 --no-dashboard \
      < /dev/null 2>&1 | grep -a "views/s\|global dB\|Error\|Traceback" >> $LOG
  cd $REPO
  $PYS rrf_gsplat/renormalize.py $REG/$DS $REG/${DS}_gpct --norm global-pct --pct 1 99.99 < /dev/null 2>&1 | tail -1 >> $LOG
  cp $REG/3dgs_MVDR_100_gpct/train_index.txt $REG/3dgs_MVDR_100_gpct/test_index.txt $REG/${DS}_gpct/
  stamp "dataset done"
fi
run e2_mvdr_db_posseed    --source $REG/${DS}_gpct --mode db --save-renders 0
run a_mvdr_db_geom_posseed --source $REG/${DS}_gpct --mode db --train-geometry --save-renders 0
stamp "all done"
