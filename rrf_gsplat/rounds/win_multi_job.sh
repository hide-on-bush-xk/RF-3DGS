#!/bin/bash
# Multi-channel datasets at the released data's setting (2.4 GHz, tutorial
# materials): MULTI (power / AoD az / AoD zen / delay, one channel each) and
# AOD3 (the tutorial's angle-times-amplitude RGB encoding), same poses.
set -u
REPO=/c/Users/Ke/Documents/GitHub/RF-3DGS
PYS=/c/Users/Ke/miniconda3/envs/rf-sionna-win/python.exe
SCENE=../sionna_tutorial/RF-3DGS_Sionna_simulation_tutorial/NIST_lobby_v1.0/NIST_lobby_V1.1_sionna12.xml
RXLOC=../sionna_tutorial/RF-3DGS_Sionna_simulation_tutorial/NIST_rx_loc.txt
REG=RF-3DGS_dataset/regenerated
export PYTHONUTF8=1
LOG=$REPO/output/rrf/win_multi_job.log
echo "== start $(date)" >> $LOG
# the generator wants most of the GPU; wait for the E1 rerun's last training
for i in $(seq 1 240); do
  [ -f $REPO/output/rrf/e1b_cbf24_global_db/results.json ] && break
  sleep 5
done
echo "== gpu free $(date)" >> $LOG
cd $REPO/sionna_port
for S in MULTI AOD3; do
  $PYS generate_dataset.py --scene-xml $SCENE --rx-loc-file $RXLOC \
      --out-dir ../$REG/3dgs_${S}_24ghz_tut --spectrum $S --num-positions 800 \
      --frequency 2.4e9 --materials tutorial --no-dashboard 2>&1 | grep -a "views/s\|channel ranges\|global dB\|Error\|error" >> $LOG
  cp ../$REG/3dgs_MVDR_100_gpct/train_index.txt ../$REG/3dgs_MVDR_100_gpct/test_index.txt ../$REG/3dgs_${S}_24ghz_tut/
done
echo "== datasets done $(date)" >> $LOG
tail -6 $LOG
