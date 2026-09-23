#!/bin/bash
# Round 35 (after round 34): MULTI generation at 300x200 and 150x100, alternating, twice each, on a quiet
# machine. Round 33 re-timed 300x200 at 58 s against round 26's 100 s, so round 26's 150x100 46 s (same
# window) is suspect too, and with it the "labels at low resolution halve the data stage" reading.
set -u
REPO=/c/Users/Ke/Documents/GitHub/RF-3DGS
PYS=/c/Users/Ke/miniconda3/envs/rf-sionna-win/python.exe
SCENE=../sionna_tutorial/RF-3DGS_Sionna_simulation_tutorial/NIST_lobby_v1.0/NIST_lobby_V1.1_sionna12.xml
RXLOC=../sionna_tutorial/RF-3DGS_Sionna_simulation_tutorial/NIST_rx_loc.txt
REG=RF-3DGS_dataset/regenerated
export PYTHONUTF8=1
LOG=$REPO/output/rrf/win_round35.log
cd $REPO
stamp() { echo "== $1 $(date +%H:%M:%S) gpu $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader | tr -d '\r')" >> $LOG; }
until grep -aq "== all done" output/rrf/win_round34.log 2>/dev/null; do sleep 20; done
stamp "round 35 start"
for rep in 1 2; do
  for spec in "300:300:200" "150:150:100"; do
    IFS=: read tag w h <<< "$spec"
    stamp "retime $tag rep $rep"; T0=$(date +%s)
    (cd sionna_port && $PYS generate_dataset.py --scene-xml $SCENE --rx-loc-file $RXLOC --out-dir ../$REG/r35_retime_$tag \
        --spectrum MULTI --num-positions 800 --frequency 2.4e9 --materials tutorial --splat-sigma 3 --no-dashboard \
        --width $w --height $h < /dev/null 2>&1 | grep -a "views/s\|Error\|Traceback" >> $LOG)
    echo "wall retime $tag rep $rep $(( $(date +%s) - T0 )) s" >> $LOG
    rm -rf $REG/r35_retime_$tag
  done
done
stamp "all done"
