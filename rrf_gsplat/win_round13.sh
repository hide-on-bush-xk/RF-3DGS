#!/bin/bash
# Round 13: localise the --tx-list slowdown. On a quiet card the batched
# generator ran its FIRST dataset at 0.7 views/s (100 positions in 10 min)
# against 12-15 views/s for the per-process path, so the shared path is
# slow before any transmitter is swapped. A/B on the same 100 positions,
# same transmitter, one after the other, progress lines time-stamped:
#   (a) --tx 6.9 -5.4 0.8            per-process path
#   (b) --tx-list P:6.9,-5.4,0.8     shared path, one transmitter
# Waits for round 12's "all done". Output under the Windows scratchpad.
set -u
REPO=/c/Users/Ke/Documents/GitHub/RF-3DGS
PYS=/c/Users/Ke/miniconda3/envs/rf-sionna-win/python.exe
SCENE=../sionna_tutorial/RF-3DGS_Sionna_simulation_tutorial/NIST_lobby_v1.0/NIST_lobby_V1.1_sionna12.xml
RXLOC=../sionna_tutorial/RF-3DGS_Sionna_simulation_tutorial/NIST_rx_loc.txt
OUT="C:/Users/Ke/AppData/Local/Temp/claude/c--Users-Ke-Documents-GitHub-RF-3DGS/ed7324c0-b974-4409-b2e5-4c5d23e75259/scratchpad"
export PYTHONUTF8=1 MSYS_NO_PATHCONV=1
LOG=$REPO/output/rrf/win_round13.log
cd $REPO
stamp() { echo "== $1 $(date +%H:%M:%S)" >> $LOG; }
stamp "start"
for i in $(seq 1 2880); do grep -q "all done" output/rrf/win_round12.log 2>/dev/null && break; sleep 5; done
stamp "round 12 finished ($(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader | tr -d '\r'))"
rm -rf "$OUT"/ab_single "$OUT"/ab_listP
cd $REPO/sionna_port
stamp "(a) per-process"
$PYS generate_dataset.py --scene-xml $SCENE --rx-loc-file $RXLOC --out-dir "$OUT"/ab_single --spectrum MVDR --num-positions 100 \
    --tx 6.9 -5.4 0.8 --no-dashboard < /dev/null 2>&1 | grep -a --line-buffered "^  [0-9]*/\|views/s\|Error\|Traceback" | while read -r l; do echo "$(date +%H:%M:%S) $l"; done >> $LOG
stamp "(b) shared path, one transmitter"
$PYS generate_dataset.py --scene-xml $SCENE --rx-loc-file $RXLOC --out-dir "$OUT"/ab_list --spectrum MVDR --num-positions 100 \
    --tx-list P:6.9,-5.4,0.8 --no-dashboard < /dev/null 2>&1 | grep -a --line-buffered "^  [0-9]*/\|views/s\|built once\|tx-list\|Error\|Traceback" | while read -r l; do echo "$(date +%H:%M:%S) $l"; done >> $LOG
cd $REPO
stamp "all done"
