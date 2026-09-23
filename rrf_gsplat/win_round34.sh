#!/bin/bash
# Round 34 (after round 33): the planner's end-to-end retrain at (8.2, -5.05, 2) with the earlier command
# (faces = 1), so round 32's 77 s has a same-transmitter, same-path baseline.
set -u
REPO=/c/Users/Ke/Documents/GitHub/RF-3DGS
PYS=/c/Users/Ke/miniconda3/envs/rf-sionna-win/python.exe
export PYTHONUTF8=1
LOG=$REPO/output/rrf/win_round34.log
cd $REPO
stamp() { echo "== $1 $(date +%H:%M:%S) gpu $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader | tr -d '\r')" >> $LOG; }
until grep -aq "== all done" output/rrf/win_round33.log 2>/dev/null; do sleep 20; done
stamp "round 34 start"
$PYS tx_planning/check_interactive_retrain.py --tx 8.2 -5.05 2.0 --faces 1 \
    --out output/tx_planning/interactive_retrain_check_8.2_-5.05_faces1.json < /dev/null 2>&1 | grep -av "jitc\|WARN" >> $LOG
stamp "all done"
