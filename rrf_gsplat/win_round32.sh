#!/bin/bash
# Round 32 (after round 31): the planner's end-to-end retrain again at a valid transmitter. Round 30 clicked
# (3, -5, 2), where 584 of 640 views have no path (PSNR 34.8 dB is the floor being fitted); the pipeline survived
# the degenerate case but the quality number is meaningless. (8.2, -5.05, 2) is the benchmark's K = 1 optimum.
set -u
REPO=/c/Users/Ke/Documents/GitHub/RF-3DGS
PYS=/c/Users/Ke/miniconda3/envs/rf-sionna-win/python.exe
export PYTHONUTF8=1
LOG=$REPO/output/rrf/win_round32.log
cd $REPO
stamp() { echo "== $1 $(date +%H:%M:%S) gpu $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader | tr -d '\r')" >> $LOG; }
until grep -aq "== all done" output/tx_planning/win_round31.log 2>/dev/null; do sleep 20; done
stamp "round 32 start"
$PYS tx_planning/check_interactive_retrain.py --tx 8.2 -5.05 2.0 --faces 4 \
    --out output/tx_planning/interactive_retrain_check_8.2_-5.05.json < /dev/null 2>&1 | grep -av "jitc\|WARN" >> $LOG
stamp "all done"
