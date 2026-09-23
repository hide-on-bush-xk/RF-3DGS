#!/bin/bash
# Round 29: guided placement on the corridor (round 26's run ran out of Dr.Jit memory in the 1020-cell fine sweep;
# now flushing every 5 solves), then its gradient refinement. After round 28, exclusive card.
set -u
REPO=/c/Users/Ke/Documents/GitHub/RF-3DGS
PYS=/c/Users/Ke/miniconda3/envs/rf-sionna-win/python.exe
export PYTHONUTF8=1
LOG=$REPO/output/tx_planning/win_round29.log
cd $REPO
stamp() { echo "== $1 $(date +%H:%M:%S) gpu $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader | tr -d '\r')" >> $LOG; }
until grep -aq "== all done" output/tx_planning/win_round28.log 2>/dev/null; do sleep 20; done
stamp "round 29 start"
cd tx_planning
$PYS -u guided_placement.py --bench ../output/tx_planning/benchmark_corridor.json --fine-step 0.5 --coarse-steps 1.0 2.0 \
    < /dev/null 2>&1 | grep -av "jitc\|WARN" >> $LOG
[ -f ../output/tx_planning/guided_corridor.json ] && $PYS guided_refine.py --bench ../output/tx_planning/benchmark_corridor.json \
    --guided ../output/tx_planning/guided_corridor.json --row guided_2m < /dev/null 2>&1 | grep -av "jitc\|WARN\|GPU memory" >> $LOG
cd $REPO
stamp "all done"
