#!/bin/bash
# Round 31 (after round 30): the corridor's guided placement again, with Dr.Jit's pool freed before every
# re-scoring (round 29 ran out of memory at the K = 3 evaluation); K = 1, 2 alone if K = 3 still does not fit.
set -u
REPO=/c/Users/Ke/Documents/GitHub/RF-3DGS
PYS=/c/Users/Ke/miniconda3/envs/rf-sionna-win/python.exe
export PYTHONUTF8=1
LOG=$REPO/output/tx_planning/win_round31.log
cd $REPO
stamp() { echo "== $1 $(date +%H:%M:%S) gpu $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader | tr -d '\r')" >> $LOG; }
until grep -aq "== all done" output/rrf/win_round30.log 2>/dev/null; do sleep 20; done
stamp "round 31 start"
cd tx_planning
$PYS -u guided_placement.py --bench ../output/tx_planning/benchmark_corridor.json --fine-step 0.5 --coarse-steps 1.0 2.0 \
    < /dev/null 2>&1 | grep -av "jitc\|WARN" >> $LOG
if [ ! -f ../output/tx_planning/guided_corridor.json ]; then
  stamp "K=3 did not fit; K = 1, 2"
  $PYS -u guided_placement.py --bench ../output/tx_planning/benchmark_corridor.json --fine-step 0.5 --coarse-steps 1.0 2.0 --k 1 2 \
      < /dev/null 2>&1 | grep -av "jitc\|WARN" >> $LOG
fi
[ -f ../output/tx_planning/guided_corridor.json ] && $PYS guided_refine.py --bench ../output/tx_planning/benchmark_corridor.json \
    --guided ../output/tx_planning/guided_corridor.json --row guided_2m < /dev/null 2>&1 | grep -av "jitc\|WARN\|GPU memory" >> $LOG
cd $REPO
stamp "all done"
