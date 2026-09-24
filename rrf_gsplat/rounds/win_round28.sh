#!/bin/bash
# Round 28: the benchmark's gradient method started from the guided 2 m placement instead of the 1 m sweep's
# best cell (guided_refine.py), lobby and corridor, K = 1, 2. After round 27, exclusive card.
set -u
REPO=/c/Users/Ke/Documents/GitHub/RF-3DGS
PYS=/c/Users/Ke/miniconda3/envs/rf-sionna-win/python.exe
export PYTHONUTF8=1
LOG=$REPO/output/tx_planning/win_round28.log
cd $REPO
stamp() { echo "== $1 $(date +%H:%M:%S) gpu $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader | tr -d '\r')" >> $LOG; }
until grep -aq "== all done" output/rrf/win_round27.log 2>/dev/null; do sleep 20; done
stamp "round 28 start"
cd tx_planning
for s in lobby corridor; do
  [ -f ../output/tx_planning/guided_$s.json ] || continue
  stamp "refine $s"
  $PYS guided_refine.py --bench ../output/tx_planning/benchmark_$s.json --guided ../output/tx_planning/guided_$s.json \
      --row guided_2m < /dev/null 2>&1 | grep -av "jitc\|WARN\|GPU memory" >> $LOG
done
cd $REPO
stamp "all done"
