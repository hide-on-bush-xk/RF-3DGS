#!/bin/bash
# Round 25 (smokes, after round 24): the data side and the planning side of the DLSS question.
#   profile_generation.py   where a generated view's time goes (solve vs synthesis, 300x200 vs 150x100), the
#                           label-side SR error, and the ray-budget error against a 4M reference
#   guided_placement.py     Tx placement by coarse solves + fine LoS guide, smoke at fine 1 m / coarse 2 m
set -u
REPO=/c/Users/Ke/Documents/GitHub/RF-3DGS
PYS=/c/Users/Ke/miniconda3/envs/rf-sionna-win/python.exe
export PYTHONUTF8=1 MSYS_NO_PATHCONV=1
LOG=$REPO/output/rrf/win_round25.log
cd $REPO
stamp() { echo "== $1 $(date +%H:%M:%S) gpu $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader | tr -d '\r')" >> $LOG; }
until grep -aq "== all done" output/rrf/win_round24.log 2>/dev/null; do sleep 20; done
stamp "round 25 start"
$PYS sionna_port/profile_generation.py --positions 10 < /dev/null 2>&1 | grep -av "jitc\|WARN" >> $LOG
stamp "profile_generation done"
cd tx_planning
$PYS guided_placement.py --bench ../output/tx_planning/benchmark_lobby.json --fine-step 1.0 --coarse-steps 2.0 \
    --out ../output/tx_planning/guided_lobby_smoke.json < /dev/null 2>&1 | grep -av "jitc\|WARN" >> $LOG
cd $REPO
stamp "all done"
