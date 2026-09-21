#!/bin/bash
# Round 23 (after round 22): Track C placement benchmarks on an exclusive card, lobby then corridor.
# The lobby run started during round 21 was stopped (the two jobs saturated the card and stalled metrics.py).
set -u
REPO=/c/Users/Ke/Documents/GitHub/RF-3DGS
PYS=/c/Users/Ke/miniconda3/envs/rf-sionna-win/python.exe
export PYTHONUTF8=1 MSYS_NO_PATHCONV=1 PYTHONUNBUFFERED=1
LOG=$REPO/output/tx_planning/win_round23.log
cd $REPO
stamp() { echo "== $1 $(date +%H:%M:%S)" >> $LOG; }
gpu_busy() {
  tasklist 2>/dev/null | grep -qi "Against the Storm" && return 0
  u=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr -d ' \r'); [ "${u:-100}" -ge 15 ]
}
until grep -aq "== all done" output/rrf/win_round22.log 2>/dev/null; do sleep 60; done
quiet=0; while [ $quiet -lt 24 ]; do if gpu_busy; then quiet=0; else quiet=$((quiet+1)); fi; sleep 5; done
stamp "start (gpu $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader | tr -d '\r'))"
cd tx_planning
if [ ! -f ../output/tx_planning/benchmark_lobby.json ]; then
  stamp "lobby"; T0=$(date +%s)
  $PYS benchmark_placement.py --scene-xml ../sionna_tutorial/RF-3DGS_Sionna_simulation_tutorial/NIST_lobby_v1.0/NIST_lobby_V1.1_sionna12.xml \
      --out ../output/tx_planning/benchmark_lobby.json --k 1 2 3 --control 8.2 -5.05 2.0 < /dev/null > ../output/tx_planning/benchmark_lobby.log 2>&1
  echo "wall $(( $(date +%s) - T0 )) s" >> $LOG; grep -v "jitc\|WARN" ../output/tx_planning/benchmark_lobby.log | grep "control\|candidate\|K=\|solves," >> $LOG
fi
if [ ! -f ../output/tx_planning/benchmark_corridor.json ]; then
  stamp "corridor"; T0=$(date +%s)
  $PYS benchmark_placement.py --scene-xml ../scene2/corridor/corridor_sionna.xml --out ../output/tx_planning/benchmark_corridor.json \
      --x-range 0.2 37.8 --y-range -5.7 5.7 --rx-z -0.088 --tx-z 0.287 --k 1 2 3 < /dev/null > ../output/tx_planning/benchmark_corridor.log 2>&1
  echo "wall $(( $(date +%s) - T0 )) s" >> $LOG; grep -v "jitc\|WARN" ../output/tx_planning/benchmark_corridor.log | grep "receivers\|candidate\|K=\|solves," >> $LOG
fi
stamp "all done"
