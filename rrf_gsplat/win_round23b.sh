#!/bin/bash
# Round 23b (after round 22b): the placement benchmarks again. Round 23's runs died in a Dr.Jit CUDA out-of-memory after
# a few hundred solves (lobby during K=3, corridor during K=2) and their JSON was only written at the end. The driver now
# frees Dr.Jit memory every 25 solves, writes the JSON after every method and resumes from it; each benchmark is retried
# up to three times so a further crash loses at most one method.
set -u
REPO=/c/Users/Ke/Documents/GitHub/RF-3DGS
PYS=/c/Users/Ke/miniconda3/envs/rf-sionna-win/python.exe
export PYTHONUTF8=1 MSYS_NO_PATHCONV=1
LOG=$REPO/output/tx_planning/win_round23b.log
cd $REPO
stamp() { echo "== $1 $(date +%H:%M:%S)" >> $LOG; }
gpu_busy() {
  tasklist 2>/dev/null | grep -qi "Against the Storm" && return 0
  u=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr -d ' \r'); [ "${u:-100}" -ge 15 ]
}
until grep -aq "== all done" output/rrf/win_round22b.log 2>/dev/null; do sleep 60; done
quiet=0; while [ $quiet -lt 24 ]; do if gpu_busy; then quiet=0; else quiet=$((quiet+1)); fi; sleep 5; done
stamp "start (gpu $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader | tr -d '\r'))"
cd tx_planning
for attempt in 1 2 3; do
  stamp "lobby attempt $attempt"; T0=$(date +%s)
  $PYS benchmark_placement.py --scene-xml ../sionna_tutorial/RF-3DGS_Sionna_simulation_tutorial/NIST_lobby_v1.0/NIST_lobby_V1.1_sionna12.xml \
      --out ../output/tx_planning/benchmark_lobby.json --k 1 2 3 --control 8.2 -5.05 2.0 --resume < /dev/null >> ../output/tx_planning/benchmark_lobby_23b.log 2>&1
  echo "wall $(( $(date +%s) - T0 )) s, exit $?" >> $LOG
  grep -av "jitc\|WARN" ../output/tx_planning/benchmark_lobby_23b.log | grep -a "control\|candidate\|K=\|GPU memory\|out of memory\|resuming" | tail -30 >> $LOG
  grep -aq '"gradient_rand1_K3"' ../output/tx_planning/benchmark_lobby.json 2>/dev/null && break
done
for attempt in 1 2 3; do
  stamp "corridor attempt $attempt"; T0=$(date +%s)
  $PYS benchmark_placement.py --scene-xml ../scene2/corridor/corridor_sionna.xml --out ../output/tx_planning/benchmark_corridor.json \
      --x-range 0.2 37.8 --y-range -5.7 5.7 --rx-z -0.088 --tx-z 0.287 --k 1 2 3 --resume < /dev/null >> ../output/tx_planning/benchmark_corridor_23b.log 2>&1
  echo "wall $(( $(date +%s) - T0 )) s, exit $?" >> $LOG
  grep -av "jitc\|WARN" ../output/tx_planning/benchmark_corridor_23b.log | grep -a "receivers\|candidate\|K=\|GPU memory\|out of memory\|resuming" | tail -30 >> $LOG
  grep -aq '"gradient_rand1_K3"' ../output/tx_planning/benchmark_corridor.json 2>/dev/null && break
done
stamp "all done"
