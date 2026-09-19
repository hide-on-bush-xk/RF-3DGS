#!/bin/bash
# Round 11: the two GPU items that need a clean card. Waits until the GPU
# has been under 15 % utilisation for a full minute (the card was shared
# with a game all evening), then:
#   1. multi-bounce power share at 40 route positions, split LoS / NLoS;
#   2. the --tx-list leak re-check: three transmitters x 100 positions in
#      one process, GPU memory sampled every 5 s alongside; a growing
#      per-dataset time or memory with the game gone is a leak.
set -u
REPO=/c/Users/Ke/Documents/GitHub/RF-3DGS
PYS=/c/Users/Ke/miniconda3/envs/rf-sionna-win/python.exe
SCENE=../sionna_tutorial/RF-3DGS_Sionna_simulation_tutorial/NIST_lobby_v1.0/NIST_lobby_V1.1_sionna12.xml
RXLOC=../sionna_tutorial/RF-3DGS_Sionna_simulation_tutorial/NIST_rx_loc.txt
SCRATCH="$1"
export PYTHONUTF8=1 MSYS_NO_PATHCONV=1
LOG=$REPO/output/rrf/win_round11.log
cd $REPO
stamp() { echo "== $1 $(date +%H:%M:%S)" >> $LOG; }
stamp "start, waiting for a quiet GPU"
quiet=0
while [ $quiet -lt 12 ]; do
  u=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr -d ' \r')
  if [ "${u:-100}" -lt 15 ]; then quiet=$((quiet+1)); else quiet=0; fi
  sleep 5
done
stamp "gpu quiet ($(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader | tr -d '\r'))"

stamp "multibounce 40 positions"
$PYS rrf_gsplat/multibounce_fraction.py --positions 40 < /dev/null 2>&1 | grep -v "jitc_llvm\|WARN\|eps_r\|classes on" >> $LOG
stamp "multibounce done"

stamp "tx-list leak check (3 x 100 positions, one process)"
rm -rf "$SCRATCH"/leak_*
( while true; do echo "$(date +%H:%M:%S) $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader | tr -d '\r')"; sleep 5; done ) > $REPO/output/rrf/win_round11_gpu.log 2>&1 &
SAMPLER=$!
cd $REPO/sionna_port
$PYS generate_dataset.py --scene-xml $SCENE --rx-loc-file $RXLOC --out-dir "$SCRATCH"/leak_ --spectrum MVDR --num-positions 100 \
    --tx-list P:6.9,-5.4,0.8 Q:2.0,-3.0,0.8 R:-3.35,0.0,0.8 --no-dashboard < /dev/null 2>&1 | grep -a "built once\|tx-list\|views/s\|Error\|Traceback" >> $LOG
cd $REPO
kill $SAMPLER 2>/dev/null
stamp "leak check done"
for n in P Q R; do
  m="$SCRATCH"/leak_$n/generation_meta.json
  [ -f "$m" ] && $PYS -c "import json; d=json.load(open(r'$m')); print('leak_$n seconds', round(d['seconds']), 'views/s', round(d['views_per_second'],1))" >> $LOG
done
stamp "all done"
