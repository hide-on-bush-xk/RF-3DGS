#!/bin/bash
# Round 14 = rounds 12 + 13 re-queued behind a stricter gate. Rounds 11-13
# started the moment the card looked idle, and the game came back seconds
# later (03:02:23), so their generation timings say nothing. This waits
# until no game process exists AND GPU utilisation stays under 15 % for two
# minutes, then:
#   1. the per-position-seed MVDR dataset and the two headline retrains
#      (third cross-view-consistency number, unfreezing gain on consistent data);
#   2. the generator A/B: --tx (per-process path) against --tx-list with one
#      transmitter (shared path), same 100 positions, progress time-stamped.
set -u
REPO=/c/Users/Ke/Documents/GitHub/RF-3DGS
PYS=/c/Users/Ke/miniconda3/envs/rf-sionna-win/python.exe
SCENE=../sionna_tutorial/RF-3DGS_Sionna_simulation_tutorial/NIST_lobby_v1.0/NIST_lobby_V1.1_sionna12.xml
RXLOC=../sionna_tutorial/RF-3DGS_Sionna_simulation_tutorial/NIST_rx_loc.txt
REG=RF-3DGS_dataset/regenerated
OUT="C:/Users/Ke/AppData/Local/Temp/claude/c--Users-Ke-Documents-GitHub-RF-3DGS/ed7324c0-b974-4409-b2e5-4c5d23e75259/scratchpad"
WSL="wsl.exe -d Ubuntu-22.04 -u ke -- bash /mnt/c/Users/Ke/Documents/GitHub/RF-3DGS/rrf_gsplat/wsl_run.sh"
export PYTHONUTF8=1 MSYS_NO_PATHCONV=1
LOG=$REPO/output/rrf/win_round14.log
cd $REPO
stamp() { echo "== $1 $(date +%H:%M:%S)" >> $LOG; }
gpu_busy() {
  tasklist 2>/dev/null | grep -qi "Against the Storm" && return 0
  u=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr -d ' \r'); [ "${u:-100}" -ge 15 ]
}
stamp "start, waiting for a card with no game and < 15 % for 2 min"
quiet=0
while [ $quiet -lt 24 ]; do if gpu_busy; then quiet=0; else quiet=$((quiet+1)); fi; sleep 5; done
stamp "gpu quiet ($(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader | tr -d '\r'))"
run() { local name=$1; shift; [ -f output/rrf/$name/results.json ] && { echo "skip $name" >> $LOG; return; }
        stamp "$name"; $WSL "$name" "$@" >> $LOG 2>&1 < /dev/null; stamp "done $name"; }

DS=3dgs_MVDR_100_posseed
if [ ! -d $REG/${DS}_gpct ]; then
  rm -rf $REG/$DS
  stamp "dataset $DS (per-position seed, otherwise as 3dgs_MVDR_100)"
  cd $REPO/sionna_port
  $PYS generate_dataset.py --scene-xml $SCENE --rx-loc-file $RXLOC --out-dir ../$REG/$DS --spectrum MVDR --num-positions 800 --no-dashboard \
      < /dev/null 2>&1 | grep -a "views/s\|global dB\|Error\|Traceback" >> $LOG
  cd $REPO
  $PYS rrf_gsplat/renormalize.py $REG/$DS $REG/${DS}_gpct --norm global-pct --pct 1 99.99 < /dev/null 2>&1 | tail -1 >> $LOG
  cp $REG/3dgs_MVDR_100_gpct/train_index.txt $REG/3dgs_MVDR_100_gpct/test_index.txt $REG/${DS}_gpct/
  stamp "dataset done"
fi
run e2_mvdr_db_posseed     --source $REG/${DS}_gpct --mode db --save-renders 0
run a_mvdr_db_geom_posseed --source $REG/${DS}_gpct --mode db --train-geometry --save-renders 0

if gpu_busy; then stamp "game back before the A/B; A/B skipped"; else
  rm -rf "$OUT"/ab_single "$OUT"/ab_list
  cd $REPO/sionna_port
  stamp "(a) per-process"
  $PYS generate_dataset.py --scene-xml $SCENE --rx-loc-file $RXLOC --out-dir "$OUT"/ab_single --spectrum MVDR --num-positions 100 \
      --tx 6.9 -5.4 0.8 --no-dashboard < /dev/null 2>&1 | grep -a --line-buffered "^  [0-9]*/\|views/s\|Error\|Traceback" | while read -r l; do echo "$(date +%H:%M:%S) $l"; done >> $LOG
  stamp "(b) shared path, one transmitter"
  $PYS generate_dataset.py --scene-xml $SCENE --rx-loc-file $RXLOC --out-dir "$OUT"/ab_list --spectrum MVDR --num-positions 100 \
      --tx-list P:6.9,-5.4,0.8 --no-dashboard < /dev/null 2>&1 | grep -a --line-buffered "^  [0-9]*/\|views/s\|built once\|tx-list\|Error\|Traceback" | while read -r l; do echo "$(date +%H:%M:%S) $l"; done >> $LOG
  cd $REPO
fi
stamp "all done"
