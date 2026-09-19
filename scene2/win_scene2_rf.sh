#!/bin/bash
# Scene 2, stage 2: the three experiments, nothing else (Ke: replicate three
# claims, not the mechanism studies). Waits for the visual chain's "all done"
# and its checkpoint, then for a quiet card.
#   A. density crossover   MULTI, 2.4 GHz, uniform scattering 0.7, Tx corridor_M,
#                          all 287 route positions, 20 % held out; training
#                          positions 230 / 115 / 58 / 29 / 15; --delay-depth;
#                          copy baselines from the same subsets.
#   B. zone structure      MVDR 60 GHz (as scene 1's Tx experiments), the ten
#                          transmitters, target room_S2: geometry adapted on each
#                          source (10k, no eval) then frozen on the target with
#                          colours reset (2k, seed 0); cold and own-unfrozen refs.
#   C. consistency         per-view normalisation (rgb, gpct vs perview) on the
#                          corridor_M MVDR dataset; per-view seed on the MULTI
#                          dataset (cs vs yawseed), both with the plain trainer.
set -u
REPO=/c/Users/Ke/Documents/GitHub/RF-3DGS
PYS=/c/Users/Ke/miniconda3/envs/rf-sionna-win/python.exe
SCENE=../scene2/corridor/corridor_sionna.xml
RXLOC=../scene2/corridor/rx_route.txt
REG=RF-3DGS_dataset/regenerated
CK=/mnt/c/Users/Ke/Documents/GitHub/RF-3DGS/scene2/visual_trained/chkpnt30000.pth
WSL="wsl.exe -d Ubuntu-22.04 -u ke -- bash /mnt/c/Users/Ke/Documents/GitHub/RF-3DGS/rrf_gsplat/wsl_run.sh"
OUT=/mnt/c/Users/Ke/Documents/GitHub/RF-3DGS/output/rrf
export PYTHONUTF8=1 MSYS_NO_PATHCONV=1
LOG=$REPO/output/rrf/win_scene2_rf.log
cd $REPO
stamp() { echo "== $1 $(date +%H:%M:%S)" >> $LOG; }
gpu_busy() {
  tasklist 2>/dev/null | grep -qi "Against the Storm" && return 0
  u=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr -d ' \r'); [ "${u:-100}" -ge 15 ]
}
stamp "start"
for i in $(seq 1 5760); do grep -q "all done" output/scene2_visual.log 2>/dev/null && [ -f scene2/visual_trained/chkpnt30000.pth ] && break; sleep 5; done
[ -f scene2/visual_trained/chkpnt30000.pth ] || { stamp "no visual checkpoint; stopping"; exit 1; }
quiet=0; while [ $quiet -lt 24 ]; do if gpu_busy; then quiet=0; else quiet=$((quiet+1)); fi; sleep 5; done
stamp "visual chain done, gpu quiet ($(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader | tr -d '\r'))"
run() { local name=$1; shift; [ -f output/rrf/$name/results.json ] && { echo "skip $name" >> $LOG; return; }
        stamp "$name"; $WSL "$name" --checkpoint $CK "$@" >> $LOG 2>&1 < /dev/null; stamp "done $name"; }
base() { local name=$1; local truth=$2; shift 2; stamp "baselines $name"
         $PYS rrf_gsplat/eval_baselines.py --multi output/rrf/$name --truth $truth --out output/rrf/baselines_$name.json "$@" < /dev/null 2>&1 | grep -v jitc >> $LOG; }
gen() { local ds=$1; shift; [ -f $REG/$ds/generation_meta.json ] && return
        stamp "dataset $ds"; cd $REPO/sionna_port
        $PYS generate_dataset.py --scene-xml $SCENE --rx-loc-file $RXLOC --out-dir ../$REG/$ds --num-positions 800 --no-dashboard "$@" < /dev/null 2>&1 | grep -a "views/s\|global dB\|channel ranges\|Error\|Traceback" >> $LOG
        cd $REPO; }
TXM="14.0 0.0 0.287"

# A. density crossover
gen s2_MULTI_corrM --spectrum MULTI --frequency 2.4e9 --splat-sigma 3 --tx $TXM
DS=$REG/s2_MULTI_corrM
for V in 0 460 232 116 60; do
  name=s2_multi_corrM_depth$( [ $V = 0 ] && echo "" || echo "_v$V" )
  if [ $V = 0 ]; then run $name --source $DS --mode multi --delay-depth --save-renders 300; base $name $DS
  else run $name --source $DS --mode multi --delay-depth --max-train-views $V --save-renders 300; base $name $DS --max-train-views $V; fi
done

# B. zone structure, MVDR 60 GHz, target room_S2
$PYS - <<'EOF' > $REPO/output/rrf/s2_tx.txt
import json; tx = json.load(open("scene2/corridor/tx_positions.json"))
for k, v in tx.items(): print(k, *v)
EOF
while read NAME X Y Z; do
  gen s2_MVDR_tx$NAME --spectrum MVDR --tx $X $Y $Z
  [ -d $REG/s2_MVDR_tx${NAME}_gpct ] || { $PYS rrf_gsplat/renormalize.py $REG/s2_MVDR_tx$NAME $REG/s2_MVDR_tx${NAME}_gpct --norm global-pct --pct 1 99.99 < /dev/null 2>&1 | tail -1 >> $LOG; }
done < $REPO/output/rrf/s2_tx.txt
TGT=$REG/s2_MVDR_txroom_S2_gpct
run s2_t_roomS2_cold_2k --source $TGT --mode db --iterations 2000 --eval-every 500 --save-renders 0 --seed 0
run s2_t_roomS2_geom_2k --source $TGT --mode db --iterations 2000 --eval-every 500 --save-renders 0 --seed 0 --train-geometry
while read NAME X Y Z; do
  [ "$NAME" = "room_S2" ] && continue
  run s2_a_tx${NAME}_geom --source $REG/s2_MVDR_tx${NAME}_gpct --mode db --train-geometry --no-eval --save-renders 0
  run s2_t_roomS2_geom${NAME}_2k --source $TGT --mode db --iterations 2000 --eval-every 500 --save-renders 0 --seed 0 \
      --init-from $OUT/s2_a_tx${NAME}_geom/rrf_state.pt --init-geometry-only
done < $REPO/output/rrf/s2_tx.txt

# C. consistency
[ -d $REG/s2_MVDR_txcorridor_M_perview ] || $PYS rrf_gsplat/renormalize.py $REG/s2_MVDR_txcorridor_M $REG/s2_MVDR_txcorridor_M_perview --norm per-view < /dev/null 2>&1 | tail -1 >> $LOG
run s2_e1_corrM_gpct_rgb    --source $REG/s2_MVDR_txcorridor_M_gpct    --mode rgb --save-renders 0
run s2_e1_corrM_perview_rgb --source $REG/s2_MVDR_txcorridor_M_perview --mode rgb --save-renders 0
gen s2_MULTI_corrM_yawseed --spectrum MULTI --frequency 2.4e9 --splat-sigma 3 --tx $TXM --seed-per-view
run s2_multi_corrM_plain   --source $DS --mode multi --save-renders 0
run s2_multi_corrM_yawseed --source $REG/s2_MULTI_corrM_yawseed --mode multi --save-renders 0
stamp "all done"
