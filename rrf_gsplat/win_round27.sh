#!/bin/bash
# Round 27: the interactive planner's "Retrain RRF here" chain (tx_planning/interactive.py) with the round-24
# training, and label-side super-resolution where it could pay: once training is ~15 s the MVDR dataset is the
# largest stage. Same transmitter as the recorded live run (live_tx_p6_2_m7_7_p2_0: data 39 s, training 47 s).
#   data   MVDR, 60 GHz, 160 positions, at 300x200 and at 150x100 (renormalised to the 300x200 range)
#   train  old   the planner's command: db, 2000 steps, torch SH, one view per step
#          new   db, 500 steps x 4 faces, lr x2, gsplat SH, grouped evaluation
#          150n  new, trained on the 150x100 labels, held-out views rendered natively at 300x200
#          150b  new, trained and rendered at 150x100, renders bilinearly upsampled to 300x200
#   score  PSNR / dB RMSE (train_rrf) and the peaks of every held-out spectrum (mvdr_peaks.py) against the 300x200 truth
set -u
REPO=/c/Users/Ke/Documents/GitHub/RF-3DGS
PYS=/c/Users/Ke/miniconda3/envs/rf-sionna-win/python.exe
WSL="wsl.exe -d Ubuntu-22.04 -u ke -- bash /mnt/c/Users/Ke/Documents/GitHub/RF-3DGS/rrf_gsplat/wsl_run.sh"
SCENE=../sionna_tutorial/RF-3DGS_Sionna_simulation_tutorial/NIST_lobby_v1.0/NIST_lobby_V1.1_sionna12.xml
RXLOC=../sionna_tutorial/RF-3DGS_Sionna_simulation_tutorial/NIST_rx_loc.txt
REG=RF-3DGS_dataset/regenerated
TX="6.232953105196451 -7.689077437535534 2.0"
export PYTHONUTF8=1 MSYS_NO_PATHCONV=1
LOG=$REPO/output/rrf/win_round27.log
cd $REPO
stamp() { echo "== $1 $(date +%H:%M:%S) gpu $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader | tr -d '\r')" >> $LOG; }
until grep -aq "== all done" output/rrf/win_round26.log 2>/dev/null; do sleep 20; done
stamp "round 27 start"
for spec in "300:300:200" "150:150:100"; do
  IFS=: read tag w h <<< "$spec"; name=r27_live_$tag
  [ -f $REG/${name}_gpct/generation_meta.json ] && continue
  stamp "generate $name"; T0=$(date +%s)
  (cd sionna_port && $PYS generate_dataset.py --scene-xml $SCENE --rx-loc-file $RXLOC --out-dir ../$REG/$name --spectrum MVDR \
      --num-positions 160 --tx $TX --no-dashboard --width $w --height $h < /dev/null 2>&1 | grep -a "views/s\|dB range\|Error\|Traceback" >> $LOG)
  T1=$(date +%s)
  if [ $tag = 300 ]; then
    $PYS rrf_gsplat/renormalize.py $REG/$name $REG/${name}_gpct --norm global-pct --pct 1 99.99 < /dev/null >> $LOG 2>&1
  else
    R=$($PYS -c "import json;m=json.load(open('$REG/r27_live_300_gpct/generation_meta.json'));print(m['spec_min_db'],m['spec_max_db'])")
    $PYS rrf_gsplat/renormalize.py $REG/$name $REG/${name}_gpct --range $R < /dev/null >> $LOG 2>&1
  fi
  echo "wall generate $name $(( T1 - T0 )) s, renormalise $(( $(date +%s) - T1 )) s" >> $LOG
done
run() {
  local name=$1; shift
  [ -f output/rrf/$name/results.json ] && return
  stamp "start $name"; local T0=$(date +%s)
  $WSL $name "$@" >> $LOG 2>&1 < /dev/null
  echo "wall $name $(( $(date +%s) - T0 )) s" >> $LOG
}
T=$REG/r27_live_300_gpct
NEW="--mode db --sh-backend gsplat --eval-group --faces-per-step 4 --lr-scale 2 --iterations 500 --eval-every 125 --save-renders -1"
run r27_live_old  --source $T --mode db --iterations 2000 --eval-every 250 --save-renders -1
run r27_live_new  --source $T $NEW
run r27_live_150n --source $REG/r27_live_150_gpct --test-source $T $NEW
run r27_live_150b --source $REG/r27_live_150_gpct $NEW
[ -d output/rrf/r27_live_150b_up ] || $PYS rrf_gsplat/upsample_renders.py output/rrf/r27_live_150b output/rrf/r27_live_150b_up --size 200 300 >> $LOG 2>&1
stamp "scoring peaks"
for n in r27_live_old r27_live_new r27_live_150n r27_live_150b_up; do
  $PYS rrf_gsplat/mvdr_peaks.py --run output/rrf/$n --truth $T >> $LOG 2>&1
done
# the labels themselves at 150x100, bilinearly upsampled: what the 2D upsampler does with no model in between
if [ ! -d output/rrf/r27_labels150_up ]; then
  mkdir -p output/rrf/r27_labels150/renders
  cp $REG/r27_live_150_gpct/spectra_float/*.npy output/rrf/r27_labels150/renders/
  echo '{}' > output/rrf/r27_labels150/results.json
  $PYS rrf_gsplat/upsample_renders.py output/rrf/r27_labels150 output/rrf/r27_labels150_up --size 200 300 >> $LOG 2>&1
fi
$PYS rrf_gsplat/mvdr_peaks.py --run output/rrf/r27_labels150_up --truth $T >> $LOG 2>&1
stamp "all done"
