#!/bin/bash
# Round 26 (full runs, launched after the round 25 smokes were read):
#   data   MULTI at the _cs poses with the current generator: 300x200 at 1M samples (the truth and the timing
#          baseline), 150x100 at 1M (labels rendered at low resolution)
#   train  the round-24 method config (4 faces per step, lr x2, 2500 steps, gsplat SH, grouped eval) on each:
#            T300      on the 1M truth itself                                  (reference)
#            T150n     on the 150x100 labels, held-out views rendered natively at 300x200 (--test-source): the
#                      3D model is the upsampler
#            T150b     on the 150x100 labels, held-out views at 150x100, bilinearly upsampled to 300x200 afterwards
#                      (upsample_renders.py): a 2D upsampler on the same model
#          every one decoded against the 1M 300x200 truth. (A 100k-sample arm was dropped after round 25: the
#          solve takes 53-60 ms from 30k to 4M samples, so a smaller ray budget buys no time here.)
#   plan   guided placement, full: lobby and corridor, fine 0.5 m, coarse 1 m and 2 m
set -u
REPO=/c/Users/Ke/Documents/GitHub/RF-3DGS
PYS=/c/Users/Ke/miniconda3/envs/rf-sionna-win/python.exe
WSL="wsl.exe -d Ubuntu-22.04 -u ke -- bash /mnt/c/Users/Ke/Documents/GitHub/RF-3DGS/rrf_gsplat/wsl_run.sh"
SCENE=../sionna_tutorial/RF-3DGS_Sionna_simulation_tutorial/NIST_lobby_v1.0/NIST_lobby_V1.1_sionna12.xml
RXLOC=../sionna_tutorial/RF-3DGS_Sionna_simulation_tutorial/NIST_rx_loc.txt
REG=RF-3DGS_dataset/regenerated
export PYTHONUTF8=1 MSYS_NO_PATHCONV=1
LOG=$REPO/output/rrf/win_round26.log
cd $REPO
stamp() { echo "== $1 $(date +%H:%M:%S) gpu $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader | tr -d '\r')" >> $LOG; }
gen() {   # name, extra args
  local name=$1; shift
  [ -f $REG/$name/generation_meta.json ] && { echo "skip $name (exists)" >> $LOG; return; }
  stamp "generate $name"; local T0=$(date +%s)
  (cd sionna_port && $PYS generate_dataset.py --scene-xml $SCENE --rx-loc-file $RXLOC --out-dir ../$REG/$name \
      --spectrum MULTI --num-positions 800 --frequency 2.4e9 --materials tutorial --splat-sigma 3 --no-dashboard "$@" \
      < /dev/null 2>&1 | grep -a "views/s\|channel ranges\|Error\|Traceback\|error" >> $LOG)
  cp $REG/3dgs_MULTI_24ghz_tut_cs/train_index.txt $REG/3dgs_MULTI_24ghz_tut_cs/test_index.txt $REG/$name/
  echo "wall generate $name $(( $(date +%s) - T0 )) s" >> $LOG
}
run() {
  local name=$1; shift
  [ -f output/rrf/$name/results.json ] && { echo "skip $name (done)" >> $LOG; return; }
  stamp "start $name"; local T0=$(date +%s)
  $WSL $name "$@" >> $LOG 2>&1 < /dev/null
  echo "wall $name $(( $(date +%s) - T0 )) s" >> $LOG
}
decode() {  # run name, truth dataset
  [ -f output/rrf/$1/results.json ] || return
  [ -f output/rrf/baselines_$1.json ] && return
  local T0=$(date +%s)
  $PYS rrf_gsplat/eval_baselines.py --multi output/rrf/$1 --truth $2 --out output/rrf/baselines_$1.json < /dev/null 2>&1 | grep -v jitc | tail -3 >> $LOG
  echo "wall decode $1 $(( $(date +%s) - T0 )) s" >> $LOG
}

stamp "round 26 start"
gen r26_MULTI_300_1M
gen r26_MULTI_150_1M --width 150 --height 100
M="--mode multi --delay-depth --delay-depth-mode ED --delay-range euclid --save-renders -1 --sh-backend gsplat --eval-group --faces-per-step 4 --lr-scale 2 --iterations 2500 --eval-every 250"
T=$REG/r26_MULTI_300_1M
run r26_T300        --source $T $M
run r26_T150n       --source $REG/r26_MULTI_150_1M --test-source $T $M
run r26_T150b       --source $REG/r26_MULTI_150_1M $M
[ -f output/rrf/r26_T150b/results.json ] && [ ! -d output/rrf/r26_T150b_up ] && \
  $PYS rrf_gsplat/upsample_renders.py output/rrf/r26_T150b output/rrf/r26_T150b_up --size 200 300 >> $LOG 2>&1
for n in r26_T300 r26_T150n r26_T150b_up; do decode $n $T; done
stamp "rrf done; planning"
cd tx_planning
for s in lobby corridor; do
  [ -f ../output/tx_planning/guided_$s.json ] && continue
  stamp "guided $s"
  $PYS guided_placement.py --bench ../output/tx_planning/benchmark_$s.json --fine-step 0.5 --coarse-steps 1.0 2.0 \
      < /dev/null 2>&1 | grep -av "jitc\|WARN\|GPU memory" >> $LOG
done
cd $REPO
stamp "all done"
