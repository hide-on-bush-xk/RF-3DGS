#!/bin/bash
# Round 21b: the AoD and Delay rows of Track A that round 21 could not run (their released pictures are 231 x 154
# under a 300 x 200 camera; the loader now scales the intrinsics to the image). Only the jet-RGB rows are valid for
# these three-channel encodings. Waits for round 21; round 22 waits for this.
set -u
REPO=/c/Users/Ke/Documents/GitHub/RF-3DGS
PYS=/c/Users/Ke/miniconda3/envs/rf-sionna-win/python.exe
CK=/mnt/c/Users/Ke/Documents/GitHub/RF-3DGS/RF-3DGS_dataset/blender_visual_trained/chkpnt30000.pth
WSL="wsl.exe -d Ubuntu-22.04 -u ke -- bash /mnt/c/Users/Ke/Documents/GitHub/RF-3DGS/rrf_gsplat/wsl_run.sh"
export PYTHONUTF8=1 MSYS_NO_PATHCONV=1
LOG=$REPO/output/rrf/win_round21b.log
cd $REPO
stamp() { echo "== $1 $(date +%H:%M:%S)" >> $LOG; }
run() { local name=$1; shift; [ -f output/rrf/$name/results.json ] && { echo "skip $name" >> $LOG; return; }
        rm -rf output/rrf/$name; stamp "$name"; T0=$(date +%s); $WSL "$name" --checkpoint $CK "$@" >> $LOG 2>&1 < /dev/null; echo "wall $(( $(date +%s) - T0 )) s" >> $LOG; stamp "done $name"; }
score() { local name=$1; local ds=$2; [ -f output/rrf/$name/inria_metrics.json ] && return
          $PYS rrf_gsplat/inria_metrics.py --run output/rrf/$name --source $ds < /dev/null 2>&1 | tail -1 >> $LOG; }
until grep -aq "== all done" output/rrf/win_round21.log 2>/dev/null; do sleep 30; done
stamp "start"
for S in AoD Delay; do
  DS=RF-3DGS_dataset/training-rf-spectrum/3dgs_${S}_100
  run sota_${S}_rgb      --source $DS --mode rgb --save-renders -1;                  score sota_${S}_rgb $DS
  run sota_${S}_rgb_geom --source $DS --mode rgb --train-geometry --save-renders -1; score sota_${S}_rgb_geom $DS
done
stamp "all done"
