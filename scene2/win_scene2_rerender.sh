#!/bin/bash
# The scene-2 density runs saved renders for 300 of their 452 held-out views,
# so their baselines scored 300. Re-render every held-out view from each run's
# saved state (0 iterations, warm start = the trained colours) and score the
# baselines on all 452, split by the receiver's space class (corridor / room /
# hall). Waits for win_scene2_extra.sh (room hold-out, seeds) to finish.
set -u
REPO=/c/Users/Ke/Documents/GitHub/RF-3DGS
PYS=/c/Users/Ke/miniconda3/envs/rf-sionna-win/python.exe
REG=RF-3DGS_dataset/regenerated
CK=/mnt/c/Users/Ke/Documents/GitHub/RF-3DGS/scene2/visual_trained/chkpnt30000.pth
WSL="wsl.exe -d Ubuntu-22.04 -u ke -- bash /mnt/c/Users/Ke/Documents/GitHub/RF-3DGS/rrf_gsplat/wsl_run.sh"
OUT=/mnt/c/Users/Ke/Documents/GitHub/RF-3DGS/output/rrf
export PYTHONUTF8=1 MSYS_NO_PATHCONV=1
LOG=$REPO/output/rrf/win_scene2_rerender.log
cd $REPO
stamp() { echo "== $1 $(date +%H:%M:%S)" >> $LOG; }
until grep -aq "== all done" output/rrf/win_scene2_extra.log 2>/dev/null; do sleep 15; done
stamp "start"
DS=$REG/s2_MULTI_corrM
for V in 0 900 452 224 112; do
  name=s2_multi_corrM_depth$( [ $V = 0 ] && echo "" || echo "_v$V" )
  full=${name}_full
  if [ ! -f output/rrf/$full/results.json ]; then
    stamp "$full"
    $WSL "$full" --checkpoint $CK --source $DS --mode multi --delay-depth --iterations 0 --init-from $OUT/$name/rrf_state.pt --save-renders 452 \
        $( [ $V = 0 ] || echo "--max-train-views $V" ) >> $LOG 2>&1 < /dev/null
    stamp "done $full"
  fi
  stamp "baselines $full"
  $PYS rrf_gsplat/eval_baselines.py --multi output/rrf/$full --truth $DS --layout scene2/corridor/layout.json \
       $( [ $V = 0 ] || echo "--max-train-views $V" ) --out output/rrf/baselines_$full.json < /dev/null 2>&1 | grep -v jitc >> $LOG
done
stamp "baselines roomS1 by space"
$PYS rrf_gsplat/eval_baselines.py --multi output/rrf/s2_multi_corrM_depth_roomS1 --truth $REG/s2_MULTI_corrM_roomS1 --layout scene2/corridor/layout.json \
     --out output/rrf/baselines_s2_multi_corrM_depth_roomS1.json < /dev/null 2>&1 | grep -v jitc >> $LOG
stamp "all done"
