#!/bin/bash
# Round 20: seed noise floor of the DECODED metrics (median azimuth / zenith /
# delay) at the two densest lobby levels, which set the crossover: the full
# training set and 160 positions, seeds 1-3 each, same flags as the density
# sweep (D range term), baselines on the same subset. Criterion (Ke): the sd
# of the medians decides how many digits a crossover may carry.
set -u
REPO=/c/Users/Ke/Documents/GitHub/RF-3DGS
PYS=/c/Users/Ke/miniconda3/envs/rf-sionna-win/python.exe
DS=RF-3DGS_dataset/regenerated/3dgs_MULTI_24ghz_tut_cs
CK=/mnt/c/Users/Ke/Documents/GitHub/RF-3DGS/RF-3DGS_dataset/blender_visual_trained/chkpnt30000.pth
WSL="wsl.exe -d Ubuntu-22.04 -u ke -- bash /mnt/c/Users/Ke/Documents/GitHub/RF-3DGS/rrf_gsplat/wsl_run.sh"
export PYTHONUTF8=1 MSYS_NO_PATHCONV=1
LOG=$REPO/output/rrf/win_round20.log
cd $REPO
stamp() { echo "== $1 $(date +%H:%M:%S)" >> $LOG; }
run() { local name=$1; shift; [ -f output/rrf/$name/results.json ] && { echo "skip $name" >> $LOG; return; }
        stamp "$name"; $WSL "$name" --checkpoint $CK "$@" >> $LOG 2>&1 < /dev/null; stamp "done $name"; }
stamp "start"
for s in 1 2 3; do
  run m_multi_24_tut_cs_depth_s$s --source $DS --mode multi --delay-depth --save-renders 640 --seed $s
  stamp "baselines s$s"; $PYS rrf_gsplat/eval_baselines.py --multi output/rrf/m_multi_24_tut_cs_depth_s$s --truth $DS --out output/rrf/baselines_m_multi_24_tut_cs_depth_s$s.json < /dev/null 2>&1 | grep -v jitc | tail -6 >> $LOG
done
for s in 1 2 3; do
  run m_multi_24_tut_cs_depth_v640_s$s --source $DS --mode multi --delay-depth --max-train-views 640 --save-renders 640 --seed $s
  stamp "baselines v640 s$s"; $PYS rrf_gsplat/eval_baselines.py --multi output/rrf/m_multi_24_tut_cs_depth_v640_s$s --truth $DS --max-train-views 640 --out output/rrf/baselines_m_multi_24_tut_cs_depth_v640_s$s.json < /dev/null 2>&1 | grep -v jitc | tail -6 >> $LOG
done
stamp "all done"
