#!/bin/bash
# Round 19: the delay range term as the Euclidean range (depth * sec theta)
# instead of the camera z, on the lobby ED run's exact settings; criterion in
# docs/stage2_notes.md round 15 section 3b (written before this run). Waits for
# the scene-2 re-render queue so the card is not shared.
set -u
REPO=/c/Users/Ke/Documents/GitHub/RF-3DGS
PYS=/c/Users/Ke/miniconda3/envs/rf-sionna-win/python.exe
DS=RF-3DGS_dataset/regenerated/3dgs_MULTI_24ghz_tut_cs
CK=/mnt/c/Users/Ke/Documents/GitHub/RF-3DGS/RF-3DGS_dataset/blender_visual_trained/chkpnt30000.pth
WSL="wsl.exe -d Ubuntu-22.04 -u ke -- bash /mnt/c/Users/Ke/Documents/GitHub/RF-3DGS/rrf_gsplat/wsl_run.sh"
export PYTHONUTF8=1 MSYS_NO_PATHCONV=1
LOG=$REPO/output/rrf/win_round19.log
cd $REPO
stamp() { echo "== $1 $(date +%H:%M:%S)" >> $LOG; }
run() { local name=$1; shift; [ -f output/rrf/$name/results.json ] && { echo "skip $name" >> $LOG; return; }
        stamp "$name"; $WSL "$name" --checkpoint $CK "$@" >> $LOG 2>&1 < /dev/null; stamp "done $name"; }
until grep -aq "== all done" output/rrf/win_scene2_rerender.log 2>/dev/null; do sleep 15; done
stamp "start"
run m_multi_24_tut_cs_depth_ed_euclid --source $DS --mode multi --delay-depth --delay-depth-mode ED --delay-range euclid --save-renders 640
stamp "baselines"
$PYS rrf_gsplat/eval_baselines.py --multi output/rrf/m_multi_24_tut_cs_depth_ed_euclid --truth $DS \
     --out output/rrf/baselines_m_multi_24_tut_cs_depth_ed_euclid.json < /dev/null 2>&1 | grep -v jitc >> $LOG
stamp "all done"
