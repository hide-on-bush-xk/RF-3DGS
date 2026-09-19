#!/bin/bash
# Stage-1 (visual 3DGS) reconstruction quality of both scenes, so a scene-2
# crossover that differs from scene 1 can be attributed to topology and not to
# geometry quality. Scene 2's test PSNR comes from its own training log
# (--eval, 44 held-out frames). The lobby checkpoint was trained on every
# frame (its cfg_args says eval=False and its test folder is empty), so the
# only number available is the PSNR on its training views, reported as such.
# Waits for scene 2's checkpoint, then renders the lobby's views.
set -u
REPO=/c/Users/Ke/Documents/GitHub/RF-3DGS
PYG=/c/Users/Ke/miniconda3/envs/rf-3dgs/python.exe
export PYTHONUTF8=1 MSYS_NO_PATHCONV=1
LOG=$REPO/output/scene2_visual.log
cd $REPO
stamp() { echo "== $1 $(date +%H:%M:%S)" >> $LOG; }
for i in $(seq 1 2880); do [ -f scene2/visual_trained/chkpnt30000.pth ] && break; sleep 5; done
stamp "stage-1 PSNR: lobby checkpoint on its training views (no held-out frames exist)"
rm -rf output/lobby_visual_eval; mkdir -p output/lobby_visual_eval
cp RF-3DGS_dataset/blender_visual_trained/cfg_args output/lobby_visual_eval/ 2>/dev/null
cp -r RF-3DGS_dataset/blender_visual_trained/point_cloud output/lobby_visual_eval/ 2>/dev/null
cp RF-3DGS_dataset/blender_visual_trained/cameras.json output/lobby_visual_eval/ 2>/dev/null
$PYG render.py -m output/lobby_visual_eval -s RF-3DGS_dataset/blender_visual_dataset --skip_test < /dev/null 2>&1 | grep -a "Error\|Traceback\|Rendering" | tail -3 >> $LOG
$PYG metrics.py -m output/lobby_visual_eval < /dev/null 2>&1 | grep -a "PSNR\|SSIM\|LPIPS\|Error\|Traceback" | tail -4 >> $LOG
stamp "stage-1 PSNR done"
