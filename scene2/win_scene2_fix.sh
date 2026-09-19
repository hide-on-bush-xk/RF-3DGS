#!/bin/bash
# Scene 2 stage 1, third pass: the camera convention passed to Mitsuba was
# rotated 180 deg about the optical axis (y,z flip instead of x,z; see the
# comment in render_visual.py and output/scene2_densify_test/conv_*.png), so
# the two earlier datasets were unfittable (loss flat at 0.2, 17-18 dB, 62-74k
# Gaussians). Same render and training settings as the redo, corrected frames.
# Then the RF queue (which waits for the checkpoint) is launched.
set -u
REPO=/c/Users/Ke/Documents/GitHub/RF-3DGS
PYS=/c/Users/Ke/miniconda3/envs/rf-sionna-win/python.exe
PYG=/c/Users/Ke/miniconda3/envs/rf-3dgs/python.exe
export PYTHONUTF8=1 MSYS_NO_PATHCONV=1 PYTHONUNBUFFERED=1
LOG=$REPO/output/scene2_visual.log
cd $REPO
stamp() { echo "== $1 $(date +%H:%M:%S)" >> $LOG; }
gpu_busy() {
  tasklist 2>/dev/null | grep -qi "Against the Storm" && return 0
  u=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr -d ' \r'); [ "${u:-100}" -ge 15 ]
}
quiet=0; while [ $quiet -lt 12 ]; do if gpu_busy; then quiet=0; else quiet=$((quiet+1)); fi; sleep 5; done
rm -rf scene2/visual_dataset_rot180 scene2/visual_trained_rot180 output/scene2_visual_rot180
mv scene2/visual_dataset scene2/visual_dataset_rot180 2>/dev/null; mv scene2/visual_trained scene2/visual_trained_rot180 2>/dev/null; mv output/scene2_visual output/scene2_visual_rot180 2>/dev/null
stamp "fix: render 800x450 with the x,z column flip (Mitsuba camera x = left)"
$PYS scene2/render_visual.py --scene scene2/corridor --out C:/Users/Ke/Documents/GitHub/RF-3DGS/scene2/visual_dataset --width 800 --height 450 --spp 64 --extra 300 --route-every 4 \
    < /dev/null 2>&1 | grep -v "jitc_llvm\|WARN" | tail -2 >> $LOG
stamp "fix: train (30k, saves at 7k and 30k)"
$PYG train.py -s scene2/visual_dataset -m output/scene2_visual --eval --iterations 30000 \
    --test_iterations 7000 30000 --save_iterations 7000 30000 --checkpoint_iterations 30000 \
    < /dev/null > output/scene2_visual_train_stdout.log 2>&1
grep -a "PSNR\|Error\|Traceback\|Number of points" output/scene2_visual_train_stdout.log | tail -8 >> $LOG
mkdir -p scene2/visual_trained && cp output/scene2_visual/chkpnt30000.pth scene2/visual_trained/chkpnt30000.pth 2>/dev/null && stamp "fix: checkpoint copied"
$PYG -c "
import torch; m,it=torch.load('scene2/visual_trained/chkpnt30000.pth', weights_only=False, map_location='cpu'); print('gaussians', m[1].shape[0], 'iteration', it)" >> $LOG 2>&1
stamp "fix: all done"
bash scene2/win_scene2_rf.sh
