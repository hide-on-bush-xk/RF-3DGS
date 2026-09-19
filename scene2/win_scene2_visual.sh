#!/bin/bash
# Scene 2, stage 1: render the visual dataset and train the visual 3DGS
# (the geometry every RF run of scene 2 starts from). Waits for round 17,
# then for a card with no game and < 15 % for two minutes.
#   render: route every 2nd position x 4 yaws + 300 random poses, 1600x900,
#           64 spp (about 3 s a frame on the 3060)
#   train:  the fork's train.py without --start_checkpoint is plain 3DGS
#           (densification on, everything trainable); 30k iterations,
#           checkpoint at 30k, --eval so the held-out frames give a PSNR
set -u
REPO=/c/Users/Ke/Documents/GitHub/RF-3DGS
PYS=/c/Users/Ke/miniconda3/envs/rf-sionna-win/python.exe
PYG=/c/Users/Ke/miniconda3/envs/rf-3dgs/python.exe
export PYTHONUTF8=1 MSYS_NO_PATHCONV=1
LOG=$REPO/output/scene2_visual.log
cd $REPO
stamp() { echo "== $1 $(date +%H:%M:%S)" >> $LOG; }
gpu_busy() {
  tasklist 2>/dev/null | grep -qi "Against the Storm" && return 0
  u=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr -d ' \r'); [ "${u:-100}" -ge 15 ]
}
stamp "start"
for i in $(seq 1 2880); do grep -q "all done" output/rrf/win_round17.log 2>/dev/null && break; sleep 5; done
quiet=0; while [ $quiet -lt 24 ]; do if gpu_busy; then quiet=0; else quiet=$((quiet+1)); fi; sleep 5; done
stamp "gpu quiet ($(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader | tr -d '\r'))"
if [ ! -f scene2/visual_dataset/transforms_train.json ]; then
  stamp "render"
  $PYS scene2/render_visual.py --scene scene2/corridor --out C:/Users/Ke/Documents/GitHub/RF-3DGS/scene2/visual_dataset --width 1600 --height 900 --spp 64 --extra 300 --route-every 2 \
      < /dev/null 2>&1 | grep -v "jitc_llvm\|WARN" | tail -4 >> $LOG
  stamp "render done"
fi
stamp "train visual 3DGS (30k)"
$PYG train.py -s scene2/visual_dataset -m output/scene2_visual --eval --iterations 30000 --checkpoint_iterations 30000 --save_iterations 30000 \
    < /dev/null 2>&1 | grep -a "PSNR\|Training progress\|Error\|Traceback\|Number of points\|Evaluating" | tail -12 >> $LOG
stamp "train done"
mkdir -p scene2/visual_trained && cp output/scene2_visual/chkpnt30000.pth scene2/visual_trained/chkpnt30000.pth 2>/dev/null && stamp "checkpoint copied"
$PYG -c "
import torch; m,it=torch.load('scene2/visual_trained/chkpnt30000.pth', weights_only=False, map_location='cpu'); print('gaussians', m[1].shape[0], 'iteration', it)" >> $LOG 2>&1
stamp "all done"
