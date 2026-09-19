#!/bin/bash
# Scene 2 visual 3DGS training only (the render exists). Two lessons from the
# first attempts: (1) the fork's Blender loader built PIL images from int8
# (patched to uint8); (2) at 1600x900 the run pinned the 12 GB card at 100 %
# for two hours without reaching a save -- memory oversubscription. So: half
# resolution (-r 2 -> 800x450, plenty for a textured synthetic scene), saves
# and test PSNR at 7k and 30k, and the trainer's own stdout kept in a file so
# progress is visible.
set -u
REPO=/c/Users/Ke/Documents/GitHub/RF-3DGS
PYG=/c/Users/Ke/miniconda3/envs/rf-3dgs/python.exe
export PYTHONUTF8=1 MSYS_NO_PATHCONV=1 PYTHONUNBUFFERED=1
LOG=$REPO/output/scene2_visual.log
cd $REPO
stamp() { echo "== $1 $(date +%H:%M:%S)" >> $LOG; }
sed -i 's/np.array(arr\*255.0, dtype=np.byte)/np.array(arr*255.0, dtype=np.uint8)/' scene/dataset_readers.py
rm -rf output/scene2_visual
stamp "train visual 3DGS (30k, resolution 1/2, saves at 7k and 30k)"
$PYG train.py -s scene2/visual_dataset -m output/scene2_visual -r 2 --eval --iterations 30000 \
    --test_iterations 7000 30000 --save_iterations 7000 30000 --checkpoint_iterations 30000 \
    < /dev/null > output/scene2_visual_train_stdout.log 2>&1
grep -a "PSNR\|Error\|Traceback\|Number of points" output/scene2_visual_train_stdout.log | tail -12 >> $LOG
stamp "train done"
mkdir -p scene2/visual_trained && cp output/scene2_visual/chkpnt30000.pth scene2/visual_trained/chkpnt30000.pth 2>/dev/null && stamp "checkpoint copied"
$PYG -c "
import torch; m,it=torch.load('scene2/visual_trained/chkpnt30000.pth', weights_only=False, map_location='cpu'); print('gaussians', m[1].shape[0], 'iteration', it)" >> $LOG 2>&1
stamp "all done"
