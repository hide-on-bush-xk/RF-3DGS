#!/bin/bash
# Scene 2 visual 3DGS training only (the render exists). The fork's Blender
# loader builds PIL images from an int8 array (np.byte), which current Pillow
# refuses ("Cannot handle this data type: (1, 1, 3), |i1"); the authors'
# lobby checkpoint was made elsewhere. Patch to uint8 (idempotent), train.
set -u
REPO=/c/Users/Ke/Documents/GitHub/RF-3DGS
PYG=/c/Users/Ke/miniconda3/envs/rf-3dgs/python.exe
export PYTHONUTF8=1 MSYS_NO_PATHCONV=1
LOG=$REPO/output/scene2_visual.log
cd $REPO
stamp() { echo "== $1 $(date +%H:%M:%S)" >> $LOG; }
sed -i 's/np.array(arr\*255.0, dtype=np.byte)/np.array(arr*255.0, dtype=np.uint8)/' scene/dataset_readers.py
grep -q "dtype=np.uint8" scene/dataset_readers.py && stamp "loader patched (np.byte -> np.uint8)"
stamp "train visual 3DGS (30k), retry"
$PYG train.py -s scene2/visual_dataset -m output/scene2_visual --eval --iterations 30000 --checkpoint_iterations 30000 --save_iterations 30000 \
    < /dev/null 2>&1 | grep -a "PSNR\|Error\|Traceback\|Number of points\|Evaluating\|Loading" | tail -20 >> $LOG
stamp "train done"
mkdir -p scene2/visual_trained && cp output/scene2_visual/chkpnt30000.pth scene2/visual_trained/chkpnt30000.pth 2>/dev/null && stamp "checkpoint copied"
$PYG -c "
import torch; m,it=torch.load('scene2/visual_trained/chkpnt30000.pth', weights_only=False, map_location='cpu'); print('gaussians', m[1].shape[0], 'iteration', it)" >> $LOG 2>&1
stamp "all done"
