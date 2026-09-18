#!/bin/bash
# The whole stage-2 queue: re-validate the fast SH path on the released data,
# then the matrix. Re-launchable; finished runs are skipped.
REPO=/mnt/c/Users/Ke/Documents/GitHub/RF-3DGS
cd $REPO
if [ ! -f output/rrf/c0b_released_mvdr_rgb/results.json ]; then
  bash rrf_gsplat/wsl_run.sh c0b_released_mvdr_rgb --source RF-3DGS_dataset/training-rf-spectrum/3dgs_MVDR_100 --mode rgb
fi
bash rrf_gsplat/run_matrix.sh colour norm ablate txmove
echo "night queue done $(date)"
