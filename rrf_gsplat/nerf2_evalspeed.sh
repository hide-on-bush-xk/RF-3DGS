#!/bin/bash
# Time NeRF2's evaluation of eight held-out views at two chunk sizes (WSL, rf-gsplat env).
cd /mnt/c/Users/Ke/Documents/GitHub/RF-3DGS
PY=/home/ke/miniconda3/envs/rf-gsplat/bin/python
for c in 4096 16384; do
  T0=$(date +%s.%N)
  $PY rrf_gsplat/nerf2_pinhole.py --source RF-3DGS_dataset/training-rf-spectrum/3dgs_MVDR_100 --out output/rrf/smoke_nerf2_evalspeed \
      --iterations 1 --eval-views 8 --chunk $c 2>&1 | grep -a "final PSNR\|loaded in"
  echo "chunk $c: $(echo "$(date +%s.%N) - $T0" | bc) s wall for load + 1 step + 8 views"
done
