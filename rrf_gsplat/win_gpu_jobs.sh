#!/bin/bash
# Windows-side GPU jobs for stage 2, run from Git Bash once the GPU is free:
#   1. the Tx-B dataset (MVDR, transmitter at the planner's best cell) + its
#      global-range copy using Tx-A's range, so a warm start sees the same dB mapping
#   2. our generator at the tutorial's 2.4 GHz for a like-for-like per-view timing
#   3. the INRIA rasteriser fine-tune (train.py) on the released MVDR data, timed
set -u
REPO=/c/Users/Ke/Documents/GitHub/RF-3DGS
PYS=/c/Users/Ke/miniconda3/envs/rf-sionna-win/python.exe
PYT=/c/Users/Ke/miniconda3/envs/rf-3dgs/python.exe
SCENE=../sionna_tutorial/RF-3DGS_Sionna_simulation_tutorial/NIST_lobby_v1.0/NIST_lobby_V1.1_sionna12.xml
RXLOC=../sionna_tutorial/RF-3DGS_Sionna_simulation_tutorial/NIST_rx_loc.txt
export PYTHONUTF8=1
mkdir -p $REPO/output/rrf
LOG=$REPO/output/rrf/win_gpu_jobs.log
echo "== start $(date)" >> $LOG

# 1. Tx-B
if [ ! -d $REPO/RF-3DGS_dataset/regenerated/3dgs_MVDR_txB_gpct ]; then
  cd $REPO/sionna_port
  $PYS generate_dataset.py --scene-xml $SCENE --rx-loc-file $RXLOC \
      --out-dir ../RF-3DGS_dataset/regenerated/3dgs_MVDR_txB --spectrum MVDR \
      --num-positions 800 --tx 8.2 -5.4 2.0 --no-dashboard 2>&1 | grep -a "views/s\|global dB" >> $LOG
  cd $REPO
  $PYS rrf_gsplat/renormalize.py RF-3DGS_dataset/regenerated/3dgs_MVDR_txB \
      RF-3DGS_dataset/regenerated/3dgs_MVDR_txB_gpct --norm global-pct --range -175.75 -101.52 2>&1 | tail -1 >> $LOG
  # same held-out positions as Tx-A, so the two are comparable
  cp RF-3DGS_dataset/regenerated/3dgs_MVDR_100_gpct/train_index.txt RF-3DGS_dataset/regenerated/3dgs_MVDR_txB_gpct/ 2>/dev/null
  cp RF-3DGS_dataset/regenerated/3dgs_MVDR_100_gpct/test_index.txt RF-3DGS_dataset/regenerated/3dgs_MVDR_txB_gpct/ 2>/dev/null
fi
echo "== txB done $(date)" >> $LOG

# 2. 2.4 GHz timing, 20 positions
cd $REPO/sionna_port
$PYS generate_dataset.py --scene-xml $SCENE --rx-loc-file $RXLOC \
    --out-dir ../output/regen_24ghz_timing --spectrum MVDR --num-positions 20 \
    --frequency 2.4e9 --no-dashboard 2>&1 | grep -a "views/s\|global dB\|paths" | tail -3 >> $LOG
echo "== 2.4 GHz timing done $(date)" >> $LOG

# 3. INRIA rasteriser, released MVDR, 30k -> 40k
cd $REPO
T0=$(date +%s)
$PYT train.py -s RF-3DGS_dataset/training-rf-spectrum/3dgs_MVDR_100 -m output/rrf/inria_released_mvdr \
    --iterations 40000 --start_checkpoint RF-3DGS_dataset/blender_visual_trained/chkpnt30000.pth \
    --eval --test_iterations 40000 --save_iterations 40000 > output/rrf/inria_released_mvdr.log 2>&1
T1=$(date +%s)
echo "== inria train.py wall $((T1-T0)) s $(date)" >> $LOG
grep -a "Evaluating test" output/rrf/inria_released_mvdr.log | tail -1 >> $LOG
echo "== all done $(date)" >> $LOG
tail -8 $LOG
