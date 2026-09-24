#!/bin/bash
# Materials ablation, Windows side: wait for the GPU, generate the MVDR dataset
# with the tutorial's per-material definitions, remap it, then run the two
# trainings in WSL.
set -u
REPO=/c/Users/Ke/Documents/GitHub/RF-3DGS
PYS=/c/Users/Ke/miniconda3/envs/rf-sionna-win/python.exe
SCENE=../sionna_tutorial/RF-3DGS_Sionna_simulation_tutorial/NIST_lobby_v1.0/NIST_lobby_V1.1_sionna12.xml
RXLOC=../sionna_tutorial/RF-3DGS_Sionna_simulation_tutorial/NIST_rx_loc.txt
export PYTHONUTF8=1
LOG=$REPO/output/rrf/win_materials_job.log
echo "== start $(date)" >> $LOG
# the CBF db training holds the GPU; the generator wants most of it
for i in $(seq 1 120); do
  [ -f $REPO/output/rrf/e2_cbf_db/results.json ] && break
  sleep 5
done
echo "== gpu free $(date)" >> $LOG
cd $REPO/sionna_port
$PYS generate_dataset.py --scene-xml $SCENE --rx-loc-file $RXLOC \
    --out-dir ../RF-3DGS_dataset/regenerated/3dgs_MVDR_tut --spectrum MVDR \
    --num-positions 800 --materials tutorial --no-dashboard 2>&1 | grep -a "views/s\|global dB\|tutorial materials" >> $LOG
cd $REPO
$PYS rrf_gsplat/renormalize.py RF-3DGS_dataset/regenerated/3dgs_MVDR_tut \
    RF-3DGS_dataset/regenerated/3dgs_MVDR_tut_gpct --norm global-pct --pct 1 99.99 2>&1 | tail -2 >> $LOG
cp RF-3DGS_dataset/regenerated/3dgs_MVDR_100_gpct/train_index.txt RF-3DGS_dataset/regenerated/3dgs_MVDR_tut_gpct/
cp RF-3DGS_dataset/regenerated/3dgs_MVDR_100_gpct/test_index.txt RF-3DGS_dataset/regenerated/3dgs_MVDR_tut_gpct/
echo "== dataset done $(date)" >> $LOG
MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu-22.04 -u ke -- bash /mnt/c/Users/Ke/Documents/GitHub/RF-3DGS/rrf_gsplat/run_matrix.sh materials >> $LOG 2>&1
echo "== all done $(date)" >> $LOG
tail -12 $LOG
