#!/bin/bash
# MULTI at splat sigma 1 (per-position seeding, one solve per position), then the training.
set -u
REPO=/c/Users/Ke/Documents/GitHub/RF-3DGS
PYS=/c/Users/Ke/miniconda3/envs/rf-sionna-win/python.exe
SCENE=../sionna_tutorial/RF-3DGS_Sionna_simulation_tutorial/NIST_lobby_v1.0/NIST_lobby_V1.1_sionna12.xml
RXLOC=../sionna_tutorial/RF-3DGS_Sionna_simulation_tutorial/NIST_rx_loc.txt
REG=RF-3DGS_dataset/regenerated
export PYTHONUTF8=1
LOG=$REPO/output/rrf/win_multi_s1.log
echo "== start $(date)" >> $LOG
cd $REPO/sionna_port
$PYS generate_dataset.py --scene-xml $SCENE --rx-loc-file $RXLOC --out-dir ../$REG/3dgs_MULTI_24ghz_tut_s1 \
    --spectrum MULTI --num-positions 800 --frequency 2.4e9 --materials tutorial --splat-sigma 1 --no-dashboard 2>&1 | grep -a "views/s\|channel ranges\|Error\|error" >> $LOG
cp ../$REG/3dgs_MVDR_100_gpct/train_index.txt ../$REG/3dgs_MVDR_100_gpct/test_index.txt ../$REG/3dgs_MULTI_24ghz_tut_s1/
echo "== dataset done $(date)" >> $LOG
MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu-22.04 -u ke -- bash /mnt/c/Users/Ke/Documents/GitHub/RF-3DGS/rrf_gsplat/wsl_run.sh m_multi_24_tut_s1 --source $REG/3dgs_MULTI_24ghz_tut_s1 --mode multi --save-renders 640 >> $LOG 2>&1
cd $REPO
$PYS rrf_gsplat/eval_encoding.py --multi output/rrf/m_multi_24_tut_s1 --aod3 output/rrf/m_aod3_24_tut --truth $REG/3dgs_MULTI_24ghz_tut_s1 --out output/rrf/encoding_comparison_s1.json >> $LOG 2>&1
echo "== all done $(date)" >> $LOG
tail -8 $LOG
