#!/bin/bash
# E1 rerun at the verified setting of the released data: CBF at 2.4 GHz with the
# tutorial's materials, global range against per-image range, rgb and db.
set -u
REPO=/c/Users/Ke/Documents/GitHub/RF-3DGS
PYS=/c/Users/Ke/miniconda3/envs/rf-sionna-win/python.exe
SCENE=../sionna_tutorial/RF-3DGS_Sionna_simulation_tutorial/NIST_lobby_v1.0/NIST_lobby_V1.1_sionna12.xml
RXLOC=../sionna_tutorial/RF-3DGS_Sionna_simulation_tutorial/NIST_rx_loc.txt
REG=RF-3DGS_dataset/regenerated
export PYTHONUTF8=1
LOG=$REPO/output/rrf/win_e1_rerun.log
echo "== start $(date)" >> $LOG
cd $REPO/sionna_port
$PYS generate_dataset.py --scene-xml $SCENE --rx-loc-file $RXLOC \
    --out-dir ../$REG/3dgs_CBF_24ghz_tut --spectrum CBF --num-positions 800 \
    --frequency 2.4e9 --materials tutorial --no-dashboard 2>&1 | grep -a "views/s\|global dB" >> $LOG
cd $REPO
$PYS rrf_gsplat/renormalize.py $REG/3dgs_CBF_24ghz_tut $REG/3dgs_CBF_24ghz_tut_gpct --norm global-pct --pct 1 99.99 2>&1 | tail -1 >> $LOG
$PYS rrf_gsplat/renormalize.py $REG/3dgs_CBF_24ghz_tut $REG/3dgs_CBF_24ghz_tut_perview --norm per-view 2>&1 | tail -1 >> $LOG
for d in 3dgs_CBF_24ghz_tut_gpct 3dgs_CBF_24ghz_tut_perview; do
  cp $REG/3dgs_MVDR_100_gpct/train_index.txt $REG/3dgs_MVDR_100_gpct/test_index.txt $REG/$d/
done
echo "== dataset done $(date)" >> $LOG
W="MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu-22.04 -u ke -- bash /mnt/c/Users/Ke/Documents/GitHub/RF-3DGS/rrf_gsplat/wsl_run.sh"
MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu-22.04 -u ke -- bash /mnt/c/Users/Ke/Documents/GitHub/RF-3DGS/rrf_gsplat/wsl_run.sh e1b_cbf24_global  --source $REG/3dgs_CBF_24ghz_tut_gpct    --mode rgb >> $LOG 2>&1
MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu-22.04 -u ke -- bash /mnt/c/Users/Ke/Documents/GitHub/RF-3DGS/rrf_gsplat/wsl_run.sh e1b_cbf24_perview --source $REG/3dgs_CBF_24ghz_tut_perview --mode rgb >> $LOG 2>&1
MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu-22.04 -u ke -- bash /mnt/c/Users/Ke/Documents/GitHub/RF-3DGS/rrf_gsplat/wsl_run.sh e1b_cbf24_global_db --source $REG/3dgs_CBF_24ghz_tut_gpct --mode db >> $LOG 2>&1
echo "== all done $(date)" >> $LOG
tail -8 $LOG
