#!/bin/bash
# Round 7 queue (Windows side, WSL for training):
#   T2  power-mode unfreezing gain (db frozen/unfrozen and power frozen exist)
#   T1  Tx-C adapted geometry carried to Tx-B
#   sigma sweep of the (cos, sin) multi-channel target: 1, 6, 13 px (3 px exists)
#   same-channel-count A/B: five channels with a seamed azimuth + a duplicate
set -u
REPO=/c/Users/Ke/Documents/GitHub/RF-3DGS
PYS=/c/Users/Ke/miniconda3/envs/rf-sionna-win/python.exe
SCENE=../sionna_tutorial/RF-3DGS_Sionna_simulation_tutorial/NIST_lobby_v1.0/NIST_lobby_V1.1_sionna12.xml
RXLOC=../sionna_tutorial/RF-3DGS_Sionna_simulation_tutorial/NIST_rx_loc.txt
REG=RF-3DGS_dataset/regenerated
WSL="wsl.exe -d Ubuntu-22.04 -u ke -- bash /mnt/c/Users/Ke/Documents/GitHub/RF-3DGS/rrf_gsplat/wsl_run.sh"
export PYTHONUTF8=1 MSYS_NO_PATHCONV=1
LOG=$REPO/output/rrf/win_round7.log
echo "== start $(date)" >> $LOG

# ---- datasets ----------------------------------------------------------
cd $REPO/sionna_port
if [ ! -d ../$REG/3dgs_MVDR_txC_gpct ]; then
  $PYS generate_dataset.py --scene-xml $SCENE --rx-loc-file $RXLOC --out-dir ../$REG/3dgs_MVDR_txC \
      --spectrum MVDR --num-positions 800 --tx 0.0 -3.0 2.0 --no-dashboard 2>&1 | grep -a "views/s\|global dB" >> $LOG
  $PYS ../rrf_gsplat/renormalize.py ../$REG/3dgs_MVDR_txC ../$REG/3dgs_MVDR_txC_gpct --norm global-pct --pct 1 99.99 2>&1 | tail -1 >> $LOG
  cp ../$REG/3dgs_MVDR_100_gpct/train_index.txt ../$REG/3dgs_MVDR_100_gpct/test_index.txt ../$REG/3dgs_MVDR_txC_gpct/
fi
for S in 1 6 13; do
  if [ ! -d ../$REG/3dgs_MULTI_24ghz_tut_cs_s$S ]; then
    $PYS generate_dataset.py --scene-xml $SCENE --rx-loc-file $RXLOC --out-dir ../$REG/3dgs_MULTI_24ghz_tut_cs_s$S \
        --spectrum MULTI --num-positions 800 --frequency 2.4e9 --materials tutorial --splat-sigma $S --no-dashboard 2>&1 | grep -a "views/s\|channel ranges" >> $LOG
    cp ../$REG/3dgs_MVDR_100_gpct/train_index.txt ../$REG/3dgs_MVDR_100_gpct/test_index.txt ../$REG/3dgs_MULTI_24ghz_tut_cs_s$S/
  fi
done
cd $REPO
$PYS rrf_gsplat/make_lin5.py $REG/3dgs_MULTI_24ghz_tut_cs $REG/3dgs_MULTI_24ghz_tut_lin5 2>&1 | tail -1 >> $LOG
echo "== datasets done $(date)" >> $LOG

# ---- trainings, by decision weight ------------------------------------
run() { local name=$1; shift; [ -f output/rrf/$name/results.json ] && { echo "skip $name" >> $LOG; return; }
        echo "== $name $(date +%H:%M:%S)" >> $LOG; $WSL "$name" "$@" >> $LOG 2>&1; }
run a_mvdr_power_geom  --source $REG/3dgs_MVDR_100_gpct --mode power --train-geometry
run a_txC_db_geom      --source $REG/3dgs_MVDR_txC_gpct --mode db --train-geometry
run t_txB_geomC_frozen --source $REG/3dgs_MVDR_txB_gpct --mode db --eval-every 250 \
                       --init-from /mnt/c/Users/Ke/Documents/GitHub/RF-3DGS/output/rrf/a_txC_db_geom/rrf_state.pt --init-geometry-only
for S in 1 6 13; do
  run m_multi_24_tut_cs_s$S --source $REG/3dgs_MULTI_24ghz_tut_cs_s$S --mode multi --save-renders 640
  $PYS rrf_gsplat/eval_encoding.py --multi output/rrf/m_multi_24_tut_cs_s$S --aod3 output/rrf/m_aod3_24_tut \
      --truth $REG/3dgs_MULTI_24ghz_tut_cs_s$S --out output/rrf/encoding_comparison_cs_s$S.json >> $LOG 2>&1
done
run m_multi_24_tut_lin5 --source $REG/3dgs_MULTI_24ghz_tut_lin5 --mode multi --save-renders 640
$PYS rrf_gsplat/eval_encoding.py --multi output/rrf/m_multi_24_tut_lin5 --aod3 output/rrf/m_aod3_24_tut \
    --truth $REG/3dgs_MULTI_24ghz_tut_lin5 --out output/rrf/encoding_comparison_lin5.json >> $LOG 2>&1
echo "== all done $(date)" >> $LOG
tail -30 $LOG
