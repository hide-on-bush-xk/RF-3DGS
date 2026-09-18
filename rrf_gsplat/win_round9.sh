#!/bin/bash
# Round 9 (Windows bash driver; training in WSL). In ROI order:
#   1. random geometry-subset controls for round 8 (needles 21.32 / discs 21.49
#      vs full 21.53: is the class irrelevant?) -- random 11 %, 3 %, 1 %;
#   2. N_eff per ray on the frozen and unfrozen Tx-A models (section 5.4);
#   3. multi-bounce power share at depth 3 (2.4 GHz both variants, 60 GHz);
#   4. the transfer curve: 8 source transmitters at 0.8 m (indoor, cleared),
#      800 positions each (parity with Tx-A/C), own percentile range, unfrozen
#      db adaptation (10k it), then the adapted geometry frozen with colours
#      reset on Tx-B (eval every 250, as t_txB_geomC_frozen);
#   5. re-attribution A/B: the cs MULTI dataset regenerated with the pre-fix
#      per-view seeds, trained identically (m_multi_24_tut_cs_yawseed).
# Nothing here is a timing measurement; runs may overlap nothing (sequential).
set -u
REPO=/c/Users/Ke/Documents/GitHub/RF-3DGS
PYS=/c/Users/Ke/miniconda3/envs/rf-sionna-win/python.exe
SCENE=../sionna_tutorial/RF-3DGS_Sionna_simulation_tutorial/NIST_lobby_v1.0/NIST_lobby_V1.1_sionna12.xml
RXLOC=../sionna_tutorial/RF-3DGS_Sionna_simulation_tutorial/NIST_rx_loc.txt
REG=RF-3DGS_dataset/regenerated
WSLPY=/home/ke/miniconda3/envs/rf-gsplat/bin/python
WSL="wsl.exe -d Ubuntu-22.04 -u ke -- bash /mnt/c/Users/Ke/Documents/GitHub/RF-3DGS/rrf_gsplat/wsl_run.sh"
export PYTHONUTF8=1 MSYS_NO_PATHCONV=1
LOG=$REPO/output/rrf/win_round9.log
cd $REPO
echo "== start $(date)" >> $LOG
run() { local name=$1; shift; [ -f output/rrf/$name/results.json ] && { echo "skip $name" >> $LOG; return; }
        echo "== $name $(date +%H:%M:%S)" >> $LOG; $WSL "$name" "$@" >> $LOG 2>&1; }

# 1. random-subset controls
run a_mvdr_db_geom_rand11 --source $REG/3dgs_MVDR_100_gpct --mode db --train-geometry --geometry-subset random --geometry-fraction 0.11 --save-renders 0
run a_mvdr_db_geom_rand3  --source $REG/3dgs_MVDR_100_gpct --mode db --train-geometry --geometry-subset random --geometry-fraction 0.03 --save-renders 0
run a_mvdr_db_geom_rand1  --source $REG/3dgs_MVDR_100_gpct --mode db --train-geometry --geometry-subset random --geometry-fraction 0.01 --save-renders 0

# 2. N_eff per ray
echo "== diag_neff $(date +%H:%M:%S)" >> $LOG
wsl.exe -d Ubuntu-22.04 -u ke -- bash -c "cd /mnt/c/Users/Ke/Documents/GitHub/RF-3DGS && $WSLPY rrf_gsplat/diag_neff.py --runs e2_mvdr_db a_mvdr_db_geom a_mvdr_db_geom_discs" >> $LOG 2>&1

# 3. multi-bounce share
echo "== multibounce $(date +%H:%M:%S)" >> $LOG
$PYS rrf_gsplat/multibounce_fraction.py 2>&1 | grep -v "jitc_llvm\|WARN" >> $LOG

# 4. transfer curve. name x y z; distances to Tx-B (8.2, -5.4, 2.0): D 1.8, E 3.2, N 4.2, L 5.3, O 6.8, H 7.4, M 9.4, K 12.8 m
SRC="D 6.9 -5.4 0.8
E 6.9 -2.7 0.8
N 5.0 -3.0 0.8
L 3.5 -7.5 0.8
O 2.0 -3.0 0.8
H 1.98 -9.27 0.8
M -1.0 -7.0 0.8
K -3.35 0.0 0.8"
echo "$SRC" | while read NAME X Y Z; do
  DS=3dgs_MVDR_tx${NAME}
  if [ ! -d $REG/${DS}_gpct ]; then
    echo "== dataset $DS ($X $Y $Z) $(date +%H:%M:%S)" >> $LOG
    cd $REPO/sionna_port
    $PYS generate_dataset.py --scene-xml $SCENE --rx-loc-file $RXLOC --out-dir ../$REG/$DS --spectrum MVDR \
        --num-positions 800 --tx $X $Y $Z --no-dashboard 2>&1 | grep -a "views/s\|global dB\|Error" >> $LOG
    $PYS ../rrf_gsplat/renormalize.py ../$REG/$DS ../$REG/${DS}_gpct --norm global-pct --pct 1 99.99 2>&1 | tail -1 >> $LOG
    cp ../$REG/3dgs_MVDR_100_gpct/train_index.txt ../$REG/3dgs_MVDR_100_gpct/test_index.txt ../$REG/${DS}_gpct/
    cd $REPO
  fi
  run a_tx${NAME}_db_geom --source $REG/${DS}_gpct --mode db --train-geometry --save-renders 0
  run t_txB_geom${NAME}_frozen --source $REG/3dgs_MVDR_txB_gpct --mode db --eval-every 250 --save-renders 0 \
      --init-from /mnt/c/Users/Ke/Documents/GitHub/RF-3DGS/output/rrf/a_tx${NAME}_db_geom/rrf_state.pt --init-geometry-only
done

# 5. re-attribution A/B (per-view seeds, everything else as 3dgs_MULTI_24ghz_tut_cs)
if [ ! -d $REG/3dgs_MULTI_24ghz_tut_cs_yawseed ]; then
  echo "== dataset cs_yawseed $(date +%H:%M:%S)" >> $LOG
  cd $REPO/sionna_port
  $PYS generate_dataset.py --scene-xml $SCENE --rx-loc-file $RXLOC --out-dir ../$REG/3dgs_MULTI_24ghz_tut_cs_yawseed \
      --spectrum MULTI --num-positions 800 --frequency 2.4e9 --materials tutorial --splat-sigma 3 --seed-per-view --no-dashboard 2>&1 \
      | grep -a "views/s\|channel ranges\|Error\|error" >> $LOG
  cp ../$REG/3dgs_MVDR_100_gpct/train_index.txt ../$REG/3dgs_MVDR_100_gpct/test_index.txt ../$REG/3dgs_MULTI_24ghz_tut_cs_yawseed/
  cd $REPO
fi
run m_multi_24_tut_cs_yawseed --source $REG/3dgs_MULTI_24ghz_tut_cs_yawseed --mode multi --save-renders 640
$PYS rrf_gsplat/eval_encoding.py --multi output/rrf/m_multi_24_tut_cs_yawseed --aod3 output/rrf/m_aod3_24_tut \
    --truth $REG/3dgs_MULTI_24ghz_tut_cs_yawseed --out output/rrf/encoding_comparison_cs_yawseed.json >> $LOG 2>&1
echo "== all done $(date)" >> $LOG
