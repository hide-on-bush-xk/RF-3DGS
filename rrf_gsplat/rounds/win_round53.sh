#!/bin/bash
# Round 53: is the field's peak failure specific to the MVDR target? (M1 vs capacity; docs/tech_paths.md 0b)
# The same scene, transmitter, positions (--poses-from the MVDR dataset) and ray tracer, but an ADDITIVE target: the
# power splat of MULTI (every path's power at its angle of arrival, channel 0; float64 path sums are not involved --
# the projection family has no covariance). The same frozen-geometry SH3 field, the same benchmarks:
#   capacity   160 positions, 3000 steps, scored on the training views (diag_train_fit.py)   [MVDR: <= 1 deg 19.2 %]
#   held-out   all 640 positions, 2500 steps x 4 faces, against NN from the same positions (t4_score.py)
#              [MVDR: field 5.56 deg / NN 1.29 deg]
# Reading, written before the runs: if the field's capacity-benchmark <= 1 deg on the power target is >= 50 %, the
# MVDR target's non-additivity (M1) is the main limiter and P1 (render power, MVDR as a known layer) is the path;
# if it stays <= 30 %, the Gaussians' capacity limits both targets and P7 (position-conditioned colour) comes first.
# In between: both. The held-out comparison with NN is reported beside it (MULTI at 2.4 GHz: the field beat NN).
set -u
REPO=/c/Users/Ke/Documents/GitHub/RF-3DGS
PYW=/c/Users/Ke/miniconda3/envs/rf-gsplat-win/python.exe
PYS=/c/Users/Ke/miniconda3/envs/rf-sionna-win/python.exe
MVD=RF-3DGS_dataset/regenerated/3dgs_MVDR_100_f64_gpct
MUL=RF-3DGS_dataset/regenerated/3dgs_MULTI_60
POW=RF-3DGS_dataset/regenerated/3dgs_POW_60
POWG=RF-3DGS_dataset/regenerated/3dgs_POW_60_gpct
export PYTHONUTF8=1
LOG=$REPO/output/rrf/win_round53.log
cd $REPO
echo "== round 53 start $(date +%H:%M:%S)" >> $LOG
if [ ! -f $MUL/generation_meta.json ]; then
  T0=$(date +%s)
  (cd sionna_port && $PYS generate_dataset.py --scene-xml ../sionna_tutorial/RF-3DGS_Sionna_simulation_tutorial/NIST_lobby_v1.0/NIST_lobby_V1.1_sionna12.xml \
      --rx-loc-file ../sionna_tutorial/RF-3DGS_Sionna_simulation_tutorial/NIST_rx_loc.txt --out-dir ../$MUL --spectrum MULTI \
      --poses-from ../$MVD/sparse/0/images.txt --no-dashboard) > output/rrf/r53_generate.log 2>&1
  echo "MULTI generated in $(( $(date +%s) - T0 )) s: $(grep -a 'channel ranges' output/rrf/r53_generate.log | cut -c1-120)" >> $LOG
fi
if [ ! -f $POWG/generation_meta.json ]; then
  $PYS sionna_port/derive_channel.py $MUL $POW --channel 0 >> $LOG 2>&1
  cp $MVD/train_index.txt $MVD/test_index.txt $POW/
  $PYW rrf_gsplat/renormalize.py $POW $POWG --norm global-pct >> $LOG 2>&1
  cp $MVD/train_index.txt $MVD/test_index.txt $POWG/
fi
B="--mode db --source $POWG --sh-backend gsplat --eval-group --faces-per-step 4 --lr-scale 2 --save-renders -1 --sh-degree 3 --seed 0"
if [ ! -f output/rrf/r53_pow_fit_160/results.json ]; then
  mkdir -p output/rrf/r53_pow_fit_160
  $PYW rrf_gsplat/train_rrf.py --out output/rrf/r53_pow_fit_160 $B --iterations 3000 --eval-every 3000 --max-train-views 640 > output/rrf/r53_pow_fit_160/train.log 2>&1
fi
echo "r53_pow_fit_160: $(tail -1 output/rrf/r53_pow_fit_160/train.log | cut -c1-100)" >> $LOG
$PYW rrf_gsplat/diag_train_fit.py --run output/rrf/r53_pow_fit_160 --train-subset 640 >> $LOG 2>&1
if [ ! -f output/rrf/r53_pow_full/results.json ]; then
  mkdir -p output/rrf/r53_pow_full
  $PYW rrf_gsplat/train_rrf.py --out output/rrf/r53_pow_full $B --iterations 2500 --eval-every 2500 > output/rrf/r53_pow_full/train.log 2>&1
fi
echo "r53_pow_full: $(tail -1 output/rrf/r53_pow_full/train.log | cut -c1-100)" >> $LOG
$PYW rrf_gsplat/diag_train_fit.py --run output/rrf/r53_pow_full --positions 160 >> $LOG 2>&1
[ -d output/rrf/t4pow_nn ] || $PYS sionna_port/t4_baselines.py --truth $POWG --which nn --out-prefix output/rrf/t4pow_ >> $LOG 2>&1
$PYS sionna_port/t4_score.py --truth $POWG --runs output/rrf/r53_pow_full output/rrf/t4pow_nn --cov output/rrf/t4_rt_cov_all.npz \
    --out output/rrf/t4_scores_pow.json >> $LOG 2>&1
echo "== all done $(date +%H:%M:%S)" >> $LOG
