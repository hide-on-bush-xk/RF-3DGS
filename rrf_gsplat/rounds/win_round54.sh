#!/bin/bash
# Round 54: the clean additive-target control for P1's premise (rounds 53 / 53b were confounded by the power splat's
# -200 dB floor and Monte-Carlo roughness). Target: the Bartlett spectrum of the SAME channels, sqrt(a^H R a) with the
# delay taps as snapshots (generate_dataset.py --spectrum CBF --cbf-variant fixed): additive over taps in linear
# power, smooth (per-view span ~30 dB, pixel roughness 0.12 dB in the 2-position smoke vs MVDR's 0.06 and the power
# splat's 0.34), no path-free floor. Same scene, transmitter, poses (--poses-from the MVDR dataset), split, field.
#   capacity   160 positions, 3000 steps, training views (diag_train_fit.py), --mode power and --mode db
#              [MVDR, db: <= 1 deg 19.2 %]
#   held-out   all positions, 2500 steps x 4 faces, --mode power, against NN from the same positions
# Reading, written before the runs (as round 53's): power-mode capacity <= 1 deg >= 50 % -> the MVDR target's
# non-additivity (M1) is the main limiter and P1 is the path; <= 30 % -> the Gaussians' capacity limits both targets
# (P7 first); in between -> both. Caveat for the reading: Bartlett peaks are ~11 deg wide (10 x 10 array), so a
# <= 1 deg argmax is harder to pin on a broad top than on MVDR's sharp one -- beam loss and the NN comparison are
# reported beside it.
set -u
REPO=/c/Users/Ke/Documents/GitHub/RF-3DGS
PYW=/c/Users/Ke/miniconda3/envs/rf-gsplat-win/python.exe
PYS=/c/Users/Ke/miniconda3/envs/rf-sionna-win/python.exe
MVD=RF-3DGS_dataset/regenerated/3dgs_MVDR_100_f64_gpct
CB=RF-3DGS_dataset/regenerated/3dgs_CBFF_60
CBG=RF-3DGS_dataset/regenerated/3dgs_CBFF_60_gpct
export PYTHONUTF8=1
LOG=$REPO/output/rrf/win_round54.log
cd $REPO
echo "== round 54 start $(date +%H:%M:%S)" >> $LOG
if [ ! -f $CB/generation_meta.json ]; then
  T0=$(date +%s)
  (cd sionna_port && $PYS generate_dataset.py --scene-xml ../sionna_tutorial/RF-3DGS_Sionna_simulation_tutorial/NIST_lobby_v1.0/NIST_lobby_V1.1_sionna12.xml \
      --rx-loc-file ../sionna_tutorial/RF-3DGS_Sionna_simulation_tutorial/NIST_rx_loc.txt --out-dir ../$CB --spectrum CBF --cbf-variant fixed \
      --poses-from ../$MVD/sparse/0/images.txt --no-dashboard) > output/rrf/r54_generate.log 2>&1
  echo "CBF fixed generated in $(( $(date +%s) - T0 )) s: $(grep -a 'global dB range' output/rrf/r54_generate.log)" >> $LOG
fi
if [ ! -f $CBG/generation_meta.json ]; then
  cp $MVD/train_index.txt $MVD/test_index.txt $CB/
  $PYW rrf_gsplat/renormalize.py $CB $CBG --norm global-pct >> $LOG 2>&1
  cp $MVD/train_index.txt $MVD/test_index.txt $CBG/
fi
run() {   # name, mode, iterations, subset (0 = all), scoring
  local name=$1 mode=$2 its=$3 sub=$4
  if [ ! -f output/rrf/$name/results.json ]; then
    mkdir -p output/rrf/$name
    local extra=""; [ "$sub" != 0 ] && extra="--max-train-views $sub"
    $PYW rrf_gsplat/train_rrf.py --out output/rrf/$name --mode $mode --source $CBG --sh-backend gsplat --eval-group --faces-per-step 4 \
        --lr-scale 2 --save-renders -1 --sh-degree 3 --seed 0 --iterations $its --eval-every $its $extra > output/rrf/$name/train.log 2>&1
  fi
  echo "$name: $(tail -1 output/rrf/$name/train.log | cut -c1-100)" >> $LOG
  if [ "$sub" != 0 ]; then $PYW rrf_gsplat/diag_train_fit.py --run output/rrf/$name --train-subset $sub >> $LOG 2>&1
  else $PYW rrf_gsplat/diag_train_fit.py --run output/rrf/$name --positions 160 >> $LOG 2>&1; fi
}
run r54_cbf_fit_160_power power 3000 640
run r54_cbf_fit_160_db db 3000 640
run r54_cbf_full_power power 2500 0
[ -d output/rrf/t4cbf_nn ] || $PYS sionna_port/t4_baselines.py --truth $CBG --which nn --out-prefix output/rrf/t4cbf_ >> $LOG 2>&1
$PYS sionna_port/t4_score.py --truth $CBG --runs output/rrf/r54_cbf_full_power output/rrf/t4cbf_nn --cov output/rrf/t4_rt_cov_all.npz \
    --out output/rrf/t4_scores_cbf.json >> $LOG 2>&1
echo "== all done $(date +%H:%M:%S)" >> $LOG
