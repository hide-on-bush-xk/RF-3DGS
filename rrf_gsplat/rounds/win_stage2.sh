#!/bin/bash
# Stage 2 (protocol_v1): can the frozen, visually placed Gaussians represent the angular power itself?
# Target 3dgs_APS_60: the instrument-free angular power spectrum (generate_dataset.py --spectrum APS): iso receive
# pattern, every path's power splatted at its angle of arrival (sigma 1 deg), 4 independent 4M-sample solves averaged
# per position, the soft floor 10 log10(P + N0) with one N0 for the dataset (median view maximum - 40 dB), dB stored
# and trained. Same poses as the MVDR datasets (--poses-from), protocol_v1's split.
# Checked before this script (sionna_port/check_aps.py, output/rrf/check_aps.json, 12 stratified positions):
#   seed-to-seed lit RMSE 0.38 dB (<= 1: pass), face seams 0.00 dB (<= 1: pass), main peak <= 1 deg in 72.9 % of views
#   (>= 90 %: FAIL -- near-ties on weak diffuse faces, not Monte-Carlo noise: 4 x 4M did not move it; reported, the
#   threshold was not changed); a single path from a known direction comes back at that direction (max 0.40 deg).
#
#   capacity   160 positions (route subset of protocol train), 3000 steps, scored on those training views
#              (diag_train_fit.py) -- the rounds 51-56 benchmark: --mode power (the APS is additive in linear
#              power) and --mode db. [MVDR db 19.2 %, Bartlett power 10.9 %, richest colour models <= 26 %]
#   held-out   protocol train (467 positions), 2500 steps x 4 faces, --mode power, scored on val (val_random +
#              val_segment, and each alone) against const / NN / IDW k = 2, 4, 8 from the same training positions,
#              and against a second draw of the val positions (--seed 43, the same N0): the target's noise floor
#
# THE GATE, written before the runs (the stage plan of 2026-09-24): capacity, power mode, main peak <= 1 deg on the
# training views >= 50 % -> stage 3 (the operator); <= 30 % -> stage 3 is cancelled, go to M3 (energy placement: P4
# ray-traced path supervision, or a per-position latent); in between -> stage 3's smoke only, no training.
# Printed beside it, not deciding it until the user rules on the metric: the same numbers on the views whose true main
# peak is distinct (its best rival >= 3 deg away is >= 1 dB lower).
set -u
REPO=/c/Users/Ke/Documents/GitHub/RF-3DGS
PYW=/c/Users/Ke/miniconda3/envs/rf-gsplat-win/python.exe
PYS=/c/Users/Ke/miniconda3/envs/rf-sionna-win/python.exe
PROT=rrf_gsplat/protocol_v1
AP=RF-3DGS_dataset/regenerated/3dgs_APS_60
APG=RF-3DGS_dataset/regenerated/3dgs_APS_60_gpct
V43=RF-3DGS_dataset/regenerated/aps_val_s43
export PYTHONUTF8=1
LOG=$REPO/output/rrf/win_stage2.log
cd $REPO
echo "== stage 2 start $(date +%H:%M:%S)" >> $LOG
[ -f $AP/generation_meta.json ] || { echo "no $AP: generate it first" >> $LOG; exit 1; }
if [ ! -f $APG/generation_meta.json ]; then
  $PYW rrf_gsplat/renormalize.py $AP $APG --norm global-pct >> $LOG 2>&1
fi
$PYS -c "import sys; sys.path.insert(0, 'rrf_gsplat'); import protocol as PR; PR.check_dataset('$PROT', '$APG'); print('protocol digest ok: $APG')" >> $LOG 2>&1 || exit 1

B="--source $APG --protocol $PROT --eval-set val --sh-backend gsplat --eval-group --faces-per-step 4 --lr-scale 2 --sh-degree 3 --seed 0"
run() {   # name, mode, iterations, subset (0 = all training positions)
  local name=$1 mode=$2 its=$3 sub=$4
  if [ ! -f output/rrf/$name/results.json ]; then
    mkdir -p output/rrf/$name
    local extra="--save-renders -1"; [ "$sub" != 0 ] && extra="--max-train-views $sub --save-renders 0"
    $PYW rrf_gsplat/train_rrf.py --out output/rrf/$name --mode $mode $B --iterations $its --eval-every $its $extra \
        > output/rrf/$name/train.log 2>&1
  fi
  echo "$name: $(tail -1 output/rrf/$name/train.log | cut -c1-110)" >> $LOG
}
# ---- capacity (the gate) ----
run s2_aps_fit_160_power power 3000 640
$PYW rrf_gsplat/diag_train_fit.py --run output/rrf/s2_aps_fit_160_power --train-subset 640 >> $LOG 2>&1
run s2_aps_fit_160_db db 3000 640
$PYW rrf_gsplat/diag_train_fit.py --run output/rrf/s2_aps_fit_160_db --train-subset 640 >> $LOG 2>&1
echo "(gate: power mode <= 1 deg >= 50 % -> stage 3; <= 30 % -> M3; MVDR db 19.2 %, Bartlett power 10.9 %)" >> $LOG

# ---- held-out on val ----
run s2_aps_full_power power 2500 0
$PYS sionna_port/t4_baselines.py --truth $APG --which const,nn,idw --protocol $PROT --eval-set val --out-prefix output/rrf/s2_aps_ >> $LOG 2>&1
# the noise floor: a second draw of the val positions on other lattices, with this dataset's N0
if [ ! -f $V43/generation_meta.json ]; then
  N0=$($PYS -c "import json; print(json.load(open('$AP/generation_meta.json'))['aps_floor_db'])")
  $PYS sionna_port/check_aps.py --make-poses $AP --protocol $PROT --sets val_random val_segment --out ${V43}_poses >> $LOG 2>&1
  (cd sionna_port && $PYS generate_dataset.py --scene-xml ../sionna_tutorial/RF-3DGS_Sionna_simulation_tutorial/NIST_lobby_v1.0/NIST_lobby_V1.1_sionna12.xml \
      --rx-loc-file ../sionna_tutorial/RF-3DGS_Sionna_simulation_tutorial/NIST_rx_loc.txt --out-dir ../$V43 --spectrum APS --rx-pattern iso \
      --samples-per-src 4000000 --aps-repeats 4 --seed 43 --aps-floor-fixed-db $N0 --poses-from ../${V43}_poses/sparse/0/images.txt \
      --no-dashboard) > output/rrf/s2_val_s43_generate.log 2>&1
fi
[ -d output/rrf/s2_aps_draw43/renders ] || $PYS sionna_port/check_aps.py --draw-as-run $V43 ${V43}_poses/names.txt output/rrf/s2_aps_draw43 >> $LOG 2>&1
RUNS="s2_aps_full_power s2_aps_draw43 s2_aps_const s2_aps_nn s2_aps_idw2 s2_aps_idw4 s2_aps_idw8"
for es in val val_random val_segment; do
  echo "-- $es" >> $LOG
  for r in $RUNS; do
    $PYS rrf_gsplat/mvdr_peaks.py --run output/rrf/$r --truth $APG --protocol $PROT --eval-set $es \
        --out output/rrf/peaks_${r}_$es.json >> $LOG 2>&1
  done
done
# ---- level 3: every val view's tap covariance R_coh (t6 re-solves the MVDR f64 dataset's own views: tr38901 array at
# the face's yaw, and checks it reproduces the stored truth), then the beam-gain loss: a beam steered to the
# predicted main peak, on the true channel, in dB below the best beam. The same R serves every target (same poses)
COV=output/rrf/t4_rt_cov_val_all.npz
[ -f $COV ] || $PYS sionna_port/t6_incoherent_mvdr.py --truth RF-3DGS_dataset/regenerated/3dgs_MVDR_100_f64_gpct --positions 0 \
    --protocol $PROT --eval-set val > output/rrf/t6_val.log 2>&1
tail -30 output/rrf/t6_val.log | grep -a -A12 '"label_noise"' >> $LOG
$PYS sionna_port/t4_score.py --truth $APG --runs $(for r in $RUNS; do echo output/rrf/$r; done) --protocol $PROT --eval-set val \
    --cov $COV --out output/rrf/t4_scores_s2_val.json >> $LOG 2>&1
echo "== all done $(date +%H:%M:%S)" >> $LOG
