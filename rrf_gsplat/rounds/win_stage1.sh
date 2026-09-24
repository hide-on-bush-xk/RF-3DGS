#!/bin/bash
# Stage 1 (protocol_v1): last night's schemes re-verified on the validation set (val_random = the old 160 held-out
# positions' random half, interpolation; val_segment = a held-out stretch of route, 1.7 m from training), trained on
# protocol train only (467 positions). Selection metrics: main-peak direction median and the beam-gain loss on the
# true channel (level 3, t4_rt_cov_val_all.npz from win_stage2.sh), not PSNR.
#   anchors    round 45's SH3 db on the stored MVDR labels (r45_base) and on the float64 labels (r50_f64), 3 seeds each
#   P7         --pcolor 8 (position-conditioned colour, round 55), stored labels, 3 seeds -- against IDW
#   1 seed     SH4, one lobe kappa 100, --peak-loss 1 (stored labels, as rounds 51 / 52 / 56): on the capacity
#              benchmark all three sat at 19-26 %, so they are table rows, not candidates
#   look-ups   const / NN / IDW k = 2, 4, 8 (t4_baselines.py, run before this script: output/rrf/s1_{mvdr,f64}_*)
#   cited      3GPP InH and the geometric LOS from T4 (12.7 / 8.2 deg on the old held-out set) -- not rerun
# Minimum difference that counts (round 50's seed spread): 3 points of <= 1 deg, 0.3 deg of direction median.
# Skipped: LM and the per-Gaussian backward (engineering, no effect on peaks in rounds 45-46).
set -u
REPO=/c/Users/Ke/Documents/GitHub/RF-3DGS
PYW=/c/Users/Ke/miniconda3/envs/rf-gsplat-win/python.exe
PYS=/c/Users/Ke/miniconda3/envs/rf-sionna-win/python.exe
PROT=rrf_gsplat/protocol_v1
MV=RF-3DGS_dataset/regenerated/3dgs_MVDR_100_gpct
MF=RF-3DGS_dataset/regenerated/3dgs_MVDR_100_f64_gpct
COV=output/rrf/t4_rt_cov_val_all.npz
export PYTHONUTF8=1
LOG=$REPO/output/rrf/win_stage1.log
cd $REPO
echo "== stage 1 start $(date +%H:%M:%S)" >> $LOG
B="--mode db --protocol $PROT --eval-set val --sh-backend gsplat --eval-group --faces-per-step 4 --lr-scale 2 --sh-degree 3 \
--iterations 2500 --eval-every 2500 --save-renders -1"
run() {   # name, source, extra args...
  local name=$1 src=$2; shift 2
  if [ ! -f output/rrf/$name/results.json ]; then
    mkdir -p output/rrf/$name
    $PYW rrf_gsplat/train_rrf.py --out output/rrf/$name --source $src $B "$@" > output/rrf/$name/train.log 2>&1
  fi
  echo "$name: $(tail -1 output/rrf/$name/train.log | cut -c1-110)" >> $LOG
}
for s in 0 1 2; do
  run s1_sh3_mvdr_s$s $MV --seed $s
  run s1_sh3_f64_s$s $MF --seed $s
  run s1_pc8_mvdr_s$s $MV --seed $s --pcolor 8
done
run s1_sh4_mvdr $MV --seed 0 --sh-degree 4
run s1_k100_mvdr $MV --seed 0 --lobes 1 --lobe-kappa 100
run s1_peak_mvdr $MV --seed 0 --peak-loss 1
score() {   # truth, runs...
  local truth=$1; shift
  for es in val val_random val_segment; do
    echo "-- $truth $es" >> $LOG
    for r in "$@"; do
      $PYS rrf_gsplat/mvdr_peaks.py --run output/rrf/$r --truth $truth --protocol $PROT --eval-set $es \
          --out output/rrf/peaks_${r}_$es.json >> $LOG 2>&1
    done
  done
}
MVRUNS="s1_sh3_mvdr_s0 s1_sh3_mvdr_s1 s1_sh3_mvdr_s2 s1_pc8_mvdr_s0 s1_pc8_mvdr_s1 s1_pc8_mvdr_s2 s1_sh4_mvdr s1_k100_mvdr s1_peak_mvdr"
MFRUNS="s1_sh3_f64_s0 s1_sh3_f64_s1 s1_sh3_f64_s2"
score $MV $MVRUNS
score $MF $MFRUNS
if [ -f $COV ]; then
  $PYS sionna_port/t4_score.py --truth $MV --protocol $PROT --eval-set val --cov $COV --out output/rrf/t4_scores_s1_mvdr_val.json \
      --runs $(for r in $MVRUNS s1_mvdr_const s1_mvdr_nn s1_mvdr_idw2 s1_mvdr_idw4 s1_mvdr_idw8; do echo output/rrf/$r; done) >> $LOG 2>&1
  $PYS sionna_port/t4_score.py --truth $MF --protocol $PROT --eval-set val --cov $COV --out output/rrf/t4_scores_s1_f64_val.json \
      --runs $(for r in $MFRUNS s1_f64_const s1_f64_nn s1_f64_idw2 s1_f64_idw4 s1_f64_idw8; do echo output/rrf/$r; done) >> $LOG 2>&1
else
  echo "no $COV: beam-gain loss MISSING (run win_stage2.sh's level-3 step first)" >> $LOG
fi
echo "== all done $(date +%H:%M:%S)" >> $LOG
