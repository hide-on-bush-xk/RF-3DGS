#!/bin/bash
# Round 50: the MVDR benchmark with float64 labels, next to the old one (nothing replaced).
# The stored labels of 3dgs_MVDR_100(_gpct) are complex64 MVDR on covariances with cond ~3e11: rounding noise in
# the weak directions, up to 35 dB, and a main peak within 1 deg of the noise-free one in only 67.5 % of held-out
# views (round 47). The generator computes MVDR in complex128 since 2d12e13.
#   1. generate 3dgs_MVDR_100_f64 with 3dgs_MVDR_100's configuration and its exact poses (--poses-from)
#   2. renormalize.py --norm global-pct with 3dgs_MVDR_100_gpct's exact dB range -> 3dgs_MVDR_100_f64_gpct, and the
#      same train / test split files
#   3. r50_f64 x 3 seeds: the round-45 SH3 configuration (now with the default NO_GEOM and TF32 off) on the new
#      labels; t4_nn on the new labels
#   4. t4_score.py against the new labels: r50_f64 (trained on clean), r45_tf32off (trained on the old labels,
#      same code path apart from NO_GEOM, which does not change the numbers), NN
# Reading, written before the run: if training on clean labels helps the field's peaks, r50_f64's direction median
# drops or its <= 1 deg share rises by more than the 3-seed range against r45_tf32off (both scored on the new
# labels). Expected: a small effect -- the field underfits for capacity (round 48), and most of the noise is in
# the floor -- but the noise reaches 28 dB within 20 dB of some peaks.
set -u
REPO=/c/Users/Ke/Documents/GitHub/RF-3DGS
PYW=/c/Users/Ke/miniconda3/envs/rf-gsplat-win/python.exe
PYS=/c/Users/Ke/miniconda3/envs/rf-sionna-win/python.exe
OLD=RF-3DGS_dataset/regenerated/3dgs_MVDR_100
NEW=RF-3DGS_dataset/regenerated/3dgs_MVDR_100_f64
NEWG=RF-3DGS_dataset/regenerated/3dgs_MVDR_100_f64_gpct
export PYTHONUTF8=1
LOG=$REPO/output/rrf/win_round50.log
cd $REPO
echo "== round 50 start $(date +%H:%M:%S)" >> $LOG
if [ ! -f $NEW/generation_meta.json ]; then
  T0=$(date +%s)
  (cd sionna_port && $PYS generate_dataset.py --scene-xml ../sionna_tutorial/RF-3DGS_Sionna_simulation_tutorial/NIST_lobby_v1.0/NIST_lobby_V1.1_sionna12.xml \
      --rx-loc-file ../sionna_tutorial/RF-3DGS_Sionna_simulation_tutorial/NIST_rx_loc.txt --out-dir ../$NEW --spectrum MVDR \
      --poses-from ../$OLD/sparse/0/images.txt --no-dashboard) > output/rrf/r50_generate.log 2>&1
  echo "generated in $(( $(date +%s) - T0 )) s: $(grep -a 'global dB range' output/rrf/r50_generate.log)" >> $LOG
fi
if [ ! -f $NEWG/generation_meta.json ]; then
  $PYW rrf_gsplat/renormalize.py $NEW $NEWG --norm global-pct --range -175.7498016357422 -101.52064493942484 >> $LOG 2>&1
  cp ${OLD}_gpct/train_index.txt ${OLD}_gpct/test_index.txt $NEWG/
fi
B="--mode db --source $NEWG --sh-backend gsplat --eval-group --faces-per-step 4 --lr-scale 2 --iterations 2500 --eval-every 2500 --save-renders -1 --sh-degree 3"
for s in 0 1 2; do
  name=r50_f64; [ $s -gt 0 ] && name=r50_f64_s$s
  if [ ! -f output/rrf/$name/results.json ]; then
    mkdir -p output/rrf/$name
    $PYW rrf_gsplat/train_rrf.py --out output/rrf/$name $B --seed $s > output/rrf/$name/train.log 2>&1
    echo "$name: $(tail -1 output/rrf/$name/train.log | cut -c1-110)" >> $LOG
  fi
done
[ -d output/rrf/t4f64_nn ] || $PYS sionna_port/t4_baselines.py --truth $NEWG --which nn --out-prefix output/rrf/t4f64_ >> $LOG 2>&1
$PYS sionna_port/t4_score.py --truth $NEWG --runs "output/rrf/r50_f64*" output/rrf/r45_tf32off output/rrf/r45_tf32off_s1 output/rrf/r45_tf32off_s2 \
    output/rrf/t4f64_nn --cov output/rrf/t4_rt_cov_all.npz --out output/rrf/t4_scores_f64labels.json >> $LOG 2>&1
echo "== all done $(date +%H:%M:%S)" >> $LOG
