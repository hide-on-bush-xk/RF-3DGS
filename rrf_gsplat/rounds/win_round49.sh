#!/bin/bash
# Round 49: the radio radiance field against the nearest-neighbour lookup as the training positions thin out.
# Round 47 (T4): with all 640 training positions (neighbours tens of cm apart along the route), the nearest
# training position's label beats the field on held-out peaks (direction median 1.29 vs 5.56 deg, beam-gain loss
# 0.19 vs 2.62 dB). A field with the scene's geometry should degrade more slowly than a lookup when positions are
# sparse -- if it has any value for peaks, it shows there.
#   r49_k{160,40,10}   SH3 db, the round-45 configuration (2500 steps x 4 faces, TF32 off and NO_GEOM now default),
#                      --max-train-views 4K: K positions evenly along the training list (route), seed 0
#   t4_nn_k{160,40,10} the nearest of those same K positions' labels (t4_baselines.py --nn-positions K)
# All scored by t4_score.py on the same 640 held-out views (stored truth; beam-gain loss on the ray-traced channel).
# Reading, written before the runs: the headline is the smallest K at which the field's direction median or
# beam-gain loss is below NN's; if NN is better at every K down to 10 positions, the field adds nothing over a
# lookup for the peaks of this dataset. One seed (round 45's seed spread: PSNR +-0.06 dB, <= 1 deg +-1.7 points);
# a crossover within that spread is reported as none.
set -u
REPO=/c/Users/Ke/Documents/GitHub/RF-3DGS
PYW=/c/Users/Ke/miniconda3/envs/rf-gsplat-win/python.exe
PYS=/c/Users/Ke/miniconda3/envs/rf-sionna-win/python.exe
MV=RF-3DGS_dataset/regenerated/3dgs_MVDR_100_gpct
export PYTHONUTF8=1
LOG=$REPO/output/rrf/win_round49.log
cd $REPO
B="--mode db --source $MV --sh-backend gsplat --eval-group --faces-per-step 4 --lr-scale 2 --iterations 2500 --eval-every 2500 --save-renders -1 --sh-degree 3 --seed 0"
echo "== round 49 start $(date +%H:%M:%S)" >> $LOG
for k in 160 40 10; do
  name=r49_k$k
  if [ ! -f output/rrf/$name/results.json ]; then
    mkdir -p output/rrf/$name
    $PYW rrf_gsplat/train_rrf.py --out output/rrf/$name $B --max-train-views $((4 * k)) > output/rrf/$name/train.log 2>&1
    echo "$name: $(tail -1 output/rrf/$name/train.log | cut -c1-110)" >> $LOG
  fi
  [ -d output/rrf/t4_nn_k$k ] || $PYS sionna_port/t4_baselines.py --truth $MV --which nn --nn-positions $k >> $LOG 2>&1
done
$PYS sionna_port/t4_score.py --truth $MV --runs output/rrf/r45_base output/rrf/t4_nn "output/rrf/r49_k*" "output/rrf/t4_nn_k*" \
    --cov output/rrf/t4_rt_cov_all.npz --out output/rrf/t4_scores_density.json >> $LOG 2>&1
echo "== all done $(date +%H:%M:%S)" >> $LOG
