#!/bin/bash
# Step 1 of the plan (Ke, 2026-09-24: "按顺序来", "都跑都做"): can the ORACLE placement beat the look-ups on the
# validation set? Smoke: win_step1_smoke.sh (passed: builder peaks 92.2 % distinct, all 480 validation views scored).
# Preceded by the path cache (path_emitters.py --cache), which makes later emitter builds CPU only.
#
# Cache checks, written before the run:
#   C1  the cache-writing build (one re-solve of the 467 training positions) reproduces emitters_train467.npz: the same
#       number of emitters, centres within 1 um (the solver may return paths in another order)
#   C2  a build from the cache alone reproduces C1's output bit for bit, with every position read from the cache
# Step 1 runs: the unclipped range (3dgs_APS_60_gp100), the full protocol training set (467 positions, 1868 views),
# 117,000 steps (~250 visits per view, as the capacity runs), power mode, seed 0:
#   A      emitters from the training positions' own paths (never a validation position) + --em-pcolor 8
#   plain  no emitters (the control)
# Reading, written before the run (primary: the validation set -- distinct <= 1 deg and the beam-gain loss median on
# the true channel; val_random / val_segment reported beside). The look-ups on the same views (s2_aps_*): NN distinct
# <= 1 deg 42 %, beam-gain loss 1.36 dB; IDW-2 42 % / 1.17 dB; IDW-8 38 % / 1.09 dB:
#   A beats the best look-up on BOTH (distinct <= 1 deg by > 3 points, and a lower beam-gain loss) -> the placement
#       route can win on new positions -> step 2 (non-oracle placement); seeds 1-2 for the record
#   A beats it on NEITHER -> even true placement does not carry to new positions: stop before step 2 and look at
#       generalisation first
#   one of the two -> report both, decide together
# Watch it live: http://localhost:8772/#live (training curves, renders, and the machine's CPU / memory / GPU).
set -u
REPO=/c/Users/Ke/Documents/GitHub/RF-3DGS
PYW=/c/Users/Ke/miniconda3/envs/rf-gsplat-win/python.exe
PYS=/c/Users/Ke/miniconda3/envs/rf-sionna-win/python.exe
AP=RF-3DGS_dataset/regenerated/3dgs_APS_60
APU=RF-3DGS_dataset/regenerated/3dgs_APS_60_gp100
PROT=rrf_gsplat/protocol_v1
CACHE=output/rrf/m3/path_cache
EM=output/rrf/m3/emitters_train467.npz
COV=output/rrf/t4_rt_cov_val_all.npz
export PYTHONUTF8=1
LOG=$REPO/output/rrf/m3/win_step1.log
cd $REPO
echo "== step 1 start $(date +%H:%M:%S)" >> $LOG
# ---- the path cache, and its two checks ----
if [ ! -f output/rrf/m3/emitters_train467_c1.npz ]; then
  T0=$(date +%s)
  $PYS sionna_port/path_emitters.py --truth $AP --protocol $PROT --capacity 467 --cap 8000 --cache $CACHE \
      --out output/rrf/m3/emitters_train467_c1.npz > output/rrf/m3/emitters_c1.log 2>&1
  echo "cache-writing build: $(( $(date +%s) - T0 )) s" >> $LOG
fi
T0=$(date +%s)
$PYS sionna_port/path_emitters.py --truth $AP --protocol $PROT --capacity 467 --cap 8000 --cache $CACHE \
    --out output/rrf/m3/emitters_train467_c2.npz > output/rrf/m3/emitters_c2.log 2>&1
echo "cache-only build: $(( $(date +%s) - T0 )) s" >> $LOG
$PYS - >> $LOG 2>&1 <<EOF
import json
import numpy as np
def load(f):
    z = np.load(f); m = z["means"].astype(np.float64)
    return m[np.lexsort(m.T[::-1])]
a, c1, c2 = (load(f"output/rrf/m3/{n}.npz") for n in ("emitters_train467", "emitters_train467_c1", "emitters_train467_c2"))
r2 = json.load(open("output/rrf/m3/emitters_train467_c2.json"))
c1ok = len(a) == len(c1) and float(np.abs(a - c1).max()) <= 1e-6
c2ok = len(c1) == len(c2) and float(np.abs(c1 - c2).max()) == 0.0 and r2["positions_from_cache"] == 467
print(f"C1 {len(a)} vs {len(c1)} emitters, max |d| {np.abs(a - c1).max() if len(a) == len(c1) else 'n/a'} m -> {'PASS' if c1ok else 'FAIL'}")
print(f"C2 {len(c1)} vs {len(c2)} emitters, max |d| {np.abs(c1 - c2).max() if len(c1) == len(c2) else 'n/a'} m, "
      f"from cache {r2['positions_from_cache']} / 467 -> {'PASS' if c2ok else 'FAIL'}")
EOF
# ---- step 1 ----
B="--mode power --source $APU --protocol $PROT --eval-set val --sh-backend gsplat --eval-group --faces-per-step 4 \
--lr-scale 2 --sh-degree 3 --seed 0 --iterations 117000 --eval-every 117000 --save-renders -1"
run() {
  local name=$1; shift
  if [ ! -f output/rrf/m3/$name/results.json ]; then
    mkdir -p output/rrf/m3/$name
    $PYW rrf_gsplat/train_rrf.py --out output/rrf/m3/$name $B "$@" > output/rrf/m3/$name/train.log 2>&1
  fi
  echo "$name: $(grep -a 'iterations in' output/rrf/m3/$name/train.log | tail -1 | cut -c1-150)" >> $LOG
  for es in val val_random val_segment; do
    $PYS rrf_gsplat/mvdr_peaks.py --run output/rrf/m3/$name --truth $APU --protocol $PROT --eval-set $es \
        --out output/rrf/m3/peaks_${name}_$es.json >> $LOG 2>&1
  done
}
run step1_A_em_pc8 --emitters $EM --em-pcolor 8
run step1_plain
RUNS="output/rrf/m3/step1_A_em_pc8 output/rrf/m3/step1_plain output/rrf/s2_aps_nn output/rrf/s2_aps_idw2 output/rrf/s2_aps_idw4 \
output/rrf/s2_aps_idw8 output/rrf/s2_aps_const output/rrf/s2_aps_draw43"
$PYS sionna_port/t4_score.py --truth $APU --runs $RUNS --protocol $PROT --eval-set val --cov $COV \
    --out output/rrf/m3/t4_scores_step1_val.json >> $LOG 2>&1
echo "== all done $(date +%H:%M:%S)" >> $LOG
