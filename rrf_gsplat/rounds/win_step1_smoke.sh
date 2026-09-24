#!/bin/bash
# Step 1 of the plan (Ke, 2026-09-24: "按顺序来"): can the ORACLE placement beat the look-ups on the validation set?
# This is its smoke: the whole path once, short, before any long run.
#   emitters   path_emitters.py on all 467 protocol training positions (never a validation position), 5 cm voxels,
#              90 % of each face sector's power, <= 8000 per position
#   training   emitters + --em-pcolor 8, power mode, the unclipped range (3dgs_APS_60_gp100), the full protocol
#              training set, 3000 steps (a smoke: far fewer visits per view than the full run will use), live on
#   scoring    mvdr_peaks.py on val / val_random / val_segment, t4_score.py with the validation tap covariances
# Criteria, written before the run: the builder's peaks check >= 85 % of distinct views (as before); every one of
# the 480 validation views scored, no MISSING in any output; the run visible in the viewer's live tab.
# Nothing here is judged on its numbers: 3000 steps over 1868 views is ~6 visits per view.
set -u
REPO=/c/Users/Ke/Documents/GitHub/RF-3DGS
PYW=/c/Users/Ke/miniconda3/envs/rf-gsplat-win/python.exe
PYS=/c/Users/Ke/miniconda3/envs/rf-sionna-win/python.exe
APU=RF-3DGS_dataset/regenerated/3dgs_APS_60_gp100
PROT=rrf_gsplat/protocol_v1
EM=output/rrf/m3/emitters_train467.npz
COV=output/rrf/t4_rt_cov_val_all.npz
export PYTHONUTF8=1
LOG=$REPO/output/rrf/m3/win_step1_smoke.log
cd $REPO
echo "== step 1 smoke start $(date +%H:%M:%S)" >> $LOG
if [ ! -f $EM ]; then
  $PYS sionna_port/path_emitters.py --truth RF-3DGS_dataset/regenerated/3dgs_APS_60 --protocol $PROT --capacity 467 \
      --cap 8000 --out $EM > output/rrf/m3/emitters_train467.log 2>&1
fi
grep -a -E '"emitters"|"seconds"|distinct_within|PASS|FAIL' output/rrf/m3/emitters_train467.log >> $LOG
name=s1smoke_em_pc8_3k
if [ ! -f output/rrf/m3/$name/results.json ]; then
  mkdir -p output/rrf/m3/$name
  $PYW rrf_gsplat/train_rrf.py --out output/rrf/m3/$name --mode power --source $APU --protocol $PROT --eval-set val \
      --sh-backend gsplat --eval-group --faces-per-step 4 --lr-scale 2 --sh-degree 3 --seed 0 --iterations 3000 \
      --eval-every 3000 --save-renders -1 --emitters $EM --em-pcolor 8 > output/rrf/m3/$name/train.log 2>&1
fi
echo "$name: $(grep -a 'iterations in' output/rrf/m3/$name/train.log | tail -1 | cut -c1-140)" >> $LOG
for es in val val_random val_segment; do
  $PYS rrf_gsplat/mvdr_peaks.py --run output/rrf/m3/$name --truth $APU --protocol $PROT --eval-set $es \
      --out output/rrf/m3/peaks_${name}_$es.json >> $LOG 2>&1
done
$PYS sionna_port/t4_score.py --truth $APU --runs output/rrf/m3/$name --protocol $PROT --eval-set val --cov $COV \
    --out output/rrf/m3/t4_scores_$name.json >> $LOG 2>&1
echo "== all done $(date +%H:%M:%S)" >> $LOG
