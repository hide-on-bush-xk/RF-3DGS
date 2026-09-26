#!/bin/bash
# Information ladder, step 2 (docs/cluster_log.md §5): one arm at one k, all repeats of that k, one A30 job.
#     sbatch -J rf-l2-<ARM>-k<K> rrf_gsplat/cluster/ladder_step2_job.sh <ARM> <K> [ITERATIONS]   (from the repo root)
# Each run: rrf_gsplat/cluster/train_rrf_ladder.py on the subset's names (fixed steps, no early stopping, no final
# validation pass, live comm off), then diag_ladder.py in-sample (the subset) and on the 307 held-out training positions.
# Inputs from rrf_gsplat/cluster/ladder_step2_prep.py (output/cluster/ladder/step2/prep/).
#SBATCH -A ai_wireless_hu_lab
#SBATCH -p a30_normal_q
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=16
#SBATCH --gres=gpu:1 --gres-flags=enforce-binding
#SBATCH -t 2:00:00
#SBATCH -o output/cluster/logs/%x-%j.out
set -u
ARM=${1:?arm}; K=${2:?k}
cd "${SLURM_SUBMIT_DIR:-.}"
[ -f rrf_gsplat/train_rrf.py ] || { echo "submit from the repository root"; exit 1; }
PY=$HOME/envs/rf-gsplat/bin/python
PREP=output/cluster/ladder/step2/prep
OUT=output/rrf/ladder_step2/$ARM
case $K in 1) REPS=6 ;; 2) REPS=4 ;; 5) REPS=4 ;; 20) REPS=2 ;; 160) REPS=1 ;; *) echo "k $K"; exit 1 ;; esac
IT=${3:-$([ "$K" = 160 ] && echo 24000 || echo 5000)}
BOX="--sh-extra $PREP/sel_box.npz --sh-extra-degree 6"
LOBE="--lobes 1 --lobe-kappa 300 --mirror-lobes $PREP/mirror_box.npz"
export OMP_NUM_THREADS=16
echo "== $ARM k=$K reps=$REPS iterations=$IT start $(date '+%F %T') on $(hostname), job ${SLURM_JOB_ID:-none}"
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used --format=csv,noheader
mkdir -p $OUT
for ((r = 0; r < REPS; r++)); do
  case $ARM in
    A0) X="" ;;
    A1) X="--peak-loss 1.0" ;;
    A2) X="$BOX" ;;
    A3) X="$BOX --peak-loss 1.0" ;;
    A4) X="$LOBE" ;;
    A5) X="$LOBE --peak-loss 1.0" ;;
    A6) X="--sh-extra $PREP/sel_dark.npz --sh-extra-degree 6" ;;
    A7) X="--emitters $PREP/emit_img_k${K}_r${r}.npz --emitter-scale 0.25" ;;
    A8) X="--emitters $PREP/emit_wall_k${K}_r${r}.npz --emitter-scale 0.25" ;;
    *) echo "arm $ARM"; exit 1 ;;
  esac
  d=$OUT/k${K}_r$r
  t0=$(date +%s.%N)
  if [ ! -f $d/results.json ]; then
    mkdir -p $d
    $PY rrf_gsplat/cluster/train_rrf_ladder.py --out $d --mode power \
      --source RF-3DGS_dataset/regenerated/3dgs_APS_60_gp100 --protocol rrf_gsplat/protocol_v1 --eval-set val \
      --sh-backend gsplat --eval-group --faces-per-step 4 --lr-scale 2 --sh-degree 3 --seed 0 \
      --train-names-file $PREP/names_k${K}_r$r.txt --iterations $IT --eval-every 1000000 --save-renders 0 \
      --live-comm-every 0 --no-eval $X > $d/train.log 2>&1 || { echo "train failed: $d"; tail -25 $d/train.log; exit 1; }
  fi
  t1=$(date +%s.%N)
  [ -f ${d}_insample.json ] || $PY rrf_gsplat/cluster/diag_ladder.py --run $d --out-tag insample > $d/insample.log 2>&1 \
    || { echo "diag in-sample failed: $d"; tail -25 $d/insample.log; exit 1; }
  t2=$(date +%s.%N)
  [ -f ${d}_heldout.json ] || $PY rrf_gsplat/cluster/diag_ladder.py --run $d --names-file $PREP/heldout.txt --out-tag heldout \
    > $d/heldout.log 2>&1 || { echo "diag held-out failed: $d"; tail -25 $d/heldout.log; exit 1; }
  t3=$(date +%s.%N)
  awk -v a=$t0 -v b=$t1 -v c=$t2 -v e=$t3 'BEGIN { printf "{\"train_s\": %.1f, \"insample_s\": %.1f, \"heldout_s\": %.1f}\n", b-a, c-b, e-c }' > $d/wall.json
  echo "$ARM k=$K r=$r done $(date +%T): $(grep -a 'iterations in' $d/train.log | tail -1 | cut -c1-140) | $(cat $d/wall.json)"
done
echo "== done $(date '+%F %T')"
