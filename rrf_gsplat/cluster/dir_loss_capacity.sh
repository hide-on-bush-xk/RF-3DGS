#!/bin/bash
# Capacity benchmark of the main-peak direction loss on Falcon (docs/cluster_handover.md §6): one run per job.
#     sbatch rrf_gsplat/cluster/dir_loss_capacity.sh <arm> <seed>        (from the repository root)
#     bash rrf_gsplat/cluster/submit_dir_loss_capacity.sh                 (all 5 arms x seeds 0 1 2)
# Configuration as the rounds 51-56 re-run (rounds/win_rerun_r51_56.sh): 160 training positions (--max-train-views
# 640), power, SH3, --faces-per-step 4 --lr-scale 2 --sh-backend gsplat --eval-group, at most 250 visits per view,
# early stopping on the fixed training subset (10 evaluations without distinct <= 1 deg rising 2 points; at least
# 5,000 steps), --save-renders 0; then diag_train_fit.py --train-subset 640 (distinct <= 1 deg the primary).
# Arms: see ARGS below. The reading, written before the run, is in dir_loss_capacity_summary.py.
#SBATCH -J rf-dircap
#SBATCH -A ai_wireless_hu_lab
#SBATCH -p a30_normal_q
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=16
#SBATCH --gres=gpu:1 --gres-flags=enforce-binding
#SBATCH -t 3:00:00
#SBATCH -o output/cluster/logs/%x-%j.out
set -u
ARM=${1:?arm}; SEED=${2:?seed}
case $ARM in
  base) ARGS="--dir-loss expgain --dir-weight 0" ;;
  eg)   ARGS="--dir-loss expgain --dir-weight 0.1 --dir-temp-start-db 20 --dir-temp-db 1 --dir-anneal-steps 1500 --dir-start 200" ;;
  eg3)  ARGS="--dir-loss expgain --dir-weight 0.3 --dir-temp-start-db 20 --dir-temp-db 1 --dir-anneal-steps 1500 --dir-start 200" ;;
  ce)   ARGS="--dir-loss ce --dir-weight 0.001 --dir-temp-db 1 --dir-start 200" ;;
  ce3)  ARGS="--dir-loss ce --dir-weight 0.003 --dir-temp-db 1 --dir-start 200" ;;
  *) echo "unknown arm $ARM"; exit 1 ;;
esac
cd "${SLURM_SUBMIT_DIR:-.}"
[ -f rrf_gsplat/train_rrf.py ] || { echo "submit from the repository root"; exit 1; }
PY=$HOME/envs/rf-gsplat/bin/python
APU=RF-3DGS_dataset/regenerated/3dgs_APS_60_gp100
d=output/rrf/dirloss_cap_falcon/${ARM}_s$SEED
export OMP_NUM_THREADS=16
echo "== $ARM seed $SEED start $(date '+%F %T') on $(hostname), job ${SLURM_JOB_ID:-none}"
nvidia-smi --query-gpu=index,name,uuid,utilization.gpu,memory.used --format=csv
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv
B="--mode power --source $APU --protocol rrf_gsplat/protocol_v1 --eval-set val --sh-backend gsplat --eval-group \
--faces-per-step 4 --lr-scale 2 --sh-degree 3 --seed $SEED --max-train-views 640 --save-renders 0 --visits-per-view 250 \
--early-stop-on train --early-stop-patience 10 --early-stop-min-steps 5000 --eval-every 1000000"
mkdir -p $d/live
echo "主峰方向 loss 容量基准(Falcon):$ARM,种子 $SEED。160 个训练位置,最多每张视图 250 次,按训练子集早停。" > $d/live/desc.txt
t0=$(date +%s.%N)
if [ ! -f $d/results.json ]; then
  $PY rrf_gsplat/train_rrf.py --out $d $B $ARGS > $d/train.log 2>&1 || { echo "train_rrf.py failed"; tail -20 $d/train.log; exit 1; }
fi
t1=$(date +%s.%N)
if [ ! -f ${d}_trainfit.json ]; then
  $PY rrf_gsplat/diag_train_fit.py --run $d --train-subset 640 > $d/trainfit.log 2>&1 || { echo "diag_train_fit.py failed"; tail -20 $d/trainfit.log; exit 1; }
fi
t2=$(date +%s.%N)
awk -v a=$t0 -v b=$t1 -v c=$t2 'BEGIN { printf "{\"train_rrf_wall_s\": %.2f, \"trainfit_wall_s\": %.2f}\n", b - a, c - b }' > $d/wall.json
grep -a 'iterations in' $d/train.log | tail -1 | cut -c1-200
grep -a -E '^train' $d/trainfit.log
echo "== done $(date '+%F %T')"
