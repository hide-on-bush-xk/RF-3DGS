#!/bin/bash
# Step 2 new-code smoke (docs/cluster_log.md §5): U2 (equal render at initialisation, mirror lobe = 1 on its ray),
# then A4 and A2 at k = 1 for 300 steps end to end (train + in-sample + held-out diag), then U4 (only selected
# Gaussians learn, axes frozen, clamps hold). Submit from the repository root.
#SBATCH -J rf-l2-smoke
#SBATCH -A ai_wireless_hu_lab
#SBATCH -p a30_normal_q
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=16
#SBATCH --gres=gpu:1 --gres-flags=enforce-binding
#SBATCH -t 0:45:00
#SBATCH -o output/cluster/logs/%x-%j.out
set -u
cd "${SLURM_SUBMIT_DIR:-.}"
PY=$HOME/envs/rf-gsplat/bin/python
echo "== smoke start $(date '+%F %T') on $(hostname)"
$PY rrf_gsplat/cluster/ladder_step2_checks.py u2 || { echo "U2 FAIL"; exit 1; }
for A in A4 A2; do
  OUTROOT=output/rrf/ladder_step2_smoke
  sed "s#OUT=output/rrf/ladder_step2/\$ARM#OUT=$OUTROOT/\$ARM#; s#case \$K in 1) REPS=6#case \$K in 1) REPS=1#" \
    rrf_gsplat/cluster/ladder_step2_job.sh > /tmp/ladder_smoke_$$.sh
  bash /tmp/ladder_smoke_$$.sh $A 1 300 || { echo "$A pipeline FAIL"; exit 1; }
done
rm -f /tmp/ladder_smoke_$$.sh
$PY rrf_gsplat/cluster/ladder_step2_checks.py u4 output/rrf/ladder_step2_smoke/A4/k1_r0 output/rrf/ladder_step2_smoke/A2/k1_r0 \
  || { echo "U4 FAIL"; exit 1; }
echo "== smoke PASS $(date '+%F %T')"
