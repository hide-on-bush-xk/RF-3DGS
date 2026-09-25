#!/bin/bash
# Submit the direction-loss capacity benchmark (dir_loss_capacity.sh): 5 arms x seeds 0 1 2, one single-A30 job each.
# Only after the cluster smoke (dir_loss_smoke.sh) has passed and its results are posted (CLAUDE.md: smoke first).
# Runs whose trainfit JSON already exists are skipped. Run from the repository root.
set -u
[ -f rrf_gsplat/train_rrf.py ] || { echo "run from the repository root"; exit 1; }
mkdir -p output/cluster/logs
for SEED in 0 1 2; do
  for ARM in base eg eg3 ce ce3; do
    [ -f output/rrf/dirloss_cap_falcon/${ARM}_s${SEED}_trainfit.json ] && { echo "$ARM s$SEED: done, skipped"; continue; }
    sbatch -J rf-dircap-${ARM}-s$SEED rrf_gsplat/cluster/dir_loss_capacity.sh $ARM $SEED
  done
done
