#!/bin/bash
# Round 8 (WSL): needle-only and disc-only unfreezing on Tx-A (db).
REPO=/mnt/c/Users/Ke/Documents/GitHub/RF-3DGS; REG=RF-3DGS_dataset/regenerated
cd $REPO
run() { local name=$1; shift; [ -f output/rrf/$name/results.json ] && { echo "skip $name"; return; }
        echo "== $name $(date +%H:%M:%S)"; bash rrf_gsplat/wsl_run.sh "$name" "$@"; }
run a_mvdr_db_geom_needles --source $REG/3dgs_MVDR_100_gpct --mode db --train-geometry --geometry-subset needles
run a_mvdr_db_geom_discs   --source $REG/3dgs_MVDR_100_gpct --mode db --train-geometry --geometry-subset discs
echo "round 8 done $(date +%H:%M:%S)"
