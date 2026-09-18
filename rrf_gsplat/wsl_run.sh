#!/bin/bash
# Run train_rrf.py in the WSL gsplat environment and log to output/rrf/<name>/train.log.
#   bash rrf_gsplat/wsl_run.sh <name> [train_rrf.py arguments...]
# From Windows:  MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu-22.04 -u ke -- bash /mnt/c/.../rrf_gsplat/wsl_run.sh <name> ...
set -u
REPO=/mnt/c/Users/Ke/Documents/GitHub/RF-3DGS
PY=/home/ke/miniconda3/envs/rf-gsplat/bin/python
NAME=$1; shift
OUT=$REPO/output/rrf/$NAME
mkdir -p "$OUT"
cd "$REPO"
PYTHONUNBUFFERED=1 "$PY" rrf_gsplat/train_rrf.py --out "$OUT" "$@" > "$OUT/train.log" 2>&1
tail -3 "$OUT/train.log"
