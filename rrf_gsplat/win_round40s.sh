#!/bin/bash
# Round 40 smoke: every experiment-B arm for 300 steps on 80 farthest-point training positions (structure kept,
# size cut), full held-out set, then mvdr_peaks. Criteria in docs/stage2_notes.md (round 40).
set -u
REPO=/c/Users/Ke/Documents/GitHub/RF-3DGS
PYS=/c/Users/Ke/miniconda3/envs/rf-sionna-win/python.exe
WSL="wsl.exe -d Ubuntu-22.04 -u ke -- bash /mnt/c/Users/Ke/Documents/GitHub/RF-3DGS/rrf_gsplat/wsl_run.sh"
MV=RF-3DGS_dataset/regenerated/3dgs_MVDR_100_gpct
MU=RF-3DGS_dataset/regenerated/3dgs_MULTI_24ghz_tut_cs
export PYTHONUTF8=1 MSYS_NO_PATHCONV=1
LOG=$REPO/output/rrf/win_round40s.log
cd $REPO
stamp() { echo "== $1 $(date +%H:%M:%S) gpu $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader | tr -d '\r')" >> $LOG; }
S="--mode db --source $MV --sh-backend gsplat --eval-group --faces-per-step 4 --lr-scale 2 --iterations 300 --eval-every 100 --save-renders -1 --max-train-views 320 --subset-mode fps"
HEAD="--head cnn --head-guides geo,phys --head-latent 4 --head-strip"
stamp "round 40 smoke start"
for spec in "B1:--sh-degree 3" "B2:--sh-degree 1" "B3:--sh-degree 1 $HEAD" "B3w8:--sh-degree 1 $HEAD --head-width 8" \
            "B4:--sh-degree 3 --peak-loss 1" "B5:--sh-degree 3 $HEAD --peak-loss 1"; do
  tag=${spec%%:*}; args=${spec#*:}; name=r40s_$tag
  stamp "start $name"
  $WSL $name $S $args >> $LOG 2>&1 < /dev/null
  grep -a "ring order\|shading head\|optimised per-Gaussian" output/rrf/$name/train.log >> $LOG
  $PYS -c "import json;r=json.load(open('output/rrf/$name/results.json'));f=r['final'];print('$name', {k:(round(v,4) if isinstance(v,float) else v) for k,v in f.items()}, 'train', round(r['train_seconds_excl_running_eval'],1))" >> $LOG 2>&1
  $PYS rrf_gsplat/mvdr_peaks.py --run output/rrf/$name --truth $MV >> $LOG 2>&1
done
stamp "multi with head"
$WSL r40s_Mhead --mode multi --source $MU --delay-depth --delay-depth-mode ED --delay-range euclid --sh-backend gsplat \
    --eval-group --faces-per-step 4 --lr-scale 2 --iterations 300 --eval-every 100 --save-renders -1 \
    --max-train-views 320 --subset-mode fps --sh-degree 1 $HEAD >> $LOG 2>&1 < /dev/null
$PYS -c "import json;r=json.load(open('output/rrf/r40s_Mhead/results.json'));f=r['final'];print('r40s_Mhead', {k:(round(v,4) if isinstance(v,float) else v) for k,v in f.items()})" >> $LOG 2>&1
stamp "all done"
