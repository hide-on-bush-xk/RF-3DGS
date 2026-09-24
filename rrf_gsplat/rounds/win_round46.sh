#!/bin/bash
# Round 46: 3DGS-LM with a trust region sized for this problem. Round 45's lm (seed 0) ran with 3DGS-LM's own
# radius settings (start 1e-3, min 1e-4, max 1e-2): the radius reached its cap after two iterations and every
# line search stopped at its maximum gamma = 1, i.e. the step was damped to about 1 / (1 + 1/radius) = 1 % of
# the Gauss-Newton step by the cap, not by the fit. 3DGS-LM moves every attribute (a nonlinear problem); here only
# the SH coefficients move and the render is linear in them, so Ceres' own defaults (start 1e4, max 1e16) are
# the natural setting. Same configuration as r45_lm otherwise (SH3 db, MVDR, Adam 1600 steps, 5 LM iterations,
# TF32 off), 3 seeds; before every run 60 s under 15 % GPU utilisation.
#   lmr   --lm-radius 1e4 --lm-radius-max 1e16
#   (--bwd-no-geom off: round 45 ran the stock backward; since round 45 train_rrf.py turns NO_GEOM on by
#   default for frozen geometry, which would give lmr a 6.5 % head start over r45_tf32off / r45_lm)
# Decision, written before the run (against r45_tf32off, as for r45_lm; the two LM rows are "as published" =
# r45_lm and "adapted trust region" = r46_lmr):
#   keep if pure training time <= 0.85 x tf32off's, PSNR >= tf32off - 0.05 dB, at-true-peak within tf32off's seed
#   range. Also reported: whether the radius still hits its cap and gamma its maximum (if so, the step is still
#   limited by the settings, not by the model).
set -u
REPO=/c/Users/Ke/Documents/GitHub/RF-3DGS
PYW=/c/Users/Ke/miniconda3/envs/rf-gsplat-win/python.exe
PYS=/c/Users/Ke/miniconda3/envs/rf-sionna-win/python.exe
MV=RF-3DGS_dataset/regenerated/3dgs_MVDR_100_gpct
export PYTHONUTF8=1
LOG=$REPO/output/rrf/win_round46.log
cd $REPO
util() { nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr -d ' \r'; }
quiet() { local q=0; while [ $q -lt 12 ]; do u=$(util); if [ "${u:-100}" -lt 15 ]; then q=$((q+1)); else q=0; fi; sleep 5; done; }
B="--mode db --source $MV --sh-backend gsplat --eval-group --faces-per-step 4 --lr-scale 2 --iterations 2500 --eval-every 250 --save-renders -1"
run() {   # name, args...
  local name=$1; shift
  if [ ! -f output/rrf/$name/results.json ]; then
    quiet; local u0=$(util); local T0=$(date +%s)
    echo "== $name $(date +%H:%M:%S) gpu $u0 %" >> $LOG
    mkdir -p output/rrf/$name
    $PYW rrf_gsplat/train_rrf.py --out output/rrf/$name $B "$@" > output/rrf/$name/train.log 2>&1
    echo "wall $name $(( $(date +%s) - T0 )) s, gpu after $(util) %, $(tail -1 output/rrf/$name/train.log | cut -c1-90)" >> $LOG
  fi
  [ -f output/rrf/peaks_$name.json ] || $PYS rrf_gsplat/mvdr_peaks.py --run output/rrf/$name --truth $MV >> $LOG 2>&1
}
echo "== round 46 start $(date +%H:%M:%S)" >> $LOG
for s in 0 1 2; do
  sfx=""; [ $s -gt 0 ] && sfx="_s$s"
  run r46_lmr$sfx --sh-degree 3 --seed $s --lm-after 1600 --lm-iters 5 --lm-radius 1e4 --lm-radius-max 1e16 --bwd-no-geom off
done
echo "== all done $(date +%H:%M:%S)" >> $LOG
