#!/bin/bash
# Round 45: keep or drop each of tonight's features, on the SH3 db configuration of rounds 41/43 (MVDR
# 3dgs_MVDR_100_gpct, 2.5k steps x 4 faces, lr x 2, gsplat SH, frozen geometry, every held-out view saved),
# 3 seeds each, in the Windows-native build (gsplat_win, env rf-gsplat-win). Before every run: 60 s under 15 %
# GPU utilisation; the utilisation before and after is logged (a busy card voids that run's time).
#   base      stock backward, cuDNN TF32 on (as every run so far)
#   nogeom    GSPLAT_BWD_NO_GEOM=1                  skip the conic / 2D-mean gradients (frozen geometry)
#   pg        GSPLAT_BWD_PERGAUSS=1                 the per-Gaussian backward (Taming 3DGS)
#   pgng      both
#   tf32off   --cudnn-tf32 off                      the exact SSIM (the TF32 bug)
#   lm        --lm-after 1600 --lm-iters 5          3DGS-LM after Adam (TF32 off, forced); against tf32off
#   S5 head   base / nogeom / pg on the fixed shading head (bound units + warm-up; 9 channels), one seed
# Decisions, written before the run (quality = PSNR(jet) and mvdr_peaks' at-true-peak, 3-seed means):
#   nogeom   keep if quality within 0.05 dB of base and pure training time >= 5 % shorter
#   pg       keep only if pgng is >= 5 % faster than nogeom at equal quality (G2 says it will not be)
#   tf32off  make it the default unless PSNR or at-true-peak is worse than base by more than base's seed range
#   lm       keep if its pure training time is <= 0.85 x tf32off's at PSNR >= tf32off - 0.05 dB and at-true-peak
#            within tf32off's seed range
set -u
REPO=/c/Users/Ke/Documents/GitHub/RF-3DGS
PYW=/c/Users/Ke/miniconda3/envs/rf-gsplat-win/python.exe
PYS=/c/Users/Ke/miniconda3/envs/rf-sionna-win/python.exe
MV=RF-3DGS_dataset/regenerated/3dgs_MVDR_100_gpct
export PYTHONUTF8=1
LOG=$REPO/output/rrf/win_round45.log
cd $REPO
util() { nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr -d ' \r'; }
quiet() { local q=0; while [ $q -lt 12 ]; do u=$(util); if [ "${u:-100}" -lt 15 ]; then q=$((q+1)); else q=0; fi; sleep 5; done; }
B="--mode db --source $MV --sh-backend gsplat --eval-group --faces-per-step 4 --lr-scale 2 --iterations 2500 --eval-every 250 --save-renders -1"
run() {   # name, env assignments ("-" for none), args...
  local name=$1 envs=$2; shift 2
  if [ ! -f output/rrf/$name/results.json ]; then
    quiet; local u0=$(util); local T0=$(date +%s)
    echo "== $name $(date +%H:%M:%S) gpu $u0 %" >> $LOG
    mkdir -p output/rrf/$name
    if [ "$envs" = "-" ]; then $PYW rrf_gsplat/train_rrf.py --out output/rrf/$name $B "$@" > output/rrf/$name/train.log 2>&1
    else env $envs $PYW rrf_gsplat/train_rrf.py --out output/rrf/$name $B "$@" > output/rrf/$name/train.log 2>&1; fi
    echo "wall $name $(( $(date +%s) - T0 )) s, gpu after $(util) %, $(tail -1 output/rrf/$name/train.log | cut -c1-90)" >> $LOG
  fi
  [ -f output/rrf/peaks_$name.json ] || $PYS rrf_gsplat/mvdr_peaks.py --run output/rrf/$name --truth $MV >> $LOG 2>&1
}
echo "== round 45 start $(date +%H:%M:%S)" >> $LOG
for s in 0 1 2; do
  sfx=""; [ $s -gt 0 ] && sfx="_s$s"
  run r45_base$sfx    -                                       --sh-degree 3 --seed $s
  run r45_nogeom$sfx  GSPLAT_BWD_NO_GEOM=1                    --sh-degree 3 --seed $s
  run r45_pg$sfx      GSPLAT_BWD_PERGAUSS=1                   --sh-degree 3 --seed $s
  run r45_pgng$sfx    "GSPLAT_BWD_PERGAUSS=1 GSPLAT_BWD_NO_GEOM=1" --sh-degree 3 --seed $s
  run r45_tf32off$sfx -                                       --sh-degree 3 --seed $s --cudnn-tf32 off
  run r45_lm$sfx      -                                       --sh-degree 3 --seed $s --lm-after 1600 --lm-iters 5
done
H="--sh-degree 1 --head cnn --head-guides geo,phys --head-latent 4 --head-strip --head-bound-units --head-warmup 250"
run r45_S5_base   -                     $H
run r45_S5_nogeom GSPLAT_BWD_NO_GEOM=1  $H
run r45_S5_pg     GSPLAT_BWD_PERGAUSS=1 $H
echo "== all done $(date +%H:%M:%S)" >> $LOG
