#!/bin/bash
# Round 43: the shading-head ablation again, with the two fixes for the heads that saturated at their bound in
# round 41 (3 of 21 dead: S1, S2, S5 seed 1): the output layer in units of the bound (--head-bound-units) and the
# head off for the first 250 steps (--head-warmup 250). Smoke: win_round43s.sh (bound units alone, FAILED c: S1 and
# S2 still dead) and win_round43s2.sh (both). Round 41's arguments otherwise (MVDR 3dgs_MVDR_100_gpct, 2.5k steps
# x 4 faces, lr x 2, gsplat SH, frozen geometry, every held-out view saved).
#   ladder     S1..S5 on SH1 (S0 has no head: round 41's run, plus seeds 1, 2 here)
#   LOO        S5 minus geo / phys / latent / ring; width 8; S5 + peak loss
#   quality    B6 (SH3 + head), B5 (+ peak loss); B1 / B4 have no head: round 41's three seeds stand
#   fixes      the fixes are added pieces too: S5 with warm-up only, and with bound units only
#   seeds      S5, B6, B5 and both fix arms: 3 seeds each
# Then mvdr_peaks on every run, and the head share (round 42's method) on every head run.
set -u
REPO=/c/Users/Ke/Documents/GitHub/RF-3DGS
PYS=/c/Users/Ke/miniconda3/envs/rf-sionna-win/python.exe
WSL="wsl.exe -d Ubuntu-22.04 -u ke -- bash /mnt/c/Users/Ke/Documents/GitHub/RF-3DGS/rrf_gsplat/wsl_run.sh"
MV=RF-3DGS_dataset/regenerated/3dgs_MVDR_100_gpct
export PYTHONUTF8=1 MSYS_NO_PATHCONV=1
LOG=$REPO/output/rrf/win_round43.log
cd $REPO
stamp() { echo "== $1 $(date +%H:%M:%S) gpu $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader | tr -d '\r')" >> $LOG; }
peaks() { [ -f output/rrf/peaks_$1.json ] || $PYS rrf_gsplat/mvdr_peaks.py --run output/rrf/$1 --truth $MV >> $LOG 2>&1; }
B="--mode db --source $MV --sh-backend gsplat --eval-group --faces-per-step 4 --lr-scale 2 --iterations 2500 --eval-every 250 --save-renders -1"
FIX="--head-bound-units --head-warmup 250"
FULL="--head cnn --head-guides geo,phys --head-latent 4 --head-strip"
run() {   # name, args
  local name=$1; shift
  if [ ! -f output/rrf/$name/results.json ]; then
    stamp "B $name"; local T0=$(date +%s)
    $WSL $name $B "$@" >> $LOG 2>&1 < /dev/null
    echo "wall $name $(( $(date +%s) - T0 )) s" >> $LOG
  fi
  peaks $name
}
seeded() {   # name, args: seeds 0, 1, 2
  local name=$1; shift
  run $name "$@"; run ${name}_s1 "$@" --seed 1; run ${name}_s2 "$@" --seed 2
}
stamp "round 43 start"
seeded r43_S5_full             --sh-degree 1 $FULL $FIX
seeded r43_B6_sh3_head         --sh-degree 3 $FULL $FIX
seeded r43_B5_sh3_head_peak    --sh-degree 3 $FULL $FIX --peak-loss 1
run r43_S1_head                --sh-degree 1 --head cnn $FIX
run r43_S2_geo                 --sh-degree 1 --head cnn --head-guides geo $FIX
run r43_S3_geo_phys            --sh-degree 1 --head cnn --head-guides geo,phys $FIX
run r43_S4_geo_phys_lat        --sh-degree 1 --head cnn --head-guides geo,phys --head-latent 4 $FIX
run r43_S5_minus_geo           --sh-degree 1 --head cnn --head-guides phys --head-latent 4 --head-strip $FIX
run r43_S5_minus_phys          --sh-degree 1 --head cnn --head-guides geo --head-latent 4 --head-strip $FIX
run r43_S5_minus_lat           --sh-degree 1 --head cnn --head-guides geo,phys --head-strip $FIX
run r43_S5_minus_strip         --sh-degree 1 --head cnn --head-guides geo,phys --head-latent 4 $FIX
run r43_S5_w8                  --sh-degree 1 $FULL --head-width 8 $FIX
run r43_S5_peak                --sh-degree 1 $FULL $FIX --peak-loss 1
seeded r43_S5_wuonly           --sh-degree 1 $FULL --head-warmup 250
seeded r43_S5_buonly           --sh-degree 1 $FULL --head-bound-units
run r43_S0_sh1_s1              --sh-degree 1 --seed 1
run r43_S0_sh1_s2              --sh-degree 1 --seed 2

# the head's share where there is signal: every head run re-rendered from its own state with the head off
stamp "head share"
for d in output/rrf/r43_S*/ output/rrf/r43_B*/; do
  run=$(basename $d); [ -f $d/rrf_state.pt ] || continue
  $PYS -c "import json,sys; sys.exit(0 if json.load(open('$d/results.json'))['config'].get('head')=='cnn' else 1)" || continue
  deg=$($PYS -c "import json; print(json.load(open('$d/results.json'))['config']['sh_degree'])")
  base=r43base_${run#r43_}
  [ -f output/rrf/$base/results.json ] || $WSL $base --mode db --source $MV --sh-backend gsplat --eval-group --sh-degree $deg \
      --init-from /mnt/c/Users/Ke/Documents/GitHub/RF-3DGS/output/rrf/$run/rrf_state.pt --iterations 0 --save-renders -1 >> $LOG 2>&1 < /dev/null
  $PYS rrf_gsplat/head_share.py --run output/rrf/$run --base output/rrf/$base --truth $MV >> $LOG 2>&1
done
stamp "all done"
