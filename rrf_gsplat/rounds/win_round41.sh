#!/bin/bash
# Round 41: experiment A (the labels) and experiment B (the shading head), full runs after the round-40 smoke.
#   A1  label upsampler ladders: bilinear -> cnn -> + coords -> + peak loss (MVDR live pair: 3 seeds for the
#       learned arms; MULTI pair: one seed), scored by mvdr_peaks / eval_baselines against the true 300x200 labels
#   B   the shading-head ablation on 3dgs_MVDR_100_gpct (2.5k steps x 4 faces, lr x2, gsplat SH):
#       speed branch (SH1): ladder S0..S5, leave-one-out from S5, width 8, peak loss on S0 and S5
#       quality branch (SH3): 2 x 2 of head x peak loss; seeds 1, 2 for B1, S5, B4, B5
#   T   timing, only on a quiet card: profile_resolution at 300x200 and 600x400 for SH3, SH1, the full head and
#       the width-8 head; A0: generation time split at 300x200 and 150x100 (MVDR live, MULTI), twice each
set -u
REPO=/c/Users/Ke/Documents/GitHub/RF-3DGS
PYS=/c/Users/Ke/miniconda3/envs/rf-sionna-win/python.exe
PYW=/home/ke/miniconda3/envs/rf-gsplat/bin/python
WSL="wsl.exe -d Ubuntu-22.04 -u ke -- bash /mnt/c/Users/Ke/Documents/GitHub/RF-3DGS/rrf_gsplat/wsl_run.sh"
WSLC="wsl.exe -d Ubuntu-22.04 -u ke -- bash -c"
REG=RF-3DGS_dataset/regenerated
MV=$REG/3dgs_MVDR_100_gpct
export PYTHONUTF8=1 MSYS_NO_PATHCONV=1
LOG=$REPO/output/rrf/win_round41.log
cd $REPO
stamp() { echo "== $1 $(date +%H:%M:%S) gpu $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader | tr -d '\r')" >> $LOG; }
quiet() {   # wait for two minutes of < 15 % utilisation before anything that is timed
  local q=0; while [ $q -lt 24 ]; do u=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr -d ' \r'); \
    if [ "${u:-100}" -lt 15 ]; then q=$((q+1)); else q=0; fi; sleep 5; done; }
peaks() { [ -f output/rrf/peaks_$1.json ] || $PYS rrf_gsplat/mvdr_peaks.py --run output/rrf/$1 --truth $2 >> $LOG 2>&1; }

stamp "round 41 start"
# ---------------- A1: the label upsampler ----------------------------------------------------------------------------
sr() {   # name, lo, hi, mode, args
  local name=$1 lo=$2 hi=$3 mode=$4; shift 4
  [ -f output/rrf/$name/results.json ] && return
  stamp "A1 $name"
  $WSLC "cd /mnt/c/Users/Ke/Documents/GitHub/RF-3DGS && $PYW rrf_gsplat/label_sr.py --lo $lo --hi $hi --mode $mode $* --out output/rrf/$name" >> $LOG 2>&1 < /dev/null
}
LO=$REG/r27_live_150_gpct; HI=$REG/r27_live_300_gpct
sr r41_A1_mvdr_bilinear $LO $HI db --arch bilinear; peaks r41_A1_mvdr_bilinear $HI
for s in 0 1 2; do
  sr r41_A1_mvdr_cnn_s$s $LO $HI db --arch cnn --seed $s;                          peaks r41_A1_mvdr_cnn_s$s $HI
  sr r41_A1_mvdr_cnn_coords_s$s $LO $HI db --arch cnn --coords --seed $s;          peaks r41_A1_mvdr_cnn_coords_s$s $HI
  sr r41_A1_mvdr_cnn_coords_peak_s$s $LO $HI db --arch cnn --coords --peak-loss 1 --seed $s; peaks r41_A1_mvdr_cnn_coords_peak_s$s $HI
done
MLO=$REG/r26_MULTI_150_1M; MHI=$REG/r26_MULTI_300_1M
for spec in "bilinear:--arch bilinear" "cnn:--arch cnn" "cnn_coords:--arch cnn --coords" "cnn_coords_peak:--arch cnn --coords --peak-loss 1"; do
  tag=${spec%%:*}; args=${spec#*:}; name=r41_A1_multi_$tag
  sr $name $MLO $MHI multi $args
  [ -f output/rrf/baselines_$name.json ] || $PYS rrf_gsplat/eval_baselines.py --multi output/rrf/$name --truth $MHI \
      --out output/rrf/baselines_$name.json < /dev/null 2>&1 | grep -v jitc | tail -1 >> $LOG
done

# ---------------- B: the shading head ---------------------------------------------------------------------------------
B="--mode db --source $MV --sh-backend gsplat --eval-group --faces-per-step 4 --lr-scale 2 --iterations 2500 --eval-every 250 --save-renders -1"
run() {   # name, args
  local name=$1; shift
  if [ ! -f output/rrf/$name/results.json ]; then
    stamp "B $name"; local T0=$(date +%s)
    $WSL $name $B "$@" >> $LOG 2>&1 < /dev/null
    echo "wall $name $(( $(date +%s) - T0 )) s" >> $LOG
  fi
  peaks $name $MV
}
run r41_B_S0_sh1                --sh-degree 1
run r41_B_S1_head               --sh-degree 1 --head cnn
run r41_B_S2_geo                --sh-degree 1 --head cnn --head-guides geo
run r41_B_S3_geo_phys           --sh-degree 1 --head cnn --head-guides geo,phys
run r41_B_S4_geo_phys_lat       --sh-degree 1 --head cnn --head-guides geo,phys --head-latent 4
run r41_B_S5_full               --sh-degree 1 --head cnn --head-guides geo,phys --head-latent 4 --head-strip
run r41_B_S5_minus_geo          --sh-degree 1 --head cnn --head-guides phys --head-latent 4 --head-strip
run r41_B_S5_minus_phys         --sh-degree 1 --head cnn --head-guides geo --head-latent 4 --head-strip
run r41_B_S5_minus_lat          --sh-degree 1 --head cnn --head-guides geo,phys --head-strip
run r41_B_S5_minus_strip        --sh-degree 1 --head cnn --head-guides geo,phys --head-latent 4
run r41_B_S5_w8                 --sh-degree 1 --head cnn --head-guides geo,phys --head-latent 4 --head-strip --head-width 8
run r41_B_S0_peak               --sh-degree 1 --peak-loss 1
run r41_B_S5_peak               --sh-degree 1 --head cnn --head-guides geo,phys --head-latent 4 --head-strip --peak-loss 1
run r41_B_B1_sh3                --sh-degree 3
run r41_B_B4_sh3_peak           --sh-degree 3 --peak-loss 1
run r41_B_B6_sh3_head           --sh-degree 3 --head cnn --head-guides geo,phys --head-latent 4 --head-strip
run r41_B_B5_sh3_head_peak      --sh-degree 3 --head cnn --head-guides geo,phys --head-latent 4 --head-strip --peak-loss 1
for s in 1 2; do
  run r41_B_B1_sh3_s$s           --sh-degree 3 --seed $s
  run r41_B_S5_full_s$s          --sh-degree 1 --head cnn --head-guides geo,phys --head-latent 4 --head-strip --seed $s
  run r41_B_B4_sh3_peak_s$s      --sh-degree 3 --peak-loss 1 --seed $s
  run r41_B_B5_sh3_head_peak_s$s --sh-degree 3 --head cnn --head-guides geo,phys --head-latent 4 --head-strip --peak-loss 1 --seed $s
done

# ---------------- T: timing, on a quiet card only -------------------------------------------------------------------
stamp "timing: waiting for a quiet card"; quiet; stamp "timing start"
P="--mode db --source $MV --faces 4 --sh-backend gsplat --scales 1 2"
for spec in "sh3:--sh-degree 3" "sh1:--sh-degree 1" "full:--sh-degree 1 --head cnn --head-guides geo,phys --head-latent 4 --head-strip" \
            "w8:--sh-degree 1 --head cnn --head-guides geo,phys --head-latent 4 --head-strip --head-width 8"; do
  args=${spec#*:}
  $WSLC "cd /mnt/c/Users/Ke/Documents/GitHub/RF-3DGS && $PYW rrf_gsplat/profile_resolution.py $P $args" 2>&1 | grep -a "x.*step\|GPU before" >> $LOG
done
# A0: where a generated view's time goes, at both label sizes, twice each, into scratch directories
SCENE=../sionna_tutorial/RF-3DGS_Sionna_simulation_tutorial/NIST_lobby_v1.0/NIST_lobby_V1.1_sionna12.xml
RXLOC=../sionna_tutorial/RF-3DGS_Sionna_simulation_tutorial/NIST_rx_loc.txt
for rep in 1 2; do
  for spec in "mvdr300:MVDR:160:300:200:--tx 6.232953105196451 -7.689077437535534 2.0" "mvdr150:MVDR:160:150:100:--tx 6.232953105196451 -7.689077437535534 2.0" \
              "multi300:MULTI:800:300:200:--frequency 2.4e9 --materials tutorial --splat-sigma 3" "multi150:MULTI:800:150:100:--frequency 2.4e9 --materials tutorial --splat-sigma 3"; do
    IFS=: read tag kind npos w h extra <<< "$spec"
    quiet; stamp "A0 $tag rep $rep"
    (cd sionna_port && $PYS generate_dataset.py --scene-xml $SCENE --rx-loc-file $RXLOC --out-dir ../$REG/r41_A0_$tag --spectrum $kind \
        --num-positions $npos --width $w --height $h --no-dashboard $extra < /dev/null 2>&1 | grep -a "time split\|views/s" | sed "s/^/A0 $tag rep $rep: /" >> $LOG)
    rm -rf $REG/r41_A0_$tag
  done
done
stamp "all done"
