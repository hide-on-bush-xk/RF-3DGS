#!/bin/bash
# Round 21: Track A of docs/sota_plan.md -- the RF-3DGS released benchmark on
# its own coordinates: released data, released 2560/640 split, the fork's
# metrics.py (PSNR / SSIM / VGG-LPIPS). Six spectra x rows:
#   A1 RF-3DGS retrained with the fork's train.py (visual checkpoint 30k -> 40k, i.e. 10k RF steps)
#   A2 ours rgb 10k       A3 ours db (jet-inverted target) 10k
#   A4 ours rgb + unfrozen geometry   A5 ours db + unfrozen geometry
# plus A2 at 40k on MVDR. Every gsplat run keeps all 640 renders and is scored
# by inria_metrics.py; the INRIA runs by render.py + metrics.py. Timings are
# wall-clock on an exclusive card (gate below), one run at a time.
set -u
REPO=/c/Users/Ke/Documents/GitHub/RF-3DGS
PYS=/c/Users/Ke/miniconda3/envs/rf-sionna-win/python.exe
PYT=/c/Users/Ke/miniconda3/envs/rf-3dgs/python.exe
CK=/mnt/c/Users/Ke/Documents/GitHub/RF-3DGS/RF-3DGS_dataset/blender_visual_trained/chkpnt30000.pth
CKW=RF-3DGS_dataset/blender_visual_trained/chkpnt30000.pth
WSL="wsl.exe -d Ubuntu-22.04 -u ke -- bash /mnt/c/Users/Ke/Documents/GitHub/RF-3DGS/rrf_gsplat/wsl_run.sh"
export PYTHONUTF8=1 MSYS_NO_PATHCONV=1
LOG=$REPO/output/rrf/win_round21.log
cd $REPO
stamp() { echo "== $1 $(date +%H:%M:%S)" >> $LOG; }
gpu_busy() {
  tasklist 2>/dev/null | grep -qi "Against the Storm" && return 0
  u=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr -d ' \r'); [ "${u:-100}" -ge 15 ]
}
run() { local name=$1; shift; [ -f output/rrf/$name/results.json ] && { echo "skip $name" >> $LOG; return; }
        stamp "$name"; T0=$(date +%s); $WSL "$name" --checkpoint $CK "$@" >> $LOG 2>&1 < /dev/null; echo "wall $(( $(date +%s) - T0 )) s" >> $LOG; stamp "done $name"; }
score() { local name=$1; local ds=$2; [ -f output/rrf/$name/inria_metrics.json ] && return
          $PYS rrf_gsplat/inria_metrics.py --run output/rrf/$name --source $ds < /dev/null 2>&1 | tail -1 >> $LOG; }
inria() { local S=$1; local ds=$2; local out=output/sota/inria_$S
          [ -f $out/results.json ] && { echo "skip inria_$S" >> $LOG; return; }
          stamp "inria_$S train"; T0=$(date +%s)
          $PYT train.py -s $ds -m $out --iterations 40000 --start_checkpoint $CKW --eval --test_iterations 40000 --save_iterations 40000 \
              < /dev/null > $out.train.log 2>&1; echo "wall $(( $(date +%s) - T0 )) s" >> $LOG
          grep -a "Evaluating test" $out.train.log | tail -1 >> $LOG
          stamp "inria_$S render+metrics"; $PYT render.py -m $out --skip_train < /dev/null > /dev/null 2>&1
          $PYT metrics.py -m $out < /dev/null 2>&1 | grep -a "PSNR\|SSIM\|LPIPS" | tr '\n' ' ' >> $LOG; echo >> $LOG; stamp "done inria_$S"; }
quiet=0; while [ $quiet -lt 24 ]; do if gpu_busy; then quiet=0; else quiet=$((quiet+1)); fi; sleep 5; done
mkdir -p output/sota
stamp "start (gpu $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader | tr -d '\r'))"
for S in MVDR CBF TCBF AoD Delay MPC; do
  DS=RF-3DGS_dataset/training-rf-spectrum/3dgs_${S}_100
  inria $S $DS
  run sota_${S}_rgb      --source $DS --mode rgb --save-renders -1;                       score sota_${S}_rgb $DS
  run sota_${S}_db       --source $DS --mode db  --save-renders -1;                       score sota_${S}_db $DS
  run sota_${S}_rgb_geom --source $DS --mode rgb --train-geometry --save-renders -1;      score sota_${S}_rgb_geom $DS
  run sota_${S}_db_geom  --source $DS --mode db  --train-geometry --save-renders -1;      score sota_${S}_db_geom $DS
done
DS=RF-3DGS_dataset/training-rf-spectrum/3dgs_MVDR_100
run sota_MVDR_rgb_40k --source $DS --mode rgb --iterations 40000 --save-renders -1;         score sota_MVDR_rgb_40k $DS
run sota_MVDR_db_geom_40k --source $DS --mode db --train-geometry --iterations 40000 --save-renders -1; score sota_MVDR_db_geom_40k $DS
stamp "all done"
