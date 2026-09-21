#!/bin/bash
# Round 22 (after round 21): Track B and Track D of docs/sota_plan.md, plus clean re-timings.
#   B  NeRF2 (their model + renderer, pinhole rays) on the released scalar spectra MVDR, CBF, TCBF, MPC, 30k iterations,
#      all 640 held-out views rendered and scored by the fork's metrics.py.
#   D  a MULTI dataset at the released poses (--poses-from, 2.4 GHz, tutorial materials = the E4 setting) and our
#      multi-channel field on it, so the decoded-angle metric is on the released poses too.
#   T  re-timing of inria_MVDR and sota_MVDR_rgb on a quiet card (their round-21 wall-clocks overlapped other jobs).
set -u
REPO=/c/Users/Ke/Documents/GitHub/RF-3DGS
PYS=/c/Users/Ke/miniconda3/envs/rf-sionna-win/python.exe
PYT=/c/Users/Ke/miniconda3/envs/rf-3dgs/python.exe
CK=/mnt/c/Users/Ke/Documents/GitHub/RF-3DGS/RF-3DGS_dataset/blender_visual_trained/chkpnt30000.pth
CKW=RF-3DGS_dataset/blender_visual_trained/chkpnt30000.pth
WSLPY="wsl.exe -d Ubuntu-22.04 -u ke -- bash -c"
WSL="wsl.exe -d Ubuntu-22.04 -u ke -- bash /mnt/c/Users/Ke/Documents/GitHub/RF-3DGS/rrf_gsplat/wsl_run.sh"
REG=RF-3DGS_dataset/regenerated
export PYTHONUTF8=1 MSYS_NO_PATHCONV=1
LOG=$REPO/output/rrf/win_round22.log
cd $REPO
stamp() { echo "== $1 $(date +%H:%M:%S)" >> $LOG; }
gpu_busy() {
  tasklist 2>/dev/null | grep -qi "Against the Storm" && return 0
  u=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr -d ' \r'); [ "${u:-100}" -ge 15 ]
}
until grep -aq "== all done" output/rrf/win_round21b.log 2>/dev/null; do sleep 30; done
quiet=0; while [ $quiet -lt 24 ]; do if gpu_busy; then quiet=0; else quiet=$((quiet+1)); fi; sleep 5; done
stamp "start (gpu $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader | tr -d '\r'))"

# B. NeRF2 on the released scalar spectra
for S in MVDR CBF TCBF MPC; do
  DS=RF-3DGS_dataset/training-rf-spectrum/3dgs_${S}_100
  if [ ! -f output/rrf/nerf2_$S/results.json ]; then
    stamp "nerf2_$S"; T0=$(date +%s)
    $WSLPY "cd /mnt/c/Users/Ke/Documents/GitHub/RF-3DGS && /home/ke/miniconda3/envs/rf-gsplat/bin/python rrf_gsplat/nerf2_pinhole.py --source $DS --out output/rrf/nerf2_$S --iterations 30000 > output/rrf/nerf2_$S.log 2>&1" < /dev/null
    echo "wall $(( $(date +%s) - T0 )) s" >> $LOG; tail -1 output/rrf/nerf2_$S.log >> $LOG; stamp "done nerf2_$S"
  fi
  [ -f output/rrf/nerf2_$S/inria_metrics.json ] || $PYS rrf_gsplat/inria_metrics.py --run output/rrf/nerf2_$S --source $DS < /dev/null 2>&1 | tail -1 >> $LOG
done

# D. MULTI at the released poses (E4 setting), our field, baselines
DS=$REG/3dgs_MULTI_relposes_24ghz_tut
if [ ! -f $DS/generation_meta.json ]; then
  stamp "dataset MULTI at released poses"; cd sionna_port
  $PYS generate_dataset.py --scene-xml ../sionna_tutorial/RF-3DGS_Sionna_simulation_tutorial/NIST_lobby_v1.0/NIST_lobby_V1.1_sionna12.xml \
      --poses-from ../RF-3DGS_dataset/training-rf-spectrum/3dgs_MVDR_100/sparse/0/images.txt --out-dir ../$DS \
      --spectrum MULTI --frequency 2.4e9 --materials tutorial --splat-sigma 3 --num-positions 800 --no-dashboard < /dev/null 2>&1 | grep -a "views/s\|channel ranges\|Error\|Traceback" >> $LOG
  cd $REPO
  cp RF-3DGS_dataset/training-rf-spectrum/3dgs_MVDR_100/train_index.txt RF-3DGS_dataset/training-rf-spectrum/3dgs_MVDR_100/test_index.txt $DS/ 2>/dev/null
fi
if [ ! -f output/rrf/m_multi_relposes_depth/results.json ]; then
  stamp "m_multi_relposes_depth"; T0=$(date +%s)
  $WSL m_multi_relposes_depth --checkpoint $CK --source $DS --mode multi --delay-depth --delay-depth-mode ED --delay-range euclid --save-renders -1 >> $LOG 2>&1 < /dev/null
  echo "wall $(( $(date +%s) - T0 )) s" >> $LOG; stamp "done m_multi_relposes_depth"
fi
[ -f output/rrf/baselines_m_multi_relposes_depth.json ] || $PYS rrf_gsplat/eval_baselines.py --multi output/rrf/m_multi_relposes_depth --truth $DS --out output/rrf/baselines_m_multi_relposes_depth.json < /dev/null 2>&1 | grep -v jitc | tail -8 >> $LOG

# T. clean re-timings (quality already measured in round 21)
stamp "retime inria_MVDR"; T0=$(date +%s)
$PYT train.py -s RF-3DGS_dataset/training-rf-spectrum/3dgs_MVDR_100 -m output/sota/inria_MVDR_retime --iterations 40000 --start_checkpoint $CKW --eval --test_iterations 40000 --save_iterations 40000 < /dev/null > output/sota/inria_MVDR_retime.train.log 2>&1
echo "wall inria_MVDR (train only, 30k->40k) $(( $(date +%s) - T0 )) s" >> $LOG
stamp "retime sota_MVDR_rgb"; T0=$(date +%s)
$WSL sota_MVDR_rgb_retime --checkpoint $CK --source RF-3DGS_dataset/training-rf-spectrum/3dgs_MVDR_100 --mode rgb --save-renders 0 >> $LOG 2>&1 < /dev/null
echo "wall sota_MVDR_rgb (train + eval, no render write) $(( $(date +%s) - T0 )) s" >> $LOG
stamp "all done"
