#!/bin/bash
# Round 22b (after round 23): the Track D block of round 22, which exited in 4 s because generate_dataset.py
# requires --rx-loc-file even with --poses-from (the filter hid argparse's message). Same commands otherwise.
set -u
REPO=/c/Users/Ke/Documents/GitHub/RF-3DGS
PYS=/c/Users/Ke/miniconda3/envs/rf-sionna-win/python.exe
CK=/mnt/c/Users/Ke/Documents/GitHub/RF-3DGS/RF-3DGS_dataset/blender_visual_trained/chkpnt30000.pth
WSL="wsl.exe -d Ubuntu-22.04 -u ke -- bash /mnt/c/Users/Ke/Documents/GitHub/RF-3DGS/rrf_gsplat/wsl_run.sh"
REG=RF-3DGS_dataset/regenerated
export PYTHONUTF8=1 MSYS_NO_PATHCONV=1
LOG=$REPO/output/rrf/win_round22b.log
cd $REPO
stamp() { echo "== $1 $(date +%H:%M:%S)" >> $LOG; }
gpu_busy() {
  tasklist 2>/dev/null | grep -qi "Against the Storm" && return 0
  u=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr -d ' \r'); [ "${u:-100}" -ge 15 ]
}
until grep -aq "== all done" output/tx_planning/win_round23.log 2>/dev/null; do sleep 60; done
quiet=0; while [ $quiet -lt 24 ]; do if gpu_busy; then quiet=0; else quiet=$((quiet+1)); fi; sleep 5; done
stamp "start (gpu $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader | tr -d '\r'))"

DS=$REG/3dgs_MULTI_relposes_24ghz_tut
if [ ! -f $DS/generation_meta.json ]; then
  stamp "dataset MULTI at released poses"; T0=$(date +%s); cd sionna_port
  $PYS generate_dataset.py --scene-xml ../sionna_tutorial/RF-3DGS_Sionna_simulation_tutorial/NIST_lobby_v1.0/NIST_lobby_V1.1_sionna12.xml \
      --rx-loc-file ../sionna_tutorial/RF-3DGS_Sionna_simulation_tutorial/NIST_rx_loc.txt \
      --poses-from ../RF-3DGS_dataset/training-rf-spectrum/3dgs_MVDR_100/sparse/0/images.txt --out-dir ../$DS \
      --spectrum MULTI --frequency 2.4e9 --materials tutorial --splat-sigma 3 --num-positions 800 --no-dashboard < /dev/null > ../output/rrf/gen_multi_relposes.log 2>&1
  cd $REPO
  grep -a "views/s\|channel ranges\|poses from\|Error\|Traceback\|error:" output/rrf/gen_multi_relposes.log >> $LOG
  echo "wall dataset $(( $(date +%s) - T0 )) s" >> $LOG
  cp RF-3DGS_dataset/training-rf-spectrum/3dgs_MVDR_100/train_index.txt RF-3DGS_dataset/training-rf-spectrum/3dgs_MVDR_100/test_index.txt $DS/ 2>/dev/null
fi
if [ -f $DS/generation_meta.json ] && [ ! -f output/rrf/m_multi_relposes_depth/results.json ]; then
  stamp "m_multi_relposes_depth"; T0=$(date +%s)
  $WSL m_multi_relposes_depth --checkpoint $CK --source $DS --mode multi --delay-depth --delay-depth-mode ED --delay-range euclid --save-renders -1 >> $LOG 2>&1 < /dev/null
  echo "wall $(( $(date +%s) - T0 )) s" >> $LOG; stamp "done m_multi_relposes_depth"
fi
[ -f output/rrf/m_multi_relposes_depth/results.json ] && [ ! -f output/rrf/baselines_m_multi_relposes_depth.json ] && $PYS rrf_gsplat/eval_baselines.py --multi output/rrf/m_multi_relposes_depth --truth $DS --out output/rrf/baselines_m_multi_relposes_depth.json < /dev/null 2>&1 | grep -v jitc | tail -8 >> $LOG
stamp "all done"
