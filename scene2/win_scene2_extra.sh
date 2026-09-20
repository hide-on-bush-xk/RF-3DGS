#!/bin/bash
# Scene 2 follow-ups after Ke's review of experiments A/B/C:
#   #3  whole-room hold-out (room S1, 47 positions, nearest training 2.2 m):
#       the registered extrapolation prediction; renders saved for all 188
#       held-out views so the baselines score the whole set.
#   seed noise floor on the zone experiment: the target's cold reference and
#       the two positive transfers (N2, S2b) at seeds 1-3 / 1-2.
set -u
REPO=/c/Users/Ke/Documents/GitHub/RF-3DGS
PYS=/c/Users/Ke/miniconda3/envs/rf-sionna-win/python.exe
REG=RF-3DGS_dataset/regenerated
CK=/mnt/c/Users/Ke/Documents/GitHub/RF-3DGS/scene2/visual_trained/chkpnt30000.pth
WSL="wsl.exe -d Ubuntu-22.04 -u ke -- bash /mnt/c/Users/Ke/Documents/GitHub/RF-3DGS/rrf_gsplat/wsl_run.sh"
OUT=/mnt/c/Users/Ke/Documents/GitHub/RF-3DGS/output/rrf
export PYTHONUTF8=1 MSYS_NO_PATHCONV=1
LOG=$REPO/output/rrf/win_scene2_extra.log
cd $REPO
stamp() { echo "== $1 $(date +%H:%M:%S)" >> $LOG; }
run() { local name=$1; shift; [ -f output/rrf/$name/results.json ] && { echo "skip $name" >> $LOG; return; }
        stamp "$name"; $WSL "$name" --checkpoint $CK "$@" >> $LOG 2>&1 < /dev/null; stamp "done $name"; }
stamp "start"
# prediction 3: hold out room S1 entirely
run s2_multi_corrM_depth_roomS1 --source $REG/s2_MULTI_corrM_roomS1 --mode multi --delay-depth --save-renders 188
stamp "baselines roomS1"
$PYS rrf_gsplat/eval_baselines.py --multi output/rrf/s2_multi_corrM_depth_roomS1 --truth $REG/s2_MULTI_corrM_roomS1 \
     --out output/rrf/baselines_s2_multi_corrM_depth_roomS1.json < /dev/null 2>&1 | grep -v jitc >> $LOG
# seed noise floor of the zone experiment (2k-step transfers on room_S2)
TGT=$REG/s2_MVDR_txroom_S2_gpct
for s in 1 2 3; do run s2_t_roomS2_cold_2k_s$s --source $TGT --mode db --iterations 2000 --eval-every 500 --save-renders 0 --seed $s; done
for src in room_N2 room_S2b; do for s in 1 2; do
  run s2_t_roomS2_geom${src}_2k_s$s --source $TGT --mode db --iterations 2000 --eval-every 500 --save-renders 0 --seed $s \
      --init-from $OUT/s2_a_tx${src}_geom/rrf_state.pt --init-geometry-only
done; done
stamp "all done"
