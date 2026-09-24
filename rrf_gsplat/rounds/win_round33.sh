#!/bin/bash
# Round 33 (after round 32): fill the cells the round 24-32 notes left MISSING or flagged.
#   retime   MULTI 300x200 generation on a quiet machine (round 26's 100 s overlapped CPU-side checks for ~20 s)
#   E1+IO    the engineering-only config with the current I/O code on the round-26 dataset (the chain table's
#            "+ engineering" row)
#   decode   eval_baselines timed on round 24's B0 (the chain table's decode cell)
#   seeds    db B0 and M2 at seeds 1 and 2, with mvdr_peaks: is M2's 0.5 deg worse main-peak median noise?
set -u
REPO=/c/Users/Ke/Documents/GitHub/RF-3DGS
PYS=/c/Users/Ke/miniconda3/envs/rf-sionna-win/python.exe
WSL="wsl.exe -d Ubuntu-22.04 -u ke -- bash /mnt/c/Users/Ke/Documents/GitHub/RF-3DGS/rrf_gsplat/wsl_run.sh"
SCENE=../sionna_tutorial/RF-3DGS_Sionna_simulation_tutorial/NIST_lobby_v1.0/NIST_lobby_V1.1_sionna12.xml
RXLOC=../sionna_tutorial/RF-3DGS_Sionna_simulation_tutorial/NIST_rx_loc.txt
REG=RF-3DGS_dataset/regenerated
export PYTHONUTF8=1 MSYS_NO_PATHCONV=1
LOG=$REPO/output/rrf/win_round33.log
cd $REPO
stamp() { echo "== $1 $(date +%H:%M:%S) gpu $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader | tr -d '\r')" >> $LOG; }
until grep -aq "== all done" output/rrf/win_round32.log 2>/dev/null; do sleep 20; done
stamp "round 33 start"
run() {
  local name=$1; shift
  [ -f output/rrf/$name/results.json ] && return
  stamp "start $name"; local T0=$(date +%s)
  $WSL $name "$@" >> $LOG 2>&1 < /dev/null
  echo "wall $name $(( $(date +%s) - T0 )) s" >> $LOG
}
# retime: into a scratch directory, deleted afterwards (same settings as r26_MULTI_300_1M)
stamp "retime generate"; T0=$(date +%s)
(cd sionna_port && $PYS generate_dataset.py --scene-xml $SCENE --rx-loc-file $RXLOC --out-dir ../$REG/r33_retime \
    --spectrum MULTI --num-positions 800 --frequency 2.4e9 --materials tutorial --splat-sigma 3 --no-dashboard \
    < /dev/null 2>&1 | grep -a "views/s\|Error\|Traceback" >> $LOG)
echo "wall retime generate $(( $(date +%s) - T0 )) s" >> $LOG
rm -rf $REG/r33_retime
M="--mode multi --delay-depth --delay-depth-mode ED --delay-range euclid --save-renders -1"
run r33_E1io_multi --source $REG/r26_MULTI_300_1M $M --sh-backend gsplat --eval-group --iterations 10000
stamp "decode timing"; T0=$(date +%s)
$PYS rrf_gsplat/eval_baselines.py --multi output/rrf/r24_B0_multi --truth $REG/3dgs_MULTI_24ghz_tut_cs \
    --out output/rrf/baselines_r24_B0_multi_retimed.json < /dev/null 2>&1 | grep -v jitc | tail -1 >> $LOG
echo "wall decode r24_B0_multi $(( $(date +%s) - T0 )) s" >> $LOG
for s in 1 2; do
  run r33_B0_db_s$s --source $REG/3dgs_MVDR_100_gpct --mode db --save-renders -1 --iterations 10000 --seed $s
  run r33_M2_db_s$s --source $REG/3dgs_MVDR_100_gpct --mode db --save-renders -1 --sh-backend gsplat --eval-group \
      --faces-per-step 4 --lr-scale 2 --iterations 2500 --eval-every 250 --seed $s
done
for n in r33_B0_db_s1 r33_B0_db_s2 r33_M2_db_s1 r33_M2_db_s2; do
  $PYS rrf_gsplat/mvdr_peaks.py --run output/rrf/$n --truth $REG/3dgs_MVDR_100_gpct >> $LOG 2>&1
done
stamp "all done"
