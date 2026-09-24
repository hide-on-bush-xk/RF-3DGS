#!/bin/bash
# Round 10a (runs beside the 9b queue on the shared GPU; results, not timings):
#   calibration -- is a 2k-step transfer budget enough to rank sources? The
#   10k-step points exist (cold 4.82, A 4.58, C 4.95, D 4.27 in-range RMSE);
#   the same four at 2k steps must give the same signs and the same order.
#   noise floor -- cold and A->B at 2k steps, seeds 1..3 (seed 0 is the
#   calibration run), so the benefit's run-to-run spread is known.
set -u
REPO=/c/Users/Ke/Documents/GitHub/RF-3DGS
REG=RF-3DGS_dataset/regenerated
WSL="wsl.exe -d Ubuntu-22.04 -u ke -- bash /mnt/c/Users/Ke/Documents/GitHub/RF-3DGS/rrf_gsplat/wsl_run.sh"
export PYTHONUTF8=1 MSYS_NO_PATHCONV=1
LOG=$REPO/output/rrf/win_round10a.log
OUT=/mnt/c/Users/Ke/Documents/GitHub/RF-3DGS/output/rrf
cd $REPO
echo "== start $(date)" >> $LOG
run() { local name=$1; shift; [ -f output/rrf/$name/results.json ] && { echo "skip $name" >> $LOG; return; }
        echo "== $name $(date +%H:%M:%S)" >> $LOG; $WSL "$name" "$@" >> $LOG 2>&1 < /dev/null; echo "== done $name $(date +%H:%M:%S)" >> $LOG; }
COMMON="--source $REG/3dgs_MVDR_txB_gpct --mode db --iterations 2000 --eval-every 500 --save-renders 0"
# calibration, seed 0
run t_txB_cold_2k          $COMMON --seed 0
run t_txB_geomA_frozen_2k  $COMMON --seed 0 --init-from $OUT/a_mvdr_db_geom/rrf_state.pt --init-geometry-only
run t_txB_geomC_frozen_2k  $COMMON --seed 0 --init-from $OUT/a_txC_db_geom/rrf_state.pt --init-geometry-only
run t_txB_geomD_frozen_2k  $COMMON --seed 0 --init-from $OUT/a_txD_db_geom/rrf_state.pt --init-geometry-only
# noise floor, seeds 1..3
for s in 1 2 3; do
  run t_txB_cold_2k_s$s         $COMMON --seed $s
  run t_txB_geomA_frozen_2k_s$s $COMMON --seed $s --init-from $OUT/a_mvdr_db_geom/rrf_state.pt --init-geometry-only
done
echo "== all done $(date)" >> $LOG
