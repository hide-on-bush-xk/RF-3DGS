#!/bin/bash
# Round 9c: run-to-run noise of the transfer curve's y axis. The benefits on
# the curve are 0.1-0.6 dB of in-range RMSE; the rasteriser's atomics make a
# repeated run differ, and that spread has never been measured. Repeats the
# cold reference and two transfer points with identical settings; each
# replicate takes its own --seed (view order), so the spread covers the
# training-order randomness as well as the rasteriser's atomics. Waits for
# round 9b's "all done" marker.
set -u
REPO=/c/Users/Ke/Documents/GitHub/RF-3DGS
REG=RF-3DGS_dataset/regenerated
WSL="wsl.exe -d Ubuntu-22.04 -u ke -- bash /mnt/c/Users/Ke/Documents/GitHub/RF-3DGS/rrf_gsplat/wsl_run.sh"
export PYTHONUTF8=1 MSYS_NO_PATHCONV=1
LOG=$REPO/output/rrf/win_round9c.log
cd $REPO
echo "== start $(date)" >> $LOG
for i in $(seq 1 1440); do grep -q "all done" output/rrf/win_round9b.log 2>/dev/null && break; sleep 5; done
echo "== round 9b finished, gpu free $(date)" >> $LOG
run() { local name=$1; shift; [ -f output/rrf/$name/results.json ] && { echo "skip $name" >> $LOG; return; }
        echo "== $name $(date +%H:%M:%S)" >> $LOG; $WSL "$name" "$@" >> $LOG 2>&1 < /dev/null; }
for rep in 1 2; do
  run t_txB_cold_rep$rep --source $REG/3dgs_MVDR_txB_gpct --mode db --eval-every 250 --save-renders 0 --seed $rep
  run t_txB_geomD_frozen_rep$rep --source $REG/3dgs_MVDR_txB_gpct --mode db --eval-every 250 --save-renders 0 --seed $rep \
      --init-from /mnt/c/Users/Ke/Documents/GitHub/RF-3DGS/output/rrf/a_txD_db_geom/rrf_state.pt --init-geometry-only
  run t_txB_geomA_frozen_rep$rep --source $REG/3dgs_MVDR_txB_gpct --mode db --eval-every 250 --save-renders 0 --seed $rep \
      --init-from /mnt/c/Users/Ke/Documents/GitHub/RF-3DGS/output/rrf/a_mvdr_db_geom/rrf_state.pt --init-geometry-only
done
echo "== all done $(date)" >> $LOG
