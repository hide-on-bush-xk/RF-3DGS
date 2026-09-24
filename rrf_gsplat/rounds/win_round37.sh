#!/bin/bash
# Round 37: two questions the notes left open.
#   peaks  does any existing MVDR variant keep the main peak? The recorded runs saved 16 renders; their states
#          are re-rendered on all 640 held-out views (--init-from, 0 steps) -- control: the re-render must
#          reproduce the run's recorded PSNR(jet) / RMSE to 0.01 dB -- and scored with mvdr_peaks.py:
#          frozen db (e2), power compositing (e2_power), unfrozen geometry (a_db_geom, a_power_geom), rgb (e2_rgb)
#   geom   does four faces per step hold with the geometry unfrozen (the configuration of the best MVDR rows)?
#          a_mvdr_db_geom (10k x 1, 302 s) against 2.5k x 4 at lr x2 (every group, the means included)
set -u
REPO=/c/Users/Ke/Documents/GitHub/RF-3DGS
PYS=/c/Users/Ke/miniconda3/envs/rf-sionna-win/python.exe
WSL="wsl.exe -d Ubuntu-22.04 -u ke -- bash /mnt/c/Users/Ke/Documents/GitHub/RF-3DGS/rrf_gsplat/wsl_run.sh"
MV=RF-3DGS_dataset/regenerated/3dgs_MVDR_100_gpct
export PYTHONUTF8=1 MSYS_NO_PATHCONV=1
LOG=$REPO/output/rrf/win_round37.log
cd $REPO
stamp() { echo "== $1 $(date +%H:%M:%S) gpu $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader | tr -d '\r')" >> $LOG; }
stamp "round 37 start"
for spec in "e2_mvdr_db:db:" "e2_mvdr_power:power:" "e2_mvdr_rgb:rgb:" "a_mvdr_db_geom:db:--train-geometry" "a_mvdr_power_geom:power:--train-geometry"; do
  IFS=: read src mode extra <<< "$spec"
  name=r37_rerender_$src
  [ -f output/rrf/$name/results.json ] && continue
  stamp "rerender $src"
  $WSL $name --source $MV --mode $mode $extra --init-from /mnt/c/Users/Ke/Documents/GitHub/RF-3DGS/output/rrf/$src/rrf_state.pt \
      --iterations 0 --save-renders -1 >> $LOG 2>&1 < /dev/null
  $PYS -c "
import json
a=json.load(open('output/rrf/$src/results.json'))['final']; b=json.load(open('output/rrf/$name/results.json'))['final']
d=max(abs(a[k]-b[k]) for k in ('psnr_rgb','rmse_db'))
print('control $src re-render: PSNR %.3f vs %.3f, RMSE %.3f vs %.3f -> %s' % (b['psnr_rgb'], a['psnr_rgb'], b['rmse_db'], a['rmse_db'], 'PASS' if d < 0.01 else 'FAIL'))" >> $LOG 2>&1
  $PYS rrf_gsplat/mvdr_peaks.py --run output/rrf/$name --truth $MV >> $LOG 2>&1
done
stamp "start r37_M2_db_geom"; T0=$(date +%s)
[ -f output/rrf/r37_M2_db_geom/results.json ] || $WSL r37_M2_db_geom --source $MV --mode db --train-geometry --eval-group \
    --faces-per-step 4 --lr-scale 2 --iterations 2500 --eval-every 250 --save-renders -1 >> $LOG 2>&1 < /dev/null
echo "wall r37_M2_db_geom $(( $(date +%s) - T0 )) s" >> $LOG
$PYS rrf_gsplat/mvdr_peaks.py --run output/rrf/r37_M2_db_geom --truth $MV >> $LOG 2>&1
stamp "all done"
