#!/bin/bash
# Round 42 (after round 41): the shading head's share of the output where there is signal. Every head run is
# re-rendered from its own state with the head switched off (0 steps, --head none: the Gaussians alone), and
# head_share.py compares the two on in-range and peak pixels. The residual RMS train_rrf.py reports pools the
# floor, where the head pushes to its bound for free, and sat at 6.00 dB.
set -u
REPO=/c/Users/Ke/Documents/GitHub/RF-3DGS
PYS=/c/Users/Ke/miniconda3/envs/rf-sionna-win/python.exe
WSL="wsl.exe -d Ubuntu-22.04 -u ke -- bash /mnt/c/Users/Ke/Documents/GitHub/RF-3DGS/rrf_gsplat/wsl_run.sh"
MV=RF-3DGS_dataset/regenerated/3dgs_MVDR_100_gpct
export PYTHONUTF8=1 MSYS_NO_PATHCONV=1
LOG=$REPO/output/rrf/win_round42.log
cd $REPO
stamp() { echo "== $1 $(date +%H:%M:%S)" >> $LOG; }
until grep -aq "== all done" output/rrf/win_round41.log 2>/dev/null; do sleep 30; done
stamp "round 42 start"
for spec in S1_head:1 S2_geo:1 S3_geo_phys:1 S4_geo_phys_lat:1 S5_full:1 S5_minus_geo:1 S5_minus_phys:1 S5_minus_lat:1 \
            S5_minus_strip:1 S5_w8:1 S5_peak:1 B6_sh3_head:3 B5_sh3_head_peak:3 S5_full_s1:1 S5_full_s2:1 \
            B5_sh3_head_peak_s1:3 B5_sh3_head_peak_s2:3; do
  tag=${spec%%:*}; deg=${spec##*:}; run=r41_B_$tag; base=r42_base_$tag
  [ -f output/rrf/$run/rrf_state.pt ] || { echo "no $run" >> $LOG; continue; }
  [ -f output/rrf/$base/results.json ] || $WSL $base --mode db --source $MV --sh-backend gsplat --eval-group --sh-degree $deg \
      --init-from /mnt/c/Users/Ke/Documents/GitHub/RF-3DGS/output/rrf/$run/rrf_state.pt --iterations 0 --save-renders -1 >> $LOG 2>&1 < /dev/null
  $PYS rrf_gsplat/head_share.py --run output/rrf/$run --base output/rrf/$base --truth $MV >> $LOG 2>&1
done
stamp "all done"
