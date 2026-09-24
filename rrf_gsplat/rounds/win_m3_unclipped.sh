#!/bin/bash
# The M3 placement oracle with an UNCLIPPED training range (Ke: "普通组一个", 2026-09-24). The visits control
# (win_m3_visits.sh) found emitters + --em-pcolor 8 at 41.7 % distinct <= 1 deg, 56.9 % on the unclipped views and
# 11.4 % on the clipped ones: global-pct (1-99.9 percentile) flat-tops the peaks of 25 % of the APS views, and a
# plateau carries no peak position. Here the same runs on 3dgs_APS_60_gp100 = the same spectra renormalised to the
# 1-100 percentile range (the top = the global maximum, -64.77 dB; span 79 instead of 55 dB):
#   A      emitters + --em-pcolor 8, 40,000 steps, seeds 0 1 2
#   plain  40,000 steps, seed 0 (the control; it has sat at ~25 % in every configuration)
# Check before training: no view's maximum above the new range's top (0 clipped views); else stop.
# Reading, written before the run (primary: distinct <= 1 deg on the 639 capacity training views, 3-seed mean of A):
#   A >= 50 %   placement + a per-position switch is a working representation at the capacity level -> next: the
#               validation set (emitters from training positions only) and placement without the oracle (P4)
#   A <= 30 %   unclipping hurts (the wider span coarsens the value's resolution) -> keep global-pct, report the
#               clipped views separately
#   between     read the former clipped / unclipped views separately before deciding
set -u
REPO=/c/Users/Ke/Documents/GitHub/RF-3DGS
PYW=/c/Users/Ke/miniconda3/envs/rf-gsplat-win/python.exe
PYS=/c/Users/Ke/miniconda3/envs/rf-sionna-win/python.exe
AP=RF-3DGS_dataset/regenerated/3dgs_APS_60
APU=RF-3DGS_dataset/regenerated/3dgs_APS_60_gp100
EM=output/rrf/m3/emitters_160.npz
export PYTHONUTF8=1
LOG=$REPO/output/rrf/m3/win_m3_unclipped.log
cd $REPO
echo "== m3 unclipped start $(date +%H:%M:%S)" >> $LOG
if [ ! -f $APU/generation_meta.json ]; then
  $PYW rrf_gsplat/renormalize.py $AP $APU --norm global-pct --pct 1 100 >> $LOG 2>&1
fi
$PYS - >> $LOG 2>&1 <<EOF || { echo "data check failed: not training" >> $LOG; exit 1; }
import json, os, sys
import numpy as np
sys.path.insert(0, "rrf_gsplat"); import protocol as PR
PR.check_dataset("rrf_gsplat/protocol_v1", "$APU")
m = json.load(open("$APU/generation_meta.json"))
d = "$APU/spectra_float"
over = sum(float(np.load(os.path.join(d, f)).max()) > m["spec_max_db"] for f in os.listdir(d))
print(f"gp100 range {m['spec_min_db']:.2f} .. {m['spec_max_db']:.2f} dB; views above the top: {over}")
sys.exit(0 if over == 0 else 1)
EOF
B="--mode power --source $APU --protocol rrf_gsplat/protocol_v1 --eval-set val --sh-backend gsplat --eval-group --faces-per-step 4 \
--lr-scale 2 --sh-degree 3 --iterations 40000 --eval-every 40000 --max-train-views 640 --save-renders 0"
run() {
  local name=$1; shift
  if [ ! -f output/rrf/m3/$name/results.json ]; then
    mkdir -p output/rrf/m3/$name
    $PYW rrf_gsplat/train_rrf.py --out output/rrf/m3/$name $B "$@" > output/rrf/m3/$name/train.log 2>&1
  fi
  echo "$name: $(grep -a 'iterations in' output/rrf/m3/$name/train.log | tail -1 | cut -c1-120)" >> $LOG
  $PYW rrf_gsplat/diag_train_fit.py --run output/rrf/m3/$name --train-subset 640 >> $LOG 2>&1
}
for s in 0 1 2; do run m3u_em_pc8_40k_s$s --seed $s --emitters $EM --em-pcolor 8; done
run m3u_plain_40k_s0 --seed 0
echo "== all done $(date +%H:%M:%S)" >> $LOG
