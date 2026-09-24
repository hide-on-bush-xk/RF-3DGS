#!/bin/bash
# M3 placement oracle, smoke (docs/tech_paths.md M3 / P4; the stage plan: stage 2's gate closed stage 3 at 26.8 %).
# Question of the full run: do Gaussians placed where the radio energy really comes from (the ray tracer's
# interaction points, sionna_port/path_emitters.py), rendered in their own pass and added in linear power, lift the
# capacity benchmark on the APS target? This smoke only checks the path, on 12 stratified positions (both ends of
# the capacity list, the flat view's position 460, the weakest position 463):
#   builder     path_emitters.py checks (index, frame, peaks) -- output/rrf/m3/emitters_smoke.json
#   model       check_emitters.py S1-S3 (init identity, gradients, loss falls) -- output/rrf/m3/check_emitters.json
#   S4          train_rrf.py on the 48 smoke views, 3000 steps, power mode, with and without --emitters, scored on
#               those views by diag_train_fit.py: <= 1 deg and step time REPORTED, not judged (12 positions say
#               nothing about 160)
set -u
REPO=/c/Users/Ke/Documents/GitHub/RF-3DGS
PYW=/c/Users/Ke/miniconda3/envs/rf-gsplat-win/python.exe
APG=RF-3DGS_dataset/regenerated/3dgs_APS_60_gpct
export PYTHONUTF8=1
LOG=$REPO/output/rrf/m3/win_m3_smoke.log
cd $REPO
echo "== m3 smoke start $(date +%H:%M:%S)" >> $LOG
B="--mode power --source $APG --protocol rrf_gsplat/protocol_v1 --eval-set val --sh-backend gsplat --eval-group --faces-per-step 4 \
--lr-scale 2 --sh-degree 3 --seed 0 --iterations 3000 --eval-every 3000 --save-renders 0 --train-names-file output/rrf/m3/smoke_names.txt"
run() {
  local name=$1; shift
  if [ ! -f output/rrf/m3/$name/results.json ]; then
    mkdir -p output/rrf/m3/$name
    $PYW rrf_gsplat/train_rrf.py --out output/rrf/m3/$name $B "$@" > output/rrf/m3/$name/train.log 2>&1
  fi
  echo "$name: $(grep -a 'iterations in' output/rrf/m3/$name/train.log | tail -1 | cut -c1-150)" >> $LOG
  $PYW rrf_gsplat/diag_train_fit.py --run output/rrf/m3/$name >> $LOG 2>&1
}
run smoke_plain
run smoke_em --emitters output/rrf/m3/emitters_smoke.npz
# round 2 (Ke, 2026-09-24: "OK先smoke"): the emitters with a position-conditioned colour (--em-pcolor 8), so an emitter
# can be on for some receiver positions and off for others; check_emitters.py --em-pcolor 8 first (S1 bit-exact
# against the emitters alone, S2 latent + MLP gradients from the second step, S3, S5)
run smoke_em_pc8 --emitters output/rrf/m3/emitters_smoke.npz --em-pcolor 8
echo "== all done $(date +%H:%M:%S)" >> $LOG
