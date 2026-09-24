#!/bin/bash
# Round 43 smoke (after round 42): the shading head with its output layer in units of the bound
# (--head-bound-units, neural_shading.ResidualCNN). Round 41 found 3 of its head runs (S1 seed 0, S2 seed 0,
# S5 seed 1) sitting at the bound, residual RMS 6.00 dB, from step 250 to the end: dead heads, and the B ladder's
# "+ phys guides" step is S2 dead -> S3 alive. 750 steps, the round-41 arguments otherwise, no renders saved.
# Criteria, written before the run:
#   a  check_shading_head.py: ALL PASS (identity at the start in both parametrisations, guides, gradients)
#   b  control, the known failures in the old parametrisation: r43s_old_S5_s1 and r43s_old_S1_s0 are at
#      >= 5.99 dB residual RMS at step 250. If not, the smoke cannot tell the fix from luck: reported, no claim
#   c  the fix on the three known failures, the live seed and the SH3 branch: residual RMS < 5.9 dB at every
#      evaluation (250, 500, 750) of every r43s_bu_* run
#   d  the fix costs the live case nothing: PSNR at 750 >= the round-41 run's at 750 minus 0.5 dB
#      (S5 seed 0: 20.49 -> 19.99; B6 seed 0: 21.04 -> 20.54)
#   expected: 1.5-2.5 min a run, about 15 min in all; residual RMS 3-5.5 dB where the head is alive
set -u
REPO=/c/Users/Ke/Documents/GitHub/RF-3DGS
PYS=/c/Users/Ke/miniconda3/envs/rf-sionna-win/python.exe
WSL="wsl.exe -d Ubuntu-22.04 -u ke -- bash /mnt/c/Users/Ke/Documents/GitHub/RF-3DGS/rrf_gsplat/wsl_run.sh"
WSLC="wsl.exe -d Ubuntu-22.04 -u ke -- bash -c"
MV=RF-3DGS_dataset/regenerated/3dgs_MVDR_100_gpct
export PYTHONUTF8=1 MSYS_NO_PATHCONV=1
LOG=$REPO/output/rrf/win_round43s.log
cd $REPO
stamp() { echo "== $1 $(date +%H:%M:%S) gpu $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader | tr -d '\r')" >> $LOG; }
until grep -aq "== all done" output/rrf/win_round42.log 2>/dev/null; do sleep 30; done
stamp "round 43 smoke start"
$WSLC "cd /mnt/c/Users/Ke/Documents/GitHub/RF-3DGS && /home/ke/miniconda3/envs/rf-gsplat/bin/python rrf_gsplat/check_shading_head.py" \
    2>&1 < /dev/null | tail -2 | sed "s/^/check_shading_head: /" >> $LOG

B="--mode db --source $MV --sh-backend gsplat --eval-group --faces-per-step 4 --lr-scale 2 --iterations 750 --eval-every 250 --save-renders 0"
S5="--sh-degree 1 --head cnn --head-guides geo,phys --head-latent 4 --head-strip"
run() {   # name, args
  local name=$1; shift
  [ -f output/rrf/$name/results.json ] && return
  stamp "S $name"; local T0=$(date +%s)
  $WSL $name $B "$@" >> $LOG 2>&1 < /dev/null
  echo "wall $name $(( $(date +%s) - T0 )) s" >> $LOG
}
run r43s_old_S5_s1  $S5 --seed 1
run r43s_old_S1_s0  --sh-degree 1 --head cnn
run r43s_bu_S5_s1   $S5 --seed 1 --head-bound-units
run r43s_bu_S1_s0   --sh-degree 1 --head cnn --head-bound-units
run r43s_bu_S2_s0   --sh-degree 1 --head cnn --head-guides geo --head-bound-units
run r43s_bu_S5_s0   $S5 --head-bound-units
run r43s_bu_B6_s0   --sh-degree 3 --head cnn --head-guides geo,phys --head-latent 4 --head-strip --head-bound-units

$PYS - >> $LOG 2>&1 <<'EOF'
import json, os
R = "output/rrf"
def hist(n):
    p = os.path.join(R, n, "results.json")
    if not os.path.isfile(p):
        return None
    return {h["iteration"]: h for h in json.load(open(p))["history"]}
ok = True
for n in ("r43s_old_S5_s1", "r43s_old_S1_s0"):
    h = hist(n); v = None if h is None else h[250]["head_residual_rms_db"]
    b = v is not None and v >= 5.99; ok &= b
    print(f"b {n}: residual RMS at 250 = {v} -> {'PASS' if b else 'FAIL'}")
for n in ("r43s_bu_S5_s1", "r43s_bu_S1_s0", "r43s_bu_S2_s0", "r43s_bu_S5_s0", "r43s_bu_B6_s0"):
    h = hist(n); v = None if h is None else [round(h[i]["head_residual_rms_db"], 2) for i in (250, 500, 750)]
    c = v is not None and max(v) < 5.9; ok &= c
    print(f"c {n}: residual RMS at 250/500/750 = {v}, PSNR {None if h is None else [round(h[i]['psnr_rgb'], 2) for i in (250, 500, 750)]} -> {'PASS' if c else 'FAIL'}")
for n, ref in (("r43s_bu_S5_s0", 20.49), ("r43s_bu_B6_s0", 21.04)):
    h = hist(n); v = None if h is None else h[750]["psnr_rgb"]
    d = v is not None and v >= ref - 0.5; ok &= d
    print(f"d {n}: PSNR at 750 = {v} (round 41: {ref}; floor {ref - 0.5:.2f}) -> {'PASS' if d else 'FAIL'}")
print("criteria b-d:", "ALL PASS" if ok else "FAIL")
EOF
stamp "all done"
