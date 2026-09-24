#!/bin/bash
# Round 43 smoke 2: --head-warmup. Smoke 1 (win_round43s.sh) FAILED criterion c: with the output layer in units of
# the bound, S5 seed 1 lives but S1 and S2 (seed 0) still sit at the bound from step 250 on, at -6 dB on every
# pixel. Diagnosis: the untrained field (SH from zero) renders above most of the truth, so nearly every pixel asks
# for "lower"; the head's constant term is the fastest way down and saturates, the Gaussians then learn
# truth + 6 dB under it (the offset is degenerate between the two), and tanh has no gradient left. Pixels below
# the range are only 1 % of the total, so it is not the floor. The fix: the head off for the first 250 steps.
# 1000 steps, eval every 250, the round-41 arguments otherwise, no renders saved. Criteria, written before the run:
#   a  check_shading_head.py: ALL PASS (the render path changed: the head_on switch)
#   b  control: smoke 1's r43s_bu_S1_s0 and r43s_bu_S2_s0 (no warm-up) are the known failures, 6.00 dB at every
#      evaluation; already recorded, not rerun
#   c  bound units + warm-up 250: head residual RMS < 5.9 dB at 500, 750 and 1000 in r43s2_S1_s0, r43s2_S2_s0,
#      r43s2_S5_s1, r43s2_S5_s0 and r43s2_B6_s0 (at 250 the head is still off)
#   d  no cost to the live case: PSNR at 1000 >= round 41's at 1000 minus 0.5 dB (S5 seed 0: 21.23 -> 20.73;
#      B6 seed 0: 21.51 -> 21.01)
#   no criterion (to say which fix is needed): warm-up alone, without bound units, on S1 seed 0 and S5 seed 1;
#   and S1 / S2 PSNR at 1000 against S0 (SH1, no head) at 1000, 17.77
#   expected: 2-2.5 min a run, about 17 min in all
set -u
REPO=/c/Users/Ke/Documents/GitHub/RF-3DGS
PYS=/c/Users/Ke/miniconda3/envs/rf-sionna-win/python.exe
WSL="wsl.exe -d Ubuntu-22.04 -u ke -- bash /mnt/c/Users/Ke/Documents/GitHub/RF-3DGS/rrf_gsplat/wsl_run.sh"
WSLC="wsl.exe -d Ubuntu-22.04 -u ke -- bash -c"
MV=RF-3DGS_dataset/regenerated/3dgs_MVDR_100_gpct
export PYTHONUTF8=1 MSYS_NO_PATHCONV=1
LOG=$REPO/output/rrf/win_round43s2.log
cd $REPO
stamp() { echo "== $1 $(date +%H:%M:%S) gpu $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader | tr -d '\r')" >> $LOG; }
stamp "round 43 smoke 2 start"
$WSLC "cd /mnt/c/Users/Ke/Documents/GitHub/RF-3DGS && /home/ke/miniconda3/envs/rf-gsplat/bin/python rrf_gsplat/check_shading_head.py" \
    2>&1 < /dev/null | tail -1 | sed "s/^/check_shading_head: /" >> $LOG

B="--mode db --source $MV --sh-backend gsplat --eval-group --faces-per-step 4 --lr-scale 2 --iterations 1000 --eval-every 250 --save-renders 0"
S5="--sh-degree 1 --head cnn --head-guides geo,phys --head-latent 4 --head-strip"
FIX="--head-bound-units --head-warmup 250"
run() {   # name, args
  local name=$1; shift
  [ -f output/rrf/$name/results.json ] && return
  stamp "S $name"; local T0=$(date +%s)
  $WSL $name $B "$@" >> $LOG 2>&1 < /dev/null
  echo "wall $name $(( $(date +%s) - T0 )) s" >> $LOG
}
run r43s2_S1_s0   --sh-degree 1 --head cnn $FIX
run r43s2_S2_s0   --sh-degree 1 --head cnn --head-guides geo $FIX
run r43s2_S5_s1   $S5 --seed 1 $FIX
run r43s2_S5_s0   $S5 $FIX
run r43s2_B6_s0   --sh-degree 3 --head cnn --head-guides geo,phys --head-latent 4 --head-strip $FIX
run r43s2_wuonly_S1_s0  --sh-degree 1 --head cnn --head-warmup 250
run r43s2_wuonly_S5_s1  $S5 --seed 1 --head-warmup 250

$PYS - >> $LOG 2>&1 <<'EOF'
import json, os
R = "output/rrf"
def hist(n):
    p = os.path.join(R, n, "results.json")
    if not os.path.isfile(p):
        return None
    return {h["iteration"]: h for h in json.load(open(p))["history"]}
def rms(h, i):
    v = h[i].get("head_residual_rms_db")
    return None if v is None else round(v, 2)
ok = True
for n in ("r43s2_S1_s0", "r43s2_S2_s0", "r43s2_S5_s1", "r43s2_S5_s0", "r43s2_B6_s0"):
    h = hist(n); v = None if h is None else [rms(h, i) for i in (500, 750, 1000)]
    c = v is not None and None not in v and max(v) < 5.9; ok &= c
    print(f"c {n}: residual RMS at 500/750/1000 = {v}, PSNR {None if h is None else [round(h[i]['psnr_rgb'], 2) for i in (250, 500, 750, 1000)]} -> {'PASS' if c else 'FAIL'}")
for n, ref in (("r43s2_S5_s0", 21.23), ("r43s2_B6_s0", 21.51)):
    h = hist(n); v = None if h is None else h[1000]["psnr_rgb"]
    d = v is not None and v >= ref - 0.5; ok &= d
    print(f"d {n}: PSNR at 1000 = {v} (round 41: {ref}; floor {ref - 0.5:.2f}) -> {'PASS' if d else 'FAIL'}")
for n in ("r43s2_wuonly_S1_s0", "r43s2_wuonly_S5_s1"):
    h = hist(n)
    print(f"info {n}: residual RMS at 500/750/1000 = {None if h is None else [rms(h, i) for i in (500, 750, 1000)]}, "
          f"PSNR at 1000 = {None if h is None else round(h[1000]['psnr_rgb'], 2)}")
print("criteria c-d:", "ALL PASS" if ok else "FAIL")
EOF
stamp "all done"
