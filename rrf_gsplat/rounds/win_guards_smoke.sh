#!/bin/bash
# Smoke for the guards and early stopping (Ke, 2026-09-24: the order agreed; "一万多iter的实验，数据应该早就收敛了").
# Criteria, written before the run:
#   E1  clip guard: on 3dgs_APS_60_gpct (25 % of views clipped) train_rrf refuses before training without --allow-clip,
#       and starts with it
#   E2  --visits-per-view 5 on the 640-view capacity subset (4 views per step) -> 800 iterations; results.json
#       iterations_run 800, visits_per_view 5.0
#   E3  --early-stop-on train (capacity run, max 20000 steps, comm every 500, patience 3, min 2000): stopped or not,
#       results.json and live/status.json agree on it; training-view comm records on 120 views; a final evaluation
#   E4  --early-stop-on val (3000 steps max, comm every 500, patience 2, min 1000): the same mechanics
#   E5  overhead: 3000 steps with both comm sets every 1000 steps (no stop) <= 3 % above the same run with
#       --live-comm-every 0 (51.48 s in win_comm_overhead_smoke.sh, same configuration, idle GPU)
set -u
REPO=/c/Users/Ke/Documents/GitHub/RF-3DGS
PYW=/c/Users/Ke/miniconda3/envs/rf-gsplat-win/python.exe
PYS=/c/Users/Ke/miniconda3/envs/rf-sionna-win/python.exe
APU=RF-3DGS_dataset/regenerated/3dgs_APS_60_gp100
APC=RF-3DGS_dataset/regenerated/3dgs_APS_60_gpct
export PYTHONUTF8=1
LOG=$REPO/output/rrf/m3/win_guards_smoke.log
cd $REPO
echo "== guards smoke $(date +%H:%M:%S)" >> $LOG
C="--mode power --protocol rrf_gsplat/protocol_v1 --eval-set val --sh-backend gsplat --eval-group --faces-per-step 4 \
--lr-scale 2 --sh-degree 3 --seed 0 --max-train-views 640 --save-renders 0"
d=output/rrf/m3/guards
mkdir -p $d
# E1
$PYW rrf_gsplat/train_rrf.py --out $d/e1_noallow $C --source $APC --iterations 20 --eval-every 20 > $d/e1_noallow.log 2>&1
e1a=$?
$PYW rrf_gsplat/train_rrf.py --out $d/e1_allow $C --source $APC --iterations 20 --eval-every 20 --allow-clip > $d/e1_allow.log 2>&1
e1b=$?
echo "E1 without --allow-clip: exit $e1a, $(grep -a 'clipped by the training range' $d/e1_noallow.log | head -1 | cut -c1-80)" >> $LOG
echo "E1 with --allow-clip: exit $e1b, $(grep -a 'training views peak above' $d/e1_allow.log | head -1)" >> $LOG
# E2
$PYW rrf_gsplat/train_rrf.py --out $d/e2 $C --source $APU --visits-per-view 5 --eval-every 100000 > $d/e2.log 2>&1
# E3
mkdir -p $d/e3/live; echo "guards smoke E3: early stop on training views (capacity run)" > $d/e3/live/desc.txt
$PYW rrf_gsplat/train_rrf.py --out $d/e3 $C --source $APU --iterations 20000 --eval-every 20000 --early-stop-on train \
    --live-comm-every 500 --early-stop-min-steps 2000 --early-stop-patience 3 > $d/e3.log 2>&1
# E4
mkdir -p $d/e4/live; echo "guards smoke E4: early stop on the validation subset" > $d/e4/live/desc.txt
$PYW rrf_gsplat/train_rrf.py --out $d/e4 $C --source $APU --iterations 3000 --eval-every 3000 --early-stop-on val \
    --live-comm-every 500 --early-stop-min-steps 1000 --early-stop-patience 2 > $d/e4.log 2>&1
# E5
$PYW rrf_gsplat/train_rrf.py --out $d/e5 $C --source $APU --iterations 3000 --eval-every 3000 --early-stop-on train \
    --live-comm-every 1000 --early-stop-min-steps 99999 > $d/e5.log 2>&1
$PYS - >> $LOG 2>&1 <<EOF
import json
d = "$d"
r2 = json.load(open(f"{d}/e2/results.json"))
print(f"E2 iterations_run {r2['iterations_run']}, visits_per_view {r2['visits_per_view']:.2f} -> "
      f"{'PASS' if r2['iterations_run'] == 800 and abs(r2['visits_per_view'] - 5) < 1e-9 else 'FAIL'}")
for e, sets in (("e3", "train"), ("e4", "val")):
    r = json.load(open(f"{d}/{e}/results.json")); st = json.load(open(f"{d}/{e}/live/status.json"))
    recs = [json.loads(l) for l in open(f"{d}/{e}/live.jsonl") if '"comm"' in l]
    mine = [x for x in recs if x.get("set") == sets]
    views = sorted({x["comm"]["views"] for x in mine})
    agree = bool(r["stopped_early"]) == bool(st.get("stopped_early", False))
    ok = agree and views == [120] and r["final"] is not None
    print(f"{e.upper()} iterations_run {r['iterations_run']}, stopped_early {r['stopped_early']} ({r['stop_reason']}), "
          f"status agrees {agree}, {len(mine)} {sets} records on {views} views, final eval {r['final'] is not None} "
          f"-> {'PASS' if ok else 'FAIL'}")
r5 = json.load(open(f"{d}/e5/results.json"))
over = 100 * (r5["train_seconds"] - 51.48) / 51.48
print(f"E5 training {r5['train_seconds']:.2f} s vs 51.48 s: overhead {over:+.2f} % -> {'PASS' if over <= 3 else 'FAIL'}")
EOF
echo "== all done $(date +%H:%M:%S)" >> $LOG
