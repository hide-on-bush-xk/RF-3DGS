#!/bin/bash
# Smoke for the live communication metrics at the new default (--live-comm-every 1000, numpy metrics on a worker
# thread; Ke, 2026-09-24: "1k步吧，最主要的还是不要影响实验的速度"). Runs after step 1, on an otherwise idle GPU.
# Criteria, written before the run:
#   O1  at 3000 steps of the capacity configuration, training time with --live-comm-every 1000 is <= 3 % above the
#       same run with --live-comm-every 0
#   O2  comm records at steps 1000, 2000 and 3000, each on 120 views, the last written before the run is marked done
set -u
REPO=/c/Users/Ke/Documents/GitHub/RF-3DGS
PYW=/c/Users/Ke/miniconda3/envs/rf-gsplat-win/python.exe
PYS=/c/Users/Ke/miniconda3/envs/rf-sionna-win/python.exe
APU=RF-3DGS_dataset/regenerated/3dgs_APS_60_gp100
export PYTHONUTF8=1
LOG=$REPO/output/rrf/m3/win_comm_overhead_smoke.log
cd $REPO
echo "== comm overhead smoke $(date +%H:%M:%S)" >> $LOG
B="--mode power --source $APU --protocol rrf_gsplat/protocol_v1 --eval-set val --sh-backend gsplat --eval-group --faces-per-step 4 \
--lr-scale 2 --sh-degree 3 --seed 0 --iterations 3000 --eval-every 3000 --max-train-views 640 --save-renders 0"
for arm in off on; do
  every=0; [ $arm = on ] && every=1000
  rm -rf output/rrf/m3/comm_overhead_$arm
  mkdir -p output/rrf/m3/comm_overhead_$arm/live
  echo "通信指标开销的 smoke（$arm）：3000 步容量配置，--live-comm-every $every，用来量实时通信指标对训练速度的影响。" \
      > output/rrf/m3/comm_overhead_$arm/live/desc.txt
  $PYW rrf_gsplat/train_rrf.py --out output/rrf/m3/comm_overhead_$arm $B --live-comm-every $every \
      > output/rrf/m3/comm_overhead_$arm/train.log 2>&1
done
$PYS - >> $LOG 2>&1 <<EOF
import json
off = json.load(open("output/rrf/m3/comm_overhead_off/results.json"))["train_seconds"]
on_r = json.load(open("output/rrf/m3/comm_overhead_on/results.json"))
on = on_r["train_seconds"]
over = 100 * (on - off) / off
print(f"O1 training {off:.2f} s off, {on:.2f} s on: overhead {over:+.2f} % -> {'PASS' if over <= 3 else 'FAIL'}")
recs = [json.loads(l) for l in open("output/rrf/m3/comm_overhead_on/live.jsonl") if '"comm"' in l]
its = [r["it"] for r in recs]; views = {r["comm"]["views"] for r in recs}
st = json.load(open("output/rrf/m3/comm_overhead_on/live/status.json"))
ok = its == [1000, 2000, 3000] and views == {120} and st.get("done")
print(f"O2 comm records at {its}, views {sorted(views)}, done {st.get('done')} -> {'PASS' if ok else 'FAIL'}")
print(f"   evaluation time outside training: {on_r.get('running_eval_seconds', 0):.2f} s")
EOF
echo "== all done $(date +%H:%M:%S)" >> $LOG
