#!/bin/bash
# Smoke of the main-peak direction loss (dir_loss.py; unit checks: check_dir_loss.py). Ke, 2026-09-25: "先做smoke" --
# smoke scale only on this PC; the capacity benchmark and the validation runs wait for the school's GPU account.
# Data: the unclipped range (3dgs_APS_60_gp100), protocol_v1 training list, a route subset of 24 positions (96 views;
# both ends of the route included, all four faces), power, SH3, --faces-per-step 4 (the ring the loss needs), 3000
# steps (~125 visits per view), seed 0, live on. Views without a distinct peak are included as they come; this APS
# dataset has no zero-path positions (the flat case is unit-checked: U7).
#   base   --dir-loss expgain --dir-weight 0 (monitored only)
#   eg     expgain, weight 0.1, T annealed 20 -> 1 dB over 1500 steps, from step 200 (U5b)
#   ce     ce, weight 0.001, T 1 dB, from step 200 (U5c)
# The weights are set so the term starts near the pixel loss's size (expgain: ~10 dB / 79 dB span x 0.1 ~ 0.013;
# ce: ~10 nats x 0.001); the smoke does not tune them.
# Criteria, written before the run:
#   S1  all three finish; the monitor (expgain at T = 1 dB, dB) finite at every logged window
#   S2  liveness: the monitor over the last 300 steps of eg and of ce is >= 0.5 dB below base's
#   S3  the pixel fit survives: training-view RMSE (diag_train_fit.py) of eg and ce within +1.0 dB of base's
#   S4  reported only (one seed, 24 positions: a smoke does not rank): distinct <= 1 deg and the main-peak angle on the
#       96 training views; step time (void when the GPU is shared)
set -u
REPO=/c/Users/Ke/Documents/GitHub/RF-3DGS
PYW=/c/Users/Ke/miniconda3/envs/rf-gsplat-win/python.exe
APU=RF-3DGS_dataset/regenerated/3dgs_APS_60_gp100
PROT=rrf_gsplat/protocol_v1
OUT=output/rrf/dirloss_smoke
export PYTHONUTF8=1
LOG=$REPO/$OUT/smoke.log
cd $REPO; mkdir -p $OUT
stamp() { echo "== $1 $(date +%H:%M:%S) gpu $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader | tr -d '\r')" >> $LOG; }
stamp "start"
B="--mode power --source $APU --protocol $PROT --eval-set val --sh-backend gsplat --eval-group --faces-per-step 4 \
--lr-scale 2 --sh-degree 3 --seed 0 --max-train-views 96 --iterations 3000 --eval-every 1000000 --save-renders 0"
run() {   # name, description, extra args...
  local name=$1 desc=$2; shift 2
  local d=$OUT/$name
  if [ ! -f $d/results.json ]; then
    mkdir -p $d/live; echo "$desc" > $d/live/desc.txt
    stamp "train $name"
    $PYW rrf_gsplat/train_rrf.py --out $d $B "$@" > $d/train.log 2>&1
  fi
  echo "$name: $(grep -a 'iterations in' $d/train.log | tail -1 | cut -c1-160)" >> $LOG
  [ -f ${d}_trainfit.json ] || $PYW rrf_gsplat/diag_train_fit.py --run $d --train-subset 96 > $d/trainfit.log 2>&1
  grep -a -E "^train" $d/trainfit.log >> $LOG
}
run base "主峰方向 loss 冒烟:只监测(权重 0)" --dir-loss expgain --dir-weight 0
run eg "主峰方向 loss 冒烟:期望增益,T 20→1 dB" --dir-loss expgain --dir-weight 0.1 --dir-temp-start-db 20 --dir-temp-db 1 --dir-anneal-steps 1500 --dir-start 200
run ce "主峰方向 loss 冒烟:交叉熵,T 1 dB" --dir-loss ce --dir-weight 0.001 --dir-temp-db 1 --dir-start 200
$PYW - >> $LOG 2>&1 <<'EOF'
import json, math, re
O = "output/rrf/dirloss_smoke"
r = {n: json.load(open(f"{O}/{n}/results.json")) for n in ("base", "eg", "ce")}
fit = {}
for n in r:
    t = open(f"{O}/{n}/trainfit.log", encoding="utf-8", errors="replace").read()
    rm = re.search(r"RMSE ([\d.]+) dB", t); dm = re.search(r"distinct main peak only \((\d+) of (\d+) views\): median ([\d.]+) deg, <= 1 deg ([\d.]+) %", t)
    fit[n] = {"rmse": float(rm.group(1)), "distinct": float(dm.group(4)), "angle": float(dm.group(3)), "nd": f"{dm.group(1)}/{dm.group(2)}"}
mon = {n: r[n]["dir_monitor_db"] for n in r}
s1 = all(all(math.isfinite(v) for _, v in mon[n]) for n in r)
last = {n: sum(v for _, v in mon[n][-3:]) / 3 for n in r}
s2 = all(last[n] <= last["base"] - 0.5 for n in ("eg", "ce"))
s3 = all(fit[n]["rmse"] <= fit["base"]["rmse"] + 1.0 for n in ("eg", "ce"))
for n in r:
    print(f"  {n:4s} monitor last 300 steps {last[n]:6.2f} dB (first window {mon[n][0][1]:.2f}); training views RMSE {fit[n]['rmse']:.2f} dB, "
          f"distinct <= 1 deg {fit[n]['distinct']:.1f} % ({fit[n]['nd']} distinct), median {fit[n]['angle']:.2f} deg; {r[n]['iters_per_second']:.1f} it/s")
print(f"  S1 finite {'PASS' if s1 else 'FAIL'}; S2 liveness (eg / ce >= 0.5 dB under base) {'PASS' if s2 else 'FAIL'}; "
      f"S3 pixel fit (+1.0 dB) {'PASS' if s3 else 'FAIL'}")
json.dump({"monitor_last_db": last, "fit": fit, "pass": {"S1": s1, "S2": s2, "S3": s3}}, open(f"{O}/summary.json", "w"), indent=1)
EOF
stamp "all done"
