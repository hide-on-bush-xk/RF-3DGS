#!/bin/bash
# Falcon replica of rounds/win_dir_loss_smoke.sh (docs/cluster_handover.md §5; stage2_notes §17), one sbatch job.
# Submit from the repository root:  sbatch rrf_gsplat/cluster/dir_loss_smoke.sh
# Configuration identical to the PC's: 3dgs_APS_60_gp100, protocol_v1, 24 positions (--max-train-views 96), power,
# SH3, --faces-per-step 4, 3000 steps, seed 0, live on; arms
#   base   --dir-loss expgain --dir-weight 0 (monitored only)
#   eg     expgain, weight 0.1, T annealed 20 -> 1 dB over 1500 steps, from step 200
#   ce     ce, weight 0.001, T 1 dB, from step 200
# The one difference (Ke, 2026-09-25: more GPUs): the three arms run at once, each alone on its own A30 of the job
# (CUDA_VISIBLE_DEVICES), 16 cores each; they share the node's host memory bandwidth.
# Criteria, written before the run (cluster_handover.md §5, 2026-09-25):
#   S1  all three finish; the monitor (expgain at T = 1 dB, dB) finite at every logged window
#   S2  the monitor over the last 300 steps of eg and of ce is >= 0.5 dB below base's
#   S3  training-view RMSE (diag_train_fit.py) of eg and ce within +1.0 dB of base's
#   C1  base training-view RMSE within 2.21 +- 0.1 dB (the PC's)
#   C2  each arm's monitor over the last 300 steps within +- 0.3 dB of the PC's 1.84 / 0.83 / 1.29 dB
#   C3  each arm's distinct <= 1 deg within +- 5 points of the PC's 46.5 / 49.3 / 52.1 %
#   Outside a range: report and look for the cause; the ranges are not changed, the full runs do not start.
# Timing (valid: the job owns its GPUs): per arm, set-up + training + final evaluation (train_rrf.py) and the
# training-fit diagnostic, each wall-clock; the step time is the median over live.jsonl's 50-step windows after
# step 200 (warm-up excluded), a step = one position's 4 faces at 300 x 200. Data generation: done on the PC, MISSING.
#SBATCH -J rf-dirloss-smoke
#SBATCH -A ai_wireless_hu_lab
#SBATCH -p a30_normal_q
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=48
#SBATCH --gres=gpu:3 --gres-flags=enforce-binding
#SBATCH -t 1:00:00
#SBATCH -o output/cluster/logs/%x-%j.out
set -u
cd "${SLURM_SUBMIT_DIR:-.}"
[ -f rrf_gsplat/train_rrf.py ] || { echo "submit from the repository root"; exit 1; }
PY=$HOME/envs/rf-gsplat/bin/python
APU=RF-3DGS_dataset/regenerated/3dgs_APS_60_gp100
PROT=rrf_gsplat/protocol_v1
OUT=output/rrf/dirloss_smoke_falcon
export OMP_NUM_THREADS=16
mkdir -p $OUT
echo "== start $(date '+%F %T') on $(hostname), job ${SLURM_JOB_ID:-none}"
nvidia-smi --query-gpu=index,name,uuid,utilization.gpu,memory.used --format=csv
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv
B="--mode power --source $APU --protocol $PROT --eval-set val --sh-backend gsplat --eval-group --faces-per-step 4 \
--lr-scale 2 --sh-degree 3 --seed 0 --max-train-views 96 --iterations 3000 --eval-every 1000000 --save-renders 0"
run() {   # gpu, name, description, extra args...
  local gpu=$1 name=$2 desc=$3; shift 3
  local d=$OUT/$name t0 t1 t2
  mkdir -p $d/live; echo "$desc" > $d/live/desc.txt
  t0=$(date +%s.%N)
  [ -f $d/results.json ] || CUDA_VISIBLE_DEVICES=$gpu $PY rrf_gsplat/train_rrf.py --out $d $B "$@" > $d/train.log 2>&1
  t1=$(date +%s.%N)
  [ -f ${d}_trainfit.json ] || CUDA_VISIBLE_DEVICES=$gpu $PY rrf_gsplat/diag_train_fit.py --run $d --train-subset 96 > $d/trainfit.log 2>&1
  t2=$(date +%s.%N)
  awk -v a=$t0 -v b=$t1 -v c=$t2 'BEGIN { printf "{\"train_rrf_wall_s\": %.2f, \"trainfit_wall_s\": %.2f}\n", b - a, c - b }' > $d/wall.json
  echo "$name done $(date +%T): $(grep -a 'iterations in' $d/train.log | tail -1 | cut -c1-160)"
}
run 0 base "主峰方向 loss 冒烟(Falcon):只监测(权重 0)" --dir-loss expgain --dir-weight 0 &
run 1 eg "主峰方向 loss 冒烟(Falcon):期望增益,T 20→1 dB" --dir-loss expgain --dir-weight 0.1 --dir-temp-start-db 20 --dir-temp-db 1 --dir-anneal-steps 1500 --dir-start 200 &
run 2 ce "主峰方向 loss 冒烟(Falcon):交叉熵,T 1 dB" --dir-loss ce --dir-weight 0.001 --dir-temp-db 1 --dir-start 200 &
wait
$PY - <<'EOF'
import json, math, statistics
O = "output/rrf/dirloss_smoke_falcon"
ARMS = ("base", "eg", "ce")
PC = {"rmse_base": 2.21, "monitor": {"base": 1.84, "eg": 0.83, "ce": 1.29}, "distinct": {"base": 46.5, "eg": 49.3, "ce": 52.1}}
r, fit, wall, step = {}, {}, {}, {}
for n in ARMS:
    try:
        r[n] = json.load(open(f"{O}/{n}/results.json"))
        fit[n] = json.load(open(f"{O}/{n}_trainfit.json"))["train"]
        wall[n] = json.load(open(f"{O}/{n}/wall.json"))
    except (OSError, KeyError, ValueError) as e:
        print(f"  {n}: missing output ({e})")
if len(r) < 3 or len(fit) < 3:
    print("  S1 FAIL: not all three arms finished"); raise SystemExit(1)
for n in ARMS:
    win = [json.loads(l) for l in open(f"{O}/{n}/live.jsonl")]
    it_s = [w["it_s"] for w in win if "it_s" in w and w["it"] > 200]
    step[n] = {"median_ms": 1000 / statistics.median(it_s), "windows": len(it_s)}
mon = {n: r[n]["dir_monitor_db"] for n in ARMS}
last = {n: sum(v for _, v in mon[n][-3:]) / 3 for n in ARMS}
rmse = {n: fit[n]["rmse_db"] for n in ARMS}
dist = {n: 100 * fit[n]["distinct_within_1deg"] for n in ARMS}
ok = {
    "S1": all(all(math.isfinite(v) for _, v in mon[n]) for n in ARMS),
    "S2": all(last[n] <= last["base"] - 0.5 for n in ("eg", "ce")),
    "S3": all(rmse[n] <= rmse["base"] + 1.0 for n in ("eg", "ce")),
    "C1": abs(rmse["base"] - PC["rmse_base"]) <= 0.1,
    "C2": all(abs(last[n] - PC["monitor"][n]) <= 0.3 for n in ARMS),
    "C3": all(abs(dist[n] - PC["distinct"][n]) <= 5 for n in ARMS),
}
print("  arm  | monitor last 300 (PC) | train RMSE dB | distinct<=1deg % (PC)   | angle med | at true peak | it/s  | step ms (median, windows) | train_rrf s | trainfit s")
for n in ARMS:
    print(f"  {n:4s} | {last[n]:5.2f} ({PC['monitor'][n]:.2f})         | {rmse[n]:5.2f}         | {dist[n]:5.1f} ({PC['distinct'][n]:.1f}, {fit[n]['distinct_views']} distinct) | "
          f"{fit[n]['distinct_angle_median']:5.2f} deg | {fit[n]['at_true_median']:+6.2f} dB    | {r[n]['iters_per_second']:5.1f} | "
          f"{step[n]['median_ms']:6.2f} ({step[n]['windows']})             | {wall[n]['train_rrf_wall_s']:7.1f}     | {wall[n]['trainfit_wall_s']:6.1f}")
    print(f"         {r[n]['gpu']}: load {r[n]['load_seconds']:.1f} s + training {r[n]['train_seconds']:.1f} s "
          f"(of which live evaluations {r[n]['running_eval_seconds']:.1f} s) + final evaluation {r[n]['eval_seconds']:.1f} s; "
          f"peak reserved {r[n]['cuda_peak_reserved_mib']:.0f} MiB")
print("  " + "; ".join(f"{k} {'PASS' if v else 'FAIL'}" for k, v in ok.items()))
passed = all(ok.values())
print(f"  SMOKE {'PASS' if passed else 'FAIL'}")
json.dump({"monitor_last_db": last, "rmse_db": rmse, "distinct_pct": dist, "fit": fit, "step": step, "wall": wall,
           "iters_per_second": {n: r[n]["iters_per_second"] for n in ARMS},
           "seconds": {n: {k: r[n][k] for k in ("load_seconds", "train_seconds", "running_eval_seconds", "eval_seconds")}
                       for n in ARMS},
           "gpu": {n: r[n]["gpu"] for n in ARMS}, "pass": ok, "smoke_pass": passed},
          open(f"{O}/summary.json", "w"), indent=1)
raise SystemExit(0 if passed else 1)
EOF
status=$?
echo "== done $(date '+%F %T'), summary exit $status"
exit $status
