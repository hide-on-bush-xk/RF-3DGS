#!/bin/bash
# Validation-set comparison (docs/cluster_log.md §7): one arm, one seed, full training set, early stopping on the
# validation subset, all 480 validation renders saved, then mvdr_peaks.py on val / val_random / val_segment.
#     sbatch -J rf-val-<ARM>-s<SEED> rrf_gsplat/cluster/val_job.sh <plain|G1> <SEED> [full|smoke]   (from the repo root)
# Uses the shared rrf_gsplat/train_rrf.py (G1 needs only --emitters). smoke = 3000 steps, early stopping exercised.
#SBATCH -A ai_wireless_hu_lab
#SBATCH -p a30_normal_q
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=16
#SBATCH --gres=gpu:1 --gres-flags=enforce-binding
#SBATCH -t 4:00:00
#SBATCH -o output/cluster/logs/%x-%j.out
set -u
ARM=${1:?arm}; SEED=${2:?seed}; MODE=${3:-full}
cd "${SLURM_SUBMIT_DIR:-.}"
[ -f rrf_gsplat/train_rrf.py ] || { echo "submit from the repository root"; exit 1; }
PY=$HOME/envs/rf-gsplat/bin/python
APU=RF-3DGS_dataset/regenerated/3dgs_APS_60_gp100
PROT=rrf_gsplat/protocol_v1
case $ARM in plain) X="" ;; G1) X="--emitters output/cluster/ladder/step3/prep/emit_geo.npz --emitter-scale 0.25" ;; *) echo "arm $ARM"; exit 1 ;; esac
case $MODE in
  full)  OUT=output/rrf/val_falcon; S="--visits-per-view 250 --early-stop-on val --early-stop-patience 10 --early-stop-min-steps 5000" ;;
  smoke) OUT=output/rrf/val_falcon_smoke; S="--iterations 3000 --early-stop-on val --early-stop-patience 1 --early-stop-min-steps 1000" ;;
  *) echo "mode $MODE"; exit 1 ;;
esac
d=$OUT/${ARM}_s$SEED
export OMP_NUM_THREADS=16
echo "== $ARM seed $SEED $MODE start $(date '+%F %T') on $(hostname), job ${SLURM_JOB_ID:-none}"
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used --format=csv,noheader
mkdir -p $d/live
t0=$(date +%s.%N)
if [ ! -f $d/results.json ]; then
  $PY rrf_gsplat/train_rrf.py --out $d --mode power --source $APU --protocol $PROT --eval-set val --sh-backend gsplat \
    --eval-group --faces-per-step 4 --lr-scale 2 --sh-degree 3 --seed $SEED --eval-every 1000000 --save-renders -1 \
    $S $X > $d/train.log 2>&1 || { echo "train failed"; tail -25 $d/train.log; exit 1; }
fi
t1=$(date +%s.%N)
for es in val val_random val_segment; do
  [ -f $OUT/peaks_${ARM}_s${SEED}_$es.json ] || $PY rrf_gsplat/mvdr_peaks.py --run $d --truth $APU --protocol $PROT --eval-set $es \
    --out $OUT/peaks_${ARM}_s${SEED}_$es.json > $d/peaks_$es.log 2>&1 || { echo "mvdr_peaks $es failed"; tail -10 $d/peaks_$es.log; exit 1; }
  tail -1 $d/peaks_$es.log
done
t2=$(date +%s.%N)
awk -v a=$t0 -v b=$t1 -v c=$t2 'BEGIN { printf "{\"train_rrf_s\": %.1f, \"peaks_s\": %.1f}\n", b-a, c-b }' > $d/wall.json
echo "$ARM s$SEED: $(grep -a 'iterations in' $d/train.log | tail -1 | cut -c1-160)"
$PY - "$d" "$OUT/peaks_${ARM}_s${SEED}_val.json" <<'PYEOF'
import json, sys
d, pk = sys.argv[1], sys.argv[2]
r = json.load(open(f"{d}/results.json"))
rows = [json.loads(l) for l in open(f"{d}/live.jsonl") if '"comm"' in l]
val_rows = [x for x in rows if x.get("set", "val") == "val" and x.get("comm")]
p = json.load(open(pk))
print(f"  check: iterations {r.get('iterations_run')}, stopped early {r.get('stopped_early')}, validation comm rows {len(val_rows)}, views scored {p['views']}")
sys.exit(0 if len(val_rows) >= 2 and p["views"] == 480 else 1)
PYEOF
echo "== done $(date '+%F %T')"
