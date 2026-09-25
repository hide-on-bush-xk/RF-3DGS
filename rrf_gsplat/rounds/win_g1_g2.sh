#!/bin/bash
# The generalisation diagnostics and P7 on the full training set (docs/stage2_notes.md "协议 v1" §13; Ke, 2026-09-24
# night: "冒烟过了就接着跑全量", the GPU shared with the URA_w_sionna runs -- timings void, scores unaffected).
# Order, each step gated by the one before; a failed smoke stops the chain there:
#   0. the emitter set of G2: path_emitters.py --sets train val (the 120 validation positions solved and cached)
#   1. G1 (g1_coverage.py; no training): do the training emitters cover the validation positions' sources?
#      Its controls (c1-c3) do not gate G2 (a measurement, not a pipeline G2 uses); reported either way.
#   2. the G2 smoke (gates A and G2) and the P7 smoke (gates P7)
#   3. P7 and A, 3 seeds each, interleaved (P7 s0, A s0, P7 s1, ...) so a short night still pairs them
#   4. G2, 3 seeds (after G1: confirmation only, see below)
#   5. plain: one seed (the control; Ke: "普通组一个")
# Order changed after G1 ran (before any training): the validation positions' main-peak sources are already within
# 10 cm of a training emitter on 100 % of the distinct views (5 cm: 98.9 %; control 100 %, shifted-1 m control 5.3 %),
# and G2's emitter set is only 5 % larger (102,530 against 97,282). G1's reading: placement is not what step 1 lacked,
# so G2 is expected close to A. The informative runs (P7, A) go first; G2 still runs, as the confirmation.
# Every full run: the unclipped range (3dgs_APS_60_gp100), the full protocol training set (467 positions, 1868 views),
# power, SH3, at most 250 visits per view (117,000 steps, step 1's budget), early stopping on the validation subset
# (120 of the 480 validation views, every 1000 steps: stop after 10 evaluations without distinct <= 1 deg rising
# 2 points or the beam-gain loss falling 0.2 dB; at least 5000 steps). Early stopping on validation views is model
# selection on the validation set: the same rule for every arm, optimistic against the look-ups (which select nothing).
#   G2     emitters from the training AND validation positions' paths + --em-pcolor 8 (the switch still trained on
#          training views only): a diagnostic, never a result -- it uses the validation positions' ray tracing
#   A      step 1's arm: emitters from the training positions only + --em-pcolor 8 (re-run under the same stopping)
#   P7     --pcolor 8, no emitters (the representation that helped capacity: 53.2 % against plain 42.1 %)
#   plain  no emitters, SH3
# Scored like step 1: mvdr_peaks.py on val / val_random / val_segment, t4_score.py (beam-gain loss on the true channel)
# with the look-ups on the same views. The look-ups (s2_aps_*): NN distinct <= 1 deg 42 %, beam-gain loss 1.36 dB;
# IDW-2 42 % / 1.17 dB; IDW-8 38 % / 1.09 dB.
#
# Smoke criteria, written before the run:
#   G2 smoke  the builder's peaks check >= 85 % of distinct views (index / frame reported, as accepted before);
#             check_emitters.py --em-pcolor 8 on the new set: S1, S2, S3, S5 pass; a 3000-step run with early stopping
#             exercised (min 1000 steps, patience 1) finishes, logs validation comm rows, and all 480 validation views
#             are scored by mvdr_peaks.py
#   P7 smoke  the same 3000-step run with --pcolor 8: finishes, comm rows logged, 480 views scored
# Reading, written before the run (3-seed means; the primary set is the whole validation set):
#   G2 beats the best look-up on BOTH distinct <= 1 deg (by > 3 points) and the beam-gain loss median -> with the right
#      placement the switch generalises: the bottleneck is placement (step 2: predict placement without the oracle)
#   G2 beats it on NEITHER -> even the right placement does not generalise: the bottleneck is how brightness varies with
#      position (representation / loss: P7, the peak-direction loss)
#   G2 - A: the validation positions' own sources help if the mean difference exceeds both arms' seed ranges
#   P7: against the best look-up by the same rule; against plain (one seed) reported only
set -u
REPO=/c/Users/Ke/Documents/GitHub/RF-3DGS
PYW=/c/Users/Ke/miniconda3/envs/rf-gsplat-win/python.exe
PYS=/c/Users/Ke/miniconda3/envs/rf-sionna-win/python.exe
AP=RF-3DGS_dataset/regenerated/3dgs_APS_60
APU=RF-3DGS_dataset/regenerated/3dgs_APS_60_gp100
PROT=rrf_gsplat/protocol_v1
CACHE=output/rrf/m3/path_cache
EM_TR=output/rrf/m3/emitters_train467.npz
EM_ALL=output/rrf/m3/emitters_trainval587.npz
COV=output/rrf/t4_rt_cov_val_all.npz
OUT=output/rrf/m3/gen
export PYTHONUTF8=1
LOG=$REPO/output/rrf/m3/win_g1_g2.log
cd $REPO; mkdir -p $OUT
stamp() { echo "== $1 $(date +%H:%M:%S) gpu $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader | tr -d '\r')" >> $LOG; }
stamp "start"

# ---- 0. G2's emitter set ----
if [ ! -f $EM_ALL ]; then
  $PYS sionna_port/path_emitters.py --truth $AP --protocol $PROT --sets train val --cap 8000 --cache $CACHE \
      --out $EM_ALL > output/rrf/m3/emitters_trainval587.log 2>&1
fi
grep -a -E '"emitters"|"seconds"|distinct_within|positions_from_cache|PASS|FAIL' output/rrf/m3/emitters_trainval587.log >> $LOG

# ---- 1. G1 ----
stamp "G1"
[ -f output/rrf/m3/g1_coverage.json ] || \
  $PYS rrf_gsplat/g1_coverage.py --truth $APU --protocol $PROT --cache $CACHE --emitters $EM_TR --out output/rrf/m3/g1_coverage.json >> $LOG 2>&1

B="--mode power --source $APU --protocol $PROT --eval-set val --sh-backend gsplat --eval-group --faces-per-step 4 \
--lr-scale 2 --sh-degree 3 --eval-every 1000000 --save-renders -1"
FULL="--visits-per-view 250 --early-stop-on val --early-stop-patience 10 --early-stop-min-steps 5000"
SMOKE="--iterations 3000 --early-stop-on val --early-stop-patience 1 --early-stop-min-steps 1000"
run() {   # name, description, extra args...
  local name=$1 desc=$2; shift 2
  local d=$OUT/$name
  if [ ! -f $d/results.json ]; then
    mkdir -p $d/live; echo "$desc" > $d/live/desc.txt
    stamp "train $name"
    $PYW rrf_gsplat/train_rrf.py --out $d $B "$@" > $d/train.log 2>&1
  fi
  echo "$name: $(grep -a 'iterations in' $d/train.log | tail -1 | cut -c1-160)" >> $LOG
  for es in val val_random val_segment; do
    [ -f $OUT/peaks_${name}_$es.json ] || $PYS rrf_gsplat/mvdr_peaks.py --run $d --truth $APU --protocol $PROT --eval-set $es \
        --out $OUT/peaks_${name}_$es.json >> $LOG 2>&1
  done
}
smoke_ok() {   # run name -> exit 0 if it finished, logged validation comm rows and scored all 480 views
  $PYW - "$OUT/$1" "$OUT/peaks_$1_val.json" <<'EOF'
import json, sys
d, pk = sys.argv[1], sys.argv[2]
ok = True
try:
    r = json.load(open(f"{d}/results.json"))
    rows = [json.loads(l) for l in open(f"{d}/live.jsonl") if '"comm"' in l]
    val_rows = [x for x in rows if x.get("set", "val") == "val" and x.get("comm")]
    p = json.load(open(pk))
    n = p.get("views") or p.get("n_views") or len(p.get("per_view", []))
    print(f"  smoke {d}: iterations {r.get('iterations_run')}, stopped early {r.get('stopped_early')}, validation comm rows {len(val_rows)}, views scored {n}")
    ok = len(val_rows) >= 2 and n == 480
except Exception as e:
    print(f"  smoke {d}: check failed: {type(e).__name__}: {e}"); ok = False
sys.exit(0 if ok else 1)
EOF
}

# ---- 2. G2 smoke, then G2 and A ----
stamp "G2 smoke"
G2_OK=1
$PYS - >> $LOG 2>&1 <<'EOF' || G2_OK=0
import json, sys
r = json.load(open("output/rrf/m3/emitters_trainval587.json"))
d = r["peaks"]["distinct_within_1deg"]
print(f"  builder: {r['emitters']:,} emitters from {len(r['positions'])} positions ({r['positions_from_cache']} from the cache); peaks {d:.1%} of distinct views (>= 85 %)")
sys.exit(0 if d is not None and d >= 0.85 else 1)
EOF
if [ ! -f $OUT/check_emitters_trainval587_pc8.json ]; then
  $PYW rrf_gsplat/check_emitters.py --emitters $EM_ALL --names output/rrf/m3/smoke_names.txt --em-pcolor 8 \
      --out $OUT/check_emitters_trainval587_pc8.json > $OUT/check_emitters_trainval587_pc8.log 2>&1
fi
tail -1 $OUT/check_emitters_trainval587_pc8.log >> $LOG
grep -q "ALL PASS" $OUT/check_emitters_trainval587_pc8.log || G2_OK=0
run smoke_g2 "G2 冒烟:训练+验证位置的发射点 + 开关,3000 步" $SMOKE --seed 0 --emitters $EM_ALL --em-pcolor 8
smoke_ok smoke_g2 >> $LOG 2>&1 || G2_OK=0
echo "G2 smoke: $([ $G2_OK = 1 ] && echo PASS || echo 'FAIL -- no emitter runs (A, G2) will start')" >> $LOG
stamp "P7 smoke"
P7_OK=1
run smoke_p7 "P7 冒烟:全部训练位置,3000 步" $SMOKE --seed 0 --pcolor 8
smoke_ok smoke_p7 >> $LOG 2>&1 || P7_OK=0
echo "P7 smoke: $([ $P7_OK = 1 ] && echo PASS || echo 'FAIL -- no P7 runs will start')" >> $LOG
if [ $G2_OK = 0 ] && [ $P7_OK = 0 ]; then stamp "both smokes failed; stopped"; exit 1; fi

# ---- 3. P7 and A, interleaved ----
for s in 0 1 2; do
  [ $P7_OK = 1 ] && run p7_s$s "P7:随接收位置变化的颜色(种子 $s),全部训练位置,按验证子集早停" $FULL --seed $s --pcolor 8
  [ $G2_OK = 1 ] && run a_s$s "A:只用训练位置的发射点 + 开关(种子 $s),按验证子集早停" $FULL --seed $s --emitters $EM_TR --em-pcolor 8
done

# ---- 4. G2 (confirmation) ----
if [ $G2_OK = 1 ]; then
  for s in 0 1 2; do
    run g2_s$s "G2(诊断):训练+验证位置的发射点 + 开关(种子 $s),按验证子集早停" $FULL --seed $s --emitters $EM_ALL --em-pcolor 8
  done
fi

# ---- 5. plain, one seed ----
run plain_s0 "普通场 SH3(种子 0,对照),全部训练位置,按验证子集早停" $FULL --seed 0

# ---- scores with the look-ups ----
stamp "t4 scores"
RUNS=$(ls -d $OUT/*_s[0-9] 2>/dev/null | tr '\n' ' ')
$PYS sionna_port/t4_score.py --truth $APU --runs $RUNS output/rrf/s2_aps_nn output/rrf/s2_aps_idw2 output/rrf/s2_aps_idw8 \
    output/rrf/s2_aps_const --protocol $PROT --eval-set val --cov $COV --out $OUT/t4_scores_val.json >> $LOG 2>&1
stamp "all done"
