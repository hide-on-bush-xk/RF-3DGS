#!/bin/bash
# Round 47: T4 (docs/t4_channel_model_plan.md) and P1's first test (docs/tech_paths.md), full scale, after the smoke.
#   1. t4_baselines.py: control, LOS, NN, 3GPP InH open / mixed x 10 realisations on all 640 held-out views
#   2. t6_incoherent_mvdr.py --positions 0: every held-out position re-solved with the dataset's configuration and
#      seeds; the coherent MVDR recomputed in float64 (the stored complex64 truth is rounding noise in its weak
#      directions), the incoherent MVDR (P1), the ray tracer's LSPs and tap covariances
#   3. t4_score.py against the stored truth and against the float64 truth: the RRF runs (SH3 base, SH3 + head B6),
#      the baselines, and the float64 truth itself (= the stored labels' own noise, as a predictor)
# Criteria and expectations: written in docs/t4_channel_model_plan.md before the smoke.
set -u
REPO=/c/Users/Ke/Documents/GitHub/RF-3DGS
PYS=/c/Users/Ke/miniconda3/envs/rf-sionna-win/python.exe
MV=RF-3DGS_dataset/regenerated/3dgs_MVDR_100_gpct
export PYTHONUTF8=1
LOG=$REPO/output/rrf/win_round47_t4.log
cd $REPO
echo "== round 47 start $(date +%H:%M:%S)" >> $LOG
T0=$(date +%s)
$PYS sionna_port/t4_baselines.py --truth $MV --realisations 10 >> $LOG 2>&1
echo "== baselines done in $(( $(date +%s) - T0 )) s $(date +%H:%M:%S)" >> $LOG
T0=$(date +%s)
$PYS sionna_port/t6_incoherent_mvdr.py --truth $MV --positions 0 > output/rrf/t6_all.log 2>&1
tail -40 output/rrf/t6_all.log >> $LOG
echo "== t6 done in $(( $(date +%s) - T0 )) s $(date +%H:%M:%S)" >> $LOG
RUNS="output/rrf/t4_los output/rrf/t4_nn output/rrf/t4_inh_open_r* output/rrf/t4_inh_mixed_r* output/rrf/r45_base output/rrf/r45_base_s1 output/rrf/r45_base_s2 output/rrf/r43_B6_sh3_head output/rrf/r43_B6_sh3_head_s1 output/rrf/r43_B6_sh3_head_s2"
echo "== scores against the stored truth" >> $LOG
$PYS sionna_port/t4_score.py --truth $MV --runs $RUNS output/rrf/t6_truth64 --cov output/rrf/t4_rt_cov_all.npz \
    --rt-stats output/rrf/t4_rt_stats_all.json --out output/rrf/t4_scores.json >> $LOG 2>&1
echo "== scores against the float64 truth" >> $LOG
$PYS sionna_port/t4_score.py --truth $MV --truth-renders output/rrf/t6_truth64 --runs $RUNS --cov output/rrf/t4_rt_cov_all.npz \
    --rt-stats output/rrf/t4_rt_stats_all.json --out output/rrf/t4_scores_truth64.json >> $LOG 2>&1
echo "== all done $(date +%H:%M:%S)" >> $LOG
