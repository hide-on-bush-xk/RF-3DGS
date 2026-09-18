#!/bin/bash
# The stage-2 experiment matrix, run sequentially in WSL; a run whose
# results.json exists is skipped, so this can be re-launched after a failure.
#   bash rrf_gsplat/run_matrix.sh [group ...]     groups: colour norm ablate txmove all
set -u
REPO=/mnt/c/Users/Ke/Documents/GitHub/RF-3DGS
RUN="bash $REPO/rrf_gsplat/wsl_run.sh"
REG=RF-3DGS_dataset/regenerated
GROUPS_WANTED=${*:-all}

want() { [[ "$GROUPS_WANTED" == *all* || "$GROUPS_WANTED" == *$1* ]]; }
run() {   # run <name> <args...>
  local name=$1; shift
  if [ -f "$REPO/output/rrf/$name/results.json" ]; then echo "skip $name"; return; fi
  echo "== $name  $(date +%H:%M:%S)"
  $RUN "$name" "$@"
}

# E2: the colour function. Same data (regenerated MVDR, global percentile
# range), same geometry, same iterations; only what a Gaussian carries and
# how it composites changes.
if want colour; then
  run e2_mvdr_rgb   --source $REG/3dgs_MVDR_100_gpct --mode rgb
  run e2_mvdr_db    --source $REG/3dgs_MVDR_100_gpct --mode db
  run e2_mvdr_power --source $REG/3dgs_MVDR_100_gpct --mode power
fi

# E1: normalisation. Global range against per-view range on the same float
# spectra; the per-view model is scored with an oracle range, which is the
# best it can ever do.
if want norm; then
  run e1_cbf_global   --source $REG/3dgs_CBF_100_gpct    --mode rgb
  run e1_cbf_perview  --source $REG/3dgs_CBF_100_perview --mode rgb
  run e1_mvdr_perview --source $REG/3dgs_MVDR_100_perview --mode rgb
fi

# Ablations on the best colour mode: SH degree, opacity, training views.
if want ablate; then
  run a_mvdr_db_sh0        --source $REG/3dgs_MVDR_100_gpct --mode db --sh-degree 0
  run a_mvdr_db_sh1        --source $REG/3dgs_MVDR_100_gpct --mode db --sh-degree 1
  run a_mvdr_db_freezeop   --source $REG/3dgs_MVDR_100_gpct --mode db --freeze-opacity
  run a_mvdr_db_views640   --source $REG/3dgs_MVDR_100_gpct --mode db --max-train-views 640
  run a_mvdr_db_views160   --source $REG/3dgs_MVDR_100_gpct --mode db --max-train-views 160
  run a_mvdr_db_iters2k    --source $REG/3dgs_MVDR_100_gpct --mode db --iterations 2000
fi

# Tx moved: cold start against warm start from the Tx-A model, on the Tx-B
# dataset (generated with --tx by sionna_port/generate_dataset.py).
if want txmove; then
  if [ -d $REPO/$REG/3dgs_MVDR_txB_gpct ]; then
    run t_txB_cold --source $REG/3dgs_MVDR_txB_gpct --mode db --eval-every 250
    run t_txB_warm --source $REG/3dgs_MVDR_txB_gpct --mode db --eval-every 250 \
                   --init-from $REPO/output/rrf/e2_mvdr_db/rrf_state.pt
    run t_txB_cold_rgb --source $REG/3dgs_MVDR_txB_gpct --mode rgb --eval-every 250
    run t_txB_warm_rgb --source $REG/3dgs_MVDR_txB_gpct --mode rgb --eval-every 250 \
                   --init-from $REPO/output/rrf/e2_mvdr_rgb/rrf_state.pt
  else
    echo "no Tx-B dataset yet"
  fi
fi
echo "matrix done $(date +%H:%M:%S)"
