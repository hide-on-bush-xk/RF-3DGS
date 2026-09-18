#!/bin/bash
# Round 4 (WSL): geometry and densification on the dB model, then the
# multi-channel targets. Waits for the multi-channel datasets, which the
# Windows side generates once the E1 rerun frees the GPU.
REPO=/mnt/c/Users/Ke/Documents/GitHub/RF-3DGS
REG=RF-3DGS_dataset/regenerated
RUN="bash $REPO/rrf_gsplat/wsl_run.sh"
cd $REPO
run() { local name=$1; shift
  if [ -f output/rrf/$name/results.json ]; then echo "skip $name"; return; fi
  echo "== $name  $(date +%H:%M:%S)"; $RUN "$name" "$@"; }

for i in $(seq 1 360); do
  [ -f $REG/3dgs_AOD3_24ghz_tut/generation_meta.json ] && [ -f $REG/3dgs_MULTI_24ghz_tut/generation_meta.json ] && break
  sleep 5
done
sleep 20    # let the generator's process exit and release its memory
echo "== datasets present $(date +%H:%M:%S)"

# geometry: unfrozen without densification, MCMC at the checkpoint's count, MCMC allowed to grow
run a_mvdr_db_geom      --source $REG/3dgs_MVDR_100_gpct --mode db --train-geometry
run a_mvdr_db_mcmc      --source $REG/3dgs_MVDR_100_gpct --mode db --densify mcmc
run a_mvdr_db_mcmc_grow --source $REG/3dgs_MVDR_100_gpct --mode db --densify mcmc --cap-max 1300000

# multi-channel: one physical quantity per channel, against the tutorial's
# angle x amplitude RGB encoding, same poses; all test renders kept for the
# decoded-angle comparison
run m_multi_24_tut --source $REG/3dgs_MULTI_24ghz_tut --mode multi --save-renders 640
run m_aod3_24_tut  --source $REG/3dgs_AOD3_24ghz_tut  --mode multi --mask-channel 2 --save-renders 640
echo "round 4 done $(date +%H:%M:%S)"
