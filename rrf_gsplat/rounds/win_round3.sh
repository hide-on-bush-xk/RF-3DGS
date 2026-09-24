#!/bin/bash
# Round 3, Windows side:
#   1. the "1.5 minutes per transmitter move" claim measured, not extrapolated:
#      Tx-B with 160 positions only, Tx-A's dB range, then a db fine-tune
#      (cold and warm) evaluated on the same 640 held-out views as before
#   2. a like-for-like dataset for the released data: 2.4 GHz, the tutorial's
#      materials, M = 10 tr38901; rgb + db fine-tunes, PSNR against 16.02
set -u
REPO=/c/Users/Ke/Documents/GitHub/RF-3DGS
PYS=/c/Users/Ke/miniconda3/envs/rf-sionna-win/python.exe
SCENE=../sionna_tutorial/RF-3DGS_Sionna_simulation_tutorial/NIST_lobby_v1.0/NIST_lobby_V1.1_sionna12.xml
RXLOC=../sionna_tutorial/RF-3DGS_Sionna_simulation_tutorial/NIST_rx_loc.txt
REG=RF-3DGS_dataset/regenerated
export PYTHONUTF8=1
LOG=$REPO/output/rrf/win_round3.log
echo "== start $(date)" >> $LOG

# 1. Tx-B, 160 positions. The generator takes the first N positions of the
# route; the held-out set must stay the one used everywhere, so the test
# images are taken from the full Tx-B dataset and only the training list
# is restricted to positions that exist here.
cd $REPO/sionna_port
T0=$(date +%s)
$PYS generate_dataset.py --scene-xml $SCENE --rx-loc-file $RXLOC \
    --out-dir ../$REG/3dgs_MVDR_txB160 --spectrum MVDR --num-positions 160 \
    --tx 8.2 -5.4 2.0 --no-dashboard 2>&1 | grep -a "views/s\|global dB" >> $LOG
T1=$(date +%s); echo "txB160 generation $((T1-T0)) s" >> $LOG
cd $REPO
$PYS rrf_gsplat/renormalize.py $REG/3dgs_MVDR_txB160 $REG/3dgs_MVDR_txB160_gpct \
    --norm global-pct --range -175.75 -101.52 2>&1 | tail -1 >> $LOG
$PYS - <<'EOF' >> $LOG 2>&1
import os, shutil
reg = "RF-3DGS_dataset/regenerated"
src, full = f"{reg}/3dgs_MVDR_txB160_gpct", f"{reg}/3dgs_MVDR_txB_gpct"
# training list: the 160 positions here, minus any that are held out
test = [l.strip() for l in open(f"{full}/test_index.txt") if l.strip()]
have = sorted(f[:-4] for f in os.listdir(f"{src}/images"))
train = [n for n in have if n not in test]
open(f"{src}/train_index.txt", "w").write("\n".join(train) + "\n")
open(f"{src}/test_index.txt", "w").write("\n".join(test) + "\n")
# the held-out views themselves come from the full Tx-B dataset
for n in test:
    for sub, ext in (("images", ".png"), ("spectra_float", ".npy")):
        d = f"{src}/{sub}/{n}{ext}"
        if not os.path.exists(d):
            shutil.copy(f"{full}/{sub}/{n}{ext}", d)
print(f"txB160: {len(train)} train views from {len(have)//4} positions, {len(test)} held-out views copied from the full set")
EOF
echo "== txB160 dataset done $(date)" >> $LOG
MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu-22.04 -u ke -- bash /mnt/c/Users/Ke/Documents/GitHub/RF-3DGS/rrf_gsplat/wsl_run.sh t_txB160_cold \
    --source $REG/3dgs_MVDR_txB160_gpct --mode db --eval-every 250 >> $LOG 2>&1
MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu-22.04 -u ke -- bash /mnt/c/Users/Ke/Documents/GitHub/RF-3DGS/rrf_gsplat/wsl_run.sh t_txB160_warm \
    --source $REG/3dgs_MVDR_txB160_gpct --mode db --eval-every 250 \
    --init-from /mnt/c/Users/Ke/Documents/GitHub/RF-3DGS/output/rrf/e2_mvdr_db/rrf_state.pt >> $LOG 2>&1
echo "== txB160 trainings done $(date)" >> $LOG

# 2. like-for-like with the released data
cd $REPO/sionna_port
$PYS generate_dataset.py --scene-xml $SCENE --rx-loc-file $RXLOC \
    --out-dir ../$REG/3dgs_MVDR_24ghz_tut --spectrum MVDR --num-positions 800 \
    --frequency 2.4e9 --materials tutorial --no-dashboard 2>&1 | grep -a "views/s\|global dB" >> $LOG
cd $REPO
$PYS rrf_gsplat/renormalize.py $REG/3dgs_MVDR_24ghz_tut $REG/3dgs_MVDR_24ghz_tut_gpct --norm global-pct --pct 1 99.99 2>&1 | tail -1 >> $LOG
cp $REG/3dgs_MVDR_100_gpct/train_index.txt $REG/3dgs_MVDR_100_gpct/test_index.txt $REG/3dgs_MVDR_24ghz_tut_gpct/
echo "== 2.4 GHz dataset done $(date)" >> $LOG
MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu-22.04 -u ke -- bash /mnt/c/Users/Ke/Documents/GitHub/RF-3DGS/rrf_gsplat/wsl_run.sh e4_24ghz_tut_rgb --source $REG/3dgs_MVDR_24ghz_tut_gpct --mode rgb >> $LOG 2>&1
MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu-22.04 -u ke -- bash /mnt/c/Users/Ke/Documents/GitHub/RF-3DGS/rrf_gsplat/wsl_run.sh e4_24ghz_tut_db  --source $REG/3dgs_MVDR_24ghz_tut_gpct --mode db  >> $LOG 2>&1
echo "== all done $(date)" >> $LOG
tail -15 $LOG
