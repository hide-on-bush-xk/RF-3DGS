#!/bin/bash
# The Python environment for RF-3DGS on VT ARC Falcon (docs/cluster_handover.md §3), built on an A30 node: gsplat's
# CUDA extension compiles at pip-install time. Submit from the repository root:
#     mkdir -p output/cluster/logs && sbatch rrf_gsplat/cluster/setup_env.sh
# Choices (2026-09-25):
#   conda   Miniforge3/25.11.0-1, conda-forge only (Ke: no Anaconda defaults channel)
#   torch   2.9.1 as on the Windows PC, the cu128 wheel; CUDA/12.8.0 for nvcc, host compiler the system GCC 11.5.
#           Not cu126 / CUDA 12.6 (the handover's guess): gsplat 28e794ca calls ::cuda::ceil_div, which CUDA 12.6's
#           CCCL lacks (job 599529 failed there); 12.8's has it. torch 2.9.1 ships no cu129 wheel.
#   gsplat  the official 28e794ca (1.6.0), cloned with its submodules to ~/src/gsplat on the login node; its default
#           channel list (Config.h) equals train_rrf.GSPLAT_CHANNELS, so NUM_CHANNELS is left unset
#   arch    8.0 (A30, A100) and 8.9 (L40S), plus PTX for newer GPUs
# Re-running is safe: an existing env is reused, pip skips what is installed.
#SBATCH -J rf-setup
#SBATCH -A ai_wireless_hu_lab
#SBATCH -p a30_normal_q
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=16
#SBATCH --gres=gpu:1 --gres-flags=enforce-binding
#SBATCH -t 3:00:00
#SBATCH -o output/cluster/logs/%x-%j.out
set -euo pipefail
ENV=$HOME/envs/rf-gsplat
GSPLAT_SRC=$HOME/src/gsplat
GSPLAT_COMMIT=28e794ca44a4c25ffc39175370c5ee7b38bfcc36

echo "== $(date '+%F %T') on $(hostname), job ${SLURM_JOB_ID:-none}"
module load Miniforge3/25.11.0-1 CUDA/12.8.0
eval "$(conda shell.bash hook)"
[ -d "$ENV" ] || conda create -y -p "$ENV" --override-channels -c conda-forge python=3.11
conda activate "$ENV"
python -m pip install --upgrade pip
pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
python -c "import torch; print('torch', torch.__version__, torch.version.cuda, torch.cuda.is_available())"
tv=$(python -c "import torch; print(torch.version.cuda)"); nv=$(nvcc --version | grep -oE 'release [0-9]+\.[0-9]+' | cut -d' ' -f2)
[ "$tv" = "$nv" ] || { echo "torch is built for CUDA $tv but nvcc is $nv"; exit 1; }
pip install numpy scipy pillow matplotlib ninja psutil

[ "$(git -C "$GSPLAT_SRC" rev-parse HEAD)" = "$GSPLAT_COMMIT" ] || { echo "gsplat source is not at $GSPLAT_COMMIT"; exit 1; }
echo "== $(date '+%T') building gsplat"
cd "$GSPLAT_SRC"
MAX_JOBS=$SLURM_CPUS_PER_TASK TORCH_CUDA_ARCH_LIST="8.0;8.9+PTX" pip install --no-build-isolation .
cd - > /dev/null
echo "== $(date '+%T') built"

echo "== versions"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
nvcc --version | tail -2 | head -1
gcc --version | head -1
# The extension must load and run, not only import: one 5-channel render (5 is in the compiled list) and its backward.
python - <<'EOF'
import matplotlib, numpy, PIL, scipy, torch
import gsplat
from gsplat import rasterization
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available(), torch.cuda.get_device_name(0))
print("gsplat", gsplat.__version__, "numpy", numpy.__version__, "scipy", scipy.__version__, "pillow", PIL.__version__,
      "matplotlib", matplotlib.__version__)
dev = "cuda"
torch.manual_seed(0)
n = 2000
means = (torch.rand(n, 3, device=dev) - 0.5) * 2 + torch.tensor([0.0, 0.0, 4.0], device=dev)
quats = torch.randn(n, 4, device=dev)
scales = torch.full((n, 3), 0.05, device=dev)
opacities = torch.full((n,), 0.8, device=dev)
colors = torch.rand(n, 5, device=dev, requires_grad=True)
viewmats = torch.eye(4, device=dev)[None]
K = torch.tensor([[150.0, 0.0, 150.0], [0.0, 150.0, 100.0], [0.0, 0.0, 1.0]], device=dev)[None]
img, alpha, _ = rasterization(means, quats, scales, opacities, colors, viewmats, K, 300, 200)
img.sum().backward()
ok = bool(torch.isfinite(img).all()) and float(img.abs().max()) > 0 and float(colors.grad.abs().sum()) > 0
print(f"render {tuple(img.shape)} max {float(img.abs().max()):.4f}, grad |sum| {float(colors.grad.abs().sum()):.1f} -> {'OK' if ok else 'FAIL'}")
raise SystemExit(0 if ok else 1)
EOF
echo "== $(date '+%F %T') done"
