#!/bin/bash
###############################################################################
# ARIES — SLM Fine-tuning using QLoRA (Interactive)
###############################################################################
set -euo pipefail

DATASET=${1:-triage}
echo "=============================================="
echo "ARIES SLM QLoRA Training : ${DATASET}"
echo "=============================================="

module purge || true
module load Mamba || true
module load CUDA/12.1.1 || true   # required for bf16 + bitsandbytes 4-bit quantisation

PROJECT_DIR="${VSC_DATA_VO_USER}/Aries_SOAR"
ENV_PATH="${PROJECT_DIR}/envs/aries"
PYTHON_BIN="${ENV_PATH}/bin/python"

if [ ! -x "${PYTHON_BIN}" ]; then
    echo "ERROR: ${PYTHON_BIN} not found. Run setup_env.sh first."
    exit 1
fi

# Prepend the conda env's nvidia libs so they take precedence over the older
# CUDA/12.1.1 module libs (which have an nvjitlink that lacks 12.8 symbols).
NVIDIA_LIBS="${ENV_PATH}/lib/python3.10/site-packages/nvidia"
export LD_LIBRARY_PATH="${NVIDIA_LIBS}/nvjitlink/lib:${NVIDIA_LIBS}/cusparse/lib:${LD_LIBRARY_PATH:-}"

# Verify a GPU is actually allocated to this session.
# If not, the user must request one via srun/salloc first.
GPU_COUNT=$("${PYTHON_BIN}" -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo 0)
if [ "${GPU_COUNT}" -eq 0 ]; then
    echo ""
    echo "ERROR: No GPU visible to PyTorch in this session."
    echo "Request an interactive GPU node first:"
    echo ""
    echo "  srun --partition=ampere_gpu --gres=gpu:1 --nodes=1 --ntasks=1 \\"
    echo "       --cpus-per-task=8 --mem=32G --time=04:00:00 --pty bash"
    echo ""
    echo "Then re-run this script inside that session."
    exit 1
fi
echo "GPU detected: ${GPU_COUNT} device(s) — proceeding with bf16 training."

# Redirect HuggingFace model cache to VSC_DATA (home quota is too small for Phi-3-mini ~3.8GB)
export HF_HOME="${PROJECT_DIR}/hf_cache"
mkdir -p "${HF_HOME}"

export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

# Train
"${PYTHON_BIN}" src/slm/finetune.py --dataset "${DATASET}"

echo "=============================================="
echo "SLM Training complete for ${DATASET}!"
echo "=============================================="