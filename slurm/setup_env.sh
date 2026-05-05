#!/bin/bash
###############################################################################
# ARIES — Environment Setup for VUB Hydra HPC (GPU-aware)
#
# Creates the conda environment with all dependencies needed for training
# all ARIES pipelines.
#
# Usage:
#   sbatch slurm/setup_env.sh
###############################################################################

#SBATCH --job-name=aries_setup
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --partition=ampere_gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00

set -euo pipefail

echo "=============================================="
echo "ARIES Environment Setup"
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "=============================================="

# ── 1. Clean environment and load Mamba ───────────────────────────────
module purge
module load Mamba

# ── 2. Paths ──────────────────────────────────────────────────────────
PROJECT_DIR="${VSC_DATA_VO_USER}/Aries_SOAR"
ENV_PATH="${PROJECT_DIR}/envs/aries"

# Create all required directories
mkdir -p "${PROJECT_DIR}/envs"
mkdir -p "${PROJECT_DIR}/mlruns"
mkdir -p "${PROJECT_DIR}/data/processed"
mkdir -p "${PROJECT_DIR}/checkpoints"/{triage,ner,bart}
mkdir -p "${PROJECT_DIR}/models"/{triage,ner,summarizer,onnx,slm_lora,slm_merged}

# ── 3. Create / recreate conda environment ────────────────────────────
MAMBA_ROOT=$(dirname $(dirname $(which mamba)))
source "${MAMBA_ROOT}/etc/profile.d/conda.sh"
source "${MAMBA_ROOT}/etc/profile.d/mamba.sh"

# Fix for Mamba lockfile error on read-only system caches
# Override the system .condarc so libmamba never tries to lock the read-only
# system package cache (/vscmnt/.../Mamba/.../pkgs/cache/cache.lock).
export CONDA_PKGS_DIRS="${PROJECT_DIR}/envs/conda_pkgs"
mkdir -p "${CONDA_PKGS_DIRS}"

export CONDARC="${PROJECT_DIR}/envs/.condarc"
cat > "${CONDARC}" << EOF
pkgs_dirs:
  - ${PROJECT_DIR}/envs/conda_pkgs
envs_dirs:
  - ${PROJECT_DIR}/envs
channels:
  - conda-forge
EOF

if [ ! -d "${ENV_PATH}" ]; then
    echo "Environment NOT found at ${ENV_PATH}. Creating now..."
    mamba create -p "${ENV_PATH}" python=3.10 -y
else
    echo "Environment already exists at ${ENV_PATH}. Recreating..."
    rm -rf "${ENV_PATH}"
    mamba create -p "${ENV_PATH}" python=3.10 -y
fi

# ── 4. Activate ──────────────────────────────────────────────────────
echo "Activating environment..."
mamba activate "${ENV_PATH}"

PYTHON_BIN="${ENV_PATH}/bin/python"
if [ ! -x "${PYTHON_BIN}" ]; then
    echo "ERROR: expected python not found at ${PYTHON_BIN}"
    exit 1
fi

echo "Using Python: $(${PYTHON_BIN} -c 'import sys; print(sys.executable)')"

echo "Upgrading pip, setuptools, wheel..."
"${PYTHON_BIN}" -m pip install --upgrade pip setuptools wheel

# ── 5. PyTorch + CUDA 12.1 ───────────────────────────────────────────
# torchvision and torchaudio are intentionally excluded: ARIES is NLP-only.
# Using cu124 wheel: vLLM installs nvidia-* CUDA 12.4/12.8 Python packages
# which conflict with cu121 bundled libs (libnvJitLink mismatch). cu124 is
# ABI-compatible with the nvidia-* packages vLLM pulls in.
echo "Installing PyTorch with CUDA 12.4 support..."
"${PYTHON_BIN}" -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124

# ── 6. ML frameworks ─────────────────────────────────────────────────
echo "Installing ML frameworks..."
"${PYTHON_BIN}" -m pip install \
    xgboost>=2.0.0 \
    scikit-learn>=1.3.0 \
    category-encoders>=2.6.0

# ── 7. HuggingFace ecosystem ─────────────────────────────────────────
echo "Installing HuggingFace stack..."
"${PYTHON_BIN}" -m pip install \
    transformers>=4.35.0 \
    accelerate>=0.25.0 \
    datasets>=2.14.0 \
    tokenizers>=0.15.0

# ── 8. ONNX ──────────────────────────────────────────────────────────
echo "Installing ONNX..."
"${PYTHON_BIN}" -m pip install \
    onnx>=1.15.0 \
    onnxruntime-gpu>=1.16.0 \
    onnxmltools>=1.12.0 \
    skl2onnx>=1.16.0

# ── 8b. SLM & Generative AI ──────────────────────────────────────────
echo "Installing SLM and GenAI frameworks..."
export VLLM_INSTALL_PUNICA_KERNELS=0
"${PYTHON_BIN}" -m pip install \
    vllm>=0.4.0 \
    llama-cpp-python>=0.2.50 \
    onnxruntime-genai>=0.2.0 \
    bitsandbytes>=0.41.0 \
    peft>=0.7.0 \
    trl>=0.7.10 \
    sentencepiece>=0.1.99

# ── 8c. Re-pin PyTorch CUDA build ────────────────────────────────────
# vLLM resolves its own torch dependency and overwrites torch with a
# potentially different build. Force-reinstall the exact CUDA 12.4 wheel
# AFTER vLLM so torch.cuda.is_available() is always True on GPU nodes.
echo "Re-pinning PyTorch CUDA 12.4 build after vLLM install..."
"${PYTHON_BIN}" -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124 --force-reinstall --no-deps

# vLLM pulls in nvidia-cusparse compiled against nvjitlink 12.8; ensure the
# matching nvjitlink is present so libcusparse.so.12 resolves its symbols.
echo "Pinning nvidia-nvjitlink to >=12.8 to match vLLM's cusparse build..."
"${PYTHON_BIN}" -m pip install "nvidia-nvjitlink-cu12>=12.8.0"

# ── 9. Evaluation metrics ────────────────────────────────────────────
echo "Installing evaluation libraries..."
"${PYTHON_BIN}" -m pip install \
    seqeval>=1.2.2 \
    rouge-score>=0.1.2

# ── 10. Utilities ────────────────────────────────────────────────────
echo "Installing utilities..."
"${PYTHON_BIN}" -m pip install \
    mlflow>=2.9.0 \
    pydantic>=2.5.0 \
    pydantic-settings>=2.1.0 \
    numpy>=1.24.0 \
    pandas>=2.0.0 \
    scipy>=1.11.0 \
    pyarrow>=14.0.0 \
    matplotlib>=3.8.0 \
    seaborn>=0.13.0 \
    tqdm>=4.66.0 \
    python-dotenv>=1.0.0

# ── 11. Verification ─────────────────────────────────────────────────
echo ""
echo "=============================================="
echo "Verifying installation..."
echo "=============================================="

# Prepend conda env nvidia libs so they shadow any system CUDA module libs
# (e.g. CUDA/12.1.1 module puts older libnvJitLink.so.12 first in LD_LIBRARY_PATH).
NVIDIA_LIBS="${ENV_PATH}/lib/python3.10/site-packages/nvidia"
export LD_LIBRARY_PATH="${NVIDIA_LIBS}/nvjitlink/lib:${NVIDIA_LIBS}/cusparse/lib:${LD_LIBRARY_PATH:-}"

"${PYTHON_BIN}" -c "
import torch
print(f'PyTorch:       {torch.__version__}')
print(f'CUDA build:    {torch.version.cuda}')
print(f'CUDA runtime:  {torch.cuda.is_available()}')
"

"${PYTHON_BIN}" -c "import xgboost; print(f'XGBoost:       {xgboost.__version__}')"
"${PYTHON_BIN}" -c "import transformers; print(f'Transformers:  {transformers.__version__}')"
"${PYTHON_BIN}" -c "import datasets; print(f'Datasets:      {datasets.__version__}')"
"${PYTHON_BIN}" -c "import onnx; print(f'ONNX:          {onnx.__version__}')"
"${PYTHON_BIN}" -c "import mlflow; print(f'MLflow:        {mlflow.__version__}')"
"${PYTHON_BIN}" -c "import seqeval; print(f'seqeval:       OK')"
"${PYTHON_BIN}" -c "import rouge_score; print(f'rouge-score:   OK')"
"${PYTHON_BIN}" -c "import category_encoders; print(f'category-enc:  OK')"
"${PYTHON_BIN}" -c "import pydantic_settings; print(f'pydantic-set:  OK')"

echo ""
echo "=============================================="
echo "Environment setup complete!"
echo "Location: ${ENV_PATH}"
echo "=============================================="
