#!/bin/bash
###############################################################################
# ARIES — XGBoost Triage Training (Interactive)
#
# Usage:
#   ./slurm/interactive/train_xgboost.sh
###############################################################################

set -euo pipefail

echo "=============================================="
echo "ARIES XGBoost Triage Training (Interactive)"
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "=============================================="

# ── Environment ───────────────────────────────────────────────────────
module purge || true
module load Mamba || true

PROJECT_DIR="${VSC_DATA_VO_USER}/Aries_SOAR"
ENV_PATH="${PROJECT_DIR}/envs/aries"
MAMBA_ROOT=$(dirname $(dirname $(which mamba 2>/dev/null || echo "")))
if [ -n "${MAMBA_ROOT}" ] && [ -d "${MAMBA_ROOT}" ]; then
    source "${MAMBA_ROOT}/etc/profile.d/conda.sh" || true
    source "${MAMBA_ROOT}/etc/profile.d/mamba.sh" || true
fi
mamba activate "${ENV_PATH}" 2>/dev/null || source activate "${ENV_PATH}" || true

export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
export ARIES_HPC_PROJECT_DIR="${PROJECT_DIR}"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# ── Verify GPU ────────────────────────────────────────────────────────
echo ""
python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"
echo ""

# ── Step 1: Preprocess GUIDE (skip if processed data already exists) ─
TRIAGE_NPZ="${PROJECT_DIR}/data/processed/triage_data.npz"
if [ -f "${TRIAGE_NPZ}" ]; then
    echo "Step 1: Skipping preprocessing — ${TRIAGE_NPZ} already exists."
else
    echo "Step 1: Preprocessing GUIDE dataset..."
    python -m src.triage.run_preprocessing
fi

# ── Step 2: Train XGBoost (GPU-accelerated) ──────────────────────────
echo ""
echo "Step 2: Training XGBoost with GPU acceleration..."
python -m src.triage.run_training \
    --tree-method hist \
    --device cuda \
    --depth 8 \
    --rounds 3000 \
    --early-stop 50 \
    --lr 0.05

echo ""
echo "=============================================="
echo "XGBoost training complete!"
echo "=============================================="