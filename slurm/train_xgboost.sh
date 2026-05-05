#!/bin/bash
###############################################################################
# ARIES — XGBoost Triage Training (GPU)
#
# Trains the XGBoost alert-triage classifier on the GUIDE dataset.
# Uses device='cuda' with tree_method='hist' for GPU-accelerated training (XGBoost 2.0+).
#
# Usage:
#   sbatch slurm/train_xgboost.sh
#   sbatch --partition=pascal_gpu slurm/train_xgboost.sh
###############################################################################

#SBATCH --job-name=aries_xgboost
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --partition=ampere_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=04:00:00

set -euo pipefail

echo "=============================================="
echo "ARIES XGBoost Triage Training"
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Partition: ${SLURM_JOB_PARTITION}"
echo "=============================================="

# ── Environment ───────────────────────────────────────────────────────
module purge
module load Mamba

PROJECT_DIR="${VSC_DATA_VO_USER}/Aries_SOAR"
ENV_PATH="${PROJECT_DIR}/envs/aries"
MAMBA_ROOT=$(dirname $(dirname $(which mamba)))
source "${MAMBA_ROOT}/etc/profile.d/conda.sh"
source "${MAMBA_ROOT}/etc/profile.d/mamba.sh"
mamba activate "${ENV_PATH}"

export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
export ARIES_HPC_PROJECT_DIR="${PROJECT_DIR}"
export CUDA_VISIBLE_DEVICES=0

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

## already available since was run before, error came in Step 2
# ── Step 1: Preprocess GUIDE (skip if processed data already exists) ─
TRIAGE_NPZ="${PROJECT_DIR}/data/processed/triage_data.npz"
if [ -f "${TRIAGE_NPZ}" ]; then
    echo "Step 1: Skipping preprocessing — ${TRIAGE_NPZ} already exists."
else
    echo "Step 1: Preprocessing GUIDE dataset..."
    srun python -m src.triage.run_preprocessing
fi

# ── Step 2: Train XGBoost (GPU-accelerated) ──────────────────────────
echo ""
echo "Step 2: Training XGBoost with GPU acceleration..."
srun python -m src.triage.run_training \
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
