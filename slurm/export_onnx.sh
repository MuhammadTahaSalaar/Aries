#!/bin/bash
###############################################################################
# ARIES — ONNX Export Pipeline
#
# Exports all trained models to ONNX format for deployment.
#
# Usage:
#   sbatch slurm/export_onnx.sh
###############################################################################

#SBATCH --job-name=aries_onnx
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --partition=ampere_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --time=01:00:00

set -euo pipefail

echo "=============================================="
echo "ARIES ONNX Export Pipeline"
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID}"
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

# ── Export all models ─────────────────────────────────────────────────
echo ""
echo "Exporting all trained models to ONNX..."
srun python -m src.export_onnx --all

echo ""
echo "=============================================="
echo "ONNX export complete!"
echo "Models saved to: ${PROJECT_DIR}/models/onnx/"
echo "=============================================="
