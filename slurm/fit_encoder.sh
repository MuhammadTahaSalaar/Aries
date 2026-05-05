#!/bin/bash
###############################################################################
# ARIES — Fit TargetEncoder only (CPU-only, no GPU needed)
#
# Reads GUIDE_Train.csv categorical columns + target, fits the TargetEncoder
# that is needed for real-time triage inference, and saves:
#   data/processed/triage_encoder.pkl
#   models/triage/triage_encoder.pkl
#
# This is much faster than running the full preprocessing pipeline.
# Expected runtime: 5-15 minutes on zen4 with 16 GB RAM.
#
# Usage:
#   sbatch slurm/fit_encoder.sh
###############################################################################

#SBATCH --job-name=aries_encoder
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --partition=zen4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00

set -euo pipefail

echo "=============================================="
echo "ARIES TargetEncoder Fitting"
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
cd "${PROJECT_DIR}"

# ── Make sure category_encoders is installed ──────────────────────────
pip install --quiet category-encoders joblib pandas 2>/dev/null || true

# ── Run ───────────────────────────────────────────────────────────────
echo ""
echo "Running fit_encoder_only.py ..."
python scripts/fit_encoder_only.py

echo ""
echo "=============================================="
echo "Done. Output files:"
ls -lh data/processed/triage_encoder.pkl models/triage/triage_encoder.pkl
echo "=============================================="
