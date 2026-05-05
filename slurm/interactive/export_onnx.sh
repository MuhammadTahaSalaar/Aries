#!/bin/bash
###############################################################################
# ARIES — ONNX Export Pipeline (Interactive)
#
# Exports all trained models to ONNX format for deployment.
#
# Usage:
#   ./slurm/interactive/export_onnx.sh
###############################################################################

set -euo pipefail

echo "=============================================="
echo "ARIES ONNX Export Pipeline (Interactive)"
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

# ── Export all models ─────────────────────────────────────────────────
echo ""
echo "Exporting all trained models to ONNX..."
python -m src.export_onnx --all

echo ""
echo "=============================================="
echo "ONNX export complete!"
echo "Models saved to: ${PROJECT_DIR}/models/onnx/"
echo "=============================================="