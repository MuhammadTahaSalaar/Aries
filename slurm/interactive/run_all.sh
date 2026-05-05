#!/bin/bash
###############################################################################
# ARIES — Run All Pipelines (Interactive)
#
# Submits all training and export jobs in sequence.
#
# Usage:
#   ./slurm/interactive/run_all.sh
###############################################################################

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=============================================="
echo "ARIES Interactive Training Pipeline"
echo "=============================================="

PROJECT_DIR="${VSC_DATA_VO_USER}/Aries_SOAR"

if [ ! -d "${PROJECT_DIR}/envs/aries" ]; then
    echo "Running setup_env.sh..."
    bash "${SCRIPT_DIR}/setup_env.sh"
fi

echo "Running train_xgboost.sh..."
bash "${SCRIPT_DIR}/train_xgboost.sh"

echo "Running train_ner.sh..."
bash "${SCRIPT_DIR}/train_ner.sh"

echo "Running train_bart.sh..."
bash "${SCRIPT_DIR}/train_bart.sh"

echo "Running export_onnx.sh..."
bash "${SCRIPT_DIR}/export_onnx.sh"

echo "All interactive pipelines completed."