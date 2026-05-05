#!/bin/bash
###############################################################################
# ARIES — Fit TargetEncoder only (Interactive)
#
# Usage:
#   ./slurm/interactive/fit_encoder.sh
###############################################################################

set -euo pipefail

echo "=============================================="
echo "ARIES TargetEncoder Fitting (Interactive)"
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
ls -lh data/processed/triage_encoder.pkl models/triage/triage_encoder.pkl || true
echo "=============================================="