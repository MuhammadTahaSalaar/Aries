#!/bin/bash
###############################################################################
# ARIES — SecureBERT NER Training (Interactive)
#
# Usage:
#   ./slurm/interactive/train_ner.sh
###############################################################################

set -euo pipefail

echo "=============================================="
echo "ARIES SecureBERT NER Training (Interactive)"
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
    props = torch.cuda.get_device_properties(0)
    print(f'Memory: {props.total_memory / 1e9:.1f} GB')
"
echo ""

# ── Detect GPU and set batch size ─────────────────────────────────────
BATCH_SIZE=$(python -c "
import torch
if torch.cuda.is_available():
    mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    if mem >= 100: print(32)      # H200
    elif mem >= 30: print(16)     # A100
    else: print(8)                # P100
else:
    print(8)
")
echo "Auto-detected batch size: ${BATCH_SIZE}"

# ── Step 1: Preprocess NER data (skip if already done) ──────────────
NER_DATASET="${PROJECT_DIR}/data/processed/ner_dataset"
echo ""
if [ -d "${NER_DATASET}" ]; then
    echo "Step 1: Skipping preprocessing — ${NER_DATASET} already exists."
else
    echo "Step 1: Preprocessing CyNER + CASIE..."
    python -m src.nlp.ner.run_preprocessing
fi

# ── Step 2: Train SecureBERT-NER ─────────────────────────────────────
echo ""
echo "Step 2: Training SecureBERT-NER..."
python -m src.nlp.ner.run_training \
    --epochs 10 \
    --lr 2e-5 \
    --batch "${BATCH_SIZE}"

echo ""
echo "=============================================="
echo "NER training complete!"
echo "=============================================="