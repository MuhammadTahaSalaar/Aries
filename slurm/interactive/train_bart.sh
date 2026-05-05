#!/bin/bash
###############################################################################
# ARIES — BART Summariser Training (Interactive)
#
# Usage:
#   ./slurm/interactive/train_bart.sh
###############################################################################

set -euo pipefail

echo "=============================================="
echo "ARIES BART Summariser Training (Interactive)"
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

# ── Verify GPUs ───────────────────────────────────────────────────────
echo ""
python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU count: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    print(f'  GPU {i}: {props.name} ({props.total_mem / 1e9:.1f} GB)')
"
echo ""

# ── Auto-detect batch size per GPU ────────────────────────────────────
read BATCH_SIZE GRAD_ACCUM <<< $(python -c "
import torch
if torch.cuda.is_available():
    mem = torch.cuda.get_device_properties(0).total_mem / 1e9
    if mem >= 100:    print('8 4')    # H200 (140GB)
    elif mem >= 30:   print('4 8')    # A100 (40GB)
    else:             print('2 16')   # P100 (16GB)
else:
    print('2 16')
")

echo "Auto-detected: batch_size=${BATCH_SIZE}  grad_accum=${GRAD_ACCUM}"

# ── Step 1: Preprocess GovReport ─────────────────────────────────────
echo ""
echo "Step 1: Preprocessing GovReport dataset..."
python -m src.nlp.summarizer.run_preprocessing

# ── Step 2: Train BART ───────────────────────────────────────────────
echo ""
echo "Step 2: Training BART summariser..."
python -m src.nlp.summarizer.run_training \
    --epochs 3 \
    --lr 3e-5 \
    --batch "${BATCH_SIZE}" \
    --grad-accum "${GRAD_ACCUM}"

echo ""
echo "=============================================="
echo "BART training complete!"
echo "=============================================="