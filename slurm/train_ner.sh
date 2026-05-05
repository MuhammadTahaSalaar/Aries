#!/bin/bash
###############################################################################
# ARIES — SecureBERT NER Training (GPU)
#
# Fine-tunes SecureBERT on CyNER + CASIE for cybersecurity NER.
# Single GPU is sufficient for the ~125M parameter model.
#
# Usage:
#   sbatch slurm/train_ner.sh
#   sbatch --partition=ampere_gpu slurm/train_ner.sh
###############################################################################

#SBATCH --job-name=aries_ner
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --partition=ampere_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=01:00:00

set -euo pipefail

echo "=============================================="
echo "ARIES SecureBERT NER Training"
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
    srun python -m src.nlp.ner.run_preprocessing
fi

# ── Step 2: Train SecureBERT-NER ─────────────────────────────────────
echo ""
echo "Step 2: Training SecureBERT-NER..."
srun python -m src.nlp.ner.run_training \
    --epochs 10 \
    --lr 2e-5 \
    --batch "${BATCH_SIZE}"

echo ""
echo "=============================================="
echo "NER training complete!"
echo "=============================================="
