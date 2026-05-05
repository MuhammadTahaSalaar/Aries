#!/bin/bash
###############################################################################
# ARIES — BART Summariser Training (Multi-GPU)
#
# Fine-tunes BART on GovReport for CTI report summarisation.
# Uses 2 GPUs with HuggingFace Trainer's built-in DataParallel.
#
# Usage:
#   sbatch slurm/train_bart.sh                          # Default: ampere_gpu
#   sbatch --partition=hopper_gpu slurm/train_bart.sh   # H200s
#   sbatch --partition=pascal_gpu slurm/train_bart.sh   # P100s
#
# Partition recommendations:
#   pascal_gpu:  batch=2  grad_accum=16  bart-base   ~24h
#   ampere_gpu:  batch=4  grad_accum=8   bart-base   ~8h
#   hopper_gpu:  batch=8  grad_accum=4   bart-base   ~3h
###############################################################################

#SBATCH --job-name=aries_bart
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --partition=ampere_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus=1
#SBATCH --mem=200G
#SBATCH --time=12:00:00

set -euo pipefail

echo "=============================================="
echo "ARIES BART Summariser Training"
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Partition: ${SLURM_JOB_PARTITION}"
echo "GPUs: ${SLURM_GPUS:-2}"
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
srun python -m src.nlp.summarizer.run_preprocessing

# ── Step 2: Train BART ───────────────────────────────────────────────
echo ""
echo "Step 2: Training BART summariser..."
srun python -m src.nlp.summarizer.run_training \
    --epochs 3 \
    --lr 3e-5 \
    --batch "${BATCH_SIZE}" \
    --grad-accum "${GRAD_ACCUM}"

echo ""
echo "=============================================="
echo "BART training complete!"
echo "=============================================="
