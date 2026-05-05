#!/bin/bash
###############################################################################
# ARIES — Job Submission Orchestrator
#
# Submits all training and export jobs to VUB Hydra HPC with dependency chains.
#
# Usage:
#   ./slurm/submit_all.sh                           # Full pipeline on ampere_gpu
#   ./slurm/submit_all.sh --partition=hopper_gpu     # Use H200s
#   ./slurm/submit_all.sh --xgboost-only             # Only triage
#   ./slurm/submit_all.sh --ner-only                 # Only NER
#   ./slurm/submit_all.sh --bart-only                # Only summariser
#   ./slurm/submit_all.sh --skip-setup               # Skip env setup
###############################################################################

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Defaults
PARTITION="ampere_gpu"
SETUP_ONLY=false
XGBOOST_ONLY=false
NER_ONLY=false
BART_ONLY=false
SKIP_SETUP=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --partition=*)  PARTITION="${1#*=}"; shift ;;
        --setup-only)   SETUP_ONLY=true; shift ;;
        --xgboost-only) XGBOOST_ONLY=true; shift ;;
        --ner-only)     NER_ONLY=true; shift ;;
        --bart-only)    BART_ONLY=true; shift ;;
        --skip-setup)   SKIP_SETUP=true; shift ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--partition=<name>] [--setup-only] [--xgboost-only] [--ner-only] [--bart-only] [--skip-setup]"
            exit 1
            ;;
    esac
done

echo "=============================================="
echo "ARIES Job Submission"
echo "Date: $(date)"
echo "Partition: ${PARTITION}"
echo "=============================================="

PROJECT_DIR="${VSC_DATA_VO_USER}/Aries_SOAR"
ENV_PATH="${PROJECT_DIR}/envs/aries"

# ── Step 1: Environment Setup ────────────────────────────────────────
DEPENDENCY=""
if [ "$SKIP_SETUP" = false ] && [ ! -d "${ENV_PATH}" ]; then
    echo ""
    echo "Step 1: Submitting environment setup..."
    SETUP_JOB=$(sbatch --parsable "${SCRIPT_DIR}/setup_env.sh")
    echo "  Job ID: ${SETUP_JOB}"
    DEPENDENCY="--dependency=afterok:${SETUP_JOB}"
else
    echo ""
    echo "Step 1: Environment exists or --skip-setup, skipping..."
fi

if [ "$SETUP_ONLY" = true ]; then
    echo "Setup-only mode, done."
    exit 0
fi

# Determine which pipelines to run
RUN_ALL=true
if [ "$XGBOOST_ONLY" = true ] || [ "$NER_ONLY" = true ] || [ "$BART_ONLY" = true ]; then
    RUN_ALL=false
fi

# ── Step 2: XGBoost Triage ───────────────────────────────────────────
if [ "$RUN_ALL" = true ] || [ "$XGBOOST_ONLY" = true ]; then
    echo ""
    echo "Step 2: Submitting XGBoost triage training..."
    XGB_JOB=$(sbatch --parsable --partition="${PARTITION}" ${DEPENDENCY} "${SCRIPT_DIR}/train_xgboost.sh")
    echo "  Job ID: ${XGB_JOB}"
fi

# ── Step 3: NER Training ─────────────────────────────────────────────
if [ "$RUN_ALL" = true ] || [ "$NER_ONLY" = true ]; then
    echo ""
    echo "Step 3: Submitting NER training..."
    NER_JOB=$(sbatch --parsable --partition="${PARTITION}" ${DEPENDENCY} "${SCRIPT_DIR}/train_ner.sh")
    echo "  Job ID: ${NER_JOB}"
fi

# ── Step 4: BART Summariser ──────────────────────────────────────────
if [ "$RUN_ALL" = true ] || [ "$BART_ONLY" = true ]; then
    echo ""
    echo "Step 4: Submitting BART summariser training..."
    BART_JOB=$(sbatch --parsable --partition="${PARTITION}" ${DEPENDENCY} "${SCRIPT_DIR}/train_bart.sh")
    echo "  Job ID: ${BART_JOB}"
fi

# ── Step 5: ONNX Export (after all training) ─────────────────────────
if [ "$RUN_ALL" = true ]; then
    # Depend on all three training jobs
    EXPORT_DEP="--dependency=afterok"
    [ -n "${XGB_JOB:-}" ]  && EXPORT_DEP="${EXPORT_DEP}:${XGB_JOB}"
    [ -n "${NER_JOB:-}" ]  && EXPORT_DEP="${EXPORT_DEP}:${NER_JOB}"
    [ -n "${BART_JOB:-}" ] && EXPORT_DEP="${EXPORT_DEP}:${BART_JOB}"

    echo ""
    echo "Step 5: Submitting ONNX export (after training completes)..."
    ONNX_JOB=$(sbatch --parsable --partition="${PARTITION}" ${EXPORT_DEP} "${SCRIPT_DIR}/export_onnx.sh")
    echo "  Job ID: ${ONNX_JOB}"
fi

echo ""
echo "=============================================="
echo "All jobs submitted. Monitor with: squeue -u \$USER"
echo "=============================================="
