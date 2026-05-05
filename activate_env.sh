# 1. Load the drivers and Mamba
module purge
module load CUDA/12.1.1  # Matches the PyTorch version we installed
module load Mamba

# 2. Universal Activation
MAMBA_ROOT=$(dirname $(dirname $(which mamba)))
source "${MAMBA_ROOT}/etc/profile.d/conda.sh"
source "${MAMBA_ROOT}/etc/profile.d/mamba.sh"


# 3. Activate using the full path to your environment
mamba activate "${VSC_DATA}/Aries/envs/aries"


# 3.5. Verify GPU is visible to PyTorch
python -c "import torch; print(f'GPU Found: {torch.cuda.is_available()}')"

# 4. Set project-specific environment variables
export PROJECT_DIR="${VSC_DATA_VO_USER}/Aries_SOAR"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
export ARIES_HPC_PROJECT_DIR="${PROJECT_DIR}"
export MLFLOW_TRACKING_URI="file://${PROJECT_DIR}/mlruns"