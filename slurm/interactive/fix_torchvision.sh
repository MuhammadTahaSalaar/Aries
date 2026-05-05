#!/bin/bash
###############################################################################
# ARIES — One-time fix: remove mismatched torchvision from existing Hydra env
#
# Run this ONCE on Hydra to fix the existing env without recreating it:
#   bash slurm/interactive/fix_torchvision.sh
###############################################################################
set -euo pipefail

PROJECT_DIR="${VSC_DATA_VO_USER}/Aries_SOAR"
ENV_PATH="${PROJECT_DIR}/envs/aries"
PIP_BIN="${ENV_PATH}/bin/pip"
PYTHON_BIN="${ENV_PATH}/bin/python"

if [ ! -x "${PIP_BIN}" ]; then
    echo "ERROR: ${PIP_BIN} not found. Is the env set up?"
    exit 1
fi

echo "Removing torchvision and torchaudio from existing env..."
"${PIP_BIN}" uninstall -y torchvision torchaudio 2>/dev/null || true

# vLLM installs nvidia-cusparse at a CUDA 12.8 build level, which links against
# libnvJitLink.so.12 and requires the versioned symbol __nvJitLinkCreate_12_8.
# If nvidia-nvjitlink-cu12 in the env is older (12.4 or below), that symbol is
# missing and torch fails to import. Upgrade nvjitlink first so the symbol is
# available before we reinstall torch.
echo "Upgrading nvidia-nvjitlink-cu12 to provide CUDA 12.8 symbols..."
"${PIP_BIN}" install "nvidia-nvjitlink-cu12>=12.8" --upgrade

# Re-pin torch AFTER nvjitlink upgrade. Keep cu124 for ABI consistency with
# vllm's other nvidia-* packages.
echo "Re-pinning torch==2.5.1+cu124 (fixes bf16/training on GPU nodes)..."
"${PIP_BIN}" install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124 --force-reinstall --no-deps

echo "Verifying torch still imports cleanly..."
"${PYTHON_BIN}" -c "import torch; print('torch', torch.__version__, '— OK')"

echo "Verifying transformers/peft import chain..."
"${PYTHON_BIN}" -c "from transformers import AutoTokenizer; from peft import AutoPeftModelForCausalLM; print('transformers + peft — OK')"

echo ""
echo "Fix applied. Re-run export_slm_gguf.sh."
