#!/bin/bash
###############################################################################
# ARIES — RunPod Direct Setup Script
#
# RunPod pods ARE Docker containers, so nested `docker build` doesn't work.
# This script installs everything directly into the RunPod Python environment
# and starts the FastAPI service with uvicorn.
#
# Usage (inside a RunPod terminal):
#   cd /workspace/Aries/apps/fastapi_service
#   bash setup_runpod.sh
###############################################################################
set -euo pipefail
export PIP_DISABLE_PIP_VERSION_CHECK=1        # suppress pip upgrade notice
export PIP_ROOT_USER_ACTION=ignore            # suppress pip root-user warning

echo "=============================================="
echo "ARIES FastAPI — RunPod Direct Install"
echo "Date: $(date)"
echo "CUDA devices: $(nvidia-smi -L 2>/dev/null || echo 'none visible')"
echo "=============================================="

# ── 1. Ensure nvcc is on PATH so cmake can find the CUDA toolkit ──────
# RunPod images have the CUDA toolkit at /usr/local/cuda but nvcc may not
# be on PATH yet. Without nvcc, cmake silently falls back to a CPU build.
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

if ! command -v nvcc &>/dev/null; then
    echo "WARNING: nvcc not found at ${CUDA_HOME}/bin — CUDA build may fall back to CPU"
else
    echo "nvcc found: $(nvcc --version | grep release)"
fi

# ── 2. llama-cpp-python with CUDA (must be BEFORE requirements.txt) ───
# FORCE_CMAKE=1 forces source compilation (never uses a pre-built wheel).
# Without it, pip may silently pick a CPU-only wheel from PyPI.
echo "Building llama-cpp-python with CUDA support..."
CMAKE_ARGS="-DGGML_CUDA=ON" FORCE_CMAKE=1 \
    pip install llama-cpp-python --force-reinstall --no-cache-dir

# ── 3. All other dependencies ─────────────────────────────────────────
echo "Installing remaining dependencies..."
pip install -r requirements.txt

# ── 4. Verify llama-cpp-python kept its CUDA build ────────────────────
echo "Verifying llama-cpp-python CUDA build..."
python -c "
import sys, io, contextlib
import llama_cpp

print(f'llama-cpp-python: {llama_cpp.__version__}')

# llama_print_system_info() format changed across versions.
# In 0.3.x it uses 'GGML_USE_CUDA' not 'CUDA'.
# Additionally, importing llama_cpp itself prints 'ggml_cuda_init: found N CUDA devices'
# to stderr — that line ONLY appears when the CUDA backend is compiled in.
try:
    info = llama_cpp.llama_cpp.llama_print_system_info().decode()
    cuda_on = ('GGML_USE_CUDA = 1' in info or
               'CUDA = 1' in info or
               'CUBLAS = 1' in info or
               'GGML_CUDA = 1' in info)
    print(f'System info CUDA flag  : {cuda_on}')
    print(f'System info (relevant) : {[s.strip() for s in info.split(\"|\") if \"CUDA\" in s or \"CUBLAS\" in s]}')
except Exception as e:
    print(f'Could not read system info: {e}')

# The definitive check: ggml_cuda_init prints to stderr on import — capture it
import subprocess, sys as _sys
result = subprocess.run(
    [_sys.executable, '-c', 'import llama_cpp'],
    capture_output=True, text=True
)
if 'ggml_cuda_init: found' in result.stderr:
    print()
    print('CUDA backend compiled: True  (ggml_cuda_init found CUDA devices in stderr)')
    print(result.stderr.strip())
else:
    print('WARNING: ggml_cuda_init not found in stderr — CUDA may not be compiled in')
    print(f'stderr: {result.stderr[:200]}')
"

echo ""
echo "=============================================="
echo "Setup complete. Start the service with:"
echo ""
echo "  bash start_runpod.sh"
echo ""
echo "Or manually:"
echo "  cd /workspace/Aries/apps/fastapi_service"
echo "  uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1"
echo "=============================================="
