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
export PIP_DISABLE_PIP_VERSION_CHECK=1   # suppress pip upgrade notice

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
import llama_cpp
print(f'llama-cpp-python: {llama_cpp.__version__}')
# llama_print_system_info() returns a byte string like '... CUDA = 1 ...'
try:
    info = llama_cpp.llama_cpp.llama_print_system_info().decode()
    cuda_on = 'CUDA = 1' in info or 'CUBLAS = 1' in info
    print(f'CUDA backend compiled: {cuda_on}')
    if not cuda_on:
        print('WARNING: CUDA = 0 in system info — running CPU-only')
        print('  Check that nvcc was available during build (see above).')
except Exception as e:
    print(f'Could not read system info: {e}')
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
