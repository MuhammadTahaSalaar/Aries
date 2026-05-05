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

echo "=============================================="
echo "ARIES FastAPI — RunPod Direct Install"
echo "Date: $(date)"
echo "CUDA devices: $(nvidia-smi -L 2>/dev/null || echo 'none visible')"
echo "=============================================="

# ── 1. llama-cpp-python with CUDA (must be BEFORE requirements.txt) ──
# pip install from requirements.txt would install a CPU-only wheel if
# llama-cpp-python appears there. We build from source with GGML_CUDA=ON
# first so the CUDA build is already present and pip skips it later.
echo "Building llama-cpp-python with CUDA support..."
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --force-reinstall --no-cache-dir

# ── 2. All other dependencies ─────────────────────────────────────────
echo "Installing remaining dependencies..."
pip install -r requirements.txt

# ── 3. Verify llama-cpp-python kept its CUDA build ────────────────────
echo "Verifying llama-cpp-python CUDA build..."
python -c "
from llama_cpp import Llama
import llama_cpp
print(f'llama-cpp-python: {llama_cpp.__version__}')
# Check if CUDA support compiled in
supports_gpu = getattr(llama_cpp.llama_cpp, 'GGML_USE_CUDA', False) or \
               getattr(llama_cpp.llama_cpp, 'GGML_USE_CUBLAS', False)
print(f'CUDA backend compiled: {supports_gpu}')
" 2>/dev/null || echo "WARNING: could not verify CUDA flag (non-fatal)"

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
