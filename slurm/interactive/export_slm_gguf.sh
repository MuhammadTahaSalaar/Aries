#!/bin/bash
###############################################################################
# ARIES — Export SLM to GGUF (Interactive)
###############################################################################
set -euo pipefail

DATASET=${1:-triage}
echo "=============================================="
echo "ARIES SLM GGUF Export : ${DATASET}"
echo "=============================================="

module purge || true
module load Mamba || true
module load CUDA/12.1.1 || true   # needed so llama.cpp make detects nvcc for GPU quant support

PROJECT_DIR="${VSC_DATA_VO_USER}/Aries_SOAR"
ENV_PATH="${PROJECT_DIR}/envs/aries"
PYTHON_BIN="${ENV_PATH}/bin/python"

if [ ! -x "${PYTHON_BIN}" ]; then
    echo "ERROR: ${PYTHON_BIN} not found. Run setup_env.sh first."
    exit 1
fi

# Redirect HuggingFace model cache to VSC_DATA (home quota is too small for Phi-3-mini ~3.8GB)
export HF_HOME="${PROJECT_DIR}/hf_cache"
mkdir -p "${HF_HOME}"

export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

echo "[1/4] Merging weights..."
"${PYTHON_BIN}" src/slm/merge_model.py --dataset "${DATASET}"

echo "[2/4] Setting up llama.cpp..."
cd "${PROJECT_DIR}"
if [ ! -d "llama.cpp" ]; then
    git clone https://github.com/ggerganov/llama.cpp.git
fi
cd llama.cpp
"${ENV_PATH}/bin/pip" install -r requirements.txt

MERGED_DIR="${PROJECT_DIR}/models/slm_merged/${DATASET}"
GGUF_OUT="${PROJECT_DIR}/models/onnx/triage_slm/${DATASET}_slm_q4.gguf"
mkdir -p "$(dirname "${GGUF_OUT}")"

if [ ! -f "${MERGED_DIR}/tokenizer.model" ]; then
    echo "ERROR: ${MERGED_DIR}/tokenizer.model is missing."
    echo "Re-run merge with base tokenizer export:"
    echo "  ${PYTHON_BIN} src/slm/merge_model.py --dataset ${DATASET}"
    echo "Then re-run this export script."
    exit 1
fi

echo "[3/4] Converting to GGUF (FP16)..."
"${PYTHON_BIN}" convert_hf_to_gguf.py "${MERGED_DIR}" --outfile "${MERGED_DIR}/model-fp16.gguf" --outtype f16

echo "[4/4] Quantizing to Q4_K_M (4-bit)..."
# llama.cpp uses CMake only (Makefile was removed).
# CMake/3.26.3-GCCcore-12.3.0 loads GCC 12, which is the highest version
# CUDA 12.1.1 supports. CMake/3.29.3-GCCcore-13.3.0 loads GCC 13 and breaks nvcc.
module load CMake/3.26.3-GCCcore-12.3.0 2>/dev/null || true

# -DLLAMA_CURL=OFF disables the libcurl/OpenSSL download feature (HuggingFace hub).
# We don't need it — we supply the model file directly.
# Without this, the linker fails on Hydra due to an older OpenSSL missing
# SSL_get1_peer_certificate (added in OpenSSL 3.x).
# -DGGML_CUDA=ON enables GPU-accelerated quantization kernels.
rm -rf build
cmake -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_CUDA=ON \
    -DLLAMA_CURL=OFF \
    -DCMAKE_DISABLE_FIND_PACKAGE_OpenSSL=ON \
    --fresh \
    2>&1 | tail -5
cmake --build build --config Release --target llama-quantize -j8
./build/bin/llama-quantize "${MERGED_DIR}/model-fp16.gguf" "${GGUF_OUT}" q4_k_m

echo "=============================================="
echo "Done! GGUF model ready for deployment."
echo "Path: ${GGUF_OUT}"
echo ""
echo "Download to local machine:"
echo "  scp vsc11249@login.hpc.vub.be:${GGUF_OUT} ~/Downloads/"
echo "=============================================="