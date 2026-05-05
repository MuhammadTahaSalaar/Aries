# ── ARIES FastAPI Service ─────────────────────────────────────────────
# Multi-stage build with CUDA 12.1 GPU support for llama-cpp-python.
# Enables full GPU offloading of the Phi-3-mini Q4_K_M GGUF model
# (~2.4 GB VRAM), delivering ~10x faster inference vs CPU.
#
# Requires: nvidia-container-toolkit on the host + `docker compose up` with
# the GPU device reservation in docker-compose.yml (already configured).

# ── Stage 1: Builder ─────────────────────────────────────────────────
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04 AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3-pip \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Use python3.11 as default python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

WORKDIR /build

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Build llama-cpp-python with CUDA backend (GGML_CUDA=ON)
# This enables full GPU offloading: set n_gpu_layers=-1 at runtime.
RUN CMAKE_ARGS="-DGGML_CUDA=ON" \
    FORCE_CMAKE=1 \
    pip install --no-cache-dir --prefix=/install "llama-cpp-python>=0.3.0"

# ── Stage 2: Runtime ─────────────────────────────────────────────────
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04 AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3-pip \
    libgomp1 \
    libstdc++6 \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Non-root user for security
RUN groupadd -r aries && useradd -r -g aries -d /app -s /sbin/nologin aries

WORKDIR /app

# Copy installed Python packages from builder
COPY --from=builder /install /usr/local

# Copy application source
COPY src/ ./src/
COPY main.py .

# Create model cache dir (will be populated from S3 at startup)
RUN mkdir -p /tmp/aries_models && chown -R aries:aries /tmp/aries_models /app

USER aries

# Environment
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    ONNX_LOG_SEVERITY_LEVEL=3

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --retries=3 --start-period=60s \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", \
    "--workers", "2", "--loop", "uvloop", "--http", "httptools"]
