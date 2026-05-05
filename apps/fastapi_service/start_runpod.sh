#!/bin/bash
###############################################################################
# ARIES — RunPod Service Start Script
#
# Exports all required env vars and starts uvicorn directly.
# Edit the variables below to match your RunPod volume paths and credentials.
###############################################################################
set -euo pipefail

# ── Model paths (on RunPod persistent volume) ────────────────────────
export ARIES_USE_SLM=true
export ARIES_SLM_MODEL_PATH=/runpod-volume/aries_models/triage_slm_q4.gguf
export ARIES_SLM_NER_MODEL_PATH=/runpod-volume/aries_models/ner_slm_q4.gguf
export ARIES_SLM_SUMMARIZER_MODEL_PATH=/runpod-volume/aries_models/triage_slm_q4.gguf

# ── Disable infrastructure services not present on RunPod ─────────────
# The service degrades gracefully when these are unreachable:
# Kafka consumers won't start, MinIO model download is skipped,
# DB pool is skipped — HTTP inference endpoints still work fine.
export ARIES_DATABASE_URL=postgresql://aries:aries@localhost:5432/aries
export ARIES_REDIS_URL=redis://localhost:6379/0
export ARIES_S3_ENDPOINT_URL=http://localhost:9000
export ARIES_MLFLOW_TRACKING_URI=http://localhost:5000
export ARIES_KAFKA_BOOTSTRAP_SERVERS=localhost:9092

# ── Logging ───────────────────────────────────────────────────────────
export ARIES_LOG_LEVEL=INFO

# ── Start service ─────────────────────────────────────────────────────
cd /workspace/Aries/apps/fastapi_service

echo "Starting ARIES FastAPI service on port 8000..."
echo "SLM model : ${ARIES_SLM_MODEL_PATH}"
echo "NER model : ${ARIES_SLM_NER_MODEL_PATH}"
echo ""

# Use 1 worker on RunPod (GGUFs are not fork-safe with multiple workers)
uvicorn main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 1 \
    --loop uvloop \
    --http httptools
