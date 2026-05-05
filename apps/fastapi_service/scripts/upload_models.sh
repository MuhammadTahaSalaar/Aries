#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════
# ARIES — Upload ONNX models to MinIO
# Run from: apps/fastapi_service/
# ═══════════════════════════════════════════════════════════════════════
set -euo pipefail

MINIO_URL="${MINIO_URL:-http://localhost:9000}"
BUCKET="${ARIES_S3_BUCKET_MODELS:-aries-models}"
ACCESS_KEY="${MINIO_ACCESS_KEY:-minioadmin}"
SECRET_KEY="${MINIO_SECRET_KEY:-minioadmin}"
MODELS_ROOT="${MODELS_ROOT:-../../models}"

# Detect mc binary
MC=""
for candidate in mc mcli; do
  if command -v "$candidate" &>/dev/null; then
    MC="$candidate"
    break
  fi
done

if [[ -z "$MC" ]]; then
  echo "ERROR: minio client (mc) not found."
  echo "Install: https://min.io/docs/minio/linux/reference/minio-mc.html"
  exit 1
fi

echo "==> Configuring MinIO alias..."
$MC alias set aries "$MINIO_URL" "$ACCESS_KEY" "$SECRET_KEY" --api S3v4

echo "==> Ensuring bucket '$BUCKET' exists..."
$MC mb --ignore-existing "aries/$BUCKET"

# ── Triage ────────────────────────────────────────────────────────────
TRIAGE_MODEL="$MODELS_ROOT/onnx/triage.onnx"
if [[ -f "$TRIAGE_MODEL" ]]; then
  echo "==> Uploading triage model..."
  $MC cp "$TRIAGE_MODEL" "aries/$BUCKET/triage/triage.onnx"
else
  echo "WARN: $TRIAGE_MODEL not found — skipping triage."
fi

# ── NER ───────────────────────────────────────────────────────────────
NER_MODEL="$MODELS_ROOT/onnx/ner.opt.onnx"
if [[ -f "$NER_MODEL" ]]; then
  echo "==> Uploading NER model..."
  $MC cp "$NER_MODEL" "aries/$BUCKET/ner/ner.opt.onnx"
else
  echo "WARN: $NER_MODEL not found — skipping NER model."
fi

# NER tokenizer
NER_TOK_DIR="$MODELS_ROOT/ner"
TOK_FILES=(tokenizer.json tokenizer_config.json special_tokens_map.json vocab.json merges.txt config.json)
echo "==> Uploading NER tokenizer files..."
for f in "${TOK_FILES[@]}"; do
  if [[ -f "$NER_TOK_DIR/$f" ]]; then
    $MC cp "$NER_TOK_DIR/$f" "aries/$BUCKET/ner/tokenizer/$f"
  fi
done

# ── Summarizer ────────────────────────────────────────────────────────
SUMM_ENC="$MODELS_ROOT/onnx/summarizer/encoder.onnx"
SUMM_DEC="$MODELS_ROOT/onnx/summarizer/decoder.onnx"

if [[ -f "$SUMM_ENC" ]]; then
  echo "==> Uploading summarizer encoder..."
  $MC cp "$SUMM_ENC" "aries/$BUCKET/summarizer/encoder.onnx"
fi
if [[ -f "$SUMM_DEC" ]]; then
  echo "==> Uploading summarizer decoder..."
  $MC cp "$SUMM_DEC" "aries/$BUCKET/summarizer/decoder.onnx"
fi

# Summarizer tokenizer
SUMM_TOK_DIR="$MODELS_ROOT/summarizer"
echo "==> Uploading summarizer tokenizer files..."
for f in tokenizer.json tokenizer_config.json; do
  if [[ -f "$SUMM_TOK_DIR/$f" ]]; then
    $MC cp "$SUMM_TOK_DIR/$f" "aries/$BUCKET/summarizer/tokenizer/$f"
  fi
done

echo ""
echo "==> Upload complete. Listing bucket contents:"
$MC ls "aries/$BUCKET/" --recursive
echo ""
echo "Done. Restart the FastAPI service to pick up new models."
