"""
ARIES FastAPI Service — Centralised configuration (pydantic-settings).

All settings are read from environment variables with the ARIES_ prefix.
Secrets must never be hardcoded; they are injected via env vars or Vault.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ServiceSettings(BaseSettings):
    """Single source of truth for the FastAPI inference service."""

    model_config = SettingsConfigDict(
        env_prefix="ARIES_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Service ───────────────────────────────────────────────────────
    service_name: str = "aries-fastapi-service"
    service_version: str = "1.0.0"
    log_level: str = "INFO"
    debug: bool = False

    # ── FastAPI / Uvicorn ─────────────────────────────────────────────
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 2

    # ── PostgreSQL (asyncpg) ──────────────────────────────────────────
    database_url: str = Field(
        default="postgresql://aries:aries@localhost:5432/aries",
        description="asyncpg-compatible connection string",
    )
    db_pool_min: int = 5
    db_pool_max: int = 20

    # ── Kafka ─────────────────────────────────────────────────────────
    kafka_bootstrap_servers: str = "localhost:9092"
    kafka_consumer_group_triage: str = "ml-triage-engine"
    kafka_consumer_group_ner: str = "nlp-ioc-extractor"
    kafka_consumer_group_feedback: str = "ml-feedback-collector"
    kafka_topic_alerts_raw: str = "alerts.raw"
    kafka_topic_alerts_enriched: str = "alerts.enriched"
    kafka_topic_cases_updated: str = "cases.updated"
    kafka_topic_ml_feedback: str = "ml.feedback"

    # ── Redis ─────────────────────────────────────────────────────────
    redis_url: str = "redis://localhost:6379/0"
    redis_cache_ttl_ner: int = 1800
    redis_cache_ttl_summary: int = 7200

    # ── MinIO / S3 ────────────────────────────────────────────────────
    s3_endpoint_url: str = "http://localhost:9000"
    s3_bucket_models: str = "mlflow-bucket"
    s3_access_key: str = "admin"
    s3_secret_key: str = "password123"
    s3_region: str = "us-east-1"

    # ── MLflow ────────────────────────────────────────────────────────
    mlflow_tracking_uri: str = "http://localhost:5000"

    # ── Model Paths (S3 keys) ────────────────────────────────────────
    model_triage_s3_key: str = "triage/triage.onnx"
    model_triage_encoder_s3_key: str = "triage/triage_encoder.pkl"
    model_ner_s3_key: str = "ner/ner.opt.onnx"
    model_ner_tokenizer_s3_prefix: str = "ner/tokenizer/"
    model_summarizer_encoder_s3_key: str = "summarizer/encoder.onnx"
    model_summarizer_decoder_s3_key: str = "summarizer/decoder.onnx"
    model_summarizer_tokenizer_s3_prefix: str = "summarizer/tokenizer/"

    # ── Local model cache (populated from S3 at startup) ─────────────
    model_cache_dir: Path = Field(default=Path("/tmp/aries_models"))

    # ── Inference ─────────────────────────────────────────────────────
    summarizer_backend: Literal["pytorch", "onnx"] = "onnx"
    use_slm: bool = Field(default=True, description="Use Local SLM for Triage/NER/Summarization")
    slm_model_path: str = Field(default="/tmp/aries_models/slm/triage_slm_q4.gguf", description="Path to triage GGUF model (used as fallback for all tasks)")
    slm_ner_model_path: str = Field(default="", description="Path to NER-specific GGUF model (falls back to slm_model_path if empty)")
    slm_summarizer_model_path: str = Field(default="", description="Path to summarizer-specific GGUF model (falls back to slm_model_path if empty)")

    @property
    def effective_ner_model_path(self) -> str:
        return self.slm_ner_model_path or self.slm_model_path

    @property
    def effective_summarizer_model_path(self) -> str:
        return self.slm_summarizer_model_path or self.slm_model_path
    auto_close_threshold: float = 0.01
    triage_weight_ml: float = 0.50
    triage_weight_asset: float = 0.30
    triage_weight_behavior: float = 0.20

    # ── NER ───────────────────────────────────────────────────────────
    ner_max_length: int = 512
    ner_batch_max: int = 32

    # ── Summarizer ────────────────────────────────────────────────────
    summarizer_max_input_tokens: int = 1024
    summarizer_executive_max_tokens: int = 100
    summarizer_executive_min_tokens: int = 50
    summarizer_analyst_max_tokens: int = 400
    summarizer_analyst_min_tokens: int = 150
    summarizer_num_beams: int = 4

    @property
    def triage_model_local(self) -> Path:
        return self.model_cache_dir / "triage" / "triage.onnx"

    @property
    def triage_encoder_local(self) -> Path:
        return self.model_cache_dir / "triage" / "triage_encoder.pkl"

    @property
    def ner_model_local(self) -> Path:
        return self.model_cache_dir / "ner" / "ner.opt.onnx"

    @property
    def ner_tokenizer_local(self) -> Path:
        return self.model_cache_dir / "ner" / "tokenizer"

    @property
    def summarizer_encoder_local(self) -> Path:
        return self.model_cache_dir / "summarizer" / "encoder.onnx"

    @property
    def summarizer_decoder_local(self) -> Path:
        return self.model_cache_dir / "summarizer" / "decoder.onnx"

    @property
    def summarizer_tokenizer_local(self) -> Path:
        return self.model_cache_dir / "summarizer" / "tokenizer"


_settings: ServiceSettings | None = None


def get_settings(**overrides: object) -> ServiceSettings:
    """Singleton accessor. Pass keyword overrides for testing."""
    global _settings
    if _settings is None or overrides:
        _settings = ServiceSettings(**overrides)  # type: ignore[arg-type]
    return _settings
