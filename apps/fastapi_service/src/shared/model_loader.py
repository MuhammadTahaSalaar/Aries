"""
ARIES — Model loader: pulls ONNX artifacts from MinIO/S3 at startup.

On startup the loader:
  1. Queries MLflow to discover the latest trained model artifacts
  2. Downloads the ONNX models + tokenizers from MinIO (mlflow-bucket)
  3. Initialises ONNX InferenceSession objects stored in app.state
"""

from __future__ import annotations

import json
from pathlib import Path

import onnxruntime as ort

from src.shared.config import ServiceSettings
from src.shared.logging import get_logger
from src.shared.mlflow_resolver import ResolvedArtifacts, resolve_latest_models
from src.shared.s3_client import S3Client

log = get_logger("model_loader")


class ModelStore:
    """
    Holds ONNX InferenceSession instances and tokenizers.
    Populated during the FastAPI lifespan startup phase.
    """

    def __init__(self) -> None:
        self.triage_session: ort.InferenceSession | None = None
        self.triage_encoder: object | None = None
        self.ner_session: ort.InferenceSession | None = None
        self.summarizer_encoder_session: ort.InferenceSession | None = None
        self.summarizer_decoder_session: ort.InferenceSession | None = None
        self.ner_tokenizer: object | None = None  # tokenizers.Tokenizer or HF AutoTokenizer
        self.summarizer_tokenizer: object | None = None
        self.ner_metadata: dict = {}
        self.triage_metadata: dict = {}
        self.summarizer_metadata: dict = {}
        # True when the SLM GGUF file is present and llama_cpp is importable
        self.slm_ready: bool = False

    @property
    def triage_loaded(self) -> bool:
        """True when the ONNX triage session is loaded (not the SLM)."""
        return self.triage_session is not None

    @property
    def ner_loaded(self) -> bool:
        """True when the ONNX NER session and tokenizer are loaded."""
        return self.ner_session is not None and self.ner_tokenizer is not None

    @property
    def summarizer_loaded(self) -> bool:
        """True when all ONNX summarizer sessions and tokenizer are loaded."""
        return (
            self.summarizer_encoder_session is not None
            and self.summarizer_decoder_session is not None
            and self.summarizer_tokenizer is not None
        )

    def status(self) -> dict[str, bool]:
        """Health-check view: shows True for any component that can serve requests."""
        return {
            "triage": self.triage_loaded or self.slm_ready,
            "ner": self.ner_loaded or self.slm_ready,
            "summarizer": self.summarizer_loaded or self.slm_ready,
            "slm": self.slm_ready,
        }


def _create_ort_session(model_path: Path) -> ort.InferenceSession:
    """Create an ONNX Runtime session with CPU provider."""
    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_opts.intra_op_num_threads = 4
    sess_opts.inter_op_num_threads = 2
    sess_opts.log_severity_level = 3  # suppress verbose ORT logs
    return ort.InferenceSession(
        str(model_path),
        sess_options=sess_opts,
        providers=["CPUExecutionProvider"],
    )


def _load_tokenizer(tokenizer_dir: Path) -> object:
    """Load a HuggingFace fast tokenizer from a local directory."""
    from tokenizers import Tokenizer

    tokenizer_json = tokenizer_dir / "tokenizer.json"
    if tokenizer_json.exists():
        return Tokenizer.from_file(str(tokenizer_json))
    # Fallback: try AutoTokenizer (slower but more compatible)
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(tokenizer_dir))


def _load_json_metadata(path: Path) -> dict:
    """Load a JSON metadata file from disk."""
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


async def download_models(s3: S3Client, settings: ServiceSettings) -> None:
    """
    Download all ONNX models and tokenizers from S3 to local cache.

    Strategy:
      1. Query MLflow for the latest model artifacts (preferred path)
      2. If MLflow is unreachable, fall back to hardcoded S3 keys from config
    """
    cache = settings.model_cache_dir
    cache.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Try MLflow resolution ─────────────────────────────────
    resolved = await resolve_latest_models(settings.mlflow_tracking_uri)

    if resolved is not None:
        log.info("using_mlflow_resolved_artifacts", runs=resolved.resolved_runs)
        # Override the S3 bucket to mlflow-bucket (artifacts live there)
        original_bucket = s3._settings.s3_bucket_models
        s3._settings.s3_bucket_models = resolved.bucket
        try:
            await _download_from_resolved(s3, settings, resolved)
        finally:
            s3._settings.s3_bucket_models = original_bucket
        return

    # ── Step 2: Fallback to hardcoded keys from config ────────────────
    log.info("mlflow_unavailable_using_fallback_keys")
    await _download_from_config(s3, settings)


# Tokenizer file extensions to filter when downloading from artifact root
_TOKENIZER_EXTS = {
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "added_tokens.json",
}


async def _download_from_resolved(
    s3: S3Client, settings: ServiceSettings, artifacts: ResolvedArtifacts
) -> None:
    """Download models using S3 keys resolved from MLflow."""

    # Triage ONNX
    if artifacts.triage_onnx_key:
        triage_path = settings.triage_model_local
        if not triage_path.exists():
            try:
                await s3.download_file(artifacts.triage_onnx_key, triage_path)
            except Exception:
                log.warning("triage_model_download_failed", key=artifacts.triage_onnx_key)

    # Triage TargetEncoder
    if artifacts.triage_encoder_key:
        enc_path = settings.triage_encoder_local
        if not enc_path.exists():
            try:
                await s3.download_file(artifacts.triage_encoder_key, enc_path)
            except Exception:
                log.warning("triage_encoder_download_failed", key=artifacts.triage_encoder_key)

    # NER ONNX
    if artifacts.ner_onnx_key:
        ner_path = settings.ner_model_local
        if not ner_path.exists():
            try:
                await s3.download_file(artifacts.ner_onnx_key, ner_path)
            except Exception:
                log.warning("ner_model_download_failed", key=artifacts.ner_onnx_key)

    # NER tokenizer (from artifact root)
    if artifacts.ner_tokenizer_prefix:
        ner_tok = settings.ner_tokenizer_local
        if not (ner_tok / "tokenizer.json").exists():
            try:
                await _download_tokenizer_files(
                    s3, artifacts.ner_tokenizer_prefix, ner_tok
                )
            except Exception:
                log.warning("ner_tokenizer_download_failed")

    # Summarizer encoder ONNX
    if artifacts.summarizer_encoder_key:
        enc_path = settings.summarizer_encoder_local
        if not enc_path.exists():
            try:
                await s3.download_file(artifacts.summarizer_encoder_key, enc_path)
            except Exception:
                log.warning("summarizer_encoder_download_failed")

    # Summarizer decoder ONNX
    if artifacts.summarizer_decoder_key:
        dec_path = settings.summarizer_decoder_local
        if not dec_path.exists():
            try:
                await s3.download_file(artifacts.summarizer_decoder_key, dec_path)
            except Exception:
                log.warning("summarizer_decoder_download_failed")

    # Summarizer tokenizer (from artifact root)
    if artifacts.summarizer_tokenizer_prefix:
        sum_tok = settings.summarizer_tokenizer_local
        if not (sum_tok / "tokenizer.json").exists():
            try:
                await _download_tokenizer_files(
                    s3, artifacts.summarizer_tokenizer_prefix, sum_tok
                )
            except Exception:
                log.warning("summarizer_tokenizer_download_failed")


async def _download_tokenizer_files(
    s3: S3Client, prefix: str, local_dir: Path
) -> None:
    """Download only tokenizer files from an S3 prefix."""
    local_dir.mkdir(parents=True, exist_ok=True)
    all_keys = await s3.list_keys(prefix)
    tokenizer_keys = [
        k for k in all_keys if k.rsplit("/", 1)[-1] in _TOKENIZER_EXTS
    ]
    for key in tokenizer_keys:
        filename = key.rsplit("/", 1)[-1]
        dest = local_dir / filename
        if not dest.exists():
            await s3.download_file(key, dest)


async def _download_from_config(s3: S3Client, settings: ServiceSettings) -> None:
    """Download models using hardcoded S3 keys from ServiceSettings (fallback)."""

    # Triage model
    triage_path = settings.triage_model_local
    if not triage_path.exists():
        try:
            await s3.download_file(settings.model_triage_s3_key, triage_path)
        except Exception:
            log.warning("triage_model_download_failed", key=settings.model_triage_s3_key)

    # Triage TargetEncoder
    encoder_path = settings.triage_encoder_local
    if not encoder_path.exists():
        try:
            await s3.download_file(settings.model_triage_encoder_s3_key, encoder_path)
        except Exception:
            log.warning("triage_encoder_download_failed", key=settings.model_triage_encoder_s3_key)

    # NER model
    ner_path = settings.ner_model_local
    if not ner_path.exists():
        try:
            await s3.download_file(settings.model_ner_s3_key, ner_path)
        except Exception:
            log.warning("ner_model_download_failed", key=settings.model_ner_s3_key)

    # NER tokenizer
    ner_tok = settings.ner_tokenizer_local
    if not (ner_tok / "tokenizer.json").exists():
        try:
            await s3.download_prefix(settings.model_ner_tokenizer_s3_prefix, ner_tok)
        except Exception:
            log.warning("ner_tokenizer_download_failed", prefix=settings.model_ner_tokenizer_s3_prefix)

    # Summarizer encoder
    enc_path = settings.summarizer_encoder_local
    if not enc_path.exists():
        try:
            await s3.download_file(settings.model_summarizer_encoder_s3_key, enc_path)
        except Exception:
            log.warning("summarizer_encoder_download_failed")

    # Summarizer decoder
    dec_path = settings.summarizer_decoder_local
    if not dec_path.exists():
        try:
            await s3.download_file(settings.model_summarizer_decoder_s3_key, dec_path)
        except Exception:
            log.warning("summarizer_decoder_download_failed")

    # Summarizer tokenizer
    sum_tok = settings.summarizer_tokenizer_local
    if not (sum_tok / "tokenizer.json").exists():
        try:
            await s3.download_prefix(settings.model_summarizer_tokenizer_s3_prefix, sum_tok)
        except Exception:
            log.warning("summarizer_tokenizer_download_failed")


def load_model_store(settings: ServiceSettings) -> ModelStore:
    """
    Synchronously load all ONNX sessions from local cache into a ModelStore.
    Call AFTER download_models() has completed.
    """
    store = ModelStore()

    # ── SLM readiness check ───────────────────────────────────────────
    # Mark slm_ready if the GGUF file is present and llama_cpp is importable.
    # When slm_ready=True the triage/ner/summarizer_loaded properties report
    # True even if ONNX sessions haven't loaded, giving an accurate /health.
    if settings.use_slm:
        slm_path = Path(settings.slm_model_path)
        try:
            import llama_cpp  # noqa: F401
            llama_available = True
        except ImportError:
            llama_available = False
        if llama_path_ok := slm_path.exists():
            log.info("slm_model_found", path=str(slm_path))
        else:
            log.warning("slm_model_not_found", path=str(slm_path))
        store.slm_ready = llama_available and llama_path_ok
        if store.slm_ready:
            log.info("slm_ready", path=str(slm_path))
        elif not llama_available:
            log.warning("llama_cpp_not_installed_slm_disabled")

    # ── Triage ────────────────────────────────────────────────────────
    triage_path = settings.triage_model_local
    if triage_path.exists():
        try:
            store.triage_session = _create_ort_session(triage_path)
            store.triage_metadata = _load_json_metadata(
                triage_path.parent / "model_metadata.json"
            )
            log.info("triage_model_loaded", path=str(triage_path))
        except Exception:
            log.exception("triage_model_load_failed")
    else:
        log.warning("triage_model_not_found", path=str(triage_path))
    # Triage TargetEncoder (required for correct categorical encoding)
    encoder_path = settings.triage_encoder_local
    if encoder_path.exists():
        try:
            import joblib
            store.triage_encoder = joblib.load(encoder_path)
            log.info("triage_encoder_loaded", path=str(encoder_path))
        except Exception:
            log.exception("triage_encoder_load_failed")
    else:
        log.warning(
            "triage_encoder_not_found_using_hash_fallback",
            path=str(encoder_path),
        )
    # ── NER ───────────────────────────────────────────────────────────
    ner_path = settings.ner_model_local
    ner_tok_path = settings.ner_tokenizer_local
    if ner_path.exists():
        try:
            store.ner_session = _create_ort_session(ner_path)
            store.ner_metadata = _load_json_metadata(
                ner_path.parent / "model_metadata.json"
            )
            log.info("ner_model_loaded", path=str(ner_path))
        except Exception:
            log.exception("ner_model_load_failed")

    if (ner_tok_path / "tokenizer.json").exists() or (ner_tok_path / "tokenizer_config.json").exists():
        try:
            store.ner_tokenizer = _load_tokenizer(ner_tok_path)
            log.info("ner_tokenizer_loaded", path=str(ner_tok_path))
        except Exception:
            log.exception("ner_tokenizer_load_failed")

    # ── Summarizer ────────────────────────────────────────────────────
    enc_path = settings.summarizer_encoder_local
    dec_path = settings.summarizer_decoder_local
    sum_tok_path = settings.summarizer_tokenizer_local

    if enc_path.exists():
        try:
            store.summarizer_encoder_session = _create_ort_session(enc_path)
            log.info("summarizer_encoder_loaded", path=str(enc_path))
        except Exception:
            log.exception("summarizer_encoder_load_failed")

    if dec_path.exists():
        try:
            store.summarizer_decoder_session = _create_ort_session(dec_path)
            store.summarizer_metadata = _load_json_metadata(
                dec_path.parent / "model_metadata.json"
            )
            log.info("summarizer_decoder_loaded", path=str(dec_path))
        except Exception:
            log.exception("summarizer_decoder_load_failed")

    if (sum_tok_path / "tokenizer.json").exists() or (sum_tok_path / "tokenizer_config.json").exists():
        try:
            store.summarizer_tokenizer = _load_tokenizer(sum_tok_path)
            log.info("summarizer_tokenizer_loaded", path=str(sum_tok_path))
        except Exception:
            log.exception("summarizer_tokenizer_load_failed")

    return store
