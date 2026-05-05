"""
ARIES — Summarization FastAPI router.

Provides:
  POST /nlp/summarize — generate executive or analyst summary
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from src.nlp.summarizer.inference import run_summarization_inference
from src.nlp.summarizer.slm_inference import run_summarization_inference_slm
from src.nlp.summarizer.schemas import SummarizeRequest, SummarizeResult
from src.shared.config import get_settings
from src.shared.dependencies import get_tenant_id
from src.shared.exceptions import ModelNotLoadedError

router = APIRouter()


@router.post(
    "/summarize",
    response_model=SummarizeResult,
    summary="Generate an incident summary",
    description="Supports 'executive' (2-4 sentences) and 'analyst' (8-15 sentences) modes.",
)
async def summarize(
    body: SummarizeRequest,
    request: Request,
    tenant_id: str = Depends(get_tenant_id),
) -> SummarizeResult:
    """Generate a summary of the input text using the BART ONNX model."""
    model_store = request.app.state.model_store
    settings = get_settings()
    redis = getattr(request.app.state, "redis", None)
    db = getattr(request.app.state, "db", None)

    # Cache key includes mode
    cache_key = f"{body.text}:{body.mode.value}"

    # Check cache
    if redis and redis.is_connected:
        cached = await redis.get_cached(tenant_id, "summarizer", cache_key)
        if cached:
            result = SummarizeResult.model_validate(cached)
            result.cached = True
            return result

    if settings.use_slm:
        result = await run_summarization_inference_slm(
            text=body.text,
            mode=body.mode,
            settings=settings,
        )
    else:
        if not model_store.summarizer_loaded:
            raise ModelNotLoadedError("summarizer").to_http()

        result = await run_summarization_inference(
            encoder_session=model_store.summarizer_encoder_session,
            decoder_session=model_store.summarizer_decoder_session,
            tokenizer=model_store.summarizer_tokenizer,
            text=body.text,
            mode=body.mode,
            settings=settings,
        )

    # Cache result
    if redis and redis.is_connected:
        await redis.set_cached(
            tenant_id,
            "summarizer",
            cache_key,
            result.model_dump(mode="json"),
            settings.redis_cache_ttl_summary,
        )

    # Persist to case if case_id provided
    if body.case_id and db:
        try:
            exec_summary = result.summary if body.mode.value == "executive" else None
            analyst_summary = result.summary if body.mode.value == "analyst" else None
            await db.upsert_case_summary(
                case_id=body.case_id,
                tenant_id=tenant_id,
                executive_summary=exec_summary,
                analyst_summary=analyst_summary,
                model_version=result.model_version,
            )
        except Exception:
            pass  # Non-critical; log and continue

    return result


@router.get("/summarize/health", summary="Summarizer pipeline health")
async def summarizer_health(request: Request) -> dict:
    model_store = request.app.state.model_store
    return {
        "pipeline": "summarizer",
        "model_loaded": model_store.summarizer_loaded,
        "metadata": model_store.summarizer_metadata,
    }
