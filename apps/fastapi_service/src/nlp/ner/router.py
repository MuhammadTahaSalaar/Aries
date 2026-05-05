"""
ARIES — NER FastAPI router.

Provides:
  POST /nlp/ner       — single-text NER + IOC extraction
  POST /nlp/ner/batch — batch NER (up to 32 texts)
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from src.nlp.ner.inference import run_ner_inference
from src.nlp.ner.slm_inference import run_ner_inference_slm
from src.nlp.ner.schemas import NERBatchRequest, NERRequest, NERResult
from src.shared.config import get_settings
from src.shared.dependencies import get_tenant_id
from src.shared.exceptions import ModelNotLoadedError

router = APIRouter()


@router.post(
    "/ner",
    response_model=NERResult,
    summary="Extract named entities and IOCs from text",
)
async def extract_entities(
    body: NERRequest,
    request: Request,
    tenant_id: str = Depends(get_tenant_id),
) -> NERResult:
    """Run NER on input text, validate IOCs via regex, and return structured entities."""
    model_store = request.app.state.model_store
    settings = get_settings()
    redis = getattr(request.app.state, "redis", None)

    # Check cache
    if redis and redis.is_connected:
        cached = await redis.get_cached(tenant_id, "ner", body.text)
        if cached:
            result = NERResult.model_validate(cached)
            result.cached = True
            return result

    if settings.use_slm:
        result = await run_ner_inference_slm(
            text=body.text,
            settings=settings,
        )
    else:
        if not model_store.ner_loaded:
            raise ModelNotLoadedError("ner").to_http()

        id2label = {int(k): v for k, v in model_store.ner_metadata.get("id2label", {}).items()}

        result = await run_ner_inference(
            session=model_store.ner_session,
            tokenizer=model_store.ner_tokenizer,
            text=body.text,
            id2label=id2label or None,
            max_length=settings.ner_max_length,
        )

    # Cache result
    if redis and redis.is_connected:
        await redis.set_cached(
            tenant_id, "ner", body.text, result.model_dump(mode="json"), settings.redis_cache_ttl_ner
        )

    return result


@router.post(
    "/ner/batch",
    response_model=list[NERResult],
    summary="Batch NER extraction (max 32 texts)",
)
async def extract_entities_batch(
    body: NERBatchRequest,
    request: Request,
    tenant_id: str = Depends(get_tenant_id),
) -> list[NERResult]:
    """Run NER on a batch of texts."""
    model_store = request.app.state.model_store
    settings = get_settings()

    if settings.use_slm:
        results: list[NERResult] = []
        for text in body.texts[: settings.ner_batch_max]:
            result = await run_ner_inference_slm(
                text=text,
                settings=settings,
            )
            results.append(result)
        return results

    if not model_store.ner_loaded:
        raise ModelNotLoadedError("ner").to_http()

    id2label = {int(k): v for k, v in model_store.ner_metadata.get("id2label", {}).items()}

    results: list[NERResult] = []
    for text in body.texts[: settings.ner_batch_max]:
        result = await run_ner_inference(
            session=model_store.ner_session,
            tokenizer=model_store.ner_tokenizer,
            text=text,
            id2label=id2label or None,
            max_length=settings.ner_max_length,
        )
        results.append(result)

    return results


@router.get("/ner/health", summary="NER pipeline health")
async def ner_health(request: Request) -> dict:
    model_store = request.app.state.model_store
    return {
        "pipeline": "ner",
        "model_loaded": model_store.ner_loaded,
        "metadata": model_store.ner_metadata,
    }
