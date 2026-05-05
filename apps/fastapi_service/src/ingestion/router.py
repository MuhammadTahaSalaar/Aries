"""
ARIES — SIEM Ingestion FastAPI router.

Provides:
  POST /ingest/siem — receive raw SIEM webhook, normalize, dedup, push to Kafka
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Body, Depends, Query, Request

from src.ingestion.normalizer import normalize_siem_alert
from src.ingestion.schemas import IngestionResult, SIEMRawPayload
from src.shared.dependencies import get_tenant_id
from src.shared.logging import get_logger

log = get_logger("ingestion_router")

router = APIRouter()


def _build_payload(
    body: dict[str, Any],
    vendor_query: str | None,
    tenant_id: str,
) -> SIEMRawPayload:
    """Build a SIEMRawPayload from the request body, query params, and header.

    If the body already wraps fields in the {vendor, raw, ...} envelope, use
    them directly.  Otherwise treat the whole body as the raw alert and pull
    the vendor from the ?vendor= query parameter.
    """
    body_vendor = body.get("vendor")
    body_raw = body.get("raw")

    vendor = vendor_query or body_vendor
    if body_raw is not None:
        raw = body_raw
    else:
        # The body itself is the raw vendor payload — strip our envelope keys
        raw = {k: v for k, v in body.items() if k not in ("vendor", "tenant_id", "raw", "timestamp")}

    return SIEMRawPayload(
        vendor=vendor,
        tenant_id=tenant_id,
        raw=raw or body,
        timestamp=body.get("timestamp"),
    )


@router.post(
    "/siem",
    response_model=IngestionResult,
    summary="Ingest a raw SIEM/EDR alert",
    description=(
        "Receives vendor-specific JSON, normalises to the ARIES Canonical Alert "
        "Schema, deduplicates via Redis, and publishes to Kafka alerts.raw.\n\n"
        "The vendor can be specified as a ?vendor= query parameter or inside the "
        "body.  The raw alert JSON can be nested under a 'raw' key or sent as "
        "the top-level body."
    ),
)
async def ingest_siem_alert(
    request: Request,
    body: dict[str, Any] = Body(...),
    vendor: str | None = Query(default=None, description="SIEM vendor (e.g. wazuh, splunk)"),
    tenant_id: str = Depends(get_tenant_id),
) -> IngestionResult:
    """Normalize and ingest a raw SIEM alert into the ARIES pipeline."""
    payload = _build_payload(body, vendor, tenant_id)

    # Normalize the raw payload
    canonical = normalize_siem_alert(payload)

    # Deduplication check via Redis
    redis = getattr(request.app.state, "redis", None)
    if redis and redis.is_connected and canonical.dedup_key:
        is_dup = await redis.get_dedup(canonical.dedup_key)
        if is_dup:
            log.info(
                "alert_deduplicated",
                alert_id=canonical.alert_id,
                dedup_key=canonical.dedup_key,
            )
            return IngestionResult(
                accepted=True,
                alert_id=canonical.alert_id,
                tenant_id=tenant_id,
                source=canonical.source,
                deduplicated=True,
            )
        await redis.set_dedup(canonical.dedup_key, ttl=3600)

    # Publish to Kafka alerts.raw
    producer = getattr(request.app.state, "kafka_producer", None)
    if producer and producer.is_connected:
        await producer.send(
            topic=request.app.state.settings.kafka_topic_alerts_raw,
            value=canonical.model_dump(mode="json"),
            key=canonical.alert_id,
        )
        log.info(
            "alert_published_to_kafka",
            alert_id=canonical.alert_id,
            topic="alerts.raw",
            tenant_id=tenant_id,
        )
    else:
        log.warning(
            "kafka_producer_unavailable",
            alert_id=canonical.alert_id,
        )

    return IngestionResult(
        accepted=True,
        alert_id=canonical.alert_id,
        tenant_id=tenant_id,
        source=canonical.source,
    )


@router.post(
    "/siem/batch",
    response_model=list[IngestionResult],
    summary="Ingest a batch of SIEM alerts",
)
async def ingest_siem_alerts_batch(
    body: list[SIEMRawPayload],
    request: Request,
    tenant_id: str = Depends(get_tenant_id),
) -> list[IngestionResult]:
    """Normalize and ingest a batch of raw SIEM alerts."""
    results: list[IngestionResult] = []
    for payload in body[:100]:  # Cap at 100 per batch
        payload.tenant_id = tenant_id
        # Reuse single-alert logic
        canonical = normalize_siem_alert(payload)

        redis = getattr(request.app.state, "redis", None)
        deduped = False
        if redis and redis.is_connected and canonical.dedup_key:
            is_dup = await redis.get_dedup(canonical.dedup_key)
            if is_dup:
                deduped = True
            else:
                await redis.set_dedup(canonical.dedup_key, ttl=3600)

        if not deduped:
            producer = getattr(request.app.state, "kafka_producer", None)
            if producer and producer.is_connected:
                await producer.send(
                    topic=request.app.state.settings.kafka_topic_alerts_raw,
                    value=canonical.model_dump(mode="json"),
                    key=canonical.alert_id,
                )

        results.append(
            IngestionResult(
                accepted=True,
                alert_id=canonical.alert_id,
                tenant_id=tenant_id,
                source=canonical.source,
                deduplicated=deduped,
            )
        )

    return results


@router.get("/siem/vendors", summary="List supported SIEM vendors")
async def list_vendors() -> dict:
    """Return the list of supported SIEM vendor mappings."""
    from src.ingestion.normalizer import VENDOR_MAPPINGS

    return {
        "vendors": list(VENDOR_MAPPINGS.keys()),
        "note": "Unsupported vendors will be rejected with a 422 error.",
    }
