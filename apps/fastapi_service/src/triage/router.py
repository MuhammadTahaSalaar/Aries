"""
ARIES — Triage FastAPI router.

Provides:
  POST /triage/score — synchronous single-alert scoring
  GET  /triage/health — pipeline health check
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from src.shared.config import get_settings, ServiceSettings
from src.shared.dependencies import get_tenant_id
from src.shared.exceptions import ModelNotLoadedError
from src.triage.feature_engineering import extract_features
from src.triage.inference import run_triage_inference
from src.triage.slm_inference import run_triage_inference_slm
from src.triage.schemas import CanonicalAlert, TriageResult

router = APIRouter()


@router.post(
    "/score",
    response_model=TriageResult,
    summary="Score a single alert",
    description="Run ML triage on a canonical alert and return ml_score + risk_score.",
)
async def score_alert(
    alert: CanonicalAlert,
    request: Request,
    tenant_id: str = Depends(get_tenant_id),
) -> TriageResult:
    """Score a single alert using the triage ONNX model."""
    model_store = request.app.state.model_store
    settings = get_settings()

    if not settings.use_slm and not model_store.triage_loaded:
        raise ModelNotLoadedError("triage").to_http()

    # Override tenant_id from header
    alert_dict = alert.model_dump()
    alert_dict["tenant_id"] = tenant_id

    features = extract_features(alert_dict, encoder=model_store.triage_encoder)

    # Derive enrichment scores from the alert context
    severity = alert.severity.value if alert.severity else "Medium"
    title_lower = (alert.normalized_title or "").lower()

    # Asset criticality from severity + entity context
    asset_base = {"Critical": 0.95, "High": 0.75, "Medium": 0.50, "Low": 0.25}.get(severity, 0.50)
    if any(kw in title_lower for kw in ["domain controller", "exchange", "database", "admin"]):
        asset_base = min(asset_base + 0.15, 1.0)
    asset_criticality = round(asset_base, 2)

    # Behavioral score from suspicion level + attack pattern
    susp = alert.suspicion_level or "Medium"
    behav_base = {"Critical": 0.95, "High": 0.70, "Medium": 0.45, "Low": 0.20}.get(susp, 0.45)
    if any(kw in title_lower for kw in ["brute force", "lateral", "exfiltration"]):
        behav_base = min(behav_base + 0.10, 1.0)
    behavioral_score = round(behav_base, 2)

    if settings.use_slm:
        result = await run_triage_inference_slm(
            alert_dict=alert_dict,
            alert_id=alert.alert_id,
            tenant_id=tenant_id,
            asset_criticality=asset_criticality,
            behavioral_score=behavioral_score,
            settings=settings,
            suspicion_level=alert.suspicion_level,
        )
    else:
        result = await run_triage_inference(
            session=model_store.triage_session,
            features=features,
            alert_id=alert.alert_id,
            tenant_id=tenant_id,
            asset_criticality=asset_criticality,
            behavioral_score=behavioral_score,
            settings=settings,
            suspicion_level=alert.suspicion_level,
        )

    return result


@router.get("/health", summary="Triage pipeline health")
async def triage_health(request: Request) -> dict:
    model_store = request.app.state.model_store
    return {
        "pipeline": "triage",
        "model_loaded": model_store.triage_loaded,
        "metadata": model_store.triage_metadata,
    }
