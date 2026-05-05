"""
ARIES — Triage ONNX inference engine.

Runs the XGBoost ONNX model in a thread-pool executor to avoid blocking
the async event loop. Includes heuristic fallback.
"""

from __future__ import annotations

import asyncio
import time
from functools import partial
from typing import Any

import numpy as np
import onnxruntime as ort

from src.shared.config import ServiceSettings
from src.shared.logging import get_logger
from src.triage.schemas import IncidentGrade, TriageResult

log = get_logger("triage_inference")

# Grade-index → label mapping for the 3-class XGBoost model
# Index matches target_map: {BenignPositive: 0, FalsePositive: 1, TruePositive: 2}
GRADE_MAP = {
    0: IncidentGrade.BENIGN_POSITIVE,
    1: IncidentGrade.FALSE_POSITIVE,
    2: IncidentGrade.TRUE_POSITIVE,
}

# Heuristic fallback scores keyed by SuspicionLevel
HEURISTIC_SCORES: dict[str, float] = {
    "Critical": 0.90,
    "High": 0.70,
    "Medium": 0.40,
    "Low": 0.10,
}


def _run_onnx_inference(
    session: ort.InferenceSession, features: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Synchronous ONNX inference (called via run_in_executor).
    Returns (predicted_labels, probabilities).
    """
    input_name = session.get_inputs()[0].name
    results = session.run(None, {input_name: features.astype(np.float32)})
    labels = results[0]  # shape: (batch,)
    probabilities = results[1]  # shape: (batch, n_classes) — list of dicts for XGBoost
    return labels, probabilities


def compute_risk_score(
    ml_score: float,
    asset_criticality: float,
    behavioral_score: float,
    settings: ServiceSettings,
) -> float:
    """
    Risk Prioritization Formula (REQ-03):
    risk_score = clamp(w_ml * ml_score + w_asset * asset + w_behav * behavior, 0, 1) * 100
    """
    raw = (
        settings.triage_weight_ml * ml_score
        + settings.triage_weight_asset * asset_criticality
        + settings.triage_weight_behavior * behavioral_score
    )
    return round(max(0.0, min(1.0, raw)) * 100, 2)


def heuristic_triage(suspicion_level: str | None) -> tuple[float, IncidentGrade]:
    """Fallback when ONNX session is unavailable."""
    level = suspicion_level or "Medium"
    score = HEURISTIC_SCORES.get(level, 0.40)
    grade = IncidentGrade.TRUE_POSITIVE if score >= 0.5 else IncidentGrade.FALSE_POSITIVE
    return score, grade


async def run_triage_inference(
    session: ort.InferenceSession | None,
    features: list[float],
    alert_id: str,
    tenant_id: str,
    asset_criticality: float,
    behavioral_score: float,
    settings: ServiceSettings,
    suspicion_level: str | None = None,
) -> TriageResult:
    """
    Run triage inference asynchronously. Falls back to heuristic if session is None.
    ONNX calls are dispatched to a thread-pool executor.
    """
    start = time.perf_counter()

    if session is None:
        ml_score, grade = heuristic_triage(suspicion_level)
        log.warning("triage_heuristic_fallback", alert_id=alert_id, ml_score=ml_score)
    else:
        loop = asyncio.get_running_loop()
        feat_array = np.array([features], dtype=np.float32)
        labels, probabilities = await loop.run_in_executor(
            None, partial(_run_onnx_inference, session, feat_array)
        )

        predicted_class = int(labels[0])
        grade = GRADE_MAP.get(predicted_class, IncidentGrade.FALSE_POSITIVE)

        # Extract TP probability (class index 2)
        if isinstance(probabilities[0], dict):
            ml_score = float(probabilities[0].get(2, 0.0))
        elif isinstance(probabilities[0], (list, np.ndarray)):
            probs = probabilities[0]
            ml_score = float(probs[2]) if len(probs) > 2 else float(probs[-1])
        else:
            ml_score = 0.5

    risk_score = compute_risk_score(ml_score, asset_criticality, behavioral_score, settings)

    # Override grade when risk_score contradicts ML prediction.
    # The XGBoost model was trained on Microsoft GUIDE data and unseen
    # categorical values from other SIEMs can collapse to the encoder's
    # prior mean, making the model under-react to clearly risky alerts.
    if (
        risk_score >= 45
        and asset_criticality >= 0.9
        and behavioral_score >= 0.75
        and grade != IncidentGrade.TRUE_POSITIVE
    ):
        grade = IncidentGrade.TRUE_POSITIVE
    elif risk_score >= 50 and grade != IncidentGrade.TRUE_POSITIVE:
        grade = IncidentGrade.TRUE_POSITIVE
    elif risk_score >= 35 and grade == IncidentGrade.BENIGN_POSITIVE:
        grade = IncidentGrade.FALSE_POSITIVE  # uncertain, upgrade from benign

    auto_closed = ml_score < settings.auto_close_threshold and risk_score < 35

    elapsed_ms = int((time.perf_counter() - start) * 1000)

    result = TriageResult(
        alert_id=alert_id,
        tenant_id=tenant_id,
        ml_score=round(ml_score, 6),
        incident_grade=grade,
        risk_score=risk_score,
        auto_closed=auto_closed,
        processing_ms=elapsed_ms,
        asset_criticality=round(asset_criticality, 4),
        behavioral_score=round(behavioral_score, 4),
    )

    log.info(
        "triage_inference_complete",
        alert_id=alert_id,
        tenant_id=tenant_id,
        ml_score=result.ml_score,
        risk_score=result.risk_score,
        grade=result.incident_grade.value,
        auto_closed=result.auto_closed,
        latency_ms=elapsed_ms,
    )

    return result
