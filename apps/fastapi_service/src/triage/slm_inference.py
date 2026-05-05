"""
ARIES — Triage SLM inference engine.

Uses a local Small Language Model (e.g., Phi-3 or Llama-3) via llama-cpp-python
to perform semantic triage on the alert narrative.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from typing import Any

from src.shared.config import ServiceSettings
from src.shared.logging import get_logger
from src.triage.inference import compute_risk_score
from src.triage.schemas import IncidentGrade, TriageResult

log = get_logger("triage_slm_inference")

try:
    from llama_cpp import Llama
except ImportError:
    Llama = None
    log.warning("llama_cpp not installed. SLM inference will use a mock implementation.")


# Global singleton for the SLM and a lock that serializes all llm() calls.
# GGML's CUDA scratch pool is a LIFO allocator — it is NOT thread-safe.
# Concurrent llm() calls from multiple run_in_executor threads corrupt the pool
# and trigger the GGML_ASSERT in ggml-cuda.cu. The lock ensures only one
# inference call touches the Llama instance at a time.
_slm_instance: Llama | None = None
_slm_lock = threading.Lock()

def get_slm(model_path: str) -> Any:
    global _slm_instance
    if Llama is None:
        return None  # Return mock
    if _slm_instance is None:
        import os
        n_threads = min(8, os.cpu_count() or 4)
        log.info("Loading SLM model", path=model_path, n_threads=n_threads)
        _slm_instance = Llama(
            model_path=model_path,
            n_ctx=4096,
            n_gpu_layers=-1,  # Offload all layers to GPU (requires CUDA build)
            n_threads=n_threads,
            n_batch=2048,
            verbose=False
        )
    return _slm_instance


def _build_triage_prompt(alert_dict: dict[str, Any]) -> str:
    """Constructs the prompt for the SLM to perform triage."""
    alert_json = json.dumps(alert_dict, indent=2, default=str)
    prompt = f"""<|system|>
You are a senior SOC analyst. Your task is to evaluate the following SIEM alert.
Classify it as TruePositive or FalsePositive, and infer the severity from context if not clearly stated.
A TruePositive is a real security threat that requires investigation.
A FalsePositive is a benign event, misconfiguration, or routine activity that poses no threat.
Respond ONLY with a valid JSON object containing exactly three keys:
1. "grade": A string, either "TruePositive" or "FalsePositive".
2. "confidence": A float between 0.0 and 1.0 representing your confidence.
3. "severity": A string, one of "Low", "Medium", "High", or "Critical", inferred from the alert context.
Do not output any markdown formatting, explanation, or other text.
<|user|>
Alert Context:
{alert_json}
<|assistant|>
"""
    return prompt


def _run_slm_inference_sync(model_path: str, alert_dict: dict[str, Any]) -> dict[str, Any]:
    prompt = _build_triage_prompt(alert_dict)
    llm = get_slm(model_path)
    
    if llm is None:
        # Mock implementation for local testing without packages
        log.info("Using mock SLM triage")
        time.sleep(0.5)
        title = alert_dict.get("normalized_title", "").lower()
        grade = "TruePositive" if any(k in title for k in ["brute force", "injection", "malware", "unauthorized"]) else "FalsePositive"
        severity = "High" if grade == "TruePositive" else "Low"
        return {"grade": grade, "confidence": 0.95 if grade == "TruePositive" else 0.85, "severity": severity}

    with _slm_lock:
        response = llm(
            prompt,
            max_tokens=50,
            stop=["<|end|>"],
            temperature=0.1
        )
    
    try:
        text_output = response["choices"][0]["text"].strip()
        # Attempt to parse JSON
        if text_output.startswith("```json"):
            text_output = text_output.replace("```json", "").replace("```", "").strip()
        result = json.loads(text_output)
        # Normalise severity from SLM in case it returns wrong case
        if "severity" in result and isinstance(result["severity"], str):
            result["severity"] = result["severity"].capitalize()
        return result
    except Exception as e:
        log.error("SLM JSON parse failed", error=str(e), output=response.get("choices", [{}])[0].get("text", ""))
        return {"grade": "FalsePositive", "confidence": 0.5}


async def run_triage_inference_slm(
    alert_dict: dict[str, Any],
    alert_id: str,
    tenant_id: str,
    asset_criticality: float | None,
    behavioral_score: float,
    settings: ServiceSettings,
    suspicion_level: str | None = None,
    asset_criticality_fallback: float = 0.50,
) -> TriageResult:
    """
    Run SLM triage inference asynchronously.
    """
    start = time.perf_counter()
    loop = asyncio.get_running_loop()
    
    slm_result = await loop.run_in_executor(
        None, _run_slm_inference_sync, settings.slm_model_path, alert_dict
    )

    grade_str = slm_result.get("grade", "FalsePositive")
    confidence = float(slm_result.get("confidence", 0.5))
    # If the caller had no explicit severity, use what the SLM inferred
    slm_severity = slm_result.get("severity")
    
    # Map back to our standard IncidentGrade
    if grade_str == "TruePositive":
        grade = IncidentGrade.TRUE_POSITIVE
        ml_score = confidence
    else:
        grade = IncidentGrade.FALSE_POSITIVE
        ml_score = 1.0 - confidence  # Lower ml_score for FP

    # If SLM returned a severity and the caller did not provide one explicitly
    # (indicated by the caller passing severity_override=None), recompute
    # asset_criticality using the SLM-inferred severity so the final risk score
    # reflects the model's contextual understanding.
    if slm_severity and asset_criticality is None:
        _sev_map = {"Critical": 0.95, "High": 0.75, "Medium": 0.50, "Low": 0.25}
        asset_criticality = _sev_map.get(slm_severity, asset_criticality_fallback)
        log.info("asset_criticality_from_slm_severity", severity=slm_severity, asset_criticality=asset_criticality)

    risk_score = compute_risk_score(ml_score, asset_criticality, behavioral_score, settings)
    
    # Heuristic safety net
    if (
        risk_score >= 50 and grade != IncidentGrade.TRUE_POSITIVE
    ):
        grade = IncidentGrade.TRUE_POSITIVE

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
        "triage_slm_inference_complete",
        alert_id=alert_id,
        tenant_id=tenant_id,
        ml_score=result.ml_score,
        risk_score=result.risk_score,
        grade=result.incident_grade.value,
        auto_closed=result.auto_closed,
        latency_ms=elapsed_ms,
    )

    return result
