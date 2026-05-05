"""
ARIES — MLflow Model Resolver.

Queries the MLflow REST API at startup to discover the latest trained model
artifacts for each pipeline (triage, NER, summarizer). Returns the S3 keys
so the model_loader can download them from the same MinIO bucket that MLflow
uses for artifact storage.

Artifact structure (set by MLOps/migrate_to_mlflow.py):
  - Triage:      <run>/artifacts/onnx/triage.onnx
  - NER:         <run>/artifacts/onnx/ner.opt.onnx  +  <run>/artifacts/*.json (tokenizer)
  - Summarizer:  <run>/artifacts/onnx/summarizer/encoder.onnx  +  decoder.onnx
                 + <run>/artifacts/*.json (tokenizer)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import httpx

from src.shared.logging import get_logger

log = get_logger("mlflow_resolver")

# Experiment names as defined in MLOps/migrate_to_mlflow.py
EXPERIMENT_TRIAGE = "aries/triage-classifier"
EXPERIMENT_NER = "aries/ner"
EXPERIMENT_SUMMARIZER = "aries/bart-summarizer"


@dataclass
class ResolvedArtifacts:
    """S3 keys for all model artifacts resolved from MLflow."""

    # Triage
    triage_onnx_key: str | None = None
    triage_encoder_key: str | None = None

    # NER
    ner_onnx_key: str | None = None
    ner_tokenizer_prefix: str | None = None

    # Summarizer
    summarizer_encoder_key: str | None = None
    summarizer_decoder_key: str | None = None
    summarizer_tokenizer_prefix: str | None = None

    # The S3 bucket containing MLflow artifacts
    bucket: str = "mlflow-bucket"

    # Metadata: which run IDs were resolved
    resolved_runs: dict[str, str] = field(default_factory=dict)


async def resolve_latest_models(
    mlflow_tracking_uri: str,
    timeout: float = 15.0,
) -> ResolvedArtifacts | None:
    """
    Query MLflow REST API to find the latest successful run for each
    experiment and build the S3 artifact keys.

    Returns None if MLflow is unreachable.
    """
    artifacts = ResolvedArtifacts()

    try:
        async with httpx.AsyncClient(
            base_url=mlflow_tracking_uri,
            timeout=timeout,
        ) as client:
            # Resolve each pipeline
            await _resolve_triage(client, artifacts)
            await _resolve_ner(client, artifacts)
            await _resolve_summarizer(client, artifacts)

    except httpx.ConnectError:
        log.warning("mlflow_unreachable", uri=mlflow_tracking_uri)
        return None
    except Exception:
        log.exception("mlflow_resolution_failed")
        return None

    resolved_count = sum(
        1
        for v in [
            artifacts.triage_onnx_key,
            artifacts.ner_onnx_key,
            artifacts.summarizer_encoder_key,
        ]
        if v is not None
    )

    if resolved_count == 0:
        log.warning("no_models_resolved_from_mlflow")
        return None

    log.info(
        "mlflow_models_resolved",
        resolved_count=resolved_count,
        runs=artifacts.resolved_runs,
    )
    return artifacts


async def _get_experiment_id(
    client: httpx.AsyncClient, experiment_name: str
) -> str | None:
    """Get experiment ID by name from MLflow REST API."""
    resp = await client.get(
        "/api/2.0/mlflow/experiments/get-by-name",
        params={"experiment_name": experiment_name},
    )
    if resp.status_code != 200:
        log.debug("experiment_not_found", name=experiment_name, status=resp.status_code)
        return None

    data = resp.json()
    return data.get("experiment", {}).get("experiment_id")


async def _get_latest_run(
    client: httpx.AsyncClient, experiment_id: str
) -> dict[str, Any] | None:
    """Search for the latest FINISHED run in an experiment."""
    resp = await client.post(
        "/api/2.0/mlflow/runs/search",
        json={
            "experiment_ids": [experiment_id],
            "filter": "attributes.status = 'FINISHED'",
            "order_by": ["attributes.end_time DESC"],
            "max_results": 1,
        },
    )
    if resp.status_code != 200:
        return None

    runs = resp.json().get("runs", [])
    return runs[0] if runs else None


async def _list_artifacts(
    client: httpx.AsyncClient, run_id: str, path: str = ""
) -> list[dict[str, Any]]:
    """List artifacts under a path in a run."""
    params: dict[str, str] = {"run_id": run_id}
    if path:
        params["path"] = path

    resp = await client.get("/api/2.0/mlflow/artifacts/list", params=params)
    if resp.status_code != 200:
        return []

    return resp.json().get("files", [])


def _artifact_s3_key(run: dict[str, Any], artifact_path: str) -> str:
    """
    Build the S3 key for an artifact.

    MLflow stores artifacts in:
      s3://mlflow-bucket/<experiment_id>/<run_id>/artifacts/<artifact_path>
    """
    info = run.get("info", {})
    experiment_id = info.get("experiment_id", "")
    run_id = info.get("run_id", "")
    return f"{experiment_id}/{run_id}/artifacts/{artifact_path}"


def _artifact_s3_prefix(run: dict[str, Any], prefix: str) -> str:
    """Build an S3 prefix for a directory of artifacts."""
    info = run.get("info", {})
    experiment_id = info.get("experiment_id", "")
    run_id = info.get("run_id", "")
    return f"{experiment_id}/{run_id}/artifacts/{prefix}"


async def _resolve_triage(
    client: httpx.AsyncClient, artifacts: ResolvedArtifacts
) -> None:
    """Resolve triage model artifact path."""
    exp_id = await _get_experiment_id(client, EXPERIMENT_TRIAGE)
    if not exp_id:
        log.info("triage_experiment_not_found", name=EXPERIMENT_TRIAGE)
        return

    run = await _get_latest_run(client, exp_id)
    if not run:
        log.info("triage_no_finished_runs")
        return

    run_id = run["info"]["run_id"]
    artifacts.triage_onnx_key = _artifact_s3_key(run, "onnx/triage.onnx")
    artifacts.triage_encoder_key = _artifact_s3_key(run, "triage_model/triage_encoder.pkl")
    artifacts.resolved_runs["triage"] = run_id

    log.info("triage_model_resolved", run_id=run_id, key=artifacts.triage_onnx_key)


async def _resolve_ner(
    client: httpx.AsyncClient, artifacts: ResolvedArtifacts
) -> None:
    """Resolve NER model + tokenizer artifact paths."""
    exp_id = await _get_experiment_id(client, EXPERIMENT_NER)
    if not exp_id:
        log.info("ner_experiment_not_found", name=EXPERIMENT_NER)
        return

    run = await _get_latest_run(client, exp_id)
    if not run:
        log.info("ner_no_finished_runs")
        return

    run_id = run["info"]["run_id"]

    # ONNX model
    artifacts.ner_onnx_key = _artifact_s3_key(run, "onnx/ner.opt.onnx")

    # Tokenizer files are logged at the run artifact root
    # (tokenizer.json, tokenizer_config.json, vocab.json, merges.txt, etc.)
    # We download them from the root prefix, filtering for tokenizer files
    artifacts.ner_tokenizer_prefix = _artifact_s3_prefix(run, "")
    artifacts.resolved_runs["ner"] = run_id

    log.info("ner_model_resolved", run_id=run_id, key=artifacts.ner_onnx_key)


async def _resolve_summarizer(
    client: httpx.AsyncClient, artifacts: ResolvedArtifacts
) -> None:
    """Resolve summarizer encoder + decoder + tokenizer artifact paths."""
    exp_id = await _get_experiment_id(client, EXPERIMENT_SUMMARIZER)
    if not exp_id:
        log.info("summarizer_experiment_not_found", name=EXPERIMENT_SUMMARIZER)
        return

    run = await _get_latest_run(client, exp_id)
    if not run:
        log.info("summarizer_no_finished_runs")
        return

    run_id = run["info"]["run_id"]

    # ONNX models under onnx/summarizer/
    artifacts.summarizer_encoder_key = _artifact_s3_key(
        run, "onnx/summarizer/encoder.onnx"
    )
    artifacts.summarizer_decoder_key = _artifact_s3_key(
        run, "onnx/summarizer/decoder.onnx"
    )

    # Tokenizer files at the run artifact root
    artifacts.summarizer_tokenizer_prefix = _artifact_s3_prefix(run, "")
    artifacts.resolved_runs["summarizer"] = run_id

    log.info(
        "summarizer_model_resolved",
        run_id=run_id,
        encoder=artifacts.summarizer_encoder_key,
        decoder=artifacts.summarizer_decoder_key,
    )
