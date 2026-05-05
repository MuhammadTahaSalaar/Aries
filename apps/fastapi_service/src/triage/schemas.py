"""
ARIES — Pydantic v2 schemas for the Triage pipeline.

Covers: AlertFeatures (input), TriageResult (output), and the
Canonical/Enriched Alert schemas used across Kafka topics.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, model_validator


# ── Enums ─────────────────────────────────────────────────────────────

class AlertStatus(str, Enum):
    NEW = "New"
    TRIAGED = "Triaged"
    CLOSED_FP = "Closed_FP"
    ESCALATED = "Escalated"


class IncidentGrade(str, Enum):
    TRUE_POSITIVE = "TruePositive"
    FALSE_POSITIVE = "FalsePositive"
    BENIGN_POSITIVE = "BenignPositive"


class Severity(str, Enum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    CRITICAL = "Critical"


# ── Canonical Alert (Kafka: alerts.raw) ──────────────────────────────

class CanonicalAlert(BaseModel):
    """
    The normalised alert schema used on the alerts.raw Kafka topic.
    Every vendor-specific payload is mapped to this schema by the
    ingestion service before being placed on the bus.
    """

    alert_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique alert identifier (UUID or vendor ID)")
    tenant_id: str = Field(default="unknown", description="Tenant isolation key (overridden by X-Tenant-ID header)")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Alert creation time")
    source: str = Field(default="unknown", description="Originating system (e.g., 'wazuh', 'splunk', 'crowdstrike')")
    normalized_title: str = Field(default="Unknown Alert", description="Human-readable alert title")
    severity: Severity = Field(default=Severity.MEDIUM, description="Alert severity level")
    raw_data: dict[str, Any] = Field(default_factory=dict, description="Original unmodified payload")

    # MITRE ATT&CK mapping
    mitre_tactic: str | None = Field(default=None, description="E.g., 'Initial Access'")
    mitre_technique: str | None = Field(default=None, description="E.g., 'T1566.001'")

    # Entity context for triage feature engineering
    entity_type: str | None = Field(default=None, description="E.g., 'Process', 'File', 'IP'")
    device_name: str | None = None
    ip_address: str | None = None
    user_name: str | None = None
    file_hash: str | None = None
    url: str | None = None
    domain: str | None = None
    category: str | None = None
    threat_family: str | None = None
    suspicion_level: str | None = None

    # Deduplication
    dedup_key: str | None = Field(default=None, description="Content hash for deduplication")

    @model_validator(mode="before")
    @classmethod
    def _extra_to_raw_data(cls, values: Any) -> Any:
        """Move unrecognised top-level fields into raw_data.

        This allows vendor-specific payloads (e.g. AlertTitle, Category) sent
        at the top level to be preserved for feature engineering.
        """
        if not isinstance(values, dict):
            return values
        known = {
            "alert_id", "tenant_id", "timestamp", "source",
            "normalized_title", "severity", "raw_data",
            "mitre_tactic", "mitre_technique", "entity_type",
            "device_name", "ip_address", "user_name", "file_hash",
            "url", "domain", "category", "threat_family",
            "suspicion_level", "dedup_key",
        }
        raw = dict(values.get("raw_data") or {})
        for k in list(values):
            if k not in known:
                raw[k] = values.pop(k)
        values["raw_data"] = raw
        return values


# ── Alert Features (input to ONNX triage model) ─────────────────────

class AlertFeatures(BaseModel):
    """
    Feature vector for the XGBoost triage classifier.
    These are produced by the feature engineering step
    from a CanonicalAlert.
    """

    alert_id: str
    tenant_id: str
    features: list[float] = Field(..., description="Float feature vector (n_features=49)")


# ── Triage Result ────────────────────────────────────────────────────

class TriageResult(BaseModel):
    """Output produced by the triage ONNX model."""

    alert_id: str
    tenant_id: str
    ml_score: float = Field(..., ge=0.0, le=1.0, description="True-positive likelihood")
    incident_grade: IncidentGrade
    risk_score: float = Field(..., ge=0.0, le=100.0, description="Final composite risk score")
    auto_closed: bool = Field(default=False, description="True if auto-closed due to low ml_score")
    model_version: str = Field(default="latest")
    processing_ms: int = Field(default=0, description="Inference latency in milliseconds")
    asset_criticality: float = Field(default=0.5, ge=0.0, le=1.0, description="Asset criticality score")
    behavioral_score: float = Field(default=0.5, ge=0.0, le=1.0, description="Behavioral anomaly score")


# ── Enriched Alert (Kafka: alerts.enriched) ──────────────────────────

class EnrichedAlert(BaseModel):
    """
    The fully enriched alert placed on alerts.enriched.
    Consumed by the Orchestration Engine, NER pipeline, and Dashboard.
    """

    alert_id: str
    tenant_id: str
    timestamp: datetime
    source: str
    normalized_title: str
    severity: Severity
    raw_data: dict[str, Any] = Field(default_factory=dict)

    # ML triage results
    ml_score: float = Field(..., ge=0.0, le=1.0)
    risk_score: float = Field(..., ge=0.0, le=100.0)
    incident_grade: IncidentGrade
    auto_closed: bool = False
    model_version: str = "latest"

    # MITRE
    mitre_tactic: str | None = None
    mitre_technique: str | None = None

    # Context
    entity_type: str | None = None
    device_name: str | None = None
    ip_address: str | None = None
    user_name: str | None = None
    category: str | None = None

    # IOCs (populated after NER pass)
    ioc_ids: list[str] = Field(default_factory=list)

    # Asset enrichment
    asset_criticality: float = Field(default=0.5, ge=0.0, le=1.0)
    behavioral_score: float = Field(default=0.5, ge=0.0, le=1.0)


# ── Health / Readiness ───────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str = "ok"
    service: str = "aries-fastapi-service"
    version: str = "1.0.0"
    models_loaded: dict[str, bool] = Field(default_factory=dict)


class ReadinessResponse(BaseModel):
    ready: bool
    kafka_connected: bool
    db_connected: bool
    redis_connected: bool
    models: dict[str, bool]
