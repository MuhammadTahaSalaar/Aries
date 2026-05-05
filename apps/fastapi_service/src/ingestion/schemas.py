"""
ARIES — Pydantic v2 schemas for the SIEM Ingestion pipeline.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class SIEMRawPayload(BaseModel):
    """Raw JSON payload received from a SIEM/EDR webhook."""

    vendor: str | None = Field(default=None, description="SIEM vendor identifier (e.g., 'wazuh', 'splunk', 'elastic_siem'). Can also be passed as ?vendor= query param")
    tenant_id: str = Field(default="unknown", description="Tenant isolation key (overridden by X-Tenant-ID header)")
    raw: dict[str, Any] | None = Field(default=None, description="Raw vendor-specific alert JSON. If omitted, the entire request body is treated as the raw payload")
    timestamp: datetime | None = Field(default=None, description="Optional override timestamp")


class IngestionResult(BaseModel):
    """Response after ingesting a SIEM alert."""

    accepted: bool = True
    alert_id: str = Field(..., description="Assigned canonical alert ID")
    tenant_id: str
    source: str
    kafka_topic: str = "alerts.raw"
    deduplicated: bool = Field(default=False, description="True if this was a duplicate, skipped")


class NormalizationMapping(BaseModel):
    """Defines how to map a vendor's JSON fields to the canonical schema."""

    vendor: str
    field_map: dict[str, str] = Field(
        ...,
        description="Map of canonical field name -> JSONPath expression in vendor payload",
    )
    severity_map: dict[str, str] = Field(
        default_factory=dict,
        description="Map of vendor severity string -> canonical Severity enum value",
    )
    default_source: str = Field(default="unknown")
