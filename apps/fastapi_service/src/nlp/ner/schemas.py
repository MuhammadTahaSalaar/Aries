"""
ARIES — Pydantic v2 schemas for the NER / IOC Extraction pipeline.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class IOCType(str, Enum):
    IP_ADDRESS = "IP_Address"
    FILE_HASH = "File_Hash"
    DOMAIN = "Domain"
    URL = "URL"
    EMAIL_ADDRESS = "Email_Address"
    CVE_ID = "CVE_ID"
    UNKNOWN = "Unknown"


class EntityLabel(str, Enum):
    MALWARE = "Malware"
    TOOL = "Tool"
    INDICATOR = "Indicator"
    SYSTEM = "System"
    VULNERABILITY = "Vulnerability"
    ORGANIZATION = "Organization"
    O = "O"


class IOCEntity(BaseModel):
    """A single extracted entity / IOC from NER inference."""

    text: str = Field(..., description="The raw text span")
    label: EntityLabel = Field(..., description="NER entity label")
    start: int = Field(..., ge=0, description="Start character offset")
    end: int = Field(..., ge=0, description="End character offset")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    ioc_type: IOCType = Field(default=IOCType.UNKNOWN, description="Validated IOC type")
    ioc_validated: bool = Field(default=False, description="True if regex-validated as IOC")


class SecurityEvent(BaseModel):
    """A CASIE-style security event inferred from NER output."""

    event_type: str = Field(..., description="E.g., Ransom, Phishing, Databreach")
    trigger: str = Field(..., description="Trigger phrase")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class NERRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=50_000, description="Input text for NER")
    tenant_id: str = Field(default="unknown", description="Tenant isolation key (overridden by X-Tenant-ID header)")


class NERBatchRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=32)
    tenant_id: str = Field(default="unknown", description="Tenant isolation key (overridden by X-Tenant-ID header)")


class NERResult(BaseModel):
    entities: list[IOCEntity] = Field(default_factory=list)
    events: list[SecurityEvent] = Field(default_factory=list)
    processing_ms: int = Field(default=0, description="Inference latency in milliseconds")
    cached: bool = Field(default=False, description="True if result was served from cache")
