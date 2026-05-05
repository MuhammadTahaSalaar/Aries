"""
ARIES — Pydantic v2 schemas for the Summarization pipeline.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class SummarizeMode(str, Enum):
    EXECUTIVE = "executive"
    ANALYST = "analyst"


class SummarizeRequest(BaseModel):
    text: str = Field(..., min_length=10, max_length=100_000, description="Input text to summarize")
    tenant_id: str = Field(default="unknown", description="Tenant isolation key (overridden by X-Tenant-ID header)")
    case_id: str | None = Field(default=None, description="Optional case ID to persist summary")
    mode: SummarizeMode = Field(default=SummarizeMode.EXECUTIVE, description="Summary mode")


class SummarizeResult(BaseModel):
    summary: str = Field(..., description="Generated summary text")
    mode: SummarizeMode
    model_version: str = Field(default="latest")
    processing_ms: int = Field(default=0, description="Inference latency in milliseconds")
    cached: bool = Field(default=False, description="True if served from cache")
    input_tokens: int = Field(default=0, description="Number of input tokens processed")
    output_tokens: int = Field(default=0, description="Number of output tokens generated")
