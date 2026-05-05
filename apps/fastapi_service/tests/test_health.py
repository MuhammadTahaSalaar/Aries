"""
ARIES — Tests for health and readiness endpoints.
"""

from __future__ import annotations

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_health_endpoint(async_client: AsyncClient) -> None:
    """Health endpoint should return 200 with model status."""
    resp = await async_client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "models_loaded" in data
    assert isinstance(data["models_loaded"], dict)


@pytest.mark.asyncio
async def test_readiness_endpoint(async_client: AsyncClient) -> None:
    """Readiness endpoint should report component status."""
    resp = await async_client.get("/ready")
    assert resp.status_code == 200
    data = resp.json()
    assert "ready" in data
    assert "models" in data
    assert "kafka_connected" in data
    assert "db_connected" in data
    assert "redis_connected" in data


@pytest.mark.asyncio
async def test_missing_tenant_id_header(async_client: AsyncClient) -> None:
    """Endpoints requiring X-Tenant-ID should reject requests without it."""
    resp = await async_client.post(
        "/triage/score",
        json={
            "alert_id": "test-1",
            "tenant_id": "t-1",
            "timestamp": "2026-01-01T00:00:00Z",
            "source": "test",
            "normalized_title": "Test",
        },
    )
    assert resp.status_code == 422  # Missing required header


@pytest.mark.asyncio
async def test_docs_endpoint(async_client: AsyncClient) -> None:
    """OpenAPI docs should be accessible."""
    resp = await async_client.get("/docs")
    assert resp.status_code == 200
