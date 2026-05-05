"""
ARIES — FastAPI dependency injection helpers.

Provides Depends()-compatible functions for extracting tenant_id,
and accessing shared resources like DB, Redis, Kafka.
"""

from __future__ import annotations

from typing import Any

from fastapi import Depends, Header, HTTPException, Request, status


async def get_tenant_id(
    x_tenant_id: str = Header(..., alias="X-Tenant-ID", description="Tenant isolation ID"),
) -> str:
    """Extract and validate Tenant ID from request header.

    In production this would validate against a JWT claim.
    For now, we require an explicit header for all requests.
    """
    if not x_tenant_id or len(x_tenant_id) < 1:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="X-Tenant-ID header is required",
        )
    return x_tenant_id


def get_db(request: Request) -> Any:
    """Retrieve the Database instance from app state."""
    return request.app.state.db


def get_redis(request: Request) -> Any:
    """Retrieve the RedisClient instance from app state."""
    return request.app.state.redis


def get_kafka_producer(request: Request) -> Any:
    """Retrieve the KafkaProducer instance from app state."""
    return request.app.state.kafka_producer
