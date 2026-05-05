"""
ARIES — Async Redis client for caching.

Wraps redis.asyncio with namespaced keys (tenant_id prefix).
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import redis.asyncio as aioredis

from src.shared.config import ServiceSettings
from src.shared.logging import get_logger

log = get_logger("redis")


class RedisClient:
    """Async Redis wrapper with tenant-namespaced caching."""

    def __init__(self, settings: ServiceSettings) -> None:
        self._url = settings.redis_url
        self._client: aioredis.Redis | None = None

    async def connect(self) -> None:
        self._client = aioredis.from_url(
            self._url,
            decode_responses=True,
            socket_timeout=5,
            socket_connect_timeout=5,
        )
        await self._client.ping()
        log.info("redis_connected", url=self._url)

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            log.info("redis_disconnected")

    @property
    def client(self) -> aioredis.Redis:
        if self._client is None:
            raise RuntimeError("Redis not connected; call connect() first")
        return self._client

    # ── Cache helpers ─────────────────────────────────────────────────

    @staticmethod
    def _hash_key(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]

    async def get_cached(self, tenant_id: str, namespace: str, text: str) -> dict[str, Any] | None:
        """Retrieve a cached JSON result."""
        key = f"{tenant_id}:{namespace}:{self._hash_key(text)}"
        raw = await self.client.get(key)
        if raw:
            log.debug("cache_hit", key=key)
            return json.loads(raw)
        return None

    async def set_cached(
        self, tenant_id: str, namespace: str, text: str, value: dict[str, Any], ttl: int
    ) -> None:
        """Store a JSON result with TTL."""
        key = f"{tenant_id}:{namespace}:{self._hash_key(text)}"
        await self.client.set(key, json.dumps(value, default=str), ex=ttl)
        log.debug("cache_set", key=key, ttl=ttl)

    async def get_dedup(self, dedup_key: str) -> bool:
        """Check if a deduplication key exists."""
        return bool(await self.client.exists(dedup_key))

    async def set_dedup(self, dedup_key: str, ttl: int = 3600) -> None:
        """Set a deduplication key with TTL."""
        await self.client.set(dedup_key, "1", ex=ttl)

    @property
    def is_connected(self) -> bool:
        return self._client is not None
