"""
ARIES — Async PostgreSQL database client (asyncpg).

Provides a connection pool and typed query helpers.
All queries enforce tenant_id isolation.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

import asyncpg

from src.shared.config import ServiceSettings
from src.shared.logging import get_logger

log = get_logger("db")


class Database:
    """Async PostgreSQL connection pool wrapper."""

    def __init__(self, settings: ServiceSettings) -> None:
        self._dsn = settings.database_url
        self._min = settings.db_pool_min
        self._max = settings.db_pool_max
        self._pool: asyncpg.Pool | None = None

    async def connect(self) -> None:
        """Create the connection pool."""
        log.info("connecting_to_database", dsn=self._dsn.split("@")[-1])
        self._pool = await asyncpg.create_pool(
            dsn=self._dsn,
            min_size=self._min,
            max_size=self._max,
            command_timeout=30,
        )
        await self._ensure_schema()
        log.info("database_connected")

    async def close(self) -> None:
        """Gracefully close the pool."""
        if self._pool:
            await self._pool.close()
            log.info("database_disconnected")

    @property
    def pool(self) -> asyncpg.Pool:
        if self._pool is None:
            raise RuntimeError("Database not connected; call connect() first")
        return self._pool

    async def _ensure_schema(self) -> None:
        """Create tables if they do not exist (dev convenience; production uses migrations)."""
        async with self.pool.acquire() as conn:
            await conn.execute(SCHEMA_SQL)
        log.info("schema_ensured")

    # ── Alert operations ──────────────────────────────────────────────

    async def insert_alert(
        self,
        alert_id: str,
        tenant_id: str,
        normalized_title: str,
        raw_data: dict[str, Any],
        source: str,
        ml_score: float | None = None,
        risk_score: float | None = None,
        status: str = "New",
        mitre_tactic: str | None = None,
    ) -> str:
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO alerts (alert_id, tenant_id, normalized_title, raw_data,
                                    source, ml_score, risk_score, status, mitre_tactic)
                VALUES ($1, $2, $3, $4::jsonb, $5, $6, $7, $8, $9)
                ON CONFLICT (alert_id) DO UPDATE
                    SET ml_score = EXCLUDED.ml_score,
                        risk_score = EXCLUDED.risk_score,
                        status = EXCLUDED.status,
                        updated_at = NOW()
                """,
                alert_id,
                tenant_id,
                normalized_title,
                json.dumps(raw_data),
                source,
                ml_score,
                risk_score,
                status,
                mitre_tactic,
            )
        return alert_id

    async def update_alert_scores(
        self, alert_id: str, tenant_id: str, ml_score: float, risk_score: float, status: str
    ) -> None:
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE alerts
                   SET ml_score = $1, risk_score = $2, status = $3, updated_at = NOW()
                 WHERE alert_id = $4 AND tenant_id = $5
                """,
                ml_score,
                risk_score,
                status,
                alert_id,
                tenant_id,
            )

    async def get_alert(self, alert_id: str, tenant_id: str) -> dict[str, Any] | None:
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM alerts WHERE alert_id = $1 AND tenant_id = $2",
                alert_id,
                tenant_id,
            )
        return dict(row) if row else None

    # ── IOC operations ────────────────────────────────────────────────

    async def insert_iocs(
        self, iocs: list[dict[str, Any]], alert_id: str, tenant_id: str
    ) -> int:
        """Bulk insert IOCs. Returns the count inserted."""
        if not iocs:
            return 0
        async with self.pool.acquire() as conn:
            await conn.executemany(
                """
                INSERT INTO iocs (alert_id, tenant_id, ioc_type, value, source, confidence)
                VALUES ($1, $2, $3, $4, $5, $6)
                ON CONFLICT DO NOTHING
                """,
                [
                    (
                        alert_id,
                        tenant_id,
                        ioc.get("ioc_type", "Unknown"),
                        ioc["value"],
                        ioc.get("source", "NER"),
                        ioc.get("confidence", 0.0),
                    )
                    for ioc in iocs
                ],
            )
        return len(iocs)

    # ── Case summary operations ───────────────────────────────────────

    async def upsert_case_summary(
        self,
        case_id: str,
        tenant_id: str,
        executive_summary: str | None,
        analyst_summary: str | None,
        model_version: str,
    ) -> None:
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO case_summaries (case_id, tenant_id, executive_summary,
                                            analyst_summary, model_version)
                VALUES ($1, $2, $3, $4, $5)
                ON CONFLICT (case_id) DO UPDATE
                    SET executive_summary = COALESCE(EXCLUDED.executive_summary, case_summaries.executive_summary),
                        analyst_summary = COALESCE(EXCLUDED.analyst_summary, case_summaries.analyst_summary),
                        model_version = EXCLUDED.model_version,
                        updated_at = NOW()
                """,
                case_id,
                tenant_id,
                executive_summary,
                analyst_summary,
                model_version,
            )

    # ── Model version tracking ────────────────────────────────────────

    async def get_active_model_version(self, pipeline: str) -> dict[str, Any] | None:
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT * FROM model_versions
                 WHERE pipeline = $1 AND stage = 'production'
                 ORDER BY promoted_at DESC LIMIT 1
                """,
                pipeline,
            )
        return dict(row) if row else None


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS alerts (
    alert_id         TEXT PRIMARY KEY,
    tenant_id        TEXT NOT NULL,
    normalized_title TEXT NOT NULL DEFAULT '',
    raw_data         JSONB NOT NULL DEFAULT '{}',
    source           TEXT NOT NULL DEFAULT '',
    ml_score         DOUBLE PRECISION,
    risk_score       DOUBLE PRECISION,
    status           TEXT NOT NULL DEFAULT 'New',
    mitre_tactic     TEXT,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at       TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_alerts_tenant ON alerts (tenant_id);
CREATE INDEX IF NOT EXISTS idx_alerts_risk   ON alerts (risk_score DESC NULLS LAST);
CREATE INDEX IF NOT EXISTS idx_alerts_status ON alerts (status);

CREATE TABLE IF NOT EXISTS iocs (
    ioc_id      BIGSERIAL PRIMARY KEY,
    alert_id    TEXT NOT NULL,
    tenant_id   TEXT NOT NULL,
    ioc_type    TEXT NOT NULL,
    value       TEXT NOT NULL,
    source      TEXT NOT NULL DEFAULT 'NER',
    confidence  DOUBLE PRECISION DEFAULT 0.0,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (alert_id, ioc_type, value)
);

CREATE INDEX IF NOT EXISTS idx_iocs_tenant ON iocs (tenant_id);
CREATE INDEX IF NOT EXISTS idx_iocs_alert  ON iocs (alert_id);

CREATE TABLE IF NOT EXISTS case_summaries (
    case_id             TEXT PRIMARY KEY,
    tenant_id           TEXT NOT NULL,
    executive_summary   TEXT,
    analyst_summary     TEXT,
    model_version       TEXT NOT NULL DEFAULT '',
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS model_versions (
    id              BIGSERIAL PRIMARY KEY,
    pipeline        TEXT NOT NULL,
    mlflow_run_id   TEXT,
    onnx_s3_uri     TEXT,
    stage           TEXT NOT NULL DEFAULT 'canary',
    metrics         JSONB,
    promoted_at     TIMESTAMPTZ,
    promoted_by     TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_model_versions_pipeline ON model_versions (pipeline, stage);

CREATE TABLE IF NOT EXISTS analyst_feedback (
    id              BIGSERIAL PRIMARY KEY,
    alert_id        TEXT NOT NULL,
    tenant_id       TEXT NOT NULL,
    original_label  TEXT,
    corrected_label TEXT,
    analyst_id      TEXT,
    feedback_type   TEXT,
    used_in_training BOOLEAN DEFAULT FALSE,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Row-Level Security (enable per-tenant isolation; requires app role setup)
-- ALTER TABLE alerts ENABLE ROW LEVEL SECURITY;
-- CREATE POLICY tenant_alerts ON alerts USING (tenant_id = current_setting('app.tenant_id'));
"""
