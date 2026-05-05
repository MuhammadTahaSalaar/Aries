"""
ARIES — Tests for the SIEM Ingestion pipeline.
"""

from __future__ import annotations

import pytest
from httpx import AsyncClient

from src.ingestion.normalizer import normalize_siem_alert, VENDOR_MAPPINGS
from src.ingestion.schemas import SIEMRawPayload
from src.triage.schemas import Severity


# ── Unit Tests: Vendor Mappings ───────────────────────────────────────

class TestVendorMappings:
    def test_known_vendors(self) -> None:
        """All expected vendors should have mappings."""
        for vendor in ("wazuh", "splunk", "elastic_siem", "crowdstrike"):
            assert vendor in VENDOR_MAPPINGS

    def test_mapping_has_required_keys(self) -> None:
        """Each mapping must define at least alert_id_path, title_path, severity_path."""
        for vendor, mapping in VENDOR_MAPPINGS.items():
            assert "alert_id_path" in mapping, f"{vendor} missing alert_id_path"
            assert "title_path" in mapping, f"{vendor} missing title_path"
            assert "severity_path" in mapping, f"{vendor} missing severity_path"


# ── Unit Tests: Normalization ─────────────────────────────────────────

class TestNormalization:
    def test_normalize_wazuh(self) -> None:
        """Wazuh alert should normalize correctly."""
        payload = SIEMRawPayload(
            vendor="wazuh",
            tenant_id="t-1",
            raw={
                "id": "wazuh-001",
                "rule": {
                    "description": "SSH Brute Force",
                    "level": 12,
                },
                "timestamp": "2026-01-15T10:30:00Z",
            },
        )
        alert = normalize_siem_alert(payload)
        assert alert.alert_id == "wazuh-001"
        assert alert.normalized_title == "SSH Brute Force"
        assert alert.source == "wazuh"
        assert alert.tenant_id == "t-1"

    def test_normalize_splunk(self) -> None:
        """Splunk alert should normalize correctly."""
        payload = SIEMRawPayload(
            vendor="splunk",
            tenant_id="t-1",
            raw={
                "sid": "splunk-002",
                "search_name": "Suspicious PowerShell",
                "result": {
                    "urgency": "high",
                },
                "_time": "2026-01-15T10:30:00Z",
            },
        )
        alert = normalize_siem_alert(payload)
        assert alert.alert_id == "splunk-002"
        assert alert.normalized_title == "Suspicious PowerShell"

    def test_normalize_elastic(self) -> None:
        """Elastic SIEM alert should normalize correctly."""
        payload = SIEMRawPayload(
            vendor="elastic_siem",
            tenant_id="t-1",
            raw={
                "kibana": {
                    "alert": {
                        "uuid": "es-003",
                        "rule": {"name": "Malware Detected"},
                        "severity": "critical",
                    },
                },
                "@timestamp": "2026-01-15T10:30:00Z",
            },
        )
        alert = normalize_siem_alert(payload)
        assert alert.alert_id == "es-003"

    def test_normalize_crowdstrike(self) -> None:
        """CrowdStrike alert should normalize correctly."""
        payload = SIEMRawPayload(
            vendor="crowdstrike",
            tenant_id="t-1",
            raw={
                "composite_id": "cs-004",
                "display_name": "Credential Theft",
                "max_severity_displayname": "High",
                "created_timestamp": "2026-01-15T10:30:00Z",
            },
        )
        alert = normalize_siem_alert(payload)
        assert alert.alert_id == "cs-004"
        assert alert.normalized_title == "Credential Theft"

    def test_normalize_unknown_vendor(self) -> None:
        """Unknown vendor should raise ValueError."""
        payload = SIEMRawPayload(
            vendor="unknown_vendor",
            tenant_id="t-1",
            raw={"id": "x"},
        )
        with pytest.raises(ValueError, match="Unsupported SIEM vendor"):
            normalize_siem_alert(payload)


# ── Unit Tests: Severity Resolution ──────────────────────────────────

class TestSeverityResolution:
    def test_wazuh_high_level(self) -> None:
        """Wazuh level >= 12 should map to Critical."""
        payload = SIEMRawPayload(
            vendor="wazuh",
            tenant_id="t-1",
            raw={
                "id": "w-1",
                "rule": {"description": "Test", "level": 14},
                "timestamp": "2026-01-15T00:00:00Z",
            },
        )
        alert = normalize_siem_alert(payload)
        assert alert.severity in (Severity.CRITICAL, Severity.HIGH)

    def test_splunk_urgency_low(self) -> None:
        """Splunk urgency 'low' should map to Low severity."""
        payload = SIEMRawPayload(
            vendor="splunk",
            tenant_id="t-1",
            raw={
                "sid": "s-1",
                "search_name": "Test",
                "result": {
                    "urgency": "low",
                },
            },
        )
        alert = normalize_siem_alert(payload)
        assert alert.severity == Severity.LOW


# ── Integration Tests: Ingestion API ─────────────────────────────────

class TestIngestionAPI:
    @pytest.mark.asyncio
    async def test_ingest_wazuh(self, async_client: AsyncClient) -> None:
        """POST /ingest/siem should normalize a Wazuh alert."""
        resp = await async_client.post(
            "/ingest/siem",
            headers={"X-Tenant-ID": "tenant-1"},
            json={
                "vendor": "wazuh",
                "tenant_id": "tenant-1",
                "raw": {
                    "id": "w-100",
                    "rule": {"description": "Test Alert", "level": 8},
                    "timestamp": "2026-01-15T10:00:00Z",
                },
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["accepted"] is True
        assert data["alert_id"] == "w-100"

    @pytest.mark.asyncio
    async def test_ingest_batch(self, async_client: AsyncClient) -> None:
        """POST /ingest/siem/batch should process multiple alerts."""
        resp = await async_client.post(
            "/ingest/siem/batch",
            headers={"X-Tenant-ID": "tenant-1"},
            json=[
                {
                    "vendor": "wazuh",
                    "tenant_id": "tenant-1",
                    "raw": {
                        "id": "w-200",
                        "rule": {"description": "Alert A", "level": 5},
                        "timestamp": "2026-01-15T10:00:00Z",
                    },
                },
                {
                    "vendor": "wazuh",
                    "tenant_id": "tenant-1",
                    "raw": {
                        "id": "w-201",
                        "rule": {"description": "Alert B", "level": 10},
                        "timestamp": "2026-01-15T10:01:00Z",
                    },
                },
            ],
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 2

    @pytest.mark.asyncio
    async def test_vendors_list(self, async_client: AsyncClient) -> None:
        """GET /ingest/siem/vendors should list all supported vendors."""
        resp = await async_client.get("/ingest/siem/vendors")
        assert resp.status_code == 200
        data = resp.json()
        assert "wazuh" in data["vendors"]
        assert "splunk" in data["vendors"]
        assert "elastic_siem" in data["vendors"]
        assert "crowdstrike" in data["vendors"]
