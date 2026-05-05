"""
ARIES — Tests for the NER (Named Entity Recognition) pipeline.
"""

from __future__ import annotations

import pytest
from httpx import AsyncClient

from src.nlp.ner.inference import IOC_PATTERNS, _recover_missed_iocs, classify_ioc, detect_events
from src.nlp.ner.slm_inference import _heuristic_ner_fallback
from src.nlp.ner.schemas import EntityLabel, IOCEntity, IOCType


# ── Unit Tests: IOC Regex Validation ──────────────────────────────────

class TestIOCRegex:
    """Validate every IOC regex pattern against known samples."""

    def test_ipv4(self) -> None:
        assert IOC_PATTERNS["ipv4"].search("192.168.1.1")
        assert IOC_PATTERNS["ipv4"].search("10.0.0.255")
        assert not IOC_PATTERNS["ipv4"].search("999.999.999.999")

    def test_ipv6(self) -> None:
        assert IOC_PATTERNS["ipv6"].search("2001:0db8:85a3:0000:0000:8a2e:0370:7334")
        assert IOC_PATTERNS["ipv6"].search("fe80::1")

    def test_md5(self) -> None:
        assert IOC_PATTERNS["md5"].search("d41d8cd98f00b204e9800998ecf8427e")
        assert not IOC_PATTERNS["md5"].search("d41d8cd98f00b204e9800998ecf8427")  # 31 chars

    def test_sha1(self) -> None:
        h = "da39a3ee5e6b4b0d3255bfef95601890afd80709"
        assert IOC_PATTERNS["sha1"].search(h)

    def test_sha256(self) -> None:
        h = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        assert IOC_PATTERNS["sha256"].search(h)

    def test_domain(self) -> None:
        assert IOC_PATTERNS["domain"].search("evil.example.com")
        assert IOC_PATTERNS["domain"].search("malware.co.uk")

    def test_url(self) -> None:
        assert IOC_PATTERNS["url"].search("https://evil.com/payload.exe")
        assert IOC_PATTERNS["url"].search("http://192.168.1.1:8080/path")

    def test_email(self) -> None:
        assert IOC_PATTERNS["email"].search("attacker@evil.com")

    def test_cve(self) -> None:
        assert IOC_PATTERNS["cve"].search("CVE-2024-12345")
        assert IOC_PATTERNS["cve"].search("CVE-2021-44228")
        assert not IOC_PATTERNS["cve"].search("CVE-20-1234")


# ── Unit Tests: IOC Classification ────────────────────────────────────

class TestIOCClassification:
    def test_classify_ipv4(self) -> None:
        ioc_type, validated = classify_ioc("192.168.1.1")
        assert ioc_type == IOCType.IP_ADDRESS
        assert validated is True

    def test_classify_sha256(self) -> None:
        h = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        ioc_type, validated = classify_ioc(h)
        assert ioc_type == IOCType.FILE_HASH
        assert validated is True

    def test_classify_cve(self) -> None:
        ioc_type, validated = classify_ioc("CVE-2021-44228")
        assert ioc_type == IOCType.CVE_ID
        assert validated is True

    def test_classify_domain(self) -> None:
        ioc_type, validated = classify_ioc("evil.example.com")
        assert ioc_type == IOCType.DOMAIN
        assert validated is True

    def test_classify_url(self) -> None:
        ioc_type, validated = classify_ioc("https://evil.com/payload")
        assert ioc_type == IOCType.URL
        assert validated is True

    def test_classify_unknown(self) -> None:
        ioc_type, validated = classify_ioc("some random text")
        assert ioc_type == IOCType.UNKNOWN
        assert validated is False


class TestIOCRecovery:
    def test_recover_merges_fragmented_ipv4(self) -> None:
        text = "C2 traffic observed connecting to 185.220.101.42."
        entities = [
            IOCEntity(
                text="185.220.101",
                label=EntityLabel.INDICATOR,
                start=34,
                end=45,
                confidence=0.5453,
                ioc_type=IOCType.UNKNOWN,
                ioc_validated=False,
            ),
            IOCEntity(
                text="42",
                label=EntityLabel.INDICATOR,
                start=46,
                end=48,
                confidence=0.6018,
                ioc_type=IOCType.UNKNOWN,
                ioc_validated=False,
            ),
        ]

        recovered = _recover_missed_iocs(text, entities)

        assert len(recovered) == 1
        assert recovered[0].text == "185.220.101.42"
        assert recovered[0].ioc_type == IOCType.IP_ADDRESS
        assert recovered[0].ioc_validated is True

    def test_slm_fallback_recovers_contextual_entities(self) -> None:
        text = (
            "The adversary pivoted laterally after a spearphishing lure. "
            "Telemetry showed outbound traffic from the compromised endpoint to a bulletproof hosting provider. "
            "A suspicious executable was found in the temp directory masquerading as a Windows DLL."
        )
        recovered = _heuristic_ner_fallback(text)

        labels = {ent["label"] for ent in recovered["entities"]}
        events = {ev["type"] for ev in recovered["events"]}

        assert "System" in labels
        assert "Tool" in labels
        assert "Organization" in labels
        assert "Phishing" in events
        assert "LateralMovement" in events


# ── Unit Tests: Event Detection ───────────────────────────────────────

class TestEventDetection:
    def test_detect_ransomware(self) -> None:
        text = "The ransomware encrypted all files on the server."
        events = detect_events(text)
        types = [e.event_type for e in events]
        assert "Ransom" in types

    def test_detect_phishing(self) -> None:
        text = "A phishing email was sent to multiple employees."
        events = detect_events(text)
        types = [e.event_type for e in events]
        assert "Phishing" in types

    def test_detect_multiple_events(self) -> None:
        text = "Data breach and ransomware attack detected at the headquarters."
        events = detect_events(text)
        types = [e.event_type for e in events]
        assert "Databreach" in types
        assert "Ransom" in types

    def test_no_events(self) -> None:
        text = "The weather is sunny today."
        events = detect_events(text)
        assert len(events) == 0


# ── Integration Tests: NER API ────────────────────────────────────────

class TestNERAPI:
    @pytest.mark.asyncio
    async def test_ner_endpoint(self, async_client: AsyncClient) -> None:
        """POST /nlp/ner returns IOC entities."""
        resp = await async_client.post(
            "/nlp/ner",
            headers={"X-Tenant-ID": "tenant-1"},
            json={
                "text": "Malware hash e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855 was found.",
                "tenant_id": "tenant-1",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "entities" in data
        assert "events" in data
        assert isinstance(data["entities"], list)

    @pytest.mark.asyncio
    async def test_ner_batch_endpoint(self, async_client: AsyncClient) -> None:
        """POST /nlp/ner/batch processes multiple texts."""
        resp = await async_client.post(
            "/nlp/ner/batch",
            headers={"X-Tenant-ID": "tenant-1"},
            json={
                "texts": [
                    "IP 192.168.1.1 seen",
                    "CVE-2021-44228 exploited",
                ],
                "tenant_id": "tenant-1",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 2

    @pytest.mark.asyncio
    async def test_ner_health(self, async_client: AsyncClient) -> None:
        """GET /nlp/ner/health should report NER model status."""
        resp = await async_client.get("/nlp/ner/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["pipeline"] == "ner"

    @pytest.mark.asyncio
    async def test_ner_empty_text(self, async_client: AsyncClient) -> None:
        """POST /nlp/ner with empty text should handle gracefully."""
        resp = await async_client.post(
            "/nlp/ner",
            headers={"X-Tenant-ID": "tenant-1"},
            json={"text": "", "tenant_id": "tenant-1"},
        )
        # Should either succeed with empty entities or return 422
        assert resp.status_code in (200, 422)
