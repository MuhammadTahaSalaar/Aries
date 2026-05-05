"""
ARIES — Tests for the Summarization pipeline.
"""

from __future__ import annotations

import pytest
from httpx import AsyncClient

from src.nlp.summarizer.inference import _clean_summary
from src.nlp.summarizer.slm_inference import _clean_slm_summary
from src.nlp.summarizer.schemas import SummarizeMode, SummarizeRequest


# ── Schema Tests ──────────────────────────────────────────────────────

class TestSummarizerSchemas:
    def test_summarize_request_defaults(self) -> None:
        req = SummarizeRequest(
            text="Test incident report.",
            tenant_id="t-1",
        )
        assert req.mode == SummarizeMode.EXECUTIVE

    def test_summarize_modes(self) -> None:
        assert SummarizeMode.EXECUTIVE == "executive"
        assert SummarizeMode.ANALYST == "analyst"

    def test_summarize_request_with_case(self) -> None:
        req = SummarizeRequest(
            text="Test incident report.",
            tenant_id="t-1",
            case_id="case-42",
            mode=SummarizeMode.ANALYST,
        )
        assert req.case_id == "case-42"
        assert req.mode == SummarizeMode.ANALYST


class TestSummarizerCleanup:
    def test_clean_summary_trims_hallucinated_trailing_clause(self) -> None:
        source = (
            "On March 15, 2024, the SOC detected a multi-stage attack. "
            "A spear-phishing email exploited CVE-2023-36884 to deploy malware. "
            "The attacker used PowerShell to download payloads from 203.0.113.50, "
            "established persistence via scheduled tasks, moved laterally using stolen "
            "credentials, accessed the domain controller, and exfiltrated 2.3GB of "
            "financial data. The attack was attributed to FIN7."
        )
        generated = (
            "On March 15, 2024, the SOC detected a multi-stage attack. "
            "A spear-phishing email exploited CVE-2023-36884 to deploy malware. "
            "The attacker used PowerShell to download payloads from 203.0.113.50, "
            "established persistence via scheduled tasks, moved laterally using stolen "
            "credentials, accessed the domain controller, and exfiltrated 2.3GB of "
            "financial data. The attack was attributed to FIN7, a security company "
            "that was responsible for the SOC's"
        )

        assert _clean_summary(generated, source) == source

    def test_clean_slm_summary_truncates_off_topic_tail(self) -> None:
        source = (
            "Wazuh detected a brute-force SSH attack on prod-db-01 from 45.33.32.156. "
            "After successful login with svc_backup, the attacker executed reconnaissance commands "
            "and fetched a shell script."
        )
        generated = (
            "Wazuh detected a brute-force SSH attack on prod-db-01 from 45.33.32.156. "
            "After successful login with svc_backup, the attacker executed reconnaissance commands. "
            "SAN FRANCISCO (KRON) A study says teenagers use smartphones 11 hours daily."
        )

        cleaned = _clean_slm_summary(generated, source)
        assert "SAN FRANCISCO" not in cleaned
        assert "brute-force SSH attack" in cleaned

    def test_clean_slm_summary_trims_mid_sentence_news_drift(self) -> None:
        source = (
            "On March 15, 2024, Wazuh detected a brute-force attack against SSH on prod-db-01 "
            "from external IP 45.33.32.156."
        )
        generated = (
            "On March 15, 2024, Wazuh detected a brute-force attack against SSH on prod-db-01 "
            "from external IP 45.33 SAN FRANCISCO (KRON) A new study says teenagers spend 11 hours on phones"
        )

        cleaned = _clean_slm_summary(generated, source)
        assert "SAN FRANCISCO" not in cleaned
        assert "A new study" not in cleaned
        assert "brute-force attack" in cleaned

    def test_clean_slm_summary_keeps_key_details_after_drift_cut(self) -> None:
        source = (
            "On March 15, 2024 at 02:34 UTC, Wazuh detected a brute-force attack against SSH service "
            "on server prod-db-01 (10.0.1.50) from external IP 45.33.32.156. "
            "At 02:41 UTC, the attacker authenticated with compromised service account svc_backup."
        )
        generated = (
            "On March 15, 2024, Wazuh detected a brute-force attack against SSH service on prod-db-01 "
            "from external IP 45.33 SAN FRANCISCO (KRON) A new study from UC Berkeley says..."
        )

        cleaned = _clean_slm_summary(generated, source)
        assert "SAN FRANCISCO" not in cleaned
        assert "brute-force attack" in cleaned
        assert len(cleaned.split()) >= 12


# ── Integration Tests: Summarizer API ─────────────────────────────────

class TestSummarizerAPI:
    @pytest.mark.asyncio
    async def test_summarize_executive(self, async_client: AsyncClient) -> None:
        """POST /nlp/summarize in executive mode."""
        resp = await async_client.post(
            "/nlp/summarize",
            headers={"X-Tenant-ID": "tenant-1"},
            json={
                "text": "A critical ransomware incident was detected on the finance server. "
                        "The malware encrypted all files and demanded a ransom payment in Bitcoin. "
                        "The SOC team initiated containment procedures and isolated the affected host.",
                "tenant_id": "tenant-1",
                "mode": "executive",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "summary" in data
        assert isinstance(data["summary"], str)
        assert data["mode"] == "executive"

    @pytest.mark.asyncio
    async def test_summarize_analyst(self, async_client: AsyncClient) -> None:
        """POST /nlp/summarize in analyst mode."""
        resp = await async_client.post(
            "/nlp/summarize",
            headers={"X-Tenant-ID": "tenant-1"},
            json={
                "text": "Investigation report for case-007. "
                        "Multiple hosts were compromised via CVE-2021-44228. "
                        "Lateral movement observed using PsExec.",
                "tenant_id": "tenant-1",
                "mode": "analyst",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["mode"] == "analyst"

    @pytest.mark.asyncio
    async def test_summarize_with_case_id(self, async_client: AsyncClient) -> None:
        """POST /nlp/summarize should accept case_id."""
        resp = await async_client.post(
            "/nlp/summarize",
            headers={"X-Tenant-ID": "tenant-1"},
            json={
                "text": "Incident timeline for persistent threat actor.",
                "tenant_id": "tenant-1",
                "case_id": "case-42",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "summary" in data

    @pytest.mark.asyncio
    async def test_summarize_health(self, async_client: AsyncClient) -> None:
        """GET /nlp/summarize/health should report model status."""
        resp = await async_client.get("/nlp/summarize/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["pipeline"] == "summarizer"
