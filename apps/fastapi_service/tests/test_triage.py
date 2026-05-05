"""
ARIES — Tests for the Triage pipeline.
"""

from __future__ import annotations

import pytest
from httpx import AsyncClient

from src.shared.config import ServiceSettings
from src.triage.feature_engineering import extract_features
from src.triage.inference import compute_risk_score, heuristic_triage
from src.triage.schemas import CanonicalAlert, IncidentGrade, Severity


# ── Unit Tests: Feature Engineering ───────────────────────────────────

class TestFeatureEngineering:
    def test_extract_features_returns_49_dims(self) -> None:
        """Feature vector must be exactly 49 elements."""
        alert = {
            "alert_id": "test-1",
            "tenant_id": "t-1",
            "timestamp": "2026-01-15T10:30:00Z",
            "source": "wazuh",
            "normalized_title": "Test Alert",
            "raw_data": {
                "AlertTitle": "Suspicious Process",
                "Category": "Malware",
                "SuspicionLevel": "High",
                "MitreTechniques": "T1566.001,T1059",
            },
        }
        features = extract_features(alert)
        assert len(features) == 49
        assert all(isinstance(f, float) for f in features)

    def test_extract_features_temporal(self) -> None:
        """Temporal features should extract correctly."""
        alert = {
            "timestamp": "2026-06-15T14:30:00Z",
            "raw_data": {},
        }
        features = extract_features(alert)
        assert features[0] == 14.0  # hour
        assert features[1] == 0.0  # Monday (weekday)
        assert features[2] == 6.0  # June

    def test_extract_features_mitre_count(self) -> None:
        """MITRE technique count should be computed."""
        alert = {
            "raw_data": {"MitreTechniques": "T1566.001,T1059,T1078"},
            "timestamp": "2026-01-01T00:00:00Z",
        }
        features = extract_features(alert)
        assert features[3] == 1.0  # has_mitre
        assert features[4] == 3.0  # 3 techniques

    def test_extract_features_empty_alert(self) -> None:
        """Should handle empty/minimal alert without crashing."""
        alert = {"raw_data": {}}
        features = extract_features(alert)
        assert len(features) == 49


# ── Unit Tests: Inference Logic ───────────────────────────────────────

class TestTriageInference:
    def test_risk_score_formula(self) -> None:
        """Risk score should follow the weighted formula."""
        settings = ServiceSettings(
            triage_weight_ml=0.5,
            triage_weight_asset=0.3,
            triage_weight_behavior=0.2,
        )
        score = compute_risk_score(
            ml_score=0.8,
            asset_criticality=0.9,
            behavioral_score=0.7,
            settings=settings,
        )
        # 0.5*0.8 + 0.3*0.9 + 0.2*0.7 = 0.4 + 0.27 + 0.14 = 0.81 → 81.0
        assert abs(score - 81.0) < 0.1

    def test_risk_score_clamped(self) -> None:
        """Risk score should be clamped to [0, 100]."""
        settings = ServiceSettings()
        score = compute_risk_score(2.0, 2.0, 2.0, settings)
        assert score <= 100.0

    def test_heuristic_fallback_critical(self) -> None:
        """Critical suspicion level should give high score."""
        score, grade = heuristic_triage("Critical")
        assert score == 0.90
        assert grade == IncidentGrade.TRUE_POSITIVE

    def test_heuristic_fallback_low(self) -> None:
        """Low suspicion level should give low score."""
        score, grade = heuristic_triage("Low")
        assert score == 0.10
        assert grade == IncidentGrade.FALSE_POSITIVE

    def test_heuristic_fallback_none(self) -> None:
        """None suspicion level should default to Medium."""
        score, grade = heuristic_triage(None)
        assert score == 0.40
        assert grade == IncidentGrade.FALSE_POSITIVE


# ── Integration Tests: Triage API ─────────────────────────────────────

class TestTriageAPI:
    @pytest.mark.asyncio
    async def test_score_endpoint(self, async_client: AsyncClient) -> None:
        """POST /triage/score should return a valid TriageResult."""
        resp = await async_client.post(
            "/triage/score",
            headers={"X-Tenant-ID": "tenant-1"},
            json={
                "alert_id": "alert-001",
                "tenant_id": "tenant-1",
                "timestamp": "2026-01-15T10:30:00Z",
                "source": "wazuh",
                "normalized_title": "Suspicious login attempt",
                "severity": "High",
                "raw_data": {"AlertTitle": "Brute force", "Category": "Credential"},
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "ml_score" in data
        assert 0.0 <= data["ml_score"] <= 1.0
        assert "risk_score" in data
        assert 0.0 <= data["risk_score"] <= 100.0
        assert "incident_grade" in data
        assert data["alert_id"] == "alert-001"

    @pytest.mark.asyncio
    async def test_triage_health(self, async_client: AsyncClient) -> None:
        """GET /triage/health should report model status."""
        resp = await async_client.get("/triage/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["pipeline"] == "triage"
        assert "model_loaded" in data


# ── Schema Validation Tests ───────────────────────────────────────────

class TestTriageSchemas:
    def test_canonical_alert_validation(self) -> None:
        """CanonicalAlert should validate correctly with all fields."""
        alert = CanonicalAlert(
            alert_id="a-1",
            tenant_id="t-1",
            source="splunk",
            normalized_title="Test",
            severity=Severity.HIGH,
            raw_data={"key": "value"},
        )
        assert alert.alert_id == "a-1"
        assert alert.severity == Severity.HIGH

    def test_canonical_alert_defaults(self) -> None:
        """CanonicalAlert should have sensible defaults."""
        alert = CanonicalAlert(
            alert_id="a-2",
            tenant_id="t-1",
            source="test",
            normalized_title="Default Test",
        )
        assert alert.severity == Severity.MEDIUM
        assert alert.raw_data == {}
