"""
ARIES — Triage Kafka consumer.

Listens on alerts.raw, runs ONNX triage inference, computes risk_score,
and publishes enriched alerts to alerts.enriched.
"""

from __future__ import annotations

import json
from typing import Any

from src.shared.config import ServiceSettings
from src.shared.db import Database
from src.shared.kafka import KafkaConsumerWorker, KafkaProducer
from src.shared.logging import get_logger
from src.shared.model_loader import ModelStore
from src.triage.feature_engineering import extract_features
from src.triage.inference import run_triage_inference
from src.triage.slm_inference import run_triage_inference_slm
from src.triage.schemas import AlertStatus, CanonicalAlert, EnrichedAlert, Severity

log = get_logger("triage_consumer")


class TriageKafkaConsumer:
    """
    Async Kafka consumer that:
    1. Reads CanonicalAlert from alerts.raw
    2. Extracts features and runs triage ONNX inference
    3. Computes composite risk_score
    4. Auto-closes low-confidence alerts
    5. Publishes EnrichedAlert to alerts.enriched
    6. Persists to PostgreSQL
    """

    def __init__(
        self,
        settings: ServiceSettings,
        model_store: ModelStore,
        producer: KafkaProducer,
        db: Database,
    ) -> None:
        self._settings = settings
        self._model_store = model_store
        self._producer = producer
        self._db = db
        self._worker = KafkaConsumerWorker(
            settings=settings,
            topic=settings.kafka_topic_alerts_raw,
            group_id=settings.kafka_consumer_group_triage,
            handler=self._handle_message,
        )

    async def start(self) -> None:
        await self._worker.start()

    async def stop(self) -> None:
        await self._worker.stop()

    async def _handle_message(self, raw_msg: dict[str, Any]) -> None:
        """Process a single alert from alerts.raw."""
        try:
            alert = CanonicalAlert.model_validate(raw_msg)
        except Exception:
            log.exception("invalid_canonical_alert", raw_msg=str(raw_msg)[:500])
            return

        log.info(
            "triage_processing_alert",
            alert_id=alert.alert_id,
            tenant_id=alert.tenant_id,
            source=alert.source,
        )

        # Extract features for ONNX model
        features = extract_features(alert.model_dump(), encoder=self._model_store.triage_encoder)

        # Asset criticality and behavioral score from enrichment context
        asset_criticality = self._get_asset_criticality(alert)
        behavioral_score = self._get_behavioral_score(alert)

        if self._settings.use_slm:
            result = await run_triage_inference_slm(
                alert_dict=alert.model_dump(),
                alert_id=alert.alert_id,
                tenant_id=alert.tenant_id,
                asset_criticality=asset_criticality,
                behavioral_score=behavioral_score,
                settings=self._settings,
                suspicion_level=alert.suspicion_level,
            )
        else:
            # Run ONNX triage inference
            result = await run_triage_inference(
                session=self._model_store.triage_session,
                features=features,
                alert_id=alert.alert_id,
                tenant_id=alert.tenant_id,
                asset_criticality=asset_criticality,
                behavioral_score=behavioral_score,
                settings=self._settings,
                suspicion_level=alert.suspicion_level,
            )

        # Determine final status
        if result.auto_closed:
            status = AlertStatus.CLOSED_FP
        else:
            status = AlertStatus.TRIAGED

        # Persist to PostgreSQL
        try:
            await self._db.insert_alert(
                alert_id=alert.alert_id,
                tenant_id=alert.tenant_id,
                normalized_title=alert.normalized_title,
                raw_data=alert.raw_data,
                source=alert.source,
                ml_score=result.ml_score,
                risk_score=result.risk_score,
                status=status.value,
                mitre_tactic=alert.mitre_tactic,
            )
        except Exception:
            log.exception("alert_persist_failed", alert_id=alert.alert_id)

        # If auto-closed, stop here — don't enriched
        if result.auto_closed:
            log.info(
                "alert_auto_closed",
                alert_id=alert.alert_id,
                ml_score=result.ml_score,
                threshold=self._settings.auto_close_threshold,
            )
            return

        # Build enriched alert
        enriched = EnrichedAlert(
            alert_id=alert.alert_id,
            tenant_id=alert.tenant_id,
            timestamp=alert.timestamp,
            source=alert.source,
            normalized_title=alert.normalized_title,
            severity=alert.severity,
            raw_data=alert.raw_data,
            ml_score=result.ml_score,
            risk_score=result.risk_score,
            incident_grade=result.incident_grade,
            auto_closed=False,
            model_version=result.model_version,
            mitre_tactic=alert.mitre_tactic,
            mitre_technique=alert.mitre_technique,
            entity_type=alert.entity_type,
            device_name=alert.device_name,
            ip_address=alert.ip_address,
            user_name=alert.user_name,
            category=alert.category,
            asset_criticality=asset_criticality,
            behavioral_score=behavioral_score,
        )

        # Publish to alerts.enriched
        try:
            await self._producer.send(
                topic=self._settings.kafka_topic_alerts_enriched,
                value=enriched.model_dump(mode="json"),
                key=alert.alert_id,
            )
            log.info(
                "enriched_alert_published",
                alert_id=alert.alert_id,
                risk_score=result.risk_score,
            )
        except Exception:
            log.exception("enriched_alert_publish_failed", alert_id=alert.alert_id)

    def _get_asset_criticality(self, alert: CanonicalAlert) -> float:
        """Estimate asset criticality from alert context.

        In a full production deployment this queries a CMDB / asset inventory.
        Here we derive a reasonable score from the alert's severity, entity
        context, and MITRE tactic — giving the risk formula meaningful
        differentiation across alert types.
        """
        base: float = {
            Severity.CRITICAL: 0.95,
            Severity.HIGH: 0.75,
            Severity.MEDIUM: 0.50,
            Severity.LOW: 0.25,
        }.get(alert.severity, 0.50)

        # Boost for high-value entity types (servers, domain controllers)
        title_lower = (alert.normalized_title or "").lower()
        raw_str = str(alert.raw_data).lower()
        context = f"{title_lower} {raw_str}"

        if any(kw in context for kw in [
            "domain controller", "active directory", "exchange",
            "database", "backup", "admin",
        ]):
            base = min(base + 0.15, 1.0)
        elif any(kw in context for kw in [
            "server", "firewall", "gateway", "dns",
        ]):
            base = min(base + 0.10, 1.0)

        return round(base, 2)

    def _get_behavioral_score(self, alert: CanonicalAlert) -> float:
        """Estimate behavioral anomaly score from alert context.

        In production this queries Elasticsearch / UBA for historical
        anomaly data. Here we derive a score from the suspicion level,
        MITRE tactic, and rule-level indicators.
        """
        base: float = {
            "Critical": 0.95, "High": 0.70, "Medium": 0.45, "Low": 0.20,
        }.get(alert.suspicion_level or "Medium", 0.45)

        # Boost for high-confidence attack patterns
        tactic = (alert.mitre_tactic or "").lower()
        title_lower = (alert.normalized_title or "").lower()

        # Lateral movement and exfiltration are higher behavioral anomalies
        if any(kw in tactic for kw in ["lateral", "exfiltration", "impact"]):
            base = min(base + 0.15, 1.0)
        elif any(kw in tactic for kw in ["execution", "persistence", "privilege"]):
            base = min(base + 0.10, 1.0)

        # Brute force or repeated failures indicate anomalous behavior
        if any(kw in title_lower for kw in [
            "brute force", "multiple fail", "repeated", "credential",
        ]):
            base = min(base + 0.10, 1.0)

        return round(base, 2)

    @property
    def is_connected(self) -> bool:
        return self._worker.is_connected
