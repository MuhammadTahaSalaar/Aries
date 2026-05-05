"""
ARIES — SIEM Ingestion Service.

Standalone module that:
1. Receives raw vendor JSON from SIEM/EDR webhooks
2. Normalises it to the ARIES Canonical Alert Schema
3. Deduplicates via Redis
4. Publishes to Kafka alerts.raw

Supported vendor mappings:
- Wazuh
- Splunk
- Elastic SIEM
- CrowdStrike Falcon
- Generic fallback
"""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timezone
from typing import Any

from src.ingestion.schemas import SIEMRawPayload
from src.shared.logging import get_logger
from src.triage.schemas import CanonicalAlert, Severity

log = get_logger("ingestion")

# ── Vendor Normalization Mappings ────────────────────────────────────

VENDOR_MAPPINGS: dict[str, dict[str, Any]] = {
    "wazuh": {
        "alert_id_path": ["id"],
        "title_path": ["rule", "description"],
        "severity_path": ["rule", "level"],
        "severity_map": {
            range(0, 4): Severity.LOW,
            range(4, 8): Severity.MEDIUM,
            range(8, 12): Severity.HIGH,
            range(12, 16): Severity.CRITICAL,
        },
        "mitre_tactic_path": ["rule", "mitre", "tactic"],
        "mitre_technique_path": ["rule", "mitre", "id"],
        "ip_address_path": ["data", "srcip"],
        "user_name_path": ["data", "srcuser"],
        "device_name_path": ["agent", "name"],
        "timestamp_path": ["timestamp"],
        "category_path": ["rule", "groups"],
    },
    "splunk": {
        "alert_id_path": ["sid"],
        "title_path": ["search_name"],
        "severity_path": ["result", "urgency"],
        "severity_map": {
            "informational": Severity.LOW,
            "low": Severity.LOW,
            "medium": Severity.MEDIUM,
            "high": Severity.HIGH,
            "critical": Severity.CRITICAL,
        },
        "ip_address_path": ["result", "src_ip"],
        "user_name_path": ["result", "user"],
        "device_name_path": ["result", "host"],
        "timestamp_path": ["_time"],
        "category_path": ["result", "source"],
    },
    "elastic_siem": {
        "alert_id_path": ["kibana", "alert", "uuid"],
        "title_path": ["kibana", "alert", "rule", "name"],
        "severity_path": ["kibana", "alert", "severity"],
        "severity_map": {
            "low": Severity.LOW,
            "medium": Severity.MEDIUM,
            "high": Severity.HIGH,
            "critical": Severity.CRITICAL,
        },
        "mitre_tactic_path": ["kibana", "alert", "rule", "threat", "tactic", "name"],
        "mitre_technique_path": ["kibana", "alert", "rule", "threat", "technique", "id"],
        "ip_address_path": ["source", "ip"],
        "user_name_path": ["user", "name"],
        "device_name_path": ["host", "name"],
        "timestamp_path": ["@timestamp"],
        "category_path": ["event", "category"],
    },
    "crowdstrike": {
        "alert_id_path": ["composite_id"],
        "title_path": ["display_name"],
        "severity_path": ["max_severity_displayname"],
        "severity_map": {
            "informational": Severity.LOW,
            "low": Severity.LOW,
            "medium": Severity.MEDIUM,
            "high": Severity.HIGH,
            "critical": Severity.CRITICAL,
        },
        "mitre_tactic_path": ["tactic"],
        "mitre_technique_path": ["technique_id"],
        "ip_address_path": ["local_ip"],
        "user_name_path": ["user_name"],
        "device_name_path": ["computer_name"],
        "file_hash_path": ["sha256"],
        "timestamp_path": ["created_timestamp"],
    },
}


def _deep_get(data: dict[str, Any], keys: list[str], default: Any = None) -> Any:
    """Safely traverse nested dicts/lists using a key path."""
    current = data
    for key in keys:
        if isinstance(current, dict):
            current = current.get(key, default)
        elif isinstance(current, list) and isinstance(key, int):
            current = current[key] if key < len(current) else default
        else:
            return default
        if current is None:
            return default
    return current


def _resolve_severity(raw_severity: Any, severity_map: dict[Any, Severity]) -> Severity:
    """Map a vendor severity value to the canonical Severity enum."""
    if raw_severity is None:
        return Severity.MEDIUM

    # String-based mapping
    if isinstance(raw_severity, str):
        mapped = severity_map.get(raw_severity.lower())
        if mapped:
            return mapped

    # Integer-based mapping (range keys)
    if isinstance(raw_severity, (int, float)):
        level = int(raw_severity)
        for key, value in severity_map.items():
            if isinstance(key, range) and level in key:
                return value
            if isinstance(key, int) and key == level:
                return value

    return Severity.MEDIUM


def _compute_dedup_key(alert_id: str, tenant_id: str, title: str) -> str:
    """Compute a deduplication hash for the alert."""
    content = f"{tenant_id}:{alert_id}:{title}"
    return f"dedup:{hashlib.sha256(content.encode()).hexdigest()[:24]}"


def _parse_timestamp(raw: Any) -> datetime:
    """Best-effort timestamp parsing."""
    if raw is None:
        return datetime.now(timezone.utc)
    if isinstance(raw, datetime):
        return raw
    if isinstance(raw, (int, float)):
        return datetime.fromtimestamp(raw, tz=timezone.utc)
    if isinstance(raw, str):
        for fmt in (
            "%Y-%m-%dT%H:%M:%S.%fZ",
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%dT%H:%M:%S%z",
            "%Y-%m-%d %H:%M:%S",
        ):
            try:
                return datetime.strptime(raw, fmt).replace(tzinfo=timezone.utc)
            except ValueError:
                continue
        try:
            return datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError:
            pass
    return datetime.now(timezone.utc)


def normalize_siem_alert(payload: SIEMRawPayload) -> CanonicalAlert:
    """
    Normalize a raw SIEM vendor payload into the ARIES Canonical Alert Schema.
    Uses vendor-specific field mappings. Falls back to generic extraction.
    """
    if not payload.vendor:
        raise ValueError(f"Vendor is required. Pass it as ?vendor=<name> query parameter or in the body. Supported: {list(VENDOR_MAPPINGS.keys())}")

    vendor = payload.vendor.lower().replace("-", "_").replace(" ", "_")
    data = payload.raw or {}
    mapping = VENDOR_MAPPINGS.get(vendor)

    if mapping is None:
        raise ValueError(f"Unsupported SIEM vendor: '{payload.vendor}'. Supported: {list(VENDOR_MAPPINGS.keys())}")

    # Extract fields using vendor mapping
    raw_alert_id = _deep_get(data, mapping.get("alert_id_path", []))
    alert_id = str(raw_alert_id) if raw_alert_id else str(uuid.uuid4())

    title = _deep_get(data, mapping.get("title_path", []), "Unknown Alert")
    if isinstance(title, list):
        title = " | ".join(str(t) for t in title)

    raw_severity = _deep_get(data, mapping.get("severity_path", []))
    severity = _resolve_severity(raw_severity, mapping.get("severity_map", {}))

    raw_ts = payload.timestamp or _deep_get(data, mapping.get("timestamp_path", []))
    timestamp = _parse_timestamp(raw_ts)

    # Optional fields
    mitre_tactic = _deep_get(data, mapping.get("mitre_tactic_path", []))
    if isinstance(mitre_tactic, list):
        mitre_tactic = mitre_tactic[0] if mitre_tactic else None

    mitre_technique = _deep_get(data, mapping.get("mitre_technique_path", []))
    if isinstance(mitre_technique, list):
        mitre_technique = mitre_technique[0] if mitre_technique else None

    ip_address = _deep_get(data, mapping.get("ip_address_path", []))
    user_name = _deep_get(data, mapping.get("user_name_path", []))
    device_name = _deep_get(data, mapping.get("device_name_path", []))
    file_hash = _deep_get(data, mapping.get("file_hash_path", []))
    category = _deep_get(data, mapping.get("category_path", []))
    if isinstance(category, list):
        category = ", ".join(str(c) for c in category)

    dedup_key = _compute_dedup_key(alert_id, payload.tenant_id, str(title))

    canonical = CanonicalAlert(
        alert_id=alert_id,
        tenant_id=payload.tenant_id,
        timestamp=timestamp,
        source=vendor,
        normalized_title=str(title),
        severity=severity,
        raw_data=data,
        mitre_tactic=str(mitre_tactic) if mitre_tactic else None,
        mitre_technique=str(mitre_technique) if mitre_technique else None,
        ip_address=str(ip_address) if ip_address else None,
        user_name=str(user_name) if user_name else None,
        device_name=str(device_name) if device_name else None,
        file_hash=str(file_hash) if file_hash else None,
        category=str(category) if category else None,
        dedup_key=dedup_key,
    )

    log.info(
        "alert_normalized",
        vendor=vendor,
        alert_id=alert_id,
        tenant_id=payload.tenant_id,
        severity=severity.value,
    )

    return canonical
