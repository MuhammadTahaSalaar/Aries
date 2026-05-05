"""
ARIES — Triage feature engineering for real-time inference.

Converts a CanonicalAlert into a 49-dim float feature vector compatible
with the trained XGBoost ONNX model.

NOTE: This mirrors the logic in src/triage/feature_engineering.py from the
training codebase but is optimised for single-alert, low-latency inference
rather than batch dataset processing.
"""

from __future__ import annotations

import hashlib
from datetime import datetime
from typing import Any

from src.shared.logging import get_logger

log = get_logger("triage_features")

# The 49 features in the exact order the model expects.
# 12 numeric + 37 categorical (target-encoded to float during training).
FEATURE_NAMES: list[str] = [
    "hour_of_day", "day_of_week", "month",
    "has_mitre", "mitre_technique_count",
    "has_EmailClusterId", "has_ThreatFamily", "has_ResourceType",
    "has_Roles", "has_AntispamDirection", "has_SuspicionLevel", "has_LastVerdict",
    # Target-encoded categoricals (37 features) → hashed to a float representation
    "OrgId", "DetectorId", "AlertTitle", "Category", "EntityType",
    "EvidenceRole", "DeviceId", "Sha256", "IpAddress", "Url",
    "AccountSid", "AccountUpn", "AccountObjectId", "AccountName",
    "DeviceName", "NetworkMessageId", "EmailClusterId", "RegistryKey",
    "RegistryValueName", "RegistryValueData", "ApplicationId",
    "ApplicationName", "OAuthApplicationId", "ThreatFamily", "FileName",
    "FolderPath", "ResourceIdName", "ResourceType", "Roles",
    "OSFamily", "OSVersion", "AntispamDirection", "SuspicionLevel",
    "LastVerdict", "CountryCode", "State", "City",
]

SUSPICION_ORDINAL = {"Low": 0.0, "Medium": 0.33, "High": 0.66, "Critical": 1.0}

FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "AlertTitle": ("normalized_title",),
    "Category": ("category",),
    "EntityType": ("entity_type",),
    "Sha256": ("file_hash",),
    "IpAddress": ("ip_address",),
    "Url": ("url",),
    "DeviceName": ("device_name",),
    "ThreatFamily": ("threat_family",),
    "SuspicionLevel": ("suspicion_level", "severity"),
}

RAW_PATH_ALIASES: dict[str, tuple[tuple[str, ...], ...]] = {
    "AlertTitle": (("rule", "description"),),
    "Category": (("rule", "groups"), ("event", "category")),
    "DeviceName": (("agent", "name"), ("host", "name")),
    "IpAddress": (("data", "srcip"), ("source", "ip")),
    "AccountName": (("data", "dstuser"), ("data", "srcuser"), ("user", "name")),
    "FileName": (("data", "filename"),),
    "FolderPath": (("data", "path"),),
}


def _hash_to_float(value: str | None, seed: int = 0) -> float:
    """Deterministic hash of a string to a float in [0, 1).

    Approximates target-encoding for inference when the full encoder
    is not available. In production, the TargetEncoder fitted during
    training should be loaded from S3 alongside the ONNX model.
    """
    if not value:
        return 0.0
    digest = hashlib.md5(f"{seed}:{value}".encode()).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


def _deep_get(data: dict[str, Any], path: tuple[str, ...]) -> Any:
    """Safely fetch a nested value from a dict using a path."""
    current: Any = data
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
        if current is None:
            return None
    return current


def _resolve_categorical_value(alert: dict[str, Any], raw: dict[str, Any], field_name: str) -> str | None:
    """Resolve a categorical field from GUIDE, canonical, or vendor-normalized aliases."""
    direct_value = raw.get(field_name) or alert.get(field_name.lower()) or alert.get(_snake(field_name))
    if direct_value is not None:
        if isinstance(direct_value, list):
            return ", ".join(str(item) for item in direct_value if item is not None) or None
        return str(direct_value)

    for alias in FIELD_ALIASES.get(field_name, ()):
        alias_value = alert.get(alias)
        if alias_value is not None:
            return str(alias_value)

    for path in RAW_PATH_ALIASES.get(field_name, ()):
        path_value = _deep_get(raw, path)
        if path_value is not None:
            if isinstance(path_value, list):
                return ", ".join(str(item) for item in path_value if item is not None) or None
            return str(path_value)

    return None


def extract_features(alert: dict[str, Any], encoder: object | None = None) -> list[float]:
    """
    Convert a canonical alert dict into a 49-element float vector.

    Mirrors the training feature engineering pipeline for online inference.
    When ``encoder`` (a fitted ``category_encoders.TargetEncoder``) is provided
    it is used for categorical encoding; otherwise a deterministic hash
    approximation is used as a fallback.
    """
    raw = alert.get("raw_data", {})
    ts = alert.get("timestamp")
    if isinstance(ts, str):
        try:
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except ValueError:
            dt = datetime.utcnow()
    elif isinstance(ts, datetime):
        dt = ts
    else:
        dt = datetime.utcnow()

    mitre_tech = alert.get("mitre_technique") or raw.get("MitreTechniques", "")
    mitre_list = [t.strip() for t in mitre_tech.split(",") if t.strip()] if mitre_tech else []

    features: list[float] = []

    # 1-3: Temporal
    features.append(float(dt.hour))
    features.append(float(dt.weekday()))
    features.append(float(dt.month))

    # 4: has_mitre (binary)
    features.append(1.0 if mitre_list else 0.0)
    # 5: mitre_technique_count
    features.append(float(len(mitre_list)))

    # 6-12: Null indicator flags
    for field in [
        "EmailClusterId", "ThreatFamily", "ResourceType",
        "Roles", "AntispamDirection", "SuspicionLevel", "LastVerdict",
    ]:
        val = raw.get(field) or alert.get(field.lower()) or alert.get(_snake(field))
        features.append(1.0 if val else 0.0)

    # 13-49: Categorical features (37 fields)
    categorical_fields = FEATURE_NAMES[12:]

    # Collect raw string values for every categorical column
    cat_values: dict[str, str] = {}
    for fname in categorical_fields:
        val = _resolve_categorical_value(alert, raw, fname)
        cat_values[fname] = val if val else "__MISSING__"

    if encoder is not None:
        # Use the TargetEncoder fitted during training for correct encoding
        try:
            import pandas as pd
            cat_df = pd.DataFrame([cat_values])
            cat_encoded = encoder.transform(cat_df)
            for fname in categorical_fields:
                features.append(float(cat_encoded[fname].iloc[0]))
        except Exception:
            log.warning("encoder_transform_failed_using_hash_fallback")
            for i, fname in enumerate(categorical_fields):
                val = cat_values[fname]
                if fname == "SuspicionLevel" and val in SUSPICION_ORDINAL:
                    features.append(SUSPICION_ORDINAL[val])
                else:
                    features.append(_hash_to_float(val if val != "__MISSING__" else None, seed=i))
    else:
        # Fallback: deterministic hash (approximation — use encoder in production)
        for i, fname in enumerate(categorical_fields):
            val = cat_values[fname]
            if fname == "SuspicionLevel" and val in SUSPICION_ORDINAL:
                features.append(SUSPICION_ORDINAL[val])
            else:
                features.append(_hash_to_float(val if val != "__MISSING__" else None, seed=i))

    assert len(features) == 49, f"Expected 49 features, got {len(features)}"
    return features


def _snake(name: str) -> str:
    """CamelCase to snake_case."""
    import re
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()
