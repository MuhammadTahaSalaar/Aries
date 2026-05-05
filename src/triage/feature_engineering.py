"""
ARIES — GUIDE dataset feature engineering.

Full pipeline for the Microsoft GUIDE dataset (45 raw columns → ~49 features).
Uses target encoding (fit on train only) and engineered MITRE / timestamp features.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.shared.utils import compute_file_hash

logger = logging.getLogger(__name__)

# ── Column taxonomy ──────────────────────────────────────────────────────────

# Drop: identifiers and secondary-task labels (99 % null)
_DROP_COLS = [
    "Id", "IncidentId", "AlertId",
    "ActionGrouped", "ActionGranular",
]

# Columns with >50 % nulls — also get a binary has_* indicator
_HIGH_NULL_COLS = [
    "MitreTechniques",   # 59 %
    "EmailClusterId",    # 99 %
    "ThreatFamily",      # 99 %
    "ResourceType",      # 99 %
    "Roles",             # 98 %
    "AntispamDirection", # 98 %
    "SuspicionLevel",    # 85 %
    "LastVerdict",       # 77 %
]

# Categoricals to target-encode (MitreTechniques excluded — multi-value)
_CATEGORICAL_COLS = [
    "OrgId", "DetectorId", "AlertTitle", "Category",
    "EntityType", "EvidenceRole",
    "DeviceId", "Sha256", "IpAddress", "Url",
    "AccountSid", "AccountUpn", "AccountObjectId", "AccountName",
    "DeviceName", "NetworkMessageId", "EmailClusterId",
    "RegistryKey", "RegistryValueName", "RegistryValueData",
    "ApplicationId", "ApplicationName", "OAuthApplicationId",
    "ThreatFamily", "FileName", "FolderPath",
    "ResourceIdName", "ResourceType", "Roles",
    "OSFamily", "OSVersion",
    "AntispamDirection", "SuspicionLevel", "LastVerdict",
    "CountryCode", "State", "City",
]

TARGET_MAP = {"BenignPositive": 0, "FalsePositive": 1, "TruePositive": 2}
LABEL_NAMES = {v: k for k, v in TARGET_MAP.items()}


# ── Loading ──────────────────────────────────────────────────────────────────

def load_guide(train_path: Path, test_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load GUIDE CSVs, drop rows with null target."""
    logger.info("Loading GUIDE  train=%s  test=%s", train_path, test_path)
    train = pd.read_csv(train_path, low_memory=False)
    test = pd.read_csv(test_path, low_memory=False)
    for name, df in [("train", train), ("test", test)]:
        if "IncidentGrade" not in df.columns:
            logger.error("IncidentGrade missing in %s", name)
            sys.exit(1)
    n_before = len(train)
    train = train.dropna(subset=["IncidentGrade"])
    test = test.dropna(subset=["IncidentGrade"])
    if n_before != len(train):
        logger.warning("Dropped %d train rows with null target", n_before - len(train))
    logger.info("Loaded  train=%s  test=%s", f"{len(train):,}", f"{len(test):,}")
    return train, test


# ── Feature engineering helpers ──────────────────────────────────────────────

def _engineer_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    if "Timestamp" not in df.columns:
        return df
    ts = pd.to_datetime(df["Timestamp"], errors="coerce", utc=True)
    df = df.copy()
    df["hour_of_day"] = ts.dt.hour.astype("float32")
    df["day_of_week"] = ts.dt.dayofweek.astype("float32")
    df["month"] = ts.dt.month.astype("float32")
    return df.drop(columns=["Timestamp"])


def _engineer_mitre(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    mitre = df.get("MitreTechniques", pd.Series(dtype=str))
    df["has_mitre"] = mitre.notna().astype("float32")
    df["mitre_technique_count"] = (
        mitre.fillna("").apply(lambda x: len(x.split(";")) if x else 0).astype("float32")
    )
    if "MitreTechniques" in df.columns:
        df = df.drop(columns=["MitreTechniques"])
    return df


def _add_null_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in _HIGH_NULL_COLS:
        if col in df.columns and col != "MitreTechniques":
            df[f"has_{col}"] = df[col].notna().astype("float32")
    return df


def _fill_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in _CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].fillna("__MISSING__").astype(str)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Full feature engineering on raw GUIDE dataframe."""
    df = df.copy()
    drop_present = [c for c in _DROP_COLS if c in df.columns]
    df = df.drop(columns=drop_present)
    df = _engineer_timestamp(df)
    df = _engineer_mitre(df)
    df = _add_null_indicators(df)
    df = _fill_categoricals(df)
    logger.info("Features engineered → %d columns", len(df.columns))
    return df


# ── Target encoding ──────────────────────────────────────────────────────────

def apply_target_encoding(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    cat_cols: list[str],
) -> tuple[np.ndarray, np.ndarray, Any]:
    """Target-encode categoricals (fit on train only). Returns (Xtr, Xte, encoder)."""
    import category_encoders as ce

    logger.info("Target-encoding %d categorical columns...", len(cat_cols))
    encoder = ce.TargetEncoder(
        cols=cat_cols,
        handle_unknown="value",
        handle_missing="value",
        min_samples_leaf=1,
        smoothing=1.0,
    )
    X_tr = encoder.fit_transform(X_train[cat_cols], y_train).values.astype("float32")
    X_te = encoder.transform(X_test[cat_cols]).values.astype("float32")
    logger.info("Target encoding done.")
    return X_tr, X_te, encoder


# ── Main pipeline ────────────────────────────────────────────────────────────

def process_guide_dataset(
    guide_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """
    End-to-end GUIDE preprocessing.

    Returns dict with X_train, y_train, X_test, y_test, metadata, feature_names,
    and saves .npz + metadata JSON to output_dir.
    """
    logger.info("=" * 60)
    logger.info("GUIDE TABULAR PREPROCESSING")
    logger.info("=" * 60)

    train_path = guide_dir / "GUIDE_Train.csv"
    test_path = guide_dir / "GUIDE_Test.csv"
    raw_hash = compute_file_hash(train_path)[:16] + compute_file_hash(test_path)[:16]

    # 1. Load
    train_df, test_df = load_guide(train_path, test_path)

    # 2. Separate target
    y_train_raw = train_df["IncidentGrade"]
    y_test_raw = test_df["IncidentGrade"]
    train_df = train_df.drop(columns=["IncidentGrade"])
    test_df = test_df.drop(columns=["IncidentGrade"])
    y_train = y_train_raw.map(TARGET_MAP).astype(int).values
    y_test = y_test_raw.map(TARGET_MAP).astype(int).values

    # 3. Feature engineering
    train_eng = engineer_features(train_df)
    test_eng = engineer_features(test_df)

    cat_cols = [c for c in _CATEGORICAL_COLS if c in train_eng.columns]
    num_cols = [c for c in train_eng.columns if c not in cat_cols]
    logger.info("Categorical: %d  Numeric: %d", len(cat_cols), len(num_cols))

    # 4. Target-encode
    X_train_cat, X_test_cat, encoder = apply_target_encoding(
        train_eng, y_train, test_eng, cat_cols
    )

    # Save encoder so it can be used for real-time inference
    import joblib
    encoder_path = output_dir / "triage_encoder.pkl"
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(encoder, encoder_path)
    logger.info("Saved TargetEncoder → %s", encoder_path)

    # 5. Combine
    X_train_num = train_eng[num_cols].values.astype("float32")
    X_test_num = test_eng[num_cols].values.astype("float32")
    X_train = np.concatenate([X_train_num, X_train_cat], axis=1)
    X_test = np.concatenate([X_test_num, X_test_cat], axis=1)
    feature_names = num_cols + cat_cols

    logger.info("Final matrix: %d features  train=%s  test=%s",
                X_train.shape[1], f"{X_train.shape[0]:,}", f"{X_test.shape[0]:,}")

    # 6. Save
    output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = output_dir / "triage_data.npz"
    np.savez_compressed(
        npz_path,
        X_train=X_train, y_train=y_train,
        X_test=X_test, y_test=y_test,
    )
    logger.info("Saved %s", npz_path)

    metadata = {
        "raw_data_hash": raw_hash,
        "encoding_method": "TargetEncoder",
        "n_features": int(X_train.shape[1]),
        "n_categorical": len(cat_cols),
        "n_numeric": len(num_cols),
        "feature_names": feature_names,
        "target_map": TARGET_MAP,
        "train_samples": int(X_train.shape[0]),
        "test_samples": int(X_test.shape[0]),
        "class_distribution": {
            int(k): int(v) for k, v in zip(*np.unique(y_train, return_counts=True))
        },
    }
    from src.shared.utils import save_json
    save_json(metadata, output_dir / "triage_metadata.json")

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "metadata": metadata,
        "feature_names": feature_names,
        "encoder": encoder,
    }
