#!/usr/bin/env python3
"""
Fit and export just the TargetEncoder from the GUIDE training CSV.

This is a fast alternative to running the full preprocessing pipeline
when only the encoder needs to be (re-)generated.  It reads only the
categorical columns + IncidentGrade from the CSV, skipping all other
feature engineering transformations.

The encoder output is identical to what process_guide_dataset produces
because TargetEncoder only sees:
  - column names (unchanged)
  - string values (filled with __MISSING__ for NaN)
  - target labels

Usage:
    python scripts/fit_encoder_only.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import pandas as pd
import category_encoders as ce

# The 37 categorical columns used by the triage model (matches training pipeline)
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

GUIDE_TRAIN_CSV = ROOT / "datasets" / "GUIDE" / "GUIDE_Train.csv"
OUTPUT_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models" / "triage"


def main() -> None:
    print(f"Loading categorical columns + target from {GUIDE_TRAIN_CSV} ...")
    print("(Reading only needed columns for speed)")

    # Only load the columns we need — much faster for a 9.5M row CSV
    cols_to_read = _CATEGORICAL_COLS + ["IncidentGrade"]
    df = pd.read_csv(
        GUIDE_TRAIN_CSV,
        usecols=lambda c: c in cols_to_read,
        low_memory=False,
    )
    print(f"Loaded {len(df):,} rows × {len(df.columns)} columns")

    # Drop rows with null target
    df = df.dropna(subset=["IncidentGrade"])
    y = df["IncidentGrade"].map(TARGET_MAP).astype(int).values
    print(f"Target distribution: {dict(zip(*[x.tolist() for x in __import__('numpy').unique(y, return_counts=True)]))}")

    # Fill NaN with __MISSING__ (matches training pipeline)
    X = df[_CATEGORICAL_COLS].copy()
    for col in _CATEGORICAL_COLS:
        if col in X.columns:
            X[col] = X[col].fillna("__MISSING__").astype(str)
        else:
            X[col] = "__MISSING__"

    print(f"Fitting TargetEncoder on {len(X):,} rows × {len(_CATEGORICAL_COLS)} columns ...")
    encoder = ce.TargetEncoder(
        cols=_CATEGORICAL_COLS,
        handle_unknown="value",
        handle_missing="value",
        min_samples_leaf=1,
        smoothing=1.0,
    )
    encoder.fit(X, y)
    print("TargetEncoder fitted successfully.")

    # Save to processed dir (matches process_guide_dataset output)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    enc_path = OUTPUT_DIR / "triage_encoder.pkl"
    joblib.dump(encoder, enc_path)
    print(f"Saved → {enc_path}")

    # Also copy to models/triage/ so migrate_to_mlflow.py can upload it
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    import shutil
    model_enc_path = MODEL_DIR / "triage_encoder.pkl"
    shutil.copy2(enc_path, model_enc_path)
    print(f"Copied → {model_enc_path}")

    print("\nDone! Run:  python MLOps/migrate_to_mlflow.py  to upload to MinIO.")


if __name__ == "__main__":
    main()
