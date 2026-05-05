#!/usr/bin/env python3
"""
cleanup_mlflow.py
=================
Prune old MLflow runs from each experiment, keeping only the latest
FINISHED run per experiment.  This deletes both the MLflow metadata
(PostgreSQL) and the artifact files (MinIO), freeing disk space.

Usage:
  python MLOps/cleanup_mlflow.py             # dry-run (default)
  python MLOps/cleanup_mlflow.py --execute   # actually delete

Prerequisites:
  cd MLOps && docker compose up -d
"""

from __future__ import annotations

import argparse
import os
import sys

import boto3
from mlflow import MlflowClient

# ── Configuration (matches MLOps/.env) ───────────────────────────────────────

MLFLOW_TRACKING_URI = "http://localhost:5000"
MLFLOW_S3_ENDPOINT_URL = "http://localhost:9000"
S3_BUCKET = "mlflow-bucket"
AWS_ACCESS_KEY_ID = "admin"
AWS_SECRET_ACCESS_KEY = "password123"

os.environ["MLFLOW_S3_ENDPOINT_URL"] = MLFLOW_S3_ENDPOINT_URL
os.environ["MLFLOW_S3_IGNORE_TLS"] = "true"
os.environ["AWS_ACCESS_KEY_ID"] = AWS_ACCESS_KEY_ID
os.environ["AWS_SECRET_ACCESS_KEY"] = AWS_SECRET_ACCESS_KEY

# Experiments used by Aries
ARIES_EXPERIMENTS = [
    "aries/triage-classifier",
    "aries/ner",
    "aries/bart-summarizer",
]


def get_s3_client():
    return boto3.client(
        "s3",
        endpoint_url=MLFLOW_S3_ENDPOINT_URL,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    )


def get_artifact_size_bytes(s3, experiment_id: str, run_id: str) -> int:
    """Sum all object sizes under a run's artifact prefix in S3."""
    prefix = f"{experiment_id}/{run_id}/"
    total = 0
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []):
            total += obj["Size"]
    return total


def delete_s3_prefix(s3, experiment_id: str, run_id: str) -> int:
    """Delete all S3 objects under a run's prefix.  Returns count deleted."""
    prefix = f"{experiment_id}/{run_id}/"
    deleted = 0
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=prefix):
        objects = [{"Key": obj["Key"]} for obj in page.get("Contents", [])]
        if objects:
            s3.delete_objects(Bucket=S3_BUCKET, Delete={"Objects": objects})
            deleted += len(objects)
    return deleted


def human_size(nbytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(nbytes) < 1024:
            return f"{nbytes:.1f} {unit}"
        nbytes /= 1024
    return f"{nbytes:.1f} TB"


def main() -> None:
    parser = argparse.ArgumentParser(description="Prune old MLflow runs, keep latest per experiment.")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually delete runs.  Without this flag, only a dry-run report is shown.",
    )
    args = parser.parse_args()

    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    s3 = get_s3_client()

    try:
        client.search_experiments()
    except Exception as exc:
        print(f"ERROR: Cannot reach MLflow at {MLFLOW_TRACKING_URI}: {exc}")
        print("Make sure you have run:  cd MLOps && docker compose up -d")
        sys.exit(1)

    total_freed = 0
    total_deleted_runs = 0

    for exp_name in ARIES_EXPERIMENTS:
        exp = client.get_experiment_by_name(exp_name)
        if exp is None:
            print(f"\n  [{exp_name}]  Not found – skipping")
            continue

        exp_id = exp.experiment_id
        print(f"\n  [{exp_name}]  (experiment_id={exp_id})")

        # Get all runs sorted by end time (newest first)
        runs = client.search_runs(
            experiment_ids=[exp_id],
            order_by=["attributes.end_time DESC"],
        )

        if not runs:
            print("    No runs found")
            continue

        # Keep the latest run, mark the rest for deletion
        latest = runs[0]
        latest_size = get_artifact_size_bytes(s3, exp_id, latest.info.run_id)
        print(f"    KEEP  run_id={latest.info.run_id}  "
              f"({latest.info.run_name or 'unnamed'})  "
              f"artifacts={human_size(latest_size)}")

        to_delete = runs[1:]
        if not to_delete:
            print("    No old runs to prune")
            continue

        for run in to_delete:
            rid = run.info.run_id
            rname = run.info.run_name or "unnamed"
            size = get_artifact_size_bytes(s3, exp_id, rid)
            total_freed += size

            if args.execute:
                # Delete artifacts from MinIO
                n_objects = delete_s3_prefix(s3, exp_id, rid)
                # Delete the run from MLflow (PostgreSQL)
                client.delete_run(rid)
                print(f"    DEL   run_id={rid}  ({rname})  "
                      f"freed={human_size(size)}  objects={n_objects}")
            else:
                print(f"    WOULD DELETE  run_id={rid}  ({rname})  "
                      f"size={human_size(size)}")
            total_deleted_runs += 1

    # Also clean up any non-Aries experiments (e.g. test runs)
    all_experiments = client.search_experiments()
    other_exps = [e for e in all_experiments if e.name not in ARIES_EXPERIMENTS and e.name != "Default"]
    if other_exps:
        print("\n  [Other experiments]")
        for exp in other_exps:
            runs = client.search_runs(experiment_ids=[exp.experiment_id])
            for run in runs:
                rid = run.info.run_id
                size = get_artifact_size_bytes(s3, exp.experiment_id, rid)
                total_freed += size
                total_deleted_runs += 1
                if args.execute:
                    n_objects = delete_s3_prefix(s3, exp.experiment_id, rid)
                    client.delete_run(rid)
                    print(f"    DEL   [{exp.name}] run_id={rid}  freed={human_size(size)}")
                else:
                    print(f"    WOULD DELETE  [{exp.name}] run_id={rid}  size={human_size(size)}")

    sep = "=" * 50
    print(f"\n{sep}")
    if args.execute:
        print(f"  Deleted {total_deleted_runs} run(s), freed ~{human_size(total_freed)}")
    else:
        print(f"  DRY RUN: Would delete {total_deleted_runs} run(s), freeing ~{human_size(total_freed)}")
        print("  Re-run with --execute to apply changes")
    print(sep)


if __name__ == "__main__":
    main()
