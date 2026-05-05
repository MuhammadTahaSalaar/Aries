#!/usr/bin/env python3
"""
migrate_to_mlflow.py
====================
Migrates all local training data to the Docker-hosted MLflow stack:
  • PostgreSQL  – stores experiment/run metadata (params, metrics, tags)
  • MinIO       – stores artifact files (model weights, plots, JSON, ONNX)
  • MLflow UI   – http://localhost:5000

What this script does
---------------------
Step 1 – Replay existing mlruns/ data
    Reads the file-based mlruns/ directory (experiments, runs, metrics,
    params, tags, artifacts) and re-creates them on the remote MLflow
    server so that all history is preserved.

Step 2 – Log NER model (SecureBERT fine-tuned)
    Creates experiment  aries/ner  and logs model files from models/ner/
    together with metrics/params from model_metadata.json.

Step 3 – Log BART summarizer
    Creates experiment  aries/bart-summarizer  and logs model files from
    models/summarizer/ together with metrics/params from model_metadata.json.

Step 4 – Upload Triage model binaries & ONNX exports
    Adds a dedicated upload run to  aries/triage-classifier  that contains
    the final XGBoost model file and all ONNX exports.

Prerequisites
-------------
1.  cd MLOps && docker compose up -d
2.  Wait ~30 s for all containers to be healthy.
3.  pip install mlflow boto3 pyyaml   (or activate your existing venv)
4.  python MLOps/migrate_to_mlflow.py
"""

import json
import os
import time
from pathlib import Path

import boto3
import mlflow
import yaml
from mlflow import MlflowClient
from mlflow.entities import Metric, Param, RunTag

# ──────────────────────────────────────────────────────────────────────────────
# Configuration – matches MLOps/.env
# ──────────────────────────────────────────────────────────────────────────────

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]   # …/Aries

MLFLOW_TRACKING_URI    = "http://localhost:5000"
MLFLOW_S3_ENDPOINT_URL = "http://localhost:9000"
AWS_ACCESS_KEY_ID      = "admin"
AWS_SECRET_ACCESS_KEY  = "password123"

# Propagate S3/MinIO credentials so boto3 can reach the local MinIO container
os.environ["MLFLOW_S3_ENDPOINT_URL"] = MLFLOW_S3_ENDPOINT_URL
os.environ["MLFLOW_S3_IGNORE_TLS"]   = "true"
os.environ["AWS_ACCESS_KEY_ID"]      = AWS_ACCESS_KEY_ID
os.environ["AWS_SECRET_ACCESS_KEY"]  = AWS_SECRET_ACCESS_KEY

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = MlflowClient()

# MLflow run status integer → string (as stored in meta.yaml)
STATUS_MAP = {
    1: "RUNNING",
    2: "SCHEDULED",
    3: "FINISHED",
    4: "FAILED",
    5: "KILLED",
}

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def get_or_create_experiment(name: str) -> str:
    """Return existing experiment id or create a new one."""
    exp = client.get_experiment_by_name(name)
    if exp is not None:
        print(f"    Experiment '{name}' already exists (id={exp.experiment_id})")
        return exp.experiment_id
    exp_id = client.create_experiment(name)
    print(f"    Created experiment '{name}' (id={exp_id})")
    return exp_id


def read_text_file(path: Path) -> str:
    """Read a single-value MLflow param/tag file."""
    return path.read_text().strip()


def parse_metric_file(path: Path) -> list[Metric]:
    """
    Parse an MLflow file-store metric file.
    Each line:  <timestamp_ms> <value> <step>
    Returns a list of mlflow.entities.Metric objects.
    """
    metrics = []
    for line in path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) == 3:
            ts, val, step = int(parts[0]), float(parts[1]), int(parts[2])
            metrics.append(Metric(path.name, val, ts, step))
    return metrics


# ──────────────────────────────────────────────────────────────────────────────
# Step 1 – Replay existing mlruns/ data
# ──────────────────────────────────────────────────────────────────────────────

def migrate_mlruns() -> None:
    """
    Walk the local mlruns/ directory and replay every experiment / run on the
    remote MLflow server.  Metrics, params, tags, and artifacts are all sent.
    """
    mlruns_dir = WORKSPACE_ROOT / "mlruns"

    for exp_dir in sorted(mlruns_dir.iterdir()):
        if not exp_dir.is_dir() or exp_dir.name.startswith("."):
            continue

        meta_yaml_path = exp_dir / "meta.yaml"
        if not meta_yaml_path.exists():
            # Experiments 1 and 3 are empty MLflow placeholders – skip
            continue

        exp_meta = yaml.safe_load(meta_yaml_path.read_text())
        exp_name = exp_meta.get("name", f"experiment-{exp_dir.name}")

        print(f"\n  [Experiment] {exp_name}  (local id: {exp_dir.name})")
        exp_id = get_or_create_experiment(exp_name)

        for run_dir in sorted(exp_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            run_meta_path = run_dir / "meta.yaml"
            if not run_meta_path.exists():
                continue

            run_meta   = yaml.safe_load(run_meta_path.read_text())
            run_name   = run_meta.get("run_name", run_dir.name)
            status_int = run_meta.get("status", 3)
            start_time = run_meta.get("start_time", int(time.time() * 1000))
            end_time   = run_meta.get("end_time")
            status_str = STATUS_MAP.get(status_int, "FINISHED")

            print(f"\n    [Run] {run_name}  (local id: {run_dir.name})  status={status_str}")

            # ── Create the run ─────────────────────────────────────────────
            run = client.create_run(
                experiment_id=exp_id,
                run_name=run_name,
                start_time=start_time,
            )
            rid = run.info.run_id

            # ── Tags ───────────────────────────────────────────────────────
            tags_dir = run_dir / "tags"
            tags: list[RunTag] = []
            if tags_dir.exists():
                for tag_file in tags_dir.iterdir():
                    if tag_file.is_file():
                        tags.append(RunTag(tag_file.name, read_text_file(tag_file)))
            if tags:
                client.log_batch(rid, tags=tags)
                print(f"      Logged {len(tags)} tag(s)")

            # ── Params ─────────────────────────────────────────────────────
            params_dir = run_dir / "params"
            params: list[Param] = []
            if params_dir.exists():
                for pf in params_dir.iterdir():
                    if pf.is_file():
                        params.append(Param(pf.name, read_text_file(pf)))
            if params:
                client.log_batch(rid, params=params)
                print(f"      Logged {len(params)} param(s)")

            # ── Metrics ────────────────────────────────────────────────────
            metrics_dir = run_dir / "metrics"
            all_metrics: list[Metric] = []
            if metrics_dir.exists():
                for mf in metrics_dir.iterdir():
                    if mf.is_file():
                        all_metrics.extend(parse_metric_file(mf))
            if all_metrics:
                # log_batch accepts at most 1000 items at once
                chunk_size = 1000
                for i in range(0, len(all_metrics), chunk_size):
                    client.log_batch(rid, metrics=all_metrics[i : i + chunk_size])
                print(f"      Logged {len(all_metrics)} metric data point(s)")

            # ── Artifacts ──────────────────────────────────────────────────
            artifacts_dir = run_dir / "artifacts"
            if artifacts_dir.exists() and any(artifacts_dir.rglob("*")):
                print(f"      Uploading artifacts: {artifacts_dir} → MinIO ...")
                client.log_artifacts(rid, str(artifacts_dir))
                print("      Artifacts uploaded")
            else:
                print("      No artifacts to upload")

            # ── Finalise run ───────────────────────────────────────────────
            if status_str in ("FINISHED", "FAILED", "KILLED"):
                client.set_terminated(rid, status=status_str, end_time=end_time)
            else:
                client.set_terminated(rid, status="FINISHED")

            print(f"      → Remote run_id: {rid}")


# ──────────────────────────────────────────────────────────────────────────────
# Step 2 – Log NER model (SecureBERT fine-tuned on CyNER)
# ──────────────────────────────────────────────────────────────────────────────

def log_ner_model() -> None:
    models_dir = WORKSPACE_ROOT / "models" / "ner"
    meta_path  = models_dir / "model_metadata.json"
    if not meta_path.exists():
        print("  [NER] model_metadata.json not found – skipping.")
        return

    meta     = json.loads(meta_path.read_text())
    exp_name = "aries/ner"
    print(f"\n  [Experiment] {exp_name}")
    exp_id = get_or_create_experiment(exp_name)

    with mlflow.start_run(experiment_id=exp_id, run_name="secureBERT-ner") as run:
        rid = run.info.run_id

        # Params
        mlflow.log_param("model_type", meta.get("model_type", ""))
        mlflow.log_param("base_model", meta.get("base_model", ""))
        mlflow.log_param("task",       meta.get("task", ""))
        for k, v in meta.get("params", {}).items():
            mlflow.log_param(k, v)

        # Metrics
        for k, v in meta.get("metrics", {}).items():
            mlflow.log_metric(k, v)

        # Tags
        mlflow.set_tags({
            "pipeline":   "ner",
            "model_type": meta.get("model_type", ""),
            "base_model": meta.get("base_model", ""),
        })

        # Artifacts – upload fine-tuned PyTorch model files
        # Skip the large pretrained base weights (secureBERT_pretrained/)
        SKIP = {"secureBERT_pretrained"}
        print("    Uploading NER model files → MinIO ...")
        for item in sorted(models_dir.iterdir()):
            if item.name in SKIP:
                print(f"      Skipping {item.name}/")
                continue
            if item.is_file():
                mlflow.log_artifact(str(item))
            elif item.is_dir():
                mlflow.log_artifacts(str(item), artifact_path=item.name)

        # ONNX exports for NER (ner.onnx = full precision, ner.opt.onnx = optimised)
        onnx_dir = WORKSPACE_ROOT / "models" / "onnx"
        ner_onnx_files = [onnx_dir / "ner.onnx", onnx_dir / "ner.opt.onnx"]
        uploaded_onnx = 0
        for onnx_file in ner_onnx_files:
            if onnx_file.exists():
                print(f"    Uploading {onnx_file.name} → MinIO (onnx/) ...")
                mlflow.log_artifact(str(onnx_file), artifact_path="onnx")
                uploaded_onnx += 1
        if uploaded_onnx == 0:
            print("    WARNING: No NER ONNX files found in models/onnx/")

        print(f"    Done – remote run_id: {rid}")


# ──────────────────────────────────────────────────────────────────────────────
# Step 3 – Log BART summarizer
# ──────────────────────────────────────────────────────────────────────────────

def log_summarizer_model() -> None:
    models_dir = WORKSPACE_ROOT / "models" / "summarizer"
    meta_path  = models_dir / "model_metadata.json"
    if not meta_path.exists():
        print("  [Summarizer] model_metadata.json not found – skipping.")
        return

    meta     = json.loads(meta_path.read_text())
    exp_name = "aries/bart-summarizer"
    print(f"\n  [Experiment] {exp_name}")
    exp_id = get_or_create_experiment(exp_name)

    with mlflow.start_run(experiment_id=exp_id, run_name="bart-summarizer") as run:
        rid = run.info.run_id

        # Params
        mlflow.log_param("model_type", meta.get("model_type", ""))
        mlflow.log_param("base_model", meta.get("base_model", ""))
        mlflow.log_param("task",       meta.get("task", ""))
        for k, v in meta.get("params", {}).items():
            mlflow.log_param(k, v)

        # Metrics
        for k, v in meta.get("metrics", {}).items():
            mlflow.log_metric(k, v)

        # Tags
        mlflow.set_tags({
            "pipeline":   "summarizer",
            "model_type": meta.get("model_type", ""),
            "base_model": meta.get("base_model", ""),
        })

        # Artifacts – upload fine-tuned PyTorch model files
        print("    Uploading summarizer model files → MinIO ...")
        mlflow.log_artifacts(str(models_dir))

        # ONNX exports for summarizer (encoder + decoder, with optimised variants)
        # Per AI_DESIGN.md §2: bart_encoder.onnx + bart_decoder.onnx
        onnx_summarizer_dir = WORKSPACE_ROOT / "models" / "onnx" / "summarizer"
        if onnx_summarizer_dir.exists() and any(onnx_summarizer_dir.iterdir()):
            print("    Uploading summarizer ONNX exports → MinIO (onnx/summarizer/) ...")
            mlflow.log_artifacts(str(onnx_summarizer_dir), artifact_path="onnx/summarizer")
        else:
            print("    WARNING: No summarizer ONNX files found in models/onnx/summarizer/")

        print(f"    Done – remote run_id: {rid}")


# ──────────────────────────────────────────────────────────────────────────────
# Step 4 – Upload Triage model binaries & ONNX exports
# ──────────────────────────────────────────────────────────────────────────────

def log_triage_model_files() -> None:
    """
    The mlruns migration (Step 1) already uploads the small artifacts that were
    logged during training (confusion_matrix.png, model_metadata.json,
    xgboost_triage.json).

    This step adds a separate, clearly labelled *model-upload* run to the same
    experiment that contains the final XGBoost model binary and all ONNX exports,
    so they are versioned and retrievable from MinIO.
    """
    triage_dir = WORKSPACE_ROOT / "models" / "triage"
    onnx_dir   = WORKSPACE_ROOT / "models" / "onnx"
    meta_path  = triage_dir / "model_metadata.json"

    if not meta_path.exists():
        print("  [Triage] model_metadata.json not found – skipping model upload.")
        return

    meta     = json.loads(meta_path.read_text())
    exp_name = "aries/triage-classifier"
    print(f"\n  [Experiment] {exp_name}")
    exp_id = get_or_create_experiment(exp_name)

    with mlflow.start_run(
        experiment_id=exp_id,
        run_name="xgboost-triage-model-upload",
    ) as run:
        rid = run.info.run_id

        # Params
        mlflow.log_param("model_type", meta.get("model_type", "xgboost"))
        mlflow.log_param("n_features", str(meta.get("n_features", "")))
        for k, v in meta.get("params", {}).items():
            mlflow.log_param(k, v)

        # Metrics
        for k, v in meta.get("metrics", {}).items():
            mlflow.log_metric(k, v)

        # Tags
        mlflow.set_tags({
            "pipeline":   "triage",
            "model_type": meta.get("model_type", "xgboost"),
        })

        # ── XGBoost model files ──────────────────────────────────────────
        print("    Uploading XGBoost model files → MinIO ...")
        mlflow.log_artifacts(str(triage_dir), artifact_path="triage_model")

        # ── Triage ONNX export only (NER + summarizer go to their own runs) ──
        # Per AI_DESIGN.md §2: models/triage/xgboost_triage.onnx
        triage_onnx = onnx_dir / "triage.onnx"
        if triage_onnx.exists():
            print("    Uploading triage.onnx → MinIO (onnx/) ...")
            mlflow.log_artifact(str(triage_onnx), artifact_path="onnx")
        else:
            print("    WARNING: triage.onnx not found in models/onnx/")

        print(f"    Done – remote run_id: {rid}")


# ──────────────────────────────────────────────────────────────────────────────
# Step 5 – Prune old runs (keep only latest per experiment in MinIO)
# ──────────────────────────────────────────────────────────────────────────────

S3_BUCKET = "mlflow-bucket"

ARIES_EXPERIMENTS = [
    "aries/triage-classifier",
    "aries/ner",
    "aries/bart-summarizer",
]


def _delete_s3_prefix(s3_client, experiment_id: str, run_id: str) -> int:
    """Delete all S3 objects under a run's prefix.  Returns count deleted."""
    prefix = f"{experiment_id}/{run_id}/"
    deleted = 0
    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=prefix):
        objects = [{"Key": obj["Key"]} for obj in page.get("Contents", [])]
        if objects:
            s3_client.delete_objects(Bucket=S3_BUCKET, Delete={"Objects": objects})
            deleted += len(objects)
    return deleted


def prune_old_runs() -> None:
    """
    For each Aries experiment, keep only the latest FINISHED run and delete
    all older runs (metadata from PostgreSQL + artifacts from MinIO).
    """
    s3 = boto3.client(
        "s3",
        endpoint_url=MLFLOW_S3_ENDPOINT_URL,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    )

    total_deleted = 0

    for exp_name in ARIES_EXPERIMENTS:
        exp = client.get_experiment_by_name(exp_name)
        if exp is None:
            continue

        exp_id = exp.experiment_id
        runs = client.search_runs(
            experiment_ids=[exp_id],
            order_by=["attributes.end_time DESC"],
        )

        if len(runs) <= 1:
            print(f"    {exp_name}: only 1 run, nothing to prune")
            continue

        latest = runs[0]
        print(f"    {exp_name}: keeping run {latest.info.run_id}, pruning {len(runs) - 1} old run(s)")

        for run in runs[1:]:
            rid = run.info.run_id
            n_objects = _delete_s3_prefix(s3, exp_id, rid)
            client.delete_run(rid)
            total_deleted += 1
            print(f"      Deleted run {rid} ({n_objects} S3 objects)")

    if total_deleted:
        print(f"\n    Pruned {total_deleted} old run(s) from MinIO + PostgreSQL")
    else:
        print("\n    Nothing to prune — all experiments have at most 1 run")


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sep = "=" * 65

    print(sep)
    print("  Aries → MLflow / PostgreSQL / MinIO migration script")
    print(sep)
    print(f"\n  Tracking URI : {MLFLOW_TRACKING_URI}")
    print(f"  Artifact store (MinIO) : {MLFLOW_S3_ENDPOINT_URL}")
    print(f"  Workspace root : {WORKSPACE_ROOT}")

    # Verify the server is reachable before doing any work
    try:
        client.search_experiments()
        print("\n  MLflow server is reachable ✓\n")
    except Exception as exc:
        print(f"\n  ERROR: Cannot reach MLflow at {MLFLOW_TRACKING_URI}")
        print(f"  Details: {exc}")
        print("\n  Make sure you have run:  cd MLOps && docker compose up -d")
        raise SystemExit(1)

    print(sep)
    print("[Step 1/5]  Replaying existing mlruns/ data")
    print(sep)
    migrate_mlruns()

    print(sep)
    print("[Step 2/5]  Logging NER model (SecureBERT)")
    print(sep)
    log_ner_model()

    print(sep)
    print("[Step 3/5]  Logging BART summarizer")
    print(sep)
    log_summarizer_model()

    print(sep)
    print("[Step 4/5]  Uploading Triage model binaries & ONNX exports")
    print(sep)
    log_triage_model_files()

    print(sep)
    print("[Step 5/5]  Pruning old runs (keeping only latest per experiment)")
    print(sep)
    prune_old_runs()

    print(f"\n{sep}")
    print("  All done!")
    print(f"  Open {MLFLOW_TRACKING_URI} to inspect your experiments.")
    print(sep)
