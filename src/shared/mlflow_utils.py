"""
ARIES — Offline MLflow tracker for firewalled HPC environments.

Uses file:// protocol so no network access to an MLflow server is needed.
All artifacts are written to local disk under mlruns/ and can be synced
to the Docker-hosted MLflow server later via rsync.
"""

from __future__ import annotations

import json
import logging
import os
import signal
import sys
from pathlib import Path
from typing import Any, Callable

import mlflow

logger = logging.getLogger(__name__)


class OfflineMLflowTracker:
    """Context-managed MLflow tracker with emergency checkpoint support."""

    def __init__(
        self,
        experiment_name: str,
        tracking_uri: str,
        run_name: str | None = None,
        checkpoint_dir: Path | None = None,
    ) -> None:
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        self.run_name = run_name
        self.checkpoint_dir = checkpoint_dir
        self._run: mlflow.ActiveRun | None = None
        self._checkpoint_fn: Callable[[], None] | None = None
        self._original_handlers: dict[int, Any] = {}

    # ── Context manager ───────────────────────────────────────────────

    def __enter__(self) -> "OfflineMLflowTracker":
        self.start_run()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.end_run(status="FAILED" if exc_type else "FINISHED")
        self._restore_signal_handlers()

    # ── Lifecycle ─────────────────────────────────────────────────────

    def start_run(self) -> None:
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)
        self._run = mlflow.start_run(run_name=self.run_name)
        self._install_signal_handlers()
        logger.info(
            "MLflow run started  experiment=%s  run_id=%s",
            self.experiment_name,
            self._run.info.run_id,
        )

    def end_run(self, status: str = "FINISHED") -> None:
        if self._run is not None:
            mlflow.end_run(status=status)
            logger.info("MLflow run ended  status=%s", status)
            self._run = None

    # ── Logging helpers ───────────────────────────────────────────────

    def log_params(self, params: dict[str, Any]) -> None:
        # MLflow truncates values > 500 chars — stringify safely
        safe = {k: str(v)[:500] for k, v in params.items()}
        mlflow.log_params(safe)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        mlflow.log_metrics(metrics, step=step)

    def log_metric(self, key: str, value: float, step: int | None = None) -> None:
        mlflow.log_metric(key, value, step=step)

    def log_artifact(self, local_path: str | Path) -> None:
        mlflow.log_artifact(str(local_path))

    def log_artifacts(self, local_dir: str | Path) -> None:
        mlflow.log_artifacts(str(local_dir))

    def set_tags(self, tags: dict[str, str]) -> None:
        mlflow.set_tags(tags)

    def log_figure(self, figure: Any, artifact_file: str) -> None:
        mlflow.log_figure(figure, artifact_file)

    def log_dict(self, data: dict, artifact_file: str) -> None:
        mlflow.log_dict(data, artifact_file)

    @property
    def run_id(self) -> str | None:
        return self._run.info.run_id if self._run else None

    # ── Emergency checkpointing ───────────────────────────────────────

    def register_checkpoint_callback(self, fn: Callable[[], None]) -> None:
        """Register a function that saves the current training state."""
        self._checkpoint_fn = fn

    def _signal_handler(self, signum: int, frame: Any) -> None:
        sig_name = signal.Signals(signum).name
        logger.warning("Received %s — saving emergency checkpoint...", sig_name)
        if self._checkpoint_fn:
            try:
                self._checkpoint_fn()
                logger.info("Emergency checkpoint saved.")
            except Exception:
                logger.exception("Failed to save emergency checkpoint")
        self.end_run(status="KILLED")
        sys.exit(128 + signum)

    def _install_signal_handlers(self) -> None:
        for sig in (signal.SIGINT, signal.SIGTERM):
            self._original_handlers[sig] = signal.getsignal(sig)
            signal.signal(sig, self._signal_handler)

    def _restore_signal_handlers(self) -> None:
        for sig, handler in self._original_handlers.items():
            signal.signal(sig, handler)
        self._original_handlers.clear()
