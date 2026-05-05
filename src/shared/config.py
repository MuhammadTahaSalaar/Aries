"""
ARIES — Centralised configuration (pydantic-settings).

Reads from environment variables or .env file.  Every path default assumes
the standard repo layout:

    Aries/
      datasets/GUIDE/, datasets/CyNER/, datasets/CASIE/, datasets/gov_reports/
      data/processed/
      models/triage/, models/ner/, models/summarizer/
      checkpoints/
      mlruns/
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _project_root() -> Path:
    """Walk up from this file until we find datasets/."""
    p = Path(__file__).resolve()
    for parent in [p] + list(p.parents):
        if (parent / "datasets").is_dir():
            return parent
    return Path.cwd()


class Settings(BaseSettings):
    """Single source of truth for all paths and hyper-parameters."""

    model_config = SettingsConfigDict(
        env_prefix="ARIES_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Paths ─────────────────────────────────────────────────────────
    project_root: Path = Field(default_factory=_project_root)

    # HPC override: on Hydra, VSC_DATA_VO_USER/Aries_SOAR is canonical
    project_root_override: str | None = Field(
        default=None,
        description="If set, overrides project_root (e.g. $VSC_DATA_VO_USER/Aries_SOAR)",
    )

    @field_validator("hpc_project_dir", mode="before")
    @classmethod
    def _expand_env(cls, v: str | Path | None) -> Path | None:
        if v is None:
            return None
        return Path(os.path.expandvars(str(v)))

    @property
    def root(self) -> Path:
        return self.hpc_project_dir or self.project_root

    # Dataset dirs
    @property
    def guide_dir(self) -> Path:
        return self.root / "datasets" / "GUIDE"

    @property
    def cyner_dir(self) -> Path:
        return self.root / "datasets" / "CyNER"

    @property
    def casie_dir(self) -> Path:
        return self.root / "datasets" / "CASIE"

    @property
    def govreport_dir(self) -> Path:
        return self.root / "datasets" / "gov_reports"

    @property
    def aptnotes_dir(self) -> Path:
        return self.root / "datasets" / "APTnotes"

    # Output dirs
    @property
    def processed_dir(self) -> Path:
        return self.root / "data" / "processed"

    @property
    def checkpoints_dir(self) -> Path:
        return self.root / "checkpoints"

    @property
    def models_dir(self) -> Path:
        return self.root / "models"

    @property
    def mlruns_dir(self) -> Path:
        return self.root / "mlruns"

    # ── XGBoost hyper-parameters ──────────────────────────────────────
    xgb_n_estimators: int = 1000
    xgb_max_depth: int = 8
    xgb_learning_rate: float = 0.05
    xgb_subsample: float = 0.8
    xgb_colsample_bytree: float = 0.7
    xgb_reg_alpha: float = 0.1
    xgb_early_stopping: int = 50
    xgb_tree_method: str = "hist"
    xgb_device: str = "cuda"  # 'cuda' for GPU, 'cpu' for CPU (XGBoost 2.0+ API)

    # ── NER hyper-parameters ──────────────────────────────────────────
    ner_base_model: str = "ehsanaghaei/SecureBERT"
    ner_epochs: int = 5
    ner_batch_size: int = 16
    ner_learning_rate: float = 2e-5
    ner_max_length: int = 512
    ner_warmup_ratio: float = 0.1
    ner_weight_decay: float = 0.01
    ner_lr: float = 2e-5  # alias kept for CLI convenience

    # ── BART hyper-parameters ─────────────────────────────────────────
    bart_base_model: str = "facebook/bart-base"
    bart_epochs: int = 3
    bart_batch_size: int = 4
    bart_gradient_accumulation: int = 8
    bart_learning_rate: float = 3e-5
    bart_warmup_steps: int = 500
    bart_max_source_length: int = 1024
    bart_max_target_length: int = 256
    bart_num_beams: int = 4

    # ── ONNX ──────────────────────────────────────────────────────────
    onnx_opset: int = 17

    # ── MLflow ────────────────────────────────────────────────────────
    mlflow_tracking_uri: str = ""
    mlflow_experiment_prefix: str = "aries"

    # ── Runtime ───────────────────────────────────────────────────────
    seed: int = 42
    device: Literal["cpu", "cuda", "auto"] = "auto"
    mixed_precision: Literal["no", "fp16", "bf16"] = "fp16"
    num_workers: int = 4
    log_level: str = "INFO"

    # ── Helpers ───────────────────────────────────────────────────────
    def ensure_dirs(self) -> None:
        """Create all output directories."""
        for d in [
            self.processed_dir,
            self.checkpoints_dir / "triage",
            self.checkpoints_dir / "ner",
            self.checkpoints_dir / "bart",
            self.models_dir / "triage",
            self.models_dir / "ner",
            self.models_dir / "summarizer",
            self.mlruns_dir,
        ]:
            d.mkdir(parents=True, exist_ok=True)

    def resolve_mlflow_uri(self) -> str:
        if self.mlflow_tracking_uri:
            return self.mlflow_tracking_uri
        # Use SQLite backend (file:// store is deprecated as of MLflow Feb 2026)
        return f"sqlite:///{self.mlruns_dir}/mlflow.db"


_settings: Settings | None = None


def get_settings(**overrides) -> Settings:
    """Singleton accessor.  Pass keyword overrides for testing / CLI."""
    global _settings
    if _settings is None or overrides:
        _settings = Settings(**overrides)
        _settings.ensure_dirs()
    return _settings
