"""
ARIES — XGBoost triage classifier training.

Trains a multi-class (BenignPositive / FalsePositive / TruePositive) XGBoost
model on the GUIDE dataset.  Supports GPU via device='cuda' (XGBoost 2.0+).
Logs everything to MLflow (offline file:// on HPC, or remote server locally).
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class XGBoostTriageTrainer:
    """Train and evaluate an XGBoost triage classifier."""

    def __init__(self, cfg) -> None:
        """
        Args:
            cfg: Settings object from src.shared.config
        """
        self.cfg = cfg
        self.model = None
        self.metadata: dict[str, Any] = {}

    # ── Data loading ──────────────────────────────────────────────────

    def load_data(self) -> dict[str, np.ndarray]:
        """Load preprocessed .npz from processed_dir."""
        npz_path = self.cfg.processed_dir / "triage_data.npz"
        if not npz_path.exists():
            raise FileNotFoundError(
                f"{npz_path} not found.  Run preprocessing first:\n"
                "  python -m src.triage.run_preprocessing"
            )
        data = np.load(npz_path)
        self.X_train = data["X_train"]
        self.y_train = data["y_train"]
        self.X_test = data["X_test"]
        self.y_test = data["y_test"]
        logger.info(
            "Loaded triage data  train=%s  test=%s  features=%d",
            f"{self.X_train.shape[0]:,}",
            f"{self.X_test.shape[0]:,}",
            self.X_train.shape[1],
        )
        # Load metadata for feature names
        meta_path = self.cfg.processed_dir / "triage_metadata.json"
        if meta_path.exists():
            from src.shared.utils import load_json
            self.metadata = load_json(meta_path)
        return {
            "X_train": self.X_train, "y_train": self.y_train,
            "X_test": self.X_test, "y_test": self.y_test,
        }

    # ── Device helpers ────────────────────────────────────────────────

    def _to_device(self, X: np.ndarray):
        """Move a numpy array to the configured XGBoost device.

        Returns a cupy array when device='cuda' (avoids the
        'mismatched devices' warning at eval/predict time), or the
        original numpy array when device='cpu'.
        """
        if "cuda" in self.cfg.xgb_device:
            try:
                import cupy as cp
                return cp.array(X)
            except ImportError:
                logger.debug("cupy not available; keeping X on CPU")
        return X

    # ── Training ──────────────────────────────────────────────────────

    def build_model(self):
        """Build XGBClassifier with current settings."""
        import xgboost as xgb
        from sklearn.utils.class_weight import compute_sample_weight

        self.sample_weights = compute_sample_weight("balanced", self.y_train)

        self.model = xgb.XGBClassifier(
            objective="multi:softprob",
            num_class=3,
            eval_metric=["mlogloss", "merror"],
            n_estimators=self.cfg.xgb_n_estimators,
            max_depth=self.cfg.xgb_max_depth,
            learning_rate=self.cfg.xgb_learning_rate,
            subsample=self.cfg.xgb_subsample,
            colsample_bytree=self.cfg.xgb_colsample_bytree,
            reg_alpha=self.cfg.xgb_reg_alpha,
            tree_method=self.cfg.xgb_tree_method,
            device=self.cfg.xgb_device,  # XGBoost 2.0+: 'cuda' enables GPU
            early_stopping_rounds=self.cfg.xgb_early_stopping,
            random_state=self.cfg.seed,
            verbosity=1,
            n_jobs=-1,
        )
        logger.info("Built XGBClassifier  tree_method=%s  device=%s  depth=%d  lr=%.4f",
                     self.cfg.xgb_tree_method, self.cfg.xgb_device,
                     self.cfg.xgb_max_depth, self.cfg.xgb_learning_rate)
        return self.model

    def train(self) -> None:
        """Fit the model with early stopping."""
        logger.info("Training XGBoost...")
        t0 = time.time()
        self.model.fit(
            self._to_device(self.X_train),
            self.y_train,
            sample_weight=self.sample_weights,
            eval_set=[(self._to_device(self.X_test), self.y_test)],
            verbose=50,
        )
        self.train_time = time.time() - t0
        logger.info("Training complete in %.1f s  best_iteration=%s",
                     self.train_time, getattr(self.model, "best_iteration", "N/A"))

    # ── Evaluation ────────────────────────────────────────────────────

    def evaluate(self) -> dict[str, Any]:
        """Compute classification metrics on the test set."""
        from sklearn.metrics import (
            accuracy_score, f1_score, precision_score, recall_score,
            classification_report, confusion_matrix,
        )

        X_eval = self._to_device(self.X_test)
        y_pred = self.model.predict(X_eval)
        y_prob = self.model.predict_proba(X_eval)

        label_names = ["BenignPositive", "FalsePositive", "TruePositive"]
        metrics = {
            "accuracy": float(accuracy_score(self.y_test, y_pred)),
            "f1_macro": float(f1_score(self.y_test, y_pred, average="macro")),
            "f1_weighted": float(f1_score(self.y_test, y_pred, average="weighted")),
            "precision_macro": float(precision_score(self.y_test, y_pred, average="macro")),
            "recall_macro": float(recall_score(self.y_test, y_pred, average="macro")),
            "train_time_seconds": self.train_time,
            "classification_report": classification_report(
                self.y_test, y_pred, target_names=label_names, output_dict=True
            ),
            "confusion_matrix": confusion_matrix(self.y_test, y_pred).tolist(),
        }

        # ml_score = P(TruePositive) — column index 2
        metrics["ml_score_mean"] = float(y_prob[:, 2].mean())

        logger.info("Accuracy=%.4f  F1(macro)=%.4f  F1(weighted)=%.4f",
                     metrics["accuracy"], metrics["f1_macro"], metrics["f1_weighted"])
        self.metrics = metrics
        return metrics

    # ── Confusion matrix plot ─────────────────────────────────────────

    def plot_confusion_matrix(self, output_path: Path) -> Path:
        """Save a confusion matrix heatmap."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns

        cm = np.array(self.metrics["confusion_matrix"])
        labels = ["BenignPositive", "FalsePositive", "TruePositive"]

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=labels, yticklabels=labels, ax=ax,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("XGBoost Triage — Confusion Matrix")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved confusion matrix → %s", output_path)
        return output_path

    # ── Save ──────────────────────────────────────────────────────────

    def save(self, output_dir: Path) -> dict[str, Path]:
        """Save model JSON + metadata."""
        output_dir.mkdir(parents=True, exist_ok=True)
        model_path = output_dir / "xgboost_triage.json"
        self.model.save_model(str(model_path))
        logger.info("Saved model → %s", model_path)

        from src.shared.utils import save_json
        meta = {
            "model_type": "xgboost",
            "task": "triage_classification",
            "n_features": int(self.X_train.shape[1]),
            "feature_names": self.metadata.get("feature_names", []),
            "target_map": {"BenignPositive": 0, "FalsePositive": 1, "TruePositive": 2},
            "metrics": {
                "accuracy": self.metrics["accuracy"],
                "f1_macro": self.metrics["f1_macro"],
            },
            "params": {
                "n_estimators": self.cfg.xgb_n_estimators,
                "max_depth": self.cfg.xgb_max_depth,
                "learning_rate": self.cfg.xgb_learning_rate,
                "tree_method": self.cfg.xgb_tree_method,
                "device": self.cfg.xgb_device,
            },
        }
        meta_path = save_json(meta, output_dir / "model_metadata.json")
        return {"model": model_path, "metadata": meta_path}

    # ── Full pipeline ─────────────────────────────────────────────────

    def run(self, tracker=None) -> dict[str, Any]:
        """
        Complete training pipeline: load → build → train → evaluate → save.

        Args:
            tracker: Optional OfflineMLflowTracker for experiment logging.
        """
        self.load_data()
        self.build_model()

        if tracker:
            tracker.register_checkpoint_callback(
                lambda: self.model.save_model(
                    str(self.cfg.checkpoints_dir / "triage" / "emergency.json")
                )
            )

        self.train()
        metrics = self.evaluate()

        # Save model
        model_dir = self.cfg.models_dir / "triage"
        paths = self.save(model_dir)

        # Copy TargetEncoder to model dir so migration can upload it together
        import shutil
        encoder_src = self.cfg.processed_dir / "triage_encoder.pkl"
        if encoder_src.exists():
            encoder_dst = model_dir / "triage_encoder.pkl"
            shutil.copy2(encoder_src, encoder_dst)
            logger.info("Copied TargetEncoder → %s", encoder_dst)
            paths["encoder"] = encoder_dst
        else:
            logger.warning(
                "triage_encoder.pkl not found at %s — run preprocessing first", encoder_src
            )

        # Confusion matrix
        cm_path = self.plot_confusion_matrix(model_dir / "confusion_matrix.png")

        # MLflow logging
        if tracker:
            tracker.log_params({
                "model_type": "xgboost",
                "dataset": "GUIDE",
                "n_estimators": self.cfg.xgb_n_estimators,
                "max_depth": self.cfg.xgb_max_depth,
                "learning_rate": self.cfg.xgb_learning_rate,
                "subsample": self.cfg.xgb_subsample,
                "colsample_bytree": self.cfg.xgb_colsample_bytree,
                "tree_method": self.cfg.xgb_tree_method,
                "device": self.cfg.xgb_device,
                "train_samples": self.X_train.shape[0],
                "test_samples": self.X_test.shape[0],
                "n_features": self.X_train.shape[1],
            })
            tracker.log_metrics({
                "accuracy": metrics["accuracy"],
                "f1_macro": metrics["f1_macro"],
                "f1_weighted": metrics["f1_weighted"],
                "precision_macro": metrics["precision_macro"],
                "recall_macro": metrics["recall_macro"],
                "ml_score_mean": metrics["ml_score_mean"],
                "train_time_seconds": metrics["train_time_seconds"],
            })
            tracker.log_artifact(str(paths["model"]))
            tracker.log_artifact(str(paths["metadata"]))
            tracker.log_artifact(str(cm_path))
            tracker.set_tags({
                "pipeline": "triage",
                "model_type": "xgboost",
                "dataset": "GUIDE",
            })

        logger.info("Triage pipeline complete.")
        return {
            "metrics": metrics,
            "paths": paths,
            "model": self.model,
        }
