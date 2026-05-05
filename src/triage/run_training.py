"""
CLI: Train XGBoost triage classifier.

Usage (local):
    python -m src.triage.run_training --debug
    python -m src.triage.run_training --tree-method gpu_hist

Usage (HPC via SLURM):
    sbatch slurm/train_xgboost.sh
"""

import argparse
import logging

from src.shared.config import get_settings
from src.shared.logging import setup_logging
from src.shared.mlflow_utils import OfflineMLflowTracker
from src.triage.trainer import XGBoostTriageTrainer


def main() -> None:
    parser = argparse.ArgumentParser(description="Train XGBoost triage model")
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--depth", type=int, default=None)
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--early-stop", type=int, default=None)
    parser.add_argument("--tree-method", type=str, default=None)
    parser.add_argument("--device", type=str, default=None,
                        help="XGBoost device: 'cuda' for GPU, 'cpu' for CPU")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    setup_logging("DEBUG" if args.debug else "INFO")
    logger = logging.getLogger(__name__)

    overrides = {}
    if args.lr is not None:
        overrides["xgb_learning_rate"] = args.lr
    if args.depth is not None:
        overrides["xgb_max_depth"] = args.depth
    if args.rounds is not None:
        overrides["xgb_n_estimators"] = args.rounds
    if args.early_stop is not None:
        overrides["xgb_early_stopping"] = args.early_stop
    if args.tree_method is not None:
        overrides["xgb_tree_method"] = args.tree_method
    if args.device is not None:
        overrides["xgb_device"] = args.device

    cfg = get_settings(**overrides)
    logger.info("Config: tree_method=%s  device=%s  depth=%d  lr=%.4f  rounds=%d",
                cfg.xgb_tree_method, cfg.xgb_device, cfg.xgb_max_depth,
                cfg.xgb_learning_rate, cfg.xgb_n_estimators)

    tracker = OfflineMLflowTracker(
        experiment_name=f"{cfg.mlflow_experiment_prefix}/triage-classifier",
        tracking_uri=cfg.resolve_mlflow_uri(),
        run_name="xgboost-triage",
    )

    with tracker:
        trainer = XGBoostTriageTrainer(cfg)
        result = trainer.run(tracker=tracker)

    logger.info(
        "Training complete.  Accuracy=%.4f  F1=%.4f",
        result["metrics"]["accuracy"],
        result["metrics"]["f1_macro"],
    )


if __name__ == "__main__":
    main()
