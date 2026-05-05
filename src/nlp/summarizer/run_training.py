"""
CLI: Train BART summariser on preprocessed GovReport data.

Usage:
    python -m src.nlp.summarizer.run_training [--epochs N] [--lr F] [--batch N] [--debug]
"""

import argparse
import logging

from src.shared.config import get_settings
from src.shared.logging import setup_logging
from src.shared.mlflow_utils import OfflineMLflowTracker
from src.nlp.summarizer.trainer import SummarizerTrainer


def main() -> None:
    parser = argparse.ArgumentParser(description="Train BART summariser")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--grad-accum", type=int, default=None,
                        help="Gradient accumulation steps")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    setup_logging("DEBUG" if args.debug else "INFO")
    logger = logging.getLogger(__name__)

    cfg = get_settings()

    # Apply CLI overrides
    if args.epochs is not None:
        cfg.bart_epochs = args.epochs
    if args.lr is not None:
        cfg.bart_learning_rate = args.lr
    if args.batch is not None:
        cfg.bart_batch_size = args.batch
    if args.grad_accum is not None:
        cfg.bart_gradient_accumulation = args.grad_accum

    logger.info(
        "Training BART — base=%s  epochs=%d  batch=%d*%d  lr=%.1e",
        cfg.bart_base_model,
        cfg.bart_epochs,
        cfg.bart_batch_size,
        cfg.bart_gradient_accumulation,
        cfg.bart_learning_rate,
    )

    tracker = OfflineMLflowTracker(
        experiment_name="summariser-bart",
        tracking_uri=cfg.resolve_mlflow_uri(),
        checkpoint_dir=cfg.checkpoints_dir / "bart",
    )

    trainer = SummarizerTrainer(cfg)
    with tracker:
        result = trainer.run(tracker=tracker)

    logger.info(
        "Summariser training complete — ROUGE-L=%.4f",
        result["metrics"].get("eval_rougeL", 0),
    )


if __name__ == "__main__":
    main()
