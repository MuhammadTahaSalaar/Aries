"""
CLI: Train SecureBERT-NER on preprocessed NER data.

Usage:
    python -m src.nlp.ner.run_training [--epochs N] [--lr F] [--batch N] [--debug]
"""

import argparse
import logging

from src.shared.config import get_settings
from src.shared.logging import setup_logging
from src.shared.mlflow_utils import OfflineMLflowTracker
from src.nlp.ner.trainer import NERTrainer


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SecureBERT NER model")
    parser.add_argument("--epochs", type=int, default=None, help="Override num epochs")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--batch", type=int, default=None, help="Override batch size")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    setup_logging("DEBUG" if args.debug else "INFO")
    logger = logging.getLogger(__name__)

    cfg = get_settings()

    # Apply CLI overrides
    if args.epochs is not None:
        cfg.ner_epochs = args.epochs
    if args.lr is not None:
        cfg.ner_learning_rate = args.lr
    if args.batch is not None:
        cfg.ner_batch_size = args.batch

    logger.info(
        "Training NER — base=%s  epochs=%d  lr=%.1e  batch=%d",
        cfg.ner_base_model, cfg.ner_epochs, cfg.ner_learning_rate, cfg.ner_batch_size,
    )

    tracker = OfflineMLflowTracker(
        experiment_name="ner-secureBERT",
        tracking_uri=cfg.resolve_mlflow_uri(),
        checkpoint_dir=cfg.checkpoints_dir / "ner",
    )

    trainer = NERTrainer(cfg)
    with tracker:
        result = trainer.run(tracker=tracker)

    logger.info(
        "NER training complete — F1=%.4f",
        result["metrics"].get("eval_f1", 0),
    )


if __name__ == "__main__":
    main()
