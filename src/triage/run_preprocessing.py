"""
CLI: Run GUIDE preprocessing.

Usage:
    python -m src.triage.run_preprocessing [--debug]
"""

import argparse
import logging

from src.shared.config import get_settings
from src.shared.logging import setup_logging
from src.triage.feature_engineering import process_guide_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="GUIDE tabular preprocessing")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    setup_logging("DEBUG" if args.debug else "INFO")
    logger = logging.getLogger(__name__)

    cfg = get_settings()
    logger.info("Project root: %s", cfg.root)

    result = process_guide_dataset(
        guide_dir=cfg.guide_dir,
        output_dir=cfg.processed_dir,
    )
    logger.info(
        "Done — %d features, %s train samples, %s test samples",
        result["metadata"]["n_features"],
        f"{result['metadata']['train_samples']:,}",
        f"{result['metadata']['test_samples']:,}",
    )


if __name__ == "__main__":
    main()
