"""
CLI: Run summarisation preprocessing (GovReport → tokenised dataset).

Usage:
    python -m src.nlp.summarizer.run_preprocessing [--debug]
"""

import argparse
import logging

from src.shared.config import get_settings
from src.shared.logging import setup_logging
from src.nlp.summarizer.preprocessor import preprocess_summarization_data


def main() -> None:
    parser = argparse.ArgumentParser(description="Summariser data preprocessing")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    setup_logging("DEBUG" if args.debug else "INFO")
    logger = logging.getLogger(__name__)

    cfg = get_settings()
    logger.info("Project root: %s", cfg.root)

    result = preprocess_summarization_data(
        govreport_dir=cfg.govreport_dir,
        output_dir=cfg.processed_dir,
        tokenizer_name=cfg.bart_base_model,
        max_source_length=cfg.bart_max_source_length,
        max_target_length=cfg.bart_max_target_length,
    )
    logger.info(
        "Done — train=%d, val=%d, test=%d",
        result["num_train"],
        result["num_val"],
        result["num_test"],
    )


if __name__ == "__main__":
    main()
