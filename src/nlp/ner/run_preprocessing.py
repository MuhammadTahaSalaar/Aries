"""
CLI: Run NER preprocessing (CyNER + CASIE → tokenized dataset).

Usage:
    python -m src.nlp.ner.run_preprocessing [--debug]
"""

import argparse
import logging

from src.shared.config import get_settings
from src.shared.logging import setup_logging
from src.nlp.ner.preprocessor import preprocess_ner_data


def main() -> None:
    parser = argparse.ArgumentParser(description="NER data preprocessing")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    setup_logging("DEBUG" if args.debug else "INFO")
    logger = logging.getLogger(__name__)

    cfg = get_settings()
    logger.info("Project root: %s", cfg.root)

    result = preprocess_ner_data(
        cyner_dir=cfg.cyner_dir,
        casie_dir=cfg.casie_dir,
        output_dir=cfg.processed_dir,
        tokenizer_name=cfg.ner_base_model,
        max_length=cfg.ner_max_length,
    )
    logger.info(
        "Done — %d labels, train=%d, val=%d, test=%d, CASIE augmented=%d",
        len(result["label_list"]),
        result["metadata"]["splits"]["train"],
        result["metadata"]["splits"]["validation"],
        result["metadata"]["splits"]["test"],
        result["metadata"]["casie_augmentation"],
    )


if __name__ == "__main__":
    main()
