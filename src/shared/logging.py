"""Structured JSON logging setup for ARIES."""

import logging
import sys


def setup_logging(level: str = "INFO") -> None:
    """Configure root logger with structured format."""
    fmt = (
        "%(asctime)s | %(levelname)-8s | %(name)-30s | %(message)s"
    )
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=fmt,
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )
    # Quieten noisy third-party loggers
    for noisy in ("urllib3", "botocore", "s3transfer", "filelock"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
