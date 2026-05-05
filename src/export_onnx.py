"""
CLI: Export all trained models to ONNX format.

Usage:
    python -m src.export_onnx [--triage] [--ner] [--summarizer] [--all] [--debug]
"""

import argparse
import logging

from src.shared.config import get_settings
from src.shared.logging import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Export models to ONNX")
    parser.add_argument("--triage", action="store_true", help="Export XGBoost triage")
    parser.add_argument("--ner", action="store_true", help="Export NER model")
    parser.add_argument("--summarizer", action="store_true", help="Export BART summariser")
    parser.add_argument("--all", action="store_true", help="Export all models")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    setup_logging("DEBUG" if args.debug else "INFO")
    logger = logging.getLogger(__name__)

    cfg = get_settings()
    onnx_dir = cfg.models_dir / "onnx"
    onnx_dir.mkdir(parents=True, exist_ok=True)

    export_all = args.all or not (args.triage or args.ner or args.summarizer)

    # ── Triage (XGBoost) ──────────────────────────────────────────────
    if args.triage or export_all:
        model_path = cfg.models_dir / "triage" / "xgboost_triage.json"
        if model_path.exists():
            from src.triage.onnx_exporter import export_xgboost_to_onnx
            from src.shared.utils import load_json

            meta_path = cfg.models_dir / "triage" / "model_metadata.json"
            meta = load_json(meta_path) if meta_path.exists() else {}
            n_features = meta.get("n_features", 49)

            out = export_xgboost_to_onnx(
                model_json_path=model_path,
                output_path=onnx_dir / "triage.onnx",
                n_features=n_features,
                opset=cfg.onnx_opset,
            )
            logger.info("Triage ONNX → %s", out)
        else:
            logger.warning("Triage model not found at %s — skipping", model_path)

    # ── NER (SecureBERT) ──────────────────────────────────────────────
    if args.ner or export_all:
        ner_dir = cfg.models_dir / "ner"
        if (ner_dir / "config.json").exists():
            from src.nlp.ner.onnx_exporter import export_ner_to_onnx

            out = export_ner_to_onnx(
                model_dir=str(ner_dir),
                output_path=str(onnx_dir / "ner.onnx"),
                opset=cfg.onnx_opset,
            )
            logger.info("NER ONNX → %s", out)
        else:
            logger.warning("NER model not found at %s — skipping", ner_dir)

    # ── Summariser (BART) ─────────────────────────────────────────────
    if args.summarizer or export_all:
        bart_dir = cfg.models_dir / "summarizer"
        if (bart_dir / "config.json").exists():
            from src.nlp.summarizer.onnx_exporter import export_bart_to_onnx

            out = export_bart_to_onnx(
                model_dir=str(bart_dir),
                output_path=str(onnx_dir / "summarizer"),
                opset=cfg.onnx_opset,
            )
            logger.info("Summariser ONNX → %s", out)
        else:
            logger.warning("Summariser model not found at %s — skipping", bart_dir)

    logger.info("ONNX export complete.")


if __name__ == "__main__":
    main()
