"""
ARIES — NER model ONNX export.

Exports SecureBERT-NER fine-tuned model to ONNX format with
onnxruntime transformer optimisation.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def export_ner_to_onnx(
    model_dir: Path,
    output_path: Path,
    opset: int = 17,
    max_length: int = 512,
    optimize: bool = True,
) -> Path:
    """
    Export NER model to ONNX.

    Args:
        model_dir: Directory containing the saved HF model + tokenizer
        output_path: Desired .onnx output path
        opset: ONNX opset version
        max_length: Max sequence length for dynamic axes
        optimize: Whether to run ORT transformer optimizer

    Returns:
        Path to the saved .onnx file
    """
    import torch
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    model_dir = Path(model_dir)
    output_path = Path(output_path)

    logger.info("Loading NER model from %s", model_dir)
    model = AutoModelForTokenClassification.from_pretrained(str(model_dir))
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model.eval()

    # Create dummy inputs
    dummy_text = "CVE-2024-1234 affects Apache Log4j resulting in remote code execution"
    inputs = tokenizer(
        dummy_text, return_tensors="pt",
        max_length=max_length, truncation=True, padding="max_length",
    )
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Exporting to ONNX  opset=%d", opset)
    torch.onnx.export(
        model,
        (input_ids, attention_mask),
        str(output_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "logits": {0: "batch", 1: "seq"},
        },
        opset_version=opset,
        do_constant_folding=True,
    )
    logger.info("Exported → %s", output_path)

    # Optimize with ORT
    if optimize:
        try:
            from onnxruntime.transformers import optimizer
            opt_path = output_path.with_suffix(".opt.onnx")
            opt_model = optimizer.optimize_model(
                str(output_path),
                model_type="bert",
                num_heads=12,
                hidden_size=768,
            )
            opt_model.save_model_to_file(str(opt_path))
            logger.info("Optimized → %s", opt_path)
            output_path = opt_path
        except ImportError:
            logger.warning("onnxruntime.transformers not available — skipping optimization")

    # Validate
    import onnx
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)
    logger.info("ONNX validation passed ✓")

    return output_path


def validate_ner_onnx(onnx_path: Path, tokenizer_dir: Path) -> dict:
    """Quick sanity check on the exported ONNX model."""
    import numpy as np
    import onnxruntime as ort
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir))
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    text = "Cobalt Strike beacon connected to 192.168.1.100"
    inputs = tokenizer(text, return_tensors="np", truncation=True, max_length=512, padding="max_length")

    logits = session.run(
        ["logits"],
        {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"]},
    )[0]

    preds = np.argmax(logits, axis=-1)
    logger.info("ONNX NER validation  output_shape=%s  sample_preds=%s",
                logits.shape, preds[0][:20])
    return {"logits_shape": logits.shape, "predictions": preds}
