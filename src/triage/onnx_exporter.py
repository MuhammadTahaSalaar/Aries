"""
ARIES — XGBoost → ONNX export.

Converts a saved XGBoost JSON model into an ONNX model suitable for
ONNX Runtime inference.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def export_xgboost_to_onnx(
    model_json_path: Path,
    output_path: Path,
    n_features: int,
    opset: int = 17,
) -> Path:
    """
    Export XGBoost JSON model to ONNX.

    Args:
        model_json_path: Path to xgboost_triage.json
        output_path: Desired .onnx output path
        n_features: Number of input features
        opset: ONNX opset version

    Returns:
        Path to the saved .onnx file
    """
    import xgboost as xgb
    import onnxmltools
    from onnxmltools.convert import convert_xgboost
    from onnxmltools.convert.common.data_types import FloatTensorType
    from onnxmltools.convert.common.onnx_ex import DEFAULT_OPSET_NUMBER

    model_json_path = Path(model_json_path)
    output_path = Path(output_path)

    # onnxmltools caps at DEFAULT_OPSET_NUMBER (currently 15)
    effective_opset = min(opset, DEFAULT_OPSET_NUMBER)

    logger.info("Loading XGBoost model from %s", model_json_path)
    model = xgb.XGBClassifier()
    model.load_model(str(model_json_path))

    logger.info("Converting to ONNX  opset=%d  n_features=%d", effective_opset, n_features)
    initial_type = [("float_input", FloatTensorType([None, n_features]))]
    onnx_model = convert_xgboost(model, initial_types=initial_type, target_opset=effective_opset)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnxmltools.utils.save_model(onnx_model, str(output_path))
    logger.info("Saved ONNX model → %s", output_path)

    # Validate
    import onnx
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)
    logger.info("ONNX model validation passed ✓")

    return output_path


def validate_onnx_output(
    onnx_path: Path,
    sample_input: np.ndarray,
) -> dict:
    """Quick sanity check that the ONNX model runs."""
    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    results = session.run(None, {input_name: sample_input.astype(np.float32)})
    logger.info("ONNX validation  output_shapes=%s", [r.shape for r in results])
    return {"labels": results[0], "probabilities": results[1]}
