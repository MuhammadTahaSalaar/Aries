"""ARIES Triage pipeline — XGBoost alert classification on GUIDE."""

from .feature_engineering import process_guide_dataset
from .trainer import XGBoostTriageTrainer
from .onnx_exporter import export_xgboost_to_onnx

__all__ = [
    "process_guide_dataset",
    "XGBoostTriageTrainer",
    "export_xgboost_to_onnx",
]
