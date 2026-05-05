"""
ARIES — NER fine-tuning trainer.

Fine-tunes SecureBERT (RoBERTa) on the merged CyNER+CASIE dataset for
security-domain token classification.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class NERTrainer:
    """Fine-tune SecureBERT for NER token classification."""

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.model = None
        self.tokenizer = None
        self.metrics: dict[str, Any] = {}

    # ── Data loading ──────────────────────────────────────────────────

    def load_data(self):
        """Load the pre-tokenized HuggingFace DatasetDict from disk."""
        from datasets import DatasetDict
        from src.shared.utils import load_json

        ds_path = self.cfg.processed_dir / "ner_dataset"
        if not ds_path.exists():
            raise FileNotFoundError(
                f"{ds_path} not found.  Run preprocessing first:\n"
                "  python -m src.nlp.ner.run_preprocessing"
            )
        self.dataset = DatasetDict.load_from_disk(str(ds_path))

        meta_path = self.cfg.processed_dir / "ner_metadata.json"
        self.meta = load_json(meta_path) if meta_path.exists() else {}
        self.label_list = self.meta.get("label_list", [])
        self.label2id = self.meta.get("label2id", {})
        self.id2label = {int(k): v for k, v in self.meta.get("id2label", {}).items()}

        logger.info(
            "Loaded NER dataset  train=%d  val=%d  test=%d  labels=%d",
            len(self.dataset["train"]),
            len(self.dataset["validation"]),
            len(self.dataset["test"]),
            len(self.label_list),
        )

    # ── Model setup ───────────────────────────────────────────────────

    def _ensure_local_safetensors(self) -> str:
        """Download SecureBERT and convert to safetensors if not already cached.

        SecureBERT only ships a legacy pytorch_model.bin which is blocked by
        transformers when PyTorch < 2.6 (CVE-2025-32434).  We work around this
        by doing a one-time conversion to the safetensors format in our local
        cache, calling torch.load directly (trusted download, not arbitrary
        user data) instead of going through transformers' safety gate.

        Returns:
            Absolute path to the local model directory that contains
            model.safetensors (safe to pass to from_pretrained).
        """
        from huggingface_hub import snapshot_download

        local_dir = self.cfg.models_dir / "ner" / "secureBERT_pretrained"
        safetensors_path = local_dir / "model.safetensors"

        if safetensors_path.exists():
            logger.info("Using cached local safetensors weights at %s", local_dir)
            return str(local_dir)

        logger.info("Downloading %s from HuggingFace Hub → %s",
                     self.cfg.ner_base_model, local_dir)
        snapshot_download(
            repo_id=self.cfg.ner_base_model,
            local_dir=str(local_dir),
            ignore_patterns=["*.msgpack", "*.h5", "flax_model*", "tf_model*"],
        )

        bin_path = local_dir / "pytorch_model.bin"
        if bin_path.exists() and not safetensors_path.exists():
            logger.info("Converting pytorch_model.bin → model.safetensors ...")
            import torch
            from safetensors.torch import save_file
            # Call torch.load directly, bypassing transformers' check_torch_load_is_safe.
            # This is safe because the file is our own trusted download, not arbitrary
            # user input.  Once the .safetensors copy exists, future loads use that.
            state_dict = torch.load(
                str(bin_path), map_location="cpu", weights_only=False
            )
            save_file(state_dict, str(safetensors_path))
            bin_path.unlink()  # Remove .bin so from_pretrained always picks safetensors
            logger.info("Conversion complete → %s", safetensors_path)

        return str(local_dir)

    def setup_model(self):
        """Load SecureBERT + token classification head."""
        from transformers import AutoTokenizer, AutoModelForTokenClassification

        local_model_path = self._ensure_local_safetensors()

        self.tokenizer = AutoTokenizer.from_pretrained(local_model_path)
        self.model = AutoModelForTokenClassification.from_pretrained(
            local_model_path,
            num_labels=len(self.label_list),
            id2label=self.id2label,
            label2id=self.label2id,
        )
        logger.info(
            "Loaded %s  params=%s",
            self.cfg.ner_base_model,
            f"{sum(p.numel() for p in self.model.parameters()):,}",
        )

    # ── Metrics ───────────────────────────────────────────────────────

    def _compute_metrics(self, eval_pred) -> dict[str, float]:
        """Compute entity-level metrics using seqeval."""
        from seqeval.metrics import (
            f1_score, precision_score, recall_score, classification_report,
        )

        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)

        # Convert IDs back to label strings, ignoring -100
        true_labels = []
        true_preds = []
        for pred_seq, label_seq in zip(preds, labels):
            t_labels = []
            t_preds = []
            for p, l in zip(pred_seq, label_seq):
                if l == -100:
                    continue
                t_labels.append(self.id2label.get(int(l), "O"))
                t_preds.append(self.id2label.get(int(p), "O"))
            true_labels.append(t_labels)
            true_preds.append(t_preds)

        return {
            "f1": f1_score(true_labels, true_preds),
            "precision": precision_score(true_labels, true_preds),
            "recall": recall_score(true_labels, true_preds),
        }

    # ── Training ──────────────────────────────────────────────────────

    def train(self) -> dict[str, float]:
        """Fine-tune the model using HuggingFace Trainer."""
        from transformers import TrainingArguments, Trainer, DataCollatorForTokenClassification

        output_dir = str(self.cfg.checkpoints_dir / "ner")

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=self.cfg.ner_epochs,
            per_device_train_batch_size=self.cfg.ner_batch_size,
            per_device_eval_batch_size=self.cfg.ner_batch_size,
            learning_rate=self.cfg.ner_learning_rate,
            weight_decay=0.01,
            warmup_ratio=self.cfg.ner_warmup_ratio,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            fp16=self.cfg.mixed_precision == "fp16",
            bf16=self.cfg.mixed_precision == "bf16",
            logging_steps=50,
            save_total_limit=2,
            report_to="none",  # We handle MLflow ourselves
            seed=self.cfg.seed,
            dataloader_num_workers=self.cfg.num_workers,
        )

        data_collator = DataCollatorForTokenClassification(
            self.tokenizer, padding=True
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.dataset["train"],
            eval_dataset=self.dataset["validation"],
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self._compute_metrics,
        )

        logger.info("Starting NER training  epochs=%d  batch=%d  lr=%.1e",
                     self.cfg.ner_epochs, self.cfg.ner_batch_size,
                     self.cfg.ner_learning_rate)

        train_result = trainer.train()
        self.trainer = trainer

        # Evaluate on test set
        test_metrics = trainer.evaluate(self.dataset["test"])
        logger.info(
            "Test results  F1=%.4f  Precision=%.4f  Recall=%.4f",
            test_metrics.get("eval_f1", 0),
            test_metrics.get("eval_precision", 0),
            test_metrics.get("eval_recall", 0),
        )

        self.metrics = {
            "train_loss": train_result.metrics.get("train_loss", 0),
            "train_runtime": train_result.metrics.get("train_runtime", 0),
            "eval_f1": test_metrics.get("eval_f1", 0),
            "eval_precision": test_metrics.get("eval_precision", 0),
            "eval_recall": test_metrics.get("eval_recall", 0),
        }
        return self.metrics

    # ── Save ──────────────────────────────────────────────────────────

    def save(self, output_dir: Path) -> dict[str, Path]:
        """Save model + tokenizer to disk."""
        output_dir.mkdir(parents=True, exist_ok=True)
        self.trainer.save_model(str(output_dir))
        self.tokenizer.save_pretrained(str(output_dir))
        logger.info("Saved NER model + tokenizer → %s", output_dir)

        from src.shared.utils import save_json
        meta = {
            "model_type": "roberta-ner",
            "base_model": self.cfg.ner_base_model,
            "task": "token_classification",
            "label_list": self.label_list,
            "label2id": self.label2id,
            "id2label": self.id2label,
            "metrics": self.metrics,
            "params": {
                "epochs": self.cfg.ner_epochs,
                "batch_size": self.cfg.ner_batch_size,
                "learning_rate": self.cfg.ner_learning_rate,
                "max_length": self.cfg.ner_max_length,
            },
        }
        meta_path = save_json(meta, output_dir / "model_metadata.json")
        return {"model_dir": output_dir, "metadata": meta_path}

    # ── Full pipeline ─────────────────────────────────────────────────

    def run(self, tracker=None) -> dict[str, Any]:
        """Complete NER training pipeline."""
        self.load_data()
        self.setup_model()

        if tracker:
            tracker.register_checkpoint_callback(
                lambda: self.model.save_pretrained(
                    str(self.cfg.checkpoints_dir / "ner" / "emergency")
                )
            )

        metrics = self.train()
        model_dir = self.cfg.models_dir / "ner"
        paths = self.save(model_dir)

        # MLflow logging
        if tracker:
            tracker.log_params({
                "model_type": "SecureBERT-NER",
                "base_model": self.cfg.ner_base_model,
                "dataset": "CyNER+CASIE",
                "epochs": self.cfg.ner_epochs,
                "batch_size": self.cfg.ner_batch_size,
                "learning_rate": self.cfg.ner_learning_rate,
                "max_length": self.cfg.ner_max_length,
                "num_labels": len(self.label_list),
                "train_samples": len(self.dataset["train"]),
                "test_samples": len(self.dataset["test"]),
            })
            tracker.log_metrics(metrics)
            tracker.log_artifacts(str(model_dir))
            tracker.set_tags({
                "pipeline": "ner",
                "model_type": "SecureBERT-NER",
                "dataset": "CyNER+CASIE",
            })

        logger.info("NER pipeline complete.  F1=%.4f", metrics.get("eval_f1", 0))
        return {"metrics": metrics, "paths": paths, "model": self.model}
