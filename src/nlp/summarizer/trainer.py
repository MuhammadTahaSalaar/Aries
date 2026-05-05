"""
ARIES — BART summariser trainer.

Fine-tunes facebook/bart-base on the tokenised GovReport dataset using
HuggingFace Seq2SeqTrainer with ROUGE evaluation.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class SummarizerTrainer:
    """Fine-tune BART for long-document report summarisation."""

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.model = None
        self.tokenizer = None
        self.dataset = None
        self.meta: dict[str, Any] = {}
        self.metrics: dict[str, float] = {}
        self._trainer = None  # HuggingFace Trainer instance

    # ── Data loading ──────────────────────────────────────────────────

    def load_data(self):
        """Load the pre-tokenised HuggingFace DatasetDict."""
        from datasets import DatasetDict
        from src.shared.utils import load_json

        ds_path = self.cfg.processed_dir / "summarizer_dataset"
        if not ds_path.exists():
            raise FileNotFoundError(
                f"{ds_path} not found.  Run preprocessing first:\n"
                "  python -m src.nlp.summarizer.run_preprocessing"
            )

        self.dataset = DatasetDict.load_from_disk(str(ds_path))
        meta_path = self.cfg.processed_dir / "summarizer_metadata.json"
        self.meta = load_json(meta_path) if meta_path.exists() else {}

        logger.info(
            "Loaded summariser dataset  train=%d  val=%d  test=%d",
            len(self.dataset["train"]),
            len(self.dataset["validation"]),
            len(self.dataset["test"]),
        )

    # ── Model setup ───────────────────────────────────────────────────

    def setup_model(self):
        """Load BART base model and tokeniser."""
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.bart_base_model)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.cfg.bart_base_model)

        logger.info(
            "Loaded %s  params=%s",
            self.cfg.bart_base_model,
            f"{sum(p.numel() for p in self.model.parameters()):,}",
        )

    # ── ROUGE metrics ─────────────────────────────────────────────────

    def _compute_metrics(self, eval_pred) -> dict[str, float]:
        """Compute ROUGE-1, ROUGE-2, ROUGE-L scores."""
        from rouge_score import rouge_scorer

        preds, labels = eval_pred
        if isinstance(preds, tuple):
            preds = preds[0]

        # Replace -100 in labels (padding) with pad token id
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)

        # Decode
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Strip whitespace
        decoded_preds = [p.strip() for p in decoded_preds]
        decoded_labels = [l.strip() for l in decoded_labels]

        # Score
        scorer = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"], use_stemmer=True
        )
        scores = {"rouge1": [], "rouge2": [], "rougeL": []}
        for pred, ref in zip(decoded_preds, decoded_labels):
            score = scorer.score(ref, pred)
            for key in scores:
                scores[key].append(score[key].fmeasure)

        return {k: float(np.mean(v)) for k, v in scores.items()}

    # ── Training ──────────────────────────────────────────────────────

    def train(self) -> dict[str, float]:
        """Fine-tune BART using Seq2SeqTrainer."""
        from transformers import (
            Seq2SeqTrainingArguments,
            Seq2SeqTrainer,
            DataCollatorForSeq2Seq,
        )

        output_dir = str(self.cfg.checkpoints_dir / "bart")

        training_args = Seq2SeqTrainingArguments(
            output_dir=output_dir,
            num_train_epochs=self.cfg.bart_epochs,
            per_device_train_batch_size=self.cfg.bart_batch_size,
            per_device_eval_batch_size=self.cfg.bart_batch_size,
            gradient_accumulation_steps=self.cfg.bart_gradient_accumulation,
            learning_rate=self.cfg.bart_learning_rate,
            warmup_steps=self.cfg.bart_warmup_steps,
            weight_decay=0.01,
            predict_with_generate=True,
            generation_max_length=self.cfg.bart_max_target_length,
            generation_num_beams=self.cfg.bart_num_beams,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="rougeL",
            greater_is_better=True,
            fp16=self.cfg.mixed_precision == "fp16",
            bf16=self.cfg.mixed_precision == "bf16",
            logging_steps=100,
            save_total_limit=2,
            report_to="none",
            seed=self.cfg.seed,
            dataloader_num_workers=self.cfg.num_workers,
            # Gradient checkpointing to fit large models in memory
            gradient_checkpointing=True,
        )

        data_collator = DataCollatorForSeq2Seq(
            self.tokenizer,
            model=self.model,
            padding=True,
            label_pad_token_id=-100,
        )

        self._trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=self.dataset["train"],
            eval_dataset=self.dataset["validation"],
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self._compute_metrics,
        )

        logger.info(
            "Starting BART training  epochs=%d  batch=%d*%d  lr=%.1e",
            self.cfg.bart_epochs,
            self.cfg.bart_batch_size,
            self.cfg.bart_gradient_accumulation,
            self.cfg.bart_learning_rate,
        )

        train_result = self._trainer.train()

        # Evaluate on test set
        test_metrics = self._trainer.evaluate(self.dataset["test"])
        logger.info(
            "Test ROUGE  R1=%.4f  R2=%.4f  RL=%.4f",
            test_metrics.get("eval_rouge1", 0),
            test_metrics.get("eval_rouge2", 0),
            test_metrics.get("eval_rougeL", 0),
        )

        self.metrics = {
            "train_loss": train_result.metrics.get("train_loss", 0),
            "train_runtime": train_result.metrics.get("train_runtime", 0),
            "eval_rouge1": test_metrics.get("eval_rouge1", 0),
            "eval_rouge2": test_metrics.get("eval_rouge2", 0),
            "eval_rougeL": test_metrics.get("eval_rougeL", 0),
        }
        return self.metrics

    # ── Generate ──────────────────────────────────────────────────────

    def generate_summary(self, text: str) -> str:
        """Generate a summary for a single input document (convenience)."""
        import torch

        device = self.model.device
        inputs = self.tokenizer(
            text,
            max_length=self.cfg.bart_max_source_length,
            truncation=True,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_length=self.cfg.bart_max_target_length,
                num_beams=self.cfg.bart_num_beams,
                early_stopping=True,
            )
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # ── Save ──────────────────────────────────────────────────────────

    def save(self, output_dir: Path) -> dict[str, Path]:
        """Save model + tokeniser to disk."""
        output_dir.mkdir(parents=True, exist_ok=True)
        self._trainer.save_model(str(output_dir))
        self.tokenizer.save_pretrained(str(output_dir))
        logger.info("Saved summariser model + tokeniser → %s", output_dir)

        from src.shared.utils import save_json

        meta = {
            "model_type": "bart-summariser",
            "base_model": self.cfg.bart_base_model,
            "task": "summarisation",
            "metrics": self.metrics,
            "params": {
                "epochs": self.cfg.bart_epochs,
                "batch_size": self.cfg.bart_batch_size,
                "gradient_accumulation": self.cfg.bart_gradient_accumulation,
                "learning_rate": self.cfg.bart_learning_rate,
                "max_source_length": self.cfg.bart_max_source_length,
                "max_target_length": self.cfg.bart_max_target_length,
                "num_beams": self.cfg.bart_num_beams,
            },
        }
        meta_path = save_json(meta, output_dir / "model_metadata.json")
        return {"model_dir": output_dir, "metadata": meta_path}

    # ── Full pipeline ─────────────────────────────────────────────────

    def run(self, tracker=None) -> dict[str, Any]:
        """Complete summariser training pipeline."""
        self.load_data()
        self.setup_model()

        if tracker:
            tracker.register_checkpoint_callback(
                lambda: self.model.save_pretrained(
                    str(self.cfg.checkpoints_dir / "bart" / "emergency")
                )
            )

        metrics = self.train()
        model_dir = self.cfg.models_dir / "summarizer"
        paths = self.save(model_dir)

        # MLflow logging
        if tracker:
            tracker.log_params({
                "model_type": "BART-summariser",
                "base_model": self.cfg.bart_base_model,
                "dataset": "GovReport",
                "epochs": self.cfg.bart_epochs,
                "batch_size": self.cfg.bart_batch_size,
                "gradient_accumulation": self.cfg.bart_gradient_accumulation,
                "learning_rate": self.cfg.bart_learning_rate,
                "max_source_length": self.cfg.bart_max_source_length,
                "max_target_length": self.cfg.bart_max_target_length,
                "num_beams": self.cfg.bart_num_beams,
                "train_samples": len(self.dataset["train"]),
                "test_samples": len(self.dataset["test"]),
            })
            tracker.log_metrics(metrics)
            tracker.log_artifacts(str(model_dir))
            tracker.set_tags({
                "pipeline": "summariser",
                "model_type": "BART",
                "dataset": "GovReport",
            })

        logger.info(
            "Summariser pipeline complete — ROUGE-L=%.4f",
            metrics.get("eval_rougeL", 0),
        )
        return {"metrics": metrics, "paths": paths, "model": self.model}
