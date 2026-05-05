"""
ARIES — Summarisation dataset preprocessing.

Loads GovReport document–summary pairs from local parquet files, tokenises
with BART tokeniser, and saves as a HuggingFace DatasetDict for Seq2SeqTrainer.

GovReport  (17 517 train / 973 val / 973 test)
  Columns:  report (source text), summary (target text)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def preprocess_summarization_data(
    govreport_dir: Path,
    output_dir: Path,
    tokenizer_name: str = "facebook/bart-base",
    max_source_length: int = 1024,
    max_target_length: int = 256,
) -> dict[str, Any]:
    """
    Load GovReport parquets, tokenise for BART, save DatasetDict.

    Returns
    -------
    dict with keys: dataset_path, metadata, num_train, num_val, num_test.
    """
    from datasets import Dataset, DatasetDict
    from transformers import AutoTokenizer
    import pyarrow.parquet as pq

    logger.info("Loading GovReport from %s", govreport_dir)

    # ── Load from parquet shards ──────────────────────────────────────
    doc_dir = govreport_dir / "document"
    splits: dict[str, Dataset] = {}
    for split_name in ("train", "validation", "test"):
        # Find all shard parquets for this split
        shards = sorted(doc_dir.glob(f"{split_name}-*.parquet"))
        if not shards:
            raise FileNotFoundError(
                f"No parquets found for split '{split_name}' in {doc_dir}"
            )
        tables = [pq.read_table(str(s)) for s in shards]
        # Concatenate shards
        import pyarrow as pa
        merged = pa.concat_tables(tables)
        splits[split_name] = Dataset.from_dict({
            "report": merged.column("report").to_pylist(),
            "summary": merged.column("summary").to_pylist(),
        })
        logger.info("  %s: %d examples from %d shard(s)", split_name,
                     len(splits[split_name]), len(shards))

    raw_ds = DatasetDict(splits)

    # ── Tokenise ──────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def tokenize_fn(batch):
        model_inputs = tokenizer(
            batch["report"],
            max_length=max_source_length,
            truncation=True,
            padding=False,          # dynamic padding via data collator
        )
        # Tokenise targets
        labels = tokenizer(
            text_target=batch["summary"],
            max_length=max_target_length,
            truncation=True,
            padding=False,
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    logger.info(
        "Tokenising with %s (source=%d, target=%d)",
        tokenizer_name, max_source_length, max_target_length,
    )
    tokenized_ds = raw_ds.map(
        tokenize_fn,
        batched=True,
        remove_columns=["report", "summary"],
        desc="Tokenising",
        num_proc=4,
    )

    # ── Save to disk ──────────────────────────────────────────────────
    ds_path = output_dir / "summarizer_dataset"
    ds_path.mkdir(parents=True, exist_ok=True)
    tokenized_ds.save_to_disk(str(ds_path))
    logger.info("Saved tokenised dataset → %s", ds_path)

    # ── Metadata ──────────────────────────────────────────────────────
    from src.shared.utils import save_json

    metadata = {
        "task": "summarisation",
        "tokenizer": tokenizer_name,
        "max_source_length": max_source_length,
        "max_target_length": max_target_length,
        "splits": {
            "train": len(tokenized_ds["train"]),
            "validation": len(tokenized_ds["validation"]),
            "test": len(tokenized_ds["test"]),
        },
    }
    meta_path = save_json(metadata, output_dir / "summarizer_metadata.json")
    logger.info("Metadata → %s", meta_path)

    return {
        "dataset_path": str(ds_path),
        "metadata": metadata,
        "num_train": metadata["splits"]["train"],
        "num_val": metadata["splits"]["validation"],
        "num_test": metadata["splits"]["test"],
    }
