"""
ARIES — NER dataset preprocessing.

Merges CyNER (CoNLL BIO format) with CASIE (JSON event annotations) into
a unified HuggingFace Dataset with BIO-tagged token classification labels.

CyNER entities:  Malware, System, Indicator, Vulnerability, Organization
CASIE entities:   mapped into the same label space via type→label rules.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ── Unified label scheme ─────────────────────────────────────────────────────

LABEL_LIST = [
    "O",
    "B-Malware", "I-Malware",
    "B-Indicator", "I-Indicator",
    "B-System", "I-System",
    "B-Vulnerability", "I-Vulnerability",
    "B-Organization", "I-Organization",
]
LABEL2ID = {l: i for i, l in enumerate(LABEL_LIST)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}

# CASIE argument type → our BIO label
_CASIE_TYPE_MAP = {
    "Malware": "Malware",
    "Vulnerability": "Vulnerability",
    "System": "System",
    "Organization": "Organization",
    "Person": "Organization",     # collapse Person → Organization for simplicity
    "Device": "System",
    "Software": "System",
}


# ── CyNER parsing ────────────────────────────────────────────────────────────

def parse_conll_file(path: Path) -> list[dict[str, list]]:
    """Parse a CoNLL-formatted BIO file into sentences."""
    sentences: list[dict[str, list]] = []
    tokens: list[str] = []
    labels: list[str] = []

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip()
            if not line:
                if tokens:
                    # Map labels to our unified scheme
                    mapped = [_map_cyner_label(l) for l in labels]
                    sentences.append({"tokens": tokens, "ner_tags": mapped})
                    tokens, labels = [], []
                continue
            parts = line.split("\t")
            if len(parts) >= 2:
                tokens.append(parts[0])
                labels.append(parts[1])
            else:
                # Some lines are space-separated
                parts = line.split()
                if len(parts) >= 2:
                    tokens.append(parts[0])
                    labels.append(parts[-1])

    if tokens:
        mapped = [_map_cyner_label(l) for l in labels]
        sentences.append({"tokens": tokens, "ner_tags": mapped})

    return sentences


def _map_cyner_label(label: str) -> str:
    """Map CyNER label to our unified label scheme."""
    if label == "O":
        return "O"
    # CyNER labels: B-Malware, I-Malware, B-Indicator, etc.
    # Keep the ones in our LABEL_LIST, drop the rest to O
    if label in LABEL2ID:
        return label
    # Handle edge cases like B-Tool → B-Malware (CyNER lumps tools with malware)
    if "Tool" in label:
        return label.replace("Tool", "Malware")
    if "Person" in label:
        return label.replace("Person", "Organization")
    if "Location" in label:
        return "O"  # We don't track Location
    return "O"


# ── CASIE parsing ────────────────────────────────────────────────────────────

def parse_casie_annotations(casie_dir: Path) -> list[dict[str, list]]:
    """
    Parse CASIE JSON annotations into BIO-tagged sentences.

    Each CASIE JSON has content text + cyberevent annotations with character offsets.
    We tokenize the content and align entity spans to token boundaries.
    """
    annotation_dir = casie_dir / "data" / "annotation"
    if not annotation_dir.exists():
        logger.warning("CASIE annotation dir not found: %s", annotation_dir)
        return []

    sentences: list[dict[str, list]] = []
    json_files = sorted(annotation_dir.glob("*.json"))
    logger.info("Processing %d CASIE annotation files...", len(json_files))

    for jf in json_files:
        try:
            with open(jf, encoding="utf-8") as f:
                doc = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue

        content = doc.get("content", "")
        if not content:
            continue

        # Collect all entity spans from all events
        entity_spans: list[dict] = []
        hoppers = doc.get("cyberevent", {}).get("hopper", [])
        for hopper in hoppers:
            for event in hopper.get("events", []):
                for arg in event.get("argument", []):
                    etype = arg.get("type", "")
                    mapped = _CASIE_TYPE_MAP.get(etype)
                    if mapped and "startOffset" in arg and "endOffset" in arg:
                        entity_spans.append({
                            "start": arg["startOffset"],
                            "end": arg["endOffset"],
                            "label": mapped,
                            "text": arg.get("text", ""),
                        })

        if not entity_spans:
            continue

        # Sort by start offset, remove overlaps (keep longer)
        entity_spans.sort(key=lambda x: (x["start"], -(x["end"] - x["start"])))
        filtered: list[dict] = []
        for span in entity_spans:
            if not filtered or span["start"] >= filtered[-1]["end"]:
                filtered.append(span)

        # Simple whitespace tokenization with offset tracking
        tokens, labels = _tokenize_and_tag(content, filtered)
        if tokens and len(tokens) > 3:
            sentences.append({"tokens": tokens, "ner_tags": labels})

    logger.info("Extracted %d sentences from CASIE", len(sentences))
    return sentences


def _tokenize_and_tag(
    text: str, spans: list[dict]
) -> tuple[list[str], list[str]]:
    """Tokenize text and assign BIO labels based on character-level spans."""
    # Find token boundaries via whitespace split
    token_offsets: list[tuple[int, int, str]] = []
    for m in re.finditer(r"\S+", text):
        token_offsets.append((m.start(), m.end(), m.group()))

    tokens = [t[2] for t in token_offsets]
    labels = ["O"] * len(tokens)

    # For each span, find overlapping tokens
    span_idx = 0
    for tok_i, (tok_start, tok_end, tok_text) in enumerate(token_offsets):
        # Advance span pointer
        while span_idx < len(spans) and spans[span_idx]["end"] <= tok_start:
            span_idx += 1

        if span_idx >= len(spans):
            break

        span = spans[span_idx]
        # Check overlap
        if tok_start >= span["start"] and tok_start < span["end"]:
            lbl = span["label"]
            if tok_start == span["start"] or (tok_i > 0 and labels[tok_i - 1] == "O"):
                labels[tok_i] = f"B-{lbl}"
            else:
                labels[tok_i] = f"I-{lbl}"

    # Validate labels are in our scheme
    labels = [l if l in LABEL2ID else "O" for l in labels]
    return tokens, labels


# ── Merge & build HuggingFace Dataset ────────────────────────────────────────

def preprocess_ner_data(
    cyner_dir: Path,
    casie_dir: Path,
    output_dir: Path,
    tokenizer_name: str = "ehsanaghaei/SecureBERT",
    max_length: int = 512,
) -> dict[str, Any]:
    """
    Build train/valid/test splits from CyNER + CASIE, tokenize for SecureBERT.

    Returns dict with datasets and metadata.
    """
    from datasets import Dataset, DatasetDict

    logger.info("=" * 60)
    logger.info("NER DATA PREPROCESSING")
    logger.info("=" * 60)

    # 1. Parse CyNER splits
    train_sents = parse_conll_file(cyner_dir / "train.txt")
    valid_sents = parse_conll_file(cyner_dir / "valid.txt")
    test_sents = parse_conll_file(cyner_dir / "test.txt")
    logger.info("CyNER — train=%d  valid=%d  test=%d",
                len(train_sents), len(valid_sents), len(test_sents))

    # 2. Parse CASIE and add to train only
    casie_sents = parse_casie_annotations(casie_dir)
    if casie_sents:
        train_sents.extend(casie_sents)
        logger.info("After CASIE augmentation → train=%d", len(train_sents))

    # 3. Convert to HuggingFace Datasets
    def _to_hf(sents: list[dict]) -> Dataset:
        return Dataset.from_dict({
            "tokens": [s["tokens"] for s in sents],
            "ner_tags": [
                [LABEL2ID.get(l, 0) for l in s["ner_tags"]]
                for s in sents
            ],
        })

    raw_ds = DatasetDict({
        "train": _to_hf(train_sents),
        "validation": _to_hf(valid_sents),
        "test": _to_hf(test_sents),
    })

    # 4. Tokenize with sub-word alignment
    from transformers import AutoTokenizer

    logger.info("Tokenizing with %s  max_length=%d", tokenizer_name, max_length)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def tokenize_and_align(examples):
        tokenized = tokenizer(
            examples["tokens"],
            truncation=True,
            is_split_into_words=True,
            max_length=max_length,
            padding="max_length",
        )
        all_labels = []
        for i, labels in enumerate(examples["ner_tags"]):
            word_ids = tokenized.word_ids(batch_index=i)
            label_ids = []
            prev_word_id = None
            for wid in word_ids:
                if wid is None:
                    label_ids.append(-100)
                elif wid != prev_word_id:
                    label_ids.append(labels[wid] if wid < len(labels) else 0)
                else:
                    # Sub-word continuation: use I- tag or -100
                    label_ids.append(-100)
                prev_word_id = wid
            all_labels.append(label_ids)
        tokenized["labels"] = all_labels
        return tokenized

    tokenized_ds = raw_ds.map(
        tokenize_and_align,
        batched=True,
        remove_columns=raw_ds["train"].column_names,
        desc="Tokenizing",
    )

    # 5. Save
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenized_ds.save_to_disk(str(output_dir / "ner_dataset"))
    logger.info("Saved tokenized dataset → %s", output_dir / "ner_dataset")

    # Save metadata
    from src.shared.utils import save_json
    meta = {
        "label_list": LABEL_LIST,
        "label2id": LABEL2ID,
        "id2label": ID2LABEL,
        "tokenizer": tokenizer_name,
        "max_length": max_length,
        "splits": {
            "train": len(train_sents),
            "validation": len(valid_sents),
            "test": len(test_sents),
        },
        "casie_augmentation": len(casie_sents),
    }
    save_json(meta, output_dir / "ner_metadata.json")

    return {
        "dataset": tokenized_ds,
        "metadata": meta,
        "label_list": LABEL_LIST,
    }
