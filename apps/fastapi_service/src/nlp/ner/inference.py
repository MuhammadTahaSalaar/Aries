"""
ARIES — NER ONNX inference engine with IOC post-processing.

Runs SecureBERT-NER via ONNX Runtime. Includes regex-based IOC validation
for IPs, hashes, domains, URLs, CVEs, and emails.
"""

from __future__ import annotations

import asyncio
import re
import time
from functools import partial
from typing import Any

import numpy as np
import onnxruntime as ort

from src.nlp.ner.schemas import EntityLabel, IOCEntity, IOCType, NERResult, SecurityEvent
from src.shared.config import ServiceSettings
from src.shared.logging import get_logger

log = get_logger("ner_inference")

# ── IOC Regex Patterns ────────────────────────────────────────────────

IPV4_PATTERN = re.compile(
    r"^(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)$"
)
IPV6_PATTERN = re.compile(
    r"^("
    r"([0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}|"            # full
    r"([0-9a-fA-F]{1,4}:){1,7}:|"                           # trailing ::
    r"([0-9a-fA-F]{1,4}:){1,6}:[0-9a-fA-F]{1,4}|"          # :: one group
    r"([0-9a-fA-F]{1,4}:){1,5}(:[0-9a-fA-F]{1,4}){1,2}|"   # :: two groups
    r"([0-9a-fA-F]{1,4}:){1,4}(:[0-9a-fA-F]{1,4}){1,3}|"
    r"([0-9a-fA-F]{1,4}:){1,3}(:[0-9a-fA-F]{1,4}){1,4}|"
    r"([0-9a-fA-F]{1,4}:){1,2}(:[0-9a-fA-F]{1,4}){1,5}|"
    r"[0-9a-fA-F]{1,4}:(:[0-9a-fA-F]{1,4}){1,6}|"
    r":((:[0-9a-fA-F]{1,4}){1,7}|:)"                        # ::... or ::
    r")$"
)
MD5_PATTERN = re.compile(r"^[a-fA-F0-9]{32}$")
SHA1_PATTERN = re.compile(r"^[a-fA-F0-9]{40}$")
SHA256_PATTERN = re.compile(r"^[a-fA-F0-9]{64}$")
DOMAIN_PATTERN = re.compile(
    r"^(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)+"
    r"[a-zA-Z]{2,}$"
)
URL_PATTERN = re.compile(r"^https?://[^\s]+$")
EMAIL_PATTERN = re.compile(r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$")
CVE_PATTERN = re.compile(r"^CVE-\d{4}-\d{4,}$", re.IGNORECASE)

# Un-anchored search patterns for extracting IOCs embedded in larger NER spans
_IOC_SEARCH_PATTERNS: list[tuple[re.Pattern, IOCType]] = [
    (re.compile(r"(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)"), IOCType.IP_ADDRESS),
    (re.compile(r"[a-fA-F0-9]{64}"), IOCType.FILE_HASH),
    (re.compile(r"[a-fA-F0-9]{40}"), IOCType.FILE_HASH),
    (re.compile(r"[a-fA-F0-9]{32}"), IOCType.FILE_HASH),
    (re.compile(r"CVE-\d{4}-\d{4,}", re.IGNORECASE), IOCType.CVE_ID),
    (re.compile(r"https?://\S+"), IOCType.URL),
    (re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"), IOCType.EMAIL_ADDRESS),
]

# Exportable dict for testing
IOC_PATTERNS: dict[str, re.Pattern] = {
    "ipv4": IPV4_PATTERN,
    "ipv6": IPV6_PATTERN,
    "md5": MD5_PATTERN,
    "sha1": SHA1_PATTERN,
    "sha256": SHA256_PATTERN,
    "domain": DOMAIN_PATTERN,
    "url": URL_PATTERN,
    "email": EMAIL_PATTERN,
    "cve": CVE_PATTERN,
}

# ── Event Detection Keywords ─────────────────────────────────────────

EVENT_KEYWORDS: dict[str, str] = {
    "encrypted": "Ransom",
    "ransomware": "Ransom",
    "ransom": "Ransom",
    "phishing": "Phishing",
    "phish": "Phishing",
    "credential harvesting": "Phishing",
    "exfiltrated": "Databreach",
    "data breach": "Databreach",
    "data leak": "Databreach",
    "exfiltration": "Databreach",
    "patched": "Patch-Vulnerability",
    "patch": "Patch-Vulnerability",
    "remediated": "Patch-Vulnerability",
    "disclosed": "Discover-Vulnerability",
    "vulnerability": "Discover-Vulnerability",
    "zero-day": "Discover-Vulnerability",
    "0day": "Discover-Vulnerability",
}

# Mapping from NER labels to the canonical label enum
NER_LABEL_MAP: dict[str, EntityLabel] = {
    "O": EntityLabel.O,
    "B-Malware": EntityLabel.MALWARE,
    "I-Malware": EntityLabel.MALWARE,
    "B-Indicator": EntityLabel.INDICATOR,
    "I-Indicator": EntityLabel.INDICATOR,
    "B-System": EntityLabel.SYSTEM,
    "I-System": EntityLabel.SYSTEM,
    "B-Vulnerability": EntityLabel.VULNERABILITY,
    "I-Vulnerability": EntityLabel.VULNERABILITY,
    "B-Organization": EntityLabel.ORGANIZATION,
    "I-Organization": EntityLabel.ORGANIZATION,
    "B-Tool": EntityLabel.TOOL,
    "I-Tool": EntityLabel.TOOL,
}

# Label ID → label string (loaded from metadata)
DEFAULT_ID2LABEL: dict[int, str] = {
    0: "O",
    1: "B-Malware",
    2: "I-Malware",
    3: "B-Indicator",
    4: "I-Indicator",
    5: "B-System",
    6: "I-System",
    7: "B-Vulnerability",
    8: "I-Vulnerability",
    9: "B-Organization",
    10: "I-Organization",
}


def classify_ioc(text: str) -> tuple[IOCType, bool]:
    """Regex-validate a text span as a specific IOC type."""
    text = text.strip()

    if IPV4_PATTERN.match(text) or IPV6_PATTERN.match(text):
        return IOCType.IP_ADDRESS, True

    if SHA256_PATTERN.match(text):
        return IOCType.FILE_HASH, True
    if SHA1_PATTERN.match(text):
        return IOCType.FILE_HASH, True
    if MD5_PATTERN.match(text):
        return IOCType.FILE_HASH, True

    if CVE_PATTERN.match(text):
        return IOCType.CVE_ID, True

    if URL_PATTERN.match(text):
        return IOCType.URL, True

    if EMAIL_PATTERN.match(text):
        return IOCType.EMAIL_ADDRESS, True

    if DOMAIN_PATTERN.match(text) and "." in text:
        return IOCType.DOMAIN, True

    return IOCType.UNKNOWN, False


def _run_ner_onnx(
    session: ort.InferenceSession,
    input_ids: np.ndarray,
    attention_mask: np.ndarray,
) -> np.ndarray:
    """Synchronous ONNX NER inference. Returns logits [batch, seq, num_labels]."""
    results = session.run(
        None,
        {
            "input_ids": input_ids.astype(np.int64),
            "attention_mask": attention_mask.astype(np.int64),
        },
    )
    return results[0]  # logits


def _align_predictions(
    tokens: list[str],
    label_ids: list[int],
    offsets: list[tuple[int, int]],
    text: str,
    id2label: dict[int, str],
) -> list[IOCEntity]:
    """
    Convert token-level label IDs back to word-level entity spans.
    Collapses BIO sub-word tokens into unified entities.

    Handles the common case where RoBERTa sub-word tokens each receive a
    ``B-`` tag instead of ``I-`` by merging contiguous same-label spans
    that have no whitespace gap between them.
    """
    entities: list[IOCEntity] = []
    current_entity: dict[str, Any] | None = None

    for idx, (token, label_id, offset) in enumerate(zip(tokens, label_ids, offsets)):
        if offset == (0, 0):
            # Special token — skip
            if current_entity:
                entities.append(_finalize_entity(current_entity, text))
                current_entity = None
            continue

        label_str = id2label.get(label_id, "O")
        bio_prefix = label_str[0] if "-" in label_str else "O"
        entity_type = label_str[2:] if "-" in label_str else "O"

        start, end = offset

        if bio_prefix == "B":
            if (
                current_entity
                and current_entity["label"] == entity_type
                and start <= current_entity["end"] + 1
            ):
                # Contiguous or overlapping same-label B- token → merge
                current_entity["end"] = end
                current_entity["tokens"].append(token)
                current_entity["label_ids"].append(label_id)
            else:
                # Genuinely new entity
                if current_entity:
                    entities.append(_finalize_entity(current_entity, text))
                current_entity = {
                    "label": entity_type,
                    "start": start,
                    "end": end,
                    "tokens": [token],
                    "label_ids": [label_id],
                }
        elif bio_prefix == "I" and current_entity and current_entity["label"] == entity_type:
            # Continuation of current entity
            current_entity["end"] = end
            current_entity["tokens"].append(token)
            current_entity["label_ids"].append(label_id)
        else:
            # O label or mismatched I — close current entity
            if current_entity:
                entities.append(_finalize_entity(current_entity, text))
                current_entity = None

    if current_entity:
        entities.append(_finalize_entity(current_entity, text))

    return entities


def _finalize_entity(entity_dict: dict[str, Any], text: str) -> IOCEntity:
    """Convert a raw entity dict into a validated IOCEntity."""
    start = entity_dict["start"]
    end = entity_dict["end"]
    span = text[start:end].strip()

    label = NER_LABEL_MAP.get(f"B-{entity_dict['label']}", EntityLabel.INDICATOR)
    ioc_type, is_validated = classify_ioc(span)

    # If the NER span has surrounding context, try to extract the IOC within
    if not is_validated and label == EntityLabel.INDICATOR:
        for pat, itype in _IOC_SEARCH_PATTERNS:
            m = pat.search(span)
            if m:
                ioc_type = itype
                is_validated = True
                match_start = m.start()
                span = m.group(0)
                start = start + match_start
                end = start + len(span)
                break

    return IOCEntity(
        text=span,
        label=label,
        start=start,
        end=end,
        confidence=0.0,  # Updated below with softmax
        ioc_type=ioc_type,
        ioc_validated=is_validated,
    )


def detect_events(text: str) -> list[SecurityEvent]:
    """Rule-based event detection from text using keyword matching."""
    events: list[SecurityEvent] = []
    text_lower = text.lower()
    seen_types: set[str] = set()

    for keyword, event_type in EVENT_KEYWORDS.items():
        if keyword in text_lower and event_type not in seen_types:
            seen_types.add(event_type)
            events.append(
                SecurityEvent(
                    event_type=event_type,
                    trigger=keyword,
                    confidence=0.7,
                )
            )

    return events


# ── Regex patterns for IOC recovery (un-anchored, for scanning full text) ──

_RECOVERY_PATTERNS: list[tuple[re.Pattern, IOCType, EntityLabel]] = [
    (re.compile(r"CVE-\d{4}-\d{4,}", re.IGNORECASE), IOCType.CVE_ID, EntityLabel.VULNERABILITY),
    (re.compile(
        r"(?<![a-fA-F0-9])[a-fA-F0-9]{64}(?![a-fA-F0-9])"), IOCType.FILE_HASH, EntityLabel.INDICATOR),
    (re.compile(
        r"(?<![a-fA-F0-9])[a-fA-F0-9]{40}(?![a-fA-F0-9])"), IOCType.FILE_HASH, EntityLabel.INDICATOR),
    (re.compile(
        r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b"
    ), IOCType.IP_ADDRESS, EntityLabel.INDICATOR),
    (re.compile(
        r"\bhttps?://[^\s\"'<>]+"), IOCType.URL, EntityLabel.INDICATOR),
    (re.compile(
        r"\b[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+\b"), IOCType.EMAIL_ADDRESS, EntityLabel.INDICATOR),
]


def _recover_missed_iocs(
    text: str, existing_entities: list[IOCEntity]
) -> list[IOCEntity]:
    """Scan the full text for IOCs that the NER model missed.

    This catches cases where sub-word tokenization splits IOCs
    (e.g. ``CVE-`` and ``2021-44228`` ending up as separate entities)
    and where the model fails to tag well-formed IOCs entirely.
    """
    # First fix partial entities (e.g. "CVE-" → "CVE-2021-44228")
    # so the covered set reflects the expanded spans.
    existing_entities = _fix_partial_entities(text, existing_entities)

    # Build a set of character ranges already covered by existing entities
    covered: set[int] = set()
    for ent in existing_entities:
        covered.update(range(ent.start, ent.end))

    suppressed_existing: set[int] = set()
    new_entities: list[IOCEntity] = []

    for pattern, ioc_type, label in _RECOVERY_PATTERNS:
        for match in pattern.finditer(text):
            start, end = match.start(), match.end()
            span_text = match.group(0)

            # Skip if this span is already (mostly) covered by an existing entity
            span_range = set(range(start, end))
            overlap = len(span_range & covered)
            overlapping_entities = [
                (idx, ent)
                for idx, ent in enumerate(existing_entities)
                if ent.start < end and ent.end > start
            ]
            fragmented_existing = bool(overlapping_entities) and all(
                ent.start >= start
                and ent.end <= end
                and ent.label == label
                and not ent.ioc_validated
                for _, ent in overlapping_entities
            )
            if overlap > len(span_range) * 0.5 and not fragmented_existing:
                continue

            # For hashes, skip if it looks like a common word/hex prefix
            if ioc_type == IOCType.FILE_HASH and len(span_text) == 40:
                # Avoid false positives — ensure it's clearly a hash context
                context_before = text[max(0, start - 20):start].lower()
                if not any(kw in context_before for kw in [
                    "hash", "sha", "md5", "checksum", "ioc", "indicator", "file"
                ]):
                    continue

            if fragmented_existing:
                suppressed_existing.update(idx for idx, _ in overlapping_entities)

            new_entities.append(IOCEntity(
                text=span_text,
                label=label,
                start=start,
                end=end,
                confidence=max(
                    0.85,
                    max((ent.confidence for _, ent in overlapping_entities), default=0.0),
                ),
                ioc_type=ioc_type,
                ioc_validated=True,
            ))
            covered.update(range(start, end))

    merged_entities = [
        ent for idx, ent in enumerate(existing_entities) if idx not in suppressed_existing
    ]
    return sorted(merged_entities + new_entities, key=lambda ent: (ent.start, ent.end))


def _fix_partial_entities(
    text: str, entities: list[IOCEntity]
) -> list[IOCEntity]:
    """Fix entities that were truncated by tokenization.

    E.g. if NER extracted ``CVE-`` but the full IOC is ``CVE-2021-44228``,
    expand the entity span.
    """
    fixed: list[IOCEntity] = []
    for ent in entities:
        if ent.text.upper().startswith("CVE-") and not CVE_PATTERN.match(ent.text):
            # Try to expand the CVE
            m = re.search(r"CVE-\d{4}-\d{4,}", text[ent.start:ent.start + 30], re.IGNORECASE)
            if m:
                full_cve = m.group(0)
                fixed.append(IOCEntity(
                    text=full_cve,
                    label=EntityLabel.VULNERABILITY,
                    start=ent.start,
                    end=ent.start + len(full_cve),
                    confidence=max(ent.confidence, 0.9),
                    ioc_type=IOCType.CVE_ID,
                    ioc_validated=True,
                ))
                continue
        fixed.append(ent)
    return fixed


async def run_ner_inference(
    session: ort.InferenceSession | None,
    tokenizer: Any,
    text: str,
    id2label: dict[int, str] | None = None,
    max_length: int = 512,
) -> NERResult:
    """
    Run NER inference asynchronously. ONNX calls dispatched to executor.
    Includes IOC post-processing and event detection.
    """
    start_time = time.perf_counter()

    if id2label is None:
        id2label = DEFAULT_ID2LABEL

    if session is None:
        # Return empty result if model not loaded
        events = detect_events(text)
        elapsed = int((time.perf_counter() - start_time) * 1000)
        return NERResult(entities=[], events=events, processing_ms=elapsed)

    # Tokenize
    if hasattr(tokenizer, "encode_batch"):
        # tokenizers.Tokenizer (fast)
        encoding = tokenizer.encode(text)
        input_ids = np.array([encoding.ids[:max_length]], dtype=np.int64)
        attention_mask = np.array([encoding.attention_mask[:max_length]], dtype=np.int64)
        offsets = encoding.offsets[:max_length]
        tokens = encoding.tokens[:max_length]
    else:
        # HuggingFace AutoTokenizer
        enc = tokenizer(
            text,
            return_tensors="np",
            truncation=True,
            max_length=max_length,
            return_offsets_mapping=True,
        )
        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]
        offsets = enc["offset_mapping"][0].tolist()
        tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())

    # Run ONNX in executor
    loop = asyncio.get_running_loop()
    logits = await loop.run_in_executor(
        None, partial(_run_ner_onnx, session, input_ids, attention_mask)
    )

    # Argmax over labels
    predictions = np.argmax(logits[0], axis=-1).tolist()

    # Compute per-token confidence via softmax
    exp_logits = np.exp(logits[0] - np.max(logits[0], axis=-1, keepdims=True))
    softmax = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
    confidences = np.max(softmax, axis=-1).tolist()

    # Align predictions back to word-level entities
    offsets_tuples = [(int(o[0]), int(o[1])) for o in offsets]
    entities = _align_predictions(
        tokens=tokens if isinstance(tokens, list) else list(tokens),
        label_ids=predictions,
        offsets=offsets_tuples,
        text=text,
        id2label=id2label,
    )

    # Update confidence scores from softmax
    for entity in entities:
        # Average confidence of tokens in the span
        token_indices = [
            i
            for i, (s, e) in enumerate(offsets_tuples)
            if s >= entity.start and e <= entity.end and (s, e) != (0, 0)
        ]
        if token_indices:
            entity.confidence = round(
                float(np.mean([confidences[i] for i in token_indices])), 4
            )

    # Post-process: recover IOCs the NER model may have missed due to
    # sub-word tokenization (e.g. CVE-2021-44228 split across tokens)
    entities = _recover_missed_iocs(text, entities)

    # Detect security events
    events = detect_events(text)

    elapsed = int((time.perf_counter() - start_time) * 1000)
    log.info(
        "ner_inference_complete",
        n_entities=len(entities),
        n_events=len(events),
        latency_ms=elapsed,
    )

    return NERResult(
        entities=entities,
        events=events,
        processing_ms=elapsed,
    )
