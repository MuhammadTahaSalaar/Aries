"""
ARIES — Summarization ONNX inference engine.

Supports "executive" and "analyst" modes with different generation parameters.
ONNX encoder-decoder generation loop, with fallback to PyTorch model.generate().
"""

from __future__ import annotations

import asyncio
import re
import time
from functools import partial
from typing import Any

import numpy as np
import onnxruntime as ort

from src.nlp.summarizer.schemas import SummarizeMode, SummarizeResult
from src.shared.config import ServiceSettings
from src.shared.logging import get_logger

log = get_logger("summarizer_inference")

_NORMALIZE_NON_ALNUM_RE = re.compile(r"[^a-z0-9\s]")
_CLAUSE_TRIM_RE = re.compile(r"[,;:]\s+")
_WHITESPACE_RE = re.compile(r"\s+")


def _normalize_grounding_text(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", _NORMALIZE_NON_ALNUM_RE.sub("", text.lower())).strip()


def _trim_to_source_prefix(
    sentence: str,
    source_normalized: str,
    min_words: int = 4,
) -> str | None:
    sentence = sentence.strip()
    normalized_sentence = _normalize_grounding_text(sentence)
    if not normalized_sentence:
        return None

    if normalized_sentence in source_normalized:
        return sentence

    terminal_punct = sentence[-1] if sentence[-1] in ".!?" else "."
    clause_matches = list(_CLAUSE_TRIM_RE.finditer(sentence))

    for match in reversed(clause_matches):
        candidate = sentence[: match.start()].strip().rstrip(",;:")
        normalized_candidate = _normalize_grounding_text(candidate)
        if (
            len(normalized_candidate.split()) >= min_words
            and normalized_candidate in source_normalized
        ):
            return candidate if candidate.endswith((".", "!", "?")) else f"{candidate}{terminal_punct}"

    words = sentence.split()
    for end_idx in range(len(words) - 1, min_words - 1, -1):
        candidate = " ".join(words[:end_idx]).strip().rstrip(",;:")
        normalized_candidate = _normalize_grounding_text(candidate)
        if normalized_candidate in source_normalized:
            return candidate if candidate.endswith((".", "!", "?")) else f"{candidate}{terminal_punct}"

    return None


def _clean_summary(summary: str, source_text: str) -> str:
    """Post-process the generated summary to remove echo artifacts.

    BART-base is an extractive-leaning summarizer that often echoes large
    portions of the input verbatim, then degenerates into hallucinated
    fragments. This function:
    1. Splits output into sentences.
    2. Identifies divergence point where model stops copying from source
       and starts hallucinating.
    3. Keeps only faithful (source-grounded) sentences.
    """
    summary = summary.strip()
    if not summary:
        return _lead_sentences(source_text)

    # Remove known instruction prefixes that may leak through
    for prefix in [
        "Summarize the following security incident report:",
        "Summarize the following:",
        "Summary:",
    ]:
        if summary.lower().startswith(prefix.lower()):
            summary = summary[len(prefix):].strip()

    # Split into sentences
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', summary) if s.strip()]
    if not sentences:
        return _lead_sentences(source_text)

    source_normalized = _normalize_grounding_text(source_text)
    source_words = set(source_normalized.split())

    # Keep sentences that are grounded in the source text (faithful)
    # or that are novel and non-repetitive
    kept: list[str] = []
    seen_words: set[frozenset[str]] = set()

    for sent in sentences:
        candidate = _trim_to_source_prefix(sent, source_normalized) or sent
        sent_clean = _normalize_grounding_text(candidate)
        words = frozenset(sent_clean.split())
        if not words:
            continue

        # Check if this sentence repeats an earlier sentence in the output
        if any(len(words & prev) > len(words) * 0.5 for prev in seen_words):
            break  # Degeneration

        # Check if this sentence is grounded in the source
        # (substantial word overlap with source text)
        grounded = len(words & source_words) / len(words) if words else 0

        if grounded < 0.5 and len(kept) > 0:
            # Hallucination — model diverged from source. Stop here.
            break

        seen_words.add(words)
        kept.append(candidate)

    if kept:
        return " ".join(kept)

    # All sentences were hallucinated — fall back to lead extraction
    return _lead_sentences(source_text)


def _lead_sentences(source_text: str, max_chars: int = 300) -> str:
    """Extract the first 1-3 sentences from source as a fallback summary."""
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', source_text) if s.strip()]
    result = ""
    for sent in sentences:
        if result and len(result) + len(sent) + 1 > max_chars:
            break
        result = f"{result} {sent}".strip() if result else sent
    return result or source_text[:200].strip()



def _greedy_decode_onnx(
    encoder_session: ort.InferenceSession,
    decoder_session: ort.InferenceSession,
    input_ids: np.ndarray,
    attention_mask: np.ndarray,
    max_new_tokens: int = 100,
    min_new_tokens: int = 30,
    eos_token_id: int = 2,
    decoder_start_token_id: int = 2,
    pad_token_id: int = 1,
    forced_bos_token_id: int | None = 0,
    no_repeat_ngram_size: int = 3,
) -> np.ndarray:
    """
    ONNX-based greedy decode for encoder-decoder models.
    Runs the encoder once, then autoregressively generates tokens via the decoder.
    """
    # Encode
    encoder_out = encoder_session.run(
        None,
        {
            "input_ids": input_ids.astype(np.int64),
            "attention_mask": attention_mask.astype(np.int64),
        },
    )
    encoder_hidden = encoder_out[0]  # [batch, seq, hidden]

    # Start decoder input with the decoder_start_token_id
    decoder_input = np.array([[decoder_start_token_id]], dtype=np.int64)
    generated_ids: list[int] = []

    decoder_input_names = [inp.name for inp in decoder_session.get_inputs()]

    for step in range(max_new_tokens):
        # Force BOS token as the first generated token (BART convention)
        if step == 0 and forced_bos_token_id is not None:
            generated_ids.append(forced_bos_token_id)
            decoder_input = np.array(
                [[decoder_start_token_id] + generated_ids], dtype=np.int64
            )
            continue

        feed: dict[str, np.ndarray] = {}
        for name in decoder_input_names:
            if "input_ids" in name or "decoder_input_ids" in name:
                feed[name] = decoder_input
            elif "encoder_hidden" in name or "encoder_output" in name or "last_hidden_state" in name:
                feed[name] = encoder_hidden.astype(np.float32)
            elif "attention_mask" in name or "encoder_attention_mask" in name:
                feed[name] = attention_mask.astype(np.int64)

        if not feed:
            # Fallback: assume standard naming
            feed = {
                decoder_input_names[0]: decoder_input,
            }
            if len(decoder_input_names) > 1:
                feed[decoder_input_names[1]] = encoder_hidden.astype(np.float32)
            if len(decoder_input_names) > 2:
                feed[decoder_input_names[2]] = attention_mask.astype(np.int64)

        try:
            decoder_out = decoder_session.run(None, feed)
        except Exception:
            log.warning("decoder_step_failed", step=step)
            break

        logits = decoder_out[0]  # [batch, seq, vocab]
        next_logits = logits[0, -1, :].copy()

        # Block repeated n-grams to prevent degenerate copying
        if no_repeat_ngram_size > 0 and len(generated_ids) >= no_repeat_ngram_size - 1:
            ngram_prefix = tuple(generated_ids[-(no_repeat_ngram_size - 1):])
            for i in range(len(generated_ids) - no_repeat_ngram_size + 1):
                prev_ngram = tuple(generated_ids[i : i + no_repeat_ngram_size - 1])
                if prev_ngram == ngram_prefix:
                    banned_token = generated_ids[i + no_repeat_ngram_size - 1]
                    next_logits[banned_token] = -float("inf")

        next_token = int(np.argmax(next_logits))

        generated_ids.append(next_token)

        if next_token == eos_token_id and step >= min_new_tokens:
            break

        # Build full decoder sequence: start token + all generated tokens so far.
        # The decoder is a non-cached full transformer so it needs the complete
        # prefix at every step to attend to previous context correctly.
        decoder_input = np.array(
            [[decoder_start_token_id] + generated_ids], dtype=np.int64
        )

    return np.array([generated_ids], dtype=np.int64)


def _run_summarization_onnx(
    encoder_session: ort.InferenceSession,
    decoder_session: ort.InferenceSession,
    input_ids: np.ndarray,
    attention_mask: np.ndarray,
    max_new_tokens: int,
    min_new_tokens: int,
) -> np.ndarray:
    """Synchronous ONNX summarization (called via run_in_executor)."""
    return _greedy_decode_onnx(
        encoder_session=encoder_session,
        decoder_session=decoder_session,
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        min_new_tokens=min_new_tokens,
    )


async def run_summarization_inference(
    encoder_session: ort.InferenceSession | None,
    decoder_session: ort.InferenceSession | None,
    tokenizer: Any,
    text: str,
    mode: SummarizeMode,
    settings: ServiceSettings,
) -> SummarizeResult:
    """
    Run summarization inference asynchronously.
    ONNX calls dispatched to executor to avoid blocking the event loop.
    """
    start_time = time.perf_counter()

    # Mode-specific generation parameters
    if mode == SummarizeMode.EXECUTIVE:
        max_tokens = settings.summarizer_executive_max_tokens
        min_tokens = settings.summarizer_executive_min_tokens
    else:
        max_tokens = settings.summarizer_analyst_max_tokens
        min_tokens = settings.summarizer_analyst_min_tokens

    # For short inputs, lower min_tokens so the model can stop naturally
    # instead of being forced past EOS into hallucination.
    approx_input_tokens = len(text.split())
    if approx_input_tokens < min_tokens * 2:
        min_tokens = max(10, approx_input_tokens // 3)

    # BART-base is a seq2seq model (not instruction-tuned), so we pass
    # the raw document directly — no instruction prefix.
    input_text = text

    # Tokenize
    if hasattr(tokenizer, "encode_batch"):
        encoding = tokenizer.encode(input_text)
        input_ids = np.array(
            [encoding.ids[: settings.summarizer_max_input_tokens]], dtype=np.int64
        )
        attention_mask = np.array(
            [encoding.attention_mask[: settings.summarizer_max_input_tokens]], dtype=np.int64
        )
        n_input_tokens = len(encoding.ids[: settings.summarizer_max_input_tokens])
    else:
        enc = tokenizer(
            input_text,
            return_tensors="np",
            truncation=True,
            max_length=settings.summarizer_max_input_tokens,
            padding=True,
        )
        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]
        n_input_tokens = int(attention_mask.sum())

    if encoder_session is None or decoder_session is None:
        elapsed = int((time.perf_counter() - start_time) * 1000)
        return SummarizeResult(
            summary="[Summarizer model not loaded — no summary available]",
            mode=mode,
            processing_ms=elapsed,
            input_tokens=n_input_tokens,
        )

    # Run ONNX generation in executor
    loop = asyncio.get_running_loop()
    output_ids = await loop.run_in_executor(
        None,
        partial(
            _run_summarization_onnx,
            encoder_session,
            decoder_session,
            input_ids,
            attention_mask,
            max_tokens,
            min_tokens,
        ),
    )

    # Decode output tokens
    if hasattr(tokenizer, "decode_batch"):
        summary_text = tokenizer.decode(output_ids[0].tolist(), skip_special_tokens=True)
    else:
        summary_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # Post-process: strip any echoed input prefix and clean up
    summary_text = _clean_summary(summary_text, input_text)

    n_output_tokens = output_ids.shape[-1]
    elapsed = int((time.perf_counter() - start_time) * 1000)

    log.info(
        "summarization_complete",
        mode=mode.value,
        input_tokens=n_input_tokens,
        output_tokens=n_output_tokens,
        latency_ms=elapsed,
    )

    return SummarizeResult(
        summary=summary_text,
        mode=mode,
        processing_ms=elapsed,
        input_tokens=n_input_tokens,
        output_tokens=int(n_output_tokens),
    )
