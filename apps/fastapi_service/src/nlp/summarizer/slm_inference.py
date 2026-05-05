"""
ARIES — Summarizer SLM inference engine.

Uses a local Small Language Model to generate factual incident summaries.
"""

from __future__ import annotations

import asyncio
import re
import time
from typing import Any

from src.nlp.summarizer.inference import _clean_summary, _lead_sentences
from src.nlp.summarizer.schemas import SummarizeMode, SummarizeResult
from src.shared.config import ServiceSettings
from src.shared.logging import get_logger

log = get_logger("summarizer_slm_inference")

_OFF_TOPIC_MARKERS = [
    "san francisco",
    "(kron)",
    "a new study",
    "according to a new study",
    "breaking news",
]
def _trim_mid_sentence_drift(summary: str, min_words: int = 8) -> str:
    """Truncate known off-topic drift that may begin mid-sentence."""
    lowered = summary.lower()
    cut_positions: list[int] = []
    for marker in _OFF_TOPIC_MARKERS:
        idx = lowered.find(marker)
        if idx > 0:
            cut_positions.append(idx)

    if not cut_positions:
        return summary

    candidate = summary[: min(cut_positions)].strip().rstrip(",;:-")
    if len(candidate.split()) < min_words:
        return summary

    return candidate if candidate.endswith((".", "!", "?")) else f"{candidate}."


def _clean_slm_summary(summary: str, source_text: str) -> str:
    """Normalize SLM output and apply source-grounded cleanup."""
    summary = summary.strip()
    for marker in ["<|user|>", "<|assistant|>", "<|system|>", "```"]:
        marker_pos = summary.find(marker)
        if marker_pos != -1:
            summary = summary[:marker_pos].strip()

    # Handle drift that begins mid-sentence (no punctuation boundary yet).
    summary = _trim_mid_sentence_drift(summary)

    cleaned = _clean_summary(summary, source_text)

    # Guardrail: if cleanup over-trims to a fragment, use source lead sentences.
    if len(cleaned.split()) < 8 and len(summary.split()) >= 12:
        return _lead_sentences(source_text, max_chars=220)

    return cleaned

try:
    from llama_cpp import Llama
    from src.triage.slm_inference import get_slm, _slm_lock
except ImportError:
    import threading
    Llama = None
    get_slm = lambda x: None
    _slm_lock = threading.Lock()


def _build_summary_prompt(text: str, mode: SummarizeMode) -> str:
    if mode == SummarizeMode.EXECUTIVE:
        instruction = "Provide a high-level executive summary in 2 to 4 sentences."
    else:
        instruction = "Provide a detailed analyst summary in 8 to 15 sentences, covering all technical details."
        
    prompt = f"""<|system|>
You are a cybersecurity analyst. Your task is to summarize the following security incident report.
{instruction}
Be completely factual. Do not hallucinate any details, IPs, or actors not present in the text.
Respond ONLY with the summary text. Do not include any JSON formatting or preamble.
<|user|>
Incident Report:
{text}
<|assistant|>
"""
    return prompt


def _run_slm_summary_sync(model_path: str, text: str, mode: SummarizeMode) -> str:
    prompt = _build_summary_prompt(text, mode)
    llm = get_slm(model_path)
    
    if llm is None:
        time.sleep(1.0)
        return f"Mock {mode.value} summary for local testing. The SLM is not installed."

    max_tokens = 96 if mode == SummarizeMode.EXECUTIVE else 220
    with _slm_lock:
        response = llm(
            prompt,
            max_tokens=max_tokens,
            stop=["<|end|>", "<|eot_id|>", "</s>", "<|user|>", "<|assistant|>"],
            temperature=0.1
        )

    raw = response["choices"][0]["text"].strip()
    return _clean_slm_summary(raw, text)


async def run_summarization_inference_slm(
    text: str,
    mode: SummarizeMode,
    settings: ServiceSettings
) -> SummarizeResult:
    start = time.perf_counter()
    loop = asyncio.get_running_loop()
    
    summary_text = await loop.run_in_executor(
        None, _run_slm_summary_sync, settings.effective_summarizer_model_path, text, mode
    )
    
    elapsed_ms = int((time.perf_counter() - start) * 1000)
    
    return SummarizeResult(
        summary=summary_text,
        mode=mode,
        model_version="slm-v1",
        processing_ms=elapsed_ms,
        cached=False,
        input_tokens=len(text.split()),  # rough estimate
        output_tokens=len(summary_text.split())
    )
