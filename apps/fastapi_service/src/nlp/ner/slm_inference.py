"""
ARIES — NER SLM inference engine.

Uses a local Small Language Model to extract STIX 2.1-aligned IOCs and entities.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from typing import Any

from src.nlp.ner.schemas import NERResult, IOCEntity, SecurityEvent, IOCType
from src.shared.config import ServiceSettings
from src.shared.logging import get_logger
from src.nlp.ner.inference import classify_ioc

log = get_logger("ner_slm_inference")

try:
    from llama_cpp import Llama
    from src.triage.slm_inference import get_slm
except ImportError:
    Llama = None
    get_slm = lambda x: None


def _extract_first_json_object(raw_text: str) -> str | None:
    """Return the first balanced JSON object found in model output."""
    start = raw_text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escaped = False
    for idx in range(start, len(raw_text)):
        ch = raw_text[idx]
        if escaped:
            escaped = False
            continue
        if ch == "\\":
            escaped = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return raw_text[start : idx + 1]

    return None


def _salvage_json(raw_text: str) -> dict[str, Any]:
    """Try multiple parsing strategies for imperfect SLM JSON output."""
    cleaned = raw_text.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned.replace("```json", "").replace("```", "").strip()

    candidate = _extract_first_json_object(cleaned) or cleaned

    try:
        return json.loads(candidate)
    except Exception:
        # Remove trailing commas before object/array closers
        candidate = re.sub(r",\s*([}\]])", r"\1", candidate)
        return json.loads(candidate)


def _heuristic_ner_fallback(text: str) -> dict[str, Any]:
    """Deterministic fallback when SLM output is unparsable."""
    entities: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()

    ioc_patterns: list[tuple[re.Pattern[str], str, str]] = [
        (re.compile(r"\b(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b"), "Indicator", "IP_Address"),
        (re.compile(r"\bCVE-\d{4}-\d{4,}\b", re.IGNORECASE), "Vulnerability", "CVE_ID"),
        (re.compile(r"\b[a-fA-F0-9]{32}\b|\b[a-fA-F0-9]{40}\b|\b[a-fA-F0-9]{64}\b"), "Indicator", "File_Hash"),
        (re.compile(r"\bhttps?://[^\s]+"), "Indicator", "URL"),
        (re.compile(r"\b[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+\b"), "Indicator", "Email_Address"),
        (re.compile(r"\b(?:[a-zA-Z0-9-]+\.)+[a-zA-Z]{2,}\b"), "Indicator", "Domain"),
    ]

    for pattern, label, ioc_type in ioc_patterns:
        for match in pattern.finditer(text):
            value = match.group(0)
            key = (value, label)
            if key in seen:
                continue
            entities.append({"text": value, "label": label, "ioc_type": ioc_type})
            seen.add(key)

    # Common malware/tool aliases often present in incident reports
    for term in ["Cobalt Strike", "Mimikatz", "Emotet", "TrickBot", "LockBit", "APT29"]:
        if term.lower() in text.lower() and (term, "Malware") not in seen:
            entities.append({"text": term, "label": "Malware", "ioc_type": "Unknown"})
            seen.add((term, "Malware"))

    contextual_patterns: list[tuple[re.Pattern[str], str]] = [
        (re.compile(r"\b(?:spear\s?-?phishing(?:\s+lure)?|phishing(?:\s+lure)?)\b", re.IGNORECASE), "Indicator"),
        (re.compile(r"\b(?:bulletproof hosting provider|threat actor|adversary)\b", re.IGNORECASE), "Organization"),
        (re.compile(r"\b(?:compromised endpoint|endpoint|domain controller|temp directory|server|host)\b", re.IGNORECASE), "System"),
        (re.compile(r"\b(?:suspicious executable|executable|windows dll|dll|powershell)\b", re.IGNORECASE), "Tool"),
    ]
    for pattern, label in contextual_patterns:
        for match in pattern.finditer(text):
            value = match.group(0)
            key = (value.lower(), label)
            if key in seen:
                continue
            entities.append({"text": value, "label": label, "ioc_type": "Unknown"})
            seen.add(key)

    events: list[dict[str, str]] = []
    event_keywords = {
        "ransomware": "Ransom",
        "spearphishing": "Phishing",
        "spear-phishing": "Phishing",
        "phishing": "Phishing",
        "exfiltration": "Databreach",
        "data breach": "Databreach",
        "brute force": "CredentialAttack",
        "credential theft": "CredentialAttack",
        "compromised": "CredentialAttack",
        "lateral": "LateralMovement",
        "lateral movement": "LateralMovement",
    }
    lower_text = text.lower()
    for keyword, event_type in event_keywords.items():
        if keyword in lower_text:
            events.append({"type": event_type, "keyword": keyword})

    return {"entities": entities, "events": events}


def _build_ner_prompt(text: str) -> str:
    prompt = f"""<|system|>
You are a cybersecurity Named Entity Recognition (NER) extractor.
Extract all STIX 2.1 aligned entities from the text. Valid labels are: Malware, Indicator, System, Vulnerability, Organization.
Also extract any broad Security Events (e.g., Ransomware, Phishing, Backdoor).
Respond ONLY with a valid JSON object matching this schema:
{{
  "entities": [
    {{"text": "string", "label": "string", "ioc_type": "string"}}
  ],
  "events": [
    {{"type": "string", "keyword": "string"}}
  ]
}}
Do not output any markdown formatting, explanation, or other text.
<|user|>
Text:
{text}
<|assistant|>
"""
    return prompt


def _run_slm_ner_sync(model_path: str, text: str) -> dict[str, Any]:
    prompt = _build_ner_prompt(text)
    llm = get_slm(model_path)
    
    if llm is None:
        time.sleep(0.5)
        # Mock output
        return {
            "entities": [{"text": "MockMalware", "label": "Malware", "ioc_type": "Unknown"}],
            "events": [{"type": "MockEvent", "keyword": "mock"}]
        }

    response = llm(
        prompt,
        max_tokens=160,
        stop=["<|end|>", "<|eot_id|>", "</s>", "<|assistant|>"],
        temperature=0.1
    )
    
    try:
        text_output = response["choices"][0]["text"].strip()
        parsed = _salvage_json(text_output)
        fallback = _heuristic_ner_fallback(text)

        parsed_entities = parsed.get("entities", [])
        parsed_events = parsed.get("events", [])
        if not isinstance(parsed_entities, list):
            parsed_entities = []
        if not isinstance(parsed_events, list):
            parsed_events = []

        # Backfill if model returns empty or structurally weak output.
        if not parsed_entities:
            parsed_entities = fallback.get("entities", [])
        if not parsed_events:
            parsed_events = fallback.get("events", [])

        parsed["entities"] = parsed_entities
        parsed["events"] = parsed_events
        return parsed
    except Exception as e:
        log.error("SLM NER parse failed", error=str(e))
        return _heuristic_ner_fallback(text)


async def run_ner_inference_slm(
    text: str,
    settings: ServiceSettings
) -> NERResult:
    start = time.perf_counter()
    loop = asyncio.get_running_loop()
    
    slm_result = await loop.run_in_executor(
        None, _run_slm_ner_sync, settings.effective_ner_model_path, text
    )
    
    entities = []
    for ent in slm_result.get("entities", []):
        raw_text = ent.get("text", "")
        label = ent.get("label", "O")
        # Validate IOC via existing logic
        ioc_validated = False
        ioc_type = "Unknown"
        if label == "Indicator":
            extracted_type, is_validated = classify_ioc(raw_text)
            if is_validated:
                ioc_type = extracted_type.value if isinstance(extracted_type, IOCType) else extracted_type
                ioc_validated = True

        entities.append(IOCEntity(
            text=raw_text,
            label=label,
            start=text.find(raw_text) if raw_text in text else 0,
            end=(text.find(raw_text) + len(raw_text)) if raw_text in text else 0,
            confidence=0.99,  # SLMs don't easily give token-level confidence here
            ioc_type=ioc_type,
            ioc_validated=ioc_validated
        ))
        
    events = [SecurityEvent(event_type=ev.get("type", "Unknown"), trigger=ev.get("keyword", "")) for ev in slm_result.get("events", [])]
    elapsed_ms = int((time.perf_counter() - start) * 1000)
    
    return NERResult(
        entities=entities,
        events=events,
        processing_ms=elapsed_ms,
        cached=False
    )
