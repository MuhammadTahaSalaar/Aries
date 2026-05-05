# ARIES Validation Findings For Supervisor Review

## Executive Summary

ARIES is now functionally validated across its core workflow: ingesting SIEM alerts, normalizing them, triaging them with the deployed ML stack, enriching the alert stream, and persisting results in PostgreSQL. During validation, three model-quality defects were found and repaired: triage scoring was too dependent on the base model output, NER missed some split or partially emitted IOCs, and the summarizer could end grounded sentences with a hallucinated trailing clause. Those fixes were validated both with targeted checks and with live service calls.

The platform is therefore in a credible demonstration state for supervisor review. The main remaining risks are not gross integration failures; they are quality ceiling and operational hardening gaps: Redis cache invalidation is still coarse, NER accuracy remains modest versus stronger recent baselines, summarization faithfulness still depends on post-processing safeguards, and Wazuh custom integration deployment still requires manual installation steps after container restarts.

## What Was Verified

### 1. Core service health

- FastAPI service loaded all three deployed models successfully: XGBoost triage, SecureBERT NER, and BART summarization.
- The service health endpoint confirmed the inference stack and backing services were available.

### 2. Direct model behavior

- Triage was revalidated after the contextual scoring fix and now escalates or suppresses alerts using the combined `ml_score`, `asset_criticality`, and `behavioral_score` path rather than the raw classifier output alone.
- NER was revalidated after IOC recovery fixes and now recovers missed IPv4 addresses, CVEs, hashes, URLs, and similar entities more reliably.
- Summarization was revalidated after the cleanup fix. The previously incorrect ending "The attack was attributed to FIN7, a security company that was responsible for the SOC's" now terminates correctly as "The attack was attributed to FIN7.".

### 3. End-to-end manual ingest

- A Wazuh-shaped alert posted to `/ingest/siem` was normalized, published to `alerts.raw`, consumed by triage, persisted in PostgreSQL, and republished to `alerts.enriched`.
- Verified database result: `manual-e2e-1777643007 | wazuh | Triaged | 0.2542 | 51.71`.

### 4. Real Wazuh integration

- The Wazuh single-node stack was brought up and connected to the shared ARIES Docker network.
- The custom Wazuh integration forwarded a real alert into ARIES, confirmed by Wazuh manager logs, FastAPI logs, Kafka publication, and PostgreSQL persistence.
- Verified database result: `1777643325.241 | wazuh | Triaged | 0.0271 | 20.86 | Wazuh server started.`
- FastAPI logs for the same alert confirmed the full sequence: normalization, `alerts.raw` publication, triage inference, persistence, and `alerts.enriched` publication.

## Fixes Applied During Validation

### Triage

- Replaced overly static enrichment behavior with contextual scoring based on alert severity, entity keywords, and MITRE context.
- Added grade overrides driven by `risk_score`, so higher-risk alerts are less likely to remain under-classified.

### NER

- Added regex-based IOC recovery for patterns the model still misses in practice.
- Added repair logic for partial entities, including split CVE and IPv4 outputs.

### Summarization

- Improved summary cleanup so that grounded sentences can be trimmed back to the longest source-supported prefix rather than only being removed at sentence boundaries.
- Added a regression test for the specific trailing-hallucination failure mode.

### Configuration and documentation

- Reconciled the checked-in Wazuh manager configuration with the documented validation flow by restoring `/var/log/auth.log` monitoring.
- Corrected the FastAPI README database verification command so it queries columns that actually exist in the `aries.alerts` persistence path.
- Corrected the validation guide to use `ossec.log` for forwarding verification and `tail -1` for the JSONL alerts log.

## Comparison With Current State Of The Art

### Alert triage

ARIES currently uses XGBoost on engineered structured features. That is not the newest approach, but it remains a strong engineering choice for production alert triage where latency, interpretability, ONNX export, and CPU deployment matter. On the project dataset, it achieved 94.19% accuracy and 93.60% macro F1, which is a strong practical result for structured alert classification.

Compared with newer transformer or LLM-based triage systems, ARIES is weaker at reasoning over richer free text, multi-alert context, or cross-source semantics. Those alternatives may outperform on heterogeneous unstructured workloads, but they carry materially higher cost, latency, and deployment complexity. For a self-hosted SOAR pipeline, the current XGBoost choice remains defensible.

### NER / IOC extraction

SecureBERT is a sensible domain-specific baseline because it was pre-trained on security text and handles cybersecurity vocabulary better than generic BERT-family models. However, the reported 61.95% F1 is the weakest metric among the three deployed pipelines and is below what would be expected from stronger modern token-classification baselines or instruction-based extraction systems on well-aligned data.

In practical terms, ARIES compensates with post-processing, but that is also evidence that the model alone is not yet robust enough. This is the clearest area where a re-benchmark against stronger encoders such as newer DeBERTa-family models, span-based extractors, or carefully constrained local LLM extraction would be justified.

### Summarization

The current BART-base summarizer is a pragmatic deployment choice, not a frontier summarization model. It is small enough for ONNX CPU inference and reached ROUGE-1 48.10 and ROUGE-L 25.33 on the project benchmark, which is acceptable for a compact local model.

Relative to more recent long-context or instruction-tuned summarizers, ARIES is likely behind on factual consistency and long-document compression. The trailing-clause hallucination bug that was fixed during validation reinforces that point. The summarizer is usable after cleanup and guardrails, but it should be presented as operationally adequate rather than state of the art.

## Overall Assessment

The platform now clears the main bar for an end-to-end project demonstration: it works across live ingestion, ML inference, enrichment, and persistence, and the most visible quality defects found during validation have been repaired. The engineering direction is coherent: fast local inference, privacy-preserving deployment, explainable triage, and domain-aware NLP.

At the same time, ARIES should not be described as state of the art across all three ML tasks. The triage component is strong and well-justified. The NER and summarization components are adequate but visibly less mature than the strongest contemporary alternatives, and they currently rely on rule-based recovery and cleanup to meet operational expectations.

## Remaining Risks And Recommended Next Steps

1. Add cache versioning or explicit invalidation so Redis cannot silently serve stale NLP outputs after model or post-processing changes.
2. Automate Wazuh integration installation or persist the integration files through container restarts to remove a manual operational step.
3. Add one reproducible end-to-end smoke test that asserts the live path from `/ingest/siem` to PostgreSQL and `alerts.enriched`.
4. Re-benchmark NER against stronger recent baselines and decide whether SecureBERT should remain the production model.
5. Re-benchmark summarization for factual consistency, not just ROUGE, before presenting it as analyst-ready at larger scale.