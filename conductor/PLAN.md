# Semantic Triage via Local Small Language Model (SLM) & NLP Overhaul

## Background & Motivation
The current alert triage engine relies on an XGBoost classifier with Target Encoding, trained exclusively on the Microsoft-centric GUIDE dataset. Local testing revealed a fundamental flaw: when the engine ingests alerts from a new SIEM (like Wazuh), the unseen categorical values (such as `normalized_title` or `device_name`) are hashed to a default, benign prior. This causes critical alerts (e.g., SSH Brute Force, SQL Injection) to receive exceptionally low ML scores (~0.09) compared to benign routine events. The current meta for cross-SIEM generalization is "Semantic Triage"—passing the textual narrative and JSON payload of an alert to a Language Model that can reason over the context, regardless of vendor-specific schema details.

Additionally, the current NER (SecureBERT) and Summarization (BART-base) models exhibit low generalization (F1 ~0.61 for NER) and factual consistency issues (hallucinating trailing clauses). This plan outlines an architectural shift to an SLM-driven pipeline for all three tasks, with preparation for GPU-optimized training on Hydra.

## Scope & Impact
- **Impacted Components:** `apps/fastapi_service/src/triage/`, `src/nlp/ner/`, `src/nlp/summarizer/`, and Hydra deployment scripts (`slurm/`).
- **Latency & Resources:** Latency per triage/NER/Summary will increase depending on the chosen model size, but will be optimized for GPU inference on the Hydra cluster. Local testing will verify the pipeline functionally before full deployment.
- **Architectural Shift:** Moving from sparse categorical vectors and task-specific models (XGBoost, SecureBERT, BART) to a prompt-based generative SLM approach for Triage, NER, and Summarization.

## Proposed Solution
Replace the XGBoost, SecureBERT, and BART models with a Small Language Model (SLM), such as **Llama-3-8B-Instruct** or **Phi-3-mini**, deployed via vLLM or ONNX Runtime. The SLM will use prompt engineering to output structured JSON for:
1. **Triage:** TP/FP classification and confidence scores.
2. **NER:** Extracting STIX 2.1-aligned IOCs and entities.
3. **Summarization:** Generating factual, grounded incident summaries.

*Crucially, the original XGBoost and task-specific code will remain intact as a fallback.* The code will simply be rerouted to use the new SLM inference path.

## Research & Datasets for Optimization (Hydra)
Since all training and optimization will occur on the Hydra cluster, the following state-of-the-art datasets will be downloaded and used by the user for fine-tuning the SLM:
- **Triage:** Augmented GUIDE + synthetically generated cross-SIEM (Wazuh, Splunk) mappings.
- **NER:** **CyberNER (2024/2025)** - A harmonized STIX 2.1-aligned dataset combining CyNER and APTNER with ~610k tokens and 21 entity types.
- **Summarization:** **Incident Dashboard 2024-2025** / **S-RM 2025 Cyber Incident Insights** - Structured incident reports for instruction-tuning factual summarization.

## Implementation Plan

### Phase 1: Environment & Script Updates
1. Update `slurm/setup_env.sh` to include new packages required for the SLM inference and training (e.g., `vllm`, `transformers`, `accelerate`, `bitsandbytes`, `onnxruntime-genai`). *No packages will be installed locally; only the scripts will be updated.*
2. Download any necessary model weights (e.g., GGUF or ONNX quantized SLM versions) for local functional testing, ensuring they run perfectly without a local GPU.

### Phase 2: Rerouting & Fallback Preservation
1. Create new modules: `src/triage/slm_inference.py`, `src/nlp/ner/slm_inference.py`, and `src/nlp/summarizer/slm_inference.py`.
2. Do **not** delete the existing XGBoost, SecureBERT, or BART inference and feature engineering files.
3. Update the routers (`src/triage/router.py`, `src/nlp/router.py`) to conditionally route requests to the new SLM modules, with a configuration flag or fallback logic to route to the original models if needed.

### Phase 3: Prompt Construction & SLM Integration
1. Implement prompt generation functions for each task:
    - **Triage Prompt:** "Review the SIEM alert. Output JSON with `grade` and `confidence`."
    - **NER Prompt:** "Extract cyber entities (Malware, Tool, Threat-Actor, IOC) from the text. Output JSON array."
    - **Summarizer Prompt:** "Summarize the incident report factually. Do not hallucinate. Output JSON."
2. Integrate local SLM inference (e.g., using `llama-cpp-python` or ONNX Runtime GenAI) to validate the pipeline functionally on the local machine.

### Phase 4: API, Orchestration, and Readiness for Hydra
1. Adjust the parsing logic to accommodate SLM JSON responses.
2. Ensure the entire pipeline (Wazuh -> Kafka -> FastAPI -> SLM -> PostgreSQL) works end-to-end locally.
3. Prepare the `slurm/` training scripts to utilize the new datasets (CyberNER) and the SLM for full optimization on the Hydra cluster.

## Verification
- **Local End-to-End:** Run `test_triage_local.py` and endpoint `curl` commands to confirm the SLM successfully processes Wazuh payloads and outputs valid JSON for Triage, NER, and Summarization without crashing the local (CPU-only) environment.
- **Hydra Handoff:** Ensure all `slurm/` scripts and dataset downloading instructions are perfectly documented and ready for the user to execute on the GPU cluster.

## Migration & Rollback
- The original XGBoost, SecureBERT, and BART artifacts and code remain entirely intact. If the SLM pipeline fails or proves too slow, a single environment variable or config switch can revert traffic back to the legacy models.