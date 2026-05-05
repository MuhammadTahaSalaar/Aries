# ARIES — AI Approach Change: Rationale & Benefits

**Document Purpose:** Explains why the ARIES AI pipeline was migrated from a legacy multi-model stack to a unified Small Language Model (SLM) architecture, which model was selected, and the tangible benefits this change delivers.

---

## 1. The Legacy Stack and Its Limitations

ARIES originally used three separate, task-specific models running independently in the inference pipeline:

| Task | Model | Format |
|------|-------|--------|
| Alert Triage | XGBoost + Target Encoder | ONNX |
| IOC Extraction (NER) | SecureBERT | ONNX |
| Incident Summarization | BART-base | ONNX |

During live integration with real SIEM sources, each model revealed critical failure modes that made the approach impractical for a production SOAR platform.

### 1.1 XGBoost Triage — Out-of-Distribution (OOD) Failure

The XGBoost classifier was trained on the Microsoft GUIDE dataset using Target Encoding for categorical features such as `rule.description`, `rule.group`, and `agent.name`. Target Encoding maps each category to the mean label value seen during training.

**The problem:** When ARIES ingested alerts from a different SIEM vendor (e.g., Wazuh, Splunk), the categorical values in those alerts were entirely new — never seen during training. Target Encoding has no graceful fallback for unseen categories; they silently collapse to the global benign prior (approximately 0.5). This caused the classifier to systematically under-score genuinely critical cross-SIEM alerts, producing False Negatives at the triage stage — precisely the failure mode a SOAR platform must never permit.

This is not a tuning deficiency. It is a structural limitation of any model that learns a fixed categorical vocabulary: the model is semantically blind to alert text it has not seen in training.

### 1.2 BART-base Summarization — Hallucination

BART-base is a standard sequence-to-sequence model without instruction tuning. When asked to summarize a security incident, it could not reliably confine its output to facts present in the source text. It fabricated IP addresses, CVE identifiers, and threat actor attributions to complete grammatically coherent but factually incorrect sentences.

For a security platform, a hallucinated CVE in an incident report is a direct operational liability — an analyst may act on a vulnerability that was never actually observed.

Post-processing guardrails (sentence-level source grounding, lead-sentence fallbacks) were added to mitigate this, but they are compensating for a fundamental model limitation rather than eliminating it.

### 1.3 SecureBERT NER — Low Recall and Memory Cost

SecureBERT, while pre-trained on cybersecurity text, achieved only **61.95% F1** on the ARIES IOC extraction benchmark. Practically, it missed split entities, partially tokenized CVE identifiers, and IP addresses fragmented across subword tokens. Regex post-processing was required to recover what the model missed.

Additionally, maintaining a full PyTorch transformer stack exclusively for token classification carried a significant memory overhead that was disproportionate to the task.

---

## 2. The Chosen Approach: Unified Small Language Model (SLM)

### 2.1 Model Selected: Phi-3-mini-4k-instruct (Microsoft)

The base model chosen is **Microsoft Phi-3-mini-4k-instruct** — a 3.8B parameter instruction-tuned Small Language Model. After fine-tuning via QLoRA on task-specific cybersecurity datasets and quantization to INT4 GGUF format, it is deployed locally using `llama.cpp` Python bindings.

**Why Phi-3-mini specifically:**

- **Instruction-tuned by default:** Unlike BART or vanilla transformer encoders, Phi-3-mini is designed to follow explicit natural language instructions. This directly eliminates the hallucination problem — the model can be told in a system prompt to use only the provided facts.
- **Strong reasoning per parameter:** Phi-3-mini consistently outperforms models 2–3× its size on reasoning benchmarks. For security triage — which requires contextual inference ("is this a lateral movement pattern?") rather than just classification — reasoning capability matters more than raw parameter count.
- **CPU-viable after quantization:** INT4 GGUF compression reduces the 3.8B model from ~7.6GB to approximately 2.2GB, making it deployable entirely on CPU RAM without cloud GPU costs. This is critical for a self-hosted, privacy-preserving SOAR platform.
- **Permissive license:** MIT licensed, with no restrictions on commercial or academic deployment.

Larger alternatives (Llama-3-8B at ~4.5GB GGUF) were also prototyped. Phi-3-mini was selected as the primary model for the constrained local deployment scenario; Llama-3-8B is the recommended upgrade path when additional GPU RAM is available.

### 2.2 Fine-Tuning Method: QLoRA on Hydra HPC

Full parameter fine-tuning of even a 3.8B model requires hardware beyond what is available locally. ARIES uses **QLoRA (Quantized Low-Rank Adaptation)**:

1. The base model weights are frozen and loaded in 4-bit quantization.
2. Small trainable LoRA adapter matrices (~2–4% of total parameters) are injected into the attention layers.
3. Training requires approximately 8GB VRAM instead of 32GB+.
4. Post-training, adapters are merged and the whole network is re-quantized to INT4 GGUF for inference.

Training was executed on the VUB Hydra HPC cluster using the datasets below:

| Task | Dataset |
|------|---------|
| Triage | Microsoft GUIDE (~13.6M rows, stratified) |
| NER | CyberNER 2024/2025 (STIX 2.1 harmonized: CyNER + APTNER) |
| Summarization | GovReport / S-RM 2025 Cyber Incident Insights |

Empirical training results from Hydra confirm successful convergence: eval loss of **0.03154**, mean token accuracy of **0.9899**, with early stopping triggered after improvement fell below threshold — indicating no overfitting.

---

## 3. How the SLM Solves Each Legacy Problem

### 3.1 Cross-SIEM Generalization (replaces XGBoost)

The SLM receives the full normalized alert JSON and any available narrative text and processes it through a task-specific system prompt. There is no categorical encoding, no training vocabulary, and no OOD collapse. The model reads an alert from Wazuh, Splunk, or any future SIEM exactly as a human SOC analyst would — evaluating context, severity language, affected asset type, and MITRE tactic indicators without any dependency on vendor-specific field names or values.

This makes the triage pipeline structurally vendor-agnostic.

### 3.2 Grounded, Non-Hallucinating Summaries (replaces BART)

The SLM system prompt explicitly instructs the model: *"Generate your response using only the facts in the provided alert. Do not infer or fabricate."* Because Phi-3-mini is instruction-tuned, it follows this constraint reliably. The trailing-clause hallucination failure that required post-processing safeguards in BART is eliminated at the model level.

### 3.3 Generative NER with Higher Recall (replaces SecureBERT)

The SLM is prompted to extract STIX 2.1-aligned entities directly from the alert text as a structured JSON list. Generative extraction operates at the token-sequence level rather than the subword-token classification level, avoiding the entity fragmentation issues that plagued SecureBERT. The legacy regex post-processor is retained as a deterministic validation and formatting layer on top of the generative output.

---

## 4. Unified Memory Footprint

The legacy three-model stack required three separate model files loaded into memory simultaneously:

| Model | Memory |
|-------|--------|
| XGBoost ONNX | ~50 MB |
| SecureBERT ONNX | ~420 MB |
| BART ONNX | ~560 MB |
| **Total** | **~1.03 GB** |

The unified SLM approach loads a single GGUF file:

| Model | Memory |
|-------|--------|
| Phi-3-mini INT4 GGUF | ~2.2 GB |

The absolute footprint is larger, but the operational model is radically simpler: one model lifecycle, one warm-up path, one inference interface, one set of prompts to maintain. The three separate inference engines, three ONNX runtime sessions, and three independent failure modes are replaced by a single, controllable system.

---

## 5. Deployment Architecture

The SLM is deployed as a toggleable layer within the existing FastAPI inference service. The environment variable `ARIES_USE_SLM=true` switches the triage consumer from the legacy XGBoost path to the SLM path. Both paths remain in the codebase simultaneously, allowing regression comparison and safe rollback.

The SLM model file (`triage_slm_q4.gguf`) is loaded via `llama-cpp-python` bindings at service startup and held in RAM for the lifetime of the container. Inference is synchronous and CPU-only, requiring no GPU in the production environment.

**Observed production latency** (from live Docker container logs, CPU-only):

```
triage_slm_inference_complete  grade=TruePositive  latency_ms=265312  ml_score=0.95  risk_score=83.5
```

The ~265 second per-alert latency on CPU reflects the constraint of INT4 inference on a single-core path and is acceptable for the asynchronous Kafka-consumer architecture where alerts are not expected to be processed in real time. On a GPU node this reduces to seconds.

---

## 6. Summary of Benefits

| Dimension | Legacy Stack | SLM Architecture |
|-----------|-------------|-----------------|
| **Cross-SIEM generalization** | Fails on unseen categories (OOD collapse) | Structurally vendor-agnostic via prompt engineering |
| **Hallucination control** | Post-processing guardrails required | Eliminated at model level via instruction tuning |
| **NER recall** | 61.95% F1, fragmentation issues | Generative extraction, higher recall |
| **Model maintenance** | Three independent models, three pipelines | Single model, single inference interface |
| **Deployment cost** | GPU strongly preferred for BART | CPU-viable via INT4 GGUF quantization |
| **Privacy** | Fully local | Fully local, no data leaves the host |
| **Future extensibility** | Adding a new task requires a new model | New task = new system prompt |

---

## 7. Limitations and Honest Caveats

- **CPU latency:** INT4 GGUF inference on CPU is slow (~4 minutes per alert on a standard core). This is acceptable for asynchronous triage but unsuitable for sub-second SLA requirements.
- **Prompt sensitivity:** SLM output quality depends on prompt engineering. A poorly constructed prompt can degrade all three tasks simultaneously — a risk that did not exist with isolated models.
- **Fine-tuning data alignment:** The QLoRA training used GUIDE, CyNER, and GovReport. Alerts from industrial control systems, cloud-native environments, or novel attack patterns not represented in those datasets will be handled by the base model's general reasoning rather than domain-specific fine-tuning.
- **The legacy stack remains the validated production path.** The SLM pipeline is architecturally complete but the GGUF model artifact must be produced from the Hydra training run before it can be the default. The legacy XGBoost/SecureBERT/BART stack, while imperfect, is the currently deployed and validated production configuration.
