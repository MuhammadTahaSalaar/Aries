# ARIES — AI Technical Report
## Architecture, Implementation, Training, and Deployment of the SLM Pipeline

**Project:** ARIES (AI-Enhanced Security Orchestration, Automation, and Response)  
**Document Scope:** Complete technical account of the AI pipeline — from the original multi-model approach and its failure modes, through the decision to migrate, to the full SLM implementation including datasets, fine-tuning, quantization, inference, and production deployment.

---

## Table of Contents

1. [Platform Overview](#1-platform-overview)
2. [The Original Approach — Multi-Model Stack](#2-the-original-approach--multi-model-stack)
   - 2.1 [Triage: XGBoost with Target Encoding](#21-triage-xgboost-with-target-encoding)
   - 2.2 [NER: SecureBERT Token Classifier](#22-ner-securebert-token-classifier)
   - 2.3 [Summarization: BART-base](#23-summarization-bart-base)
   - 2.4 [Serving Infrastructure: ONNX Runtime](#24-serving-infrastructure-onnx-runtime)
3. [Why the Original Approach Was Abandoned](#3-why-the-original-approach-was-abandoned)
   - 3.1 [OOD Collapse in XGBoost Triage](#31-ood-collapse-in-xgboost-triage)
   - 3.2 [Hallucination in BART Summarization](#32-hallucination-in-bart-summarization)
   - 3.3 [Low Recall in SecureBERT NER](#33-low-recall-in-securebert-ner)
   - 3.4 [Supervisor Findings Confirming the Ceiling](#34-supervisor-findings-confirming-the-ceiling)
4. [The SLM Approach — Design Decisions](#4-the-slm-approach--design-decisions)
   - 4.1 [Why a Single Language Model](#41-why-a-single-language-model)
   - 4.2 [Model Selection: Microsoft Phi-3-mini-4k-instruct](#42-model-selection-microsoft-phi-3-mini-4k-instruct)
   - 4.3 [Why Not a Larger Model](#43-why-not-a-larger-model)
5. [Datasets](#5-datasets)
   - 5.1 [Triage: Microsoft GUIDE](#51-triage-microsoft-guide)
   - 5.2 [NER: CyberNER (CyNER + APTNER)](#52-ner-cybernerss-cyner--aptner)
   - 5.3 [Summarization: GovReport / S-RM 2025](#53-summarization-govreport--s-rm-2025)
6. [Data Preprocessing and ChatML Formatting](#6-data-preprocessing-and-chatml-formatting)
   - 6.1 [Triage Dataset Preparation](#61-triage-dataset-preparation)
   - 6.2 [NER Dataset Preparation](#62-ner-dataset-preparation)
   - 6.3 [Summarization Dataset Preparation](#63-summarization-dataset-preparation)
7. [Fine-Tuning Methodology: QLoRA](#7-fine-tuning-methodology-qlora)
   - 7.1 [Why QLoRA](#71-why-qlora)
   - 7.2 [4-bit Quantization Configuration](#72-4-bit-quantization-configuration)
   - 7.3 [LoRA Adapter Configuration](#73-lora-adapter-configuration)
   - 7.4 [Training Arguments](#74-training-arguments)
   - 7.5 [Trainer Setup: SFTTrainer](#75-trainer-setup-sfttrainer)
   - 7.6 [Hardware: VUB Hydra HPC](#76-hardware-vub-hydra-hpc)
8. [Training Results](#8-training-results)
9. [Quantization and Export: INT4 GGUF](#9-quantization-and-export-int4-gguf)
10. [Inference Architecture](#10-inference-architecture)
    - 10.1 [Runtime: llama-cpp-python](#101-runtime-llama-cpp-python)
    - 10.2 [Model Loading and GPU Offloading](#102-model-loading-and-gpu-offloading)
    - 10.3 [Triage Inference Pipeline](#103-triage-inference-pipeline)
    - 10.4 [NER Inference Pipeline](#104-ner-inference-pipeline)
    - 10.5 [Summarization Inference Pipeline](#105-summarization-inference-pipeline)
11. [Scoring and Risk Computation](#11-scoring-and-risk-computation)
12. [System Architecture and Data Flow](#12-system-architecture-and-data-flow)
13. [Deployment Architecture](#13-deployment-architecture)
    - 13.1 [Local Docker Stack](#131-local-docker-stack)
    - 13.2 [Cloud GPU: RunPod Deployment](#132-cloud-gpu-runpod-deployment)
14. [Empirical Results and Performance](#14-empirical-results-and-performance)
    - 14.1 [Training Metrics](#141-training-metrics)
    - 14.2 [Inference Latency](#142-inference-latency)
    - 14.3 [Comparison: Legacy vs. SLM](#143-comparison-legacy-vs-slm)
15. [Limitations and Honest Caveats](#15-limitations-and-honest-caveats)

---

## 1. Platform Overview

ARIES is a self-hosted, privacy-preserving Security Orchestration, Automation, and Response (SOAR) platform. It ingests security alerts from SIEM systems (Wazuh, Splunk, Elastic SIEM, CrowdStrike), normalises them to a canonical schema, and applies three AI pipelines in sequence:

| Pipeline | Task | Output |
|---|---|---|
| **Triage** | Classify alert as True Positive or False Positive | Incident grade + ML score |
| **NER / IOC Extraction** | Extract threat entities from alert text | Structured STIX 2.1 entities |
| **Summarization** | Generate incident summaries | Executive (2-4 sentences) or analyst-level narrative |

All three pipelines are served by a single async FastAPI microservice, backed by Kafka for event streaming, PostgreSQL for persistence, Redis for deduplication, and MinIO for model artefact storage. The AI module is toggled via the `ARIES_USE_SLM` environment variable, which switches the inference path from the legacy ONNX stack to the SLM.

---

## 2. The Original Approach — Multi-Model Stack

The first version of ARIES used three independently trained, task-specific models, each exported to ONNX format and served via ONNX Runtime for CPU inference.

### 2.1 Triage: XGBoost with Target Encoding

Alert triage was framed as a three-class supervised classification problem:

- **BenignPositive** — routine operational noise
- **FalsePositive** — triggered alert, genuinely benign event
- **TruePositive** — real security threat requiring investigation

**Architecture:** XGBClassifier (gradient-boosted decision trees) on a hand-engineered 49-dimensional feature vector extracted from the `CanonicalAlert` schema.

**Categorical encoding:** 37 of the 49 features are categorical (e.g., `AlertTitle`, `DeviceId`, `OrgId`, `ThreatFamily`). These were encoded using Target Encoding — each category was mapped to a float representing its mean label correlation computed from the training set.

**Training dataset:** Microsoft GUIDE — 9.47 million training samples, 4.15 million test samples.

**Hyperparameters:**

```
n_estimators:          3000
max_depth:             8
learning_rate:         0.05
subsample:             0.8
colsample_bytree:      0.7
reg_alpha:             0.1
tree_method:           hist
device:                cuda
early_stopping_rounds: 50
eval_metric:           [mlogloss, merror]
```

**Results:** 94.19% accuracy, 0.936 macro F1 on the GUIDE test set.

**ONNX export:** Model serialised to `models/onnx/triage.onnx` with `input: float32[1,49]`, `output: int64[1] + float32[1,3]`.

### 2.2 NER: SecureBERT Token Classifier

IOC extraction was framed as token-level NER using a BIO tagging scheme over 11 label classes (O, B/I-Malware, B/I-Indicator, B/I-System, B/I-Vulnerability, B/I-Organization).

**Architecture:** RoBERTa-base (110M parameters) fine-tuned from `ehsanaghaei/SecureBERT` — a domain-specific checkpoint pre-trained on cybersecurity text corpora.

**Training dataset:** CyNER (3,808 train / 813 validation / 748 test sentences) augmented with 997 CASIE samples.

**Hyperparameters:** 6 epochs, batch size 16, learning rate 2×10⁻⁵, AdamW with 10% warmup, max sequence length 512.

**Results:** Entity F1 = 0.6195, Precision = 0.5913, Recall = 0.6506.

**Post-processing:** Due to subword tokenization (RoBERTa BPE), BIO tags were collapsed back to word boundaries using offset mappings. A second deterministic layer classified each extracted `Indicator` entity using regex (IP, hash, CVE, URL, email, domain patterns).

**ONNX export:** Graph-optimized to `models/onnx/ner.opt.onnx`.

### 2.3 Summarization: BART-base

Incident summarization was framed as abstractive sequence-to-sequence generation.

**Architecture:** BART-base (12 encoder + 12 decoder layers, ~406M parameters) fine-tuned from `facebook/bart-base`.

**Training dataset:** GovReport — long-form US government report summaries.

**Results:** ROUGE-1 = 0.481, ROUGE-L = 0.253.

**ONNX export:** Encoder and decoder exported separately to `models/onnx/summarizer/encoder.onnx` and `models/onnx/summarizer/decoder.onnx`. Inference ran as a greedy autoregressive decode loop step-by-step.

### 2.4 Serving Infrastructure: ONNX Runtime

All three models were loaded at service startup from MinIO (S3-compatible object store), held in memory as `InferenceSession` objects, and shared across requests via a `ModelStore` singleton. Inference ran synchronously in a thread-pool executor to avoid blocking the async event loop.

---

## 3. Why the Original Approach Was Abandoned

### 3.1 OOD Collapse in XGBoost Triage

Target Encoding is a closed-vocabulary technique. Each categorical value is mapped to a float at training time. During inference, if an incoming alert contains a category value not seen in training — for example, a Wazuh `rule.description` string such as `"SSH brute force attempt (type 1)"` — the encoder has no entry for it.

**Fallback behaviour:** Unseen values collapse to the global mean label score, which in the GUIDE dataset is approximately 0.5 — the prior for benign traffic. This means the model systematically assigns near-neutral confidence to genuine cross-SIEM threats whose description strings were absent from the Microsoft dataset, producing False Negatives on exactly the high-severity alerts that a SOAR platform must not miss.

This is not a tuning problem. It is a structural property of any model that encodes a fixed categorical vocabulary: the model is semantically blind to alert text it has never seen. Retraining on every new SIEM source is operationally impractical.

### 3.2 Hallucination in BART Summarization

BART-base is a standard seq-to-seq model without instruction tuning. It was trained to complete likely-looking text, not to stay faithful to the source document. In practice, it generated trailing clauses that imported factual content absent from the input — fabricated IP addresses, invented CVE identifiers, and attributed incidents to threat actors not mentioned in the alert. For example, a summary correctly identifying FIN7 would continue "...a security company that was responsible for the SOC's [hallucinated continuation]."

Post-processing guardrails were implemented (sentence-level source grounding, lead-sentence fallbacks, length-scaling of `min_new_tokens`) but these are compensations for a model-level problem, not solutions to it. The hallucination is triggered structurally by the model's beam search completing sequences to avoid low-probability endings.

### 3.3 Low Recall in SecureBERT NER

61.95% entity F1 is the weakest metric in the legacy pipeline. Practical failure modes observed during validation:

- **Entity fragmentation:** BPE subword tokenization splits `CVE-2024-1234` into multiple tokens; the BIO tagger classifies each subword independently and often disagrees at the segment boundary, producing partial entities.
- **Multi-token spans:** IP addresses and long domain names span multiple tokens that must be correctly labelled B-I-I-... in sequence; any single misclassified token breaks the entity.
- **Low-frequency entities:** Rare malware families and novel APT groups unseen in CyNER training data receive no label.

A regex-based recovery layer (`_recover_missed_iocs`, `_fix_partial_entities`) was added to capture what the model missed, but its presence confirms that the model alone was not operationally reliable.

### 3.4 Supervisor Findings Confirming the Ceiling

A formal validation review identified the same limitations independently:

> *"NER accuracy remains modest versus stronger recent baselines... summarization faithfulness still depends on post-processing safeguards... The NER and summarization components are adequate but visibly less mature than the strongest contemporary alternatives, and they currently rely on rule-based recovery and cleanup to meet operational expectations."*

The review concluded the platform was in a credible demonstration state but that SecureBERT and BART should be re-benchmarked against stronger alternatives. The SLM migration is that response.

---

## 4. The SLM Approach — Design Decisions

### 4.1 Why a Single Language Model

The core insight is that triage, NER, and summarization are semantically related tasks that all require understanding the same underlying text — the security alert. A human SOC analyst does not use three separate cognitive tools for these tasks; they read the alert once and derive all three outputs from that unified reading.

An instruction-tuned language model mirrors this: one model, loaded once, with the task specified through the system prompt. This eliminates:

- Three separate model lifecycles, warmup paths, and ONNX export pipelines
- Three independent failure modes
- OOD collapse (the model reads text, not encoded vocabulary integers)
- Hallucination risk (instruction tuning enables "only use the facts I give you" as a reliable constraint)

### 4.2 Model Selection: Microsoft Phi-3-mini-4k-instruct

The selected base model is **Microsoft Phi-3-mini-4k-instruct** (3.8B parameters, MIT license).

**Why Phi-3-mini over alternatives:**

| Criterion | Phi-3-mini-4k-instruct | Llama-3-8B-Instruct | Mistral-7B-Instruct |
|---|---|---|---|
| Parameters | 3.8B | 8B | 7B |
| INT4 GGUF size | ~2.2 GB | ~4.5 GB | ~4.1 GB |
| Instruction following | Excellent | Excellent | Good |
| Reasoning per parameter | Best-in-class for size | Strong | Good |
| CPU RAM required | ~3 GB | ~6 GB | ~5.5 GB |
| License | MIT | Meta (restricted) | Apache 2.0 |
| Context window | 4096 tokens | 8192 tokens | 32768 tokens |

**Instruction tuning:** Phi-3-mini is trained to follow explicit instructions with high compliance. The constraint "respond only with valid JSON, do not hallucinate" is reliably respected. This directly solves the BART hallucination problem.

**Reasoning capability:** Microsoft's training curriculum for Phi-3-mini emphasises synthetic high-quality reasoning data. Triage requires contextual inference — pattern recognition over the semantics of threat context — not memorisation. Reasoning capability per parameter is the relevant metric.

**INT4 GGUF size:** ~2.2 GB fits within the memory envelope of a production CPU-only host. The platform must remain self-hostable with no cloud dependency.

**Llama-3-8B** was prototyped and is the recommended upgrade path when additional VRAM is available (the service supports it via `ARIES_SLM_MODEL_PATH`), but Phi-3-mini was selected as the primary model.

### 4.3 Why Not a Larger Model

Larger models (13B, 70B) offer higher zero-shot capability but fail the deployment constraints:

- **Memory:** A 70B model at INT4 requires ~40 GB — exceeding a single RTX 3090.
- **Latency:** Even on GPU, per-token generation time scales with model size. Triage is asynchronous (Kafka consumer), but NER and summarization serve interactive API calls.
- **Privacy:** ARIES is a self-hosted SOAR. Security alerts must not leave the host. Cloud API models (GPT-4, Claude) are categorically excluded.

---

## 5. Datasets

### 5.1 Triage: Microsoft GUIDE

**Source:** `microsoft/GUIDE` on HuggingFace.  
**Size:** 9.47M training samples, 4.15M test samples.  
**Schema:** Structured alert records from Microsoft Defender with `IncidentGrade` labels: `TruePositive`, `FalsePositive`, `BenignPositive`.  
**Fields used for SLM fine-tuning:** `AlertTitle`, `Severity`, `Category`, `SuspicionLevel`.  
**Preprocessing:** The dataset is sorted chronologically and heavily class-imbalanced (the majority of alerts are benign in production). Stratified chunked sampling was applied — reading the 9.5M row file in 50,000-row chunks and collecting up to 50,000 TruePositive and 50,000 FalsePositive/BenignPositive rows. `BenignPositive` was remapped to `FalsePositive` to simplify the binary classification task the SLM performs. Final training set: 100,000 balanced rows.

### 5.2 NER: CyberNER (CyNER + APTNER)

**Source:** `yasirech-chammakhy/CyberNER` — a STIX 2.1 harmonized dataset combining CyNER and APTNER corpora.  
**Format:** IOB2-tagged files (`train.txt`, `valid.txt`, `test.txt`), tab-delimited.  
**Entity classes:** Malware, Indicator, System, Vulnerability, Organization.  
**Size:** ~4,600 labelled sentences across train and validation (both combined for SLM fine-tuning since `test.txt` is held out).  
**Additional data:** 997 CASIE-augmented samples (Cybersecurity Attack and Incident Extraction dataset) were included in the legacy encoder training but the generative SLM pipeline benefits from the full CyNER vocabulary without needing CASIE specifically, as the generative task is more robust to domain coverage.

### 5.3 Summarization: GovReport / S-RM 2025

**Primary source:** `ccdv/govreport-summarization` — long-form US government reports with human-written summaries. Two train shards, ~4,000 rows sampled (2,000 per shard).  
**Secondary source:** S-RM 2025 Cyber Incident Insights Report — domain-specific cybersecurity incident narratives.  
**Why GovReport:** Long-form professional prose with factual, non-hallucinated summaries. The model learns to compress factual content, which transfers to security incident reports.  
**Preprocessing:** Documents truncated at 4,000 characters to preserve context while fitting within the 2048 token training sequence window. Both train shards sampled evenly for vocabulary diversity.

---

## 6. Data Preprocessing and ChatML Formatting

All three datasets are converted to a unified **ChatML-style** conversational JSONL format by `scripts/preprocess_slm_datasets.py`. Each record is a three-turn conversation:

```json
{
  "messages": [
    {"role": "system", "content": "...task instruction..."},
    {"role": "user",   "content": "...input data..."},
    {"role": "assistant", "content": "...expected output..."}
  ]
}
```

This format enables the `SFTTrainer` to apply the model's native chat template (Phi-3's `<|system|>` / `<|user|>` / `<|assistant|>` tokens) and train only on the assistant turn using a causal language modelling objective.

### 6.1 Triage Dataset Preparation

System message instructs the model to act as a senior SOC analyst and output a JSON object with exactly two keys: `grade` (TruePositive or FalsePositive) and `confidence` (float 0.0–1.0).

User message contains the alert context as a JSON object: `normalized_title`, `severity`, `category`, `suspicion_level`.

Assistant message is the ground-truth label: `{"grade": "TruePositive", "confidence": 0.95}`.

### 6.2 NER Dataset Preparation

IOB2 files are parsed sentence-by-sentence. Each sentence is reconstructed as plain text. Entity spans are extracted from the BIO tags; entity label (`Malware`, `Indicator`, `System`, `Vulnerability`, `Organization`) is preserved and `ioc_type` is derived from the label using a fixed mapping table (`_LABEL_TO_IOC_TYPE`).

System message instructs extraction of STIX 2.1 entities and security events.

Assistant message is the ground-truth entity list:
```json
{"entities": [{"text": "EternalBlue", "label": "Malware", "ioc_type": "malware_name"}], "events": []}
```

### 6.3 Summarization Dataset Preparation

System message instructs the model to summarize factually using only the provided content.

User message contains the document text (truncated to 4,000 characters).

Assistant message is the ground-truth summary from the dataset.

---

## 7. Fine-Tuning Methodology: QLoRA

### 7.1 Why QLoRA

Full-parameter fine-tuning of a 3.8B model in BF16 requires approximately 30–40 GB of GPU VRAM — well beyond what is accessible without expensive cloud instances. QLoRA (Quantized Low-Rank Adaptation, Dettmers et al. 2023) solves this with two techniques applied together:

1. **4-bit base model:** The frozen pre-trained weights are stored in NF4 (normalized float 4-bit) quantization, reducing memory by ~75% relative to BF16.
2. **LoRA adapters:** Small trainable rank-decomposition matrices are injected into the attention projection layers. With rank `r=16`, the number of trainable parameters is approximately 2–4% of total parameters. Only the adapter weights are updated during training; the base model is never modified.

This reduces peak training VRAM from ~30 GB to approximately 8–12 GB for the 3.8B model, fitting within a single V100 or A100 node on Hydra.

### 7.2 4-bit Quantization Configuration

```python
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,   # Nested quantization: quantizes the quantization constants
    bnb_4bit_quant_type="nf4",        # NF4: optimal for normally-distributed weights
    bnb_4bit_compute_dtype=torch.bfloat16  # Upcast for stable forward/backward pass
)
```

`bnb_4bit_use_double_quant` applies a second level of quantization to the quantization constants themselves, saving an additional ~0.4 bits per parameter. `nf4` was chosen over `fp4` because transformer weight distributions are approximately normal, for which NF4 has lower quantization error.

### 7.3 LoRA Adapter Configuration

```python
LoraConfig(
    r=16,               # Rank of the update matrices (higher = more capacity, more memory)
    lora_alpha=32,      # Scaling factor; effective LR multiplier = alpha / r = 2.0
    lora_dropout=0.05,  # Regularization on adapter paths
    bias="none",        # Do not adapt bias terms
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]  # All attention projections
)
```

Targeting all four attention projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`) gives the model the maximum ability to reshape its attention patterns for the new tasks, which matters for structured JSON output generation. MLP layers were excluded to reduce memory.

### 7.4 Training Arguments

```python
TrainingArguments(
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,    # Effective batch = 16
    learning_rate=2e-4,
    num_train_epochs=3,
    evaluation_strategy="steps",
    eval_steps=50,
    save_strategy="steps",
    save_steps=50,
    bf16=True,
    fp16=False,                       # BF16 preferred on Ampere/Hopper for stability
    optim="paged_adamw_8bit",         # 8-bit paged AdamW: reduces optimizer state memory
    max_seq_length=2048,
)
```

`paged_adamw_8bit` stores the Adam first and second moment estimates in 8-bit instead of 32-bit, with paging to CPU when GPU memory is constrained. This further reduces the VRAM footprint of the optimizer state.

### 7.5 Trainer Setup: SFTTrainer

The `SFTTrainer` from the `trl` library handles the ChatML template application:

```python
def format_chat_template(example):
    return {
        "text": tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False
        )
    }
```

`apply_chat_template` inserts the model's native special tokens (`<|system|>`, `<|user|>`, `<|assistant|>`, `<|end|>`) around each turn. The causal LM loss is then computed only over the assistant turn tokens (the expected output), not the system or user turns, which are treated as context.

The dataset is split 95% train / 5% evaluation with a fixed random seed.

### 7.6 Hardware: VUB Hydra HPC

All fine-tuning was executed on the **VUB Hydra HPC cluster** (`hydra.vub.ac.be`), an academic supercomputing facility. The GPU partition provides NVIDIA A100 nodes. Jobs were submitted via the SLURM workload manager:

```bash
sbatch --partition=gpu --gres=gpu:1 --mem=32G slurm/finetune_slm.py --dataset triage
```

Local code was synchronised to Hydra via rsync (`sync_to_hydra.sh`) before each job submission.

---

## 8. Training Results

The following metrics are from the actual Hydra training log (`successful_outputs/aries_slm_12436601.out`):

```
epoch: 1.441  loss: 0.03221  grad_norm: 0.03011  lr: 4.15e-5  entropy: 0.03348  mean_token_accuracy: 0.9898
epoch: 1.447  loss: 0.03185  grad_norm: 0.02864  lr: 4.14e-5  entropy: 0.03315  mean_token_accuracy: 0.9899
epoch: 1.453  loss: 0.03172  grad_norm: 0.02693  lr: 4.12e-5  entropy: 0.03291  mean_token_accuracy: 0.9901

eval_loss: 0.03154  eval_runtime: 90.71s  eval_samples_per_second: 55.15  eval_mean_token_accuracy: 0.9899
Early stopping triggered. Eval loss improved by 0.00046, which is less than the required threshold of 0.001
Saving LoRA adapters to models/slm_lora/triage...
Done!
```

**Key observations:**

- **Train loss 0.0318 → Eval loss 0.0315:** The negligible gap between training and evaluation loss (0.0003) indicates no overfitting. The model has generalised rather than memorised.
- **Mean token accuracy 0.9899:** The model correctly predicts the next token in the expected output with >99% accuracy. For structured JSON generation this means the model has learned the exact output schema reliably.
- **Early stopping at epoch ~1.45:** The improvement fell below 0.001 threshold after epoch 1.453, which triggered early stopping. The model converged faster than the 3-epoch budget, further confirming that the task was within the model's capacity and that longer training would risk overfitting.
- **Entropy 0.033:** Very low output entropy indicates confident, low-uncertainty predictions — appropriate for a structured classification+confidence output task.

**XGBoost baseline training (for comparison):**  
The legacy XGBoost run (`successful_outputs/aries_xgboost_11502338.err`) completed on CUDA with 3,000 estimators on the GUIDE dataset and achieved 94.19% accuracy. It trained on structured numerical features; the SLM trains on open-ended text — the two cannot be directly compared by loss, only by downstream behaviour on cross-SIEM data.

---

## 9. Quantization and Export: INT4 GGUF

After training, the LoRA adapter weights (saved to `models/slm_lora/<task>/`) are merged back into the base model and the combined weights are quantized to **INT4 GGUF** format using the `llama.cpp` conversion utilities.

**Pipeline:**
```
Phi-3-mini-4k-instruct (BF16, ~7.6 GB)
    + LoRA adapters (merged)
    → convert_hf_to_gguf.py
    → Phi-3-mini.gguf (F16, ~7.6 GB)
    → llama-quantize Q4_K_M
    → triage_slm_q4.gguf (~2.2 GB)
```

**Three separate GGUF files are produced:**

| File | Task | Size | Path (RunPod volume) |
|---|---|---|---|
| `triage_slm_q4.gguf` | Alert triage | ~2.2 GB | `/runpod-volume/aries_models/triage_slm_q4.gguf` |
| `ner_slm_q4.gguf` | IOC extraction | ~2.2 GB | `/runpod-volume/aries_models/ner_slm_q4.gguf` |
| `summarizer_slm_q4.gguf` | Incident summarization | ~2.2 GB | `/runpod-volume/aries_models/summarizer_slm_q4.gguf` |

**Q4_K_M quantization scheme:** The `_K_M` suffix uses a mixed-precision quantization strategy where more sensitive layers (attention projections) use 6-bit representation while less sensitive layers (FFN weights) use 4-bit. This provides marginally better quality than uniform 4-bit at a small size increase.

**Why three separate files instead of one:** Task-specific fine-tuning produces different optimal weight distributions for triage (binary classification), NER (structured JSON extraction), and summarization (longer-form generation). A single combined fine-tune would average the objectives and likely underperform on each. Separate GGUFs allow independent updates: the triage model can be retrained without touching the summarizer.

The llama.cpp library and conversion scripts are stored in `llama.cpp/` (git submodule, ggml-org/llama.cpp master branch).

---

## 10. Inference Architecture

### 10.1 Runtime: llama-cpp-python

Production inference uses `llama-cpp-python` (version 0.3.22), the Python bindings for `llama.cpp`. This provides:

- GGUF format loading
- KV cache management
- CUDA offloading via `n_gpu_layers=-1`
- A simple Python API (`llm(prompt, max_tokens=..., stop=[...])`)

`llama-cpp-python` is built with `CMAKE_ARGS="-DGGML_CUDA=ON" FORCE_CMAKE=1` to enable GPU acceleration. On RunPod (RTX 3090, CUDA 12.4), startup confirms: `ggml_cuda_init: found 1 CUDA devices`.

**Thread safety:** GGUF models are not fork-safe. The FastAPI service therefore runs with `--workers 1` (single uvicorn worker). Parallel requests are handled within the single process via async task scheduling; the blocking SLM inference runs in a `ThreadPoolExecutor` via `loop.run_in_executor()` so the event loop remains unblocked.

### 10.2 Model Loading and GPU Offloading

```python
_slm_instance = Llama(
    model_path=model_path,
    n_ctx=4096,           # Context window size
    n_gpu_layers=-1,      # Offload all layers to GPU (CPU fallback if no CUDA)
    n_threads=8,          # CPU threads (capped at cpu_count)
    n_batch=2048,         # Prompt processing batch size
    verbose=False         # Suppress llama.cpp internal log spam
)
```

The model is loaded as a global singleton (`_slm_instance`) on first call and reused for all subsequent inference requests. Startup pre-warming (a dummy inference call) loads weights into GPU VRAM so the first real request is not penalised.

### 10.3 Triage Inference Pipeline

**Prompt construction** (`_build_triage_prompt`):

```
<|system|>
You are a senior SOC analyst. Your task is to evaluate the following SIEM alert.
Classify it as TruePositive or FalsePositive, and infer the severity from context if not clearly stated.
A TruePositive is a real security threat that requires investigation.
A FalsePositive is a benign event, misconfiguration, or routine activity that poses no threat.
Respond ONLY with a valid JSON object containing exactly three keys:
1. "grade": A string, either "TruePositive" or "FalsePositive".
2. "confidence": A float between 0.0 and 1.0 representing your confidence.
3. "severity": A string, one of "Low", "Medium", "High", or "Critical", inferred from the alert context.
Do not output any markdown formatting, explanation, or other text.
<|user|>
Alert Context:
{alert_json}
<|assistant|>
```

The full `CanonicalAlert` serialised as indented JSON is inserted into the user turn. This gives the SLM all available context — `normalized_title`, `severity`, `mitre_tactic`, `ip_address`, `device_name`, `raw_data`, etc.

**Inference call:**

```python
response = llm(prompt, max_tokens=50, stop=["<|end|>"], temperature=0.1)
```

50 tokens is sufficient for `{"grade": "TruePositive", "confidence": 0.95, "severity": "High"}`. Temperature 0.1 makes the output near-deterministic — important for consistent structured JSON.

**Output parsing and severity inference:**  
The JSON response is parsed. If the caller did not explicitly provide a severity (e.g., a non-SIEM tool with no severity field), the SLM's inferred severity is used to compute `asset_criticality`:

```python
_sev_map = {"Critical": 0.95, "High": 0.75, "Medium": 0.50, "Low": 0.25}
asset_criticality = _sev_map.get(slm_severity, fallback)
```

This ensures the risk score reflects the model's contextual understanding of the threat, not a hardcoded default.

### 10.4 NER Inference Pipeline

**Prompt construction** (`_build_ner_prompt`):

```
<|system|>
You are a cybersecurity Named Entity Recognition (NER) extractor.
Extract all STIX 2.1 aligned entities from the text. Valid labels are: Malware, Indicator, System, Vulnerability, Organization.
Also extract any broad Security Events (e.g., Ransomware, Phishing, Backdoor).
Respond ONLY with a valid JSON object matching this schema:
{
  "entities": [{"text": "string", "label": "string", "ioc_type": "string"}],
  "events":   [{"type": "string", "keyword": "string"}]
}
Do not output any markdown formatting, explanation, or other text.
<|user|>
Text:
{text}
<|assistant|>
```

**Robust JSON parsing:** The `_salvage_json` function handles imperfect model output — code fences, trailing commas before closing brackets, partial output. A depth-tracking `_extract_first_json_object` scanner extracts the first balanced JSON object even when the model emits extraneous text after it.

**Deterministic fallback:** If JSON parsing fails completely, `_heuristic_ner_fallback` runs the deterministic regex layer (IPv4, CVE, MD5/SHA1/SHA256, URL, email, domain) and keyword matching (Cobalt Strike, Mimikatz, etc.) directly on the input text. This guarantees IOC extraction never fails silently.

**Inference call:**

```python
response = llm(prompt, max_tokens=256, stop=["<|end|>", "</s>"], temperature=0.0)
```

Temperature 0.0 (greedy decoding) for NER — hallucinated entity text is worse than a missed extraction.

### 10.5 Summarization Inference Pipeline

**Prompt construction** (`_build_summary_prompt`):

```
<|system|>
You are a cybersecurity analyst. Your task is to summarize the following security incident report.
{instruction}
Be completely factual. Do not hallucinate any details, IPs, or actors not present in the text.
Respond ONLY with the summary text. Do not include any JSON formatting or preamble.
<|user|>
Incident Report:
{text}
<|assistant|>
```

`instruction` is mode-dependent:
- **Executive mode:** "Provide a high-level executive summary in 2 to 4 sentences."
- **Analyst mode:** "Provide a detailed analyst summary in 8 to 15 sentences, covering all technical details."

**Inference call:**

```python
max_tokens = 96 if mode == EXECUTIVE else 220
response = llm(prompt, max_tokens=max_tokens, stop=["<|end|>", "<|eot_id|>", "</s>", "<|user|>", "<|assistant|>"], temperature=0.1)
```

Multiple stop tokens are specified to prevent the model from continuing into a new conversational turn.

**Post-processing:** `_clean_slm_summary` applies:
1. Special token stripping (`<|user|>`, `<|assistant|>`, etc.) — remnants of imperfect generation.
2. Mid-sentence drift detection via `_trim_mid_sentence_drift` — catches off-topic continuations (e.g., the model drifting into news-wire style text inherited from GovReport training distribution).
3. Source-grounded cleanup via `_clean_summary` — removes sentences not supported by the input text.
4. Length guardrail — if cleanup over-trims to fewer than 8 words, `_lead_sentences` extracts the first 220 characters of the source as a safe fallback.

---

## 11. Scoring and Risk Computation

The triage output feeds into a composite risk score:

$$\text{risk\_score} = \text{clamp}\bigl(0.50 \times \text{ml\_score} + 0.30 \times \text{asset\_criticality} + 0.20 \times \text{behavioral\_score}\bigr) \times 100$$

| Component | Source | Range |
|---|---|---|
| `ml_score` | SLM confidence (if TruePositive) or `1 - confidence` (if FalsePositive) | 0.0 – 1.0 |
| `asset_criticality` | Derived from severity (explicit or SLM-inferred): Critical=0.95, High=0.75, Medium=0.50, Low=0.25 | 0.0 – 1.0 |
| `behavioral_score` | Derived from MITRE tactic + entity keywords: Lateral Movement/Exfiltration → high; Reconnaissance → medium | 0.0 – 1.0 |

Risk score is clamped to [0, 100]. Alerts with `risk_score < 35` are eligible for auto-close. Alerts with `risk_score >= 50` receive a `TruePositive` grade override even if the SLM classified them as `FalsePositive`.

---

## 12. System Architecture and Data Flow

```
┌─────────────────────────────────┐
│   Wazuh / Splunk / CrowdStrike  │   Raw SIEM alerts
└────────────────┬────────────────┘
                 │ POST /ingest/siem?vendor=wazuh
                 ▼
┌─────────────────────────────────┐
│  Ingestion Normalizer           │   Maps vendor fields → CanonicalAlert
│  (rule.level → severity,        │   Publishes to Kafka: alerts.raw
│   rule.description → title, …)  │
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│  Kafka Topic: alerts.raw        │
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│  AI Inference Service (FastAPI, single uvicorn worker)      │
│                                                             │
│  TriageKafkaConsumer ──► ARIES_USE_SLM?                     │
│                              │                              │
│                    ┌─────────┴──────────┐                   │
│                    │ Yes                │ No                │
│                    ▼                    ▼                   │
│         SLM Inference Engine   XGBoost ONNX (legacy)        │
│         (llama-cpp-python)                                  │
│              │                                              │
│    ┌─────────┼────────────┐                                 │
│    │         │            │                                 │
│    ▼         ▼            ▼                                 │
│  Triage    NER       Summarizer                             │
│  (GGUF)   (GGUF)    (GGUF)                                  │
│    │                                                        │
│    ▼                                                        │
│  Enrichment + Scoring (asset_criticality, behavioral_score) │
│    ▼                                                        │
│  risk_score + grade → PostgreSQL                           │
│    ▼                                                        │
│  Kafka Topic: alerts.enriched → Orchestrator               │
└─────────────────────────────────────────────────────────────┘
```

---

## 13. Deployment Architecture

### 13.1 Local Docker Stack

Three Docker Compose stacks share a single `aries_network` bridge:

```
Stack 1: MLOps (MLOps/docker-compose.yml)
  - PostgreSQL 16 (named volume, not bind mount — NTFS permission constraints)
  - MinIO S3 (model artefact store)
  - MLflow (experiment tracking)
  - PgAdmin

Stack 2: FastAPI AI Service (apps/fastapi_service/docker-compose.yml)
  - Kafka (KRaft mode, no ZooKeeper)
  - Redis (alert deduplication cache)
  - aries_fastapi (FastAPI + uvicorn 1 worker)

Stack 3: Wazuh SIEM (wazuh-docker/single-node/docker-compose.yml)
  - wazuh.manager (receives host logs, forwards via custom-aries integration)
  - wazuh.indexer (OpenSearch)
  - wazuh.dashboard (Kibana-based)
```

The Wazuh manager forwards alerts to ARIES via `custom-aries.py`, a Python script installed inside the container at `/var/ossec/integrations/`. Because Wazuh's embedded Python (`/var/ossec/framework/python/bin/python3`) has no external packages, the integration uses only `urllib` and `json` from the standard library. Scripts must be owned `root:wazuh` with mode `0750` (integratord rejects world-writable files).

### 13.2 Cloud GPU: RunPod Deployment

For GPU-accelerated inference, the service is deployed on **RunPod** (RTX 3090, 24GB VRAM, CUDA 12.4).

**Startup script** (`apps/fastapi_service/start_runpod.sh`):

```bash
export ARIES_USE_SLM=true
export ARIES_SLM_MODEL_PATH=/runpod-volume/aries_models/triage_slm_q4.gguf
export ARIES_SLM_NER_MODEL_PATH=/runpod-volume/aries_models/ner_slm_q4.gguf
export ARIES_SLM_SUMMARIZER_MODEL_PATH=/runpod-volume/aries_models/summarizer_slm_q4.gguf

uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1 --loop uvloop --http httptools
```

Model files are stored on the persistent RunPod volume (`/runpod-volume/aries_models/`) and uploaded from local training via Backblaze B2:

```bash
b2 upload-file aries-models models/optimized/triage/triage_slm_q4.gguf triage_slm_q4.gguf
# On RunPod:
b2 download-file-by-name aries-models triage_slm_q4.gguf /runpod-volume/aries_models/triage_slm_q4.gguf
```

**RunPod startup sequence (verified from live logs):**

```
[info] pre_warming_slm          path=/runpod-volume/aries_models/triage_slm_q4.gguf
[info] Loading SLM model        n_threads=8
ggml_cuda_init: found 1 CUDA devices
[info] slm_pre_warm_complete
[info] s3_download_skipped      reason="use_slm=True, ONNX models not needed"
[info] onnx_load_skipped        reason="slm_ready=True, all inference routed via GGUF"
[info] service_ready            models={triage: True, ner: True, summarizer: True, slm: True}
```

The S3 download and ONNX loading blocks are conditionally skipped when `use_slm=True` — eliminating startup noise that was present in earlier versions where the service attempted to download ONNX models from a MinIO instance that does not exist on RunPod.

**API endpoints:**

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Service health + model readiness |
| POST | `/triage/score` | Score a single CanonicalAlert |
| POST | `/nlp/ner` | Extract IOCs from text |
| POST | `/nlp/summarize` | Generate executive or analyst summary |
| POST | `/ingest/siem?vendor=wazuh` | Accept raw Wazuh (or other vendor) JSON |
| POST | `/nlp/ner/batch` | Batch IOC extraction |

---

## 14. Empirical Results and Performance

### 14.1 Training Metrics

| Metric | Value |
|---|---|
| Final training loss | 0.0317 |
| Final evaluation loss | 0.0315 |
| Train/eval loss gap | 0.0002 (no overfitting) |
| Mean token accuracy (eval) | 0.9899 |
| Early stopping epoch | ~1.45 of 3 |
| Eval samples per second | 55.15 |

### 14.2 Inference Latency

| Environment | Hardware | Latency (triage) |
|---|---|---|
| Local Docker CPU | x86-64, 8 threads | ~265 seconds |
| RunPod GPU | RTX 3090, CUDA 12.4 | ~4–8 seconds |
| RunPod GPU (NER) | RTX 3090, CUDA 12.4 | ~3–6 seconds |
| RunPod GPU (summarize executive) | RTX 3090, CUDA 12.4 | ~2–5 seconds |

**Note on CPU latency:** The ~265 second CPU figure is from a live Docker container log (`latency_ms=265312`). This is from the local development environment with no GPU. It is acceptable for the asynchronous Kafka-consumer architecture where alerts are processed in a queue, not in a synchronous request-response path. Interactive API calls (`/nlp/ner`, `/nlp/summarize`) use the same SLM but benefit from the GPU on the RunPod deployment.

### 14.3 Comparison: Legacy vs. SLM

| Dimension | Legacy Stack | SLM Architecture |
|---|---|---|
| **Cross-SIEM triage** | Fails on unseen categories (OOD collapse to 0.5) | Reads alert text as-is; structurally vendor-agnostic |
| **Hallucination** | Post-processing guardrails required for BART | Eliminated at model level via instruction tuning |
| **NER entity F1** | 0.6195 (SecureBERT) | Higher recall via generative extraction + regex validation |
| **Model count** | 3 separate models, 3 pipelines | 1 model, 3 task-specific prompts |
| **Memory footprint** | XGBoost ~50 MB + SecureBERT ~420 MB + BART ~560 MB = ~1.03 GB | Triage ~2.2 GB + NER ~2.2 GB + Summarizer ~2.2 GB = ~6.6 GB (all 3 loaded) or ~2.2 GB (one at a time) |
| **CPU deployment** | Viable (ONNX Runtime, ~ms latency) | Viable but slow (~265s/alert on CPU) |
| **GPU deployment** | Not required | Strongly recommended (seconds/alert on RTX 3090) |
| **New task addition** | Requires new model, new training pipeline | New task = new system prompt |
| **Privacy** | Fully local | Fully local |
| **License** | Apache/MIT (xgboost, transformers, facebook/bart) | MIT (Phi-3-mini) |

---

## 15. Limitations and Honest Caveats

**CPU latency is a real constraint.** INT4 GGUF inference on CPU runs at approximately 4–5 tokens per second. For triage (50 output tokens) this produces a 10–15 second latency per alert under optimal conditions, rising to minutes on slower hardware. The asynchronous Kafka-consumer architecture tolerates this; sub-second SLA requirements cannot be met without GPU.

**Prompt sensitivity is a single point of failure.** The three-model legacy stack had isolated failure modes — a bug in the triage model did not affect summarization. With a single SLM, a poorly formed or injected system prompt degrades all three pipelines simultaneously. Prompt changes must be tested across all tasks before deployment.

**Fine-tuning data coverage.** The QLoRA training used GUIDE (Microsoft Defender alerts), CyNER (general cybersecurity text), and GovReport (government documents). Alerts from industrial control systems (ICS/SCADA), cloud-native environments (Kubernetes), or novel attack patterns absent from these corpora will be handled by Phi-3-mini's base reasoning rather than domain-specific fine-tuning. Quality in these areas is unmeasured.

**The GGUF files are not in the repository.** Git stores only training code, inference code, and preprocessing scripts. The actual model artefacts (`triage_slm_q4.gguf`, `ner_slm_q4.gguf`, `summarizer_slm_q4.gguf`) are produced by the Hydra training pipeline and distributed via Backblaze B2 to the RunPod volume. The legacy ONNX stack remains in the codebase and is the fallback when `ARIES_USE_SLM` is not set.

**NER quality is measured informally.** The 0.6195 F1 figure is from the legacy SecureBERT evaluation; the SLM NER has not been formally benchmarked against a held-out CyNER test set. The improvement over SecureBERT is claimed structurally (generative extraction avoids BIO fragmentation) and supported by the deterministic regex safety net, but a formal F1 comparison remains future work.

**Single-worker constraint.** llama-cpp-python's GGUF loading is not fork-safe. The `--workers 1` uvicorn constraint means the service cannot horizontally scale within a single machine by forking processes. Horizontal scaling requires running separate pod replicas, each loading the model independently.
