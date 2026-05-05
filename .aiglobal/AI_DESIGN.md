# ARIES — AI Subsystem Design Specification

> **Scope:** Full implementation design for all three AI/ML pipelines in the ARIES SOAR platform.
> Aligns with: `REQUIREMENTS.md` (REQ-01..12, REQ-21..24), `DESIGN.md` (§3.4, §6.4),
> `aries-global.mdc`, `fastapi-service.mdc`.
> **Authors:** Sameed Ilyas, Muhammad Taha Salaar, Muhammad Zain | **Version:** 1.0 (Mar 2026)

---

## 0. Quick Reference

| Pipeline | Model | Dataset(s) | Serves | REQ |
|---|---|---|---|---|
| **Alert Triage** | XGBoost → ONNX | GUIDE (13.6 M rows) | `src/triage/` | REQ-02, 03, 04 |
| **NER / IOC Extraction** | SecureBERT-NER fine-tuned → ONNX | CyNER (MITRE) + CASIE | `src/nlp/ner/` | REQ-09, 11, 12 |
| **Incident Summarization** | BART fine-tuned → ONNX | GovReport, evaluated on APTnotes | `src/nlp/summarizer/` | REQ-10 |

All pipelines live inside `apps/fastapi_service/`. Training artifacts are persisted in
MLflow backed by MinIO/S3. Production inference uses ONNX Runtime exclusively.
Feedback is routed via the `ml.feedback` Kafka topic.

---

## 1. Dataset Registry

### 1.1 GUIDE — Alert Triage

| Property | Value |
|---|---|
| **Location** | `datasets/GUIDE/GUIDE_Train.csv` · `GUIDE_Test.csv` |
| **Size** | ~9.5 M train rows · ~4.1 M test rows |
| **Label column** | `IncidentGrade` → `TruePositive` · `FalsePositive` · `BenignPositive` |
| **Key features** | `MitreTechniques`, `AlertTitle`, `Category`, `ThreatFamily`, `SuspicionLevel`, `EntityType`, `DeviceName`, `IpAddress`, `Sha256` |
| **License** | Microsoft GUIDE (research) |
| **Use** | Train XGBoost triage classifier; label maps to `TP=1, FP/BP=0` for binary scoring |

### 1.2 CyNER (MITRE Dataset) — NER Fine-Tuning

| Property | Value |
|---|---|
| **Source** | `github.com/aiforsec/CyNER` (install: `pip install git+https://...`) |
| **Location** | `datasets/CyNER/` after download |
| **Format** | BIO-tagged CoNLL-style splits: `train.txt`, `valid.txt`, `test.txt` |
| **Entities** | `Malware`, `Tool`, `Indicator`, `Vulnerability`, `Organization`, `Person`, `Location` |
| **License** | MIT |
| **Use** | Fine-tune SecureBERT-NER token classifier |

### 1.3 CASIE — Cybersecurity Event Extraction

| Property | Value |
|---|---|
| **Source** | `github.com/Ebiquity/CASIE` |
| **Location** | `datasets/CASIE/` after download |
| **Format** | 1,000 JSON-annotated cybersecurity news articles |
| **Event types** | `Databreach`, `Phishing`, `Ransom`, `Discover-Vulnerability`, `Patch-Vulnerability` |
| **License** | Public (AAAI 2020) |
| **Use** | Augment NER fine-tuning; drive MITRE ATT&CK timeline reconstruction (REQ-12) |

### 1.4 GovReport — Summarization Fine-Tuning

| Property | Value |
|---|---|
| **Source** | `huggingface.co/datasets/ccdv/govreport-summarization` |
| **Location** | `datasets/gov_reports/` |
| **Size** | 17,517 train · 973 val · 973 test |
| **Format** | `{"document": "...", "summary": "..."}` · up to 9,000 token input |
| **License** | Public (arXiv 2104.02112) |
| **Use** | Primary fine-tuning corpus for BART summarization model |

### 1.5 APTnotes — Evaluation Corpus

| Property | Value |
|---|---|
| **Source** | `github.com/kbandla/APTnotes` (400+ PDFs · 2008–2023) |
| **Location** | `datasets/APTnotes/` after PDF extraction |
| **Format** | Raw text extracted from PDFs; pairs built from report body (input) + executive summary section (target) |
| **License** | Public (vendor-published reports) |
| **Use** | Held-out evaluation of summarization model on real-world CTI documents |

### 1.6 Base Backbone — SecureBERT

| Property | Value |
|---|---|
| **HF identifier** | `ehsanaghaei/SecureBERT` (RoBERTa retrained on cybersecurity corpus) |
| **Successor** | `cisco-ai/SecureBERT2.0-base` |
| **License** | openrail-m |
| **Use** | Initialisation weights for NER fine-tuning; outperforms RoBERTa-base/large on security NLP benchmarks |

> **Datasets to delete:** `datasets/CTISum/` is removed — only ~10 pairs, incomplete release.
> **Datasets to keep:** `datasets/GUIDE/`, `datasets/CTI_hf/` (kept for downstream LLM evaluation benchmarks only; never used for training).

---

## 2. Repository Layout for AI Code

```
apps/fastapi_service/
├── src/
│   ├── triage/                        # Pipeline 1 — Alert Triage
│   │   ├── __init__.py
│   │   ├── feature_engineering.py     # extract/encode GUIDE features
│   │   ├── trainer.py                 # XGBoost training + MLflow logging
│   │   ├── onnx_exporter.py           # export XGBoost → ONNX
│   │   ├── inference.py               # ONNX Runtime inference session
│   │   ├── router.py                  # FastAPI router: POST /triage/score
│   │   └── schemas.py                 # Pydantic v2 AlertFeatures, TriageResult
│   │
│   ├── nlp/
│   │   ├── __init__.py
│   │   ├── ner/                       # Pipeline 2 — NER / IOC Extraction
│   │   │   ├── preprocessor.py        # CyNER + CASIE → HuggingFace Dataset
│   │   │   ├── trainer.py             # fine-tune SecureBERT-NER
│   │   │   ├── onnx_exporter.py       # export to ONNX (opset 17)
│   │   │   ├── inference.py           # ONNX NER inference + IOC post-processing
│   │   │   ├── router.py              # FastAPI router: POST /nlp/ner
│   │   │   └── schemas.py             # NERRequest, NERResult, IOCEntity
│   │   │
│   │   └── summarizer/                # Pipeline 3 — Incident Summarization
│   │       ├── preprocessor.py        # GovReport + APTnotes pair builder
│   │       ├── trainer.py             # BART fine-tune via HF Trainer
│   │       ├── onnx_exporter.py       # export encoder + decoder to ONNX
│   │       ├── inference.py           # ONNX generative inference
│   │       ├── router.py              # FastAPI router: POST /nlp/summarize
│   │       └── schemas.py             # SummarizeRequest, SummarizeResult
│   │
│   ├── training/                      # Shared training utilities
│   │   ├── mlflow_client.py           # MLflow run creation, artifact logging
│   │   ├── feedback_consumer.py       # Kafka ml.feedback consumer → retraining trigger
│   │   ├── drift_monitor.py           # PSI-based drift detection
│   │   └── scheduler.py              # APScheduler for scheduled retraining
│   │
│   ├── analytics/                     # KPI calculations (MTTD/MTTR, FP rate)
│   │   └── ...
│   │
│   └── shared/
│       ├── config.py                  # pydantic-settings BaseSettings
│       ├── db.py                      # asyncpg pool
│       ├── kafka.py                   # aiokafka producer/consumer helpers
│       ├── redis_client.py            # Redis async client
│       ├── s3_client.py               # aioboto3 MinIO/S3 client
│       └── exceptions.py             # domain exceptions → HTTPException mappings
│
├── models/                            # ONNX model artifacts (loaded at startup)
│   ├── triage/
│   │   └── xgboost_triage.onnx
│   ├── ner/
│   │   └── secureBERT_ner.onnx
│   └── summarizer/
│       ├── bart_encoder.onnx
│       └── bart_decoder.onnx
│
├── data/                              # Processed/cached dataset splits (gitignored)
├── checkpoints/                       # HF Trainer checkpoints (gitignored)
├── main.py                            # FastAPI app factory; mounts all routers
├── Dockerfile
└── requirements.txt
```

> **Convention:** `models/` holds ONNX `.onnx` files bundled into the Docker image or
> mounted from an S3-backed PVC in Kubernetes. The canonical source of truth for all
> model versions is MLflow (backed by MinIO/S3). The `models/` directory is populated
> by the CI deploy step after a successful model promotion.

---

## 3. Pipeline 1 — Alert Triage (XGBoost on GUIDE)

### 3.1 Overview — What It Does

Takes a normalized alert (from `alerts.raw` Kafka topic) and produces:
- `ml_score` (float 0–1): True-Positive likelihood.
- `incident_grade_label`: `TruePositive | FalsePositive | BenignPositive`.
- `shap_top_features`: top-5 SHAP feature importances for explainability (REQ-02).

### 3.2 Preprocessing Pipeline

**Input:** `datasets/GUIDE/GUIDE_Train.csv`

**Steps in `src/triage/feature_engineering.py`:**

```
1. Load CSV in chunks (chunk_size=100_000) via pandas to avoid OOM on 9.5 M rows.
2. Drop duplicates on (AlertId, TenantId).
3. Binary label encode: TruePositive=1, FalsePositive=0, BenignPositive=0.
4. Categorical encoding:
   - High-cardinality text: AlertTitle, Category, ThreatFamily, EntityType
     → TF-IDF (max_features=500) or OrdinalEncoder with unknown handling.
   - Low-cardinality categoricals: SuspicionLevel (Low/Medium/High/Critical)
     → OrdinalEncoder (in severity order).
5. Hash-type features (Sha256, IpAddress, Domain):
   → FeatureHasher (n_features=1024) — preserves signal without requiring a
     fixed vocabulary, important for novel IOCs.
6. Numerical passthrough: Confidence, LasteventTime (Unix epoch diff).
7. Missing-value imputation: median for numerics, 'unknown' literal for strings.
8. Output: scipy sparse matrix X, numpy array y.
9. Persist processed splits as .npz to data/processed/triage_{train,val,test}.npz.
   Log dataset artifact to MLflow.
```

**Validation split:** Stratified 80/10/10 train/val/test on `IncidentGrade`.

### 3.3 Training

**File:** `src/triage/trainer.py`

```python
# Key training configuration
params = {
    "objective":       "binary:logistic",
    "eval_metric":     ["auc", "logloss"],
    "max_depth":       8,
    "eta":             0.05,
    "subsample":       0.8,
    "colsample_bytree":0.7,
    "n_estimators":    1000,
    "early_stopping_rounds": 50,
    "tree_method":     "hist",   # GPU: "gpu_hist"
    "scale_pos_weight": neg_count / pos_count,  # class imbalance
}
```

**MLflow Tracking (runs inside `src/training/mlflow_client.py`):**
- Experiment name: `aries/triage-classifier`.
- Logged per run: all hyperparams, AUC-ROC, F1 (macro), confusion matrix PNG,
  SHAP summary plot, training time, dataset hash.
- Model artifact logged with `mlflow.xgboost.log_model(model, "xgboost-triage")`.
- Tags: `dataset=GUIDE`, `model_type=xgboost`, `pipeline=triage`.

**SLURM script:** `slurm/train_xgboost.sh` (already scaffolded in repo).

### 3.4 ONNX Export

**File:** `src/triage/onnx_exporter.py`

```python
import onnxmltools
from onnxmltools.convert import convert_xgboost
from onnxmltools.utils import save_model

onnx_model = convert_xgboost(
    xgb_model,
    initial_types=[("float_input", FloatTensorType([None, n_features]))]
)
save_model(onnx_model, "models/triage/xgboost_triage.onnx")
mlflow.log_artifact("models/triage/xgboost_triage.onnx")
```

The exported model is then registered in MLflow Model Registry under the name
`aries-triage-classifier` with stage `Production` after canary validation.

### 3.5 Production Inference — FastAPI Endpoint

**File:** `src/triage/inference.py` + `src/triage/router.py`

```
POST /triage/score
Content-Type: application/json
Authorization: Bearer <JWT>

Body: AlertFeatures (Pydantic v2)
Response: TriageResult { ml_score, incident_grade_label, shap_top_features }
```

**Inference path:**
1. `InferenceSession` loaded once at startup into `app.state.triage_session`
   (ONNX Runtime, CPUExecutionProvider / CUDAExecutionProvider).
2. Request features vectorised using the same `FeatureHasher` / encoders serialised
   alongside the ONNX model in MinIO (loaded at startup from S3).
3. `session.run(None, {"float_input": X_float32})` returns `[labels, probabilities]`.
4. `ml_score = probabilities[:, 1]` (TP class probability).
5. SHAP values computed on-demand using `shap.TreeExplainer` (PyTorch/CPU only,
   cached per alert hash in Redis with TTL=3600s).
6. Result published back via caller (Kafka consumer response pattern) and
   persisted to PostgreSQL `alerts` table (`ml_score`, `incident_grade_label`).

**Heuristic fallback:** If ONNX session fails, a rule-based scorer assigns `ml_score`
based on `SuspicionLevel` mapping (Critical=0.9, High=0.7, Medium=0.4, Low=0.1).

### 3.6 Kafka Integration

**Consumer group:** `ml-triage-engine`

```
alerts.raw  ──►  TriageKafkaConsumer
                  │  normalise + feature engineer
                  │  run ONNX inference
                  │  calculate risk_score = w1*ml_score + w2*asset_criticality + w3*behavioral_anomaly
                  ▼
             alerts.enriched  (enriched alert JSON with ml_score, risk_score)
                  │
                  ├──► Elasticsearch  (index: alerts-{tenant_id})
                  └──► PostgreSQL     (INSERT INTO alerts)
```

**Auto-closure (REQ-04):** Alerts with `ml_score < AUTO_CLOSE_THRESHOLD` (default 0.01)
are status-set to `Closed_FP` immediately and not forwarded to `alerts.enriched`.

### 3.7 Risk Prioritization Formula (REQ-03)

```
risk_score = clamp(
    w_ml    * ml_score           +   # default weight 0.50
    w_asset * asset_criticality  +   # default weight 0.30  (0..1 from CMDB)
    w_behav * behavioral_score,      # default weight 0.20  (anomaly from ES)
    min=0, max=100
) * 100
```

Weights are configurable per-tenant via PostgreSQL `tenant_settings` JSONB column.

---

## 4. Pipeline 2 — NER / IOC Extraction

### 4.1 Overview — What It Does

Takes free-form text (alert body, log snippets, ticket descriptions, CTI reports)
and returns a structured list of tagged entities and IOCs (REQ-09, REQ-11, REQ-12):

```json
{
  "entities": [
    {"text": "192.168.1.100", "label": "Indicator", "start": 4, "end": 17, "ioc_type": "IP_Address"},
    {"text": "Cobalt Strike", "label": "Tool", "start": 32, "end": 45},
    {"text": "CVE-2024-1234",  "label": "Vulnerability", "start": 60, "end": 72}
  ],
  "events": [
    {"type": "Ransom", "trigger": "encrypted files", "arguments": {...}}
  ]
}
```

### 4.2 NER Fine-Tuning on CyNER

#### 4.2.1 Dataset Preparation

**File:** `src/nlp/ner/preprocessor.py`

```
1. Load CyNER CoNLL splits (train.txt, valid.txt, test.txt) via HuggingFace datasets
   `load_dataset("aiforsec/cyner")` or from local CONLL files in datasets/CyNER/.

2. Label scheme (BIO):
   B-Malware, I-Malware
   B-Tool, I-Tool
   B-Indicator, I-Indicator
   B-Vulnerability, I-Vulnerability
   B-Organization, I-Organization
   B-Person, I-Person
   B-Location, I-Location
   O

3. Augmentation from CASIE:
   - CASIE JSONs parsed to extract entity spans for Malware, Tool, Vulnerability.
   - Converted to BIO format and appended to CyNER train split.
   - Deduplication on sentence hash to prevent leakage.

4. Tokenise with SecureBERT tokenizer (RoBERTa BPE):
   - align_labels_with_tokens() using "first" sub-word strategy.
   - max_length=512, truncation=True, padding="max_length".

5. Persist HuggingFace Dataset to data/processed/ner_{train,val,test}.arrow.
```

#### 4.2.2 Fine-Tuning Strategy

**File:** `src/nlp/ner/trainer.py`

- **Base model:** `ehsanaghaei/SecureBERT` (RoBERTa architecture)
- **Head:** `RobertaForTokenClassification` with `num_labels=len(label_list)`
- **Training via HuggingFace `Trainer`:**

```python
training_args = TrainingArguments(
    output_dir          = "checkpoints/ner",
    num_train_epochs    = 5,
    per_device_train_batch_size = 16,
    per_device_eval_batch_size  = 16,
    learning_rate       = 2e-5,
    weight_decay        = 0.01,
    warmup_ratio        = 0.1,
    evaluation_strategy = "epoch",
    save_strategy       = "epoch",
    load_best_model_at_end = True,
    metric_for_best_model  = "f1",
    fp16                = True,           # GPU only
    report_to           = "mlflow",
)
```

- **Compute metrics:** `seqeval` library — entity-level F1, precision, recall per class.
- **MLflow:** Experiment `aries/ner-finetune`. Log: all training args, per-epoch F1,
  confusion matrix, model artifact `secureBERT-ner-finetuned`.

**SLURM script:** `slurm/train_ner.sh` (to be created).

#### 4.2.3 ONNX Export

**File:** `src/nlp/ner/onnx_exporter.py`

```python
torch.onnx.export(
    model,
    (input_ids, attention_mask),
    "models/ner/secureBERT_ner.onnx",
    input_names  = ["input_ids", "attention_mask"],
    output_names = ["logits"],
    dynamic_axes = {
        "input_ids":      {0: "batch", 1: "seq"},
        "attention_mask": {0: "batch", 1: "seq"},
        "logits":         {0: "batch", 1: "seq"},
    },
    opset_version = 17,
)
# Optimise with onnxruntime tools
from onnxruntime.transformers import optimizer
opt = optimizer.optimize_model("models/ner/secureBERT_ner.onnx", model_type="bert")
opt.save_model_to_file("models/ner/secureBERT_ner_opt.onnx")
```

#### 4.2.4 Production Inference

**Inference pipeline (ONNX):**

```
Input text
  │
  ├─ Tokenise (RoBERTa tokenizer, loaded from saved tokenizer in S3)
  ├─ ONNX session.run → logits [batch, seq, num_labels]
  ├─ argmax → label_ids
  ├─ Align label_ids back to word boundaries (sub-word collapse)
  ├─ IOC post-processor:
  │    - regex validation: IP_Address (IPv4/v6), Domain, URL, File_Hash (SHA256/MD5),
  │      Email_Address, CVE-ID
  │    - label → ioc_type mapping
  │    - threat-intel stub lookup (Redis cache → external feed in Phase 5)
  └─ Return NERResult

```

**Redis caching:** `ner:{sha256_of_input}` → NERResult JSON, TTL=1800s.
Prevents re-running the ONNX session for repeated identical alert texts.

### 4.3 CASIE Event Extraction

**Approach:** A second ONNX model fine-tuned solely on CASIE for event detection.
Or (FYP-scoped simpler approach): rule-based event classifier on top of NER output —
map Malware/Tool entities + action verbs to `EventType` using regex trigger phrases.

The classification uses a simple keyword→event mapping table:
```
"encrypted"   → Ransom
"phishing"    → Phishing
"exfiltrated" → Databreach
"patched"     → Patch-Vulnerability
"disclosed"   → Discover-Vulnerability
```

This bridges CASIE events to MITRE ATT&CK tactic mapping for timeline reconstruction (REQ-12).

### 4.4 FastAPI Endpoint

```
POST /nlp/ner
Body: NERRequest { text: str, tenant_id: str }
Response: NERResult { entities: List[IOCEntity], events: List[Event], processing_ms: int }

POST /nlp/ner/batch
Body: NERBatchRequest { texts: List[str], tenant_id: str }  (max 32 items)
Response: List[NERResult]
```

### 4.5 Kafka Integration

**Consumer group:** `nlp-ioc-extractor`

```
alerts.enriched ──► NERKafkaConsumer
                     │  extract entities from alert.raw_data.event_details
                     │  validate IOCs
                     ▼
                PostgreSQL  ioc table (INSERT per extracted IOC)
                     │
                     └──► alerts.enriched  (updated message with ioc_ids attached)
```

Extracted IOCs → `PostgreSQL.ioc` table with columns:
`(ioc_id, alert_id, tenant_id, type, value, source="NER", reputation_score, created_at)`

---

## 5. Pipeline 3 — Incident Summarization

### 5.1 Overview — What It Does

Takes an assembled incident context (case title + associated alert bodies + IOC list,
concatenated as structured prompt) and produces two distinct summaries (REQ-10):

- **Executive summary:** 2–4 sentences. Non-technical. Who was affected, what category of
  attack, recommended management action.
- **Analyst summary:** 8–15 sentences. Technical detail. Attack vector, tools used,
  affected assets, IOCs (linked), MITRE ATT&CK techniques, recommended containment steps.

### 5.2 GovReport Preprocessing

**File:** `src/nlp/summarizer/preprocessor.py`

```
1. Load: datasets.load_dataset("ccdv/govreport-summarization", split="train")
2. Filter out documents longer than MODEL_MAX_INPUT_TOKENS=1024 (BART input limit).
   This retains ~70% of GovReport; the rest are truncated with suffix "[TRUNCATED]".
3. Prepend a domain-adaptation prefix to input:
   "Summarize the following security incident report: {document}"
4. No label prefix needed for BART (seq2seq; encoder receives document, decoder generates summary).
5. Tokenise with BART tokenizer:
   - Input:  max_length=1024, truncation=True, padding=True
   - Target: max_length=256, truncation=True
6. Persist to data/processed/summarizer_{train,val,test}.arrow.
```

**APTnotes pair construction:**
```
1. PDFs in datasets/APTnotes/ extracted to plain text using pdfminer.six.
2. Executive summary section detected via regex heuristics
   (section headings: "Executive Summary", "Overview", "Key Findings").
3. Body text (remaining document) = input; executive section = target.
4. Hold out as evaluation-only set — never used in training.
```

### 5.3 BART Fine-Tuning

**File:** `src/nlp/summarizer/trainer.py`

- **Base model:** `facebook/bart-large-cnn` (pre-trained on CNN/DailyMail summarization).
  Starting from a summarization checkpoint reduces GovReport epochs needed.
- **Training via HuggingFace `Seq2SeqTrainer`:**

```python
training_args = Seq2SeqTrainingArguments(
    output_dir              = "checkpoints/bart",
    num_train_epochs        = 3,
    per_device_train_batch_size = 4,      # BART-large is memory intensive
    per_device_eval_batch_size  = 4,
    gradient_accumulation_steps = 8,      # effective batch = 32
    learning_rate           = 3e-5,
    warmup_steps            = 500,
    predict_with_generate   = True,
    generation_max_length   = 256,
    generation_num_beams    = 4,
    fp16                    = True,
    evaluation_strategy     = "epoch",
    save_strategy           = "epoch",
    load_best_model_at_end  = True,
    metric_for_best_model   = "rougeL",
    report_to               = "mlflow",
)
```

**Compute metrics:** `rouge-score` library — ROUGE-1, ROUGE-2, ROUGE-L on validation.
**APTnotes evaluation:** After training, run inference on APTnotes pairs and log
  ROUGE-L separately as `eval/aptNotes_rougeL` to MLflow.

**MLflow:** Experiment `aries/summarizer-finetune`. Logs: hyperparams, per-epoch ROUGE,
  APTnotes ROUGE-L, model artifact `bart-incident-summarizer`.

**SLURM script:** `slurm/train_bart.sh` (already scaffolded in repo).

### 5.4 ONNX Export

BART is an encoder-decoder architecture; both halves are exported and used in a
generate loop at inference time.

**File:** `src/nlp/summarizer/onnx_exporter.py`

```python
# Export encoder
torch.onnx.export(bart_encoder, (input_ids, attention_mask),
    "models/summarizer/bart_encoder.onnx",
    input_names=["input_ids","attention_mask"],
    output_names=["last_hidden_state"],
    dynamic_axes={"input_ids":{0:"batch",1:"seq"},...},
    opset_version=17)

# Export decoder (with past key-values for KV-cache efficiency)
torch.onnx.export(bart_decoder_with_past, (decoder_input_ids, encoder_hidden_states, ...),
    "models/summarizer/bart_decoder.onnx",
    opset_version=17, ...)
```

> **FYP Scoping note:** For initial FYP delivery, ONNX export for BART generation is
> complex. An acceptable alternative is to serve the fine-tuned PyTorch checkpoint
> directly (via `model.generate()`) behind the FastAPI endpoint on a GPU node, and
> defer ONNX conversion to a post-FYP optimisation. Flag with environment variable
> `SUMMARIZER_BACKEND=pytorch|onnx`.

### 5.5 Production Inference

**Request modes:**

| Mode | Trigger | Response time target |
|---|---|---|
| On-demand (REST) | Analyst clicks "Generate Summary" in dashboard | < 10 s |
| Background (Kafka) | Case reaches `Investigating` status automatically | Async, < 60 s |

**On-demand path:**
```
POST /nlp/summarize
Body: SummarizeRequest { case_id, text, mode: "executive"|"analyst", tenant_id }
  │
  ├─ Check Redis cache: summarizer:{sha256(text)}:{mode} → hit → return immediately
  ├─ ONNX / PyTorch generate()
  │     executive: num_beams=4, max_new_tokens=100, min_length=50
  │     analyst:   num_beams=4, max_new_tokens=400, min_length=150
  ├─ Store result in Redis (TTL=7200s)
  ├─ Persist to PostgreSQL cases.summary
  └─ Return SummarizeResult { executive: str, analyst: str, model_version: str }
```

**Background Kafka path:**
```
cases.updated (status=Investigating)
  └──► SummarizationKafkaConsumer
         │  assemble case context (title + alert bodies + IOC list)
         │  run generate (analyst mode first, then executive)
         │  persist to PostgreSQL cases.summary + cases.executive_summary
         └──► cases.updated (status=summarized, payload includes summary)
```

### 5.6 Prompt Construction

The input fed to BART is assembled as:

```
CASE: {case.title}
SEVERITY: {case.severity}
ALERTS ({n} alerts):
  - {alert_1.normalized_title} [{alert_1.mitre_technique}]
  - {alert_2.normalized_title} [...]
EXTRACTED IOCs:
  - IP: {ioc_value_1}
  - FILE_HASH: {ioc_value_2}
  - DOMAIN: {ioc_value_3}
RAW DETAIL (truncated to 700 tokens):
  {top_alert.event_details[:700_tokens]}
```

This structured prompt maximizes BART's ability to produce grounded, factual summaries
that cite specific evidence rather than hallucinating generic security text.

---

## 6. Model Storage & Versioning (MLflow + MinIO/S3)

### 6.1 Architecture

```
Training Job (SLURM / K8s Job)
  │
  ├──► MLflow Tracking Server  (apps/ ml-tracking, port 5000)
  │         │  metrics, params, artifacts
  │         ▼
  │     MinIO (S3-compatible)  s3://mlflow/
  │         Buckets:
  │           mlflow/           ← MLflow artifact store
  │           models/           ← promoted ONNX artifacts
  │           datasets/         ← processed .arrow / .npz files
  │
  └──► MLflow Model Registry
            Registered models:
              aries-triage-classifier   (XGBoost ONNX)
              aries-ner-extractor       (SecureBERT ONNX)
              aries-summarizer          (BART ONNX or PyTorch)
```

### 6.2 Promotion Workflow

```
Training → Registered in MLflow (stage: None)
  │
  ├─ Canary validation (compare vs current Production on held-out test set)
  │     triage:    AUC-ROC ≥ 0.92
  │     NER:       entity F1 ≥ 0.80 on CyNER test
  │     summarizer: ROUGE-L ≥ 0.35 on GovReport test
  │
  ├─ Pass → transition to stage: Production
  │         copy ONNX to s3://models/{pipeline}/latest.onnx
  │
  └─ Fail → stage: Archived  (no deployment, alert SRE)
```

**Fast rollback:** MLflow Model Registry supports one-click rollback to any prior
`Production` version. The `models/` directory in the Docker image is a volume mount
pointing to the S3 path; swapping the S3 pointer (via `transition_model_version_stage`)
triggers a rolling restart of the FastAPI pods.

### 6.3 Model Metadata in PostgreSQL

All deployed model versions are tracked in a `model_versions` table:

```sql
CREATE TABLE model_versions (
    id             UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    pipeline       TEXT NOT NULL,                      -- 'triage' | 'ner' | 'summarizer'
    mlflow_run_id  TEXT NOT NULL,
    mlflow_model_version INT NOT NULL,
    onnx_s3_uri    TEXT NOT NULL,
    stage          TEXT NOT NULL,                      -- 'canary' | 'production' | 'archived'
    metrics        JSONB,                              -- AUC, F1, ROUGE-L etc.
    promoted_at    TIMESTAMPTZ,
    promoted_by    TEXT,
    created_at     TIMESTAMPTZ DEFAULT NOW()
);
```

This table drives the dashboard widget showing model version + live performance (REQ-23).

---

## 7. Retraining Loop (REQ-21..24)

### 7.1 Feedback Collection

Every analyst override (closing a TP as FP, or escalating a FP) publishes to the
`ml.feedback` Kafka topic with schema:

```json
{
  "alert_id":       "uuid",
  "tenant_id":      "uuid",
  "original_label": "TruePositive",
  "corrected_label":"FalsePositive",
  "analyst_id":     "uuid",
  "timestamp":      "ISO-8601",
  "feedback_type":  "triage_override | approval_rejection | manual_close"
}
```

**Consumer group:** `ml-feedback-collector` (in `src/training/feedback_consumer.py`)
Feedback rows are stored in PostgreSQL `analyst_feedback` table and used as
additional labeled rows in the next retraining cycle.

### 7.2 Drift Detection

**File:** `src/training/drift_monitor.py`

- **Metric:** Population Stability Index (PSI) computed daily on `ml_score` distribution
  (7-day sliding window vs training baseline).
- **Threshold:** PSI > 0.2 → `DriftEvent` published; triggers an unscheduled retrain.
- **Feature drift:** KL-divergence on `ThreatFamily` distribution.
- Scheduled via APScheduler every 24 hours.

### 7.3 Retraining Trigger Conditions

| Condition | Action |
|---|---|
| `feedback_count >= 500` (new labels since last train) | Schedule retrain run |
| PSI > 0.2 (model drift) | Immediate retrain run |
| Scheduled cron (configurable via tenant_settings) | Weekly / monthly |
| Manual trigger via SOC Manager dashboard button | Immediate retrain run |

### 7.4 Retraining Execution

1. `feedback_consumer.py` triggers MLflow `run_training_pipeline(pipeline="triage")`.
2. New GUIDE+feedback combined dataset assembled (feedback rows override GUIDE labels for matching `alert_id`).
3. Full training pipeline re-runs (steps 3.2 → 3.3 → 3.4).
4. Canary validation against held-out test set.
5. If passed: promote to Production, rolling restart of FastAPI pods.
6. SOC Manager notified via WebSocket push: model version, AUC delta.
7. Feedback rows marked `used_in_training=True` in PostgreSQL.

---

## 8. FastAPI Service Integration Architecture

### 8.1 Service Entry Point

`main.py` wires all routers and starts shared resources at startup:

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load ONNX sessions once per process
    app.state.triage_session   = ort.InferenceSession("models/triage/xgboost_triage.onnx")
    app.state.ner_session      = ort.InferenceSession("models/ner/secureBERT_ner_opt.onnx")
    # BART: load PyTorch or ONNX depending on SUMMARIZER_BACKEND env var
    app.state.summarizer       = load_summarizer(settings.SUMMARIZER_BACKEND)
    # Load tokenizers / encoders from S3
    app.state.ner_tokenizer    = load_tokenizer_from_s3("models/ner/tokenizer/")
    app.state.bart_tokenizer   = load_tokenizer_from_s3("models/summarizer/tokenizer/")
    # Start Kafka consumers as background tasks
    asyncio.create_task(triage_consumer.run())
    asyncio.create_task(ner_consumer.run())
    asyncio.create_task(summarization_consumer.run())
    asyncio.create_task(feedback_consumer.run())
    yield
    # Graceful shutdown: drain consumer, close DB pool
    ...

app = FastAPI(lifespan=lifespan)
app.include_router(triage_router,     prefix="/triage",    tags=["Triage"])
app.include_router(ner_router,        prefix="/nlp",       tags=["NLP"])
app.include_router(summarizer_router, prefix="/nlp",       tags=["NLP"])
app.include_router(analytics_router,  prefix="/analytics", tags=["Analytics"])
```

### 8.2 Tenant Isolation

Every FastAPI endpoint extracts `tenant_id` from the JWT (via `Depends(get_tenant_id)`).
Neither ONNX inference results nor IOC records are persisted or returned without
`tenant_id` in scope. Redis cache keys are namespaced: `{tenant_id}:ner:{hash}`.

### 8.3 Service-to-Service Communication

```
Go Orchestration Engine          fastapi_service
(gRPC client)         ──────►    /triage/score     (sync, low-latency path)
                                 /nlp/ner           (sync, on-demand)
                                 /nlp/summarize     (sync, analyst-triggered)

Go Orchestration Engine          Apache Kafka
(Kafka producer)      ──────►    alerts.raw         (async, high-throughput path)
                                 cases.updated
                                 ml.feedback

fastapi_service       ──────►    alerts.enriched    (after triage scoring)
                      ──────►    PostgreSQL          (alerts, ioc, cases, model_versions)
                      ──────►    Elasticsearch       (alert index: alerts-{tenant_id})
                      ──────►    Redis               (SHAP cache, NER cache, dedup keys)
                      ──────►    MinIO/S3            (model artifact retrieval on startup)
                      ──────►    MLflow              (metrics logging during training)
```

### 8.4 Asynchronous Patterns

- **All FastAPI handlers:** `async def`. No blocking calls in the event loop.
- **DB access:** `asyncpg` connection pool. Pool size: `min=5, max=20` per instance.
- **Kafka:** `aiokafka` `AIOKafkaProducer` / `AIOKafkaConsumer`. Consumers run as
  `asyncio.Task` launched in `lifespan`.
- **ONNX inference:** Wrapped in `asyncio.get_event_loop().run_in_executor(None, ...)` so
  it runs in a thread pool (ONNX Runtime is not async-native) without blocking the loop.
- **SHAP/PyTorch generate:** Same executor pattern.

---

## 9. Deployment — Kubernetes & Production Readiness

### 9.1 Docker Image

`apps/fastapi_service/Dockerfile`:

```dockerfile
FROM python:3.11-slim AS base
# System deps for ONNX / PyTorch
RUN apt-get update && apt-get install -y libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

FROM base AS production
COPY src/ ./src/
COPY main.py .
# ONNX artifacts baked in (or mounted from PVC — see below)
COPY models/ ./models/

ENV PYTHONUNBUFFERED=1
ENV ONNX_LOG_SEVERITY_LEVEL=3
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "2", "--loop", "uvloop", "--http", "httptools"]
```

> For GPU nodes (summarizer BART), use `FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime`
> as the base and install `onnxruntime-gpu` instead of `onnxruntime`.

### 9.2 Kubernetes Manifests (Helm chart: `helm/fastapi-service/`)

**Deployment (CPU — Triage + NER):**
```yaml
resources:
  requests:  { cpu: "1",  memory: "2Gi" }
  limits:    { cpu: "4",  memory: "4Gi" }
replicas: 2
strategy:
  type: RollingUpdate
  rollingUpdate: { maxSurge: 1, maxUnavailable: 0 }
```

**Deployment (GPU — Summarizer):**
```yaml
resources:
  requests:  { cpu: "2", memory: "8Gi", nvidia.com/gpu: "1" }
  limits:    { cpu: "4", memory: "16Gi", nvidia.com/gpu: "1" }
replicas: 1   # GPU nodes are expensive; scale up on demand
```

**HPA (CPU pods):**
```yaml
minReplicas: 2
maxReplicas: 10
metrics:
  - type: Resource
    resource:
      name: cpu
      target: { type: Utilization, averageUtilization: 60 }
  - type: Pods
    pods:
      metricName: kafka_consumer_lag
      target: { type: AverageValue, averageValue: "500" }
```

**Model PVC (`PersistentVolumeClaim`):**
For heavier models (BART), ONNX files are stored in S3 and mounted via a
`ReadOnlyMany` PVC backed by a CSI S3 driver, avoiding baking large files into the
image. Triage and NER ONNX files (< 100 MB each) are baked into the Docker image.

**Environment variables (from Kubernetes Secret / ConfigMap):**
```
DATABASE_URL           asyncpg connection string (Vault-injected)
KAFKA_BOOTSTRAP        kafka:9092
REDIS_URL              redis://redis:6379/0
MLFLOW_TRACKING_URI    http://mlflow:5000
S3_ENDPOINT_URL        http://minio:9000
S3_BUCKET_MODELS       models
AWS_ACCESS_KEY_ID      (Vault-injected)
AWS_SECRET_ACCESS_KEY  (Vault-injected)
SUMMARIZER_BACKEND     pytorch     # or onnx
AUTO_CLOSE_THRESHOLD   0.01
TRIAGE_WEIGHTS         {"w_ml":0.5,"w_asset":0.3,"w_behav":0.2}
```

### 9.3 Readiness & Liveness Probes

```
GET /health        → 200 { status: "ok", models_loaded: {...} }
GET /ready         → 200 only when ONNX sessions loaded AND Kafka consumer connected
                   → 503 during cold-start model loading
```

The `/ready` probe prevents Kubernetes from routing traffic to a pod that hasn't yet
pulled and loaded its ONNX model from S3.

### 9.4 Model Hot-Reload (Zero-Downtime Promotion)

When a new model version is promoted in MLflow:

1. MLflow webhook (or CI step) updates Kubernetes ConfigMap `model-version-config`
   with new S3 URI.
2. A `ConfigMap`-watch sidecar in the pod detects the change.
3. Background coroutine in `main.py` (`model_reload_watcher`) picks up the new path,
   loads the new ONNX session into a staging slot, validates with 100 shadow requests,
   then atomically swaps `app.state.triage_session` reference.
4. Old session is dereferenced and GC'd.
5. No pod restart required. Live traffic experiences zero downtime.

---

## 10. Monitoring & Observability

### 10.1 Metrics (Prometheus)

All metrics exposed at `GET /metrics` via `prometheus_fastapi_instrumentator`:

| Metric | Type | Labels |
|---|---|---|
| `aries_triage_requests_total` | Counter | `tenant_id`, `grade` |
| `aries_triage_latency_seconds` | Histogram | `pipeline` |
| `aries_ml_score_distribution` | Histogram | `tenant_id`, `grade` |
| `aries_ner_entities_total` | Counter | `label`, `tenant_id` |
| `aries_summarizer_tokens_input` | Histogram | `mode` |
| `aries_model_version` | Gauge | `pipeline`, `version` |
| `aries_kafka_consumer_lag` | Gauge | `consumer_group`, `topic` |
| `aries_drift_psi` | Gauge | `pipeline`, `tenant_id` |

### 10.2 Logging

All logs emit structured JSON (via `structlog`):

```json
{
  "timestamp": "2026-03-01T12:00:00Z",
  "level": "info",
  "service": "fastapi-service",
  "pipeline": "triage",
  "tenant_id": "uuid",
  "alert_id": "uuid",
  "ml_score": 0.94,
  "latency_ms": 18,
  "model_version": "3"
}
```

Log level controlled via `LOG_LEVEL` env var. Ingested by Loki / CloudWatch.

### 10.3 Distributed Tracing

OpenTelemetry SDK (`opentelemetry-sdk`, `opentelemetry-instrumentation-fastapi`)
auto-instruments all FastAPI requests. Traces propagated to Jaeger / Tempo.
`correlation_id` header passed through from API Gateway → all services.

### 10.4 Alerting Rules (Grafana/Prometheus)

| Rule | Threshold | Severity |
|---|---|---|
| Triage P95 latency > 2 s | sustained 5 min | critical |
| NER inference error rate > 1% | sustained 10 min | warning |
| Kafka consumer lag > 5,000 | sustained 10 min | warning |
| PSI drift alarm | any trigger | info |
| Model AUC < 0.85 | post-retrain | critical |

---

## 11. Integration Flow — End to End

```
External Alert (SIEM/EDR)
  │
  ▼
[Go Alert Ingestion Service]
  Normalise → Kafka: alerts.raw
  │
  ▼
[FastAPI Triage Engine] ← (Consumer: ml-triage-engine)
  XGBoost ONNX → ml_score, risk_score
  Auto-close if ml_score < 0.01
  → Kafka: alerts.enriched
  → PostgreSQL: alerts
  → Elasticsearch: alerts-{tenant_id}
  │
  ▼
[FastAPI NER Consumer] ← (Consumer: nlp-ioc-extractor)
  SecureBERT-NER ONNX → entities, IOCs
  → PostgreSQL: ioc
  → alerts.enriched (updated with ioc_ids)
  │
  ├──────────────────────────────────────────────┐
  ▼                                              ▼
[Go Orchestration Engine]             [Next.js Dashboard]
  Select Playbook                        Analyst views risk-sorted
  Execute Actions                        Triage Queue (from ES)
  Create Case in PostgreSQL              Real-time updates (WebSocket)
  → cases.updated                        NLP Summaries on Case Detail
  │
  ▼
[FastAPI Summarization Consumer] ← (trigger: case status=Investigating)
  BART generate → executive + analyst summaries
  → PostgreSQL: cases.summary
  → cases.updated
  │
  ▼
[Dashboard: Case Detail Page]
  Display summaries, highlighted IOCs,
  MITRE ATT&CK timeline, evidence citations

  [Analyst overrides triage label]
  → Kafka: ml.feedback
  │
  ▼
[FastAPI Feedback Consumer]
  Store in analyst_feedback table
  Run drift monitor
  Trigger retraining if threshold exceeded
  MLflow run → ONNX promotion → hot-reload
```

---

## 12. Coding & Quality Conventions

All AI Python code in `apps/fastapi_service/` must follow:

- **Async everywhere:** `async def` for all FastAPI handlers; ONNX/torch in executor.
- **Pydantic v2:** All request/response models are `pydantic.BaseModel` with strict typing.
- **Config:** `pydantic_settings.BaseSettings` loaded from environment; never hardcode URIs.
- **Linting:** `ruff check` + `ruff format`; `mypy --strict` (CI enforced).
- **Testing:** `pytest` + `pytest-asyncio`; mock ONNX sessions in unit tests;
  integration tests use `testcontainers-python` for PostgreSQL + Redis.
- **No secrets in code:** All credentials from environment / Vault-injected K8s Secrets.
- **ONNX only in production:** PyTorch imports gated behind `if settings.TRAINING_MODE`.
- **Tenant isolation:** Every DB query/ES query/Redis key includes `tenant_id`.
- **Explainability:** SHAP values stored with every triage result in PostgreSQL for audit.
- **Model reproducibility:** All training runs log `random_seed`, `dataset_hash`,
  `git_commit_sha` to MLflow.

---

## 13. Recommended Implementation Order (FYP Milestones)

| Milestone | Deliverable |
|---|---|
| **M1** | GUIDE preprocessing + XGBoost training + MLflow logging. Triage ONNX endpoint live. Kafka consumer consuming `alerts.raw`. |
| **M2** | SecureBERT-NER fine-tuning on CyNER. NER ONNX endpoint live. IOC persistence to PostgreSQL. Dashboard IOC highlighting (REQ-11). |
| **M3** | BART fine-tuning on GovReport. Summarization endpoint live (PyTorch backend). Case detail page displays summaries. APTnotes evaluation logged to MLflow. |
| **M4** | Feedback loop (`ml.feedback` consumer), drift monitor, scheduled retrain trigger. Model registry promotion workflow. Dashboard model-version widget. |
| **M5** | Kubernetes Helm charts, HPA, GPU node for summarizer, model hot-reload sidecar, Prometheus metrics, Grafana dashboards. |

---

*End of AI Design Specification*
