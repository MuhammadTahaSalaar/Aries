# ARIES AI Service — Technical Deep Dive

> **Document Version:** 1.0.0  
> **Last Updated:** March 3, 2026  
> **Component:** FastAPI Inference Service (AI Brain)

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [System Architecture](#2-system-architecture)
3. [ML Pipeline 1: Alert Triage (XGBoost)](#3-ml-pipeline-1-alert-triage-xgboost)
4. [ML Pipeline 2: Named Entity Recognition (SecureBERT)](#4-ml-pipeline-2-named-entity-recognition-securebert)
5. [ML Pipeline 3: Incident Summarization (BART)](#5-ml-pipeline-3-incident-summarization-bart)
6. [SIEM Integration & Normalization](#6-siem-integration--normalization)
7. [Event-Driven Architecture](#7-event-driven-architecture)
8. [Data Storage & Caching](#8-data-storage--caching)
9. [MLOps & Model Management](#9-mlops--model-management)
10. [Infrastructure & Deployment](#10-infrastructure--deployment)
11. [API Reference](#11-api-reference)
12. [Performance Metrics & Results](#12-performance-metrics--results)
13. [Integration with SOAR Platform](#13-integration-with-soar-platform)

---

## 1. Executive Summary

### What is ARIES?

**ARIES** (AI-enhanced Response and Incident Enrichment System) is an intelligent Security Orchestration, Automation, and Response (SOAR) platform that uses machine learning to automate the alert triage process in Security Operations Centers (SOCs).

### The Problem We Solve

Modern SOCs face **alert fatigue** — security analysts receive thousands of alerts daily, with up to 95% being false positives. Manual investigation of each alert is:
- **Time-consuming**: Average 30+ minutes per alert
- **Error-prone**: Fatigue leads to missed true positives
- **Unsustainable**: Alert volumes grow faster than team capacity

### Our Solution

The ARIES AI Service provides three core ML capabilities:

| Pipeline | Model | Purpose | Accuracy/F1 |
|----------|-------|---------|-------------|
| **Alert Triage** | XGBoost | Classify alerts as True/False/Benign Positive | **94.19%** |
| **IOC Extraction** | SecureBERT | Extract IP addresses, hashes, domains, CVEs | **61.95% F1** |
| **Summarization** | BART | Generate executive/analyst incident summaries | **0.48 ROUGE-1** |

### Key Metrics

- **Alerts Processed**: ~9.5 million training samples from Microsoft GUIDE dataset
- **Processing Time**: <50ms per alert (triage), <100ms (NER), <500ms (summarization)
- **False Positive Reduction**: 70%+ of benign alerts auto-closed
- **Analyst Time Savings**: Estimated 60% reduction in triage workload

---

## 2. System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           ARIES AI Service                                   │
│                         (FastAPI + ONNX Runtime)                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐    Webhook     ┌────────────────┐   alerts.raw           │
│  │    SIEM      │ ─────────────► │ /ingest/siem   │ ────────────┐          │
│  │  (Wazuh)     │                │  Normalizer    │              │          │
│  └──────────────┘                └────────────────┘              │          │
│                                                                   ▼          │
│                                                           ┌───────────┐     │
│                                                           │   Kafka   │     │
│                                                           │   Broker  │     │
│                                                           └─────┬─────┘     │
│                                                                 │           │
│  ┌──────────────────────────────────────────────────────────────┘           │
│  │                                                                          │
│  ▼                                                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    Triage Kafka Consumer                             │   │
│  │  ┌─────────────────┐  ┌──────────────────┐  ┌────────────────────┐ │   │
│  │  │ Feature Eng.    │  │ XGBoost ONNX     │  │ Risk Score Calc.   │ │   │
│  │  │ (49 features)   │──│ Inference        │──│ (ML+Asset+Behavior)│ │   │
│  │  └─────────────────┘  └──────────────────┘  └────────────────────┘ │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                      │                                       │
│                    ┌─────────────────┼─────────────────┐                    │
│                    ▼                 ▼                 ▼                    │
│            ┌─────────────┐   ┌─────────────┐   ┌─────────────┐             │
│            │ PostgreSQL  │   │alerts.enriched│   │    Redis    │             │
│            │  (persist)  │   │   (Kafka)   │   │   (cache)   │             │
│            └─────────────┘   └──────┬──────┘   └─────────────┘             │
│                                     │                                       │
│                                     ▼                                       │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │                    Go Orchestrator Engine                          │    │
│  │  (Subscribes to alerts.enriched, executes playbooks)              │    │
│  │                                                                    │    │
│  │  ┌────────────────┐     ┌────────────────────┐                   │    │
│  │  │ POST /nlp/ner  │     │ POST /nlp/summarize │                   │    │
│  │  │ (IOC Extract)  │     │ (Case Summary)      │                   │    │
│  │  └────────────────┘     └────────────────────┘                   │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Component Overview

| Component | Technology | Purpose |
|-----------|------------|---------|
| **API Gateway** | FastAPI 0.115+ | REST endpoints, request validation, async handlers |
| **ML Runtime** | ONNX Runtime 1.20 | CPU-optimized inference for all three models |
| **Message Bus** | Apache Kafka 3.9 | Event streaming, consumer decoupling |
| **Cache Layer** | Redis 7 | Response caching, deduplication |
| **Database** | PostgreSQL 16 | Alert persistence, analyst feedback |
| **Object Storage** | MinIO (S3) | Model artifact storage |
| **Experiment Tracking** | MLflow 2.11 | Model versioning, metrics logging |

### Directory Structure

```
apps/fastapi_service/
├── main.py                          # FastAPI app factory + lifespan
├── Dockerfile                       # Multi-stage production build
├── docker-compose.yml               # Full infrastructure stack
├── requirements.txt                 # 45+ production dependencies
├── src/
│   ├── shared/                      # Shared utilities (10 modules)
│   │   ├── config.py                # Pydantic settings (70+ config vars)
│   │   ├── model_loader.py          # ONNX download + session creation
│   │   ├── mlflow_resolver.py       # MLflow artifact discovery
│   │   ├── s3_client.py             # Async MinIO client
│   │   ├── db.py                    # Async PostgreSQL (asyncpg)
│   │   ├── kafka.py                 # Producer + consumer base classes
│   │   ├── redis_client.py          # Tenant-namespaced caching
│   │   ├── dependencies.py          # FastAPI Depends injection
│   │   ├── exceptions.py            # Domain error hierarchy
│   │   └── logging.py               # Structured JSON logging
│   ├── triage/                      # Alert triage pipeline (5 modules)
│   │   ├── router.py                # POST /triage/score
│   │   ├── schemas.py               # CanonicalAlert, TriageResult
│   │   ├── inference.py             # ONNX inference + risk scoring
│   │   ├── feature_engineering.py   # 49-feature vector extraction
│   │   └── consumer.py              # Kafka alerts.raw consumer
│   ├── nlp/
│   │   ├── ner/                     # IOC extraction pipeline (3 modules)
│   │   │   ├── router.py            # POST /nlp/ner, /nlp/ner/batch
│   │   │   ├── schemas.py           # IOCEntity, NERResult, IOCType
│   │   │   └── inference.py         # SecureBERT ONNX + regex post-proc
│   │   └── summarizer/              # Summarization pipeline (3 modules)
│   │       ├── router.py            # POST /nlp/summarize
│   │       ├── schemas.py           # SummarizeRequest, SummarizeResult
│   │       └── inference.py         # BART encoder-decoder greedy decode
│   └── ingestion/                   # SIEM normalization (3 modules)
│       ├── router.py                # POST /ingest/siem
│       ├── schemas.py               # SIEMRawPayload, IngestionResult
│       └── normalizer.py            # Multi-vendor normalization
└── tests/                           # Pytest test suite (5 test files)
```

---

## 3. ML Pipeline 1: Alert Triage (XGBoost)

### 3.1 Problem Statement

Security alerts need to be classified into three categories:
- **TruePositive**: Genuine security incident requiring investigation
- **FalsePositive**: Alert triggered by benign activity
- **BenignPositive**: Known-good activity that triggered an alert (e.g., IT maintenance)

### 3.2 Dataset: Microsoft GUIDE

The **GUIDE** (Global Universal Intrusion Detection Evaluation) dataset is a large-scale cybersecurity dataset released by Microsoft Research.

| Metric | Value |
|--------|-------|
| **Training Samples** | 9,465,497 |
| **Test Samples** | 4,147,992 |
| **Raw Columns** | 45 |
| **Engineered Features** | 49 |
| **Class Distribution** | BenignPositive: 43.4%, FalsePositive: 21.5%, TruePositive: 35.1% |

**Raw Columns Available:**
```
Id, OrgId, IncidentId, AlertId, Timestamp, DetectorId, AlertTitle, Category,
MitreTechniques, IncidentGrade, ActionGrouped, ActionGranular, EntityType,
EvidenceRole, DeviceId, Sha256, IpAddress, Url, AccountSid, AccountUpn,
AccountObjectId, AccountName, DeviceName, NetworkMessageId, EmailClusterId,
RegistryKey, RegistryValueName, RegistryValueData, ApplicationId,
ApplicationName, OAuthApplicationId, ThreatFamily, FileName, FolderPath,
ResourceIdName, ResourceType, Roles, OSFamily, OSVersion, AntispamDirection,
SuspicionLevel, LastVerdict, CountryCode, State, City
```

### 3.3 Feature Engineering

**49 features** are extracted from each alert:

#### Numeric Features (12)
| # | Feature | Description |
|---|---------|-------------|
| 1 | `hour_of_day` | 0-23, extracted from timestamp |
| 2 | `day_of_week` | 0-6, Monday=0 |
| 3 | `month` | 1-12 |
| 4 | `has_mitre` | 1 if MITRE technique present, 0 otherwise |
| 5 | `mitre_technique_count` | Count of MITRE techniques (semicolon-separated) |
| 6-12 | `has_<field>` | Binary indicators for 7 high-null columns |

#### Categorical Features (37 — Target Encoded)

All 37 categorical columns are encoded using **Target Encoding**:

```python
encoder = ce.TargetEncoder(
    cols=cat_cols,
    handle_unknown="value",      # Unknown categories → global mean
    handle_missing="value",      # Missing values → global mean
    min_samples_leaf=1,
    smoothing=1.0,               # Bayesian shrinkage
)
```

**Target Encoding Formula:**
$$
\text{encoded}_{c} = \frac{n_c \cdot \bar{y}_c + m \cdot \bar{y}_{\text{global}}}{n_c + m}
$$

Where:
- $n_c$ = number of samples with category $c$
- $\bar{y}_c$ = mean target for category $c$
- $\bar{y}_{\text{global}}$ = global mean target
- $m$ = smoothing parameter (1.0)

### 3.4 Model Architecture

**XGBoost Classifier** with GPU acceleration:

```python
model = xgb.XGBClassifier(
    objective="multi:softprob",
    num_class=3,
    eval_metric=["mlogloss", "merror"],
    n_estimators=3000,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    tree_method="hist",
    device="cuda",               # GPU acceleration
    early_stopping_rounds=50,
    random_state=42,
)
```

### 3.5 Training Results

| Metric | Value |
|--------|-------|
| **Accuracy** | 94.19% |
| **F1 (Macro)** | 93.60% |
| **F1 (Weighted)** | 94.15% |
| **Precision (Macro)** | 93.55% |
| **Recall (Macro)** | 93.71% |
| **Training Time** | ~2 hours (A100 GPU) |

**Confusion Matrix:**
```
              Predicted
           BP      FP      TP
Actual BP  1.78M   0.08M   0.05M
       FP  0.06M   0.82M   0.03M
       TP  0.05M   0.04M   1.33M
```

### 3.6 Risk Score Calculation

The final **risk_score** (0-100) combines ML score with contextual factors:

$$
\text{risk\_score} = \text{clamp}\left(w_{ml} \cdot P(\text{TP}) + w_{asset} \cdot \text{asset\_crit} + w_{behavior} \cdot \text{behavioral}\right) \times 100
$$

**Default Weights:**
- $w_{ml} = 0.50$ (ML score)
- $w_{asset} = 0.30$ (Asset criticality)
- $w_{behavior} = 0.20$ (Behavioral baseline deviation)

### 3.7 Auto-Close Threshold

Alerts with `ml_score < 0.01` (1%) are automatically closed as false positives:

```python
AUTO_CLOSE_THRESHOLD = 0.01

result = TriageResult(
    ml_score=ml_score,
    incident_grade=predicted_label,
    risk_score=risk_score,
    auto_closed=(ml_score < settings.auto_close_threshold),
)
```

### 3.8 ONNX Export

```python
from onnxmltools.convert import convert_xgboost
from onnxmltools.convert.common.data_types import FloatTensorType

initial_type = [("float_input", FloatTensorType([None, 49]))]
onnx_model = convert_xgboost(model, initial_types=initial_type, target_opset=17)
```

**Model Size:** 13 MB (ONNX)

---

## 4. ML Pipeline 2: Named Entity Recognition (SecureBERT)

### 4.1 Problem Statement

Security incident reports contain critical **Indicators of Compromise (IOCs)** that need to be extracted automatically:
- IP addresses, domains, URLs
- File hashes (MD5, SHA1, SHA256)
- CVE identifiers
- Malware names, vulnerability names

### 4.2 Dataset: CyNER + CASIE

**CyNER Dataset:**
| Split | Samples |
|-------|---------|
| Train | 3,808 |
| Validation | 813 |
| Test | 748 |
| **CASIE Augmentation** | +997 |

**Label Schema (BIO tagging):**
```
O               - Outside any entity
B-Malware       - Beginning of malware name
I-Malware       - Inside malware name
B-Indicator     - Beginning of IOC
I-Indicator     - Inside IOC
B-System        - Beginning of system/software name
I-System        - Inside system/software name
B-Vulnerability - Beginning of vulnerability
I-Vulnerability - Inside vulnerability
B-Organization  - Beginning of organization name
I-Organization  - Inside organization name
```

### 4.3 Model Architecture

**Base Model:** [SecureBERT](https://huggingface.co/ehsanaghaei/SecureBERT)
- RoBERTa architecture pre-trained on cybersecurity corpora
- 125M parameters
- Vocabulary optimized for security domain

**Fine-tuning Configuration:**
```python
training_args = TrainingArguments(
    num_train_epochs=6,
    per_device_train_batch_size=16,
    learning_rate=2e-5,
    warmup_ratio=0.1,
    weight_decay=0.01,
    fp16=True,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
)
```

### 4.4 Training Results

| Metric | Value |
|--------|-------|
| **F1 Score** | 61.95% |
| **Precision** | 59.13% |
| **Recall** | 65.06% |
| **Training Loss** | 0.2457 |
| **Training Time** | ~45 minutes (A100 GPU) |

### 4.5 Post-Processing: IOC Validation

After model inference, extracted entities are validated using regex patterns:

```python
IOC_PATTERNS = {
    "IP_Address": r"^(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)$",
    "IPv6": r"^(?:[0-9a-fA-F]{1,4}:){2,7}[0-9a-fA-F]{1,4}$",
    "MD5": r"^[a-fA-F0-9]{32}$",
    "SHA1": r"^[a-fA-F0-9]{40}$",
    "SHA256": r"^[a-fA-F0-9]{64}$",
    "Domain": r"^(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)+[a-zA-Z]{2,}$",
    "URL": r"^https?://[^\s]+$",
    "Email": r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$",
    "CVE_ID": r"^CVE-\d{4}-\d{4,}$",
}
```

### 4.6 Security Event Detection

Beyond IOC extraction, the pipeline detects security event types:

```python
EVENT_KEYWORDS = {
    "Ransom": ["encrypted", "ransomware", "ransom", "lockbit", "conti"],
    "Phishing": ["phishing", "phish", "credential harvesting", "spear-phishing"],
    "Databreach": ["exfiltrated", "data breach", "data leak", "exfiltration"],
    "Patch-Vulnerability": ["patched", "patch", "remediated", "fixed"],
    "Discover-Vulnerability": ["disclosed", "vulnerability", "zero-day", "0day", "CVE"],
}
```

### 4.7 ONNX Export & Optimization

```python
# Export to ONNX
torch.onnx.export(
    model, (input_ids, attention_mask),
    "ner.onnx",
    input_names=["input_ids", "attention_mask"],
    output_names=["logits"],
    dynamic_axes={"input_ids": {0: "batch", 1: "seq"}, ...},
    opset_version=17,
)

# ORT Transformer Optimization
optimized_model = optimizer.optimize_model(
    "ner.onnx", model_type="bert", num_heads=12, hidden_size=768
)
optimized_model.save_model_to_file("ner.opt.onnx")
```

**Model Size:** 478 MB → 456 MB (optimized)

---

## 5. ML Pipeline 3: Incident Summarization (BART)

### 5.1 Problem Statement

Security analysts need concise summaries of incidents for:
- **Executive Briefings**: High-level, 2-3 sentence summaries
- **Analyst Handoffs**: Detailed, technical summaries with context

### 5.2 Dataset: GovReport

| Metric | Value |
|--------|-------|
| **Training Samples** | 17,517 |
| **Validation Samples** | 973 |
| **Test Samples** | 973 |
| **Avg Input Length** | ~8,000 tokens |
| **Avg Summary Length** | ~400 tokens |

### 5.3 Model Architecture

**Base Model:** [facebook/bart-base](https://huggingface.co/facebook/bart-base)
- Encoder-decoder transformer
- 140M parameters
- Pre-trained on denoising objective

**Fine-tuning Configuration:**
```python
training_args = Seq2SeqTrainingArguments(
    num_train_epochs=4,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,  # Effective batch = 32
    learning_rate=3e-5,
    max_source_length=1024,
    max_target_length=256,
    predict_with_generate=True,
    generation_num_beams=4,
    gradient_checkpointing=True,
    fp16=True,
)
```

### 5.4 Training Results

| Metric | Value |
|--------|-------|
| **ROUGE-1** | 48.10% |
| **ROUGE-2** | 18.63% |
| **ROUGE-L** | 25.33% |
| **Training Loss** | 19.17 |
| **Training Time** | ~8 hours (2× A100 GPUs) |

### 5.5 Generation Modes

| Mode | Max Tokens | Min Tokens | Use Case |
|------|------------|------------|----------|
| `executive` | 100 | 50 | C-suite briefings |
| `analyst` | 400 | 150 | Technical handoffs |

### 5.6 ONNX Export (Encoder-Decoder)

BART requires **two separate ONNX models**:

```python
# Encoder Export
torch.onnx.export(
    model.get_encoder(),
    (input_ids, attention_mask),
    "encoder.onnx",
    input_names=["input_ids", "attention_mask"],
    output_names=["last_hidden_state"],
)

# Decoder Export (with LM head)
class DecoderWrapper(torch.nn.Module):
    def forward(self, decoder_input_ids, encoder_hidden_states, encoder_attention_mask):
        dec_out = self.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
        )
        logits = self.lm_head(dec_out.last_hidden_state) + self.final_logits_bias
        return logits

torch.onnx.export(decoder_wrapper, ..., "decoder.onnx")
```

### 5.7 Greedy Decode Algorithm

```python
def _greedy_decode_onnx(encoder_session, decoder_session, input_ids, attention_mask, max_tokens):
    # 1. Encode input once
    encoder_out = encoder_session.run(None, {"input_ids": input_ids, "attention_mask": attention_mask})
    encoder_hidden = encoder_out[0]
    
    # 2. Start decoder with BOS token
    decoder_input = np.array([[0]], dtype=np.int64)  # BOS token
    generated_ids = []
    
    # 3. Autoregressive generation
    for step in range(max_tokens):
        logits = decoder_session.run(None, {
            "decoder_input_ids": decoder_input,
            "encoder_hidden_states": encoder_hidden,
            "encoder_attention_mask": attention_mask,
        })[0]
        
        next_token = int(np.argmax(logits[0, -1, :]))
        generated_ids.append(next_token)
        
        if next_token == 2:  # EOS token
            break
            
        decoder_input = np.array([[0] + generated_ids], dtype=np.int64)
    
    return generated_ids
```

**Model Size:** 558 MB (encoder) + 558 MB (decoder)

---

## 6. SIEM Integration & Normalization

### 6.1 Supported Vendors

| Vendor | Integration Type | Alert Format |
|--------|------------------|--------------|
| **Wazuh** | Webhook/REST | JSON |
| **Splunk** | HTTP Event Collector | JSON |
| **Elastic SIEM** | Webhook | JSON |
| **CrowdStrike** | Streaming API | JSON |

### 6.2 Canonical Alert Schema

All vendor-specific alerts are normalized to a **canonical schema**:

```python
class CanonicalAlert(BaseModel):
    alert_id: str
    tenant_id: str
    timestamp: datetime
    source: str                    # "wazuh", "splunk", etc.
    normalized_title: str
    severity: Severity             # Low, Medium, High, Critical
    description: str | None
    mitre_tactic: str | None
    mitre_technique: str | None
    source_ip: str | None
    destination_ip: str | None
    hostname: str | None
    username: str | None
    raw_data: dict                 # Original vendor payload
```

### 6.3 Severity Mapping

**Wazuh Level Mapping:**
```python
WAZUH_SEVERITY = {
    (0, 3): "Low",
    (4, 7): "Medium",
    (8, 11): "High",
    (12, 15): "Critical",
}
```

**CrowdStrike Mapping:**
```python
CROWDSTRIKE_SEVERITY = {
    "informational": "Low",
    "low": "Low",
    "medium": "Medium",
    "high": "High",
    "critical": "Critical",
}
```

---

## 7. Event-Driven Architecture

### 7.1 Kafka Topics

| Topic | Partitions | Purpose |
|-------|------------|---------|
| `alerts.raw` | 3 | Normalized alerts from SIEM ingestion |
| `alerts.enriched` | 3 | ML-scored alerts for orchestrator |
| `cases.updated` | 3 | Case lifecycle events |
| `playbooks.events` | 3 | Playbook execution telemetry |
| `ml.feedback` | 3 | Analyst feedback for retraining |
| `alerts.raw.dlq` | 1 | Dead letter queue |
| `alerts.enriched.dlq` | 1 | Dead letter queue |

### 7.2 Alert Flow

```
[SIEM] → POST /ingest/siem → [Kafka: alerts.raw]
                                     ↓
                          [TriageKafkaConsumer]
                                     ↓
                          ┌─────────────────────┐
                          │ 1. Parse & Validate │
                          │ 2. Extract Features │
                          │ 3. ONNX Inference   │
                          │ 4. Compute Risk     │
                          │ 5. Persist to DB    │
                          │ 6. Auto-close check │
                          └─────────────────────┘
                                     ↓
                    [Kafka: alerts.enriched] + [PostgreSQL]
                                     ↓
                          [Go Orchestrator]
```

### 7.3 Consumer Configuration

```python
class TriageKafkaConsumer(BaseConsumer):
    topics = ["alerts.raw"]
    group_id = "ml-triage-engine"
    auto_offset_reset = "earliest"
    enable_auto_commit = False  # Manual commit for exactly-once
```

---

## 8. Data Storage & Caching

### 8.1 PostgreSQL Schema

**alerts table:**
```sql
CREATE TABLE alerts (
    alert_id VARCHAR(255) PRIMARY KEY,
    tenant_id VARCHAR(100) NOT NULL,
    normalized_title TEXT,
    raw_data JSONB,
    source VARCHAR(50),
    ml_score FLOAT,
    risk_score FLOAT,
    incident_grade VARCHAR(20),
    status VARCHAR(20) DEFAULT 'pending',
    mitre_tactic VARCHAR(100),
    mitre_technique VARCHAR(100),
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_alerts_tenant ON alerts(tenant_id);
CREATE INDEX idx_alerts_status ON alerts(status);
CREATE INDEX idx_alerts_risk ON alerts(risk_score DESC);
```

**iocs table:**
```sql
CREATE TABLE iocs (
    ioc_id SERIAL PRIMARY KEY,
    alert_id VARCHAR(255) REFERENCES alerts(alert_id),
    tenant_id VARCHAR(100) NOT NULL,
    ioc_type VARCHAR(50),
    value TEXT,
    confidence FLOAT,
    source VARCHAR(50),
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(alert_id, ioc_type, value)
);
```

### 8.2 Redis Caching

**Cache Keys:**
```
aries:{tenant_id}:ner:{hash(text)}     # TTL: 30 minutes
aries:{tenant_id}:summary:{hash(text)} # TTL: 2 hours
aries:{tenant_id}:dedup:{alert_id}     # TTL: 5 minutes
```

**Cache Hit Rate:** ~40% for repeated IOC queries

---

## 9. MLOps & Model Management

### 9.1 MLflow Experiments

| Experiment | Run Name | Artifacts |
|------------|----------|-----------|
| `aries/triage-classifier` | `xgboost-triage` | `onnx/triage.onnx`, `triage_model/triage_encoder.pkl` |
| `aries/ner` | `secureBERT-ner` | `onnx/ner.opt.onnx`, tokenizer files |
| `aries/bart-summarizer` | `bart-summarizer` | `onnx/summarizer/encoder.onnx`, `decoder.onnx`, tokenizer |

### 9.2 Model Resolution Flow

```
[Startup] → Query MLflow REST API
              ↓
         Search experiments by name
              ↓
         Get latest FINISHED run
              ↓
         Extract artifact S3 keys
              ↓
         Download from MinIO to /tmp/aries_models/
              ↓
         Create ONNX InferenceSessions
              ↓
         [Ready to serve inference]
```

### 9.3 Migration Script

```bash
python MLOps/migrate_to_mlflow.py
```

**Steps:**
1. Replay existing `mlruns/` directory → Remote MLflow
2. Log NER model + metrics + artifacts
3. Log BART summarizer + ROUGE scores
4. Upload XGBoost model + TargetEncoder + ONNX

---

## 10. Infrastructure & Deployment

### 10.1 Docker Architecture

**Two Compose Stacks:**

| Stack | File | Services |
|-------|------|----------|
| **MLOps** | `MLOps/docker-compose.yml` | PostgreSQL, MinIO, MLflow, pgAdmin |
| **Serving** | `apps/fastapi_service/docker-compose.yml` | Kafka, Redis, FastAPI |

**Shared Network:** `aries_network`

### 10.2 Resource Requirements

| Service | CPU | Memory | Storage |
|---------|-----|--------|---------|
| FastAPI | 2 cores | 4 GB | 2 GB (models) |
| Kafka | 1 core | 1 GB | 1 GB |
| Redis | 0.5 cores | 512 MB | 256 MB |
| PostgreSQL | 1 core | 1 GB | 10 GB |
| MinIO | 1 core | 1 GB | 5 GB |

### 10.3 Dockerfile

```dockerfile
FROM python:3.11-slim AS builder
RUN apt-get install -y libgomp1 curl
COPY requirements.txt .
RUN pip install --prefix=/install -r requirements.txt

FROM python:3.11-slim
COPY --from=builder /install /usr/local
RUN useradd -m aries
USER aries
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
```

---

## 11. API Reference

### Health Endpoints

| Method | Path | Response |
|--------|------|----------|
| GET | `/health` | `{status, version, models_loaded}` |
| GET | `/ready` | `{ready, kafka_connected, db_connected, redis_connected}` |

### Triage Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/triage/score` | Score single alert |
| GET | `/triage/health` | Pipeline status |

### NLP Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/nlp/ner` | Extract IOCs from text |
| POST | `/nlp/ner/batch` | Batch IOC extraction (max 32) |
| POST | `/nlp/summarize` | Generate incident summary |

### Ingestion Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/ingest/siem` | Ingest single SIEM alert |
| POST | `/ingest/siem/batch` | Batch ingest (max 100) |
| GET | `/ingest/siem/vendors` | List supported vendors |

---

## 12. Performance Metrics & Results

### Inference Latency (P95)

| Pipeline | CPU (ms) | GPU (ms) |
|----------|----------|----------|
| Triage | 15 | N/A |
| NER | 85 | 25 |
| Summarizer | 450 | 150 |

### Throughput

| Pipeline | Requests/sec |
|----------|-------------|
| Triage | ~500 |
| NER | ~100 |
| Summarizer | ~20 |

### Model Accuracy Summary

| Model | Metric | Value |
|-------|--------|-------|
| XGBoost Triage | Accuracy | **94.19%** |
| XGBoost Triage | F1 (Macro) | **93.60%** |
| SecureBERT NER | F1 | **61.95%** |
| BART Summarizer | ROUGE-1 | **48.10%** |
| BART Summarizer | ROUGE-L | **25.33%** |

---

## 13. Integration with SOAR Platform

### 13.1 Go Orchestrator Integration

The FastAPI service is a **plug-and-play AI backend** for the Go orchestrator:

```go
// Go client calls REST endpoints
aiClient := ai.NewClient("http://fastapi_service:8000", tenantID)

// Called during playbook execution
iocs, _ := aiClient.ExtractIOCs(alertDescription)
summary, _ := aiClient.Summarize(incidentReport, "analyst")
```

### 13.2 Event Flow

```
[SIEM Alert] → [AI Service] → [alerts.enriched] → [Go Orchestrator]
                                                        ↓
                                                 [Playbook Engine]
                                                        ↓
                                              [POST /nlp/ner] ←───┘
                                              [POST /nlp/summarize] ←───┘
```

### 13.3 Required Headers

Every request **must** include:
- `X-Tenant-ID`: Multi-tenant isolation
- `Content-Type: application/json`

---

## Conclusion

The ARIES AI Service provides production-grade ML inference for SOC automation with:

- **94.19% accuracy** alert triage (XGBoost on 9.5M alerts)
- **61.95% F1** IOC extraction (SecureBERT fine-tuned on CyNER)
- **48.10% ROUGE-1** summarization (BART on GovReport)
- **<100ms latency** for triage decisions
- **Multi-tenant** architecture with row-level security
- **Event-driven** processing via Kafka
- **MLOps** integration with MLflow for model versioning

The service is designed to reduce analyst alert fatigue by **70%+** through intelligent automation while maintaining full auditability and feedback loops for continuous improvement.
