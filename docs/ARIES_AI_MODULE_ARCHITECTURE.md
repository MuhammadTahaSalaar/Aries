# ARIES AI Module — Architecture & API Reference

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Models Available](#2-models-available)
   - [2.1 Triage Classifier (XGBoost)](#21-triage-classifier-xgboost)
   - [2.2 NER / IOC Extractor (SecureBERT)](#22-ner--ioc-extractor-securebert)
   - [2.3 Incident Summarizer (BART)](#23-incident-summarizer-bart)
3. [API Reference](#3-api-reference)
   - [3.1 Health & Readiness](#31-health--readiness)
   - [3.2 Triage Endpoints](#32-triage-endpoints)
   - [3.3 NER Endpoints](#33-ner-endpoints)
   - [3.4 Summarization Endpoints](#34-summarization-endpoints)
   - [3.5 SIEM Ingestion Endpoints](#35-siem-ingestion-endpoints)
4. [Architecture & Data Flow](#4-architecture--data-flow)
   - [4.1 Component Diagram](#41-component-diagram)
   - [4.2 End-to-End Alert Flow](#42-end-to-end-alert-flow)
   - [4.3 Triage Pipeline Flow](#43-triage-pipeline-flow)
   - [4.4 NER Pipeline Flow](#44-ner-pipeline-flow)
   - [4.5 Summarization Pipeline Flow](#45-summarization-pipeline-flow)
5. [Model Serving Infrastructure](#5-model-serving-infrastructure)
   - [5.1 Model Loading at Startup](#51-model-loading-at-startup)
   - [5.2 ONNX Runtime Configuration](#52-onnx-runtime-configuration)
   - [5.3 Caching Strategy](#53-caching-strategy)
6. [Database Schema](#6-database-schema)
7. [Kafka Topics](#7-kafka-topics)
8. [Training Pipelines](#8-training-pipelines)
9. [Configuration Reference](#9-configuration-reference)
10. [Performance Characteristics](#10-performance-characteristics)

---

## 1. System Overview

**ARIES** (AI-Enhanced SOAR) is a security orchestration platform that combines three ML pipelines to automate cybersecurity operations:

| Capability | Model | Purpose |
|---|---|---|
| Alert Triage | XGBoost | Classify alerts as True Positive, False Positive, or Benign Positive |
| IOC Extraction | SecureBERT (NER) | Extract malware names, indicators, systems, vulnerabilities, & organizations from text |
| Incident Summarization | BART | Generate executive or analyst-level incident summaries |

The AI module is served as an **async FastAPI microservice** backed by **ONNX Runtime** for optimized CPU inference. It integrates with **Kafka** for event streaming, **PostgreSQL** for persistence, **Redis** for caching/deduplication, **MinIO (S3)** for model storage, and **MLflow** for experiment tracking.

---

## 2. Models Available

### 2.1 Triage Classifier (XGBoost)

| Property | Value |
|---|---|
| **Architecture** | XGBClassifier (gradient-boosted trees) |
| **Task** | 3-class classification |
| **Classes** | `BenignPositive` (0), `FalsePositive` (1), `TruePositive` (2) |
| **Training Dataset** | Microsoft GUIDE (9.47M train / 4.15M test samples) |
| **ONNX Path** | `models/onnx/triage.onnx` |
| **Input Shape** | `float32 [1, 49]` — a 49-dimensional feature vector |
| **Output** | Predicted label + per-class probabilities (3 classes) |
| **Accuracy** | 94.2% |
| **F1 (macro)** | 0.936 |

#### Input: 49-Feature Vector

The model expects a `float32[1, 49]` array constructed from a `CanonicalAlert`:

| Index | Feature | Type | Description |
|---|---|---|---|
| 0 | `hour_of_day` | int (0–23) | Hour the alert was created |
| 1 | `day_of_week` | int (0–6) | Day of week (Monday=0) |
| 2 | `month` | int (1–12) | Month |
| 3 | `has_mitre` | binary (0/1) | Whether MITRE ATT&CK mapping exists |
| 4 | `mitre_technique_count` | int | Count of MITRE techniques |
| 5 | `has_EmailClusterId` | binary | Null-flag for EmailClusterId field |
| 6 | `has_ThreatFamily` | binary | Null-flag for ThreatFamily field |
| 7 | `has_ResourceType` | binary | Null-flag for ResourceType field |
| 8 | `has_Roles` | binary | Null-flag for Roles field |
| 9 | `has_AntispamDirection` | binary | Null-flag for AntispamDirection field |
| 10 | `has_SuspicionLevel` | binary | Null-flag for SuspicionLevel field |
| 11 | `has_LastVerdict` | binary | Null-flag for LastVerdict field |
| 12–48 | 37 categorical fields | float (target-encoded) | OrgId, DetectorId, AlertTitle, Category, EntityType, EvidenceRole, DeviceId, Sha256, IpAddress, Url, AccountSid, AccountUpn, AccountObjectId, AccountName, DeviceName, NetworkMessageId, EmailClusterId, RegistryKey, RegistryValueName, RegistryValueData, ApplicationId, ApplicationName, OAuthApplicationId, ThreatFamily, FileName, FolderPath, ResourceIdName, ResourceType, Roles, OSFamily, OSVersion, AntispamDirection, SuspicionLevel, LastVerdict, CountryCode, State, City |

**Categorical encoding**: A `TargetEncoder` (fitted during training, serialized to `models/triage/triage_encoder.pkl`) maps each categorical value to a float in [0, 1] representing its mean target correlation. At inference time, unseen values fall back to a deterministic MD5 hash → float in [0, 1).

#### Output

| Field | Type | Description |
|---|---|---|
| `labels` | `int64[1]` | Predicted class (0, 1, or 2) |
| `probabilities` | `float32[1, 3]` | Softmax over (BenignPositive, FalsePositive, TruePositive) |

The `TruePositive` probability (index 2) is used as `ml_score`. A composite `risk_score` is then derived:

```
risk_score = clamp(0.50 × ml_score + 0.30 × asset_criticality + 0.20 × behavioral_score) × 100
```

Alerts with `ml_score < 0.01` are auto-closed as false positives.

#### Training Hyperparameters

```
n_estimators:         3000
max_depth:            8
learning_rate:        0.05
subsample:            0.8
colsample_bytree:     0.7
reg_alpha:            0.1
tree_method:          hist
device:               cuda
early_stopping_rounds: 50
eval_metric:          [mlogloss, merror]
```

---

### 2.2 NER / IOC Extractor (SecureBERT)

| Property | Value |
|---|---|
| **Architecture** | RoBERTa-base (110M params), fine-tuned for token classification |
| **Base Model** | `ehsanaghaei/SecureBERT` |
| **Task** | Token-level multi-label NER (BIO tagging) |
| **Labels** | 11 tags: O + B/I for Malware, Indicator, System, Vulnerability, Organization |
| **Training Dataset** | CyNER (3,808 train / 813 val / 748 test) + 997 CASIE-augmented samples |
| **ONNX Path** | `models/onnx/ner.opt.onnx` (graph-optimized) |
| **Input** | `input_ids` int64[1, ≤512], `attention_mask` int64[1, ≤512] |
| **Output** | `logits` float32[1, seq_len, 11] |
| **Entity F1** | 0.6195 |
| **Precision** | 0.5913 |
| **Recall** | 0.6506 |

#### Input: Text String

Raw security text (1–50,000 characters). The service tokenizes it using the SecureBERT fast tokenizer with a max sequence length of 512 tokens.

#### Output: Entity Labels (11 classes, BIO scheme)

| ID | Label | Description |
|---|---|---|
| 0 | `O` | No entity |
| 1 | `B-Malware` | Beginning of malware/tool name |
| 2 | `I-Malware` | Inside malware/tool name |
| 3 | `B-Indicator` | Beginning of IOC/threat indicator |
| 4 | `I-Indicator` | Inside IOC/threat indicator |
| 5 | `B-System` | Beginning of OS/software/system |
| 6 | `I-System` | Inside OS/software/system |
| 7 | `B-Vulnerability` | Beginning of CVE/exploit |
| 8 | `I-Vulnerability` | Inside CVE/exploit |
| 9 | `B-Organization` | Beginning of APT group/organization |
| 10 | `I-Organization` | Inside APT group/organization |

#### Post-Processing

1. **BIO Tag Collapsing**: Sub-word tokens are aligned back to word boundaries using offset mappings; B/I tags are merged into contiguous entity spans.
2. **IOC Classification via Regex**: Each extracted `Indicator` entity is validated and typed:

   | IOC Type | Pattern |
   |---|---|
   | `IP_Address` | IPv4/IPv6 regex |
   | `File_Hash` | MD5 (32 hex), SHA1 (40 hex), SHA256 (64 hex) |
   | `CVE_ID` | `CVE-\d{4}-\d{4,}` |
   | `URL` | URL pattern |
   | `Email_Address` | Email pattern |
   | `Domain` | Domain pattern |
   | `Unknown` | No regex match |

3. **Confidence Score**: Mean softmax confidence across all tokens in the entity span.
4. **Event Detection**: Keyword-based rules infer security event categories (ransomware, phishing, etc.).

#### Training Hyperparameters

```
epochs:          6
batch_size:      16
learning_rate:   2e-5
max_length:      512
warmup_ratio:    0.1
weight_decay:    0.01
optimizer:       AdamW
loss:            CrossEntropyLoss (token-level)
```

---

### 2.3 Incident Summarizer (BART)

| Property | Value |
|---|---|
| **Architecture** | BART-base (12 encoder + 12 decoder layers, ~406M params) |
| **Base Model** | `facebook/bart-base` |
| **Task** | Abstractive seq-to-seq summarization |
| **Training Dataset** | GovReport (long-form government reports) |
| **ONNX Paths** | `models/onnx/summarizer/encoder.onnx`, `models/onnx/summarizer/decoder.onnx` |
| **Encoder Input** | `input_ids` int64[1, ≤1024], `attention_mask` int64[1, ≤1024] |
| **Encoder Output** | `last_hidden_state` float32[1, seq, 768] |
| **Decoder Input** | `decoder_input_ids`, `encoder_hidden_states`, `attention_mask` |
| **Decoder Output** | `logits` float32[1, seq, 50264] (vocabulary size) |
| **ROUGE-1** | 0.481 |
| **ROUGE-2** | 0.186 |
| **ROUGE-L** | 0.253 |

#### Input: Text String

Security incident text (10–100,000 characters). Prefixed with `"Summarize the following security incident report: "` before tokenization. Truncated to 1,024 tokens max.

#### Output: Generated Summary

The decoder runs greedy autoregressive decoding:

1. Encoder produces hidden states from input text (single forward pass).
2. Decoder generates one token at a time, conditioned on encoder output + all previously generated tokens.
3. No-repeat n-gram blocking (n=3) prevents repetitive text.
4. Decoding continues until EOS token or max tokens reached.

**Modes**:

| Mode | Min Tokens | Max Tokens | Typical Output |
|---|---|---|---|
| `executive` | 50 | 100 | 2–4 sentence high-level summary |
| `analyst` | 150 | 400 | 8–15 sentence detailed summary |

#### Training Hyperparameters

```
epochs:                     4
batch_size:                 4
gradient_accumulation_steps: 8
learning_rate:              3e-5
warmup_steps:               500
max_source_length:          1024
max_target_length:          256
num_beams:                  4
gradient_checkpointing:     true
optimizer:                  AdamW
```

---

## 3. API Reference

**Base URL**: `http://<host>:8000`
**Docs**: `/docs` (Swagger UI), `/redoc` (ReDoc)

### 3.1 Health & Readiness

#### `GET /health`

Returns basic service health.

**Response** (`HealthResponse`):
```json
{
  "status": "ok",
  "version": "1.0.0",
  "models_loaded": {
    "triage": true,
    "ner": true,
    "summarizer": true
  }
}
```

#### `GET /ready`

Deep readiness check for all dependencies.

**Response** (`ReadinessResponse`):
```json
{
  "db_ok": true,
  "redis_ok": true,
  "kafka_ok": true,
  "models_status": {
    "triage": true,
    "ner": true,
    "summarizer": true
  }
}
```

---

### 3.2 Triage Endpoints

#### `POST /triage/score`

Score a single alert using the XGBoost ONNX model.

**Request Body** (`CanonicalAlert`):

```json
{
  "alert_id": "abc-123",
  "tenant_id": "tenant-01",
  "timestamp": "2025-01-15T10:30:00Z",
  "source": "wazuh",
  "normalized_title": "Suspicious PowerShell execution",
  "severity": "High",
  "raw_data": { /* original vendor payload */ },
  "mitre_tactic": "Execution",
  "mitre_technique": "T1059.001",
  "entity_type": "Process",
  "device_name": "WORKSTATION-01",
  "ip_address": "10.0.1.50",
  "user_name": "jdoe",
  "file_hash": null,
  "url": null,
  "domain": null,
  "category": "SuspiciousActivity",
  "threat_family": null,
  "suspicion_level": "High",
  "dedup_key": null
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `alert_id` | string | Yes | Unique alert identifier |
| `tenant_id` | string | Yes | Tenant for multi-tenant isolation |
| `timestamp` | datetime | Yes | Alert creation time |
| `source` | string | Yes | SIEM vendor (`wazuh`, `splunk`, `elastic_siem`, `crowdstrike`) |
| `normalized_title` | string | Yes | Human-readable alert title |
| `severity` | enum | Yes | `Low` / `Medium` / `High` / `Critical` |
| `raw_data` | object | Yes | Original unmodified vendor payload |
| `mitre_tactic` | string | No | MITRE ATT&CK tactic name |
| `mitre_technique` | string | No | MITRE technique ID (e.g., T1566.001) |
| `entity_type` | string | No | `Process`, `File`, `IP`, etc. |
| `device_name` | string | No | Host/device name |
| `ip_address` | string | No | Source/destination IP |
| `user_name` | string | No | Associated username |
| `file_hash` | string | No | SHA256 / MD5 / SHA1 hash |
| `url` | string | No | Suspicious URL |
| `domain` | string | No | Domain name |
| `category` | string | No | Alert category |
| `threat_family` | string | No | Malware family name |
| `suspicion_level` | string | No | `Low` / `Medium` / `High` / `Critical` |
| `dedup_key` | string | No | Pre-computed deduplication hash |

**Response** (`TriageResult`):

```json
{
  "alert_id": "abc-123",
  "tenant_id": "tenant-01",
  "ml_score": 0.847321,
  "incident_grade": "TruePositive",
  "risk_score": 72.15,
  "auto_closed": false,
  "model_version": "latest",
  "processing_ms": 12
}
```

| Field | Type | Description |
|---|---|---|
| `alert_id` | string | Echo of input alert ID |
| `tenant_id` | string | Echo of tenant ID |
| `ml_score` | float [0.0–1.0] | Probability of being a True Positive |
| `incident_grade` | enum | `TruePositive` / `FalsePositive` / `BenignPositive` |
| `risk_score` | float [0.0–100.0] | Composite risk score (ML + asset criticality + behavioral) |
| `auto_closed` | bool | `true` if `ml_score < 0.01` |
| `model_version` | string | Model version tag |
| `processing_ms` | int | Inference latency in milliseconds |

#### `GET /triage/health`

```json
{
  "pipeline": "triage",
  "model_loaded": true,
  "metadata": { /* triage_metadata.json content */ }
}
```

---

### 3.3 NER Endpoints

#### `POST /nlp/ner`

Extract named entities and IOCs from text.

**Request Body** (`NERRequest`):

```json
{
  "text": "APT29 deployed SUNBURST backdoor via SolarWinds Orion update. C2 server at 203.0.113.42. Hash: a25cadd48d70f6ea0c4a58c167dce8e7.",
  "tenant_id": "tenant-01"
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `text` | string (1–50,000 chars) | Yes | Text to analyze for entities/IOCs |
| `tenant_id` | string | No | Defaults to `"unknown"` |

**Response** (`NERResult`):

```json
{
  "entities": [
    {
      "text": "APT29",
      "label": "Organization",
      "start": 0,
      "end": 5,
      "confidence": 0.94,
      "ioc_type": "Unknown",
      "ioc_validated": false
    },
    {
      "text": "SUNBURST",
      "label": "Malware",
      "start": 15,
      "end": 23,
      "confidence": 0.91,
      "ioc_type": "Unknown",
      "ioc_validated": false
    },
    {
      "text": "SolarWinds Orion",
      "label": "System",
      "start": 37,
      "end": 53,
      "confidence": 0.88,
      "ioc_type": "Unknown",
      "ioc_validated": false
    },
    {
      "text": "203.0.113.42",
      "label": "Indicator",
      "start": 75,
      "end": 87,
      "confidence": 0.96,
      "ioc_type": "IP_Address",
      "ioc_validated": true
    },
    {
      "text": "a25cadd48d70f6ea0c4a58c167dce8e7",
      "label": "Indicator",
      "start": 95,
      "end": 127,
      "confidence": 0.93,
      "ioc_type": "File_Hash",
      "ioc_validated": true
    }
  ],
  "events": [
    {
      "type": "Backdoor",
      "keyword": "backdoor"
    }
  ],
  "processing_ms": 87,
  "cached": false
}
```

**IOCEntity fields**:

| Field | Type | Description |
|---|---|---|
| `text` | string | Raw text span of the extracted entity |
| `label` | enum | `Malware` / `Indicator` / `System` / `Vulnerability` / `Organization` / `O` |
| `start` | int | Character offset (0-indexed, inclusive) |
| `end` | int | Character offset (exclusive) |
| `confidence` | float [0.0–1.0] | Mean softmax confidence across entity tokens |
| `ioc_type` | enum | `IP_Address` / `File_Hash` / `Domain` / `URL` / `Email_Address` / `CVE_ID` / `Unknown` |
| `ioc_validated` | bool | `true` if the IOC value passed regex validation |

#### `POST /nlp/ner/batch`

Batch extraction for up to 32 texts at once.

**Request Body** (`NERBatchRequest`):

```json
{
  "texts": [
    "WannaCry exploiting CVE-2017-0144 on Windows 7 systems.",
    "Phishing email from attacker@evil.com with link to http://malicious.example.com/payload"
  ],
  "tenant_id": "tenant-01"
}
```

**Response**: `list[NERResult]` — one result per input text.

#### `GET /nlp/ner/health`

```json
{
  "pipeline": "ner",
  "model_loaded": true,
  "metadata": { /* ner_metadata.json content */ }
}
```

---

### 3.4 Summarization Endpoints

#### `POST /nlp/summarize`

Generate a summary of a security incident report.

**Request Body** (`SummarizeRequest`):

```json
{
  "text": "On January 15, 2025, multiple alerts were triggered on WORKSTATION-01 indicating suspicious PowerShell execution... (long incident report text)",
  "tenant_id": "tenant-01",
  "case_id": "CASE-2025-001",
  "mode": "executive"
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `text` | string (10–100,000 chars) | Yes | Incident report text |
| `tenant_id` | string | No | Defaults to `"unknown"` |
| `case_id` | string | No | If provided, summary is persisted to PostgreSQL |
| `mode` | enum | No | `executive` (default, short) or `analyst` (detailed) |

**Response** (`SummarizeResult`):

```json
{
  "summary": "A suspicious PowerShell execution was detected on WORKSTATION-01 targeting credential stores. The attack pattern matches Mimikatz-style credential harvesting with lateral movement indicators.",
  "mode": "executive",
  "model_version": "latest",
  "processing_ms": 720,
  "cached": false,
  "input_tokens": 487,
  "output_tokens": 38
}
```

| Field | Type | Description |
|---|---|---|
| `summary` | string | Generated summary text |
| `mode` | enum | `executive` or `analyst` |
| `model_version` | string | Model version tag |
| `processing_ms` | int | Total generation latency |
| `cached` | bool | `true` if served from Redis cache |
| `input_tokens` | int | Token count of the input |
| `output_tokens` | int | Token count of the generated summary |

#### `GET /nlp/summarize/health`

```json
{
  "pipeline": "summarizer",
  "model_loaded": true,
  "metadata": { /* summarizer model_metadata.json */ }
}
```

---

### 3.5 SIEM Ingestion Endpoints

#### `POST /ingest/siem`

Receive a raw SIEM webhook, normalize it to the canonical schema, deduplicate, and publish to Kafka.

**Request Body** (`SIEMRawPayload`):

```json
{
  "vendor": "wazuh",
  "tenant_id": "tenant-01",
  "raw": {
    "id": "1705312200.12345",
    "rule": {
      "level": 12,
      "description": "Possible rootkit activity detected",
      "id": "510"
    },
    "agent": {
      "id": "003",
      "name": "web-server-01"
    },
    "data": {
      "srcip": "192.168.1.100"
    }
  },
  "timestamp": "2025-01-15T10:30:00Z"
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `vendor` | string | Yes | One of: `wazuh`, `splunk`, `elastic_siem`, `crowdstrike` |
| `tenant_id` | string | Yes | Tenant identifier |
| `raw` | object | Yes | Vendor-specific alert JSON payload |
| `timestamp` | datetime | No | Override timestamp |

**Vendor-Specific Field Mapping**:

| Vendor | Alert ID Path | Title Path | Severity Path |
|---|---|---|---|
| **Wazuh** | `["id"]` | `["rule"]["description"]` | `["rule"]["level"]` (int 0–15) |
| **Splunk** | `["sid"]` | `["search_name"]` | `["result"]["urgency"]` (string) |
| **Elastic SIEM** | `["kibana"]["alert"]["uuid"]` | `["kibana"]["alert"]["rule"]["name"]` | `["kibana"]["alert"]["severity"]` |
| **CrowdStrike** | `["composite_id"]` | `["display_name"]` | `["max_severity_displayname"]` |

**Severity Normalization**:

| Vendor | Input | → Normalized |
|---|---|---|
| Wazuh | Level 0–3 | Low |
| Wazuh | Level 4–7 | Medium |
| Wazuh | Level 8–11 | High |
| Wazuh | Level 12–15 | Critical |
| Splunk | `"informational"` / `"low"` | Low |
| Splunk | `"medium"` | Medium |
| Splunk | `"high"` | High |
| Splunk | `"critical"` | Critical |
| Elastic / CrowdStrike | Direct string mapping | Low / Medium / High / Critical |

**Deduplication**: SHA-256 of `(tenant_id + alert_id + title)` checked against Redis with 1-hour TTL.

**Response** (`IngestionResult`):

```json
{
  "accepted": true,
  "alert_id": "1705312200.12345",
  "tenant_id": "tenant-01",
  "source": "wazuh",
  "kafka_topic": "alerts.raw",
  "deduplicated": false
}
```

#### `POST /ingest/siem/batch`

Batch ingest up to 100 raw alerts.

**Request**: `list[SIEMRawPayload]`
**Response**: `list[IngestionResult]`

#### `GET /ingest/siem/vendors`

```json
{
  "vendors": ["wazuh", "splunk", "elastic_siem", "crowdstrike"]
}
```

---

## 4. Architecture & Data Flow

### 4.1 Component Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          ARIES AI MODULE                                │
│                                                                         │
│  ┌──────────────────── FastAPI Service (port 8000) ──────────────────┐  │
│  │                                                                    │  │
│  │  ┌─────────────┐  ┌──────────────┐  ┌──────────────────────────┐  │  │
│  │  │ /ingest/siem│  │ /triage/score│  │ /nlp/ner  /nlp/summarize│  │  │
│  │  │  Normalizer │  │  XGBoost ONNX│  │ SecureBERT    BART ONNX │  │  │
│  │  └──────┬──────┘  └──────┬───────┘  └─────────┬────────────────┘  │  │
│  │         │                │                     │                    │  │
│  │         │         ┌──────┴───────┐             │                    │  │
│  │         │         │  ModelStore  │◄────────────┘                    │  │
│  │         │         │ (ONNX Sess.) │                                  │  │
│  │         │         └──────────────┘                                  │  │
│  └─────────┼──────────────────────────────────────────────────────────┘  │
│            │                                                             │
│  ┌─────────▼───────┐  ┌────────────────┐  ┌──────────────────────────┐  │
│  │      Kafka      │  │  PostgreSQL    │  │        Redis             │  │
│  │  alerts.raw     │  │  alerts table  │  │  NER/summary cache      │  │
│  │  alerts.enriched│  │  iocs table    │  │  dedup keys             │  │
│  │  ml.feedback    │  │  case_summaries│  │  TTL: 30m / 2h / 1h     │  │
│  └────────┬────────┘  └────────────────┘  └──────────────────────────┘  │
│           │                                                              │
│  ┌────────▼────────┐  ┌────────────────┐  ┌──────────────────────────┐  │
│  │ TriageConsumer  │  │    MinIO (S3)  │  │       MLflow             │  │
│  │ (background)    │  │  ONNX model    │  │  Experiment tracking     │  │
│  │ alerts.raw →    │  │  artifacts     │  │  Model registry          │  │
│  │ alerts.enriched │  │                │  │                          │  │
│  └─────────────────┘  └────────────────┘  └──────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4.2 End-to-End Alert Flow

```
  SIEM Vendor                    ARIES AI Module                   Orchestrator
  (Wazuh/Splunk/                                                   (Go Engine)
   Elastic/CS)
       │
       │  Raw alert webhook
       ▼
  POST /ingest/siem ──► Normalize ──► Dedup (Redis) ──► Kafka [alerts.raw]
                                                              │
                                                              ▼
                                                   TriageKafkaConsumer
                                                              │
                                                    extract_features()
                                                    (49-dim vector)
                                                              │
                                                    ONNX XGBoost inference
                                                    (ml_score, grade)
                                                              │
                                                    Compute risk_score
                                                              │
                                              ┌───────────────┼──────────────┐
                                              │               │              │
                                         PostgreSQL     auto_closed?    Kafka
                                         (persist)       ml < 0.01   [alerts.enriched]
                                                              │              │
                                                         Close alert         ▼
                                                                     Orchestrator
                                                                          │
                                                              ┌───────────┼──────────┐
                                                              │                      │
                                                     POST /nlp/ner         POST /nlp/summarize
                                                              │                      │
                                                     IOC entities           Summary text
                                                              │                      │
                                                              ▼                      ▼
                                                      Enriched Response     Case Report
```

### 4.3 Triage Pipeline Flow

```
CanonicalAlert
       │
       ▼
┌─────────────────────────┐
│   Feature Engineering   │
│                         │
│  Temporal: hour, day,   │
│    month from timestamp │
│  MITRE: flags + count   │
│  Null flags: 7 binary   │
│  Categorical: 37 fields │
│    → TargetEncoder      │
│    → float [0, 1]       │
│                         │
│  Output: float32[1, 49] │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│   ONNX XGBoost Session  │
│                         │
│  Input:  float_input    │
│          [1, 49]        │
│  Output: labels [1]     │
│          probs  [1, 3]  │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│   Post-Processing       │
│                         │
│  ml_score = P(class=2)  │
│  grade = label → enum   │
│  risk = weighted combo  │
│  auto_close if < 0.01   │
└─────────────────────────┘
```

### 4.4 NER Pipeline Flow

```
Input Text (str)
       │
       ▼
┌──────────────────────────┐
│  Redis Cache Check       │
│  Key: {tenant}:ner:{hash}│
│  Hit? → return cached    │
└────────────┬─────────────┘
             │ miss
             ▼
┌──────────────────────────┐
│  SecureBERT Tokenizer    │
│                          │
│  text → input_ids        │
│       → attention_mask   │
│       → offset_mapping   │
│  Max length: 512 tokens  │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│  ONNX NER Session        │
│                          │
│  Input:  input_ids       │
│          attention_mask   │
│          [1, ≤512]       │
│                          │
│  Output: logits          │
│          [1, seq, 11]    │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│  Post-Processing         │
│                          │
│  1. Argmax → label IDs   │
│  2. Softmax → confidence │
│  3. BIO tag collapsing   │
│     (sub-words → spans)  │
│  4. Regex IOC typing     │
│  5. Event keyword detect │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│  Cache in Redis (30 min) │
│  Return NERResult        │
└──────────────────────────┘
```

### 4.5 Summarization Pipeline Flow

```
Input Text (str) + mode
       │
       ▼
┌──────────────────────────┐
│  Redis Cache Check       │
│  Key: {tenant}:summarizer│
│       :{hash(text+mode)} │
│  Hit? → return cached    │
└────────────┬─────────────┘
             │ miss
             ▼
┌──────────────────────────────┐
│  Prefix + Tokenize           │
│                              │
│  "Summarize the following    │
│   security incident report:  │
│   {text}"                    │
│                              │
│  → input_ids [1, ≤1024]     │
│  → attention_mask [1, ≤1024] │
└────────────┬─────────────────┘
             │
             ▼
┌──────────────────────────────┐
│  ONNX Encoder (single pass)  │
│                              │
│  Input:  input_ids,          │
│          attention_mask       │
│  Output: hidden_state        │
│          [1, seq, 768]       │
└────────────┬─────────────────┘
             │
             ▼
┌──────────────────────────────┐
│  ONNX Decoder (autoregress.) │
│                              │
│  For each step:              │
│    Input: decoder_input_ids, │
│           encoder_hidden,    │
│           attention_mask      │
│    Output: logits            │
│            [1, seq, 50264]   │
│                              │
│  Next token = argmax(logits) │
│  Block repeat 3-grams       │
│  Stop at EOS or max tokens   │
│                              │
│  executive: 50–100 tokens    │
│  analyst:   150–400 tokens   │
└────────────┬─────────────────┘
             │
             ▼
┌──────────────────────────────┐
│  Decode token IDs → text     │
│  Cache in Redis (2 hours)    │
│  Persist to DB (if case_id)  │
│  Return SummarizeResult      │
└──────────────────────────────┘
```

---

## 5. Model Serving Infrastructure

### 5.1 Model Loading at Startup

On application startup (FastAPI lifespan), models are loaded in this order:

1. **Resolve model artifacts via MLflow** — Query the MLflow REST API for the latest successful run of each experiment (`aries/triage-classifier`, `aries/ner`, `aries/summarizer`). Extract S3 artifact keys.
2. **Download from MinIO/S3** — Using `aioboto3`, download ONNX files to `/tmp/aries_models/`.
3. **Load ONNX InferenceSession** — Create `ort.InferenceSession` for each model with CPU provider.
4. **Load tokenizers** — Load fast tokenizers (`tokenizers.Tokenizer.from_file()`) for NER and Summarizer.
5. **Load TargetEncoder** — Deserialize `triage_encoder.pkl` via `joblib.load()`.
6. **Fallback** — If MLflow is unreachable, use hardcoded S3 keys from config.

All sessions are stored in a singleton `ModelStore` instance shared across requests.

### 5.2 ONNX Runtime Configuration

```python
SessionOptions:
  graph_optimization_level: ORT_ENABLE_ALL
  intra_op_num_threads:     4
  inter_op_num_threads:     2

Providers: ["CPUExecutionProvider"]
```

All ONNX inference runs in a thread pool executor (`loop.run_in_executor(None, ...)`) to avoid blocking the async event loop.

### 5.3 Caching Strategy

| Resource | Cache Key Pattern | TTL |
|---|---|---|
| NER results | `{tenant_id}:ner:{SHA256(text)[:32]}` | 1,800s (30 min) |
| Summaries | `{tenant_id}:summarizer:{SHA256(text+mode)[:32]}` | 7,200s (2 hours) |
| Dedup keys | `dedup:{SHA256(tenant_id+alert_id+title)[:24]}` | 3,600s (1 hour) |

Cache hits return in **< 1ms**, bypassing model inference entirely.

---

## 6. Database Schema

### `alerts`

| Column | Type | Description |
|---|---|---|
| `alert_id` | TEXT (PK) | Unique alert identifier |
| `tenant_id` | TEXT (NOT NULL) | Tenant isolation key |
| `normalized_title` | TEXT | Human-readable title |
| `raw_data` | JSONB | Original vendor payload |
| `source` | TEXT | Vendor name |
| `ml_score` | DOUBLE PRECISION | [0–1] TruePositive probability |
| `risk_score` | DOUBLE PRECISION | [0–100] composite score |
| `status` | TEXT | `New` / `Triaged` / `Closed_FP` / `Escalated` |
| `mitre_tactic` | TEXT | MITRE ATT&CK tactic |
| `created_at` | TIMESTAMPTZ | Row creation time |
| `updated_at` | TIMESTAMPTZ | Last update time |

**Indexes**: `tenant_id`, `risk_score DESC`, `status`

### `iocs`

| Column | Type | Description |
|---|---|---|
| `ioc_id` | BIGSERIAL (PK) | Auto-increment ID |
| `alert_id` | TEXT | Foreign key to alerts |
| `tenant_id` | TEXT | Tenant isolation |
| `ioc_type` | TEXT | `IP_Address` / `File_Hash` / `Domain` / `URL` / `Email_Address` / `CVE_ID` |
| `value` | TEXT | Actual IOC value |
| `source` | TEXT | Always `"NER"` |
| `confidence` | DOUBLE PRECISION | Model confidence |
| `created_at` | TIMESTAMPTZ | Extraction time |

**Unique constraint**: `(alert_id, ioc_type, value)`

### `case_summaries`

| Column | Type | Description |
|---|---|---|
| `case_id` | TEXT (PK) | Case identifier |
| `tenant_id` | TEXT | Tenant isolation |
| `executive_summary` | TEXT | Short summary (2–4 sentences) |
| `analyst_summary` | TEXT | Detailed summary (8–15 sentences) |
| `model_version` | TEXT | Model version used |
| `created_at` | TIMESTAMPTZ | Creation time |
| `updated_at` | TIMESTAMPTZ | Last update |

### `model_versions`

| Column | Type | Description |
|---|---|---|
| `id` | BIGSERIAL (PK) | Auto-increment ID |
| `pipeline` | TEXT | `triage` / `ner` / `summarizer` |
| `mlflow_run_id` | TEXT | MLflow run reference |
| `onnx_s3_uri` | TEXT | S3 path to ONNX artifact |
| `stage` | TEXT | `canary` / `production` |
| `metrics` | JSONB | Training metrics snapshot |
| `promoted_at` | TIMESTAMPTZ | Promotion timestamp |
| `promoted_by` | TEXT | Who promoted the model |

### `analyst_feedback`

| Column | Type | Description |
|---|---|---|
| `id` | BIGSERIAL (PK) | Auto-increment ID |
| `alert_id` | TEXT | Related alert |
| `tenant_id` | TEXT | Tenant isolation |
| `original_label` | TEXT | Model's prediction |
| `corrected_label` | TEXT | Analyst's correction |
| `analyst_id` | TEXT | Who provided feedback |
| `feedback_type` | TEXT | Feedback category |
| `used_in_training` | BOOLEAN | Whether consumed for retraining |

---

## 7. Kafka Topics

| Topic | Partitions | Purpose |
|---|---|---|
| `alerts.raw` | 3 | Normalized SIEM alerts awaiting triage |
| `alerts.enriched` | 3 | Triage-scored alerts sent to Orchestrator |
| `cases.updated` | 3 | Case lifecycle events |
| `playbooks.events` | 3 | Playbook execution events |
| `ml.feedback` | 3 | Analyst feedback for model retraining |
| `alerts.raw.dlq` | 1 | Dead-letter queue for failed raw alerts |
| `alerts.enriched.dlq` | 1 | Dead-letter queue for failed enriched alerts |

**Consumer Group**: `ml-triage-engine`
**Mode**: KRaft (no ZooKeeper)

---

## 8. Training Pipelines

All three models can be trained via Slurm HPC scripts or locally. Training is tracked in MLflow with artifacts uploaded to MinIO.

### XGBoost Triage Trainer

```
Input:  data/processed/triage_data.npz  (X_train, y_train, X_test, y_test)
Steps:  load → build XGBClassifier → fit (GPU) → evaluate → save JSON + metadata
Output: models/triage/xgboost_triage.json
        models/triage/model_metadata.json
        models/triage/triage_encoder.pkl
```

### SecureBERT NER Trainer

```
Input:  datasets/CyNER/{train,valid,test}.txt + datasets/CASIE/
Steps:  load → download SecureBERT → fine-tune (HuggingFace Trainer) → evaluate → save
Output: models/ner/model.safetensors
        models/ner/tokenizer.json
        models/ner/model_metadata.json
```

### BART Summarizer Trainer

```
Input:  datasets/gov_reports/
Steps:  load → download bart-base → fine-tune (Seq2SeqTrainer) → evaluate ROUGE → save
Output: models/summarizer/model.safetensors
        models/summarizer/tokenizer.json
        models/summarizer/model_metadata.json
```

### ONNX Export

After training, models are exported to ONNX (opset 17):

| Model | Export Tool | Output |
|---|---|---|
| XGBoost | `onnxmltools.convert_xgboost()` | `models/onnx/triage.onnx` |
| SecureBERT | HuggingFace → `torch.onnx.export()` + ONNX RT optimization | `models/onnx/ner.opt.onnx` |
| BART Encoder | `torch.onnx.export()` | `models/onnx/summarizer/encoder.onnx` |
| BART Decoder | `torch.onnx.export()` | `models/onnx/summarizer/decoder.onnx` |

---

## 9. Configuration Reference

All settings are configurable via environment variables with the `ARIES_` prefix.

| Variable | Default | Description |
|---|---|---|
| `ARIES_SERVICE_VERSION` | `1.0.0` | Service version string |
| `ARIES_HOST` | `0.0.0.0` | Bind address |
| `ARIES_PORT` | `8000` | Bind port |
| `ARIES_WORKERS` | `2` | Uvicorn worker count |
| `ARIES_LOG_LEVEL` | `INFO` | Structured logging level |
| `ARIES_DATABASE_URL` | `postgresql://aries:aries@localhost:5432/aries` | PostgreSQL connection |
| `ARIES_DB_POOL_MIN` | `5` | Min connection pool size |
| `ARIES_DB_POOL_MAX` | `20` | Max connection pool size |
| `ARIES_KAFKA_BOOTSTRAP_SERVERS` | `localhost:9092` | Kafka brokers |
| `ARIES_KAFKA_TOPIC_ALERTS_RAW` | `alerts.raw` | Raw alerts topic |
| `ARIES_KAFKA_TOPIC_ALERTS_ENRICHED` | `alerts.enriched` | Enriched alerts topic |
| `ARIES_REDIS_URL` | `redis://localhost:6379/0` | Redis connection |
| `ARIES_REDIS_CACHE_TTL_NER` | `1800` | NER cache TTL (seconds) |
| `ARIES_REDIS_CACHE_TTL_SUMMARY` | `7200` | Summary cache TTL (seconds) |
| `ARIES_S3_ENDPOINT_URL` | `http://localhost:9000` | MinIO endpoint |
| `ARIES_S3_BUCKET_MODELS` | `aries-models` | S3 bucket for ONNX models |
| `ARIES_MLFLOW_TRACKING_URI` | `http://localhost:5000` | MLflow server |
| `ARIES_AUTO_CLOSE_THRESHOLD` | `0.01` | Auto-close alerts below this ml_score |
| `ARIES_NER_MAX_LENGTH` | `512` | Max NER token length |
| `ARIES_NER_BATCH_MAX` | `32` | Max batch size for NER |
| `ARIES_SUMMARIZER_MAX_INPUT_TOKENS` | `1024` | Max input tokens for summarizer |
| `ARIES_SUMMARIZER_EXECUTIVE_MAX_TOKENS` | `100` | Max output tokens (executive mode) |
| `ARIES_SUMMARIZER_ANALYST_MAX_TOKENS` | `400` | Max output tokens (analyst mode) |

---

## 10. Performance Characteristics

### Inference Latencies (CPU, single request)

| Pipeline | Latency | Throughput | ONNX Model Size |
|---|---|---|---|
| Triage (XGBoost) | 5–20 ms | ~600 req/s | ~5 MB |
| NER (SecureBERT) | 50–150 ms | 30–50 req/s | ~100 MB (optimized) |
| Summarizer (BART) | 500–1,000 ms | 2–5 req/s | ~540 MB (enc + dec) |
| Redis cache hit | < 1 ms | — | — |

### Memory Footprint

| Component | Memory |
|---|---|
| ONNX NER session | ~200 MB |
| ONNX Summarizer sessions | ~600 MB |
| ONNX Triage session | ~10 MB |
| TargetEncoder | ~5 MB |
| **Total model footprint** | **~800 MB** |

### Deployment

- **Container**: Multi-stage Docker build (Python 3.11-slim)
- **Server**: Uvicorn with `uvloop` + `httptools`, 2 workers
- **Health check**: `GET /health` every 30s
- **Networking**: Connected to `aries_network` Docker bridge for inter-service communication
