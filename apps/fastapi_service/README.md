# ARIES FastAPI Service — AI Brain

The **AI Brain** microservice for the ARIES (AI-Enhanced SOAR) platform.
Provides real-time alert triage, IOC extraction via NER, incident summarization,
and multi-vendor SIEM ingestion — all powered by ONNX Runtime for production inference.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Quick Start](#quick-start)
3. [Infrastructure Setup (Docker)](#infrastructure-setup-docker)
4. [Kafka Setup Guide](#kafka-setup-guide)
5. [Uploading Models to MinIO (via MLflow)](#uploading-models-to-minio-via-mlflow)
6. [Connecting a SIEM (Wazuh)](#connecting-a-siem-wazuh)
7. [End-to-End Alert Flow](#end-to-end-alert-flow)
8. [Orchestrator Integration](#orchestrator-integration-go-engine)
9. [API Reference](#api-reference)
10. [Running Tests](#running-tests)
11. [Environment Variables](#environment-variables)
12. [Troubleshooting](#troubleshooting)

---

## Architecture Overview

```
┌──────────────┐    webhook     ┌──────────────────┐   alerts.raw    ┌───────────┐
│  SIEM        │ ──────────────>│  /ingest/siem    │ ───────────────>│   Kafka   │
│  (Wazuh)     │                │  (normalizer)    │                 │           │
└──────────────┘                └──────────────────┘                 └─────┬─────┘
                                                                          │
                                                            ┌─────────────┘
                                                            ▼
                                                    ┌───────────────────┐
                                                    │  Triage Consumer  │
                                                    │  (XGBoost ONNX)   │
                                                    └────────┬──────────┘
                                                             │
                                alerts.enriched              │  insert to PostgreSQL
                        ┌────────────────────────────────────┤
                        ▼                                    ▼
                 ┌─────────────┐                      ┌──────────┐
                 │   Kafka     │                      │ Postgres │
                 │  (enriched) │                      │          │
                 └──────┬──────┘                      └──────────┘
                        │
                        ▼
              ┌───────────────────┐        ┌─────────────────────┐
              │  Go Orchestrator  │ ──────>│  /nlp/ner           │
              │  Engine           │ ──────>│  /nlp/summarize     │
              └───────────────────┘        └─────────────────────┘
```

**Pipelines:**

| Pipeline       | Model            | Format    | Endpoint              |
|---------------|------------------|-----------|-----------------------|
| Alert Triage  | XGBoost          | ONNX      | `POST /triage/score`  |
| NER (IOC)     | SecureBERT       | ONNX      | `POST /nlp/ner`       |
| Summarizer    | BART             | ONNX      | `POST /nlp/summarize` |
| SIEM Ingest   | Rule-based       | —         | `POST /ingest/siem`   |

---

## Quick Start

```bash
# 1 — Start the MLOps infrastructure stack first (postgres, minio, mlflow)
cd MLOps && docker compose up -d
# Wait ~30s for all services to be healthy

# 2 — Run the migration script to populate MLflow with trained models
python MLOps/migrate_to_mlflow.py

# 3 — Navigate to the FastAPI service
cd apps/fastapi_service

# 4 — Copy environment file
cp .env.example .env

# 5 — Start the serving stack (kafka, redis, fastapi)
docker compose up -d --build

# 6 — Verify
curl http://localhost:8000/health
curl http://localhost:8000/ready
```

> **Important:** The MLOps stack must be running first — it provides PostgreSQL,
> MinIO, and MLflow that the FastAPI service connects to via a shared Docker network
> (`aries_network`).

The service will be available at:
- **FastAPI Service:** http://localhost:8000
- **API Docs (Swagger):** http://localhost:8000/docs
- **API Docs (ReDoc):** http://localhost:8000/redoc
- **Kafka UI:** http://localhost:8090
- **MLflow UI:** http://localhost:5000
- **MinIO Console:** http://localhost:9001 (admin / password123)

---

## Infrastructure Setup (Docker)

### Two-Stack Architecture

ARIES uses **two separate Docker Compose files** sharing a single Docker network (`aries_network`):

| Stack | Location | Purpose | Services |
|-------|----------|---------|----------|
| **MLOps** | `MLOps/docker-compose.yml` | Training & experiment tracking | PostgreSQL, MinIO, MLflow, pgAdmin |
| **Serving** | `apps/fastapi_service/docker-compose.yml` | Production inference | Kafka, Redis, FastAPI |

The **MLOps stack defines** the shared `aries_network`. The **serving stack joins** it as an external network. This means:
- FastAPI connects to `postgres:5432`, `minio:9000`, and `mlflow:5000` by hostname
- No port conflicts or duplicated services
- Credentials come from `MLOps/.env` (default: admin / password123)

### MLOps Stack Services

| Service | Image | Port(s) | Purpose |
|---------|-------|---------|---------|
| `postgres` | postgres:latest | 5432 | MLflow metadata + ARIES application DB |
| `minio` | minio/minio | 9000/9001 | S3-compatible storage for model artifacts |
| `minio-setup` | minio/mc | — | Creates `mlflow-bucket` |
| `mlflow` | custom | 5000 | Experiment tracking & model registry |
| `pgadmin` | dpage/pgadmin4 | 8080 | Database admin UI |

### Serving Stack Services

| Service | Image | Port(s) | Purpose |
|---------|-------|---------|---------|
| `kafka` | apache/kafka:3.9.0 | 9092/9094 | Event bus (KRaft, no ZooKeeper) |
| `kafka-ui` | provectuslabs/kafka-ui | 8090 | Kafka topic browser |
| `kafka-init` | apache/kafka:3.9.0 | — | Creates all required topics |
| `redis` | redis:7-alpine | 6379 | Inference caching & deduplication |
| `fastapi_service` | custom (this Dockerfile) | 8000 | ARIES AI Brain |

### Starting Infrastructure Only

If you want to develop locally against real infrastructure:

```bash
# Start MLOps stack
cd MLOps && docker compose up -d

# Start only Kafka + Redis (not the FastAPI container)
cd apps/fastapi_service
docker compose up -d kafka kafka-init kafka-ui redis

# Run service locally
pip install -r requirements.txt
ARIES_KAFKA_BOOTSTRAP_SERVERS=localhost:9094 \
ARIES_S3_ENDPOINT_URL=http://localhost:9000 \
ARIES_S3_ACCESS_KEY=admin \
ARIES_S3_SECRET_KEY=password123 \
ARIES_S3_BUCKET_MODELS=mlflow-bucket \
ARIES_DATABASE_URL=postgresql://admin:password123@localhost:5432/aries \
ARIES_REDIS_URL=redis://localhost:6379/0 \
ARIES_MLFLOW_TRACKING_URI=http://localhost:5000 \
  uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

> **Note:** Use port `9094` for Kafka when connecting from the host (EXTERNAL listener).
> Inside Docker, services use port `9092` (PLAINTEXT listener).

### Model Loading on Startup

On startup, the FastAPI service:
1. **Queries MLflow** to discover the latest finished run for each experiment (`aries/triage-classifier`, `aries/ner`, `aries/bart-summarizer`)
2. **Resolves S3 artifact paths** from the run metadata
3. **Downloads ONNX models + tokenizers** from `mlflow-bucket` in MinIO
4. **Falls back** to hardcoded S3 keys in `.env` if MLflow is unreachable

---

## Kafka Setup Guide

### How Kafka is Configured

ARIES uses **Apache Kafka in KRaft mode** (no ZooKeeper required). The `docker-compose.yml` handles everything automatically.

### Topics

The `kafka-init` container creates these topics on first startup:

| Topic                 | Partitions | Purpose                                    |
|-----------------------|------------|---------------------------------------------|
| `alerts.raw`          | 3          | Normalized SIEM alerts for triage           |
| `alerts.enriched`     | 3          | Triage-scored alerts for orchestrator       |
| `cases.updated`       | 3          | Case lifecycle events                       |
| `playbooks.events`    | 3          | Playbook execution events                   |
| `ml.feedback`         | 3          | Analyst feedback for model retraining       |
| `alerts.raw.dlq`      | 1          | Dead letter queue for failed raw alerts     |
| `alerts.enriched.dlq` | 1          | Dead letter queue for failed enriched alerts|

### Verifying Kafka

```bash
# List all topics
docker exec -it aries_kafka /opt/kafka/bin/kafka-topics.sh \
  --bootstrap-server localhost:9092 --list

# Produce a test message to alerts.raw  (-i not -it when using <<<)
docker exec -i aries_kafka /opt/kafka/bin/kafka-console-producer.sh \
  --bootstrap-server localhost:9092 \
  --topic alerts.raw <<< '{"alert_id":"test-001","tenant_id":"t1","source":"manual","normalized_title":"Test Alert","raw_data": {}}'

# Consume from alerts.enriched (Ctrl+C to stop)
docker exec -it aries_kafka /opt/kafka/bin/kafka-console-consumer.sh \
  --bootstrap-server localhost:9092 \
  --topic alerts.enriched \
  --from-beginning
```

### Kafka UI

Open http://localhost:8090 to:
- View all topics, partitions, and consumer groups
- Browse messages in any topic
- Monitor consumer lag

### Manual Topic Management

```bash
# Create a custom topic
docker exec -it aries_kafka /opt/kafka/bin/kafka-topics.sh \
  --bootstrap-server localhost:9092 \
  --create --topic my.custom.topic \
  --partitions 3 --replication-factor 1

# Describe a topic
docker exec -it aries_kafka /opt/kafka/bin/kafka-topics.sh \
  --bootstrap-server localhost:9092 \
  --describe --topic alerts.raw

# Delete a topic
docker exec -it aries_kafka /opt/kafka/bin/kafka-topics.sh \
  --bootstrap-server localhost:9092 \
  --delete --topic my.custom.topic
```

### Connecting from Host vs Docker

| Context       | Bootstrap Server  | Why                              |
|--------------|-------------------|----------------------------------|
| Inside Docker | `kafka:9092`      | Internal PLAINTEXT listener      |
| From host     | `localhost:9094`  | External EXTERNAL listener       |
| Python local  | `localhost:9094`  | Same as host                     |

---

## Uploading Models to MinIO (via MLflow)

Models are managed through **MLflow** and stored in the `mlflow-bucket` on MinIO.
On startup, the FastAPI service automatically queries MLflow to find and download
the latest trained models.

### Using the Migration Script (Recommended)

The migration script uploads all local model artifacts to MLflow:

```bash
# 1 — Make sure the MLOps stack is running
cd MLOps && docker compose up -d

# 2 — Run the migration script  
python MLOps/migrate_to_mlflow.py

# This uploads to:
#   Experiment: aries/triage-classifier  →  onnx/triage.onnx
#   Experiment: aries/ner               →  onnx/ner.opt.onnx + tokenizer files
#   Experiment: aries/bart-summarizer   →  onnx/summarizer/encoder.onnx + decoder.onnx + tokenizer
```

### Verifying in MLflow UI

1. Open http://localhost:5000
2. You should see three experiments: `aries/triage-classifier`, `aries/ner`, `aries/bart-summarizer`
3. Each experiment has runs with artifacts (ONNX models + tokenizer files)

### Verifying in MinIO Console

1. Open http://localhost:9001 — login: `admin` / `password123`
2. Navigate to `mlflow-bucket`
3. You'll see experiment/run directories with the artifacts

### Manual Upload (Fallback)

If you need to manually upload models without MLflow, use the `mc` CLI:

```bash
mc alias set aries http://localhost:9000 admin password123

# Upload to mlflow-bucket (or configure ARIES_S3_BUCKET_MODELS in .env)
mc cp models/onnx/triage.onnx aries/mlflow-bucket/triage/triage.onnx
mc cp models/onnx/ner.opt.onnx aries/mlflow-bucket/ner/ner.opt.onnx
mc cp models/onnx/summarizer/encoder.onnx aries/mlflow-bucket/summarizer/encoder.onnx
mc cp models/onnx/summarizer/decoder.onnx aries/mlflow-bucket/summarizer/decoder.onnx
```

> When using manual upload, set the fallback S3 keys in `.env` (see `.env.example`).
> The service will use these if MLflow is unreachable.

---

## Connecting a SIEM (Wazuh)

### Step 1: Deploy Wazuh with Docker

Wazuh provides an official Docker deployment. Add it alongside ARIES:

```bash
# In your project root — clone Wazuh's Docker repo
git clone https://github.com/wazuh/wazuh-docker.git -b v4.9.0 wazuh-docker
cd wazuh-docker/single-node

# Generate SSL certificates
docker compose -f generate-indexer-certs.yml run --rm generator

# Start Wazuh
docker compose up -d
```

Wazuh will be available at:
- **Dashboard:** https://localhost:443 (admin / SecretPassword)
- **API:** https://localhost:55000

### Step 2: Configure Wazuh Webhook Integration

Create a custom integration in Wazuh to forward alerts to the ARIES ingestion endpoint.

**File: `/var/ossec/etc/ossec.conf`** (inside the Wazuh Manager container):

```xml
<ossec_config>
  <integration>
    <name>custom-aries</name>
    <hook_url>http://host.docker.internal:8000/ingest/siem</hook_url>
    <level>3</level>
    <alert_format>json</alert_format>
  </integration>
</ossec_config>
```

> **Note:** `host.docker.internal` resolves to the Docker host.
> On Linux without Docker Desktop, use the Docker bridge IP instead (usually `172.17.0.1`).

**Create the integration script** at `/var/ossec/integrations/custom-aries`:

```python
#!/usr/bin/env python3
"""
Wazuh integration to forward alerts to ARIES FastAPI service.
"""
import json
import sys
import requests

ARIES_URL = "http://host.docker.internal:8000/ingest/siem"
TENANT_ID = "default"

def main():
    # Wazuh passes alert file path as first argument
    alert_file = sys.argv[1]
    with open(alert_file) as f:
        alert = json.load(f)

    payload = {
        "vendor": "wazuh",
        "tenant_id": TENANT_ID,
        "raw": alert,
    }

    resp = requests.post(
        ARIES_URL,
        json=payload,
        headers={"X-Tenant-ID": TENANT_ID},
        timeout=10,
    )
    resp.raise_for_status()

if __name__ == "__main__":
    main()
```

Make it executable:
```bash
docker exec -it wazuh-manager chmod +x /var/ossec/integrations/custom-aries
docker exec -it wazuh-manager /var/ossec/bin/wazuh-control restart
```

### Step 3: Alternative — Use a Simple Webhook Forwarder

If you don't want to modify Wazuh config, use a lightweight sidecar:

```yaml
# Add to docker-compose.yml
  wazuh-forwarder:
    image: python:3.11-slim
    container_name: aries_wazuh_forwarder
    volumes:
      - ./scripts/wazuh_forwarder.py:/app/forwarder.py
    command: python /app/forwarder.py
    environment:
      WAZUH_API_URL: https://wazuh-manager:55000
      WAZUH_USER: wazuh-wui
      WAZUH_PASSWORD: MyS3cr37P450r.*-
      ARIES_INGEST_URL: http://fastapi_service:8000/ingest/siem
      TENANT_ID: default
      POLL_INTERVAL: 10
    depends_on:
      fastapi_service:
        condition: service_healthy
```

### Step 4: Test the SIEM Connection

```bash
# Simulate a Wazuh alert via the ingest endpoint
curl -X POST http://localhost:8000/ingest/siem \
  -H "Content-Type: application/json" \
  -H "X-Tenant-ID: default" \
  -d '{
    "vendor": "wazuh",
    "tenant_id": "default",
    "raw": {
      "id": "1734567890.123456",
      "rule": {
        "description": "SSH brute-force attack detected",
        "level": 12,
        "mitre": {
          "id": ["T1110"],
          "tactic": ["Credential Access"]
        }
      },
      "agent": {"name": "web-server-01", "ip": "192.168.1.100"},
      "data": {"srcip": "10.0.0.55"},
      "timestamp": "2026-01-15T10:30:00+0000"
    }
  }'
```

Expected response:
```json
{
  "accepted": true,
  "alert_id": "1734567890.123456",
  "topic": "alerts.raw",
  "dedup_key": "...",
  "vendor": "wazuh"
}
```

### Step 5: Verify End-to-End

```bash
# 1. Check alerts.raw topic
docker exec -it aries_kafka /opt/kafka/bin/kafka-console-consumer.sh \
  --bootstrap-server localhost:9092 \
  --topic alerts.raw --from-beginning --max-messages 5

# 2. Check alerts.enriched (after triage)
docker exec -it aries_kafka /opt/kafka/bin/kafka-console-consumer.sh \
  --bootstrap-server localhost:9092 \
  --topic alerts.enriched --from-beginning --max-messages 5

# 3. Check PostgreSQL
docker exec -it mlflow_db psql -U admin -d aries -c "SELECT alert_id, source, status, ml_score, risk_score FROM alerts ORDER BY created_at DESC LIMIT 10;"
```

`incident_grade` is emitted in the triage API response and the `alerts.enriched` Kafka event. The persisted `alerts` table stores the workflow status together with `ml_score` and `risk_score`.

---

## End-to-End Alert Flow

Here's exactly what happens when an alert enters the system:

```
     ┌───────┐          ┌──────────────┐         ┌───────────┐
     │ SIEM  │ ──POST──>│ /ingest/siem │ ──pub──>│ alerts.raw│
     └───────┘          └──────────────┘         └─────┬─────┘
                                                       │
                     ┌─────────────────────────────────┘
                     ▼
             ┌───────────────────┐
             │ TriageKafkaConsumer│
             │  1. Validate alert │
             │  2. Extract 49 feat│
             │  3. ONNX inference │
             │  4. Compute risk   │
             │  5. Insert to PG   │
             │  6. Auto-close?    │
             └────────┬──────────┘
                      │ publish
                      ▼
             ┌────────────────┐
             │alerts.enriched │──> Go Orchestrtor consumes
             └────────────────┘
                      │
                      ▼
             ┌────────────────────────┐
             │ Orchestrator calls:    │
             │  POST /nlp/ner         │
             │  POST /nlp/summarize   │
             │  (as part of playbook) │
             └────────────────────────┘
```

### Manual Testing the Full Pipeline

```bash
# Step 1: Ingest a Wazuh alert
curl -s -X POST http://localhost:8000/ingest/siem \
  -H "Content-Type: application/json" \
  -H "X-Tenant-ID: default" \
  -d '{
    "vendor": "wazuh",
    "tenant_id": "default",
    "raw": {
      "id": "manual-test-001",
      "rule": {"description": "Malware detected on endpoint", "level": 13},
      "timestamp": "2026-01-15T10:30:00Z"
    }
  }' | python3 -m json.tool

# Step 2: Run NER on some text
curl -s -X POST http://localhost:8000/nlp/ner \
  -H "Content-Type: application/json" \
  -H "X-Tenant-ID: default" \
  -d '{
    "text": "The malware was downloaded from https://evil.example.com/payload.exe and communicated with 185.220.101.1. Hash: e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "tenant_id": "default"
  }' | python3 -m json.tool

# Step 3: Summarize an incident report
curl -s -X POST http://localhost:8000/nlp/summarize \
  -H "Content-Type: application/json" \
  -H "X-Tenant-ID: default" \
  -d '{
    "text": "At 10:30 UTC on January 15, a critical ransomware incident was detected on the finance server. The malware variant, identified as LockBit 3.0, encrypted all files on the shared drive and deployed a ransom note demanding 2 BTC. The SOC team immediately initiated containment procedures, isolating the affected host from the network. Further investigation revealed the initial access vector was a phishing email containing a malicious Excel macro.",
    "tenant_id": "default",
    "mode": "executive"
  }' | python3 -m json.tool
```

---

## Orchestrator Integration (Go Engine)

The FastAPI service is designed as a **plug-and-play AI backend** for the Go orchestrator engine.

### Communication Pattern

The orchestrator (Go) communicates with this service in two ways:

1. **Kafka Event Bus** — The orchestrator subscribes to `alerts.enriched` to receive triage-scored alerts.
2. **HTTP REST API** — The orchestrator calls NLP endpoints synchronously during playbook execution.

### Go Client Example

```go
package ai

import (
    "bytes"
    "encoding/json"
    "fmt"
    "net/http"
    "time"
)

const baseURL = "http://fastapi_service:8000"

type AIClient struct {
    client   *http.Client
    tenantID string
}

func NewAIClient(tenantID string) *AIClient {
    return &AIClient{
        client:   &http.Client{Timeout: 30 * time.Second},
        tenantID: tenantID,
    }
}

// TriageScore calls POST /triage/score
func (c *AIClient) TriageScore(alert map[string]interface{}) (map[string]interface{}, error) {
    return c.post("/triage/score", alert)
}

// ExtractIOCs calls POST /nlp/ner
func (c *AIClient) ExtractIOCs(text string) (map[string]interface{}, error) {
    return c.post("/nlp/ner", map[string]interface{}{
        "text":      text,
        "tenant_id": c.tenantID,
    })
}

// Summarize calls POST /nlp/summarize
func (c *AIClient) Summarize(text, caseID, mode string) (map[string]interface{}, error) {
    return c.post("/nlp/summarize", map[string]interface{}{
        "text":      text,
        "tenant_id": c.tenantID,
        "case_id":   caseID,
        "mode":      mode,
    })
}

func (c *AIClient) post(path string, body interface{}) (map[string]interface{}, error) {
    data, _ := json.Marshal(body)
    req, _ := http.NewRequest("POST", baseURL+path, bytes.NewReader(data))
    req.Header.Set("Content-Type", "application/json")
    req.Header.Set("X-Tenant-ID", c.tenantID)

    resp, err := c.client.Do(req)
    if err != nil {
        return nil, err
    }
    defer resp.Body.Close()

    if resp.StatusCode != 200 {
        return nil, fmt.Errorf("AI service returned %d", resp.StatusCode)
    }

    var result map[string]interface{}
    json.NewDecoder(resp.Body).Decode(&result)
    return result, nil
}
```

### Kafka Consumer in Go

```go
// The Go orchestrator subscribes to alerts.enriched
// Each message contains the full EnrichedAlert:
// {
//   "alert_id": "...",
//   "tenant_id": "...",
//   "ml_score": 0.87,
//   "incident_grade": "TruePositive",
//   "risk_score": 72.5,
//   "auto_closed": false,
//   "normalized_title": "...",
//   "severity": "High",
//   ...
// }
```

### Required Headers

Every request from the orchestrator **must** include:

| Header          | Required | Description              |
|----------------|----------|--------------------------|
| `X-Tenant-ID`  | Yes      | Tenant identification    |
| `Content-Type` | Yes      | `application/json`       |

### Docker Networking

When running everything in Docker Compose, the orchestrator reaches the AI service at:

```
http://fastapi_service:8000
```

If the Go orchestrator runs on the Docker host:

```
http://localhost:8000
```

---

## API Reference

### Health & Readiness

| Method | Path     | Description                     |
|--------|----------|---------------------------------|
| GET    | /health  | Liveness probe + model status   |
| GET    | /ready   | Full readiness (DB, Kafka, etc.)|
| GET    | /docs    | Swagger UI                      |
| GET    | /redoc   | ReDoc documentation             |

### Triage

| Method | Path           | Description                        |
|--------|----------------|------------------------------------|
| POST   | /triage/score  | Score a single alert (sync)        |
| GET    | /triage/health | Triage pipeline health status      |

**POST /triage/score** — Request:
```json
{
  "alert_id": "alert-001",
  "tenant_id": "default",
  "timestamp": "2026-01-15T10:30:00Z",
  "source": "wazuh",
  "normalized_title": "SSH brute force detected",
  "severity": "High",
  "raw_data": {
    "AlertTitle": "Brute Force",
    "Category": "CredentialAccess"
  }
}
```

**Response:**
```json
{
  "alert_id": "alert-001",
  "ml_score": 0.87,
  "incident_grade": "TruePositive",
  "risk_score": 72.5,
  "auto_closed": false,
  "model_version": "xgboost_triage_v1"
}
```

### NER (IOC Extraction)

| Method | Path            | Description                     |
|--------|-----------------|----------------------------------|
| POST   | /nlp/ner        | Extract IOCs from text           |
| POST   | /nlp/ner/batch  | Batch NER (up to 32 texts)       |
| GET    | /nlp/ner/health | NER pipeline health              |

**POST /nlp/ner** — Request:
```json
{
  "text": "Malware connected to 185.220.101.1 exploiting CVE-2021-44228",
  "tenant_id": "default"
}
```

**Response:**
```json
{
  "text": "Malware connected to 185.220.101.1 exploiting CVE-2021-44228",
  "entities": [
    {"text": "185.220.101.1", "label": "Indicator", "ioc_type": "IP_Address", "ioc_validated": true, "confidence": 0.95, "start": 25, "end": 38},
    {"text": "CVE-2021-44228", "label": "Vulnerability", "ioc_type": "CVE_ID", "ioc_validated": true, "confidence": 0.92, "start": 50, "end": 64}
  ],
  "events": [{"event_type": "Exploit", "trigger": "exploiting"}],
  "processing_time_ms": 45.2
}
```

### Summarizer

| Method | Path                  | Description                       |
|--------|-----------------------|------------------------------------|
| POST   | /nlp/summarize        | Summarize incident text            |
| GET    | /nlp/summarize/health | Summarizer pipeline health         |

**Modes:** `executive` (short, 50-100 tokens) and `analyst` (detailed, 150-400 tokens).

### SIEM Ingestion

| Method | Path                | Description                          |
|--------|---------------------|---------------------------------------|
| POST   | /ingest/siem        | Ingest & normalize one SIEM alert    |
| POST   | /ingest/siem/batch  | Batch ingest (up to 100 alerts)      |
| GET    | /ingest/siem/vendors| List supported SIEM vendors          |

**Supported vendors:** `wazuh`, `splunk`, `elastic_siem`, `crowdstrike`

---

## Running Tests

### Prerequisites

```bash
cd apps/fastapi_service
pip install -r requirements.txt
pip install -r tests/requirements.txt
```

### Run All Tests

```bash
# Run all tests with verbose output
pytest

# Run with coverage report
pytest --cov=src --cov-report=term-missing

# Run a specific test file
pytest tests/test_triage.py -v

# Run only unit tests (skip integration)
pytest -m "not integration"
```

### Test Structure

```
tests/
├── conftest.py           # Shared fixtures (mock DB, Redis, Kafka, ONNX sessions)
├── test_health.py        # Health/readiness endpoints
├── test_triage.py        # Triage inference, features, risk score
├── test_ner.py           # IOC regex, classification, event detection
├── test_summarizer.py    # Summarization schemas and API
└── test_ingestion.py     # SIEM normalization, vendor mappings
```

### What the Tests Cover

- **Unit tests:** Feature engineering (49-dim vectors), IOC regex patterns (IPv4/v6, MD5, SHA1, SHA256, domain, URL, email, CVE), risk score computation, heuristic fallback, event detection, normalization for all 4 vendors
- **Integration tests:** All API endpoints with mocked ONNX sessions, tenant header enforcement, caching behavior, batch processing

---

## Environment Variables

All variables are prefixed with `ARIES_` and managed via `pydantic-settings`.

| Variable                              | Default                          | Description                    |
|---------------------------------------|----------------------------------|--------------------------------|
| `ARIES_SERVICE_NAME`                  | `aries-fastapi-service`          | Service identifier             |
| `ARIES_SERVICE_VERSION`               | `1.0.0`                          | Semantic version               |
| `ARIES_LOG_LEVEL`                     | `INFO`                           | Logging level                  |
| `ARIES_DEBUG`                         | `false`                          | Debug mode                     |
| `ARIES_DATABASE_URL`                  | `postgresql://...`               | PostgreSQL connection string   |
| `ARIES_KAFKA_BOOTSTRAP_SERVERS`       | `localhost:9094`                 | Kafka brokers                  |
| `ARIES_REDIS_URL`                     | `redis://localhost:6379/0`       | Redis connection               |
| `ARIES_S3_ENDPOINT_URL`              | `http://localhost:9000`          | MinIO/S3 endpoint              |
| `ARIES_S3_BUCKET_MODELS`             | `aries-models`                   | Model storage bucket           |
| `ARIES_S3_ACCESS_KEY`                | `minioadmin`                     | S3 access key                  |
| `ARIES_S3_SECRET_KEY`                | `minioadmin`                     | S3 secret key                  |
| `ARIES_AUTO_CLOSE_THRESHOLD`         | `0.01`                           | ML score below which to auto-close |
| `ARIES_TRIAGE_WEIGHT_ML`             | `0.50`                           | ML weight in risk formula      |
| `ARIES_TRIAGE_WEIGHT_ASSET`          | `0.30`                           | Asset weight in risk formula   |
| `ARIES_TRIAGE_WEIGHT_BEHAVIOR`       | `0.20`                           | Behavior weight in risk formula|

See `.env.example` for the complete list.

---

## Troubleshooting

### Service won't start — "S3 model download failed"

Models aren't in MinIO yet. Run `python MLOps/migrate_to_mlflow.py` to upload them via MLflow.
See [Uploading Models to MinIO (via MLflow)](#uploading-models-to-minio-via-mlflow).
The service will still start but inference endpoints will return 503.

### Kafka connection refused

Ensure Kafka is healthy:
```bash
docker compose ps kafka
docker logs aries_kafka | tail -20
```

If connecting from the host, use port `9094` (EXTERNAL listener), not `9092`.

### ONNX Runtime error: "Invalid model"

Ensure you're uploading the correct `.onnx` files:
- `triage.onnx` (from XGBoost export)
- `ner.opt.onnx` (from SecureBERT export, optimized)
- `summarizer/encoder.onnx` + `summarizer/decoder.onnx` (from BART export)

### Models not loading on startup

1. **Check MLflow is reachable**: `curl http://localhost:5000/api/2.0/mlflow/experiments/list`
2. **Check migration was run**: Ensure experiments exist in MLflow UI at http://localhost:5000
3. **Check MinIO has artifacts**: Open http://localhost:9001 (admin/password123), browse `mlflow-bucket`
4. If MLflow is unavailable, set fallback S3 keys in `.env` (see `.env.example`)

### PostgreSQL: "relation does not exist"

The schema is auto-created on first connection. If it fails, create the `aries` database manually:
```bash
docker exec -it mlflow_db psql -U admin -c "CREATE DATABASE aries;"
```

### Tenant header missing

All endpoints (except /health, /ready, /docs) require the `X-Tenant-ID` header.
Missing it returns `400 Bad Request`.

### Redis cache not working

Check Redis connectivity:
```bash
docker exec -it aries_redis redis-cli ping  # Should return PONG
```

The service degrades gracefully — if Redis is down, inference still works without caching.

---

## Project Structure

```
apps/fastapi_service/
├── main.py                          # App factory + lifespan handler
├── Dockerfile                       # Multi-stage production build
├── docker-compose.yml               # Full infrastructure stack
├── requirements.txt                 # Production dependencies
├── .env.example                     # Environment variable template
├── pyproject.toml                   # Pytest + coverage config
├── src/
│   ├── shared/
│   │   ├── config.py                # ServiceSettings (pydantic-settings)
│   │   ├── logging.py               # structlog JSON logging
│   │   ├── exceptions.py            # Custom error hierarchy
│   │   ├── s3_client.py             # Async MinIO/S3 client
│   │   ├── db.py                    # PostgreSQL with asyncpg
│   │   ├── kafka.py                 # Kafka producer + consumer base
│   │   ├── redis_client.py          # Redis caching + dedup
│   │   ├── dependencies.py          # FastAPI Depends injection
│   │   ├── mlflow_resolver.py       # MLflow REST API model discovery
│   │   └── model_loader.py          # ONNX model download + session loader
│   ├── triage/
│   │   ├── schemas.py               # CanonicalAlert, TriageResult, enums
│   │   ├── inference.py             # XGBoost ONNX inference + risk score
│   │   ├── feature_engineering.py   # 49-feature vector extraction
│   │   ├── consumer.py              # Kafka consumer (alerts.raw → enriched)
│   │   └── router.py                # POST /triage/score, GET /triage/health
│   ├── nlp/
│   │   ├── ner/
│   │   │   ├── schemas.py           # IOCEntity, IOCType, NERRequest/Result
│   │   │   ├── inference.py         # SecureBERT ONNX + IOC regex + events
│   │   │   └── router.py            # POST /nlp/ner, batch, health
│   │   └── summarizer/
│   │       ├── schemas.py           # SummarizeRequest/Result, modes
│   │       ├── inference.py         # BART encoder-decoder ONNX greedy decode
│   │       └── router.py            # POST /nlp/summarize, health
│   └── ingestion/
│       ├── schemas.py               # SIEMRawPayload, IngestionResult
│       ├── normalizer.py            # Multi-vendor normalization (4 vendors)
│       └── router.py                # POST /ingest/siem, batch, vendors
└── tests/
    ├── conftest.py                  # Comprehensive mock fixtures
    ├── test_health.py               # Health + readiness tests
    ├── test_triage.py               # Triage pipeline tests
    ├── test_ner.py                  # NER + IOC regex tests
    ├── test_summarizer.py           # Summarizer API tests
    └── test_ingestion.py            # Ingestion + normalization tests
```
