# ARIES AI Service — Step-by-Step Demo Guide

> **Document Version:** 1.0.0  
> **Last Updated:** March 3, 2026  
> **Estimated Demo Time:** 20-30 minutes

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Environment Setup](#2-environment-setup)
3. [Starting the MLOps Stack](#3-starting-the-mlops-stack)
4. [Running Model Migration](#4-running-model-migration)
5. [Starting the FastAPI Service](#5-starting-the-fastapi-service)
6. [Verifying System Health](#6-verifying-system-health)
7. [Demo: Alert Triage](#7-demo-alert-triage)
8. [Demo: Named Entity Recognition](#8-demo-named-entity-recognition)
9. [Demo: Summarization](#9-demo-summarization)
10. [Demo: Full SIEM Pipeline](#10-demo-full-siem-pipeline)
11. [Inspecting Results](#11-inspecting-results)
12. [Cleanup](#12-cleanup)
13. [Troubleshooting](#13-troubleshooting)

---

## 1. Prerequisites

### 1.1 Required Software

| Software | Minimum Version | Check Command |
|----------|----------------|---------------|
| Docker | 24.0+ | `docker --version` |
| Docker Compose | 2.20+ | `docker compose version` |
| Python | 3.11+ | `python --version` |
| curl | Any | `curl --version` |
| jq | Any | `jq --version` |

### 1.2 System Requirements

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| RAM | 8 GB | 16 GB |
| Disk | 10 GB free | 20 GB free |
| CPU | 4 cores | 8 cores |

### 1.3 Clone Repository

```bash
git clone https://github.com/your-org/aries.git
cd aries
```

### 1.4 Directory Structure Overview

```
aries/
├── MLOps/                    # MLOps stack (MLflow, MinIO, Postgres)
│   ├── docker-compose.yml
│   └── migrate_to_mlflow.py
├── apps/fastapi_service/     # Inference service
│   ├── docker-compose.yml
│   └── src/
└── models/                   # Local model artifacts
```

---

## 2. Environment Setup

### 2.1 Create Python Virtual Environment

```bash
# From repository root
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# OR: .venv\Scripts\activate  # Windows
```

### 2.2 Install Dependencies

```bash
pip install -r requirements.txt
pip install -r MLOps/requirements.txt
```

### 2.3 Configure Docker DNS (If Needed)

If Docker container builds fail with DNS errors:

```bash
# Create Docker daemon config
sudo tee /etc/docker/daemon.json << 'EOF'
{
  "dns": ["8.8.8.8", "8.8.4.4"]
}
EOF

# Restart Docker
sudo systemctl restart docker
```

---

## 3. Starting the MLOps Stack

### 3.1 Navigate to MLOps Directory

```bash
cd MLOps
```

### 3.2 Start Services

```bash
docker compose up -d
```

**Expected Output:**
```
[+] Running 5/5
 ✔ Network mlops_network  Created
 ✔ Container minio        Started
 ✔ Container postgres     Started
 ✔ Container createbucket Started
 ✔ Container mlflow_server Started
```

### 3.3 Wait for Services to Initialize

```bash
# Wait 30 seconds for services to stabilize
sleep 30
```

### 3.4 Verify MLOps Stack

```bash
docker compose ps
```

**Expected Output (All Services Healthy):**
```
NAME            STATUS                   PORTS
createbucket    Exited (0) ...           
minio           Up X minutes (healthy)   0.0.0.0:9000->9000/tcp
mlflow_server   Up X minutes (healthy)   0.0.0.0:5000->5000/tcp
postgres        Up X minutes (healthy)   0.0.0.0:5432->5432/tcp
```

### 3.5 Access MLflow UI

Open in browser: **http://localhost:5000**

You should see the MLflow Tracking UI (experiments list may be empty initially).

---

## 4. Running Model Migration

### 4.1 Navigate Back to Root

```bash
cd ..
# Ensure you're in: /path/to/aries
```

### 4.2 Activate Virtual Environment

```bash
source .venv/bin/activate
```

### 4.3 Run Migration Script

```bash
python MLOps/migrate_to_mlflow.py
```

**Expected Output (4 Steps):**
```
=== Aries MLflow Migration ===

Step 1/4 — Triage model
  artifacts/triage/xgboost_triage.json → 46e824ab/triage_model/
  artifacts/triage/triage_encoder.pkl → 46e824ab/triage_model/
  ✔ triage_model@Production

Step 2/4 — NER model
  artifacts/ner/*.json,*.safetensors → 46e824ab/ner_model/
  ✔ ner_model@Production

Step 3/4 — Summarizer model
  artifacts/summarizer/*.json,*.safetensors → 46e824ab/summarizer_model/
  ✔ summarizer_model@Production

Step 4/4 — Summarizer ONNX
  artifacts/onnx/summarizer/* → 46e824ab/bart_onnx/
  ✔ bart_onnx@Production

Migration complete!
```

### 4.4 Verify in MLflow UI

1. Open **http://localhost:5000**
2. Click **"Models"** in left sidebar
3. You should see 4 registered models:
   - `triage_model` (Production)
   - `ner_model` (Production)
   - `summarizer_model` (Production)
   - `bart_onnx` (Production)

---

## 5. Starting the FastAPI Service

### 5.1 Navigate to FastAPI Directory

```bash
cd apps/fastapi_service
```

### 5.2 Build and Start Services

```bash
docker compose up -d --build
```

**First Build Time:** ~5-7 minutes (downloads dependencies)  
**Subsequent Builds:** ~30 seconds

**Expected Output:**
```
[+] Running 8/8
 ✔ Network aries_network    Created
 ✔ Container kafka          Started
 ✔ Container mlflow_db      Started
 ✔ Container redis          Started
 ✔ Container kafka-ui       Started
 ✔ Container pg-admin       Started
 ✔ Container aries          Started
 ✔ Container aries-consumer Started
```

### 5.3 Wait for Model Loading

```bash
# Wait for models to download from MinIO
sleep 60
```

### 5.4 Check Container Logs

```bash
docker compose logs aries | head -50
```

**Look For (Indicates Successful Model Loading):**
```
INFO:     loading model store …
INFO:     downloading artifacts from mlflow …
INFO:     triage ONNX loaded: (2, 49) → (2, 3)
INFO:     NER ONNX loaded: RobertaForTokenClassification
INFO:     summarizer ONNX loaded: encoder + decoder
INFO:     model store ready in 12.4 seconds
INFO:     Uvicorn running on http://0.0.0.0:8000
```

---

## 6. Verifying System Health

### 6.1 Check All Containers Running

```bash
docker compose ps
```

**Expected (All Healthy/Running):**
```
NAME             STATUS                    PORTS
aries            Up X min (healthy)        0.0.0.0:8000->8000/tcp
aries-consumer   Up X min                  
kafka            Up X min (healthy)        9092, 9094
kafka-ui         Up X min                  0.0.0.0:8080->8080/tcp
pg-admin         Up X min                  0.0.0.0:8088->80/tcp
mlflow_db        Up X min (healthy)        5432
redis            Up X min (healthy)        6379
```

### 6.2 Check API Health Endpoint

```bash
curl http://localhost:8000/health | jq
```

**Expected Response:**
```json
{
  "status": "healthy",
  "timestamp": "2026-03-03T10:30:45.123456",
  "models": {
    "triage": true,
    "ner": true,
    "summarizer": true
  }
}
```

### 6.3 Check Readiness Endpoint

```bash
curl http://localhost:8000/ready | jq
```

**Expected Response:**
```json
{
  "ready": true,
  "components": {
    "database": "connected",
    "kafka": "connected",
    "redis": "connected",
    "models": "loaded"
  }
}
```

### 6.4 Access Supporting UIs

| Service | URL | Credentials |
|---------|-----|-------------|
| **API Docs** | http://localhost:8000/docs | None |
| **Kafka UI** | http://localhost:8080 | None |
| **pgAdmin** | http://localhost:8088 | admin@aries.local / admin123 |
| **MLflow** | http://localhost:5000 | None |
| **MinIO** | http://localhost:9001 | minioadmin / minioadmin |

---

## 7. Demo: Alert Triage

### 7.1 Single Alert Scoring

```bash
curl -X POST http://localhost:8000/triage/score \
  -H "Content-Type: application/json" \
  -H "X-Tenant-ID: demo-tenant" \
  -d '{
    "AlertTitle": "Suspicious PowerShell execution",
    "Category": "Execution",
    "Severity": "High",
    "ServiceSource": "MicrosoftDefenderForEndpoint",
    "DeviceId": "device-123",
    "Timestamp": "2026-03-03T10:30:00Z"
  }' | jq
```

**Expected Response:**
```json
{
  "alert_id": "a1b2c3d4...",
  "ml_score": 0.873,
  "prediction": "TruePositive",
  "confidence": {
    "TruePositive": 0.873,
    "FalsePositive": 0.092,
    "BenignPositive": 0.035
  },
  "auto_close": false,
  "latency_ms": 12.5
}
```

### 7.2 Understanding the Response

| Field | Meaning |
|-------|---------|
| `ml_score` | Probability of True Positive (0-1) |
| `prediction` | Highest probability class |
| `auto_close` | True if score < threshold (0.01) |
| `latency_ms` | Inference time in milliseconds |

### 7.3 Test Auto-Close Scenario

```bash
# Benign alert example
curl -X POST http://localhost:8000/triage/score \
  -H "Content-Type: application/json" \
  -H "X-Tenant-ID: demo-tenant" \
  -d '{
    "AlertTitle": "Sign-in from known location",
    "Category": "InitialAccess",
    "Severity": "Informational",
    "ServiceSource": "AzureAD",
    "Timestamp": "2026-03-03T10:30:00Z"
  }' | jq
```

Low-scoring alerts will show `"auto_close": true`.

---

## 8. Demo: Named Entity Recognition

### 8.1 Extract IOCs from Text

```bash
curl -X POST http://localhost:8000/nlp/ner \
  -H "Content-Type: application/json" \
  -H "X-Tenant-ID: demo-tenant" \
  -d '{
    "text": "The attacker used mimikatz to dump credentials. They connected to C2 server 192.168.1.100:4444 and downloaded payload from hxxp://evil.com/malware.exe. The malware exploited CVE-2021-44228."
  }' | jq
```

**Expected Response:**
```json
{
  "entities": [
    {"text": "mimikatz", "label": "TOOL", "start": 18, "end": 26},
    {"text": "192.168.1.100:4444", "label": "IP", "start": 77, "end": 95},
    {"text": "evil.com", "label": "DOMAIN", "start": 125, "end": 133},
    {"text": "malware.exe", "label": "FILE", "start": 134, "end": 145},
    {"text": "CVE-2021-44228", "label": "VULNERABILITY", "start": 170, "end": 184}
  ],
  "latency_ms": 45.2,
  "cached": false
}
```

### 8.2 Batch NER Processing

```bash
curl -X POST http://localhost:8000/nlp/ner/batch \
  -H "Content-Type: application/json" \
  -H "X-Tenant-ID: demo-tenant" \
  -d '{
    "texts": [
      "APT41 exploited ProxyShell vulnerability CVE-2021-34473",
      "Ransomware encrypted files with .lockbit extension"
    ]
  }' | jq
```

### 8.3 NER Entity Types

| Label | Example | Description |
|-------|---------|-------------|
| `TOOL` | mimikatz, cobalt strike | Attack tools |
| `VULNERABILITY` | CVE-2021-44228 | CVE identifiers |
| `IP` | 192.168.1.100 | IP addresses |
| `DOMAIN` | evil.com | Domain names |
| `FILE` | malware.exe | File names |
| `MALWARE` | WannaCry, Emotet | Malware names |
| `THREAT_ACTOR` | APT41, Lazarus | Threat groups |

---

## 9. Demo: Summarization

### 9.1 Summarize Security Report

```bash
curl -X POST http://localhost:8000/nlp/summarize \
  -H "Content-Type: application/json" \
  -H "X-Tenant-ID: demo-tenant" \
  -d '{
    "tenant_id": "demo-tenant",
    "text": "On March 1, 2026, at approximately 14:30 UTC, our security operations center detected anomalous network activity originating from workstation WS-ACCT-042 in the Finance department. The initial alert was triggered by Microsoft Defender for Endpoint, which identified suspicious PowerShell execution patterns. Upon investigation, we discovered that the threat actor had gained initial access through a phishing email containing a malicious macro-enabled document. The document, disguised as an invoice from a known vendor, exploited CVE-2022-30190 (Follina) to execute arbitrary code without user interaction beyond opening the file. The attacker established persistence through a scheduled task that executed a Base64-encoded PowerShell script every 4 hours. This script beaconed to command-and-control infrastructure at IP address 185.234.72.15, which resolved to a bulletproof hosting provider in Eastern Europe. During the 6-hour window before detection, the attacker performed Kerberoasting to harvest service account hashes, moved laterally to two additional workstations, and exfiltrated approximately 2.3GB of data including financial records and employee personal information. Containment actions included immediate isolation of affected systems, forced password resets for all Finance department accounts, and blocking of identified IOCs at the firewall. The root cause analysis revealed that the phishing email bypassed email security due to the sender address closely mimicking a legitimate vendor domain."
  }' | jq
```

**Expected Response:**
```json
{
  "summary": "A security incident occurred on March 1, 2026, when a phishing attack exploiting CVE-2022-30190 led to unauthorized access to Finance department systems. The attacker established persistence, performed credential harvesting, and exfiltrated 2.3GB of sensitive data before containment.",
  "input_length": 1847,
  "summary_length": 285,
  "compression_ratio": 6.48,
  "latency_ms": 450.3,
  "cached": false
}
```

### 9.2 Summarization Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_length` | 142 | Maximum summary tokens |
| `min_length` | 56 | Minimum summary tokens |

---

## 10. Demo: Full SIEM Pipeline

### 10.1 Ingest Single Alert (Wazuh Format)

```bash
curl -X POST "http://localhost:8000/ingest/siem?vendor=wazuh" \
  -H "Content-Type: application/json" \
  -H "X-Tenant-ID: demo-tenant" \
  -d '{
    "rule": {
      "id": "87105",
      "level": 12,
      "description": "Windows: Process created with suspicious arguments"
    },
    "agent": {
      "id": "003",
      "name": "win-workstation",
      "ip": "192.168.1.50"
    },
    "data": {
      "win": {
        "eventdata": {
          "commandLine": "powershell.exe -enc SQBFAFgAIAAoAE4AZQB3AC...",
          "user": "DOMAIN\\john.doe"
        }
      }
    },
    "timestamp": "2026-03-03T10:45:00.000Z"
  }' | jq
```

**Expected Response:**
```json
{
  "status": "accepted",
  "alert_id": "wazuh_87105_1709462700",
  "kafka_partition": 0,
  "kafka_offset": 42
}
```

### 10.2 Batch Ingest (Multiple Alerts)

```bash
curl -X POST "http://localhost:8000/ingest/siem/batch?vendor=wazuh" \
  -H "Content-Type: application/json" \
  -H "X-Tenant-ID: demo-tenant" \
  -d '[
    {
      "rule": {"id": "87105", "level": 10, "description": "Suspicious process"},
      "agent": {"name": "server-01"},
      "timestamp": "2026-03-03T10:45:00.000Z"
    },
    {
      "rule": {"id": "87106", "level": 8, "description": "Network connection"},
      "agent": {"name": "server-02"},
      "timestamp": "2026-03-03T10:45:01.000Z"
    }
  ]' | jq
```

### 10.3 Supported SIEM Vendors

```bash
curl http://localhost:8000/ingest/siem/vendors | jq
```

**Response:**
```json
{
  "vendors": ["wazuh", "splunk", "elastic", "crowdstrike"]
}
```

### 10.4 Wait for Async Processing

The SIEM endpoint returns immediately after queueing to Kafka. The `aries-consumer` service processes alerts asynchronously.

```bash
# Wait for consumer to process
sleep 5
```

---

## 11. Inspecting Results

### 11.1 View Kafka Topics

Open **http://localhost:8080** (Kafka UI)

**Topics to Check:**
| Topic | Purpose |
|-------|---------|
| `alerts.raw` | Ingested alerts (raw) |
| `alerts.normalized` | After SIEM normalization |
| `alerts.enriched` | After ML enrichment |
| `alerts.auto_closed` | Low-scoring alerts |

Click on `alerts.enriched` → Messages to see processed alerts.

### 11.2 Query Database

**Option A: Using psql**
```bash
docker exec -i mlflow_db psql -U admin -d aries -c \
  "SELECT alert_id, tenant_id, ml_score, prediction, created_at 
   FROM alerts 
   ORDER BY created_at DESC 
   LIMIT 5;"
```

**Option B: Using pgAdmin**
1. Open **http://localhost:8088**
2. Login: `admin@aries.local` / `admin123`
3. Add Server:
   - Host: `mlflow_db`
   - Port: `5432`
   - Database: `aries`
   - Username: `admin`
   - Password: `password123`
4. Run query in Query Tool

### 11.3 View Consumer Logs

```bash
docker compose logs -f aries-consumer --since 1m
```

**Look For:**
```
INFO: processing alert_id=wazuh_87105_... ml_score=0.82 prediction=TruePositive
INFO: alert persisted to database
INFO: published to alerts.enriched partition=0 offset=43
```

### 11.4 Check Redis Cache

```bash
docker exec redis redis-cli KEYS "aries:*" | head -20
```

---

## 12. Cleanup

### 12.1 Stop FastAPI Stack

```bash
cd apps/fastapi_service
docker compose down
```

### 12.2 Stop MLOps Stack

```bash
cd ../../MLOps
docker compose down
```

### 12.3 Remove All Data (Full Reset)

```bash
# WARNING: This deletes all data!
cd MLOps
docker compose down -v  # Removes volumes

cd ../apps/fastapi_service
docker compose down -v
```

### 12.4 Remove Docker Networks

```bash
docker network rm aries_network mlops_network 2>/dev/null || true
```

---

## 13. Troubleshooting

### 13.1 Container Won't Start

**Check Logs:**
```bash
docker compose logs <service-name>
```

**Common Issues:**

| Error | Cause | Fix |
|-------|-------|-----|
| `port already in use` | Another service on same port | Stop conflicting service or change port |
| `network not found` | MLOps stack not running | Start MLOps stack first |
| `connection refused` | Dependency not ready | Wait and retry, or restart |

### 13.2 Models Not Loading

**Check MinIO Connection:**
```bash
curl http://localhost:9000/minio/health/live
```

**Check Models in MinIO:**
```bash
docker exec minio mc ls local/mlflow-artifacts/
```

### 13.3 Kafka Errors

**Verify Kafka is Running:**
```bash
docker exec kafka /opt/kafka/bin/kafka-topics.sh --bootstrap-server localhost:9092 --list
```

**Re-create Topics:**
```bash
docker exec kafka /opt/kafka/bin/kafka-topics.sh --bootstrap-server localhost:9092 --create --topic alerts.raw --partitions 3 --replication-factor 1
```

### 13.4 Database Connection Issues

**Check Database:**
```bash
docker exec -i mlflow_db psql -U admin -d aries -c "SELECT 1;"
```

**Reset Database:**
```bash
cd apps/fastapi_service
docker compose down -v
docker compose up -d
```

### 13.5 ml_score Always 0.0

**Cause:** TargetEncoder not loaded

**Fix:**
1. Ensure migration completed successfully
2. Check encoder exists: `docker exec minio mc ls local/mlflow-artifacts/ | grep encoder`
3. Rebuild container: `docker compose up -d --build`

### 13.6 Summarizer Echoing Input

**Cause:** BART decoder bug (fixed in current version)

**Fix:**
1. Pull latest code
2. Rebuild: `docker compose up -d --build`

---

## Quick Reference Commands

```bash
# Start everything
cd MLOps && docker compose up -d
cd ../apps/fastapi_service && docker compose up -d

# Check health
curl http://localhost:8000/health | jq

# View logs
docker compose logs -f aries

# Test triage
curl -X POST http://localhost:8000/triage/score \
  -H "Content-Type: application/json" \
  -H "X-Tenant-ID: demo" \
  -d '{"AlertTitle":"Test","Category":"Execution","Severity":"High"}'

# Stop everything
docker compose down
cd ../MLOps && docker compose down
```

---

## Demo Checklist

- [ ] MLOps stack running (5000, 9000, 5432)
- [ ] Migration completed (4 models in MLflow UI)
- [ ] FastAPI stack running (8000, 8080, 8088)
- [ ] Health endpoint returns healthy
- [ ] Triage endpoint returns ml_score > 0
- [ ] NER extracts entities correctly
- [ ] Summarizer produces summaries (not echoes)
- [ ] SIEM ingestion accepted
- [ ] Kafka UI shows messages in topics
- [ ] Database has alert records
