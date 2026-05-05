# ARIES Pipeline — End-to-End Validation & Changes Log

**Date:** 2026-03-30  
**Scope:** Full pipeline validation: Wazuh SIEM → ARIES AI-Enhanced SOAR  
**Result:** ✅ ALL SYSTEMS OPERATIONAL — Full end-to-end pipeline verified

---

## Summary

Brought up the entire ARIES SOAR platform (11 containers across 3 Docker Compose
stacks) and validated the complete alert pipeline from Wazuh SIEM through Kafka to
AI inference (triage, NER, summarizer). Identified and fixed all integration issues.
Final test: 28 Wazuh alerts automatically forwarded, normalized, triaged via ONNX
ML models, and stored in PostgreSQL — all without manual intervention.

---

## Files Modified

### 1. `apps/fastapi_service/src/shared/config.py`
**Change:** Fixed S3/MinIO credential defaults  
- `S3_ACCESS_KEY`: `minioadmin` → `admin`  
- `S3_SECRET_KEY`: `minioadmin` → `password123`  
- `S3_BUCKET`: `aries-models` → `mlflow-bucket`  
**Reason:** Credentials must match the MLOps docker-compose MinIO configuration.

### 2. `wazuh-integration/custom-aries` (REWRITTEN)
**Change:** Converted from standalone Python script to standard Wazuh bash wrapper  
**Before:** Python script with `#!/usr/bin/env python3` shebang (failed silently because `python3` is not in PATH inside Wazuh container)  
**After:** Standard Wazuh shell wrapper that invokes `custom-aries.py` via Wazuh's embedded Python (`/var/ossec/framework/python/bin/python3`)  
**Reason:** Wazuh integratord requires the two-file pattern (bash wrapper + `.py` logic) used by all built-in integrations (slack, pagerduty, virustotal, etc.). The embedded Python is the only Python available inside the Wazuh Docker container.

### 3. `wazuh-integration/custom-aries.py` (NEW)
**Change:** Created Python logic file for the ARIES integration  
**Content:** Reads alert JSON from temp file (arg 1), wraps in ARIES envelope (`{vendor, tenant_id, raw}`), POSTs to `http://aries_fastapi:8000/ingest/siem?vendor=wazuh` using stdlib `urllib` (no external dependencies needed). Configurable via `ARIES_INGESTION_URL` and `ARIES_TENANT_ID` env vars.  
**Deployment:** Must be installed to `/var/ossec/integrations/custom-aries.py` inside the Wazuh container with `chmod 750` and `chown root:wazuh`.

### 4. `wazuh-docker/single-node/config/wazuh_cluster/wazuh_manager.conf`
**Change:** Added ARIES integration block and auth.log monitoring  
**Added (integration block before `<cluster>`):**
```xml
<integration>
  <name>custom-aries</name>
  <level>3</level>
  <alert_format>json</alert_format>
</integration>
```
**Added (localfile for SSH log monitoring):**
```xml
<localfile>
  <log_format>syslog</log_format>
  <location>/var/log/auth.log</location>
</localfile>
```
**Reason:** Tells Wazuh integratord to forward all alerts with level ≥ 3 to ARIES via the custom-aries script. Auth.log monitoring enables SSH brute-force detection rules.

### 5. `wazuh-docker/single-node/docker-compose.yml`
**Change:** Added external `aries_network` for cross-stack communication  
**Added to wazuh.manager service networks:** `aries_network`  
**Added to top-level networks:**
```yaml
aries_network:
  external: true
```
**Note:** A bind mount for custom-aries was added then removed. NTFS filesystems give 777 permissions which Wazuh integratord rejects ("file has write permissions for everyone"). The scripts must be installed via `docker cp` + `chmod 750` instead.

### 6. `MLOps/docker-compose.yml`
**Change:** PostgreSQL volume switched from NTFS bind mount to named Docker volume  
**Before:** `/data/docker-volumes/aries/postgres:/var/lib/postgresql`  
**After:** `mlflow_postgres_data:/var/lib/postgresql` (named volume)  
**Added:** `volumes: mlflow_postgres_data:` declaration  
**Reason:** PostgreSQL requires specific file permissions (0700 for data directory) that NTFS/fuseblk filesystems cannot provide.

---

## Infrastructure State

### Running Containers (11 total)

| Stack | Container | Port | Status |
|-------|-----------|------|--------|
| MLOps | mlflow_db (PostgreSQL) | 5432 | ✅ healthy |
| MLOps | mlflow_minio (MinIO) | 9000/9001 | ✅ healthy |
| MLOps | mlflow_server (MLflow) | 5000 | ✅ running |
| MLOps | mlflow_pgadmin | 8088 | ✅ running |
| FastAPI | aries_kafka (KRaft) | 9092 | ✅ healthy |
| FastAPI | aries_kafka_ui | 8090 | ✅ running |
| FastAPI | aries_redis | 6379 | ✅ healthy |
| FastAPI | aries_fastapi | 8000 | ✅ healthy |
| Wazuh | wazuh.manager | 1514/1515/55000 | ✅ running |
| Wazuh | wazuh.indexer | 9200 | ✅ running |
| Wazuh | wazuh.dashboard | 443 | ✅ running |

### Docker Networks
- `aries_network` — shared bridge connecting all 3 stacks
- Each stack also has its own default network

### Models Loaded in FastAPI
| Model | Type | S3 Key | Status |
|-------|------|--------|--------|
| Triage | XGBoost (ONNX) | triage/triage.onnx + triage/triage_encoder.pkl | ✅ loaded |
| NER | SecureBERT (ONNX) | ner/ner.opt.onnx + 5 tokenizer files | ✅ loaded |
| Summarizer | BART (ONNX) | summarizer/encoder.onnx + decoder.onnx + 2 tokenizer files | ✅ loaded |

### Kafka Topics (8)
`alerts.raw`, `alerts.enriched`, `cases.updated`, `playbooks.events`, `ml.feedback`, `alerts.raw.dlq`, `alerts.enriched.dlq`, `__consumer_offsets`

### Database Tables (5)
`alerts`, `iocs`, `case_summaries`, `model_versions`, `analyst_feedback`

---

## Pipeline Verification Results

### Test: SSH Brute Force Attack Simulation
Injected SSH failure logs into `/var/log/auth.log` inside Wazuh container.

**Flow validated:**
```
/var/log/auth.log  →  Wazuh logcollector  →  analysisd (rule 5710: level 5, rule 5712: level 10)
  →  integratord  →  custom-aries wrapper  →  custom-aries.py
  →  POST /ingest/siem?vendor=wazuh  →  normalize (severity mapping)
  →  Kafka alerts.raw  →  TriageKafkaConsumer
  →  ONNX XGBoost inference (2-3ms latency)
  →  DB INSERT (status=Triaged, ml_score, risk_score)
  →  Kafka alerts.enriched
```

**Database result:** 28 alerts, all source=wazuh, all status=Triaged  
**Average ML score:** 0.0715 | **Average risk score:** 28.57  
**Severity mapping:** Level 5 → Medium (risk 27.88), Level 10 → High (risk 35.38)

### API Endpoint Tests
| Endpoint | Input | Result |
|----------|-------|--------|
| `POST /ingest/siem?vendor=wazuh` | Wazuh alert JSON | ✅ 200, alert_id returned |
| `POST /nlp/ner` | CVE/IP/domain text | ✅ Detected entities (CVE, IP, domain, systems) |
| `POST /nlp/summarize` | Incident description | ✅ Generated executive summary |
| `GET /health` | — | ✅ All models loaded, all services connected |

---

## Deployment Notes

### Installing Integration Scripts (after container restart)
The Wazuh container does not persist changes to `/var/ossec/integrations/` across
restarts unless using a volume. After each `docker compose up`:

```bash
# Copy scripts into running container
docker cp wazuh-integration/custom-aries single-node-wazuh.manager-1:/var/ossec/integrations/custom-aries
docker cp wazuh-integration/custom-aries.py single-node-wazuh.manager-1:/var/ossec/integrations/custom-aries.py

# Fix permissions (REQUIRED — integratord rejects files with world-write)
docker exec single-node-wazuh.manager-1 bash -c '
  chmod 750 /var/ossec/integrations/custom-aries
  chmod 750 /var/ossec/integrations/custom-aries.py
  chown root:wazuh /var/ossec/integrations/custom-aries
  chown root:wazuh /var/ossec/integrations/custom-aries.py
'

# Restart Wazuh to pick up the integration
docker exec single-node-wazuh.manager-1 /var/ossec/bin/wazuh-control restart
```

### Stack Startup Order
```bash
# 1. MLOps (creates aries_network)
cd MLOps && docker compose up -d

# 2. FastAPI (joins aries_network as external)
cd apps/fastapi_service && docker compose up -d

# 3. Wazuh (joins aries_network as external)
cd wazuh-docker/single-node && docker compose up -d

# 4. Install integration scripts (see above)
```

### Known Limitations (acceptable for FYP)
- **Triage asset_criticality / behavioral_score:** Deterministic stubs (not connected to live CMDB/UBA). Asset score = hash-based, behavioral = random-seeded. Acceptable — the ML pipeline and risk formula are fully functional.
- **CORS wildcard:** `allow_origins=["*"]` in FastAPI — acceptable for Docker-internal traffic. Restrict in production.
- **NTFS filesystem:** `/data` partition is NTFS, which prevents `chmod` operations. PostgreSQL uses a named Docker volume; Wazuh scripts installed via `docker cp`.

---

## Root Cause: Wazuh integratord Silent Failure

**Problem:** Wazuh integratord acknowledged the integration (`Enabling integration for: 'custom-aries'`) but never executed the script despite generating level 10 alerts.

**Root cause:** The custom-aries file was a standalone Python script with `#!/usr/bin/env python3` shebang. Inside the Wazuh 4.9.0 Docker container, `python3` is NOT in the system PATH — Wazuh ships its own embedded Python at `/var/ossec/framework/python/bin/python3`. When integratord executed the script, the shebang failed to resolve, and the error was silently swallowed.

**Fix:** Adopted Wazuh's standard two-file integration pattern:
1. `custom-aries` — bash wrapper (identical pattern to built-in slack/pagerduty/virustotal)
2. `custom-aries.py` — Python logic using stdlib only (urllib, json, sys, os)

The wrapper script's `*/integrations)` case resolves `WAZUH_PATH` and invokes `${WAZUH_PATH}/framework/python/bin/python3 ${DIR_NAME}/${SCRIPT_NAME}.py "$@"`.
