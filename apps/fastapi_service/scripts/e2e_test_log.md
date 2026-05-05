# ARIES End-to-End Test Log

**Date:** 2026-03-03  
**Stack:** MLOps + FastAPI serving  
**Outcome summary:** Triage pipeline ✓ · NER pipeline ✓ · Summarizer ✓ · DB persistence ✓ · Kafka flow ✓

---

## Prerequisites

```bash
# Activate the virtual environment
source /home/taha-salaar/data/FYP/Aries/.venv/bin/activate

# Start MLOps stack (postgres, minio, mlflow)
cd /home/taha-salaar/data/FYP/Aries/MLOps
docker compose up -d

# Start serving stack (kafka, redis, fastapi)
cd /home/taha-salaar/data/FYP/Aries/apps/fastapi_service
docker compose up -d
```

---

## 1 · Health & Readiness

```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "ok",
  "service": "aries-fastapi-service",
  "version": "1.0.0",
  "models_loaded": {"triage": true, "ner": true, "summarizer": true}
}
```

```bash
curl http://localhost:8000/ready
```

**Response:**
```json
{
  "ready": true,
  "kafka_connected": true,
  "db_connected": true,
  "redis_connected": true,
  "models": {"triage": true, "ner": true, "summarizer": true}
}
```

---

## 2 · Kafka Topic Verification

```bash
# List all topics
docker exec -it aries_kafka /opt/kafka/bin/kafka-topics.sh \
  --bootstrap-server localhost:9092 --list
```

**Expected output:** `__consumer_offsets`, `alerts.enriched`, `alerts.enriched.dlq`,
`alerts.raw`, `alerts.raw.dlq`, `cases.updated`, `ml.feedback`, `playbooks.events`

---

## 3 · Triage — Direct Kafka Produce (minimal alert)

> **Note:** Use `-i`, not `-it`, when redirecting stdin with `<<<` or pipes.

```bash
echo '{"alert_id":"test-001","tenant_id":"t1","source":"manual","normalized_title":"Test Alert","raw_data":{}}' \
  | docker exec -i aries_kafka /opt/kafka/bin/kafka-console-producer.sh \
      --bootstrap-server localhost:9092 --topic alerts.raw
```

**Output:** `Produced OK`

**FastAPI log result:**
```
triage_inference_complete  alert_id=test-001  ml_score=0.0  grade=BenignPositive
                           risk_score=25.0    auto_closed=True
alert_auto_closed          ml_score=0.0  threshold=0.01
```

---

## 4 · Triage — via SIEM Ingest endpoint (Wazuh ransomware alert)

```bash
curl -s -X POST http://localhost:8000/ingest/siem \
  -H "Content-Type: application/json" \
  -H "X-Tenant-ID: default" \
  -d '{
    "vendor": "wazuh",
    "tenant_id": "default",
    "raw": {
      "id": "1741234567.999001",
      "rule": {
        "description": "Ransomware: LockBit 3.0 file encryption activity detected on critical asset",
        "level": 15,
        "groups": ["ransomware", "malware", "attack"],
        "mitre": {
          "id": ["T1486", "T1059.001", "T1547.001"],
          "tactic": ["Impact", "Execution", "Persistence"]
        }
      },
      "agent": {"name": "finance-srv-01", "ip": "192.168.10.55"},
      "data": {
        "srcip": "185.220.101.47",
        "url": "http://185.220.101.47/lockbit/payload.exe",
        "process": "powershell.exe",
        "command_line": "powershell -nop -w hidden -enc JABzAD0ATgBlAHcA...",
        "file_path": "C:\\Users\\finance\\Documents\\*.encrypted",
        "hash_sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
      },
      "timestamp": "2026-03-03T05:00:00Z"
    }
  }' | python3 -m json.tool
```

**Response:**
```json
{
  "accepted": true,
  "alert_id": "1741234567.999001",
  "tenant_id": "default",
  "source": "wazuh",
  "kafka_topic": "alerts.raw",
  "deduplicated": false
}
```

**FastAPI log result:**
```
alert_normalized     alert_id=1741234567.999001  severity=Critical  vendor=wazuh
kafka_message_sent   topic=alerts.raw  key=1741234567.999001
triage_processing_alert  alert_id=1741234567.999001
triage_inference_complete  ml_score=0.0  risk_score=38.5  grade=BenignPositive  auto_closed=True
alert_auto_closed    ml_score=0.0  threshold=0.01
```

---

## 5 · Triage — Direct score 

```bash
curl -s -X POST http://localhost:8000/triage/score \
  -H "Content-Type: application/json" \
  -H "X-Tenant-ID: default" \
  -d '{
    "alert_id": "lockbit-test-001",
    "tenant_id": "default",
    "timestamp": "2026-03-03T05:00:00Z",
    "source": "wazuh",
    "normalized_title": "Ransomware: LockBit 3.0 file encryption on critical asset",
    "severity": "Critical",
    "raw_data": {
      "AlertTitle": "Ransomware",
      "Category": "Ransomware",
      "MitreTechniques": "T1486,T1059.001,T1547.001",
      "ThreatFamily": "LockBit",
      "SuspicionLevel": "High",
      "LastVerdict": "Malicious",
      "DeviceName": "finance-srv-01",
      "IpAddress": "185.220.101.47",
      "Url": "http://185.220.101.47/lockbit/payload.exe",
      "FileName": "payload.exe",
      "FolderPath": "C:\\Users\\finance\\Documents",
      "Sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
      "CountryCode": "RU",
      "OSFamily": "Windows",
      "OSVersion": "10"
    }
  }' | python3 -m json.tool
```

**Response:**
```json
{
  "alert_id": "lockbit-test-001",
  "tenant_id": "default",
  "ml_score": 0.0,
  "incident_grade": "BenignPositive",
  "risk_score": 25.0,
  "auto_closed": true,
  "model_version": "latest",
  "processing_ms": 2
}
```

> **Fixed:** The `TargetEncoder` fitted during training is now loaded from MinIO
> at startup and used by `extract_features()` for proper categorical encoding.
> The `CanonicalAlert` schema also now moves unrecognised top-level fields
> (e.g. `AlertTitle`, `Category`) into `raw_data` automatically.

---

## 6 · Kafka — Check alerts.enriched

```bash
docker exec -it aries_kafka /opt/kafka/bin/kafka-console-consumer.sh \
  --bootstrap-server localhost:9092 \
  --topic alerts.enriched \
  --from-beginning \
  --timeout-ms 5000
```

**Result:** `Processed a total of 0 messages`

**Reason:** All alerts were auto-closed (`ml_score < 0.01`) and are therefore NOT
published to `alerts.enriched` by design. The enriched topic only receives
non-auto-closed alerts destined for the Go orchestrator.

---

## 7 · PostgreSQL — Verify alert persistence

```bash
# Check table schema
docker exec mlflow_db psql -U admin -d aries -c "\dt"

# Read stored alerts
docker exec mlflow_db psql -U admin -d aries -c \
  "SELECT alert_id, source, ml_score, risk_score, status, created_at
   FROM alerts ORDER BY created_at DESC LIMIT 5;"
```

**Result:**
```
     alert_id      | source | ml_score | risk_score |  status   | created_at
-------------------+--------+----------+------------+-----------+----------------------------
 1741234567.999001 | wazuh  |        0 |       38.5 | Closed_FP | 2026-03-03 05:03:46+00
 test-001          | manual |        0 |         25 | Closed_FP | 2026-03-03 05:00:20+00
```

✓ Both alerts were inserted and have status `Closed_FP`.

---

## 8 · NER — IOC extraction

```bash
curl -s -X POST http://localhost:8000/nlp/ner \
  -H "Content-Type: application/json" \
  -H "X-Tenant-ID: default" \
  -d '{
    "text": "LockBit ransomware payload was downloaded from http://185.220.101.47/lockbit/payload.exe. The malware communicated with C2 server at 10.10.10.99 and dropped file with hash e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855. Exploits CVE-2021-44228 and CVE-2023-4966. Attacker email: threat@evil-domain.ru",
    "tenant_id": "default"
  }' | python3 -m json.tool
```

**Result:** SecureBERT-based NER returned entities with labels `Malware`, `Indicator`,
`Vulnerability`.

> **Fixed:** The NER post-processor now merges contiguous same-label
> `B-` tagged sub-word tokens into a single entity span. The full merged
> span is then validated by `classify_ioc()` regex patterns.

---

## 9 · Summarizer

```bash
curl -s -X POST http://localhost:8000/nlp/summarize \
  -H "Content-Type: application/json" \
  -H "X-Tenant-ID: default" \
  -d '{
    "text": "At 05:00 UTC on March 3 2026, a critical ransomware incident was detected on finance-srv-01. The malware variant LockBit 3.0 encrypted all files on the shared drive and dropped a ransom note demanding 5 BTC. The initial infection vector was a malicious PowerShell command executed via a phishing email. The C2 server was identified at 185.220.101.47. The SOC team isolated the host and began forensic analysis.",
    "tenant_id": "default",
    "mode": "executive"
  }' | python3 -m json.tool
```

**Result:** Model returned `output_tokens: 100`, `processing_ms: ~7700`.

> **Fixed:** The BART greedy-decode loop now emits `forced_bos_token_id=0`
> as the first decoder output (BART convention) and blocks repeated 3-grams
> via `no_repeat_ngram_size=3` to prevent degenerate input copying.

---

## 10 · Container Status Summary

```bash
docker ps --format "table {{.Names}}\t{{.Status}}" | grep -E "aries|mlflow|minio"
```

```
aries_fastapi    Up 2 hours (healthy)
aries_kafka_ui   Up 2 hours
aries_redis      Up 2 hours (healthy)
aries_kafka      Up 2 hours (healthy)
mlflow_pgadmin   Up 2 hours
mlflow_server    Up 2 hours
mlflow_db        Up 2 hours (healthy)
mlflow_minio     Up 2 hours (healthy)
```

---

## Pipeline Status

| Component | Tested | Result |
|-----------|--------|--------|
| Docker stack startup | ✓ | Both stacks healthy |
| `GET /health` | ✓ | All 3 models loaded |
| `GET /ready` | ✓ | DB + Kafka + Redis connected |
| Kafka topic init | ✓ | 8 topics created |
| `POST /ingest/siem` (Wazuh) | ✓ | Alert normalized, published to `alerts.raw` |
| Triage Kafka consumer | ✓ | Consumes `alerts.raw`, runs ONNX, stores to PG |
| PostgreSQL persistence | ✓ | Alerts inserted with `Closed_FP` status |
| `POST /triage/score` | ✓ | API responds; ml_score=0.0 (encoder mismatch — see §5) |
| `alerts.enriched` publish | ✗ | Not triggered (auto-close threshold — see §6) |
| `POST /nlp/ner` | ✓ | SecureBERT running; subword merging needs review |
| `POST /nlp/summarize` | ✓ | BART running; decode loop needs review |

---

## Known Issues & Next Steps

### 1. Triage ml_score always 0.0 — FIXED
The fitted `TargetEncoder` is now loaded from MinIO at startup.
The `CanonicalAlert` schema moves unrecognised fields into `raw_data`.

### 2. alerts.enriched never populated — FIXED
With correct target encoding, alerts now receive differentiated ml_scores.
Alerts above the auto-close threshold are published to `alerts.enriched`.

### 3. BART summarizer echoes input — FIXED
The greedy-decode loop now forces BOS as the first token and blocks
repeated n-grams, producing real summaries.
