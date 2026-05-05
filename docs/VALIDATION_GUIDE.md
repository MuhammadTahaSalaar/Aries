# ARIES SOAR — Validation & Testing Guide

Complete commands to start, test, and validate the entire ARIES pipeline:
Wazuh → Kafka → AI (Triage + NER + Summarizer) → Database.

---

## Table of Contents

1. [Starting the Platform](#1-starting-the-platform)
2. [Health Checks](#2-health-checks)
3. [Direct Endpoint Testing](#3-direct-endpoint-testing)
4. [Attack Scenario Injection](#4-attack-scenario-injection)
5. [Pipeline Verification](#5-pipeline-verification)
6. [Triggering Real Wazuh Alerts](#6-triggering-real-wazuh-alerts)
7. [Useful Debuging Commands](#7-useful-debugging-commands)

---

## 1. Starting the Platform

All three Docker Compose stacks must be running (11 containers total).

```bash
# 1. MLOps stack (PostgreSQL, MinIO, MLflow, PgAdmin)
cd ~/data/FYP/Aries/MLOps
docker compose up -d

# 2. FastAPI stack (Kafka, Redis, FastAPI service)
cd ~/data/FYP/Aries/apps/fastapi_service
docker compose up -d

# 3. Wazuh stack (Manager, Indexer, Dashboard)
cd ~/data/FYP/Aries/wazuh-docker/single-node
docker compose up -d
```
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

### Verify all containers are running

```bash
docker ps --format "table {{.Names}}\t{{.Status}}" | sort
```

Expected (11 containers):
| Container | Status |
|-----------|--------|
| `aries_fastapi` | Up (healthy) |
| `aries_kafka` | Up (healthy) |
| `aries_kafka_ui` | Up |
| `aries_redis` | Up (healthy) |
| `mlflow_db` | Up (healthy) |
| `mlflow_minio` | Up (healthy) |
| `mlflow_pgadmin` | Up |
| `mlflow_server` | Up |
| `single-node-wazuh.dashboard-1` | Up |
| `single-node-wazuh.indexer-1` | Up |
| `single-node-wazuh.manager-1` | Up |

---

## 2. Health Checks

```bash
# FastAPI overall health
curl -s http://localhost:8000/health | python3 -m json.tool

# Individual pipeline health
curl -s http://localhost:8000/triage/health | python3 -m json.tool
curl -s http://localhost:8000/nlp/ner/health | python3 -m json.tool
curl -s http://localhost:8000/nlp/summarize/health | python3 -m json.tool

# Kafka UI (browser)
# http://localhost:9090

# Wazuh Dashboard (browser)
# https://localhost:443  (user: admin, password: SecretPassword)

# MLflow UI (browser)
# http://localhost:5000
```

---

## 3. Direct Endpoint Testing

### 3.1 Triage Scoring

Scores an alert and returns ML probability, risk score, and incident grade.

```bash
# Critical alert on a domain controller
curl -s -X POST http://localhost:8000/triage/score \
  -H "Content-Type: application/json" \
  -H "X-Tenant-ID: default" \
  -d '{
    "normalized_title": "Brute force SSH login on domain controller",
    "severity": "Critical",
    "category": "authentication",
    "mitre_tactic": "Credential Access",
    "mitre_technique": "T1110",
    "source": "wazuh",
    "suspicion_level": "High"
  }' | python3 -m json.tool
```

```bash
# Low-severity routine event (should score much lower)
curl -s -X POST http://localhost:8000/triage/score \
  -H "Content-Type: application/json" \
  -H "X-Tenant-ID: default" \
  -d '{
    "normalized_title": "Log rotation completed",
    "severity": "Low",
    "category": "system",
    "source": "wazuh"
  }' | python3 -m json.tool
```

**Key fields in response:**
- `ml_score`: XGBoost true-positive probability (0-1)
- `risk_score`: Composite score (0-100) from ML + asset criticality + behavioral analysis
- `incident_grade`: `TruePositive`, `BenignPositive`, or `FalsePositive`
- `asset_criticality`: How critical the targeted asset is (0-1)
- `behavioral_score`: How suspicious the behavior pattern is (0-1)

### 3.2 NER (Named Entity Recognition)

Extracts cybersecurity entities (IPs, CVEs, malware, organizations) and security events from text.

```bash
curl -s -X POST http://localhost:8000/nlp/ner \
  -H "Content-Type: application/json" \
  -H "X-Tenant-ID: default" \
  -d '{
    "text": "The APT group Lazarus deployed TrickBot malware targeting Windows 10 systems at Acme Corp. The attack exploited CVE-2021-44228 (Log4Shell) to gain initial access. C2 traffic observed connecting to 185.220.101.42. File hash a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2 was found."
  }' | python3 -m json.tool
```

**Key fields in response:**
- `entities[]`: Each with `text`, `label` (Malware/Indicator/Vulnerability/System/Organization), `ioc_type`, `ioc_validated`, `confidence`
- `events[]`: Security events like `Patch-Vulnerability`, `Discover-Vulnerability`
- `cached`: Whether result came from Redis cache

### 3.3 Summarization

Generates a summary of a security incident report.

```bash
# Executive mode (short, 1-3 sentences)
curl -s -X POST http://localhost:8000/nlp/summarize \
  -H "Content-Type: application/json" \
  -H "X-Tenant-ID: default" \
  -d '{
    "text": "On March 15, 2024, the SOC detected a multi-stage attack. A spear-phishing email exploited CVE-2023-36884 to deploy malware. The attacker used PowerShell to download payloads from 203.0.113.50, established persistence via scheduled tasks, moved laterally using stolen credentials, accessed the domain controller, and exfiltrated 2.3GB of financial data. The attack was attributed to FIN7."
  }' | python3 -m json.tool

# Analyst mode (detailed, full analysis — takes ~100s)
curl -s --max-time 180 -X POST http://localhost:8000/nlp/summarize \
  -H "Content-Type: application/json" \
  -H "X-Tenant-ID: default" \
  -d '{
    "text": "On March 15, 2024, the SOC detected a multi-stage attack...",
    "mode": "analyst"
  }' | python3 -m json.tool
```

---

## 4. Attack Scenario Injection

These inject alerts through the full pipeline: Ingestion → Kafka → Triage → DB → alerts.enriched.

### Scenario 1: SSH Brute Force on Domain Controller

```bash
curl -s -X POST 'http://localhost:8000/ingest/siem?vendor=wazuh' \
  -H 'Content-Type: application/json' \
  -H 'X-Tenant-ID: default' \
  -d @- <<'EOF' | python3 -m json.tool
{
  "vendor": "wazuh",
  "raw": {
    "timestamp": "2026-03-30T15:00:00+0000",
    "rule": {
      "level": 10,
      "description": "sshd: Multiple authentication failures on domain controller.",
      "id": "5763",
      "mitre": {
        "id": ["T1110"],
        "tactic": ["Credential Access"],
        "technique": ["Brute Force"]
      },
      "groups": ["syslog", "sshd", "authentication_failures"]
    },
    "agent": {"id": "001", "name": "dc-prod-01", "ip": "10.0.0.10"},
    "data": {"srcip": "185.220.101.42", "dstuser": "administrator"},
    "location": "/var/log/auth.log",
    "full_log": "dc-prod-01 sshd: Failed password for administrator from 185.220.101.42 port 22 ssh2"
  }
}
EOF
```

### Scenario 2: Privilege Escalation via Sudo

```bash
curl -s -X POST 'http://localhost:8000/ingest/siem?vendor=wazuh' \
  -H 'Content-Type: application/json' \
  -H 'X-Tenant-ID: default' \
  -d @- <<'EOF' | python3 -m json.tool
{
  "vendor": "wazuh",
  "raw": {
    "timestamp": "2026-03-30T15:01:00+0000",
    "rule": {
      "level": 12,
      "description": "Unauthorized root access attempt via sudo on database server",
      "id": "5403",
      "mitre": {
        "id": ["T1548"],
        "tactic": ["Privilege Escalation"],
        "technique": ["Abuse Elevation Control Mechanism"]
      },
      "groups": ["syslog", "sudo", "privilege_escalation"]
    },
    "agent": {"id": "003", "name": "db-prod-01", "ip": "10.0.0.30"},
    "data": {"srcuser": "www-data", "dstuser": "root", "command": "/bin/bash"},
    "location": "/var/log/auth.log",
    "full_log": "db-prod-01 sudo: www-data : user NOT in sudoers"
  }
}
EOF
```

### Scenario 3: SQL Injection

```bash
curl -s -X POST 'http://localhost:8000/ingest/siem?vendor=wazuh' \
  -H 'Content-Type: application/json' \
  -H 'X-Tenant-ID: default' \
  -d @- <<'EOF' | python3 -m json.tool
{
  "vendor": "wazuh",
  "raw": {
    "timestamp": "2026-03-30T15:02:00+0000",
    "rule": {
      "level": 15,
      "description": "SQL injection attempt detected on web application",
      "id": "31103",
      "mitre": {
        "id": ["T1190"],
        "tactic": ["Initial Access"],
        "technique": ["Exploit Public-Facing Application"]
      },
      "groups": ["web", "attack", "sql_injection"]
    },
    "agent": {"id": "002", "name": "web-prod-01", "ip": "10.0.0.20"},
    "data": {"srcip": "203.0.113.50", "url": "/api/users?id=1 OR 1=1 --", "method": "GET"},
    "location": "/var/log/nginx/access.log",
    "full_log": "203.0.113.50 GET /api/users?id=1 OR 1=1 -- HTTP/1.1 200"
  }
}
EOF
```

### Scenario 4: Lateral Movement (Pass-the-Hash)

```bash
curl -s -X POST 'http://localhost:8000/ingest/siem?vendor=wazuh' \
  -H 'Content-Type: application/json' \
  -H 'X-Tenant-ID: default' \
  -d @- <<'EOF' | python3 -m json.tool
{
  "vendor": "wazuh",
  "raw": {
    "timestamp": "2026-03-30T15:03:00+0000",
    "rule": {
      "level": 14,
      "description": "Pass-the-hash lateral movement targeting Active Directory domain controller",
      "id": "92201",
      "mitre": {
        "id": ["T1550.002"],
        "tactic": ["Lateral Movement"],
        "technique": ["Pass the Hash"]
      },
      "groups": ["windows", "authentication", "lateral_movement"]
    },
    "agent": {"id": "004", "name": "ad-srv-01", "ip": "10.0.0.40"},
    "data": {"srcip": "10.0.0.100", "dstuser": "DOMAIN\\\\admin", "logon_type": "3"},
    "location": "WinEvtLog",
    "full_log": "EventID 4624: Logon Type 3 Source: 10.0.0.100 Target: DOMAIN admin"
  }
}
EOF
```

### Scenario 5: Data Exfiltration

```bash
curl -s -X POST 'http://localhost:8000/ingest/siem?vendor=wazuh' \
  -H 'Content-Type: application/json' \
  -H 'X-Tenant-ID: default' \
  -d @- <<'EOF' | python3 -m json.tool
{
  "vendor": "wazuh",
  "raw": {
    "timestamp": "2026-03-30T15:05:00+0000",
    "rule": {
      "level": 15,
      "description": "Large outbound data transfer to external IP from backup server",
      "id": "100003",
      "mitre": {
        "id": ["T1041"],
        "tactic": ["Exfiltration"],
        "technique": ["Exfiltration Over C2 Channel"]
      },
      "groups": ["network", "data_exfiltration"]
    },
    "agent": {"id": "005", "name": "backup-srv-01", "ip": "10.0.0.50"},
    "data": {"srcip": "10.0.0.50", "dstip": "198.51.100.99", "bytes_out": "2415919104", "protocol": "https"},
    "location": "network_monitor",
    "full_log": "Large outbound transfer: 2.3GB to 198.51.100.99 over HTTPS from backup-srv-01"
  }
}
EOF
```

### Expected Results

After ingestion, check logs for triage processing:

```bash
docker logs aries_fastapi --tail 20 2>&1 | grep triage_inference_complete
```

Each alert should show:
- `accepted: true` from ingestion endpoint
- `triage_inference_complete` log with `risk_score` and `grade`
- `enriched_alert_published` to `alerts.enriched` topic

---

## 5. Pipeline Verification

### Check Kafka Topics

```bash
# List all topics
docker exec aries_kafka kafka-topics.sh --bootstrap-server localhost:9092 --list

# Count messages on alerts.raw
docker exec aries_kafka kafka-console-consumer.sh \
  --bootstrap-server localhost:9092 \
  --topic alerts.raw \
  --from-beginning \
  --timeout-ms 3000 2>/dev/null | wc -l

# Count messages on alerts.enriched
docker exec aries_kafka kafka-console-consumer.sh \
  --bootstrap-server localhost:9092 \
  --topic alerts.enriched \
  --from-beginning \
  --timeout-ms 3000 2>/dev/null | wc -l

# Read latest enriched alert
docker exec aries_kafka kafka-console-consumer.sh \
  --bootstrap-server localhost:9092 \
  --topic alerts.enriched \
  --from-beginning \
  --timeout-ms 3000 2>/dev/null | tail -1 | python3 -m json.tool
```

### Check Redis Cache

```bash
# Count cached keys
docker exec aries_redis redis-cli DBSIZE

# List all keys
docker exec aries_redis redis-cli KEYS '*'

# Clear cache (for fresh testing)
docker exec aries_redis redis-cli FLUSHALL
```

### Check FastAPI Logs

```bash
# Recent logs
docker logs aries_fastapi --tail 50

# Filter for specific events
docker logs aries_fastapi 2>&1 | grep "triage_inference_complete"
docker logs aries_fastapi 2>&1 | grep "enriched_alert_published"
docker logs aries_fastapi 2>&1 | grep "error\|ERROR\|exception"
```

### View Kafka UI

Open http://localhost:9090 in your browser to see:
- Topic `alerts.raw` — raw ingested alerts
- Topic `alerts.enriched` — triage-scored alerts
- Consumer group `ml-triage-engine` lag

---

## 6. Triggering Real Wazuh Alerts

To generate alerts that flow from Wazuh through the full integration:

### Prerequisites

1. Install the custom integration on the Wazuh manager:

```bash
# Copy integration files to Wazuh manager container
docker cp wazuh-integration/custom-aries single-node-wazuh.manager-1:/var/ossec/integrations/custom-aries
docker cp wazuh-integration/custom-aries.py single-node-wazuh.manager-1:/var/ossec/integrations/custom-aries.py

# Set permissions
docker exec single-node-wazuh.manager-1 chmod 750 /var/ossec/integrations/custom-aries
docker exec single-node-wazuh.manager-1 chmod 750 /var/ossec/integrations/custom-aries.py
docker exec single-node-wazuh.manager-1 chown root:wazuh /var/ossec/integrations/custom-aries
docker exec single-node-wazuh.manager-1 chown root:wazuh /var/ossec/integrations/custom-aries.py
```

2. The `wazuh_manager.conf` already has the integration configured (level ≥ 3).
   It also monitors `/var/log/auth.log`, which is the quickest way to trigger SSH authentication rules during validation.

### Trigger SSH Brute Force (on Wazuh manager itself)

```bash
# Inject failed SSH log lines into auth.log (triggers sshd authentication rules)
docker exec single-node-wazuh.manager-1 bash -lc '
for i in $(seq 1 8); do
  echo "Mar 30 15:00:0${i} wazuh-manager sshd[12345]: Failed password for invalid user baduser from 185.220.101.42 port 22 ssh2" >> /var/log/auth.log
done
'
```

### Trigger File Integrity Alert

```bash
# Modify a monitored file (triggers rules 550/554)
docker exec single-node-wazuh.manager-1 bash -c 'echo "# test" >> /etc/hosts'
```

### Verify Alerts in Wazuh

```bash
# Check the latest Wazuh alert payload
docker exec single-node-wazuh.manager-1 tail -1 /var/ossec/logs/alerts/alerts.json | python3 -m json.tool

# Check ARIES forwarding in the manager log
docker exec single-node-wazuh.manager-1 bash -lc "grep -E 'custom-aries|integratord' /var/ossec/logs/ossec.log | tail -20"
```

---

## 7. Useful Debugging Commands

### Rebuild FastAPI Service

```bash
cd ~/data/FYP/Aries/apps/fastapi_service
docker compose up -d --build
```

### Check Model Loading

```bash
docker logs aries_fastapi 2>&1 | grep "model_loaded\|service_ready"
```

### Check Kafka Consumer Health

```bash
docker logs aries_fastapi 2>&1 | grep "kafka_consumer"
```

### View API Documentation

Open http://localhost:8000/docs for the Swagger UI with all endpoints.

### Stop Everything

```bash
cd ~/data/FYP/Aries/wazuh-docker/single-node && docker compose down
cd ~/data/FYP/Aries/apps/fastapi_service && docker compose down
cd ~/data/FYP/Aries/MLOps && docker compose down
```

---

## Architecture Flow

```
┌─────────────────┐      integratord       ┌──────────────────────┐
│  Wazuh Manager   │ ─── (level ≥ 3) ────→ │  custom-aries.py     │
│  Alerts Engine   │     alert JSON         │  POST /ingest/siem   │
└─────────────────┘                         └──────────┬───────────┘
                                                       │
                    ┌──────────────────────────────────┘
                    ▼
┌──────────────────────────┐    Kafka: alerts.raw    ┌─────────────────────────┐
│  Ingestion Router        │ ──────────────────────→ │  Triage Kafka Consumer  │
│  • Vendor normalization  │                         │  • XGBoost ML scoring   │
│  • Redis deduplication   │                         │  • Risk computation     │
│  • Schema validation     │                         │  • Context enrichment   │
└──────────────────────────┘                         └───────────┬─────────────┘
                                                                 │
                                                    Kafka: alerts.enriched
                                                                 │
                                                                 ▼
                                                    ┌─────────────────────────┐
                                                    │  PostgreSQL + Dashboard │
                                                    │  NER / Summarizer       │
                                                    │  (on-demand via API)    │
                                                    └─────────────────────────┘
```

### AI Endpoints (direct API calls)

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/triage/score` | POST | Score a single alert (ML + risk) |
| `/nlp/ner` | POST | Extract entities & IOCs from text |
| `/nlp/summarize` | POST | Summarize incident reports |
| `/ingest/siem?vendor=wazuh` | POST | Ingest a SIEM alert (full pipeline) |
| `/health` | GET | Service health check |
