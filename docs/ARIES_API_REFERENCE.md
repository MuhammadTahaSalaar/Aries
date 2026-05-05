# ARIES FastAPI Service — API Reference

## Connection Details

| Field | Value |
|-------|-------|
| **RunPod Proxy URL** | `https://756vdeq94fibsd-8000.proxy.runpod.net` |
| **Port** | 8000 |
| **Protocol** | HTTPS (RunPod proxy handles TLS) |
| **Interactive Docs** | `https://756vdeq94fibsd-8000.proxy.runpod.net/docs` |

> **Note**: The proxy URL changes every time the RunPod pod is restarted. Update it from the RunPod dashboard under **Connect → HTTP Service (Port 8000)**.

---

## Required Headers (all endpoints)

| Header | Required | Description |
|--------|----------|-------------|
| `Content-Type` | Yes (POST) | Must be `application/json` |
| `X-Tenant-ID` | Yes | Tenant isolation key. Use `demo` for testing; use your Wazuh agent group name in production. |

---

## Endpoints

### 1. Health Check

```
GET /health
```

No body, no auth required. Returns model load status.

**Response**

```json
{
  "status": "ok",
  "service": "aries-fastapi-service",
  "version": "1.0.0",
  "models_loaded": {
    "triage": true,
    "ner": true,
    "summarizer": true,
    "slm": true
  }
}
```

**Successful curl**

```bash
curl -s https://756vdeq94fibsd-8000.proxy.runpod.net/health | python3 -m json.tool
```

---

### 2. Triage — Score an Alert

```
POST /triage/score
```

Takes a **CanonicalAlert** and returns an ML risk score. All fields except `alert_id` are optional and have defaults — but the more context you provide, the better the SLM can reason.

**Request body (CanonicalAlert)**

| Field | Type | Required | Notes |
|-------|------|----------|-------|
| `alert_id` | string | No | UUID auto-generated if omitted |
| `tenant_id` | string | No | Overridden by `X-Tenant-ID` header |
| `source` | string | No | `"wazuh"`, `"splunk"`, `"crowdstrike"`, etc. |
| `normalized_title` | string | No | Human-readable alert title |
| `severity` | string | No | **Exact case**: `"Low"`, `"Medium"`, `"High"`, `"Critical"` |
| `raw_data` | object | No | Original vendor payload (pass-through) |
| `mitre_tactic` | string | No | e.g. `"Initial Access"` |
| `mitre_technique` | string | No | e.g. `"T1110.001"` |
| `entity_type` | string | No | `"Process"`, `"File"`, `"IP"`, etc. |
| `device_name` | string | No | Hostname / agent name |
| `ip_address` | string | No | Source IP |
| `user_name` | string | No | Affected username |
| `file_hash` | string | No | SHA256 hash |
| `url` | string | No | Related URL |
| `domain` | string | No | Related domain |
| `category` | string | No | Alert category / Wazuh rule group |
| `threat_family` | string | No | Malware family |

> **Severity enum is title-case**. Sending `"high"` returns a 422 validation error. Use `"High"`.

**Successful example — SSH brute-force**

```bash
curl -s -X POST https://756vdeq94fibsd-8000.proxy.runpod.net/triage/score \
  -H "Content-Type: application/json" \
  -H "X-Tenant-ID: demo" \
  -d '{
    "normalized_title": "SSH brute-force attempt",
    "severity": "High",
    "source": "wazuh",
    "ip_address": "192.168.1.50",
    "user_name": "root",
    "mitre_tactic": "Credential Access",
    "mitre_technique": "T1110.001",
    "category": "authentication_failed"
  }' | python3 -m json.tool
```

**Successful example — Malware detected**

```bash
curl -s -X POST https://756vdeq94fibsd-8000.proxy.runpod.net/triage/score \
  -H "Content-Type: application/json" \
  -H "X-Tenant-ID: demo" \
  -d '{
    "normalized_title": "Ransomware binary execution detected",
    "severity": "Critical",
    "source": "wazuh",
    "device_name": "WORKSTATION-42",
    "user_name": "jdoe",
    "file_hash": "a1b2c3d4e5f67890a1b2c3d4e5f67890a1b2c3d4e5f67890a1b2c3d4e5f67890",
    "mitre_tactic": "Execution",
    "mitre_technique": "T1059",
    "threat_family": "LockBit"
  }' | python3 -m json.tool
```

---

### 3. NER — Extract Entities and IOCs

```
POST /nlp/ner
```

Extracts IP addresses, domains, hashes, CVEs, malware names, and security events from free text.

**Request body**

| Field | Type | Required | Notes |
|-------|------|----------|-------|
| `text` | string | Yes | 1–50,000 chars |
| `tenant_id` | string | No | Overridden by `X-Tenant-ID` header |

**Successful example — log line**

```bash
curl -s -X POST https://756vdeq94fibsd-8000.proxy.runpod.net/nlp/ner \
  -H "Content-Type: application/json" \
  -H "X-Tenant-ID: demo" \
  -d '{"text": "Malware callback from 10.0.0.5 contacted evil.com:443 using CVE-2021-44228 (Log4Shell). SHA256: a1b2c3d4e5f67890a1b2c3d4e5f67890."}' \
  | python3 -m json.tool
```

**Successful example — Wazuh alert text**

```bash
curl -s -X POST https://756vdeq94fibsd-8000.proxy.runpod.net/nlp/ner \
  -H "Content-Type: application/json" \
  -H "X-Tenant-ID: demo" \
  -d '{"text": "CRIT sshd[1234]: Failed password for root from 203.0.113.50 port 22 ssh2. Repeated 47 times in 60 seconds from same source."}' \
  | python3 -m json.tool
```

---

### 4. Summarize — Generate Incident Summary

```
POST /nlp/summarize
```

Generates an executive (2-4 sentence) or analyst (8-15 sentence) summary of incident text.

**Request body**

| Field | Type | Required | Notes |
|-------|------|----------|-------|
| `text` | string | Yes | 10–100,000 chars |
| `mode` | string | No | `"executive"` (default) or `"analyst"` |
| `case_id` | string | No | Optional — links summary to a case for persistence |
| `tenant_id` | string | No | Overridden by `X-Tenant-ID` header |

**Successful example — executive summary**

```bash
curl -s -X POST https://756vdeq94fibsd-8000.proxy.runpod.net/nlp/summarize \
  -H "Content-Type: application/json" \
  -H "X-Tenant-ID: demo" \
  -d '{
    "text": "Multiple failed SSH login attempts detected from 203.0.113.50 targeting root account on web-server-01. Source IP is a known Tor exit node. 47 failed attempts in 60 seconds. Following failed authentication, a successful login occurred using a different credential at 02:14 UTC. Post-login activity included /etc/passwd read and wget of an unknown binary.",
    "mode": "executive"
  }' | python3 -m json.tool
```

**Successful example — analyst summary**

```bash
curl -s -X POST https://756vdeq94fibsd-8000.proxy.runpod.net/nlp/summarize \
  -H "Content-Type: application/json" \
  -H "X-Tenant-ID: demo" \
  -d '{
    "text": "Multiple failed SSH login attempts detected from 203.0.113.50 targeting root account on web-server-01. Source IP is a known Tor exit node. 47 failed attempts in 60 seconds. Following failed authentication, a successful login occurred using a different credential at 02:14 UTC. Post-login activity included /etc/passwd read and wget of an unknown binary.",
    "mode": "analyst"
  }' | python3 -m json.tool
```

---

### 5. SIEM Ingestion — Ingest Raw Wazuh/SIEM Alert

```
POST /ingest/siem?vendor=wazuh
```

Receives a raw vendor JSON payload, normalises it to the Canonical Alert schema, deduplicates via Redis (if available), and publishes to Kafka `alerts.raw` (if available). Returns the normalised canonical form.

The vendor can be passed as:
- `?vendor=wazuh` query parameter (recommended for webhook integration)
- `"vendor": "wazuh"` field in the request body

Supported vendors: `wazuh`, `splunk`, `elastic_siem`, `crowdstrike`

**Wazuh field mapping** (how raw Wazuh JSON is normalised)

| Canonical field | Wazuh JSON path |
|----------------|-----------------|
| `alert_id` | `id` |
| `normalized_title` | `rule.description` |
| `severity` | `rule.level` → mapped: 0-3→Low, 4-7→Medium, 8-11→High, 12-15→Critical |
| `mitre_tactic` | `rule.mitre.tactic` |
| `mitre_technique` | `rule.mitre.id` |
| `ip_address` | `data.srcip` |
| `user_name` | `data.srcuser` |
| `device_name` | `agent.name` |
| `category` | `rule.groups` |
| `timestamp` | `timestamp` |

**Successful example — Wazuh SSH brute-force alert (raw Wazuh JSON)**

```bash
curl -s -X POST "https://756vdeq94fibsd-8000.proxy.runpod.net/ingest/siem?vendor=wazuh" \
  -H "Content-Type: application/json" \
  -H "X-Tenant-ID: demo" \
  -d '{
    "id": "1716854400.123456",
    "timestamp": "2026-05-05T02:00:00.000Z",
    "rule": {
      "level": 10,
      "description": "Multiple failed SSH logins (brute force)",
      "id": "5712",
      "mitre": {
        "tactic": ["Credential Access"],
        "id": ["T1110.001"]
      },
      "groups": ["authentication_failed", "ssh"]
    },
    "agent": {
      "id": "001",
      "name": "web-server-01",
      "ip": "10.0.1.10"
    },
    "data": {
      "srcip": "203.0.113.50",
      "srcuser": "root",
      "srcport": "54321"
    },
    "full_log": "May  5 02:00:00 web-server-01 sshd[1234]: Failed password for root from 203.0.113.50 port 22 ssh2"
  }' | python3 -m json.tool
```

**Successful example — Wazuh malware/FIM alert**

```bash
curl -s -X POST "https://756vdeq94fibsd-8000.proxy.runpod.net/ingest/siem?vendor=wazuh" \
  -H "Content-Type: application/json" \
  -H "X-Tenant-ID: demo" \
  -d '{
    "id": "1716854500.654321",
    "timestamp": "2026-05-05T02:01:00.000Z",
    "rule": {
      "level": 12,
      "description": "File integrity monitoring: suspicious binary added to /tmp",
      "id": "550",
      "mitre": {
        "tactic": ["Persistence"],
        "id": ["T1574"]
      },
      "groups": ["fim", "syscheck"]
    },
    "agent": {
      "id": "002",
      "name": "workstation-42",
      "ip": "10.0.1.42"
    },
    "data": {
      "path": "/tmp/update_agent",
      "sha256_after": "a1b2c3d4e5f67890a1b2c3d4e5f67890a1b2c3d4e5f67890a1b2c3d4e5f67890",
      "event": "added"
    }
  }' | python3 -m json.tool
```

---

## Wazuh SOAR Integration

To forward all Wazuh alerts to ARIES automatically, configure a **custom active-response integration** in Wazuh. The `wazuh-integration/custom-aries.py` script in this repo does exactly this. It:
1. Receives a Wazuh alert JSON from the active-response socket
2. POSTs it to `/ingest/siem?vendor=wazuh`
3. Calls `/triage/score` with the normalised alert if needed

**Wazuh `ossec.conf` integration block** (add to manager's `ossec.conf`):

```xml
<integration>
  <name>custom-aries</name>
  <hook_url>https://756vdeq94fibsd-8000.proxy.runpod.net</hook_url>
  <level>7</level>
  <alert_format>json</alert_format>
</integration>
```

This triggers ARIES for all alerts at Wazuh level 7 and above.

---

## Common Mistakes to Avoid

| Mistake | Correct Usage |
|---------|---------------|
| `"severity": "high"` | `"severity": "High"` (title-case enum) |
| `"severity": "CRITICAL"` | `"severity": "Critical"` |
| POST to `/triage/predict` | POST to `/triage/score` |
| POST to `/ner/extract` | POST to `/nlp/ner` |
| Missing `Content-Type` header | Always include `-H "Content-Type: application/json"` |
| Missing `X-Tenant-ID` header | Always include `-H "X-Tenant-ID: <tenant>"` |
| Sending Wazuh rule level as `"severity"` in `/triage/score` | Use `/ingest/siem?vendor=wazuh` which auto-maps level→severity; or set severity manually to `"High"` / `"Critical"` |
| Sending log lines (not JSON) to `/triage/score` | Send log text to `/nlp/ner` or `/nlp/summarize`; `/triage/score` expects structured JSON |

---

## Batch NER

```
POST /nlp/ner/batch
```

Accepts up to 32 texts in one call — useful for processing a burst of Wazuh log lines.

```bash
curl -s -X POST https://756vdeq94fibsd-8000.proxy.runpod.net/nlp/ner/batch \
  -H "Content-Type: application/json" \
  -H "X-Tenant-ID: demo" \
  -d '{
    "texts": [
      "Failed password for root from 203.0.113.50 port 22",
      "Outbound connection to known C2 domain evil-c2.ru:8080",
      "Privilege escalation detected: sudo su - root by user jdoe"
    ]
  }' | python3 -m json.tool
```
