# ARIES SOAR — Master AI & SLM Integration Report

This document serves as the comprehensive master report for the AI infrastructure of the ARIES (AI-Enhanced Security Orchestration, Automation, and Response) platform. It details the theoretical foundations, architectural design, fine-tuning methodologies, empirical results, and deployment workflows for the newly integrated Small Language Model (SLM) pipeline. 

This guide is structured to provide both a high-level overview for stakeholders and deep technical execution steps for engineers, making it suitable for final academic and technical reporting.

---

## 1. System Architecture

The following diagram illustrates the end-to-end data flow of an alert traversing the ARIES platform, highlighting the transition from raw SIEM ingestion to SLM-driven semantic reasoning, and finally to orchestration.

```mermaid
graph TD
    subgraph Ingestion Layer
        sensors[Network Sensors / Endpoints] -->|Syslog / API| wazuh[Wazuh Manager]
        wazuh -->|Custom Integration Script| fastapi_ingest[FastAPI POST /ingest/siem]
    end
    
    subgraph Message Broker Layer
        fastapi_ingest -->|Publish JSON| kafka_raw[(Kafka: alerts.raw)]
        enrichment -->|Publish Enriched| kafka_enriched[(Kafka: alerts.enriched)]
    end
    
    subgraph AI Inference Service aries_fastapi
        kafka_raw -->|Consume| consumer[TriageKafkaConsumer]
        
        consumer -->|Check ARIES_USE_SLM| router{Use SLM?}
        
        router -->|No| xgboost[Legacy XGBoost + Target Encoder]
        router -->|Yes| slm[SLM Inference Engine<br/>INT4 GGUF via llama.cpp CPU]
        
        subgraph Semantic Tasks
            slm -.-> triage_logic[Semantic Triage<br/>Identify TP/FP/BP]
            slm -.-> ner_logic[Generative NER<br/>Extract STIX 2.1 IOCs]
            slm -.-> sum_logic[Fact-Grounded Summarization]
        end
        
        triage_logic --> enrichment[Enrichment & Scoring<br/>+ Asset Criticality<br/>+ Behavioral Score]
    end
    
    subgraph Persistence & Orchestration
        enrichment -->|Persist State| pgsql[(PostgreSQL: aries DB)]
        kafka_enriched --> orchestration[Go Orchestrator / Playbooks]
        orchestration --> minio[MLflow / MinIO Artifacts]
    end

    classDef infrastructure fill:#f9f9f9,stroke:#333,stroke-width:2px;
    classDef ai fill:#e1f5fe,stroke:#0288d1,stroke-width:2px;
    classDef broker fill:#fff3e0,stroke:#0288d1,stroke-width:2px;
    
    class Ingestion Layer,Persistence & Orchestration infrastructure;
    class AI Inference Service aries_fastapi ai;
    class Message Broker Layer broker;
```

---

## 2. The Paradigm Shift: Legacy vs. Semantic AI

ARIES initially utilized isolated, task-specific machine learning models. The transition to a unified Small Language Model (SLM) addresses critical limitations discovered during cross-SIEM integration.

### 2.1 Limitations of the Legacy Stack
*   **XGBoost (Triage)**: Relied heavily on Target Encoding trained on the Microsoft GUIDE dataset. When ARIES ingested alerts from new SIEMs (e.g., Wazuh), unseen categorical values (like specific `rule.description` strings) collapsed to benign priors. This out-of-distribution (OOD) failure degraded the detection of critical alerts.
*   **BART-base (Summarization)**: As a standard sequence-to-sequence model without instruction tuning, it suffered from severe hallucination, often generating fabricated IP addresses or CVEs when trying to complete sentences.
*   **SecureBERT (NER)**: While accurate, maintaining a separate PyTorch transformer stack exclusively for token classification was computationally heavy and memory-inefficient.

### 2.2 The Unified SLM Solution
The new **Semantic Architecture** replaces these disparate models with a single, instruction-tuned Small Language Model (e.g., Phi-3-mini or Llama-3-8B). 
*   **Decoupled Schema**: Instead of relying on strict categorical encoding, the SLM parses the raw JSON and narrative text of an alert directly using zero-shot and few-shot prompt engineering. It reads the alert exactly as a human SOC analyst would.
*   **Unified Memory Footprint**: A single quantized model handles classification, extraction, and generation tasks, drastically reducing the system's memory overhead.

---

## 3. AI Sub-Pipelines & Prompt Engineering

### 3.1 Semantic Triage (`src/triage/slm_inference.py`)
The SLM evaluates the alert payload and outputs a structured JSON response dictating the `incident_grade` (TruePositive, FalsePositive, BenignPositive) and a confidence metric. 
*   **Contextual Scoring**: The pipeline supplements the ML output with an `asset_criticality` score (derived from entity types like "Domain Controller") and a `behavioral_score` (derived from MITRE tactics like "Lateral Movement"). These combine to form the final `risk_score`.

### 3.2 Generative NER (`src/nlp/ner/slm_inference.py`)
The model is prompted to extract cybersecurity entities aligned with the STIX 2.1 standard (Malware, Indicators, Systems, Vulnerabilities) directly from unstructured text.
*   **Safety Net**: To ensure reliability, the legacy regex logic remains active as a post-processing step to validate and strictly format extracted IPs, hashes, and domains, bridging the gap between generative AI and deterministic security requirements.

### 3.3 Fact-Grounded Summarization (`src/nlp/summarizer/slm_inference.py`)
The SLM generates executive (1-3 sentences) and analyst (detailed) summaries. By utilizing strict system prompts instructing the model to *only* use provided facts, the hallucination issues present in the legacy BART model have been eliminated.

---

## 4. Fine-Tuning Methodology (Hydra HPC)

Training large language models requires significant GPU resources, which are handled via the VUB Hydra HPC cluster.

### 4.1 Datasets Utilized
*   **Triage**: Augmented GUIDE dataset (~13.6M rows), stratified and sampled to prevent class imbalance priors.
*   **NER**: CyberNER (2024/2025) - A STIX 2.1 harmonized dataset combining CyNER and APTNER.
*   **Summarization**: S-RM 2025 Cyber Incident Insights / GovReport.

### 4.2 Training Mechanics: QLoRA & PEFT
Full parameter fine-tuning of an 8B parameter model is computationally prohibitive. ARIES utilizes **QLoRA (Quantized Low-Rank Adaptation)**:
1.  The base model weights are frozen and quantized to 4-bit precision.
2.  Small, trainable "adapter" matrices are injected into the attention layers.
3.  This reduces VRAM requirements from ~32GB to ~8GB during training while maintaining near 100% of the model's performance capability.

### 4.3 Deployment Optimization: INT4 GGUF
Post-training, the LoRA adapters are merged with the base model, and the entire network is quantized to the **INT4 GGUF** format using `llama.cpp`. This compresses the model from ~16GB down to ~4.5GB, enabling it to run entirely on CPU RAM in the production environment.

---

## 5. Empirical Results

**Extracted from HPC Training Output (`successful_outputs/aries_slm_12436601.out`):**
The training logs demonstrate the SLM successfully converging over multiple epochs, achieving excellent token accuracy and low evaluation loss without overfitting.

```text
...
{'loss': '0.03221', 'grad_norm': '0.03011', 'learning_rate': '0.0000415', 'entropy': '0.03348', 'num_tokens': '2.423e+07', 'mean_token_accuracy': '0.9898', 'epoch': '1.441'}          
{'loss': '0.03185', 'grad_norm': '0.02864', 'learning_rate': '0.0000414', 'entropy': '0.03315', 'num_tokens': '2.426e+07', 'mean_token_accuracy': '0.9899', 'epoch': '1.447'}          
{'loss': '0.03172', 'grad_norm': '0.02693', 'learning_rate': '0.0000412', 'entropy': '0.03291', 'num_tokens': '2.429e+07', 'mean_token_accuracy': '0.9901', 'epoch': '1.453'}          
{'eval_loss': '0.03154', 'eval_runtime': '90.71', 'eval_samples_per_second': '55.15', 'eval_steps_per_second': '13.78', 'eval_entropy': '0.03285', 'eval_num_tokens': '2.429e+07', 'eval_mean_token_accuracy': '0.9899', 'epoch': '1.453'}                                                                                                                          
Early stopping triggered. Eval loss improved by 0.00046, which is less than the required threshold of 0.001 for 1 consecutive evaluation step.
Saving LoRA adapters to /home/user/project/models/slm_lora/triage...
Done!
```

---

## 6. Production Execution Logs (Docker)

In the local production stack, the SLM executes using CPU RAM via `llama.cpp` bindings in Python. The logs below prove the successful loading of the INT4 GGUF model, ingestion of a Wazuh alert from Kafka, and complete triage processing via the SLM.

**Extracted from `aries_fastapi` container logs:**
```text
2026-05-04 08:13:53 [info     ] pre_warming_slm                component=main path=/tmp/aries_models/slm/triage_slm_q4.gguf
2026-05-04 08:13:53 [info     ] Loading SLM model              component=triage_slm_inference n_threads=8 path=/tmp/aries_models/slm/triage_slm_q4.gguf
llama_context: n_ctx_seq (2048) < n_ctx_train (4096) -- the full capacity of the model will not be utilized
2026-05-04 08:14:03 [info     ] slm_pre_warm_complete          component=main
2026-05-04 08:14:03 [info     ] kafka_producer_started         bootstrap=kafka:9092 component=kafka
2026-05-04 08:14:03 [info     ] triage_consumer_started        component=main
2026-05-04 08:14:03 [info     ] service_ready                  component=main models={'triage': True, 'ner': True, 'summarizer': True, 'slm': True}
INFO:     Application startup complete.

# Alert Ingestion and Kafka Routing
2026-05-04 08:14:34 [info     ] alert_normalized               alert_id=0bf147f0-7baf-4bee-b7c8-065fe825adb2 component=ingestion severity=High tenant_id=default vendor=wazuh
2026-05-04 08:14:34 [info     ] alert_published_to_kafka       alert_id=0bf147f0-7baf-4bee-b7c8-065fe825adb2 component=ingestion_router tenant_id=default topic=alerts.raw
2026-05-04 08:14:34 [info     ] triage_processing_alert        alert_id=0bf147f0-7baf-4bee-b7c8-065fe825adb2 component=triage_consumer source=wazuh tenant_id=default

# SLM Inference Completion (Latency reflects local CPU processing time)
2026-05-04 08:19:02 [info     ] triage_slm_inference_complete  alert_id=0bf147f0-7baf-4bee-b7c8-065fe825adb2 auto_closed=False component=triage_slm_inference grade=TruePositive latency_ms=265312 ml_score=0.95 risk_score=83.5 tenant_id=default
2026-05-04 08:19:02 [info     ] enriched_alert_published       alert_id=0bf147f0-7baf-4bee-b7c8-065fe825adb2 component=triage_consumer risk_score=83.5
```

---

## 7. Integration and Execution Commands

To start the platform and validate the pipeline, use the following deployment sequence:

### Step 1: Start the Platform Stacks
All three Docker Compose stacks must be running to establish the network and dependencies.

```bash
# 1. MLOps stack (PostgreSQL, MinIO, MLflow, PgAdmin)
cd /data/FYP/Aries/MLOps
docker compose up -d

# 2. FastAPI AI stack (Kafka, Redis, FastAPI service)
cd /data/FYP/Aries/apps/fastapi_service
docker compose up -d --build

# 3. Wazuh SIEM stack (Manager, Indexer, Dashboard)
cd /data/FYP/Aries/wazuh-docker/single-node
docker compose up -d
```

### Step 2: Configure Wazuh Integration
The Wazuh manager container requires the custom Python integration scripts. Because of permission constraints in `integratord`, they must be copied into the running container and secured.

```bash
cd /data/FYP/Aries
docker cp wazuh-integration/custom-aries single-node-wazuh.manager-1:/var/ossec/integrations/custom-aries
docker cp wazuh-integration/custom-aries.py single-node-wazuh.manager-1:/var/ossec/integrations/custom-aries.py

# Fix permissions for integratord
docker exec single-node-wazuh.manager-1 bash -c '
  chmod 750 /var/ossec/integrations/custom-aries
  chmod 750 /var/ossec/integrations/custom-aries.py
  chown root:wazuh /var/ossec/integrations/custom-aries
  chown root:wazuh /var/ossec/integrations/custom-aries.py
'

# Restart Wazuh Manager
docker exec single-node-wazuh.manager-1 /var/ossec/bin/wazuh-control restart
```

### Step 3: Trigger a Simulated Attack for SLM Validation
Inject a raw SIEM alert to test the full pipeline (Wazuh Ingestion → Kafka `alerts.raw` → SLM Triage → Kafka `alerts.enriched`).

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

### Step 4: Verify Success in PostgreSQL Database
After allowing local CPU inference to complete, verify the alert was scored and enriched successfully:

```bash
docker exec mlflow_db psql -U admin -d aries -c "\x" -c "SELECT alert_id, normalized_title, ml_score, risk_score, status FROM alerts ORDER BY created_at DESC LIMIT 1;"
```
