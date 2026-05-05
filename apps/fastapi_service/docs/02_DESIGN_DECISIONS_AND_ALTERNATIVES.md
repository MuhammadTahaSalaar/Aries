# ARIES AI Service — Design Decisions & Alternatives

> **Document Version:** 1.0.0  
> **Last Updated:** March 3, 2026  
> **Purpose:** Comprehensive rationale for all technical choices

---

## Table of Contents

1. [ML Model Selection](#1-ml-model-selection)
2. [Feature Engineering Approach](#2-feature-engineering-approach)
3. [Dataset Choices](#3-dataset-choices)
4. [Inference Runtime](#4-inference-runtime)
5. [Web Framework](#5-web-framework)
6. [Message Broker](#6-message-broker)
7. [Database Selection](#7-database-selection)
8. [Caching Strategy](#8-caching-strategy)
9. [Model Serving Architecture](#9-model-serving-architecture)
10. [MLOps Stack](#10-mlops-stack)
11. [Containerization Strategy](#11-containerization-strategy)
12. [Security Considerations](#12-security-considerations)

---

## 1. ML Model Selection

### 1.1 Alert Triage: Why XGBoost?

**Chosen:** XGBoost  
**Alternatives Considered:** LightGBM, CatBoost, Random Forest, Neural Networks (MLP/Transformer)

| Model | Pros | Cons | Decision |
|-------|------|------|----------|
| **XGBoost** | Best accuracy on GUIDE, GPU support, handles missing values, interpretable | Slightly larger model size | ✅ **Selected** |
| LightGBM | Faster training, smaller model | 1-2% lower accuracy on our data | ❌ Rejected |
| CatBoost | Native categorical handling | Slower GPU implementation | ❌ Rejected |
| Random Forest | Simple, no tuning needed | 5%+ lower accuracy, no GPU | ❌ Rejected |
| MLP/Transformer | Could learn complex patterns | Overfitted on tabular data, slower | ❌ Rejected |

**Key Reasons:**
1. **Accuracy**: XGBoost achieved **94.19%** vs LightGBM's 92.8% and CatBoost's 93.1%
2. **GPU Acceleration**: XGBoost 2.0+ `device="cuda"` enabled 10x faster training on 9.5M samples
3. **ONNX Export**: Mature `onnxmltools` support vs experimental for others
4. **Interpretability**: Feature importance scores for compliance/audit

### 1.2 NER: Why SecureBERT over BERT?

**Chosen:** SecureBERT (RoBERTa variant)  
**Alternatives Considered:** BERT-base, RoBERTa-base, SciBERT, CyBERT, GPT-3.5 API

| Model | Pros | Cons | Decision |
|-------|------|------|----------|
| **SecureBERT** | Pre-trained on security corpora, domain vocabulary | Slightly larger than BERT | ✅ **Selected** |
| BERT-base | Well-understood, fast | Generic vocabulary misses security terms | ❌ Rejected |
| RoBERTa-base | Better pre-training than BERT | No security domain adaptation | ❌ Rejected |
| SciBERT | Good for scientific text | Wrong domain | ❌ Rejected |
| CyBERT | Security-focused | Not publicly available, paper only | ❌ Rejected |
| GPT-3.5 API | High accuracy, zero-shot | Cost per request, latency, data privacy | ❌ Rejected |

**Key Reasons:**
1. **Domain Adaptation**: SecureBERT was pre-trained on CVE descriptions, security blogs, threat reports — its tokenizer handles terms like "CVE-2021-44228", "mimikatz", "APT41" correctly
2. **Vocabulary**: Security-specific tokens are single tokens, not subword-split
3. **Cost**: Self-hosted model = $0/query vs $0.002+/query for API calls
4. **Privacy**: Customer data never leaves our infrastructure

### 1.3 Summarization: Why BART over T5/GPT?

**Chosen:** BART-base  
**Alternatives Considered:** T5-base, PEGASUS, LED (Longformer-Encoder-Decoder), GPT-3.5, Llama 2

| Model | Pros | Cons | Decision |
|-------|------|------|----------|
| **BART-base** | Great for abstractive summarization, 140M params | 1024 token limit | ✅ **Selected** |
| T5-base | Unified text-to-text format | Higher memory usage, slower | ❌ Rejected |
| PEGASUS | SOTA summarization | Gap Sentence Generation pre-training not ideal for security | ❌ Rejected |
| LED | 16K token context | 4x parameters, slow inference | ❌ Rejected |
| GPT-3.5 | Best quality, long context | Cost, latency, data privacy | ❌ Rejected |
| Llama 2 7B | Open weights, good quality | Too large for CPU inference (13GB+) | ❌ Rejected |

**Key Reasons:**
1. **Size vs Quality**: BART-base (140M params) fits in 2GB RAM while achieving competitive ROUGE scores
2. **ONNX Export**: Clean encoder-decoder export vs T5's more complex architecture
3. **Latency**: ~450ms on CPU vs LED's ~2000ms
4. **Security Domain**: Fine-tuned on GovReport which includes incident-style reports

---

## 2. Feature Engineering Approach

### 2.1 Why Target Encoding over One-Hot?

**Chosen:** Target Encoding with `category_encoders`  
**Alternatives Considered:** One-Hot Encoding, Label Encoding, Feature Hashing, Embeddings

| Method | Pros | Cons | Decision |
|--------|------|------|----------|
| **Target Encoding** | Captures target correlation, handles high cardinality, 37 features → 37 floats | Target leakage risk (mitigated by smoothing) | ✅ **Selected** |
| One-Hot Encoding | Simple, no leakage | 37 cols × ~50K unique values = explosion | ❌ Rejected |
| Label Encoding | Compact | Introduces false ordinal relationship | ❌ Rejected |
| Feature Hashing | Fixed size | Collisions lose information | ❌ Rejected |
| Embeddings (NN) | Learns relationships | Requires NN model, not compatible with XGBoost | ❌ Rejected |

**Key Reasons:**
1. **Cardinality**: `AlertTitle` has ~50,000 unique values — One-Hot would create 50K sparse features
2. **Leakage Mitigation**: Smoothing parameter (m=1.0) and fit-only-on-train prevents target leakage
3. **XGBoost Compatibility**: Dense float vectors work perfectly with gradient boosting

### 2.2 Why 49 Features?

**Feature Breakdown:**
- 3 temporal (hour, day_of_week, month)
- 2 MITRE (has_mitre, mitre_count)
- 7 null indicators (has_EmailClusterId, has_ThreatFamily, etc.)
- 37 target-encoded categoricals

**Why not more/fewer?**
- Adding raw text features (TF-IDF) increased training time 5x with <1% accuracy gain
- Removing temporal features dropped accuracy by 2%
- The 7 null indicators capture missingness patterns that are predictive

---

## 3. Dataset Choices

### 3.1 Triage: Why Microsoft GUIDE?

**Chosen:** Microsoft GUIDE Dataset  
**Alternatives Considered:** CICIDS, NSL-KDD, CTU-13, Synthetic data

| Dataset | Samples | Pros | Cons | Decision |
|---------|---------|------|------|----------|
| **GUIDE** | 9.5M | Real enterprise alerts, labeled by Microsoft | Proprietary format | ✅ **Selected** |
| CICIDS 2017 | 2.8M | Academic standard | Network flows only, not alerts | ❌ Rejected |
| NSL-KDD | 125K | Classic benchmark | Dated (1999), network-focused | ❌ Rejected |
| CTU-13 | 2M | Botnet-focused | Single attack type | ❌ Rejected |
| Synthetic | Any | Full control | Not representative of real alerts | ❌ Rejected |

**Key Reasons:**
1. **Scale**: 9.5M samples = robust generalization
2. **Realism**: Actual Microsoft Defender alerts from enterprise environments
3. **Labels**: Expert-labeled as TruePositive/FalsePositive/BenignPositive
4. **Features**: 45 columns covering endpoint, identity, email, network

### 3.2 NER: Why CyNER + CASIE?

**Chosen:** CyNER (primary) + CASIE (augmentation)  
**Alternatives Considered:** CoNLL-2003, OntoNotes, Custom annotation

| Dataset | Entities | Pros | Cons | Decision |
|---------|----------|------|------|----------|
| **CyNER** | Security IOCs | Security-specific labels | Small (5K samples) | ✅ **Selected** |
| **CASIE** | Security events | Security domain | Different label schema | ✅ **Augmentation** |
| CoNLL-2003 | Person, Org, Loc | Large, well-annotated | Wrong domain (news) | ❌ Rejected |
| OntoNotes | 18 entity types | Large | No security entities | ❌ Rejected |
| Custom | Our needs | Perfect fit | Expensive ($50K+ annotation) | ❌ Rejected |

### 3.3 Summarization: Why GovReport?

**Chosen:** GovReport  
**Alternatives Considered:** CNN/DailyMail, XSum, arXiv, PubMed

| Dataset | Samples | Avg Length | Pros | Cons | Decision |
|---------|---------|------------|------|------|----------|
| **GovReport** | 19K | 8K/400 | Long documents, formal style | Smaller dataset | ✅ **Selected** |
| CNN/DailyMail | 300K | 800/50 | Large | Short docs, news style | ❌ Rejected |
| XSum | 227K | 400/20 | Large | Extreme compression | ❌ Rejected |
| arXiv | 215K | 6K/200 | Long docs | Scientific, not security | ❌ Rejected |

**Key Reasons:**
1. **Document Length**: Security incident reports are typically 1000-5000 words — GovReport's 8K average is closest
2. **Formal Style**: Government reports match incident report tone better than news
3. **Abstractive**: GovReport requires true abstractive summarization (not extractive)

---

## 4. Inference Runtime

### 4.1 Why ONNX Runtime over PyTorch/TensorFlow?

**Chosen:** ONNX Runtime  
**Alternatives Considered:** PyTorch (native), TensorFlow Serving, TorchServe, Triton

| Runtime | Pros | Cons | Decision |
|---------|------|------|----------|
| **ONNX Runtime** | Framework-agnostic, optimized CPU kernels, 2-5x faster | Export complexity | ✅ **Selected** |
| PyTorch | Simple, no export | Slower inference, larger memory | ❌ Rejected |
| TF Serving | Production-ready | TensorFlow ecosystem lock-in | ❌ Rejected |
| TorchServe | PyTorch native serving | Heavier infrastructure | ❌ Rejected |
| Triton | Multi-model serving | Overkill for 3 models | ❌ Rejected |

**Key Reasons:**
1. **Performance**: ONNX Runtime with ORT optimizations is 2-5x faster than raw PyTorch
2. **Memory**: ONNX models use ~40% less memory (no autograd graph)
3. **Cross-Framework**: XGBoost, PyTorch (SecureBERT, BART) all export to ONNX
4. **Simplicity**: Single runtime for all models

### 4.2 Why CPU Inference over GPU?

**Chosen:** CPU inference (CPUExecutionProvider)  
**Rationale:**
1. **Cost**: No GPU instances required ($0.40/hr CPU vs $3/hr GPU)
2. **Latency**: Our models are small enough that CPU latency is acceptable (<100ms for NER)
3. **Scaling**: CPU scales horizontally more easily than GPU
4. **Availability**: GPU instances have availability issues in some regions

**Note:** GPU can be enabled by changing provider to `CUDAExecutionProvider` if latency requirements increase.

---

## 5. Web Framework

### 5.1 Why FastAPI over Flask/Django?

**Chosen:** FastAPI  
**Alternatives Considered:** Flask, Django, Starlette, aiohttp

| Framework | Pros | Cons | Decision |
|-----------|------|------|----------|
| **FastAPI** | Async native, auto-docs, Pydantic validation, fastest | Newer, smaller community | ✅ **Selected** |
| Flask | Simple, mature | No async, manual validation | ❌ Rejected |
| Django | Full-featured | Heavy for microservice, sync-first | ❌ Rejected |
| Starlette | FastAPI's base | No automatic validation | ❌ Rejected |
| aiohttp | Pure async | No auto-docs, more boilerplate | ❌ Rejected |

**Key Reasons:**
1. **Async/Await**: Kafka consumers, DB queries, S3 downloads all benefit from async
2. **Pydantic v2**: Automatic request validation catches malformed payloads
3. **OpenAPI**: Auto-generated Swagger/ReDoc documentation
4. **Performance**: 3-5x faster than Flask for I/O-bound workloads

---

## 6. Message Broker

### 6.1 Why Kafka over RabbitMQ/Redis Streams?

**Chosen:** Apache Kafka  
**Alternatives Considered:** RabbitMQ, Redis Streams, AWS SQS, NATS

| Broker | Pros | Cons | Decision |
|--------|------|------|----------|
| **Kafka** | High throughput, replay, partitioned consumers | Heavier setup | ✅ **Selected** |
| RabbitMQ | Simple, AMQP standard | No replay, lower throughput | ❌ Rejected |
| Redis Streams | Already have Redis | Limited feature set | ❌ Rejected |
| AWS SQS | Managed | Cloud lock-in, no replay | ❌ Rejected |
| NATS | Lightweight | Smaller ecosystem | ❌ Rejected |

**Key Reasons:**
1. **Replay**: Can re-process alerts if ML model is updated
2. **Consumer Groups**: Multiple services can consume `alerts.enriched` independently
3. **Durability**: Alerts are not lost if consumer crashes
4. **Throughput**: Handles 100K+ alerts/second (future-proof)

### 6.2 Why KRaft over ZooKeeper?

**Chosen:** KRaft mode (Kafka 3.9)  
**Rationale:**
1. **Simplicity**: No separate ZooKeeper cluster to manage
2. **Resource**: One less container/process
3. **Future**: ZooKeeper is deprecated in Kafka 4.0
4. **Performance**: Faster metadata operations

---

## 7. Database Selection

### 7.1 Why PostgreSQL over MongoDB/MySQL?

**Chosen:** PostgreSQL  
**Alternatives Considered:** MongoDB, MySQL, DynamoDB, CockroachDB

| Database | Pros | Cons | Decision |
|----------|------|------|----------|
| **PostgreSQL** | JSONB for raw_data, strong ACID, RLS | Scaling complexity | ✅ **Selected** |
| MongoDB | Native JSON, flexible schema | No joins, eventual consistency | ❌ Rejected |
| MySQL | Familiar, fast reads | No JSONB, weaker JSON support | ❌ Rejected |
| DynamoDB | Serverless scaling | AWS lock-in, expensive | ❌ Rejected |
| CockroachDB | Distributed SQL | Overkill, learning curve | ❌ Rejected |

**Key Reasons:**
1. **JSONB**: Store raw SIEM payloads efficiently with indexing
2. **Row-Level Security**: Multi-tenant isolation at database level
3. **MLflow Compatible**: MLflow uses PostgreSQL for metadata
4. **Mature**: 30+ years of battle-tested reliability

---

## 8. Caching Strategy

### 8.1 Why Redis over Memcached?

**Chosen:** Redis  
**Alternatives Considered:** Memcached, In-process cache, PostgreSQL materialized views

| Solution | Pros | Cons | Decision |
|----------|------|------|----------|
| **Redis** | Data structures, TTL, persistence option | Single-threaded | ✅ **Selected** |
| Memcached | Multi-threaded, simple | No persistence, limited data types | ❌ Rejected |
| In-process | Zero latency | Not shared across workers | ❌ Rejected |
| PG views | SQL integration | Slower, no TTL | ❌ Rejected |

**Key Reasons:**
1. **TTL**: Automatic cache expiration (30min NER, 2hr summaries)
2. **Deduplication**: `SETNX` for idempotent alert processing
3. **Tenant Namespacing**: Key prefixes for multi-tenant isolation

### 8.2 What Do We Cache?

| Data | TTL | Key Pattern | Rationale |
|------|-----|-------------|-----------|
| NER results | 30 min | `aries:{tenant}:ner:{hash}` | Same text = same IOCs |
| Summaries | 2 hours | `aries:{tenant}:summary:{hash}` | Expensive to generate |
| Alert dedup | 5 min | `aries:{tenant}:dedup:{alert_id}` | Prevent duplicate processing |

**Not Cached:**
- Triage scores (each alert is unique)
- Database writes (must be durable)

---

## 9. Model Serving Architecture

### 9.1 Why Embedded vs Dedicated ML Server?

**Chosen:** Embedded (models loaded in FastAPI process)  
**Alternatives Considered:** Seldon Core, KServe, BentoML, Ray Serve

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| **Embedded** | Simple, low latency, no network hop | Scaling couples API + ML | ✅ **Selected** |
| Seldon Core | Production MLOps | Kubernetes-heavy | ❌ Rejected |
| KServe | Standard ML serving | Complex setup | ❌ Rejected |
| BentoML | Python-native packaging | Additional abstraction layer | ❌ Rejected |
| Ray Serve | Distributed serving | Overkill for 3 models | ❌ Rejected |

**Key Reasons:**
1. **Latency**: No network hop to separate ML service
2. **Simplicity**: One container instead of 4
3. **Scale**: At our volume (1000s alerts/day), single instance suffices
4. **Cost**: Fewer containers = lower infrastructure cost

### 9.2 Future Scaling Path

If throughput exceeds single-instance capacity:
1. Increase Uvicorn workers (already at 2)
2. Horizontal scaling with Kubernetes replicas
3. Extract ML to dedicated service (BentoML)

---

## 10. MLOps Stack

### 10.1 Why MLflow over Weights & Biases/Neptune?

**Chosen:** MLflow  
**Alternatives Considered:** Weights & Biases, Neptune, Comet ML, DVC

| Tool | Pros | Cons | Decision |
|------|------|------|----------|
| **MLflow** | Open source, self-hosted, model registry | UI less polished | ✅ **Selected** |
| W&B | Beautiful UI, collaboration | SaaS cost, data leaves infra | ❌ Rejected |
| Neptune | Good experiment tracking | SaaS cost | ❌ Rejected |
| Comet ML | Similar to W&B | SaaS cost | ❌ Rejected |
| DVC | Git-like versioning | No model registry, tracking UI | ❌ Rejected |

**Key Reasons:**
1. **Self-Hosted**: Data sovereignty — customer alerts never leave our infrastructure
2. **Cost**: $0 vs $500+/month for SaaS alternatives
3. **Model Registry**: Built-in model staging (Staging → Production)
4. **S3 Artifacts**: Direct integration with MinIO

### 10.2 Why MinIO over AWS S3?

**Chosen:** MinIO (self-hosted S3-compatible)  
**Rationale:**
1. **On-Premises**: Some customers require on-prem deployment
2. **Cost**: Free vs $0.023/GB/month
3. **Compatibility**: 100% S3 API compatible
4. **Dev Experience**: Works identically in local dev and production

---

## 11. Containerization Strategy

### 11.1 Why Multi-Stage Dockerfile?

**Chosen:** Multi-stage build  
**Benefits:**
```dockerfile
# Stage 1: Builder (1.2 GB)
FROM python:3.11-slim AS builder
RUN pip install --prefix=/install -r requirements.txt

# Stage 2: Runtime (600 MB)
FROM python:3.11-slim
COPY --from=builder /install /usr/local
```

**Size Reduction:** 1.2 GB → 600 MB (50% smaller)

### 11.2 Why Two Docker Compose Files?

**Chosen:** Separate `MLOps/` and `apps/fastapi_service/` stacks  
**Rationale:**
1. **Lifecycle**: MLOps stack rarely changes; serving stack updates frequently
2. **Resources**: Can run MLOps on different machine/cluster
3. **Isolation**: MLflow failure doesn't affect live inference
4. **Development**: Can develop without full MLOps stack

---

## 12. Security Considerations

### 12.1 Why Non-Root Container User?

```dockerfile
RUN useradd -m aries
USER aries
```

**Rationale:**
1. **Least Privilege**: Container compromise doesn't give root
2. **K8s PSP**: Required by many Kubernetes security policies
3. **Best Practice**: CIS Docker Benchmark recommendation

### 12.2 Why Tenant Header vs JWT?

**Chosen:** `X-Tenant-ID` header  
**Alternatives Considered:** JWT tokens, API keys, OAuth2

| Method | Pros | Cons | Decision |
|--------|------|------|----------|
| **Header** | Simple, performant | No built-in auth | ✅ **Selected** (with Gateway) |
| JWT | Standard auth | Token parsing overhead | ❌ Deferred to Gateway |
| API Keys | Simple auth | Key management complexity | ❌ Deferred to Gateway |

**Rationale:**
- Authentication is handled by API Gateway (Kong/Nginx) in front of FastAPI
- FastAPI trusts the gateway-injected `X-Tenant-ID`
- Keeps ML service stateless and simple

### 12.3 Data Privacy

| Concern | Mitigation |
|---------|-----------|
| Customer data in logs | Structured logging filters PII |
| Data at rest | PostgreSQL + MinIO encryption |
| Data in transit | TLS between services |
| Model extraction | ONNX models stored in private MinIO bucket |

---

## Summary

| Decision | Chosen | Top Alternative | Key Differentiator |
|----------|--------|-----------------|-------------------|
| Triage Model | XGBoost | LightGBM | 1.5% higher accuracy |
| NER Model | SecureBERT | BERT-base | Security domain adaptation |
| Summarization | BART-base | T5-base | Lower memory, faster inference |
| Encoding | Target Encoding | One-Hot | Handles high cardinality |
| Runtime | ONNX | PyTorch | 2-5x faster inference |
| Framework | FastAPI | Flask | Async native, auto-docs |
| Message Broker | Kafka | RabbitMQ | Replay capability |
| Database | PostgreSQL | MongoDB | JSONB + RLS |
| Cache | Redis | Memcached | TTL + data structures |
| MLOps | MLflow | W&B | Self-hosted, free |
| Storage | MinIO | AWS S3 | On-prem compatible |

Each decision was made considering:
1. **Performance**: Latency and throughput requirements
2. **Cost**: Infrastructure and operational costs
3. **Complexity**: Team expertise and maintenance burden
4. **Security**: Data sovereignty and privacy
5. **Scalability**: Future growth path
