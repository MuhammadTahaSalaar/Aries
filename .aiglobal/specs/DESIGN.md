# ARIES — Software Design Specification

> **AI-Enhanced SOAR: Optimizing Security Operations through Intelligent Automation**
> Source: SDS v1.0 (11-Dec-2025) | Authors: Sameed Ilyas, Muhammad Taha Salaar, Muhammad Zain

---

## 1. Design Methodology

### Object-Oriented Approach
- **Modularity & Encapsulation:** Each module (Alert Ingestion, Orchestration, AI/Analytics) modeled as classes with encapsulated state and behavior.
- **Extensibility & Polymorphism:** Generic `IntegrationConnector` interface — OrchestrationEngine interacts with any connector implementing this interface. New connectors plug in with zero changes to core.
- **Reusability & Maintainability:** Core domain concepts (Alert, Case, Playbook) modeled as reusable classes, reducing duplication.

### Software Process Model — Iterative and Incremental
- Aligns with FYP milestones: demonstrable working software at each milestone.
- Early risk mitigation: complex components (ML, orchestration) validated in early increments.
- Adapts to evolving requirements and third-party library changes.

---

## 2. Architectural Design

### Pattern: Modular Microservices (Event-Driven)

**Justification:**
- **Scalability:** Independent horizontal scaling (500 alerts/sec burst, 100 concurrent playbooks).
- **Extensibility:** New integrations deployed as isolated services.
- **Resilience:** Service failure isolation prevents system-wide outages (target: 98% uptime).
- **Polyglot Stack:** Go for orchestration concurrency, Python/PyTorch/FastAPI for AI/ML.
- **Iterative Delivery:** Each microservice is a discrete, deliverable increment.

### High-Level Component Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    EXTERNAL ALERT SOURCES                       │
│            (SIEM, EDR, NTA, Threat-Intel Feeds)                 │
└───────────────────────┬─────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│              ALERT INGESTION & TRIAGE LAYER                     │
│  ┌──────────────┐ ┌──────────────┐ ┌─────────────────────────┐  │
│  │ Alert        │ │ Context      │ │ ML Triage Engine        │  │
│  │ Normalizer   │ │ Enrichment   │ │ (FastAPI + PyTorch/ONNX)│  │
│  │ (Logstash)   │ │              │ │                         │  │
│  └──────┬───────┘ └──────┬───────┘ └────────────┬────────────┘  │
└─────────┼────────────────┼──────────────────────┼───────────────┘
          │                │                      │
          ▼                ▼                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                     EVENT BUS (Apache Kafka)                     │
│  Topics: alerts.raw | alerts.enriched | cases.updated |         │
│          playbooks.events | ml.feedback                         │
└────────┬───────────────────┬────────────────────┬───────────────┘
         │                   │                    │
         ▼                   ▼                    ▼
┌────────────────┐  ┌────────────────┐  ┌─────────────────────────┐
│ ORCHESTRATION  │  │ AI & ANALYTICS │  │ OPERATIONS LAYER        │
│ LAYER (Go)     │  │ LAYER (Python) │  │ (Next.js + React)       │
│                │  │                │  │                         │
│ • Orch Engine  │  │ • ML Scoring   │  │ • Analyst Dashboard     │
│ • Playbook     │  │ • NLP/NER      │  │ • Triage Queue          │
│   Executor     │  │ • Summarizer   │  │ • Case Management       │
│ • Connectors   │  │ • Retraining   │  │ • Reporting Dashboard   │
│ • Approval Svc │  │   Loop (MLflow)│  │ • Approval Workflow     │
│ • Go Workers   │  │ • Analytics    │  │ • WebSocket RT Updates  │
└────────┬───────┘  └────────┬───────┘  └────────────┬────────────┘
         │                   │                       │
         ▼                   ▼                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                    PERSISTENCE LAYER (Polyglot)                  │
│  ┌────────────┐ ┌──────────────┐ ┌───────┐ ┌───────┐ ┌──────┐  │
│  │ PostgreSQL │ │Elasticsearch │ │ Neo4j │ │ Redis │ │  S3  │  │
│  │ (System of │ │ (Time-series │ │(Graph │ │(Cache,│ │(Bulk │  │
│  │  Record)   │ │  Search)     │ │ Corr.)│ │ PubSub│ │ Data)│  │
│  └────────────┘ └──────────────┘ └───────┘ └───────┘ └──────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Component Descriptions

### 3.1 Alert Ingestion & Triage Layer
**Purpose:** Primary entry point for all external security data. Normalizes, enriches, and applies ML to prioritize incoming alerts.

- **Alert Normalizer (Logstash):** Parses vendor-specific payloads into canonical alert schema (IDs, entities, indicators, timestamps, MITRE ATT&CK mapping). Validates required fields, stamps source provenance, attaches deduplication keys.
- **Context Enrichment:** Fetches asset inventory, user identity, geo/IP intel, threat-intel matches. Adds playbook hints (e.g., "isolate endpoint").
- **ML Triage Engine (FastAPI + PyTorch/ONNX):** Applies trained models for true-positive likelihood, severity, urgency. Consumes features: entity reputation, behavioral anomalies, asset criticality, historical outcomes. Produces triage labels and confidence scores.
- **Risk Prioritization:** Weighted policies (business criticality, blast radius, compliance impact). Emits normalized, enriched, prioritized alerts to Event Bus.

### 3.2 Event Bus (Apache Kafka)
**Purpose:** Decouple producers (ingestion/ML) from consumers (orchestration, dashboards, analytics).

**Topics:**
| Topic | Purpose |
|---|---|
| `alerts.raw` | Raw normalized alerts from ingestion |
| `alerts.enriched` | ML-scored and context-enriched alerts |
| `cases.updated` | Case state change events |
| `playbooks.events` | Playbook execution state changes |
| `ml.feedback` | Analyst feedback for model retraining |

Supports backpressure handling, replay, exactly-once/at-least-once semantics, and stream processors for correlation (e.g., multi-stage kill-chain join).

### 3.3 Orchestration Layer (Go)
**Purpose:** Execute automated responses safely and consistently while preserving analyst control.

- **API Gateway:** AuthN/Z, rate limiting, request signing, per-tenant routing. REST/gRPC endpoints for alert submission, case updates, playbook triggers.
- **Orchestration Engine:** State machine / workflow runtime for playbooks (fan-out, retries, timeouts, compensations). Correlates events, opens/updates cases, coordinates multi-system actions. Stateless controller — scales horizontally.
- **Playbook Execution:** Library of actions (EDR isolate, IAM disable, M365 purge, firewall rule, ticket ops). Sandboxed runners with secrets management, circuit breakers, idempotent side-effects.
- **Integration Connectors:** Adapters to SIEMs, EDRs, email gateways, cloud providers, ticketing (Jira/ServiceNow), chat (Slack/Teams). Async IO, pagination, quota-aware schedulers.
- **Approval Service:** Dedicated component for managing approval logic and user notifications. Decoupled from orchestration engine to prevent blocking.

### 3.4 AI & Analytics Layer (Python)
**Purpose:** Continuous learning and decision support.

- **ML Model (Scoring/Triage/Recommendation):** Feature store reads from ES/SQL; model artifacts stored/versioned in MLflow; online inference served behind low-latency API (ONNX Runtime).
- **NLP Services:**
  - **Entity Extraction:** NER (Hugging Face/spaCy) over alert text, tickets, email threads to extract IOC/IOA, users, hosts, processes.
  - **Incident Summarization:** Executive and analyst-level summaries with citations to evidence. Case timelines and "next best action" aligned to MITRE ATT&CK.
- **Analytics Engine:** Detection efficacy, MTTD/MTTR, alert volume heatmaps, suppression savings. KPIs to Reporting Dashboard + governance/compliance reports.
- **Model Retraining Loop:** Consumes `ml.feedback` topic (case outcomes, analyst overrides, drift monitors). Schedules retraining, canary/batch validation, rollback on degradation, staged ring promotion.

### 3.5 Persistence Layer (Polyglot)

| Store | Role | Data |
|---|---|---|
| **PostgreSQL** | System of record (ACID) | Cases, playbook runs, approvals, audit trail, user roles. RLS for multi-tenant isolation. |
| **Elasticsearch** | Time-series search | Alerts, events, telemetry. Powers investigations, pivoting, near-real-time dashboards. |
| **Neo4j** | Graph correlation | IOC↔Asset↔User relationships. Attack-path analysis, lateral movement detection. |
| **Redis Cluster** | Cache + pub/sub | Hot entities (hosts/users), dedupe keys, rate-limits, workflow locks, session state. |
| **Amazon S3** | Object storage | Log archives, forensic reports, threat-intel files, raw alert data, ML model artifacts. |

### 3.6 Operations Layer (Next.js + React)
**Purpose:** Human-in-the-loop control with strong governance.

- **Analyst Dashboard:** Risk-sorted triage queue, entity context, one-click playbook triggers, annotation threads. Inline NLP summaries, evidence attachments, timeline reconstruction.
- **Reporting Dashboard:** SOC metrics, SLA compliance, executive summaries, customizable widgets.
- **Approval Workflow:** Policy-based gates, e-signatures, immutable audit trail, auto-escalation paths.
- **Case Management:** Full lifecycle: detection → enrichment → action → verification → closure → post-incident review. Links to ES artifacts, orchestration action logs, generated summaries.
- **Real-Time:** WebSockets/SSE for live updates to queues, cases, and dashboards.

---

## 4. Domain Class Model

### Core Classes

#### Alert
| Attribute | Type | Description |
|---|---|---|
| `alert_id` | UUID | Unique identifier |
| `tenant_id` | UUID | Tenant isolation key |
| `rawPayload` | JSON | Original unmodified payload |
| `normalized_title` | String | Standardized title after normalization |
| `ml_score` | Float (0-1) | True-positive likelihood from ML Triage Engine |
| `risk_score` | Integer | Final risk score (ML + asset criticality + context) |
| `status` | Enum | New, Triaged, Closed_FP, Escalated |

**Methods:** `normalize(rawData)`, `calculateRiskScore(assetData, threatIntel)`, `updateStatus(newStatus)`

#### Case
| Attribute | Type | Description |
|---|---|---|
| `case_id` | UUID | Unique identifier |
| `tenant_id` | UUID | Tenant isolation key |
| `title` | String | Descriptive case title |
| `summary` | Text | NLP-generated incident summary |
| `status` | Enum | Detected, Open, Investigating, Awaiting_Approval, Contained, Closed |
| `severity` | Enum | Low, Medium, High, Critical |

**Methods:** `addAlert(alert)`, `assignAnalyst(analyst)`, `updateStatus(newState)`, `generateSummary()`

#### Playbook
| Attribute | Type | Description |
|---|---|---|
| `playbook_id` | UUID | Unique identifier |
| `tenant_id` | UUID | Tenant isolation key |
| `name` | String | Human-readable name |
| `description` | Text | Purpose and actions description |
| `trigger_conditions` | JSON/RuleSet | Rules for automatic selection |

**Methods:** `getActions()`, `matchesTrigger(alert)`

#### Action
| Attribute | Type | Description |
|---|---|---|
| `action_id` | UUID | Unique identifier |
| `action_type` | String | e.g., "enrich_ip", "quarantine_host", "disable_user" |
| `status` | Enum | Pending, Running, Completed, Failed, Awaiting_Approval |
| `result` | JSON | Output from execution |
| `timestamp` | DateTime | Execution timestamp |

**Methods:** `execute(connector)`, `getStatus()`, `getResult()`

#### SecurityAnalyst
| Attribute | Type | Description |
|---|---|---|
| `user_id` | UUID | Unique identifier |
| `name` | String | Full name |
| `role` | Enum | Tier-1_Analyst, Tier-2_Analyst, SOC_Manager, Security_Engineer |

**Methods:** `assignCase(case)`, `approveAction(action)`

#### IOC (Indicator of Compromise)
| Attribute | Type | Description |
|---|---|---|
| `ioc_id` | UUID | Unique identifier |
| `tenant_id` | UUID | Tenant isolation key |
| `type` | Enum | IP_Address, File_Hash, Domain, URL, Email_Address |
| `value` | String | Actual IOC value |
| `source` | String | Extraction source |

**Methods:** `validate()`, `getReputation(threatFeedConnector)`

#### OrchestrationEngine (Controller)
Stateless controller — scales horizontally without state replication.

**Methods:** `selectPlaybook(alert)`, `executePlaybook(playbook, case)`, `pauseForApproval(action)`

#### IntegrationConnector (Interface)
Generic contract for all external tool integrations. Concrete implementations: `EDRConnector`, `SIEMConnector`, etc.

**Methods:** `executeAction(actionType, parameters)` → translates generic command to specific API call.

### Key Relationships
- **SecurityAnalyst → Case:** One-to-many (analyst manages multiple cases).
- **Case → Alert:** One-to-many (case aggregates related alerts).
- **Alert → IOC:** One-to-many (alert contains multiple IOCs).
- **Playbook → Action:** Composition (actions don't exist independently of playbook).
- **Playbook → Case:** One-to-many (playbook definition reused across cases).
- **OrchestrationEngine → IntegrationConnector:** Dependency on interface (polymorphism).

---

## 5. Incident Lifecycle State Machine

```
                     ┌──────────┐
       new alert ──► │ Detected │
       triaged       └────┬─────┘
                          │ analyst validates
                          ▼
                     ┌──────────┐
                     │   Open   │
                     └────┬─────┘
                          │ analyst assigned
                          ▼
                     ┌──────────────┐
              ┌─────►│ Investigating │◄────────┐
              │      └──────┬───────┘         │
              │             │                 │
              │  critical action requires     │ manager approves
              │  sign-off   │                 │
              │             ▼                 │
              │     ┌───────────────────┐     │
              │     │ Awaiting Approval │─────┘
              │     └───────────────────┘
              │             │ manager rejects → closes/escalates
              │             │
              │  response actions completed
              │             │
              │             ▼
              │      ┌─────────────┐
              │      │  Contained  │
              │      └──────┬──────┘
              │             │ fully resolved
              │             ▼
              │      ┌─────────────┐
              └──────│   Closed    │
                     └─────────────┘
```

| Transition | Event |
|---|---|
| (Initial) → Detected | New alert triaged into incident |
| Detected → Open | Analyst validates incident |
| Open → Investigating | Analyst assigned to investigate |
| Investigating → Awaiting Approval | Critical action requires sign-off |
| Awaiting Approval → Investigating | Manager approves action |
| Investigating → Contained | Response actions completed |
| Contained → Closed | Incident fully resolved |

---

## 6. Key Sequence Flows

### 6.1 Alert Triage Flow
1. **Alert Source** → `send(raw_alert)` → **Ingestion Layer**
2. **Ingestion Layer** → `normalize()` → `enrich()` → **ML Triage Engine**
3. **ML Triage Engine** → `score()` → `persist(enriched_alert)` → **Elasticsearch**
4. **Analyst Dashboard** → `query(alerts)` → **Elasticsearch** (asynchronous, decoupled from ingestion)

### 6.2 Playbook Orchestration Flow
1. **OrchestrationEngine** → `selectPlaybook()` → **Playbook Library**
2. For each action: **OrchestrationEngine** → `executeAction(type, params)` → **IntegrationConnector**
3. **IntegrationConnector** → API call → **External System** → result returned
4. **OrchestrationEngine** logs result, proceeds to next step

### 6.3 Approval Workflow Flow
1. **OrchestrationEngine** encounters critical action → `requestApproval(actionDetails)` → **Approval Service** (pauses workflow)
2. **Approval Service** → `notify(request)` → **SOC Manager**
3. **SOC Manager** → `approve()` or `deny()` → **Approval Service**
4. **Approval Service** → signals **OrchestrationEngine** to resume or terminate

### 6.4 NLP Summarization Flow (Parallel)
1. Unstructured data input → **NLP Service**
2. **Fork:** NER (entity extraction) ‖ Summarization Model (executive + analyst summaries)
3. **Join:** Structured output (IOCs + summaries) returned

---

## 7. Data Design

### 7.1 Canonical Alert Schema
```json
{
  "alert_id": "UUID",
  "tenant_id": "UUID",
  "normalized_title": "String",
  "risk_score": 95,
  "ml_score": 0.92,
  "status": "New",
  "raw_data": {
    "source": "EDR",
    "event_details": "..."
  }
}
```

### 7.2 Canonical Case Schema
```json
{
  "case_id": "UUID",
  "tenant_id": "UUID",
  "title": "String",
  "summary": "NLP-generated summary text...",
  "status": "Investigating",
  "severity": "High",
  "alert_ids": ["UUID", "UUID"]
}
```

### 7.3 ERD Summary
- **Case** (1) → (many) **Alert** → (many) **IOC**
- **SecurityAnalyst** (1) → (many) **Case**
- **Playbook** (1) → (many) **Action**
- **Playbook** (1) → (many) **Case** (reuse across incidents)
- **Action** → (logs to) **Case** (auditable timeline)
- All entities tagged with `tenant_id` for isolation.

---

## 8. Communication Patterns

| Pattern | Use |
|---|---|
| **gRPC** | Internal microservice-to-microservice (low-latency, strongly typed, bi-directional streaming) |
| **REST** | External integrations, client-facing APIs (universal compatibility) |
| **Kafka** | Asynchronous event-driven decoupling between all layers |
| **WebSockets** | Real-time push updates to Analyst Dashboard |
| **HTTPS/TLS** | All public-facing traffic (port 443) |

---

## 9. Security Architecture

| Layer | Mechanism |
|---|---|
| **Authentication** | OAuth2/OIDC via API Gateway |
| **Authorization** | RBAC enforced at API Gateway (per-role, per-action) |
| **Secrets** | HashiCorp Vault (dynamic rotation, never hardcoded) |
| **Network** | TLS Everywhere (mTLS between services) |
| **Data at Rest** | Per-tenant encryption |
| **Multi-Tenancy** | RLS in PostgreSQL, index-per-tenant in ES, scoped connector tokens, quota controls on Kafka |
| **Audit** | Immutable audit trail in PostgreSQL for all actions and approvals |
| **Compliance** | Retention and export policies per regulation |

---

## 10. Quality Attributes

| Attribute | How Achieved |
|---|---|
| **Scalability** | Kafka absorbs bursts; independent horizontal scaling of microservices on K8s |
| **Resilience** | Dead-letter topics, retry/backoff, idempotent playbook steps, saga compensations |
| **Observability** | Centralized logs/traces/metrics (Prometheus, Grafana, OpenTelemetry, Sentry) |
| **Cost Control** | Storage tiering (hot/warm/cold), sampled telemetry, burst-aware scaling |
| **Extensibility** | Playbook SDK (YAML/DSL), Model Plug-ins (stable inference interface), Connector Framework (drop-in adapters) |

---

## 11. Human Interface Design

### Screen 1: Workflow Management
- Workflow list table with metadata (name, alias, creation time, save state, version).
- Search and filtering controls.
- Create Workflow control for structured creation flow.
- Save state and version indicators to prevent accidental execution of incomplete logic.

### Screen 2: Workflow Builder
- Visual node-based canvas for composing execution steps.
- Trigger Node (webhook trigger with configurable HTTP methods, IP allowlisting, API key auth).
- Trigger Configuration Panel with security settings.
- Run, Save, and Draft controls separating authoring from execution.

### Screen 3: AI-Driven Decision & Enforcement Workflow
- Decision Flow Graph: directed execution graph with sequential steps, conditional branches, converging paths.
- AI/Data Processing Nodes: entity extraction, ML triage, response parsing, risk score calculation.
- Policy and Approval Gates: embedded governance directly in workflow execution.
- Human-in-the-Loop: simulation and feedback nodes capture approval outcomes and rejection reasoning.
- Enforcement Nodes: execute only after risk evaluation and policy checks satisfied.
- Audit/Notification Nodes: publish feedback, generate immutable audit records, send notifications.

### Design Philosophy
> "Automation should accelerate operations while remaining transparent, reviewable, and aligned with organizational policy."
