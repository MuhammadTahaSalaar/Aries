# ARIES — Software Requirements Specification

> **AI-Enhanced SOAR: Optimizing Security Operations through Intelligent Automation**
> Source: SRS v1.0 (21-Oct-2025) | Authors: Sameed Ilyas, Muhammad Taha Salaar, Muhammad Zain

---

## 1. Product Scope

### 1.1 Goal
Develop a modular, cloud-native SOAR platform that enables **intelligent prioritization of alerts**, the **recommendation and execution of adaptive playbooks**, and the **extraction of insights via NLP**, reducing MTTD and MTTR for security incidents.

### 1.2 Benefits
- **Reduce False Positives** — ML filters, classifies, and prioritizes alerts for better signal-to-noise.
- **Adaptive Response** — Dynamic workflows adjust based on context, severity, and historical outcomes.
- **Accelerate Investigations** — NLP summarizes incidents and extracts IOCs from verbose security data.
- **Alleviate Analyst Fatigue** — Automate up to 70% of Tier-1 and Tier-2 triage tasks.
- **Strengthen Resilience** — Extensible solution that measurably strengthens cyber defense.

### 1.3 Objectives
1. Automatically filter and classify alerts using supervised and unsupervised ML models.
2. Dynamic orchestration that selects and executes context-aware Adaptive Playbooks.
3. NLP services for automated incident summarization and key entity extraction.
4. Human-in-the-Loop approval system maintaining analyst oversight for critical decisions.

---

## 2. User Classes

| User Role | Frequency | Primary Function |
|---|---|---|
| **Security Analyst (Tier-1, Tier-2)** | High | Triage, investigation, executing playbooks, Human-in-the-Loop approvals |
| **SOC Manager/Director** | Moderate/High | Dashboards, key metrics (MTTR, FP rate), managing playbooks, critical approvals |
| **Security Engineer/Integrator** | Moderate | Creating/maintaining connectors, deploying ML models, managing platform architecture |

### Security Analyst
- Views and interacts with prioritized alerts in the Triage Queue.
- Executes one-click playbooks, reviews NLP summaries and IOCs, provides feedback on alert classifications.
- Can initiate automated actions but requires approval for critical, destructive actions.

### SOC Manager
- High-level analytics and reporting views.
- Manages Approval Workflow for escalated incidents, customizes dashboards, reviews audit logs.
- Can access all performance and audit data, configure playbooks, manage user roles.

### Security Engineer
- Develops and deploys new Integration Connectors (adapters).
- Manages Kubernetes clusters, CI/CD pipelines, monitors service health via Prometheus/Grafana.
- Has technical access to backend, API keys (via Vault), and deployment manifests (Helm/Terraform).

---

## 3. Design and Implementation Constraints

| Constraint | Detail |
|---|---|
| **ML Model Accuracy** | >95% true-positive classification for high-severity alerts. Explainability via SHAP/LIME. |
| **Multitenancy** | Strict tenant isolation — row-level security in SQL, index-per-tenant in ES. |
| **Real-Time Performance** | End-to-end alert processing ≤ 2 seconds (P95 latency) for high-severity alerts. |
| **Integration Flexibility** | Connector Framework supporting Python and Go action runners for heterogeneous SIEMs/EDRs. |
| **Security & Compliance** | OAuth2/OIDC, TLS Everywhere, immutable audit logging. |

---

## 4. System Features & Functional Requirements

### 4.1 Intelligent Alert Triage & Prioritization (Priority: HIGH)

**Flow:** Raw alert → Ingestion Layer normalizes to canonical schema → Kafka Event Bus → ML Triage Engine enriches (asset criticality, IP reputation) and applies trained model → Risk Prioritization Score computed → Enriched alert indexed in Elasticsearch → Analyst Dashboard displays risk-sorted queue.

| ID | Requirement |
|---|---|
| **REQ-01** | **Canonical Alert Normalization** — Parse and convert all incoming vendor-specific alert formats into a single, standardized canonical alert schema. |
| **REQ-02** | **ML Score Generation** — Apply the currently deployed ML Triage Model to generate a confidence score for the True-Positive Likelihood of the alert. |
| **REQ-03** | **Risk Prioritization Formula** — Calculate the final Risk Prioritization Score using a configurable weighted formula of the ML score, asset criticality, and behavioral anomalies. |
| **REQ-04** | **Auto-Closure Threshold** — Automatically close alerts whose True-Positive Likelihood score falls below a configurable low threshold (e.g., <1%) to reduce noise. |

### 4.2 Adaptive Playbook Orchestration (Priority: HIGH)

**Flow:** Enriched alert on Kafka → Orchestration Engine queries Playbook Library → Selects matching playbook (e.g., "Phishing Triage and Isolation") → Initializes state machine → Executes actions → At critical decision points, pauses for Approval → On approval, resumes and logs immutable audit record → Executes remaining actions.

| ID | Requirement |
|---|---|
| **REQ-05** | **Dynamic Playbook Selection** — Select the most appropriate playbook based on normalized alert type, severity, and ML-generated playbook hints. |
| **REQ-06** | **Workflow Persistence** — Maintain state of long-running playbooks, supporting retries, timeouts, and compensatory actions on failure. |
| **REQ-07** | **Idempotent Actions** — All core response actions executed by connectors shall be idempotent to prevent adverse effects from retries. |
| **REQ-08** | **Case Creation and Correlation** — Automatically open a new Case record in PostgreSQL and correlate the current playbook run and all associated action logs with that Case ID. |

### 4.3 NLP-Powered Incident Summarization & IOC Extraction (Priority: HIGH)

**Flow:** Enriched alert with verbose log data → NLP Services (Hugging Face/spaCy) → NER identifies IPs, file hashes, usernames → Verified against threat intelligence → Marked as IOCs → On analyst request, generates executive-level summary citing affected systems and recommended actions.

| ID | Requirement |
|---|---|
| **REQ-09** | **NLP Entity Extraction** — Use NLP models to accurately extract IOCs, hosts, users, and processes from raw, unstructured security data and log snippets. |
| **REQ-10** | **Automatic Summarization** — Generate two distinct narrative summaries: a brief high-level executive summary and a detailed analyst summary. |
| **REQ-11** | **IOC Highlighting** — Dynamically highlight and annotate all verified IOCs and extracted entities within the raw alert data for quick visual identification. |
| **REQ-12** | **Timeline Reconstruction** — Use extracted entities and timestamps to reconstruct and display a case timeline aligned with MITRE ATT&CK. |

### 4.4 Human-in-the-Loop Approval Workflow (Priority: HIGH)

**Flow:** Playbook attempts critical action → Orchestration Engine pauses playbook → Sends notification to Approval Workflow service → SOC Manager reviews context and NLP summary → Approves (immutable record in SQL, playbook resumes) or Rejects (mandatory justification, logged to ML feedback topic).

| ID | Requirement |
|---|---|
| **REQ-13** | **Policy-Based Gates** — Enforce policy-based gates requiring approval for specific actions based on asset criticality, alert severity, or action type. |
| **REQ-14** | **Immutable Audit Trail** — Record all approval and rejection actions, including user, timestamp, and context, as an immutable audit trail in the SQL database. |
| **REQ-15** | **Contextual Justification** — On rejection, require mandatory textual justification fed back to the Model Retraining Loop. |
| **REQ-16** | **Auto-Escalation** — If critical approval not received within configurable timeout (e.g., 5 minutes), automatically escalate to next tier of management. |

### 4.5 Real-Time Case Update Streaming (Priority: HIGH)

**Flow:** Analyst opens Triage Queue → Persistent WebSocket connection established → Playbook action completes → Orchestration Engine publishes `cases.updated` to Kafka → WebSocket service consumes event → Pushes JSON payload to all active sessions viewing that case → Dashboard updates status from "Running" to "Completed" instantly.

| ID | Requirement |
|---|---|
| **REQ-17** | **WebSocket Connectivity** — Utilize WebSockets as the primary method for streaming real-time case updates and new alert notifications to the UI. |
| **REQ-18** | **State Change Events** — Orchestration Engine publishes a structured event to Kafka upon every significant state change. |
| **REQ-19** | **Fault Tolerance** — Client must automatically re-establish lost WebSocket connections and retrieve missed state updates. |
| **REQ-20** | **Event Throttling** — WebSocket service implements throttling to manage high volumes without degrading browser performance. |

### 4.6 ML Model Retraining Loop (Priority: MEDIUM)

**Flow:** Analyst overrides ML classification → `ml.feedback` event published to Kafka → Analytics Engine monitors accumulated feedback → Threshold exceeded → Schedules retraining job in MLflow → Fetches labeled data + feedback → Trains new model version → Canary/batch validation → Promotes new model artifact to inference API.

| ID | Requirement |
|---|---|
| **REQ-21** | **Feedback Collection** — Capture and publish explicit and implicit analyst feedback (manual triage, approvals, overrides) to a dedicated ML feedback Kafka topic. |
| **REQ-22** | **Drift Detection** — Monitor live ML Triage Model performance and detect significant accuracy drops to trigger retraining. |
| **REQ-23** | **Model Versioning** — Version all trained models using MLflow. Allow quick rollback to previous version if newly deployed model shows degraded performance. |
| **REQ-24** | **Scheduled Retraining** — Allow SOC Manager to configure a fixed schedule (weekly, monthly) for automatic model retraining. |

### 4.7 Multi-Tenant Data Isolation (Priority: HIGH)

**Flow:** User (Tenant X) sends request via API Gateway → Token verified, Tenant X ID attached → SQL Cluster enforces Row-Level Security (RLS), returns only Tenant X records → Elasticsearch restricts search to Tenant Y's index/alias → Playbook execution scoped only to intended tenant.

| ID | Requirement |
|---|---|
| **REQ-25** | **Tenant ID Enforcement** — All data records (alerts, cases, logs) immutably tagged with Tenant ID upon creation. |
| **REQ-26** | **Row-Level Security** — PostgreSQL implements RLS ensuring users/services only query data matching their assigned Tenant ID. |
| **REQ-27** | **Logical Storage Separation** — Elasticsearch uses index-per-tenant or filtered aliases to prevent cross-tenant data leakage. |
| **REQ-28** | **Resource Quotas** — Implement quotas on shared resources (Kafka topics, worker queues) to prevent one tenant's load from degrading others. |

---

## 5. Non-Functional Requirements

### 5.1 Performance

| Metric | Requirement |
|---|---|
| **Alert Processing Latency** | Median end-to-end ≤ 2 seconds (P50) for high-severity alerts |
| **Playbook Execution Speed** | Automated action completes within 150ms (P95), excluding external API latency |
| **UI/Dashboard Rendering** | Full load and render within 3 seconds |
| **System Scalability** | ≥ 500 alerts/second burst capacity, 100 simultaneous active playbook executions |

### 5.2 Safety
- Comprehensive error handling with dead-letter queues (Kafka) and graceful degradation.
- Reliable data backup and recovery for all persistent stores (PostgreSQL, Elasticsearch).
- Prompt notification of significant system errors to analysts and managers.
- Strict input validation on API Gateway for all incoming payloads.

### 5.3 Security
- **Authentication:** OAuth2/OpenID Connect (OIDC) for enterprise identity provider integration.
- **Authorization:** Role-Based Access Control (RBAC) enforced at API Gateway.
- **Encryption in Transit:** TLS Everywhere for all internal and external communications.
- **Encryption at Rest:** Sensitive data encrypted at rest. HashiCorp Vault for dynamic secret management.
- **API Security:** Scoped per-tenant API tokens managed by Vault. Trivy for container vulnerability scanning.

### 5.4 Software Quality Attributes

| Attribute | Target |
|---|---|
| **Usability** | SUS score ≥ 75 |
| **Availability** | Core services uptime ≥ 98% |
| **Reliability** | ML Triage Engine reliability coefficient ≥ 0.90 |
| **Testability** | Unit, Integration, Load/Stress, and UAT testing required |

### 5.5 Business Rules
- Critical actions (isolating production server, disabling executive account) require L2 (SOC Manager) approval. (REQ-13)
- Approval requests not actioned within 5 minutes auto-escalate to on-call SOC Director. (REQ-16)
- Alerts with True-Positive Likelihood < 1% auto-closed as "Informational" or "False Positive". (REQ-04)

---

## 6. External Interface Requirements

### 6.1 User Interfaces (Analyst Dashboard)
Built with **React.js + Next.js**, styled with **TailwindCSS + shadcn/ui**, real-time via **WebSockets**.

#### Triage and Queue Management Screen
- Risk-sorted alert list (filterable by source, severity, ML score).
- Alert cards with normalized title, primary IOCs, ML confidence visualization.
- One-click action bar for low-risk playbook actions + link to Approval Workflow for critical actions.
- Real-time updates via WebSockets.

#### Case Management and Timeline View
- Case details with normalized data, asset criticality, selected playbook.
- Chronological action timeline (immutable audit trail).
- NLP Summarization output: incident summary + verified IOC list.
- Evidence Viewer linking to raw data in Elasticsearch.

#### Reporting and Analytics Dashboard
- MTTD, MTTR, False-Positive Rate trends.
- Automation efficacy charts (auto-closed vs manual, by playbook/type).
- Model performance monitoring (drift, version, confidence).
- Governance: Audit Logs for all critical approvals and actions.

### 6.2 Communications Interfaces
- **WebSockets (port 8000):** Real-time push updates to dashboard.
- **HTTPS (port 443):** Secure client-to-API-Gateway communication with TLS/SSL.
- **gRPC:** Internal microservice-to-microservice traffic (high speed, strongly typed).
- **REST:** Public API for universal compatibility with third-party tools.
- **TCP/IP:** Internal cloud infrastructure communication (Kafka brokers, database clusters).

---

## 7. Tech Stack Summary

| Component | Technology |
|---|---|
| Orchestration/Control | Go (Golang), Temporal.io or Custom Engine |
| API/ML Gateway | FastAPI (Python), Next.js API Routes (BFF) |
| AI/ML Core | PyTorch, Hugging Face Transformers, spaCy, ONNX Runtime, MLflow |
| Messaging | Apache Kafka |
| Databases | PostgreSQL, Elasticsearch, Neo4j |
| Caching | Redis Cluster |
| Object Storage | Amazon S3 |
| Frontend/UI | React.js, Next.js, TypeScript, TailwindCSS, shadcn/ui, WebSockets |
| Visualization | Kibana (ELK Stack) |
| DevOps/Infra | Docker, Kubernetes, Helm, Terraform |
| CI/CD | GitHub Actions, Nx Monorepo, Trivy |
| Monitoring | Prometheus, Grafana, OpenTelemetry, Sentry |
| Security | OAuth2/OIDC, HashiCorp Vault, TLS Everywhere |
| Data Ingestion | Logstash |
| Streaming Analytics | Apache Spark Structured Streaming |

---

## 8. Assumptions and Dependencies

### Assumptions
- Sufficient historical security alert data (labeled TP/FP) available for initial ML model training.
- Cloud infrastructure (GKE, managed databases, object storage, GPU access) available.
- Reliable and documented API access to at least one major SIEM and one EDR product.

### Dependencies
- ML Libraries: PyTorch, Hugging Face Transformers, spaCy, ONNX Runtime.
- Containerization: Docker and Kubernetes maturity and support.
- Message Broker: Apache Kafka reliability and performance.

---

## 9. TBD Items
1. Specific ML model accuracy threshold for non-high-severity alerts.
2. Full list of third-party SIEM/EDR products to be supported by v1.0.
3. Specific resource quotas for tenants (REQ-28).
