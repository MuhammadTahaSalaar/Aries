# ARIES — Implementation Tasks

> Master task list for building the AI-Enhanced SOAR platform.
> Each task references requirements from `REQUIREMENTS.md` and design from `DESIGN.md`.

---

## Current Status (as of Feb 2026)

**Overall: ~5% scaffolded, 0% ARIES business logic implemented.**

### What Exists
- Nx monorepo with Next.js dashboard app, Go service stub, FastAPI Dockerfile
- Next.js 15 + React 19 + TypeScript + TailwindCSS + shadcn/ui
- tRPC client/server setup with a hello endpoint
- Better Auth configured with Prisma adapter (email/password only)
- Prisma schema with User, Session, Account, Verification models (auth only)
- Login form UI component (non-functional submit)
- Minimal Go HTTP server ("Hello from Go Service")
- Docker Compose for three app services (no infra services)
- GitHub Actions CI (lint, test, build)

### What Is Missing
Everything in the task list below.

---

## Phase 1 — Foundation & Infrastructure

### T-001: PostgreSQL Schema — Full ARIES Domain Model
**Priority:** Critical | **Refs:** REQ-01, REQ-08, REQ-14, REQ-25, REQ-26 | **Design:** §4, §7

Extend the existing Prisma schema (or raw SQL migrations) to include:

- [ ] `Tenant` table (id, name, settings, created_at)
- [ ] `Alert` table (alert_id, tenant_id, normalized_title, raw_data JSON, ml_score, risk_score, status enum [New, Triaged, Closed_FP, Escalated], source, mitre_tactic, created_at, updated_at)
- [ ] `Case` table (case_id, tenant_id, title, summary, status enum [Detected, Open, Investigating, Awaiting_Approval, Contained, Closed], severity enum [Low, Medium, High, Critical], assigned_analyst_id FK, created_at, updated_at)
- [ ] `Playbook` table (playbook_id, tenant_id, name, description, trigger_conditions JSON, is_active, created_at)
- [ ] `PlaybookRun` table (run_id, playbook_id, case_id, tenant_id, status, started_at, completed_at)
- [ ] `Action` table (action_id, playbook_run_id, action_type, status enum [Pending, Running, Completed, Failed, Awaiting_Approval], result JSON, started_at, completed_at)
- [ ] `IOC` table (ioc_id, alert_id, tenant_id, type enum [IP_Address, File_Hash, Domain, URL, Email_Address], value, source, reputation_score, created_at)
- [ ] `ApprovalRequest` table (id, action_id, case_id, tenant_id, requested_by, approved_by, status, justification, created_at, resolved_at)
- [ ] `AuditLog` table (id, tenant_id, actor_id, action, entity_type, entity_id, details JSON, created_at) — immutable
- [ ] `AlertCase` junction table (alert_id, case_id) — many-to-many
- [ ] Row-Level Security (RLS) policies on all tenant-scoped tables
- [ ] Indexes: risk_score DESC on Alert, status on Case, created_at on AuditLog
- [ ] Seed data for development (sample tenant, alerts, cases)

### T-002: Docker Compose — Full Infrastructure Stack
**Priority:** Critical | **Refs:** Design §3.2, §3.5

Extend `docker-compose.yml` to include all infrastructure services:

- [ ] PostgreSQL 16 with persistent volume, health check, init SQL for RLS
- [ ] Apache Kafka (KRaft mode, no ZooKeeper) + kafka-ui for dev visibility
- [ ] Elasticsearch 8 with single-node config for dev
- [ ] Redis 7 cluster (or standalone for dev)
- [ ] Neo4j 5 with persistent volume
- [ ] MinIO (S3-compatible) for local object storage
- [ ] Network configuration so all services can communicate
- [ ] Environment variable files (.env.example) for all service connections
- [ ] Health check dependencies (apps wait for infra)

### T-003: Nx Workspace — Service Scaffolding
**Priority:** Critical | **Refs:** Design §2

Ensure proper Nx project structure for all services:

- [ ] `apps/dashboard` — Next.js (exists, needs restructure)
- [ ] `apps/api-gateway` — Go API Gateway service
- [ ] `apps/orchestration-engine` — Go Orchestration Engine
- [ ] `apps/ml-triage-service` — FastAPI ML Triage service
- [ ] `apps/nlp-service` — FastAPI NLP service
- [ ] `apps/ingestion-service` — Go/Python Alert Ingestion service
- [ ] `libs/shared-types` — Shared TypeScript types/schemas (tRPC, API contracts)
- [ ] `libs/proto` — gRPC protobuf definitions
- [ ] `libs/kafka-schemas` — Avro/JSON schemas for Kafka topics
- [ ] Dockerfiles for each service
- [ ] Nx project.json for each service (build, serve, test, lint, docker-build targets)

### T-004: Kafka Topic Setup & Event Schemas
**Priority:** Critical | **Refs:** Design §3.2

- [ ] Topic creation script: `alerts.raw`, `alerts.enriched`, `cases.updated`, `playbooks.events`, `ml.feedback`
- [ ] JSON Schema or Avro definitions for each topic's message format
- [ ] Dead-letter topics for each main topic
- [ ] Producer/consumer helper libraries for Go and Python

### T-005: Authentication & Authorization — Full RBAC
**Priority:** High | **Refs:** REQ-13, Design §9

Extend the existing Better Auth setup:

- [ ] Add `role` field to User model (Tier1_Analyst, Tier2_Analyst, SOC_Manager, Security_Engineer, Admin)
- [ ] Add `tenant_id` field to User model
- [ ] Implement RBAC middleware for tRPC routes
- [ ] Implement RBAC middleware for API Gateway (Go)
- [ ] OAuth2/OIDC provider integration (at least one: Google or GitHub)
- [ ] Session management with Redis
- [ ] API key generation and validation for service-to-service auth
- [ ] Protected route wrappers for dashboard pages

---

## Phase 2 — Core Pipeline (Alert Ingestion → Triage → Dashboard)

### T-006: Alert Ingestion Service
**Priority:** Critical | **Refs:** REQ-01, Design §3.1

- [ ] HTTP/webhook endpoint to receive raw alerts (JSON payloads)
- [ ] Alert Normalizer: parse vendor-specific payloads into canonical alert schema
- [ ] Field validation and deduplication (using Redis dedupe keys)
- [ ] MITRE ATT&CK tactic/technique mapping (basic lookup table)
- [ ] Publish normalized alert to `alerts.raw` Kafka topic
- [ ] Logstash config for log collection from external sources (optional, can be Phase 3)
- [ ] Health check and metrics endpoints (Prometheus format)
- [ ] Unit tests and integration tests

### T-007: ML Triage Engine — FastAPI Service
**Priority:** Critical | **Refs:** REQ-02, REQ-03, REQ-04, Design §3.4

- [ ] FastAPI service scaffolding with proper project structure
- [ ] Kafka consumer: subscribe to `alerts.raw`
- [ ] Context Enrichment module: fetch asset inventory, IP reputation (stub initially)
- [ ] ML Model inference endpoint (start with a rule-based/heuristic scorer, replace with trained model later)
- [ ] True-Positive Likelihood score calculation (0.0–1.0)
- [ ] Risk Prioritization Score formula: weighted combination of ml_score, asset_criticality, behavioral_anomalies
- [ ] Auto-closure logic: alerts with ml_score < configurable threshold auto-closed
- [ ] Publish enriched alert to `alerts.enriched` Kafka topic
- [ ] Persist enriched alert to Elasticsearch
- [ ] Persist alert record to PostgreSQL
- [ ] ONNX Runtime integration for model serving (Phase 3 refinement)
- [ ] Health check and metrics endpoints
- [ ] Unit tests with mock data

### T-008: Analyst Dashboard — Triage Queue
**Priority:** Critical | **Refs:** REQ-17, Design §3.6, §11

- [ ] Triage Queue page: risk-sorted list of alerts
- [ ] Alert card component: normalized title, severity badge, ml_score visualization (progress bar/gauge), primary IOCs, timestamp
- [ ] Filtering: by source, severity, status, date range
- [ ] Sorting: by risk_score, ml_score, created_at
- [ ] Search: full-text search across alert titles
- [ ] Alert detail drawer/modal: full enriched data, raw payload viewer
- [ ] One-click action buttons: "Enrich IP", "Create Case", "Close as FP"
- [ ] Real-time updates via WebSocket (new alerts appear, statuses update live)
- [ ] tRPC routes for alert CRUD operations
- [ ] Pagination (cursor-based for performance)

### T-009: WebSocket Real-Time Service
**Priority:** High | **Refs:** REQ-17, REQ-18, REQ-19, REQ-20

- [ ] WebSocket server (Next.js API route or standalone Go service)
- [ ] Kafka consumer: subscribe to `alerts.enriched`, `cases.updated`, `playbooks.events`
- [ ] Broadcast events to connected clients (scoped by tenant)
- [ ] Auto-reconnection logic on client side
- [ ] Event throttling for high-volume periods
- [ ] Connection health monitoring

### T-010: Elasticsearch Integration
**Priority:** High | **Refs:** REQ-27, Design §3.5

- [ ] Elasticsearch client setup for Go and Python services
- [ ] Index templates for alerts (per-tenant index strategy or filtered aliases)
- [ ] Alert document mapping (all canonical fields + full-text searchable)
- [ ] Search API: full-text search, filtered queries, aggregations
- [ ] Dashboard integration: feed Triage Queue from ES queries
- [ ] Index lifecycle management (hot/warm/cold tiering for cost)

---

## Phase 3 — Orchestration & Case Management

### T-011: Orchestration Engine — Go Service
**Priority:** Critical | **Refs:** REQ-05, REQ-06, REQ-07, REQ-08, Design §3.3

- [ ] Kafka consumer: subscribe to `alerts.enriched` (high-priority alerts)
- [ ] Playbook selection logic: match alert characteristics to playbook trigger_conditions
- [ ] State machine / workflow runtime (custom Go engine or Temporal.io SDK)
- [ ] Playbook execution loop: sequential and parallel action support
- [ ] Retry logic with exponential backoff and configurable max retries
- [ ] Timeout handling per action and per playbook
- [ ] Compensation/rollback actions on failure (saga pattern)
- [ ] Idempotent action execution (REQ-07)
- [ ] Case creation: auto-create Case record in PostgreSQL on playbook start
- [ ] Alert-to-Case correlation: link triggering alert(s) to created case
- [ ] Publish state changes to `playbooks.events` and `cases.updated` Kafka topics
- [ ] gRPC server for internal service communication
- [ ] Health check and metrics endpoints
- [ ] Unit and integration tests

### T-012: Integration Connector Framework
**Priority:** High | **Refs:** Design §3.3, §4 (IntegrationConnector interface)

- [ ] Define `IntegrationConnector` interface in Go (and Python equivalent)
- [ ] Connector registry: discover and load connectors dynamically
- [ ] Base connector with common functionality: auth, retry, rate-limiting, error mapping
- [ ] Stub connectors for development:
  - [ ] `MockEDRConnector` — simulates quarantine_host, scan_host
  - [ ] `MockSIEMConnector` — simulates log_query, alert_fetch
  - [ ] `MockEmailConnector` — simulates purge_email, block_sender
  - [ ] `MockTicketConnector` — simulates create_ticket (Jira/ServiceNow)
- [ ] Async IO and pagination support
- [ ] Secrets retrieval from environment/Vault for connector credentials

### T-013: Case Management — Dashboard & API
**Priority:** High | **Refs:** REQ-08, Design §3.6

- [ ] Case list page: filterable by status, severity, assigned analyst
- [ ] Case detail page:
  - [ ] Case metadata (title, severity, status, assigned analyst)
  - [ ] Associated alerts list
  - [ ] Action Timeline: chronological log of all automated and manual actions (immutable)
  - [ ] NLP Summary section (placeholder until NLP service is built)
  - [ ] IOC list with highlighting
  - [ ] Evidence viewer linking to Elasticsearch raw data
- [ ] Case status transitions (following state machine: Detected → Open → Investigating → Contained → Closed)
- [ ] Analyst assignment
- [ ] tRPC routes for all case operations
- [ ] Case creation from alert (one-click)

### T-014: Approval Workflow
**Priority:** High | **Refs:** REQ-13, REQ-14, REQ-15, REQ-16, Design §6.3

- [ ] Approval Service (Go or Next.js API)
- [ ] When orchestration engine hits critical action → create ApprovalRequest record
- [ ] Notification to SOC Manager (WebSocket push + optional email/Slack)
- [ ] Approval dashboard screen:
  - [ ] Pending approvals list
  - [ ] Action context display (incident summary, requested action, risk assessment)
  - [ ] Approve / Reject buttons
  - [ ] Mandatory justification field on rejection
- [ ] Immutable audit trail: all approve/reject decisions logged to AuditLog table
- [ ] Auto-escalation: configurable timeout (default 5 min) → escalate to next tier
- [ ] Signal back to Orchestration Engine (resume or terminate playbook)
- [ ] Rejection feedback published to `ml.feedback` Kafka topic

### T-015: Playbook Management UI
**Priority:** Medium | **Refs:** Design §11

- [ ] Playbook list page: all playbooks with metadata
- [ ] Playbook builder screen:
  - [ ] Visual node-based canvas for composing workflow steps
  - [ ] Trigger configuration (alert type, severity, source matching)
  - [ ] Action nodes (drag-and-drop from available actions)
  - [ ] Decision/branch nodes (conditional logic)
  - [ ] Approval gate nodes
  - [ ] Save/Draft/Publish workflow
- [ ] Playbook YAML/JSON import/export (SDK)
- [ ] Pre-built playbook templates:
  - [ ] Phishing Triage and Isolation
  - [ ] Brute-Force Login Response
  - [ ] Malware Containment
  - [ ] Suspicious IP Investigation

---

## Phase 4 — AI & NLP Services

### T-016: NLP Service — Entity Extraction & Summarization
**Priority:** High | **Refs:** REQ-09, REQ-10, REQ-11, REQ-12, Design §3.4

- [ ] FastAPI service scaffolding
- [ ] NER endpoint: extract IOCs (IPs, file hashes, domains, URLs, email addresses), hosts, users, processes from unstructured text
  - [ ] Hugging Face Transformers + spaCy pipeline
  - [ ] Custom security-domain NER model or fine-tuned model
- [ ] Summarization endpoint:
  - [ ] Executive summary (1-3 sentences)
  - [ ] Analyst summary (detailed with citations to evidence)
- [ ] IOC verification: cross-reference extracted IOCs against threat intelligence feeds (stub/mock initially)
- [ ] Timeline reconstruction: extract timestamps and events, align to MITRE ATT&CK
- [ ] Kafka integration: consume from `alerts.enriched`, publish enriched IOCs
- [ ] REST API for on-demand summarization (called from dashboard)
- [ ] Persist extracted IOCs to PostgreSQL IOC table
- [ ] Unit tests with sample security logs

### T-017: ML Model Training Pipeline
**Priority:** Medium | **Refs:** REQ-21, REQ-22, REQ-23, REQ-24, Design §3.4

- [ ] MLflow setup (containerized, connected to S3/MinIO for artifact storage)
- [ ] Training data pipeline:
  - [ ] Data loader from PostgreSQL (historical alerts with labels)
  - [ ] Feature engineering: entity reputation, behavioral patterns, asset criticality
- [ ] Model training scripts (PyTorch):
  - [ ] Alert classification (True Positive vs False Positive)
  - [ ] Severity scoring
  - [ ] Playbook recommendation
- [ ] Model evaluation and validation pipeline
- [ ] ONNX export for production inference
- [ ] Model versioning and registry in MLflow
- [ ] Canary deployment logic: compare new model vs current in A/B fashion
- [ ] Drift detection: monitor prediction distribution changes
- [ ] Scheduled retraining (configurable cron)

### T-018: ML Feedback Loop
**Priority:** Medium | **Refs:** REQ-21, REQ-22

- [ ] Kafka consumer for `ml.feedback` topic
- [ ] Aggregate analyst feedback (triage overrides, approval decisions)
- [ ] Store feedback as training labels in PostgreSQL
- [ ] Trigger retraining when feedback volume or drift exceeds threshold
- [ ] Dashboard widget: model performance metrics (accuracy, F1, drift indicator)

---

## Phase 5 — Advanced Features & Integrations

### T-019: Neo4j Graph Database — Threat Correlation
**Priority:** Medium | **Refs:** Design §3.5

- [ ] Neo4j schema: Node types (Alert, IOC, Asset, User, Case), Relationship types (CONTAINS, AFFECTS, ASSOCIATED_WITH)
- [ ] Data ingestion pipeline: sync relevant data from PostgreSQL/ES to Neo4j
- [ ] Graph queries: attack-path analysis, lateral movement detection, IOC correlation
- [ ] API endpoint for graph queries (consumed by dashboard)
- [ ] Dashboard visualization: interactive graph view of threat relationships

### T-020: Reporting & Analytics Dashboard
**Priority:** High | **Refs:** Design §3.6

- [ ] Dashboard page for SOC Manager:
  - [ ] MTTD trend chart (line graph over time)
  - [ ] MTTR trend chart
  - [ ] False-Positive Rate trend
  - [ ] Alert volume heatmap (by hour, day)
  - [ ] Automation efficacy: % auto-closed vs manual, by playbook type
  - [ ] Model performance: accuracy, version, confidence levels
- [ ] Customizable widgets (drag-and-drop layout)
- [ ] Date range selector for all charts
- [ ] Export to PDF/CSV
- [ ] Role-based view (Manager sees different default than Analyst)

### T-021: Multi-Tenant Isolation — Full Implementation
**Priority:** High | **Refs:** REQ-25, REQ-26, REQ-27, REQ-28

- [ ] Tenant management API (CRUD tenants)
- [ ] Tenant context middleware: extract tenant_id from JWT, inject into all queries
- [ ] PostgreSQL RLS policies enforced and tested
- [ ] Elasticsearch index-per-tenant strategy (or filtered aliases)
- [ ] Kafka topic partitioning by tenant (or message filtering)
- [ ] Redis key namespacing by tenant
- [ ] Resource quotas per tenant (configurable limits)
- [ ] Tenant switching UI for admin users
- [ ] Integration tests verifying cross-tenant data isolation

### T-022: API Gateway — Go Service
**Priority:** High | **Refs:** Design §3.3

- [ ] HTTP/REST + gRPC gateway
- [ ] OAuth2/OIDC token validation middleware
- [ ] RBAC enforcement (role → allowed endpoints mapping)
- [ ] Rate limiting (per-tenant, per-user)
- [ ] Request signing and validation
- [ ] Per-tenant routing to appropriate service instances
- [ ] OpenAPI/Swagger auto-generation
- [ ] Health check aggregation (checks all downstream services)
- [ ] Request logging and correlation ID propagation

---

## Phase 6 — DevOps, Monitoring & Deployment

### T-023: CI/CD Pipeline — Full Implementation
**Priority:** High | **Refs:** Design §10

- [ ] Fix existing CI pipeline (directory naming: `fastapi_service` vs `fastapi-service`)
- [ ] Nx affected builds only (selective CI based on changed files)
- [ ] Docker image builds for all services
- [ ] Container vulnerability scanning with Trivy
- [ ] Integration test stage (spin up Docker Compose, run E2E tests)
- [ ] Deploy stage (to staging environment)
- [ ] Environment-specific configs (dev, staging, production)

### T-024: Observability Stack
**Priority:** Medium | **Refs:** Design §10

- [ ] Prometheus: metrics collection from all services
- [ ] Grafana: dashboards for system health, Kafka lag, DB performance, API latency
- [ ] OpenTelemetry: distributed tracing across Go, Python, Next.js services
- [ ] Sentry: runtime error tracking for frontend and backend
- [ ] Structured logging (JSON format) across all services
- [ ] Alerting rules (Prometheus/Grafana) for critical system conditions
- [ ] Docker Compose additions for monitoring stack

### T-025: Kubernetes Deployment
**Priority:** Medium | **Refs:** Design §10

- [ ] Helm charts for each service
- [ ] Kubernetes manifests: Deployments, Services, ConfigMaps, Secrets
- [ ] Horizontal Pod Autoscaler (HPA) configs for ML inference and ingestion
- [ ] Ingress controller configuration
- [ ] Persistent Volume Claims for databases
- [ ] Network Policies for service-to-service security
- [ ] Health check probes (liveness, readiness, startup)

### T-026: Terraform — Infrastructure as Code
**Priority:** Low (for production) | **Refs:** Design §10

- [ ] GKE cluster provisioning
- [ ] Cloud SQL (PostgreSQL) provisioning
- [ ] Elasticsearch service provisioning
- [ ] Redis (Memorystore) provisioning
- [ ] Cloud Storage (S3) buckets
- [ ] VPC and networking
- [ ] IAM roles and service accounts

---

## Phase 7 — Testing & Documentation

### T-027: Testing Strategy
**Priority:** High

- [ ] Unit tests for all services (Jest for TS, pytest for Python, Go testing for Go)
- [ ] Integration tests (service-to-service with test containers)
- [ ] E2E tests with Playwright (full user workflows):
  - [ ] Login → View Triage Queue → Click Alert → Create Case → Execute Playbook → Approve Action → Close Case
- [ ] Load/stress testing (k6 or Locust): validate 500 alerts/sec, 100 concurrent playbooks
- [ ] Security testing: OWASP ZAP scan, dependency audit
- [ ] ML model testing: accuracy benchmarks, bias detection

### T-028: Documentation
**Priority:** Medium

- [ ] OpenAPI/Swagger specs for all REST APIs
- [ ] gRPC protobuf documentation
- [ ] Playbook SDK guide (YAML/DSL format, how to create custom playbooks)
- [ ] Integration Connector development guide
- [ ] Deployment guide (Docker Compose local + Kubernetes production)
- [ ] User manual (Analyst workflow, Manager workflow, Engineer workflow)

---

## Dependency Graph (Suggested Build Order)

```
T-002 (Docker Compose Infra)
  │
  ├─► T-001 (PostgreSQL Schema)
  │     │
  │     ├─► T-005 (Auth & RBAC)
  │     │     │
  │     │     └─► T-008 (Triage Queue UI)
  │     │
  │     ├─► T-004 (Kafka Topics)
  │     │     │
  │     │     ├─► T-006 (Alert Ingestion)
  │     │     │     │
  │     │     │     └─► T-007 (ML Triage Engine)
  │     │     │           │
  │     │     │           └─► T-010 (Elasticsearch)
  │     │     │                 │
  │     │     │                 └─► T-008 (Triage Queue UI)
  │     │     │
  │     │     └─► T-009 (WebSocket Service)
  │     │
  │     └─► T-003 (Nx Scaffolding)
  │
  ├─► T-011 (Orchestration Engine) ─► T-012 (Connectors) ─► T-014 (Approval Workflow)
  │
  ├─► T-013 (Case Management)
  │
  ├─► T-016 (NLP Service) ─► T-017 (ML Training Pipeline) ─► T-018 (Feedback Loop)
  │
  ├─► T-019 (Neo4j) ─► T-020 (Reporting Dashboard)
  │
  └─► T-021 (Multi-Tenant) ─► T-022 (API Gateway)
        │
        └─► T-023 (CI/CD) ─► T-024 (Observability) ─► T-025 (K8s) ─► T-026 (Terraform)
```

---

## Quick Reference: Requirement → Task Mapping

| Requirement | Task(s) |
|---|---|
| REQ-01 (Canonical Normalization) | T-006 |
| REQ-02 (ML Score Generation) | T-007 |
| REQ-03 (Risk Prioritization) | T-007 |
| REQ-04 (Auto-Closure) | T-007 |
| REQ-05 (Playbook Selection) | T-011 |
| REQ-06 (Workflow Persistence) | T-011 |
| REQ-07 (Idempotent Actions) | T-011, T-012 |
| REQ-08 (Case Creation) | T-011, T-013 |
| REQ-09 (NLP Entity Extraction) | T-016 |
| REQ-10 (Auto Summarization) | T-016 |
| REQ-11 (IOC Highlighting) | T-016, T-008 |
| REQ-12 (Timeline Reconstruction) | T-016, T-013 |
| REQ-13 (Policy-Based Gates) | T-014 |
| REQ-14 (Immutable Audit Trail) | T-001, T-014 |
| REQ-15 (Contextual Justification) | T-014 |
| REQ-16 (Auto-Escalation) | T-014 |
| REQ-17 (WebSocket) | T-009 |
| REQ-18 (State Change Events) | T-009, T-011 |
| REQ-19 (Fault Tolerance) | T-009 |
| REQ-20 (Event Throttling) | T-009 |
| REQ-21 (Feedback Collection) | T-018 |
| REQ-22 (Drift Detection) | T-017, T-018 |
| REQ-23 (Model Versioning) | T-017 |
| REQ-24 (Scheduled Retraining) | T-017 |
| REQ-25 (Tenant ID Enforcement) | T-001, T-021 |
| REQ-26 (Row-Level Security) | T-001, T-021 |
| REQ-27 (Logical Storage Separation) | T-010, T-021 |
| REQ-28 (Resource Quotas) | T-021 |
