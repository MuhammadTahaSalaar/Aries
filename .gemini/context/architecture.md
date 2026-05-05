# Architecture

## System Overview

ARIES is an event-driven microservices platform. Services communicate
asynchronously via Apache Kafka. All persistence is polyglot.

## Services

| App                      | Stack                                         | Port      | Owns                                               |
| ------------------------ | --------------------------------------------- | --------- | -------------------------------------------------- |
| `apps/dashboard`         | Next.js 15, React 19, tRPC, Prisma, shadcn/ui | 3000      | UI, auth, Prisma schema                            |
| `apps/go_service`        | Go 1.25+, Fiber/Gin, gRPC, pgx               | 8080/9090 | Orchestration, API Gateway, Connectors, Ingestion  |
| `apps/fastapi_service`   | FastAPI, PyTorch, HuggingFace, ONNX Runtime   | 8000      | ML Triage, NLP, Training, Analytics                |

## Kafka Topics

`alerts.raw`, `alerts.enriched`, `cases.updated`, `playbooks.events`,
`ml.feedback`

## Persistence Layer

- **PostgreSQL 16** — System of record. Row-Level Security (RLS) enforced.
  Every table has `tenant_id`. Database: `aries`.
- **Elasticsearch 8** — Search and telemetry. Index-per-tenant pattern.
- **Neo4j 5** — Graph relationships (IOC correlation, kill-chain timelines).
- **Redis 7** — Cache, idempotency keys, session state.
- **MinIO / S3** — MLflow artifact store. Bucket: `mlflow-bucket`.

## Multi-Tenancy

Every database query MUST include `WHERE tenant_id = $1` even when RLS is
active (defence-in-depth). `tenant_id` is extracted from the JWT on every
request.

## Auth

OAuth2/OIDC + RBAC. Roles: `Tier1_Analyst`, `Tier2_Analyst`, `SOC_Manager`,
`Security_Engineer`, `Admin`.

## Docker Stack Startup Order

1. `MLOps/docker-compose.yml` — creates `aries_network`
2. `apps/fastapi_service/docker-compose.yml` — joins `aries_network` as external
3. `wazuh-docker/single-node/docker-compose.yml` — joins `aries_network` as external

## Domain Entities (consistent across all services)

`Alert`, `Case`, `Playbook`, `PlaybookRun`, `Action`, `IOC`,
`ApprovalRequest`, `AuditLog`, `Tenant`, `User`

## Credentials (dev only — never commit to production)

| Service    | User    | Password         | Notes               |
| ---------- | ------- | ---------------- | ------------------- |
| MinIO      | admin   | password123      | bucket: mlflow-bucket |
| PostgreSQL | admin   | password123      | db: aries           |
| Wazuh API  | wazuh-wui | MyS3cr37P450r.*- |                     |
