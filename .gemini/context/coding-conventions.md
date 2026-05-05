# Coding Conventions

These apply across **all services** in the ARIES monorepo.

## Universal Rules

- **No secrets in code.** All credentials come from environment variables.
- **Tenant isolation everywhere.** Every query or data operation must scope to
  `tenant_id`.
- **Prefer async I/O.** Use async/await in Python (FastAPI), Go goroutines +
  context cancellation, and async in Next.js server components.
- **Fail fast on misconfiguration.** Validate all required env vars at startup.
- **Idempotency.** All connector actions and Kafka message handlers must be
  idempotent.

## Python / FastAPI

- Python 3.11+.
- `ruff` for linting and formatting; `mypy` for type checking.
- Pydantic v2 for all schemas. Validate at system boundaries only.
- Use `asyncpg` for PostgreSQL, `aiokafka` for Kafka, `httpx.AsyncClient` for
  outbound HTTP.
- `pytest` + `pytest-asyncio` for tests; mock all external services.
- ONNX Runtime for production inference. PyTorch only for training.

## Go Service

- Go 1.25+.
- Wrap errors with context: `fmt.Errorf("selecting playbook: %w", err)`.
- Define interfaces at the consumer, not the provider.
- Pass `context.Context` as first parameter to every function that does I/O.
- Use `zap.Logger` with structured fields (`zap.String("tenant_id", tid)`).
  No `fmt.Println`.
- Table-driven tests with `testify/assert`.

## Next.js Dashboard

- Next.js 15 + React 19 + tRPC + Prisma + shadcn/ui.
- TypeScript strict mode.
- All DB access goes through Prisma. Never write raw SQL in components.
- RBAC checks in tRPC middleware; never trust client-side role data.

## Docker / Infrastructure

- Multi-stage Dockerfiles. Non-root runtime user.
- `HEALTHCHECK` in every Dockerfile.
- Pin base image versions.
- `.env.example` committed; `.env` gitignored.

## Git

- Conventional Commits (`feat:`, `fix:`, `chore:`, etc.).
- Keep PRs small and focused.
- Never force-push to `main`.

## AI / ML Specific

- ONNX Runtime is the production inference engine — do not load PyTorch models
  at inference time.
- Do not add instruction prefixes to BART-base (not instruction-tuned).
- IOC post-processing (regex fallback) runs **after** model inference, not
  before.
- Risk score thresholds: `>= 50 → TruePositive`; `>= 35 + BenignPositive →
  FalsePositive`; `auto_close` only when `risk_score < 35`.
