# FastAPI Service Conventions

The FastAPI service lives at `apps/fastapi_service/` and runs on port **8000**.
It owns all ML inference, training lifecycle, and analytics KPIs.

## Stack

Python 3.11+, FastAPI, asyncpg, aiokafka, PyTorch (training only),
ONNX Runtime (inference), HuggingFace Transformers, spaCy, MLflow,
pydantic-settings, ruff, mypy

## Module Layout

```
apps/fastapi_service/src/
├── triage/       — ML Triage Engine (scoring, enrichment, feature engineering)
│   ├── feature_engineering.py
│   ├── trainer.py
│   ├── onnx_exporter.py
│   ├── inference.py           — ONNX Runtime session
│   ├── router.py              — POST /triage/score
│   └── schemas.py             — AlertFeatures, TriageResult (Pydantic v2)
│
├── nlp/
│   ├── ner/                   — NER / IOC Extraction
│   │   ├── preprocessor.py    — CyNER + CASIE → HuggingFace Dataset
│   │   ├── trainer.py
│   │   ├── onnx_exporter.py
│   │   ├── inference.py       — ONNX NER + IOC post-processing
│   │   ├── router.py          — POST /nlp/ner
│   │   └── schemas.py         — NERRequest, NERResult, IOCEntity
│   └── summarizer/            — Incident Summarization
│       ├── preprocessor.py
│       ├── trainer.py
│       ├── onnx_exporter.py
│       ├── inference.py       — ONNX generative inference
│       ├── router.py          — POST /nlp/summarize
│       └── schemas.py         — SummarizeRequest, SummarizeResult
│
├── training/                  — Model lifecycle
│   ├── mlflow_client.py
│   ├── feedback_consumer.py   — Kafka ml.feedback consumer
│   ├── drift_monitor.py       — PSI-based drift detection
│   └── scheduler.py           — APScheduler for retraining
│
├── analytics/                 — KPI calculations (MTTD, MTTR, FP rate)
│
└── shared/
    ├── config.py              — pydantic-settings BaseSettings
    ├── db.py                  — asyncpg pool
    ├── kafka.py               — aiokafka producer/consumer helpers
    ├── redis_client.py
    ├── s3_client.py           — aioboto3 MinIO/S3
    └── exceptions.py          — domain exceptions → HTTPException mappings
```

## Kafka Consumer Groups

`ml-triage-engine`, `nlp-ioc-extractor`, `ml-feedback-collector`

## Coding Patterns

- **Async everywhere**: use `async def` for all handlers; use `asyncpg`,
  `aiokafka`, `httpx.AsyncClient`.
- **Pydantic v2**: all request/response bodies are Pydantic models; validate
  strictly.
- **Dependency injection**: use `Depends()` for DB pool, Redis, Kafka producer.
- **Config**: `pydantic-settings` BaseSettings from environment variables only.
- **Error handling**: raise `HTTPException` with specific codes; custom handlers
  for domain exceptions.
- **ML inference**: ONNX Runtime for production. PyTorch only during training.
  Always provide a heuristic fallback.
- **Tenant isolation**: every DB query includes `tenant_id` extracted from JWT.
- **Linting**: `ruff` for lint + format; `mypy` for type checking.
- **Testing**: `pytest` + `pytest-asyncio`; `httpx.AsyncClient` with
  `TestClient`; mock all external services.

## Key Alerts Table Schema

The `alerts` table in the `aries` database persists: `status`, `ml_score`,
`risk_score`. It does **not** have an `incident_grade` column.
