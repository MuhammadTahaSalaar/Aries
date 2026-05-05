"""
ARIES — FastAPI Service Entry Point.

This is the "AI Brain" of the ARIES SOAR platform.
Mounts all routers and manages the service lifecycle:
  1. Startup: Download ONNX models from MinIO, initialise sessions
  2. Runtime: Serve inference endpoints + run Kafka consumers as background tasks
  3. Shutdown: Drain consumers, close pools
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.ingestion.router import router as ingestion_router
from src.nlp.ner.router import router as ner_router
from src.nlp.summarizer.router import router as summarizer_router
from src.shared.config import ServiceSettings, get_settings
from src.shared.db import Database
from src.shared.exceptions import AriesBaseError
from src.shared.kafka import KafkaProducer
from src.shared.logging import get_logger, setup_logging
from src.shared.model_loader import ModelStore, download_models, load_model_store
from src.shared.redis_client import RedisClient
from src.shared.s3_client import S3Client
from src.triage.consumer import TriageKafkaConsumer
from src.triage.router import router as triage_router
from src.triage.schemas import HealthResponse, ReadinessResponse

log = get_logger("main")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    FastAPI lifespan event handler.

    STARTUP:
      1. Configure structured logging
      2. Connect to PostgreSQL, Redis
      3. Download ONNX models from MinIO/S3
      4. Load ONNX InferenceSession objects
      5. Start Kafka producer + consumer background tasks

    SHUTDOWN:
      1. Stop Kafka consumers
      2. Stop Kafka producer
      3. Close Redis connection
      4. Close PostgreSQL pool
    """
    settings = get_settings()
    setup_logging(settings.log_level, settings.service_name)
    log.info("service_starting", version=settings.service_version)

    # Store settings on app state
    app.state.settings = settings

    # ── PostgreSQL ────────────────────────────────────────────────────
    db = Database(settings)
    try:
        await db.connect()
        app.state.db = db
    except Exception:
        log.exception("database_connection_failed")
        app.state.db = None

    # ── Redis ─────────────────────────────────────────────────────────
    redis = RedisClient(settings)
    try:
        await redis.connect()
        app.state.redis = redis
    except Exception:
        log.warning("redis_connection_failed")
        app.state.redis = RedisClient(settings)  # Unconnected placeholder

    # ── S3 / MinIO — Download ONNX models (MLflow-resolved) ────────────
    s3 = S3Client(settings)
    try:
        await download_models(s3, settings)
        log.info("models_downloaded_from_s3")
    except Exception:
        log.warning("s3_model_download_failed_using_local_cache")

    # ── Load ONNX sessions ────────────────────────────────────────────
    model_store = load_model_store(settings)
    app.state.model_store = model_store
    log.info("models_loaded", status=model_store.status())

    # ── SLM pre-warm: load GGUF into RAM now so the first HTTP call
    #    does not pay the 30-second model-load penalty ─────────────────
    if settings.use_slm and model_store.slm_ready:
        log.info("pre_warming_slm", path=settings.slm_model_path)
        from src.triage.slm_inference import get_slm
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, get_slm, settings.slm_model_path)
            log.info("slm_pre_warm_complete")
        except Exception:
            log.exception("slm_pre_warm_failed")

    # ── Kafka Producer ────────────────────────────────────────────────
    producer = KafkaProducer(settings)
    try:
        await producer.start()
        app.state.kafka_producer = producer
    except Exception:
        log.warning("kafka_producer_start_failed")
        app.state.kafka_producer = KafkaProducer(settings)  # Unconnected placeholder

    # ── Kafka Consumers (background tasks) ────────────────────────────
    consumer_tasks: list[asyncio.Task] = []
    triage_consumer: TriageKafkaConsumer | None = None

    if producer.is_connected and app.state.db is not None:
        try:
            triage_consumer = TriageKafkaConsumer(
                settings=settings,
                model_store=model_store,
                producer=producer,
                db=db,
            )
            task = asyncio.create_task(triage_consumer.start())
            consumer_tasks.append(task)
            log.info("triage_consumer_started")
        except Exception:
            log.exception("triage_consumer_start_failed")

    app.state.consumer_tasks = consumer_tasks
    app.state.triage_consumer = triage_consumer

    log.info("service_ready", models=model_store.status())

    yield  # ── Application is running ────────────────────────────────

    # ── SHUTDOWN ──────────────────────────────────────────────────────
    log.info("service_shutting_down")

    # Stop Kafka consumers
    if triage_consumer:
        await triage_consumer.stop()

    for task in consumer_tasks:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    # Stop Kafka producer
    if producer.is_connected:
        await producer.stop()

    # Close Redis
    if redis.is_connected:
        await redis.close()

    # Close DB pool
    if app.state.db:
        await db.close()

    log.info("service_stopped")


def create_app() -> FastAPI:
    """Application factory."""
    settings = get_settings()

    app = FastAPI(
        title="ARIES AI Service",
        description=(
            "AI Brain for the ARIES SOAR platform. Provides alert triage (XGBoost), "
            "NER-based IOC extraction (SecureBERT), incident summarization (BART), "
            "and SIEM ingestion normalization."
        ),
        version=settings.service_version,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # ── CORS ──────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Tighten in production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Exception handlers ────────────────────────────────────────────
    @app.exception_handler(AriesBaseError)
    async def aries_error_handler(request: Request, exc: AriesBaseError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail},
        )

    # ── Health & Readiness ────────────────────────────────────────────
    @app.get("/health", response_model=HealthResponse, tags=["Health"])
    async def health() -> HealthResponse:
        model_store: ModelStore = app.state.model_store
        return HealthResponse(
            status="ok",
            version=settings.service_version,
            models_loaded=model_store.status(),
        )

    @app.get("/ready", response_model=ReadinessResponse, tags=["Health"])
    async def readiness() -> ReadinessResponse:
        model_store: ModelStore = app.state.model_store
        db_ok = app.state.db is not None
        redis_ok = getattr(app.state, "redis", None) is not None and app.state.redis.is_connected
        kafka_ok = getattr(app.state, "kafka_producer", None) is not None and app.state.kafka_producer.is_connected

        models = model_store.status()
        all_ready = db_ok and any(models.values())

        return ReadinessResponse(
            ready=all_ready,
            kafka_connected=kafka_ok,
            db_connected=db_ok,
            redis_connected=redis_ok,
            models=models,
        )

    # ── Mount Routers ─────────────────────────────────────────────────
    app.include_router(triage_router, prefix="/triage", tags=["Triage"])
    app.include_router(ner_router, prefix="/nlp", tags=["NLP"])
    app.include_router(summarizer_router, prefix="/nlp", tags=["NLP"])
    app.include_router(ingestion_router, prefix="/ingest", tags=["Ingestion"])

    return app


app = create_app()
