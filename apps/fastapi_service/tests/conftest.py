"""
ARIES — Shared test fixtures and configuration.
"""

from __future__ import annotations

import asyncio
from typing import AsyncGenerator, Generator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from src.shared.config import ServiceSettings
from src.shared.model_loader import ModelStore


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create a single event loop for the entire test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def test_settings() -> ServiceSettings:
    """Service settings pre-configured for testing."""
    return ServiceSettings(
        database_url="postgresql://test:test@localhost:5432/test_aries",
        kafka_bootstrap_servers="localhost:9092",
        redis_url="redis://localhost:6379/1",
        s3_endpoint_url="http://localhost:9000",
        s3_access_key="minioadmin",
        s3_secret_key="minioadmin",
        log_level="DEBUG",
    )


@pytest.fixture
def mock_model_store() -> ModelStore:
    """A model store with mock ONNX sessions."""
    store = ModelStore()

    # Mock triage session
    mock_triage = MagicMock()
    mock_triage.get_inputs.return_value = [MagicMock(name="float_input")]
    mock_triage.run.return_value = [
        [2],  # labels: TruePositive
        [{0: 0.05, 1: 0.10, 2: 0.85}],  # probabilities
    ]
    store.triage_session = mock_triage
    store.triage_metadata = {
        "model_type": "xgboost",
        "n_features": 49,
    }

    # Mock NER session
    import numpy as np

    mock_ner = MagicMock()
    mock_ner.run.return_value = [
        np.zeros((1, 20, 11), dtype=np.float32),  # logits [batch, seq, labels]
    ]
    store.ner_session = mock_ner
    store.ner_metadata = {
        "id2label": {
            "0": "O", "1": "B-Malware", "2": "I-Malware",
            "3": "B-Indicator", "4": "I-Indicator",
            "5": "B-System", "6": "I-System",
            "7": "B-Vulnerability", "8": "I-Vulnerability",
            "9": "B-Organization", "10": "I-Organization",
        }
    }

    # Mock NER tokenizer
    mock_tokenizer = MagicMock()
    mock_encoding = MagicMock()
    mock_encoding.ids = list(range(20))
    mock_encoding.attention_mask = [1] * 20
    mock_encoding.offsets = [(0, 0)] + [(i, i + 3) for i in range(0, 57, 3)] + [(0, 0)]
    mock_encoding.tokens = ["<s>"] + [f"tok{i}" for i in range(18)] + ["</s>"]
    mock_tokenizer.encode.return_value = mock_encoding
    mock_tokenizer.decode.return_value = "decoded text"
    store.ner_tokenizer = mock_tokenizer

    # Mock summarizer
    mock_encoder = MagicMock()
    mock_encoder.run.return_value = [np.zeros((1, 10, 768), dtype=np.float32)]

    mock_decoder = MagicMock()
    mock_decoder.get_inputs.return_value = [
        MagicMock(name="input_ids"),
        MagicMock(name="encoder_hidden_states"),
        MagicMock(name="encoder_attention_mask"),
    ]
    # Return logits with EOS at position 2
    vocab_logits = np.zeros((1, 1, 50265), dtype=np.float32)
    vocab_logits[0, 0, 2] = 100.0  # EOS token
    mock_decoder.run.return_value = [vocab_logits]

    store.summarizer_encoder_session = mock_encoder
    store.summarizer_decoder_session = mock_decoder
    store.summarizer_tokenizer = mock_tokenizer
    store.summarizer_metadata = {"model_type": "bart-summariser"}

    return store


@pytest.fixture
def mock_db() -> MagicMock:
    """Mock database."""
    db = MagicMock()
    db.insert_alert = AsyncMock(return_value="test-alert-id")
    db.update_alert_scores = AsyncMock()
    db.get_alert = AsyncMock(return_value=None)
    db.insert_iocs = AsyncMock(return_value=0)
    db.upsert_case_summary = AsyncMock()
    return db


@pytest.fixture
def mock_redis() -> MagicMock:
    """Mock Redis client."""
    redis = MagicMock()
    redis.is_connected = True
    redis.get_cached = AsyncMock(return_value=None)
    redis.set_cached = AsyncMock()
    redis.get_dedup = AsyncMock(return_value=False)
    redis.set_dedup = AsyncMock()
    return redis


@pytest.fixture
def mock_kafka_producer() -> MagicMock:
    """Mock Kafka producer."""
    producer = MagicMock()
    producer.is_connected = True
    producer.send = AsyncMock()
    return producer


@pytest_asyncio.fixture
async def async_client(
    test_settings: ServiceSettings,
    mock_model_store: ModelStore,
    mock_db: MagicMock,
    mock_redis: MagicMock,
    mock_kafka_producer: MagicMock,
) -> AsyncGenerator[AsyncClient, None]:
    """Create an httpx AsyncClient with a fully mocked app."""
    with patch("src.shared.config.get_settings", return_value=test_settings):
        with patch("src.shared.model_loader.download_models", new_callable=AsyncMock):
            with patch("src.shared.model_loader.load_model_store", return_value=mock_model_store):
                from main import create_app

                app = create_app()

                # Manually set state to skip lifespan for unit tests
                app.state.settings = test_settings
                app.state.model_store = mock_model_store
                app.state.db = mock_db
                app.state.redis = mock_redis
                app.state.kafka_producer = mock_kafka_producer
                app.state.consumer_tasks = []
                app.state.triage_consumer = None

                async with AsyncClient(
                    transport=ASGITransport(app=app),
                    base_url="http://test",
                ) as client:
                    yield client
