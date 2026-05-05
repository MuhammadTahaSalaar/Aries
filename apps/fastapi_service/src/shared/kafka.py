"""
ARIES — Async Kafka producer/consumer helpers (aiokafka).

Provides a typed producer wrapper and a base consumer class that
handles graceful shutdown, back-pressure, and structured logging.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Callable, Coroutine

from aiokafka import AIOKafkaConsumer, AIOKafkaProducer

from src.shared.config import ServiceSettings
from src.shared.logging import get_logger

log = get_logger("kafka")


class KafkaProducer:
    """Async Kafka producer wrapper with JSON serialisation."""

    def __init__(self, settings: ServiceSettings) -> None:
        self._bootstrap = settings.kafka_bootstrap_servers
        self._producer: AIOKafkaProducer | None = None

    async def start(self) -> None:
        self._producer = AIOKafkaProducer(
            bootstrap_servers=self._bootstrap,
            value_serializer=lambda v: json.dumps(v, default=str).encode("utf-8"),
            key_serializer=lambda k: k.encode("utf-8") if k else None,
            acks="all",
            enable_idempotence=True,
            max_request_size=5_242_880,  # 5 MB
        )
        await self._producer.start()
        log.info("kafka_producer_started", bootstrap=self._bootstrap)

    async def stop(self) -> None:
        if self._producer:
            await self._producer.stop()
            log.info("kafka_producer_stopped")

    async def send(self, topic: str, value: dict[str, Any], key: str | None = None) -> None:
        if self._producer is None:
            raise RuntimeError("Kafka producer not started")
        await self._producer.send_and_wait(topic, value=value, key=key)
        log.debug("kafka_message_sent", topic=topic, key=key)

    @property
    def is_connected(self) -> bool:
        return self._producer is not None


MessageHandler = Callable[[dict[str, Any]], Coroutine[Any, Any, None]]


class KafkaConsumerWorker:
    """
    Base async Kafka consumer that deserialises JSON messages
    and dispatches them to a handler coroutine.

    Runs as an asyncio.Task. Handles graceful shutdown via stop().
    """

    def __init__(
        self,
        settings: ServiceSettings,
        topic: str,
        group_id: str,
        handler: MessageHandler,
    ) -> None:
        self._bootstrap = settings.kafka_bootstrap_servers
        self._topic = topic
        self._group_id = group_id
        self._handler = handler
        self._consumer: AIOKafkaConsumer | None = None
        self._running = False

    async def start(self) -> None:
        """Start consuming in a loop. Call as asyncio.create_task(worker.start())."""
        self._consumer = AIOKafkaConsumer(
            self._topic,
            bootstrap_servers=self._bootstrap,
            group_id=self._group_id,
            value_deserializer=lambda v: json.loads(v.decode("utf-8")),
            auto_offset_reset="earliest",
            enable_auto_commit=True,
            auto_commit_interval_ms=5000,
            max_poll_records=50,
        )
        await self._consumer.start()
        self._running = True
        log.info("kafka_consumer_started", topic=self._topic, group=self._group_id)

        try:
            async for msg in self._consumer:
                if not self._running:
                    break
                try:
                    await self._handler(msg.value)
                except Exception:
                    log.exception(
                        "kafka_message_handler_error",
                        topic=self._topic,
                        offset=msg.offset,
                        partition=msg.partition,
                    )
        finally:
            await self._consumer.stop()
            log.info("kafka_consumer_stopped", topic=self._topic)

    async def stop(self) -> None:
        """Signal the consumer loop to stop."""
        self._running = False
        if self._consumer:
            await self._consumer.stop()

    @property
    def is_connected(self) -> bool:
        return self._consumer is not None and self._running
