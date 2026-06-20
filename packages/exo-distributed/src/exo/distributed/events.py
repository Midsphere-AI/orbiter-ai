"""Redis-backed event publishing and subscription for distributed streaming."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import AsyncIterator
from typing import Any

from exo.distributed._redis_mixin import (  # pyright: ignore[reportMissingImports]
    RedisConnectionMixin,
)
from exo.distributed.metrics import (  # pyright: ignore[reportMissingImports]
    increment_counter,
    record_histogram_value,
)
from exo.observability.semconv import (  # pyright: ignore[reportMissingImports]
    METRIC_STREAM_EVENT_PUBLISH_DURATION,
    METRIC_STREAM_EVENTS_EMITTED,
    STREAM_EVENT_TYPE,
)
from exo.types import (  # pyright: ignore[reportMissingImports]
    ErrorEvent,
    ExoError,
    ReasoningEvent,
    StatusEvent,
    StepEvent,
    StreamEvent,
    TextEvent,
    ToolCallDeltaEvent,
    ToolCallEvent,
    ToolResultEvent,
    UsageEvent,
)

logger = logging.getLogger(__name__)

# Mapping from event type discriminator to the concrete class.
_EVENT_TYPE_MAP: dict[str, type[Any]] = {
    "text": TextEvent,
    "tool_call": ToolCallEvent,
    "tool_call_delta": ToolCallDeltaEvent,
    "step": StepEvent,
    "tool_result": ToolResultEvent,
    "reasoning": ReasoningEvent,
    "error": ErrorEvent,
    "status": StatusEvent,
    "usage": UsageEvent,
}


def _record_event_published(event_type: str, duration: float) -> None:
    """Record metrics for a single event publish operation."""
    attrs: dict[str, str] = {STREAM_EVENT_TYPE: event_type}
    increment_counter(
        METRIC_STREAM_EVENTS_EMITTED,
        attrs,
        description="Number of streaming events emitted",
    )
    if duration > 0:
        record_histogram_value(
            METRIC_STREAM_EVENT_PUBLISH_DURATION,
            duration,
            attrs,
            description="Duration of event publish operations",
        )


def _deserialize_event(data: dict[str, Any]) -> StreamEvent:
    """Reconstruct a ``StreamEvent`` from a JSON-decoded dict.

    Uses the ``type`` field as a discriminator to pick the correct class.
    """
    event_type = data.get("type")
    cls = _EVENT_TYPE_MAP.get(event_type)  # type: ignore[arg-type]
    if cls is None:
        msg = f"Unknown event type: {event_type!r}"
        raise ExoError(msg)
    return cls(**data)  # type: ignore[return-value]


class EventPublisher(RedisConnectionMixin):
    """Publishes streaming events to Redis Pub/Sub and Streams.

    Each event is published to:
    - A Pub/Sub channel ``exo:events:{task_id}`` for live consumption.
    - A Redis Stream ``exo:stream:{task_id}`` for persistent replay.

    Args:
        redis_url: Redis connection URL.
        stream_ttl_seconds: TTL for the persistent stream key (default 3600 = 1 hour).
    """

    def __init__(
        self,
        redis_url: str,
        *,
        stream_ttl_seconds: int = 3600,
    ) -> None:
        super().__init__()
        self._redis_url = redis_url
        self._stream_ttl_seconds = stream_ttl_seconds

    async def connect(self) -> None:
        """Connect to Redis."""
        await super().connect()

    def _pubsub_channel(self, task_id: str) -> str:
        return f"exo:events:{task_id}"

    def _stream_key(self, task_id: str) -> str:
        return f"exo:stream:{task_id}"

    async def publish(self, task_id: str, event: StreamEvent) -> None:
        """Publish *event* to both Pub/Sub and the persistent Stream."""
        r = self._client()
        payload = json.dumps(event.model_dump())

        start = time.monotonic()

        # Publish to Pub/Sub for live subscribers.
        await r.publish(self._pubsub_channel(task_id), payload)  # type: ignore[misc]

        # Append to Stream for replay.
        stream_key = self._stream_key(task_id)
        await r.xadd(stream_key, {"event": payload})
        await r.expire(stream_key, self._stream_ttl_seconds)  # type: ignore[misc]

        duration = time.monotonic() - start
        logger.debug(
            "EventPublisher published %s event for task %s (%.3fs)",
            event.type,
            task_id,
            duration,
        )
        _record_event_published(event.type, duration)


class EventSubscriber(RedisConnectionMixin):
    """Subscribes to streaming events from Redis.

    Provides two consumption modes:

    - :meth:`subscribe` — live Pub/Sub for real-time events.
    - :meth:`replay` — reads persisted events from a Redis Stream.

    Args:
        redis_url: Redis connection URL.
    """

    def __init__(self, redis_url: str) -> None:
        super().__init__()
        self._redis_url = redis_url

    async def connect(self) -> None:
        """Connect to Redis."""
        await super().connect()

    async def subscribe(self, task_id: str) -> AsyncIterator[StreamEvent]:
        """Yield live events via Redis Pub/Sub.

        Listens on channel ``exo:events:{task_id}`` and yields
        deserialized ``StreamEvent`` instances.  The iterator ends when
        a ``StatusEvent`` with status ``"completed"`` or ``"error"`` or
        ``"cancelled"`` is received.
        """
        r = self._client()
        channel_name = f"exo:events:{task_id}"
        pubsub = r.pubsub()
        await pubsub.subscribe(channel_name)  # type: ignore[misc]
        logger.debug("EventSubscriber subscribed to %s", channel_name)
        terminal_statuses = {"completed", "error", "cancelled"}
        try:
            while True:
                msg = await pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
                if msg is None:
                    await asyncio.sleep(0.01)
                    continue
                if msg["type"] != "message":
                    continue
                data = json.loads(msg["data"])
                event = _deserialize_event(data)
                yield event
                # Stop on terminal status events.
                if isinstance(event, StatusEvent) and event.status in terminal_statuses:
                    logger.debug(
                        "EventSubscriber received terminal event for task %s (status=%s)",
                        task_id,
                        event.status,
                    )
                    break
        finally:
            await pubsub.unsubscribe(channel_name)
            await pubsub.aclose()  # type: ignore[misc]

    async def replay(self, task_id: str, from_id: str = "0") -> AsyncIterator[StreamEvent]:
        """Yield persisted events from the Redis Stream.

        Reads all entries in ``exo:stream:{task_id}`` starting from
        *from_id* (default ``"0"`` = beginning).
        """
        r = self._client()
        stream_key = f"exo:stream:{task_id}"
        entries = await r.xrange(stream_key, min=from_id)
        logger.debug(
            "EventSubscriber replaying %d events for task %s from %s",
            len(entries),
            task_id,
            from_id,
        )
        for _msg_id, fields in entries:
            data = json.loads(fields["event"])
            yield _deserialize_event(data)
