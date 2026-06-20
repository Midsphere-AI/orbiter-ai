"""Tests for streaming event metrics in EventPublisher."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from exo.distributed.events import (  # pyright: ignore[reportMissingImports]
    EventPublisher,
    _record_event_published,
)
from exo.observability.metrics import (  # pyright: ignore[reportMissingImports]
    get_metrics_snapshot,
    reset_metrics,
)
from exo.observability.semconv import (  # pyright: ignore[reportMissingImports]
    METRIC_STREAM_EVENT_PUBLISH_DURATION,
    METRIC_STREAM_EVENTS_EMITTED,
    STREAM_EVENT_TYPE,
)
from exo.types import (  # pyright: ignore[reportMissingImports]
    StatusEvent,
    TextEvent,
    ToolCallEvent,
)

# Patch HAS_OTEL to False so all recording helpers use the in-memory collector.
# events.py now delegates to metrics.py helpers, so only patching metrics is needed.
_NO_OTEL_METRICS = patch("exo.distributed.metrics.HAS_OTEL", False)


@pytest.fixture(autouse=True)
def _reset() -> None:
    """Reset in-memory metrics and force in-memory path for each test."""
    reset_metrics()
    with _NO_OTEL_METRICS:
        yield  # type: ignore[misc]


# ---------------------------------------------------------------------------
# _record_event_published
# ---------------------------------------------------------------------------


class TestRecordEventPublished:
    def test_increments_counter_with_event_type(self) -> None:
        _record_event_published("text", 0.01)
        snap = get_metrics_snapshot()
        assert snap["counters"][METRIC_STREAM_EVENTS_EMITTED] == 1.0

    def test_records_duration_histogram(self) -> None:
        _record_event_published("text", 0.05)
        snap = get_metrics_snapshot()
        durations = snap["histograms"][METRIC_STREAM_EVENT_PUBLISH_DURATION]
        assert len(durations) == 1
        assert durations[0]["value"] == 0.05
        assert durations[0]["attributes"][STREAM_EVENT_TYPE] == "text"

    def test_skips_zero_duration(self) -> None:
        _record_event_published("text", 0.0)
        snap = get_metrics_snapshot()
        assert snap["counters"][METRIC_STREAM_EVENTS_EMITTED] == 1.0
        assert METRIC_STREAM_EVENT_PUBLISH_DURATION not in snap["histograms"]

    def test_multiple_event_types(self) -> None:
        _record_event_published("text", 0.01)
        _record_event_published("text", 0.02)
        _record_event_published("tool_call", 0.03)
        snap = get_metrics_snapshot()
        # Counter aggregates all types
        assert snap["counters"][METRIC_STREAM_EVENTS_EMITTED] == 3.0
        # Histogram has individual entries with type attributes
        durations = snap["histograms"][METRIC_STREAM_EVENT_PUBLISH_DURATION]
        assert len(durations) == 3


# ---------------------------------------------------------------------------
# EventPublisher.publish() metrics integration
# ---------------------------------------------------------------------------


class TestEventPublisherMetrics:
    @pytest.fixture
    async def connected_publisher(self) -> EventPublisher:
        import fakeredis.aioredis

        pub = EventPublisher("redis://localhost:6379")
        pub._redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
        return pub

    @pytest.mark.asyncio
    async def test_publish_records_event_count(self, connected_publisher: EventPublisher) -> None:
        pub = connected_publisher
        await pub.publish("task-1", TextEvent(text="hello"))
        snap = get_metrics_snapshot()
        assert snap["counters"][METRIC_STREAM_EVENTS_EMITTED] == 1.0

    @pytest.mark.asyncio
    async def test_publish_records_duration(self, connected_publisher: EventPublisher) -> None:
        pub = connected_publisher
        await pub.publish("task-1", TextEvent(text="hello"))
        snap = get_metrics_snapshot()
        durations = snap["histograms"][METRIC_STREAM_EVENT_PUBLISH_DURATION]
        assert len(durations) == 1
        assert durations[0]["value"] > 0

    @pytest.mark.asyncio
    async def test_publish_tracks_event_type_attribute(
        self, connected_publisher: EventPublisher
    ) -> None:
        pub = connected_publisher
        await pub.publish("task-1", TextEvent(text="hello"))
        snap = get_metrics_snapshot()
        durations = snap["histograms"][METRIC_STREAM_EVENT_PUBLISH_DURATION]
        assert durations[0]["attributes"][STREAM_EVENT_TYPE] == "text"

    @pytest.mark.asyncio
    async def test_publish_multiple_types(self, connected_publisher: EventPublisher) -> None:
        pub = connected_publisher
        await pub.publish("task-1", TextEvent(text="hello"))
        await pub.publish("task-1", ToolCallEvent(tool_name="search", tool_call_id="tc-1"))
        await pub.publish("task-1", StatusEvent(status="completed", message="done"))
        snap = get_metrics_snapshot()
        assert snap["counters"][METRIC_STREAM_EVENTS_EMITTED] == 3.0
        durations = snap["histograms"][METRIC_STREAM_EVENT_PUBLISH_DURATION]
        assert len(durations) == 3
        types = {d["attributes"][STREAM_EVENT_TYPE] for d in durations}
        assert types == {"text", "tool_call", "status"}
