"""Tests for EventPublisher and EventSubscriber using fakeredis."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from exo.distributed.events import (  # pyright: ignore[reportMissingImports]
    EventPublisher,
    EventSubscriber,
    _deserialize_event,
)
from exo.types import (  # pyright: ignore[reportMissingImports]
    ErrorEvent,
    ExoError,
    ReasoningEvent,
    StatusEvent,
    StepEvent,
    TextEvent,
    ToolCallDeltaEvent,
    ToolCallEvent,
    ToolResultEvent,
    Usage,
    UsageEvent,
)

# ---------------------------------------------------------------------------
# _deserialize_event tests
# ---------------------------------------------------------------------------


class TestDeserializeEvent:
    def test_text_event(self) -> None:
        data = {"type": "text", "text": "hello", "agent_name": "agent-1"}
        event = _deserialize_event(data)
        assert isinstance(event, TextEvent)
        assert event.text == "hello"
        assert event.agent_name == "agent-1"

    def test_tool_call_event(self) -> None:
        data = {
            "type": "tool_call",
            "tool_name": "search",
            "tool_call_id": "tc-1",
            "agent_name": "agent-1",
        }
        event = _deserialize_event(data)
        assert isinstance(event, ToolCallEvent)
        assert event.tool_name == "search"

    def test_step_event(self) -> None:
        data = {
            "type": "step",
            "step_number": 1,
            "agent_name": "a",
            "status": "started",
            "started_at": 100.0,
        }
        event = _deserialize_event(data)
        assert isinstance(event, StepEvent)
        assert event.step_number == 1

    def test_tool_result_event(self) -> None:
        data = {
            "type": "tool_result",
            "tool_name": "search",
            "tool_call_id": "tc-1",
            "result": "found",
            "success": True,
            "duration_ms": 50.0,
            "agent_name": "a",
        }
        event = _deserialize_event(data)
        assert isinstance(event, ToolResultEvent)
        assert event.result == "found"

    def test_reasoning_event(self) -> None:
        data = {"type": "reasoning", "text": "thinking...", "agent_name": "a"}
        event = _deserialize_event(data)
        assert isinstance(event, ReasoningEvent)
        assert event.text == "thinking..."

    def test_error_event(self) -> None:
        data = {
            "type": "error",
            "error": "boom",
            "error_type": "RuntimeError",
            "agent_name": "a",
        }
        event = _deserialize_event(data)
        assert isinstance(event, ErrorEvent)
        assert event.error == "boom"

    def test_status_event(self) -> None:
        data = {"type": "status", "status": "running", "agent_name": "a", "message": ""}
        event = _deserialize_event(data)
        assert isinstance(event, StatusEvent)
        assert event.status == "running"

    def test_usage_event(self) -> None:
        data = {
            "type": "usage",
            "usage": {"input_tokens": 10, "output_tokens": 20, "total_tokens": 30},
            "agent_name": "a",
            "step_number": 1,
            "model": "gpt-4",
        }
        event = _deserialize_event(data)
        assert isinstance(event, UsageEvent)
        assert event.usage.total_tokens == 30

    def test_tool_call_delta_event(self) -> None:
        data = {
            "type": "tool_call_delta",
            "index": 0,
            "tool_call_id": "tc-1",
            "tool_name": "search",
            "arguments_delta": '{"query":',
            "agent_name": "agent-1",
        }
        event = _deserialize_event(data)
        assert isinstance(event, ToolCallDeltaEvent)
        assert event.tool_name == "search"
        assert event.arguments_delta == '{"query":'

    def test_unknown_type_raises(self) -> None:
        with pytest.raises(ExoError, match="Unknown event type"):
            _deserialize_event({"type": "unknown_xyz"})

    def test_round_trip_all_types(self) -> None:
        """model_dump() -> JSON -> deserialize produces equivalent event."""
        events = [
            TextEvent(text="hi"),
            ToolCallEvent(tool_name="t", tool_call_id="tc1"),
            ToolCallDeltaEvent(index=0, tool_name="t", arguments_delta="{}"),
            StepEvent(step_number=1, agent_name="a", status="started", started_at=1.0),
            ToolResultEvent(tool_name="t", tool_call_id="tc1"),
            ReasoningEvent(text="hmm"),
            ErrorEvent(error="err", error_type="E"),
            StatusEvent(status="completed"),
            UsageEvent(usage=Usage(input_tokens=5, output_tokens=5, total_tokens=10)),
        ]
        for original in events:
            dumped = original.model_dump()
            restored = _deserialize_event(dumped)
            assert restored.model_dump() == dumped


# ---------------------------------------------------------------------------
# EventPublisher tests
# ---------------------------------------------------------------------------


class TestEventPublisherInit:
    def test_defaults(self) -> None:
        pub = EventPublisher("redis://localhost:6379")
        assert pub._stream_ttl_seconds == 3600
        assert pub._redis is None

    def test_custom_ttl(self) -> None:
        pub = EventPublisher("redis://localhost:6379", stream_ttl_seconds=7200)
        assert pub._stream_ttl_seconds == 7200

    def test_not_connected_raises(self) -> None:
        pub = EventPublisher("redis://localhost:6379")
        with pytest.raises(ExoError, match="not connected"):
            pub._client()


class TestEventPublisherConnect:
    @pytest.mark.asyncio
    async def test_connect_creates_client(self) -> None:
        pub = EventPublisher("redis://localhost:6379")
        with patch("exo.distributed._redis_mixin.aioredis.from_url") as mock_from_url:
            mock_redis = AsyncMock()
            mock_from_url.return_value = mock_redis

            await pub.connect()

            mock_from_url.assert_called_once_with("redis://localhost:6379", decode_responses=True)
            assert pub._redis is mock_redis

    @pytest.mark.asyncio
    async def test_disconnect(self) -> None:
        pub = EventPublisher("redis://localhost:6379")
        with patch("exo.distributed._redis_mixin.aioredis.from_url") as mock_from_url:
            mock_redis = AsyncMock()
            mock_from_url.return_value = mock_redis

            await pub.connect()
            await pub.disconnect()

            mock_redis.aclose.assert_called_once()
            assert pub._redis is None


class TestEventPublisherWithFakeRedis:
    """Integration-style tests using fakeredis."""

    @pytest.fixture
    async def connected_publisher(self) -> EventPublisher:
        import fakeredis.aioredis

        pub = EventPublisher("redis://localhost:6379", stream_ttl_seconds=3600)
        pub._redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
        return pub

    @pytest.mark.asyncio
    async def test_publish_adds_to_stream(self, connected_publisher: EventPublisher) -> None:
        pub = connected_publisher
        event = TextEvent(text="hello", agent_name="agent-1")
        await pub.publish("task-1", event)

        # Verify the stream has the event.
        entries = await pub._redis.xrange("exo:stream:task-1")  # type: ignore[union-attr]
        assert len(entries) == 1
        _msg_id, fields = entries[0]
        data = json.loads(fields["event"])
        assert data["type"] == "text"
        assert data["text"] == "hello"
        assert data["agent_name"] == "agent-1"

    @pytest.mark.asyncio
    async def test_publish_multiple_events(self, connected_publisher: EventPublisher) -> None:
        pub = connected_publisher
        await pub.publish("task-1", TextEvent(text="a"))
        await pub.publish("task-1", TextEvent(text="b"))
        await pub.publish("task-1", TextEvent(text="c"))

        entries = await pub._redis.xrange("exo:stream:task-1")  # type: ignore[union-attr]
        assert len(entries) == 3
        texts = [json.loads(e[1]["event"])["text"] for e in entries]
        assert texts == ["a", "b", "c"]

    @pytest.mark.asyncio
    async def test_publish_different_tasks(self, connected_publisher: EventPublisher) -> None:
        pub = connected_publisher
        await pub.publish("task-A", TextEvent(text="for A"))
        await pub.publish("task-B", TextEvent(text="for B"))

        entries_a = await pub._redis.xrange("exo:stream:task-A")  # type: ignore[union-attr]
        entries_b = await pub._redis.xrange("exo:stream:task-B")  # type: ignore[union-attr]
        assert len(entries_a) == 1
        assert len(entries_b) == 1
        assert json.loads(entries_a[0][1]["event"])["text"] == "for A"
        assert json.loads(entries_b[0][1]["event"])["text"] == "for B"

    @pytest.mark.asyncio
    async def test_publish_complex_event(self, connected_publisher: EventPublisher) -> None:
        pub = connected_publisher
        event = ToolResultEvent(
            tool_name="search",
            tool_call_id="tc-1",
            arguments={"query": "test"},
            result="found 3 results",
            success=True,
            duration_ms=123.4,
            agent_name="agent-1",
        )
        await pub.publish("task-1", event)

        entries = await pub._redis.xrange("exo:stream:task-1")  # type: ignore[union-attr]
        data = json.loads(entries[0][1]["event"])
        assert data["type"] == "tool_result"
        assert data["tool_name"] == "search"
        assert data["arguments"] == {"query": "test"}
        assert data["duration_ms"] == 123.4

    @pytest.mark.asyncio
    async def test_publish_sets_stream_ttl(self, connected_publisher: EventPublisher) -> None:
        pub = connected_publisher
        await pub.publish("task-1", TextEvent(text="hello"))

        ttl = await pub._redis.ttl("exo:stream:task-1")  # type: ignore[union-attr]
        # TTL should be set (positive value).
        assert ttl > 0
        assert ttl <= 3600


# ---------------------------------------------------------------------------
# EventSubscriber tests
# ---------------------------------------------------------------------------


class TestEventSubscriberInit:
    def test_defaults(self) -> None:
        sub = EventSubscriber("redis://localhost:6379")
        assert sub._redis is None

    def test_not_connected_raises(self) -> None:
        sub = EventSubscriber("redis://localhost:6379")
        with pytest.raises(ExoError, match="not connected"):
            sub._client()


class TestEventSubscriberReplay:
    """Tests for replay (persistent Stream reading)."""

    @pytest.fixture
    async def redis_client(self):  # type: ignore[no-untyped-def]
        import fakeredis.aioredis

        return fakeredis.aioredis.FakeRedis(decode_responses=True)

    @pytest.fixture
    async def publisher_and_subscriber(self, redis_client):  # type: ignore[no-untyped-def]
        pub = EventPublisher("redis://localhost:6379")
        pub._redis = redis_client
        sub = EventSubscriber("redis://localhost:6379")
        sub._redis = redis_client
        return pub, sub

    @pytest.mark.asyncio
    async def test_replay_empty_stream(
        self, publisher_and_subscriber: tuple[EventPublisher, EventSubscriber]
    ) -> None:
        _pub, sub = publisher_and_subscriber
        events = [e async for e in sub.replay("nonexistent")]
        assert events == []

    @pytest.mark.asyncio
    async def test_replay_returns_all_events(
        self, publisher_and_subscriber: tuple[EventPublisher, EventSubscriber]
    ) -> None:
        pub, sub = publisher_and_subscriber
        await pub.publish("task-1", TextEvent(text="first"))
        await pub.publish("task-1", TextEvent(text="second"))
        await pub.publish("task-1", StatusEvent(status="completed", message="done"))

        events = [e async for e in sub.replay("task-1")]
        assert len(events) == 3
        assert isinstance(events[0], TextEvent)
        assert events[0].text == "first"
        assert isinstance(events[1], TextEvent)
        assert events[1].text == "second"
        assert isinstance(events[2], StatusEvent)
        assert events[2].status == "completed"

    @pytest.mark.asyncio
    async def test_replay_preserves_event_types(
        self, publisher_and_subscriber: tuple[EventPublisher, EventSubscriber]
    ) -> None:
        pub, sub = publisher_and_subscriber
        await pub.publish("task-1", TextEvent(text="hi"))
        await pub.publish(
            "task-1",
            ToolCallEvent(tool_name="search", tool_call_id="tc-1"),
        )
        await pub.publish("task-1", ErrorEvent(error="oops", error_type="E"))

        events = [e async for e in sub.replay("task-1")]
        assert isinstance(events[0], TextEvent)
        assert isinstance(events[1], ToolCallEvent)
        assert isinstance(events[2], ErrorEvent)

    @pytest.mark.asyncio
    async def test_replay_different_tasks(
        self, publisher_and_subscriber: tuple[EventPublisher, EventSubscriber]
    ) -> None:
        pub, sub = publisher_and_subscriber
        await pub.publish("task-A", TextEvent(text="A"))
        await pub.publish("task-B", TextEvent(text="B"))

        events_a = [e async for e in sub.replay("task-A")]
        events_b = [e async for e in sub.replay("task-B")]
        assert len(events_a) == 1
        assert len(events_b) == 1
        assert events_a[0].text == "A"  # type: ignore[union-attr]
        assert events_b[0].text == "B"  # type: ignore[union-attr]

    @pytest.mark.asyncio
    async def test_replay_usage_event_nested_model(
        self, publisher_and_subscriber: tuple[EventPublisher, EventSubscriber]
    ) -> None:
        pub, sub = publisher_and_subscriber
        usage = Usage(input_tokens=10, output_tokens=20, total_tokens=30)
        event = UsageEvent(usage=usage, agent_name="a", step_number=1, model="gpt-4")
        await pub.publish("task-1", event)

        events = [e async for e in sub.replay("task-1")]
        assert len(events) == 1
        assert isinstance(events[0], UsageEvent)
        assert events[0].usage.total_tokens == 30
        assert events[0].model == "gpt-4"


class TestEventSubscriberSubscribe:
    """Tests for live Pub/Sub subscription."""

    @pytest.fixture
    async def redis_client(self):  # type: ignore[no-untyped-def]
        import fakeredis.aioredis

        return fakeredis.aioredis.FakeRedis(decode_responses=True)

    @pytest.mark.asyncio
    async def test_subscribe_receives_published_events(
        self,
        redis_client,  # type: ignore[no-untyped-def]
    ) -> None:
        """Publish events and verify subscribe() yields them."""
        pub = EventPublisher("redis://localhost:6379")
        pub._redis = redis_client
        sub = EventSubscriber("redis://localhost:6379")
        sub._redis = redis_client

        received: list = []

        async def consume() -> None:
            async for event in sub.subscribe("task-1"):
                received.append(event)

        async def produce() -> None:
            # Small delay to let subscribe set up.
            import asyncio

            await asyncio.sleep(0.05)
            await pub.publish("task-1", TextEvent(text="hello"))
            await pub.publish("task-1", TextEvent(text="world"))
            await pub.publish(
                "task-1",
                StatusEvent(status="completed", message="done"),
            )

        import asyncio

        # Run producer and consumer concurrently with a timeout.
        await asyncio.wait_for(
            asyncio.gather(consume(), produce()),
            timeout=5.0,
        )

        assert len(received) == 3
        assert isinstance(received[0], TextEvent)
        assert received[0].text == "hello"
        assert isinstance(received[1], TextEvent)
        assert received[1].text == "world"
        assert isinstance(received[2], StatusEvent)
        assert received[2].status == "completed"

    @pytest.mark.asyncio
    async def test_subscribe_stops_on_error_status(
        self,
        redis_client,  # type: ignore[no-untyped-def]
    ) -> None:
        pub = EventPublisher("redis://localhost:6379")
        pub._redis = redis_client
        sub = EventSubscriber("redis://localhost:6379")
        sub._redis = redis_client

        received: list = []

        async def consume() -> None:
            async for event in sub.subscribe("task-2"):
                received.append(event)

        async def produce() -> None:
            import asyncio

            await asyncio.sleep(0.05)
            await pub.publish("task-2", TextEvent(text="before error"))
            await pub.publish(
                "task-2",
                StatusEvent(status="error", message="something failed"),
            )

        import asyncio

        await asyncio.wait_for(
            asyncio.gather(consume(), produce()),
            timeout=5.0,
        )

        assert len(received) == 2
        assert isinstance(received[0], TextEvent)
        assert isinstance(received[1], StatusEvent)
        assert received[1].status == "error"

    @pytest.mark.asyncio
    async def test_subscribe_stops_on_cancelled_status(
        self,
        redis_client,  # type: ignore[no-untyped-def]
    ) -> None:
        pub = EventPublisher("redis://localhost:6379")
        pub._redis = redis_client
        sub = EventSubscriber("redis://localhost:6379")
        sub._redis = redis_client

        received: list = []

        async def consume() -> None:
            async for event in sub.subscribe("task-3"):
                received.append(event)

        async def produce() -> None:
            import asyncio

            await asyncio.sleep(0.05)
            await pub.publish(
                "task-3",
                StatusEvent(status="cancelled", message="user cancelled"),
            )

        import asyncio

        await asyncio.wait_for(
            asyncio.gather(consume(), produce()),
            timeout=5.0,
        )

        assert len(received) == 1
        assert isinstance(received[0], StatusEvent)
        assert received[0].status == "cancelled"
