"""Tests for distributed task tracing — trace context propagation."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from exo.distributed._propagation import (  # pyright: ignore[reportMissingImports]
    BaggagePropagator,
    DictCarrier,
    clear_baggage,
    get_baggage,
    set_baggage,
)
from exo.distributed.cancel import CancellationToken  # pyright: ignore[reportMissingImports]
from exo.distributed.client import distributed  # pyright: ignore[reportMissingImports]
from exo.distributed.models import (  # pyright: ignore[reportMissingImports]
    TaskPayload,
    TaskResult,
    TaskStatus,
)
from exo.distributed.worker import Worker  # pyright: ignore[reportMissingImports]


@pytest.fixture(autouse=True)
def _clear_baggage():
    """Clear baggage context before and after each test."""
    clear_baggage()
    yield
    clear_baggage()


# ---------------------------------------------------------------------------
# Client — submission span and trace context injection
# ---------------------------------------------------------------------------


class TestDistributedSubmissionSpan:
    """Verify distributed() creates a submission span."""

    @pytest.mark.asyncio
    async def test_creates_submit_span(self) -> None:
        """distributed() should create an 'exo.distributed.submit' span."""
        agent = MagicMock()
        agent.to_dict.return_value = {"name": "test-agent"}
        agent.name = "test-agent"

        mock_span = MagicMock()

        with (
            patch("exo.distributed.client.TaskBroker") as mock_broker_cls,
            patch("exo.distributed.client.TaskStore") as mock_store_cls,
            patch("exo.distributed.client.EventSubscriber") as mock_sub_cls,
            patch("exo.distributed.client.aspan") as mock_aspan,
        ):
            mock_broker = MagicMock()
            mock_broker.connect = AsyncMock()
            mock_broker.submit = AsyncMock()
            mock_broker_cls.return_value = mock_broker

            mock_store = MagicMock()
            mock_store.connect = AsyncMock()
            mock_store_cls.return_value = mock_store

            mock_sub = MagicMock()
            mock_sub.connect = AsyncMock()
            mock_sub_cls.return_value = mock_sub

            # Make aspan work as async context manager
            mock_aspan.return_value.__aenter__ = AsyncMock(return_value=mock_span)
            mock_aspan.return_value.__aexit__ = AsyncMock(return_value=False)

            await distributed(agent, "hello", redis_url="redis://localhost:6379")

            # Verify aspan was called with correct span name
            mock_aspan.assert_called_once()
            call_args = mock_aspan.call_args
            assert call_args[0][0] == "exo.distributed.submit"

            # Verify attributes include task_id and agent_name
            attrs = call_args[1].get("attributes", call_args[0][1] if len(call_args[0]) > 1 else {})
            assert attrs["dist.agent_name"] == "test-agent"
            assert "dist.task_id" in attrs

    @pytest.mark.asyncio
    async def test_submit_span_attributes_with_unnamed_agent(self) -> None:
        """Span should have empty agent_name when agent has no name attribute."""
        agent = MagicMock(spec=[])  # no name attribute
        agent.to_dict = MagicMock(return_value={"model": "gpt-4"})

        with (
            patch("exo.distributed.client.TaskBroker") as mock_broker_cls,
            patch("exo.distributed.client.TaskStore") as mock_store_cls,
            patch("exo.distributed.client.EventSubscriber") as mock_sub_cls,
            patch("exo.distributed.client.aspan") as mock_aspan,
        ):
            mock_broker = MagicMock()
            mock_broker.connect = AsyncMock()
            mock_broker.submit = AsyncMock()
            mock_broker_cls.return_value = mock_broker

            mock_store = MagicMock()
            mock_store.connect = AsyncMock()
            mock_store_cls.return_value = mock_store

            mock_sub = MagicMock()
            mock_sub.connect = AsyncMock()
            mock_sub_cls.return_value = mock_sub

            mock_aspan.return_value.__aenter__ = AsyncMock(return_value=MagicMock())
            mock_aspan.return_value.__aexit__ = AsyncMock(return_value=False)

            await distributed(agent, "hello", redis_url="redis://localhost:6379")

            attrs = mock_aspan.call_args[1]["attributes"]
            assert attrs["dist.agent_name"] == ""


class TestDistributedTraceContextInjection:
    """Verify distributed() injects trace context into TaskPayload metadata."""

    @pytest.mark.asyncio
    async def test_injects_trace_context_when_baggage_set(self) -> None:
        """When baggage is set, trace_context should appear in metadata."""
        set_baggage("user.id", "u123")
        set_baggage("session.id", "s456")

        agent = MagicMock()
        agent.to_dict.return_value = {"name": "test"}

        with (
            patch("exo.distributed.client.TaskBroker") as mock_broker_cls,
            patch("exo.distributed.client.TaskStore") as mock_store_cls,
            patch("exo.distributed.client.EventSubscriber") as mock_sub_cls,
            patch("exo.distributed.client.aspan") as mock_aspan,
        ):
            mock_broker = MagicMock()
            mock_broker.connect = AsyncMock()
            mock_broker.submit = AsyncMock()
            mock_broker_cls.return_value = mock_broker

            mock_store = MagicMock()
            mock_store.connect = AsyncMock()
            mock_store_cls.return_value = mock_store

            mock_sub = MagicMock()
            mock_sub.connect = AsyncMock()
            mock_sub_cls.return_value = mock_sub

            mock_aspan.return_value.__aenter__ = AsyncMock(return_value=MagicMock())
            mock_aspan.return_value.__aexit__ = AsyncMock(return_value=False)

            await distributed(agent, "hello", redis_url="redis://localhost:6379")

            # Check the submitted payload has trace_context in metadata
            payload = mock_broker.submit.call_args[0][0]
            assert "trace_context" in payload.metadata
            assert "baggage" in payload.metadata["trace_context"]

    @pytest.mark.asyncio
    async def test_no_trace_context_when_no_baggage(self) -> None:
        """When no baggage is set, trace_context should not appear in metadata."""
        agent = MagicMock()
        agent.to_dict.return_value = {"name": "test"}

        with (
            patch("exo.distributed.client.TaskBroker") as mock_broker_cls,
            patch("exo.distributed.client.TaskStore") as mock_store_cls,
            patch("exo.distributed.client.EventSubscriber") as mock_sub_cls,
            patch("exo.distributed.client.aspan") as mock_aspan,
        ):
            mock_broker = MagicMock()
            mock_broker.connect = AsyncMock()
            mock_broker.submit = AsyncMock()
            mock_broker_cls.return_value = mock_broker

            mock_store = MagicMock()
            mock_store.connect = AsyncMock()
            mock_store_cls.return_value = mock_store

            mock_sub = MagicMock()
            mock_sub.connect = AsyncMock()
            mock_sub_cls.return_value = mock_sub

            mock_aspan.return_value.__aenter__ = AsyncMock(return_value=MagicMock())
            mock_aspan.return_value.__aexit__ = AsyncMock(return_value=False)

            await distributed(agent, "hello", redis_url="redis://localhost:6379")

            payload = mock_broker.submit.call_args[0][0]
            assert "trace_context" not in payload.metadata

    @pytest.mark.asyncio
    async def test_preserves_existing_metadata(self) -> None:
        """Trace context injection should not overwrite existing metadata."""
        set_baggage("key", "val")

        agent = MagicMock()
        agent.to_dict.return_value = {"name": "test"}

        with (
            patch("exo.distributed.client.TaskBroker") as mock_broker_cls,
            patch("exo.distributed.client.TaskStore") as mock_store_cls,
            patch("exo.distributed.client.EventSubscriber") as mock_sub_cls,
            patch("exo.distributed.client.aspan") as mock_aspan,
        ):
            mock_broker = MagicMock()
            mock_broker.connect = AsyncMock()
            mock_broker.submit = AsyncMock()
            mock_broker_cls.return_value = mock_broker

            mock_store = MagicMock()
            mock_store.connect = AsyncMock()
            mock_store_cls.return_value = mock_store

            mock_sub = MagicMock()
            mock_sub.connect = AsyncMock()
            mock_sub_cls.return_value = mock_sub

            mock_aspan.return_value.__aenter__ = AsyncMock(return_value=MagicMock())
            mock_aspan.return_value.__aexit__ = AsyncMock(return_value=False)

            await distributed(
                agent,
                "hello",
                redis_url="redis://localhost:6379",
                metadata={"user": "alice"},
            )

            payload = mock_broker.submit.call_args[0][0]
            assert payload.metadata["user"] == "alice"
            assert "trace_context" in payload.metadata


# ---------------------------------------------------------------------------
# Worker — trace context extraction and execution span
# ---------------------------------------------------------------------------


class TestWorkerTraceContextExtraction:
    """Verify Worker extracts trace context from task metadata."""

    @pytest.mark.asyncio
    async def test_extracts_trace_context(self) -> None:
        """Worker should extract baggage from trace_context in metadata."""
        w = Worker("redis://localhost", worker_id="w1")
        w._broker = AsyncMock()
        w._store = AsyncMock()
        w._publisher = AsyncMock()

        # Create a payload with trace context
        propagator = BaggagePropagator()
        carrier = DictCarrier()
        propagator.inject(carrier, {"user.id": "u123", "session.id": "s456"})

        task = TaskPayload(
            task_id="task-trace-1",
            agent_config={"name": "agent", "model": "openai:gpt-4o"},
            input="hello",
            metadata={"trace_context": carrier.headers},
        )

        # Clear baggage before execution to verify extraction
        clear_baggage()

        captured_baggage: dict[str, str] = {}

        async def _capture_run(agent: object, t: TaskPayload, token: CancellationToken) -> str:
            nonlocal captured_baggage
            captured_baggage = get_baggage()
            return "done"

        with (
            patch.object(w, "_reconstruct_agent", return_value=MagicMock()),
            patch.object(w, "_run_agent", side_effect=_capture_run),
            patch.object(w, "_listen_for_cancel", new_callable=AsyncMock),
        ):
            await w._execute_task(task)

        assert captured_baggage["user.id"] == "u123"
        assert captured_baggage["session.id"] == "s456"

    @pytest.mark.asyncio
    async def test_no_trace_context_in_metadata(self) -> None:
        """Worker should handle tasks without trace_context gracefully."""
        w = Worker("redis://localhost", worker_id="w1")
        w._broker = AsyncMock()
        w._store = AsyncMock()
        w._publisher = AsyncMock()

        task = TaskPayload(
            task_id="task-no-trace",
            agent_config={"name": "agent", "model": "openai:gpt-4o"},
            input="hello",
        )

        async def _fake_run(agent: object, t: TaskPayload, token: CancellationToken) -> str:
            return "done"

        with (
            patch.object(w, "_reconstruct_agent", return_value=MagicMock()),
            patch.object(w, "_run_agent", side_effect=_fake_run),
            patch.object(w, "_listen_for_cancel", new_callable=AsyncMock),
        ):
            # Should not raise
            await w._execute_task(task)

        # Verify it completed normally
        calls = w._store.set_status.call_args_list
        statuses = [c.args[1] for c in calls]
        assert TaskStatus.COMPLETED in statuses

    @pytest.mark.asyncio
    async def test_invalid_trace_context_ignored(self) -> None:
        """Worker should ignore non-dict trace_context values."""
        w = Worker("redis://localhost", worker_id="w1")
        w._broker = AsyncMock()
        w._store = AsyncMock()
        w._publisher = AsyncMock()

        task = TaskPayload(
            task_id="task-bad-trace",
            agent_config={"name": "agent", "model": "openai:gpt-4o"},
            input="hello",
            metadata={"trace_context": "not-a-dict"},
        )

        async def _fake_run(agent: object, t: TaskPayload, token: CancellationToken) -> str:
            return "done"

        with (
            patch.object(w, "_reconstruct_agent", return_value=MagicMock()),
            patch.object(w, "_run_agent", side_effect=_fake_run),
            patch.object(w, "_listen_for_cancel", new_callable=AsyncMock),
        ):
            # Should not raise
            await w._execute_task(task)

        calls = w._store.set_status.call_args_list
        statuses = [c.args[1] for c in calls]
        assert TaskStatus.COMPLETED in statuses


class TestWorkerExecutionSpan:
    """Verify Worker creates an execution span."""

    @pytest.mark.asyncio
    async def test_creates_execute_span(self) -> None:
        """Worker should create an 'exo.distributed.execute' span."""
        w = Worker("redis://localhost", worker_id="w1")
        w._broker = AsyncMock()
        w._store = AsyncMock()
        w._publisher = AsyncMock()

        task = TaskPayload(
            task_id="task-span-1",
            agent_config={"name": "agent", "model": "openai:gpt-4o"},
            input="hello",
        )

        async def _fake_run(agent: object, t: TaskPayload, token: CancellationToken) -> str:
            return "done"

        with (
            patch.object(w, "_reconstruct_agent", return_value=MagicMock()),
            patch.object(w, "_run_agent", side_effect=_fake_run),
            patch.object(w, "_listen_for_cancel", new_callable=AsyncMock),
            patch("exo.distributed.worker.aspan") as mock_aspan,
        ):
            mock_span = MagicMock()
            mock_aspan.return_value.__aenter__ = AsyncMock(return_value=mock_span)
            mock_aspan.return_value.__aexit__ = AsyncMock(return_value=False)

            await w._execute_task(task)

            mock_aspan.assert_called_once_with(
                "exo.distributed.execute",
                attributes={
                    "dist.task_id": "task-span-1",
                    "dist.worker_id": "w1",
                },
            )

    @pytest.mark.asyncio
    async def test_execute_span_records_exception_on_failure(self) -> None:
        """Execution span should record exceptions when task fails."""
        w = Worker("redis://localhost", worker_id="w1")
        w._broker = AsyncMock()
        w._broker.max_retries = 3
        w._store = AsyncMock()
        w._store.get_status.return_value = TaskResult(
            task_id="task-fail", status=TaskStatus.RUNNING, retries=3
        )
        w._publisher = AsyncMock()

        task = TaskPayload(
            task_id="task-fail",
            agent_config={"name": "agent", "model": "openai:gpt-4o"},
            input="fail",
        )

        with (
            patch.object(w, "_reconstruct_agent", side_effect=ValueError("bad config")),
            patch.object(w, "_listen_for_cancel", new_callable=AsyncMock),
            patch("exo.distributed.worker.aspan") as mock_aspan,
        ):
            mock_span = MagicMock()
            mock_aspan.return_value.__aenter__ = AsyncMock(return_value=mock_span)
            mock_aspan.return_value.__aexit__ = AsyncMock(return_value=False)

            await w._execute_task(task)

            # Span should have recorded the exception
            mock_span.record_exception.assert_called_once()
            exc = mock_span.record_exception.call_args[0][0]
            assert isinstance(exc, ValueError)
            assert str(exc) == "bad config"


# ---------------------------------------------------------------------------
# End-to-end trace context propagation
# ---------------------------------------------------------------------------


class TestTraceContextPropagation:
    """Verify trace context flows from client to worker."""

    @pytest.mark.asyncio
    async def test_baggage_round_trip(self) -> None:
        """Baggage set before distributed() should be available in worker."""
        # 1. Set baggage in the "client" context
        set_baggage("request.id", "req-abc")
        set_baggage("tenant", "acme")

        # 2. Simulate what distributed() does: inject into metadata
        propagator = BaggagePropagator()
        carrier = DictCarrier()
        propagator.inject(carrier)
        metadata = {"trace_context": carrier.headers}

        # 3. Create a task payload (as distributed() would)
        task = TaskPayload(
            task_id="task-e2e",
            agent_config={"name": "agent", "model": "openai:gpt-4o"},
            input="hello",
            metadata=metadata,
        )

        # 4. Clear baggage to simulate a different process (worker)
        clear_baggage()
        assert get_baggage() == {}

        # 5. Simulate what the worker does: extract from metadata
        trace_context = task.metadata.get("trace_context")
        assert isinstance(trace_context, dict)
        worker_carrier = DictCarrier(trace_context)
        worker_propagator = BaggagePropagator()
        worker_propagator.extract(worker_carrier)

        # 6. Verify baggage is restored
        baggage = get_baggage()
        assert baggage["request.id"] == "req-abc"
        assert baggage["tenant"] == "acme"

    def test_empty_baggage_no_trace_context(self) -> None:
        """When no baggage is set, no trace_context should be injected."""
        propagator = BaggagePropagator()
        carrier = DictCarrier()
        propagator.inject(carrier)
        assert carrier.headers == {}
