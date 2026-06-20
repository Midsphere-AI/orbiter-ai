"""Tests for distributed() API and TaskHandle."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from exo.distributed.client import (  # pyright: ignore[reportMissingImports]
    TaskHandle,
    distributed,
)
from exo.distributed.models import (  # pyright: ignore[reportMissingImports]
    TaskResult,
    TaskStatus,
)
from exo.types import ExoError  # pyright: ignore[reportMissingImports]

# ---------------------------------------------------------------------------
# TaskHandle.__init__ / properties
# ---------------------------------------------------------------------------


class TestTaskHandleInit:
    def test_task_id_property(self) -> None:
        handle = TaskHandle(
            "abc123",
            broker=MagicMock(),
            store=MagicMock(),
            subscriber=MagicMock(),
        )
        assert handle.task_id == "abc123"

    def test_stores_components(self) -> None:
        broker = MagicMock()
        store = MagicMock()
        subscriber = MagicMock()
        handle = TaskHandle(
            "t1",
            broker=broker,
            store=store,
            subscriber=subscriber,
        )
        assert handle._broker is broker
        assert handle._store is store
        assert handle._subscriber is subscriber


# ---------------------------------------------------------------------------
# TaskHandle.close / async context manager
# ---------------------------------------------------------------------------


class TestTaskHandleClose:
    @pytest.mark.asyncio
    async def test_close_disconnects_all_components(self) -> None:
        broker = MagicMock()
        broker.disconnect = AsyncMock()
        store = MagicMock()
        store.disconnect = AsyncMock()
        subscriber = MagicMock()
        subscriber.disconnect = AsyncMock()

        handle = TaskHandle(
            "t1",
            broker=broker,
            store=store,
            subscriber=subscriber,
        )
        await handle.close()

        broker.disconnect.assert_awaited_once()
        store.disconnect.assert_awaited_once()
        subscriber.disconnect.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_close_is_idempotent_on_component_error(self) -> None:
        """close() must continue disconnecting remaining components even if one raises."""
        broker = MagicMock()
        broker.disconnect = AsyncMock(side_effect=RuntimeError("redis gone"))
        store = MagicMock()
        store.disconnect = AsyncMock()
        subscriber = MagicMock()
        subscriber.disconnect = AsyncMock()

        handle = TaskHandle(
            "t1",
            broker=broker,
            store=store,
            subscriber=subscriber,
        )
        # Should not propagate the RuntimeError
        await handle.close()

        broker.disconnect.assert_awaited_once()
        store.disconnect.assert_awaited_once()
        subscriber.disconnect.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_close_tolerates_all_components_failing(self) -> None:
        """close() must not raise even when every component's disconnect raises."""
        broker = MagicMock()
        broker.disconnect = AsyncMock(side_effect=RuntimeError("broker err"))
        store = MagicMock()
        store.disconnect = AsyncMock(side_effect=RuntimeError("store err"))
        subscriber = MagicMock()
        subscriber.disconnect = AsyncMock(side_effect=RuntimeError("subscriber err"))

        handle = TaskHandle(
            "t1",
            broker=broker,
            store=store,
            subscriber=subscriber,
        )
        await handle.close()  # must not raise

    @pytest.mark.asyncio
    async def test_async_context_manager_calls_close_on_exit(self) -> None:
        broker = MagicMock()
        broker.disconnect = AsyncMock()
        store = MagicMock()
        store.disconnect = AsyncMock()
        subscriber = MagicMock()
        subscriber.disconnect = AsyncMock()

        async with TaskHandle(
            "t1",
            broker=broker,
            store=store,
            subscriber=subscriber,
        ) as handle:
            assert handle.task_id == "t1"

        broker.disconnect.assert_awaited_once()
        store.disconnect.assert_awaited_once()
        subscriber.disconnect.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_async_context_manager_returns_handle(self) -> None:
        handle = TaskHandle(
            "t2",
            broker=MagicMock(disconnect=AsyncMock()),
            store=MagicMock(disconnect=AsyncMock()),
            subscriber=MagicMock(disconnect=AsyncMock()),
        )
        async with handle as h:
            assert h is handle

    @pytest.mark.asyncio
    async def test_close_called_on_exception_exit(self) -> None:
        """__aexit__ should still call close() when the body raises."""
        broker = MagicMock()
        broker.disconnect = AsyncMock()
        store = MagicMock()
        store.disconnect = AsyncMock()
        subscriber = MagicMock()
        subscriber.disconnect = AsyncMock()

        with pytest.raises(ValueError, match="oops"):
            async with TaskHandle(
                "t1",
                broker=broker,
                store=store,
                subscriber=subscriber,
            ):
                raise ValueError("oops")

        broker.disconnect.assert_awaited_once()
        store.disconnect.assert_awaited_once()
        subscriber.disconnect.assert_awaited_once()


# ---------------------------------------------------------------------------
# TaskHandle.status
# ---------------------------------------------------------------------------


class TestTaskHandleStatus:
    @pytest.mark.asyncio
    async def test_returns_task_result(self) -> None:
        store = MagicMock()
        expected = TaskResult(task_id="t1", status=TaskStatus.RUNNING)
        store.get_status = AsyncMock(return_value=expected)

        handle = TaskHandle(
            "t1",
            broker=MagicMock(),
            store=store,
            subscriber=MagicMock(),
        )
        result = await handle.status()
        assert result == expected
        store.get_status.assert_awaited_once_with("t1")

    @pytest.mark.asyncio
    async def test_returns_none_when_not_found(self) -> None:
        store = MagicMock()
        store.get_status = AsyncMock(return_value=None)

        handle = TaskHandle(
            "t1",
            broker=MagicMock(),
            store=store,
            subscriber=MagicMock(),
        )
        result = await handle.status()
        assert result is None


# ---------------------------------------------------------------------------
# TaskHandle.cancel
# ---------------------------------------------------------------------------


class TestTaskHandleCancel:
    @pytest.mark.asyncio
    async def test_delegates_to_broker(self) -> None:
        broker = MagicMock()
        broker.cancel = AsyncMock()

        handle = TaskHandle(
            "t1",
            broker=broker,
            store=MagicMock(),
            subscriber=MagicMock(),
        )
        await handle.cancel()
        broker.cancel.assert_awaited_once_with("t1")


# ---------------------------------------------------------------------------
# TaskHandle.result
# ---------------------------------------------------------------------------


class TestTaskHandleResult:
    @pytest.mark.asyncio
    async def test_returns_result_on_completed(self) -> None:
        store = MagicMock()
        store.get_status = AsyncMock(
            return_value=TaskResult(
                task_id="t1",
                status=TaskStatus.COMPLETED,
                result={"output": "hello"},
            )
        )

        handle = TaskHandle(
            "t1",
            broker=MagicMock(),
            store=store,
            subscriber=MagicMock(),
        )
        result = await handle.result()
        assert result == {"output": "hello"}

    @pytest.mark.asyncio
    async def test_returns_empty_dict_when_result_is_none(self) -> None:
        store = MagicMock()
        store.get_status = AsyncMock(
            return_value=TaskResult(
                task_id="t1",
                status=TaskStatus.COMPLETED,
                result=None,
            )
        )

        handle = TaskHandle(
            "t1",
            broker=MagicMock(),
            store=store,
            subscriber=MagicMock(),
        )
        result = await handle.result()
        assert result == {}

    @pytest.mark.asyncio
    async def test_raises_on_failed(self) -> None:
        store = MagicMock()
        store.get_status = AsyncMock(
            return_value=TaskResult(
                task_id="t1",
                status=TaskStatus.FAILED,
                error="boom",
            )
        )

        handle = TaskHandle(
            "t1",
            broker=MagicMock(),
            store=store,
            subscriber=MagicMock(),
        )
        with pytest.raises(ExoError, match="failed: boom"):
            await handle.result()

    @pytest.mark.asyncio
    async def test_raises_on_cancelled(self) -> None:
        store = MagicMock()
        store.get_status = AsyncMock(
            return_value=TaskResult(
                task_id="t1",
                status=TaskStatus.CANCELLED,
            )
        )

        handle = TaskHandle(
            "t1",
            broker=MagicMock(),
            store=store,
            subscriber=MagicMock(),
        )
        with pytest.raises(ExoError, match="was cancelled"):
            await handle.result()

    @pytest.mark.asyncio
    async def test_polls_until_terminal(self) -> None:
        """result() should poll until a terminal status is reached."""
        pending = TaskResult(task_id="t1", status=TaskStatus.PENDING)
        running = TaskResult(task_id="t1", status=TaskStatus.RUNNING)
        completed = TaskResult(
            task_id="t1",
            status=TaskStatus.COMPLETED,
            result={"output": "done"},
        )

        store = MagicMock()
        store.get_status = AsyncMock(side_effect=[pending, running, completed])

        handle = TaskHandle(
            "t1",
            broker=MagicMock(),
            store=store,
            subscriber=MagicMock(),
        )
        result = await handle.result(poll_interval=0.01)
        assert result == {"output": "done"}
        assert store.get_status.await_count == 3


# ---------------------------------------------------------------------------
# TaskHandle.stream
# ---------------------------------------------------------------------------


class TestTaskHandleStream:
    @pytest.mark.asyncio
    async def test_yields_events_from_subscriber(self) -> None:
        event1 = MagicMock()
        event1.type = "text"
        event2 = MagicMock()
        event2.type = "status"

        async def fake_subscribe(task_id: str):
            yield event1
            yield event2

        subscriber = MagicMock()
        subscriber.subscribe = fake_subscribe

        handle = TaskHandle(
            "t1",
            broker=MagicMock(),
            store=MagicMock(),
            subscriber=subscriber,
        )

        events = [e async for e in handle.stream()]
        assert events == [event1, event2]


# ---------------------------------------------------------------------------
# distributed() function
# ---------------------------------------------------------------------------


class TestDistributed:
    @pytest.mark.asyncio
    async def test_raises_without_redis_url(self) -> None:
        agent = MagicMock()
        with (
            patch.dict("os.environ", {}, clear=True),
            pytest.raises(ExoError, match="redis_url must be provided"),
        ):
            await distributed(agent, "hello")

    @pytest.mark.asyncio
    async def test_uses_env_var_for_redis_url(self) -> None:
        agent = MagicMock()
        agent.to_dict.return_value = {"name": "test"}

        with (
            patch.dict("os.environ", {"EXO_REDIS_URL": "redis://env:6379"}),
            patch("exo.distributed.client.TaskBroker") as mock_broker_cls,
            patch("exo.distributed.client.TaskStore") as mock_store_cls,
            patch("exo.distributed.client.EventSubscriber") as mock_sub_cls,
        ):
            mock_broker = MagicMock()
            mock_broker.connect = AsyncMock()
            mock_broker.submit = AsyncMock(return_value="task123")
            mock_broker_cls.return_value = mock_broker

            mock_store = MagicMock()
            mock_store.connect = AsyncMock()
            mock_store_cls.return_value = mock_store

            mock_sub = MagicMock()
            mock_sub.connect = AsyncMock()
            mock_sub_cls.return_value = mock_sub

            handle = await distributed(agent, "hello")

            mock_broker_cls.assert_called_once_with("redis://env:6379")
            assert isinstance(handle, TaskHandle)

    @pytest.mark.asyncio
    async def test_explicit_redis_url_overrides_env(self) -> None:
        agent = MagicMock()
        agent.to_dict.return_value = {"name": "test"}

        with (
            patch.dict("os.environ", {"EXO_REDIS_URL": "redis://env:6379"}),
            patch("exo.distributed.client.TaskBroker") as mock_broker_cls,
            patch("exo.distributed.client.TaskStore") as mock_store_cls,
            patch("exo.distributed.client.EventSubscriber") as mock_sub_cls,
        ):
            mock_broker = MagicMock()
            mock_broker.connect = AsyncMock()
            mock_broker.submit = AsyncMock(return_value="task123")
            mock_broker_cls.return_value = mock_broker

            mock_store = MagicMock()
            mock_store.connect = AsyncMock()
            mock_store_cls.return_value = mock_store

            mock_sub = MagicMock()
            mock_sub.connect = AsyncMock()
            mock_sub_cls.return_value = mock_sub

            handle = await distributed(agent, "hello", redis_url="redis://explicit:6379")

            mock_broker_cls.assert_called_once_with("redis://explicit:6379")
            assert isinstance(handle, TaskHandle)

    @pytest.mark.asyncio
    async def test_submits_task_with_correct_payload(self) -> None:
        agent = MagicMock()
        agent.to_dict.return_value = {"name": "myagent", "model": "gpt-4"}

        with (
            patch("exo.distributed.client.TaskBroker") as mock_broker_cls,
            patch("exo.distributed.client.TaskStore") as mock_store_cls,
            patch("exo.distributed.client.EventSubscriber") as mock_sub_cls,
        ):
            mock_broker = MagicMock()
            mock_broker.connect = AsyncMock()
            mock_broker.submit = AsyncMock(return_value="task123")
            mock_broker_cls.return_value = mock_broker

            mock_store = MagicMock()
            mock_store.connect = AsyncMock()
            mock_store_cls.return_value = mock_store

            mock_sub = MagicMock()
            mock_sub.connect = AsyncMock()
            mock_sub_cls.return_value = mock_sub

            handle = await distributed(
                agent,
                "hello world",
                redis_url="redis://localhost:6379",
                detailed=True,
                timeout=120.0,
                metadata={"user": "test"},
            )

            # Verify submit was called with a TaskPayload
            mock_broker.submit.assert_awaited_once()
            payload = mock_broker.submit.call_args[0][0]
            assert payload.agent_config == {"name": "myagent", "model": "gpt-4"}
            assert payload.input == "hello world"
            assert payload.detailed is True
            assert payload.timeout_seconds == 120.0
            assert payload.metadata == {"user": "test"}
            assert payload.task_id  # auto-generated
            assert handle.task_id == payload.task_id

    @pytest.mark.asyncio
    async def test_connects_all_components(self) -> None:
        agent = MagicMock()
        agent.to_dict.return_value = {"name": "test"}

        with (
            patch("exo.distributed.client.TaskBroker") as mock_broker_cls,
            patch("exo.distributed.client.TaskStore") as mock_store_cls,
            patch("exo.distributed.client.EventSubscriber") as mock_sub_cls,
        ):
            mock_broker = MagicMock()
            mock_broker.connect = AsyncMock()
            mock_broker.submit = AsyncMock(return_value="task123")
            mock_broker_cls.return_value = mock_broker

            mock_store = MagicMock()
            mock_store.connect = AsyncMock()
            mock_store_cls.return_value = mock_store

            mock_sub = MagicMock()
            mock_sub.connect = AsyncMock()
            mock_sub_cls.return_value = mock_sub

            await distributed(agent, "hello", redis_url="redis://localhost:6379")

            mock_broker.connect.assert_awaited_once()
            mock_store.connect.assert_awaited_once()
            mock_sub.connect.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_works_with_swarm(self) -> None:
        swarm = MagicMock()
        swarm.to_dict.return_value = {
            "agents": [{"name": "a1"}, {"name": "a2"}],
            "mode": "workflow",
        }

        with (
            patch("exo.distributed.client.TaskBroker") as mock_broker_cls,
            patch("exo.distributed.client.TaskStore") as mock_store_cls,
            patch("exo.distributed.client.EventSubscriber") as mock_sub_cls,
        ):
            mock_broker = MagicMock()
            mock_broker.connect = AsyncMock()
            mock_broker.submit = AsyncMock(return_value="task123")
            mock_broker_cls.return_value = mock_broker

            mock_store = MagicMock()
            mock_store.connect = AsyncMock()
            mock_store_cls.return_value = mock_store

            mock_sub = MagicMock()
            mock_sub.connect = AsyncMock()
            mock_sub_cls.return_value = mock_sub

            handle = await distributed(
                swarm,
                "analyze this",
                redis_url="redis://localhost:6379",
            )

            payload = mock_broker.submit.call_args[0][0]
            assert "agents" in payload.agent_config
            assert handle.task_id == payload.task_id

    @pytest.mark.asyncio
    async def test_messages_forwarded(self) -> None:
        agent = MagicMock()
        agent.to_dict.return_value = {"name": "test"}
        msgs = [{"role": "user", "content": "hi"}]

        with (
            patch("exo.distributed.client.TaskBroker") as mock_broker_cls,
            patch("exo.distributed.client.TaskStore") as mock_store_cls,
            patch("exo.distributed.client.EventSubscriber") as mock_sub_cls,
        ):
            mock_broker = MagicMock()
            mock_broker.connect = AsyncMock()
            mock_broker.submit = AsyncMock(return_value="task123")
            mock_broker_cls.return_value = mock_broker

            mock_store = MagicMock()
            mock_store.connect = AsyncMock()
            mock_store_cls.return_value = mock_store

            mock_sub = MagicMock()
            mock_sub.connect = AsyncMock()
            mock_sub_cls.return_value = mock_sub

            await distributed(
                agent,
                "hello",
                redis_url="redis://localhost:6379",
                messages=msgs,
            )

            payload = mock_broker.submit.call_args[0][0]
            assert payload.messages == msgs
