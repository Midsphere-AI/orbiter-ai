"""Tests for TaskBroker using fakeredis."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from exo.distributed.broker import TaskBroker  # pyright: ignore[reportMissingImports]
from exo.distributed.models import TaskPayload  # pyright: ignore[reportMissingImports]
from exo.types import ExoError  # pyright: ignore[reportMissingImports]


@pytest.fixture
def broker() -> TaskBroker:
    return TaskBroker("redis://localhost:6379")


class TestTaskBrokerInit:
    def test_defaults(self, broker: TaskBroker) -> None:
        assert broker._queue_name == "exo:tasks"
        assert broker._max_retries == 3
        assert broker._group_name == "exo:tasks:group"
        assert broker._redis is None

    def test_custom_params(self) -> None:
        b = TaskBroker(
            "redis://host:1234",
            queue_name="custom:queue",
            max_retries=5,
        )
        assert b._queue_name == "custom:queue"
        assert b._max_retries == 5
        assert b.max_retries == 5

    def test_not_connected_raises(self, broker: TaskBroker) -> None:
        with pytest.raises(ExoError, match="not connected"):
            broker._client()


class TestTaskBrokerConnect:
    @pytest.mark.asyncio
    async def test_connect_creates_consumer_group(self) -> None:
        broker = TaskBroker("redis://localhost:6379")
        with patch("exo.distributed.broker.aioredis.from_url") as mock_from_url:
            mock_redis = AsyncMock()
            mock_from_url.return_value = mock_redis
            mock_redis.xgroup_create = AsyncMock()

            await broker.connect()

            mock_from_url.assert_called_once_with("redis://localhost:6379", decode_responses=True)
            mock_redis.xgroup_create.assert_called_once_with(
                "exo:tasks", "exo:tasks:group", id="0", mkstream=True
            )
            assert broker._redis is mock_redis

    @pytest.mark.asyncio
    async def test_connect_ignores_busygroup(self) -> None:
        broker = TaskBroker("redis://localhost:6379")
        with patch("exo.distributed.broker.aioredis.from_url") as mock_from_url:
            import redis.asyncio as aioredis

            mock_redis = AsyncMock()
            mock_from_url.return_value = mock_redis
            mock_redis.xgroup_create = AsyncMock(
                side_effect=aioredis.ResponseError("BUSYGROUP Consumer Group name already exists")
            )

            await broker.connect()
            # Should not raise — BUSYGROUP is expected when reconnecting.
            assert broker._redis is mock_redis

    @pytest.mark.asyncio
    async def test_connect_raises_other_errors(self) -> None:
        broker = TaskBroker("redis://localhost:6379")
        with patch("exo.distributed.broker.aioredis.from_url") as mock_from_url:
            import redis.asyncio as aioredis

            mock_redis = AsyncMock()
            mock_from_url.return_value = mock_redis
            mock_redis.xgroup_create = AsyncMock(
                side_effect=aioredis.ResponseError("Some other error")
            )

            with pytest.raises(aioredis.ResponseError, match="Some other error"):
                await broker.connect()

    @pytest.mark.asyncio
    async def test_disconnect(self) -> None:
        broker = TaskBroker("redis://localhost:6379")
        with patch("exo.distributed.broker.aioredis.from_url") as mock_from_url:
            mock_redis = AsyncMock()
            mock_from_url.return_value = mock_redis
            mock_redis.xgroup_create = AsyncMock()

            await broker.connect()
            await broker.disconnect()

            mock_redis.aclose.assert_called_once()
            assert broker._redis is None


class TestTaskBrokerWithFakeRedis:
    """Integration-style tests using fakeredis."""

    @pytest.fixture
    async def connected_broker(self) -> TaskBroker:
        import fakeredis.aioredis

        broker = TaskBroker("redis://localhost:6379")
        broker._redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
        await broker._redis.xgroup_create(
            broker._queue_name, broker._group_name, id="0", mkstream=True
        )
        return broker

    @pytest.mark.asyncio
    async def test_submit_returns_task_id(self, connected_broker: TaskBroker) -> None:
        broker = connected_broker
        task = TaskPayload(input="hello")
        task_id = await broker.submit(task)
        assert task_id == task.task_id

    @pytest.mark.asyncio
    async def test_submit_adds_to_stream(self, connected_broker: TaskBroker) -> None:
        broker = connected_broker
        task = TaskPayload(input="hello")
        await broker.submit(task)

        # Read raw stream entries.
        entries = await broker._redis.xrange(broker._queue_name)  # type: ignore[union-attr]
        assert len(entries) == 1
        _msg_id, fields = entries[0]
        data = json.loads(fields["payload"])
        assert data["task_id"] == task.task_id
        assert data["input"] == "hello"

    @pytest.mark.asyncio
    async def test_claim_returns_task(self, connected_broker: TaskBroker) -> None:
        broker = connected_broker
        task = TaskPayload(input="test-claim")
        await broker.submit(task)

        claimed = await broker.claim("worker-1", timeout=1.0)
        assert claimed is not None
        assert claimed.task_id == task.task_id
        assert claimed.input == "test-claim"

    @pytest.mark.asyncio
    async def test_claim_returns_none_on_empty(self, connected_broker: TaskBroker) -> None:
        broker = connected_broker
        claimed = await broker.claim("worker-1", timeout=0.1)
        assert claimed is None

    @pytest.mark.asyncio
    async def test_claim_fifo_order(self, connected_broker: TaskBroker) -> None:
        broker = connected_broker
        t1 = TaskPayload(input="first")
        t2 = TaskPayload(input="second")
        await broker.submit(t1)
        await broker.submit(t2)

        c1 = await broker.claim("worker-1", timeout=1.0)
        c2 = await broker.claim("worker-1", timeout=1.0)
        assert c1 is not None and c1.input == "first"
        assert c2 is not None and c2.input == "second"

    @pytest.mark.asyncio
    async def test_ack_removes_from_pending(self, connected_broker: TaskBroker) -> None:
        broker = connected_broker
        task = TaskPayload(input="ack-test")
        await broker.submit(task)

        claimed = await broker.claim("worker-1", timeout=1.0)
        assert claimed is not None
        await broker.ack(claimed.task_id)

        # Pending list should be empty after ack.
        pending = await broker._redis.xpending(  # type: ignore[union-attr]
            broker._queue_name, broker._group_name
        )
        assert pending["pending"] == 0

    @pytest.mark.asyncio
    async def test_nack_requeues_task(self, connected_broker: TaskBroker) -> None:
        broker = connected_broker
        task = TaskPayload(input="nack-test")
        await broker.submit(task)

        claimed = await broker.claim("worker-1", timeout=1.0)
        assert claimed is not None
        await broker.nack(claimed.task_id)

        # The task should be re-claimable by another consumer.
        reclaimed = await broker.claim("worker-2", timeout=1.0)
        assert reclaimed is not None
        assert reclaimed.task_id == task.task_id
        assert reclaimed.input == "nack-test"

    @pytest.mark.asyncio
    async def test_multiple_workers_claim_different_tasks(
        self, connected_broker: TaskBroker
    ) -> None:
        broker = connected_broker
        t1 = TaskPayload(input="worker1-task")
        t2 = TaskPayload(input="worker2-task")
        await broker.submit(t1)
        await broker.submit(t2)

        c1 = await broker.claim("worker-A", timeout=1.0)
        c2 = await broker.claim("worker-B", timeout=1.0)
        assert c1 is not None and c2 is not None
        assert {c1.input, c2.input} == {"worker1-task", "worker2-task"}

    @pytest.mark.asyncio
    async def test_submit_multiple_and_claim_all(self, connected_broker: TaskBroker) -> None:
        broker = connected_broker
        tasks = [TaskPayload(input=f"task-{i}") for i in range(5)]
        for t in tasks:
            await broker.submit(t)

        claimed_inputs = []
        for _ in range(5):
            c = await broker.claim("worker-1", timeout=1.0)
            assert c is not None
            claimed_inputs.append(c.input)
            await broker.ack(c.task_id)

        assert claimed_inputs == [f"task-{i}" for i in range(5)]

    @pytest.mark.asyncio
    async def test_ack_unknown_task_id_is_noop(self, connected_broker: TaskBroker) -> None:
        broker = connected_broker
        await broker.ack("nonexistent-task-id")  # Should not raise.

    @pytest.mark.asyncio
    async def test_nack_unknown_task_id_is_noop(self, connected_broker: TaskBroker) -> None:
        broker = connected_broker
        await broker.nack("nonexistent-task-id")  # Should not raise.
