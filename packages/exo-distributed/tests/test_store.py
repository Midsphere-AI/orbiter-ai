"""Tests for TaskStore using fakeredis."""

from __future__ import annotations

import time

import pytest

from exo.distributed.models import (  # pyright: ignore[reportMissingImports]
    TaskResult,
    TaskStatus,
)
from exo.distributed.store import TaskStore  # pyright: ignore[reportMissingImports]
from exo.types import ExoError  # pyright: ignore[reportMissingImports]


@pytest.fixture
def store() -> TaskStore:
    return TaskStore("redis://localhost:6379")


class TestTaskStoreInit:
    def test_defaults(self, store: TaskStore) -> None:
        assert store._prefix == "exo:task:"
        assert store._ttl_seconds == 86400
        assert store._redis is None

    def test_custom_params(self) -> None:
        s = TaskStore(
            "redis://host:1234",
            prefix="custom:prefix:",
            ttl_seconds=3600,
        )
        assert s._prefix == "custom:prefix:"
        assert s._ttl_seconds == 3600

    def test_not_connected_raises(self, store: TaskStore) -> None:
        with pytest.raises(ExoError, match="not connected"):
            store._client()


class TestTaskStoreWithFakeRedis:
    """Integration-style tests using fakeredis."""

    @pytest.fixture
    async def connected_store(self) -> TaskStore:
        import fakeredis.aioredis

        store = TaskStore("redis://localhost:6379")
        store._redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
        return store

    @pytest.mark.asyncio
    async def test_set_and_get_status(self, connected_store: TaskStore) -> None:
        store = connected_store
        await store.set_status("task-1", TaskStatus.PENDING)
        result = await store.get_status("task-1")
        assert result is not None
        assert result.task_id == "task-1"
        assert result.status == TaskStatus.PENDING

    @pytest.mark.asyncio
    async def test_get_status_nonexistent(self, connected_store: TaskStore) -> None:
        store = connected_store
        result = await store.get_status("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_set_status_with_kwargs(self, connected_store: TaskStore) -> None:
        store = connected_store
        now = time.time()
        await store.set_status(
            "task-2",
            TaskStatus.RUNNING,
            worker_id="worker-1",
            started_at=now,
        )
        result = await store.get_status("task-2")
        assert result is not None
        assert result.status == TaskStatus.RUNNING
        assert result.worker_id == "worker-1"
        assert result.started_at == now

    @pytest.mark.asyncio
    async def test_set_status_with_result_dict(self, connected_store: TaskStore) -> None:
        store = connected_store
        await store.set_status(
            "task-3",
            TaskStatus.COMPLETED,
            result={"output": "done", "steps": 3},
        )
        result = await store.get_status("task-3")
        assert result is not None
        assert result.status == TaskStatus.COMPLETED
        assert result.result == {"output": "done", "steps": 3}

    @pytest.mark.asyncio
    async def test_set_status_with_error(self, connected_store: TaskStore) -> None:
        store = connected_store
        await store.set_status(
            "task-4",
            TaskStatus.FAILED,
            error="Connection timeout",
            retries=2,
        )
        result = await store.get_status("task-4")
        assert result is not None
        assert result.status == TaskStatus.FAILED
        assert result.error == "Connection timeout"
        assert result.retries == 2

    @pytest.mark.asyncio
    async def test_update_status(self, connected_store: TaskStore) -> None:
        store = connected_store
        await store.set_status("task-5", TaskStatus.PENDING)
        await store.set_status("task-5", TaskStatus.RUNNING, worker_id="w1")
        await store.set_status("task-5", TaskStatus.COMPLETED)

        result = await store.get_status("task-5")
        assert result is not None
        assert result.status == TaskStatus.COMPLETED
        # worker_id should persist from previous set_status.
        assert result.worker_id == "w1"

    @pytest.mark.asyncio
    async def test_list_tasks_all(self, connected_store: TaskStore) -> None:
        store = connected_store
        await store.set_status("t1", TaskStatus.PENDING)
        await store.set_status("t2", TaskStatus.RUNNING)
        await store.set_status("t3", TaskStatus.COMPLETED)

        tasks = await store.list_tasks()
        assert len(tasks) == 3
        task_ids = {t.task_id for t in tasks}
        assert task_ids == {"t1", "t2", "t3"}

    @pytest.mark.asyncio
    async def test_list_tasks_filtered_by_status(self, connected_store: TaskStore) -> None:
        store = connected_store
        await store.set_status("t1", TaskStatus.PENDING)
        await store.set_status("t2", TaskStatus.RUNNING)
        await store.set_status("t3", TaskStatus.COMPLETED)
        await store.set_status("t4", TaskStatus.RUNNING)

        running = await store.list_tasks(status=TaskStatus.RUNNING)
        assert len(running) == 2
        assert all(t.status == TaskStatus.RUNNING for t in running)

    @pytest.mark.asyncio
    async def test_list_tasks_with_limit(self, connected_store: TaskStore) -> None:
        store = connected_store
        for i in range(10):
            await store.set_status(f"task-{i}", TaskStatus.PENDING)

        tasks = await store.list_tasks(limit=5)
        assert len(tasks) == 5

    @pytest.mark.asyncio
    async def test_list_tasks_empty(self, connected_store: TaskStore) -> None:
        store = connected_store
        tasks = await store.list_tasks()
        assert tasks == []

    @pytest.mark.asyncio
    async def test_list_tasks_no_match(self, connected_store: TaskStore) -> None:
        store = connected_store
        await store.set_status("t1", TaskStatus.PENDING)

        tasks = await store.list_tasks(status=TaskStatus.FAILED)
        assert tasks == []

    @pytest.mark.asyncio
    async def test_set_status_none_kwargs(self, connected_store: TaskStore) -> None:
        store = connected_store
        await store.set_status("task-n", TaskStatus.PENDING, error=None)
        result = await store.get_status("task-n")
        assert result is not None
        assert result.error is None

    @pytest.mark.asyncio
    async def test_task_result_types(self, connected_store: TaskStore) -> None:
        store = connected_store
        now = time.time()
        await store.set_status(
            "task-types",
            TaskStatus.COMPLETED,
            started_at=now - 10.0,
            completed_at=now,
            worker_id="worker-2",
            retries=1,
            result={"output": "ok"},
        )
        result = await store.get_status("task-types")
        assert result is not None
        assert isinstance(result, TaskResult)
        assert isinstance(result.started_at, float)
        assert isinstance(result.completed_at, float)
        assert isinstance(result.retries, int)
        assert isinstance(result.result, dict)
