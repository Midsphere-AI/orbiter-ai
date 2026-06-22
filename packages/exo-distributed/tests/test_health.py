"""Tests for worker health checks and fleet monitoring."""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from exo.distributed.health import (  # pyright: ignore[reportMissingImports]
    WorkerHealth,
    WorkerHealthCheck,
    _parse_worker_health,
    get_worker_fleet_status,
)
from exo.observability.health import (  # pyright: ignore[reportMissingImports]
    HealthCheck,
    HealthStatus,
)

# ---------------------------------------------------------------------------
# WorkerHealth dataclass
# ---------------------------------------------------------------------------


class TestWorkerHealth:
    def test_defaults(self) -> None:
        wh = WorkerHealth(worker_id="w1", status="running")
        assert wh.worker_id == "w1"
        assert wh.status == "running"
        assert wh.tasks_processed == 0
        assert wh.tasks_failed == 0
        assert wh.current_task_id is None
        assert wh.started_at is None
        assert wh.last_heartbeat is None
        assert wh.concurrency == 1
        assert wh.hostname == ""
        assert wh.alive is True

    def test_all_fields(self) -> None:
        wh = WorkerHealth(
            worker_id="w2",
            status="running",
            tasks_processed=10,
            tasks_failed=2,
            current_task_id="task-1",
            started_at=1000.0,
            last_heartbeat=2000.0,
            concurrency=4,
            hostname="myhost",
            alive=False,
        )
        assert wh.tasks_processed == 10
        assert wh.tasks_failed == 2
        assert wh.current_task_id == "task-1"
        assert wh.started_at == 1000.0
        assert wh.concurrency == 4
        assert wh.hostname == "myhost"
        assert wh.alive is False

    def test_frozen(self) -> None:
        wh = WorkerHealth(worker_id="w1", status="running")
        with pytest.raises(AttributeError):
            wh.status = "stopped"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# WorkerHealthCheck — implements HealthCheck protocol
# ---------------------------------------------------------------------------


class TestWorkerHealthCheck:
    def test_implements_health_check_protocol(self) -> None:
        check = WorkerHealthCheck("redis://localhost", "w1")
        assert isinstance(check, HealthCheck)

    def test_name(self) -> None:
        check = WorkerHealthCheck("redis://localhost", "my-worker")
        assert check.name == "worker:my-worker"

    def test_healthy_worker(self) -> None:
        mock_redis = MagicMock()
        now = time.time()
        mock_redis.hgetall.return_value = {
            "status": "running",
            "last_heartbeat": str(now),
            "tasks_processed": "5",
            "tasks_failed": "1",
        }

        check = WorkerHealthCheck("redis://localhost", "w1")

        with patch("redis.from_url", return_value=mock_redis):
            result = check.check()

        assert result.status == HealthStatus.HEALTHY
        assert "w1" in result.message
        assert result.metadata["worker_id"] == "w1"

    def test_unhealthy_no_heartbeat(self) -> None:
        mock_redis = MagicMock()
        mock_redis.hgetall.return_value = {}

        check = WorkerHealthCheck("redis://localhost", "w1")

        with patch("redis.from_url", return_value=mock_redis):
            result = check.check()

        assert result.status == HealthStatus.UNHEALTHY
        assert "not found" in result.message

    def test_unhealthy_expired_heartbeat(self) -> None:
        mock_redis = MagicMock()
        old_time = time.time() - 120  # 120 seconds ago
        mock_redis.hgetall.return_value = {
            "status": "running",
            "last_heartbeat": str(old_time),
            "tasks_processed": "0",
            "tasks_failed": "0",
        }

        check = WorkerHealthCheck("redis://localhost", "w1", heartbeat_timeout=60.0)

        with patch("redis.from_url", return_value=mock_redis):
            result = check.check()

        assert result.status == HealthStatus.UNHEALTHY
        assert "expired" in result.message

    def test_degraded_stale_heartbeat(self) -> None:
        mock_redis = MagicMock()
        # 50 seconds ago, with 60s timeout → 83% of timeout → degraded
        stale_time = time.time() - 50
        mock_redis.hgetall.return_value = {
            "status": "running",
            "last_heartbeat": str(stale_time),
            "tasks_processed": "0",
            "tasks_failed": "0",
        }

        check = WorkerHealthCheck("redis://localhost", "w1", heartbeat_timeout=60.0)

        with patch("redis.from_url", return_value=mock_redis):
            result = check.check()

        assert result.status == HealthStatus.DEGRADED
        assert "stale" in result.message

    @pytest.mark.asyncio
    async def test_acheck_healthy(self) -> None:
        """acheck() must work from an async context (uses async Redis, not blocking)."""
        from unittest.mock import patch as async_patch

        import fakeredis.aioredis

        now = time.time()
        fake_redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
        await fake_redis.hset(  # type: ignore[misc]
            "exo:workers:w-async",
            mapping={
                "status": "running",
                "last_heartbeat": str(now),
                "tasks_processed": "3",
                "tasks_failed": "0",
            },
        )

        check = WorkerHealthCheck("redis://localhost", "w-async", heartbeat_timeout=60.0)
        with async_patch("exo.distributed.health.aioredis.from_url", return_value=fake_redis):
            result = await check.acheck()

        assert result.status == HealthStatus.HEALTHY
        assert "w-async" in result.message

    @pytest.mark.asyncio
    async def test_acheck_unhealthy_no_heartbeat(self) -> None:
        from unittest.mock import patch as async_patch

        import fakeredis.aioredis

        fake_redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
        check = WorkerHealthCheck("redis://localhost", "missing-w", heartbeat_timeout=60.0)
        with async_patch("exo.distributed.health.aioredis.from_url", return_value=fake_redis):
            result = await check.acheck()

        assert result.status == HealthStatus.UNHEALTHY
        assert "not found" in result.message


# ---------------------------------------------------------------------------
# _parse_worker_health
# ---------------------------------------------------------------------------


class TestParseWorkerHealth:
    def test_full_data(self) -> None:
        now = time.time()
        data = {
            "status": "running",
            "tasks_processed": "10",
            "tasks_failed": "2",
            "current_task_id": "task-abc",
            "started_at": str(now - 100),
            "last_heartbeat": str(now),
            "concurrency": "4",
            "hostname": "myhost",
        }
        wh = _parse_worker_health("w1", data)
        assert wh.worker_id == "w1"
        assert wh.status == "running"
        assert wh.tasks_processed == 10
        assert wh.tasks_failed == 2
        assert wh.current_task_id == "task-abc"
        assert wh.concurrency == 4
        assert wh.hostname == "myhost"
        assert wh.alive is True

    def test_empty_current_task(self) -> None:
        now = time.time()
        data = {
            "status": "running",
            "tasks_processed": "0",
            "tasks_failed": "0",
            "current_task_id": "",
            "started_at": str(now),
            "last_heartbeat": str(now),
            "concurrency": "1",
            "hostname": "host",
        }
        wh = _parse_worker_health("w1", data)
        assert wh.current_task_id is None

    def test_dead_worker(self) -> None:
        old = time.time() - 120  # 2 minutes ago, past 60s threshold
        data = {
            "status": "running",
            "last_heartbeat": str(old),
        }
        wh = _parse_worker_health("w1", data)
        assert wh.alive is False

    def test_missing_fields(self) -> None:
        data: dict[str, str] = {}
        wh = _parse_worker_health("w1", data)
        assert wh.status == "unknown"
        assert wh.tasks_processed == 0
        assert wh.last_heartbeat is None
        assert wh.started_at is None


# ---------------------------------------------------------------------------
# get_worker_fleet_status
# ---------------------------------------------------------------------------


class TestGetWorkerFleetStatus:
    @pytest.mark.asyncio
    async def test_no_workers(self) -> None:
        import fakeredis.aioredis

        r = fakeredis.aioredis.FakeRedis(decode_responses=True)

        with patch("exo.distributed.health.aioredis.from_url", return_value=r):
            workers = await get_worker_fleet_status("redis://localhost")

        assert workers == []

    @pytest.mark.asyncio
    async def test_multiple_workers(self) -> None:
        import fakeredis.aioredis

        r = fakeredis.aioredis.FakeRedis(decode_responses=True)

        now = time.time()
        await r.hset(  # type: ignore[misc]
            "exo:workers:w1",
            mapping={
                "status": "running",
                "tasks_processed": "5",
                "tasks_failed": "1",
                "current_task_id": "",
                "started_at": str(now - 100),
                "last_heartbeat": str(now),
                "concurrency": "2",
                "hostname": "host1",
            },
        )
        await r.hset(  # type: ignore[misc]
            "exo:workers:w2",
            mapping={
                "status": "running",
                "tasks_processed": "3",
                "tasks_failed": "0",
                "current_task_id": "task-abc",
                "started_at": str(now - 50),
                "last_heartbeat": str(now),
                "concurrency": "1",
                "hostname": "host2",
            },
        )

        with patch("exo.distributed.health.aioredis.from_url", return_value=r):
            workers = await get_worker_fleet_status("redis://localhost")

        assert len(workers) == 2
        ids = {w.worker_id for w in workers}
        assert ids == {"w1", "w2"}

        # Verify parsed correctly
        w1 = next(w for w in workers if w.worker_id == "w1")
        assert w1.tasks_processed == 5
        assert w1.hostname == "host1"
        assert w1.alive is True

        w2 = next(w for w in workers if w.worker_id == "w2")
        assert w2.current_task_id == "task-abc"

    @pytest.mark.asyncio
    async def test_dead_worker_detected(self) -> None:
        import fakeredis.aioredis

        r = fakeredis.aioredis.FakeRedis(decode_responses=True)

        old = time.time() - 120  # 2 minutes ago
        await r.hset(  # type: ignore[misc]
            "exo:workers:dead-w",
            mapping={
                "status": "running",
                "last_heartbeat": str(old),
                "tasks_processed": "0",
                "tasks_failed": "0",
                "concurrency": "1",
                "hostname": "gone",
            },
        )

        with patch("exo.distributed.health.aioredis.from_url", return_value=r):
            workers = await get_worker_fleet_status("redis://localhost")

        assert len(workers) == 1
        assert workers[0].alive is False
