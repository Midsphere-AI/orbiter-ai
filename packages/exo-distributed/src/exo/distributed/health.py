"""Worker health checks and fleet monitoring.

Publishes worker health data to Redis and provides functions to query the
health of all active workers.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any

import redis.asyncio as aioredis

from exo.observability.health import (  # pyright: ignore[reportMissingImports]
    HealthResult,
    HealthStatus,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WorkerHealth:
    """Snapshot of a single worker's health data."""

    worker_id: str
    status: str
    tasks_processed: int = 0
    tasks_failed: int = 0
    current_task_id: str | None = None
    started_at: float | None = None
    last_heartbeat: float | None = None
    concurrency: int = 1
    hostname: str = ""
    alive: bool = True


class WorkerHealthCheck:
    """Health check for a distributed worker, implementing the HealthCheck protocol.

    Reads the worker's heartbeat hash from Redis and determines health based
    on heartbeat freshness.

    Args:
        redis_url: Redis connection URL.
        worker_id: The worker ID to check.
        heartbeat_timeout: Seconds after which a missing heartbeat means unhealthy (default 60).
    """

    def __init__(
        self,
        redis_url: str,
        worker_id: str,
        *,
        heartbeat_timeout: float = 60.0,
    ) -> None:
        self._redis_url = redis_url
        self._worker_id = worker_id
        self._heartbeat_timeout = heartbeat_timeout

    @property
    def name(self) -> str:
        return f"worker:{self._worker_id}"

    def check(self) -> HealthResult:
        """Check worker health by reading its heartbeat from Redis.

        Since the HealthCheck protocol requires a sync ``check()`` method,
        this creates a temporary sync Redis connection.

        .. warning::
            This method is **synchronous** and performs a blocking network call.
            Do **not** call it from an async context (e.g. inside an
            ``asyncio`` event loop) — use :meth:`acheck` instead.

        Raises:
            RuntimeError: If called from within a running ``asyncio`` event loop.
                Use :meth:`acheck` instead.
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass  # No running loop — safe to proceed with blocking call.
        else:
            raise RuntimeError(
                "WorkerHealthCheck.check() was called from inside a running asyncio event loop, "
                "which would block the loop. Use 'await health_check.acheck()' instead."
            )

        import redis

        logger.debug("WorkerHealthCheck checking worker %s", self._worker_id)
        r = redis.from_url(self._redis_url, decode_responses=True, socket_timeout=5)
        try:
            key = f"exo:workers:{self._worker_id}"
            data: dict[str, str] = r.hgetall(key)  # type: ignore[assignment]
            return self._evaluate(data)
        finally:
            r.close()

    async def acheck(self) -> HealthResult:
        """Async variant of :meth:`check` — safe to call from a running event loop.

        Uses an async Redis connection instead of a blocking sync one.
        """
        logger.debug("WorkerHealthCheck (async) checking worker %s", self._worker_id)
        r: aioredis.Redis = aioredis.from_url(
            self._redis_url, decode_responses=True, socket_connect_timeout=5, socket_timeout=5
        )
        try:
            key = f"exo:workers:{self._worker_id}"
            data: dict[str, str] = await r.hgetall(key)  # type: ignore[misc,assignment]
            return self._evaluate(data)
        finally:
            await r.aclose()

    def _evaluate(self, data: dict[str, str]) -> HealthResult:
        """Compute a :class:`HealthResult` from the raw heartbeat hash."""
        if not data:
            return HealthResult(
                status=HealthStatus.UNHEALTHY,
                message=f"Worker {self._worker_id} not found (no heartbeat)",
                metadata={"worker_id": self._worker_id},
            )

        last_hb = float(data.get("last_heartbeat", "0"))
        age = time.time() - last_hb
        metadata: dict[str, Any] = {
            "worker_id": self._worker_id,
            "heartbeat_age_seconds": round(age, 2),
            "tasks_processed": int(data.get("tasks_processed", "0")),
            "tasks_failed": int(data.get("tasks_failed", "0")),
        }

        if age > self._heartbeat_timeout:
            return HealthResult(
                status=HealthStatus.UNHEALTHY,
                message=f"Worker {self._worker_id} heartbeat expired ({age:.1f}s ago)",
                metadata=metadata,
            )

        if age > self._heartbeat_timeout * 0.8:
            return HealthResult(
                status=HealthStatus.DEGRADED,
                message=f"Worker {self._worker_id} heartbeat stale ({age:.1f}s ago)",
                metadata=metadata,
            )

        return HealthResult(
            status=HealthStatus.HEALTHY,
            message=f"Worker {self._worker_id} healthy (heartbeat {age:.1f}s ago)",
            metadata=metadata,
        )


def _parse_worker_health(worker_id: str, data: dict[str, str]) -> WorkerHealth:
    """Parse a Redis hash into a WorkerHealth dataclass."""
    last_hb_str = data.get("last_heartbeat", "")
    last_hb = float(last_hb_str) if last_hb_str else None
    started_str = data.get("started_at", "")
    started = float(started_str) if started_str else None
    current = data.get("current_task_id", "") or None

    alive = True
    if last_hb is not None:
        alive = (time.time() - last_hb) < 60.0

    return WorkerHealth(
        worker_id=worker_id,
        status=data.get("status", "unknown"),
        tasks_processed=int(data.get("tasks_processed", "0")),
        tasks_failed=int(data.get("tasks_failed", "0")),
        current_task_id=current,
        started_at=started,
        last_heartbeat=last_hb,
        concurrency=int(data.get("concurrency", "1")),
        hostname=data.get("hostname", ""),
        alive=alive,
    )


async def get_worker_fleet_status(redis_url: str) -> list[WorkerHealth]:
    """Return health info for all active workers.

    Scans for ``exo:workers:*`` keys in Redis and parses each into a
    :class:`WorkerHealth` instance.  Workers whose heartbeat TTL has expired
    are automatically cleaned up by Redis and will not appear.

    Uses a Redis pipeline to batch ``HGETALL`` calls across all keys found
    in a scan page, reducing round-trips from O(n) to O(scan_pages).

    Args:
        redis_url: Redis connection URL.

    Returns:
        List of :class:`WorkerHealth` for all workers found.
    """
    logger.debug("Scanning fleet status from Redis")
    r: aioredis.Redis = aioredis.from_url(
        redis_url, decode_responses=True, socket_connect_timeout=5, socket_timeout=5
    )
    try:
        workers: list[WorkerHealth] = []
        prefix = "exo:workers:"

        # SCAN for all worker keys, then batch HGETALL via pipeline per page.
        cursor: int | bytes = 0
        while True:
            cursor, keys = await r.scan(cursor=cursor, match=f"{prefix}*", count=100)  # type: ignore[misc]
            if keys:
                pipe = r.pipeline(transaction=False)
                for key in keys:
                    pipe.hgetall(key)  # type: ignore[misc]
                results: list[dict[str, str]] = await pipe.execute()  # type: ignore[misc]
                for key, data in zip(keys, results, strict=False):
                    if data:
                        worker_id = str(key).removeprefix(prefix)
                        workers.append(_parse_worker_health(worker_id, data))
            if cursor == 0:
                break

        logger.debug("Fleet status: found %d workers", len(workers))
        return workers
    finally:
        await r.aclose()
