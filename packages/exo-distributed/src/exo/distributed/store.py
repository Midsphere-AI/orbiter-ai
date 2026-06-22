"""Redis-backed task state store for tracking distributed task status."""

from __future__ import annotations

import json
import logging
from typing import Any

from exo.distributed._redis_mixin import (  # pyright: ignore[reportMissingImports]
    RedisConnectionMixin,
)
from exo.distributed.models import (  # pyright: ignore[reportMissingImports]
    TaskResult,
    TaskStatus,
)

logger = logging.getLogger(__name__)


class TaskStore(RedisConnectionMixin):
    """Tracks task status in Redis hashes with TTL-based auto-cleanup.

    Each task is stored as a Redis hash at ``{prefix}{task_id}``.  A secondary
    index set (``{prefix}index``) records all known task IDs for listing.
    """

    def __init__(
        self,
        redis_url: str,
        *,
        prefix: str = "exo:task:",
        ttl_seconds: int = 86400,
    ) -> None:
        super().__init__()
        self._redis_url = redis_url
        self._prefix = prefix
        self._ttl_seconds = ttl_seconds

    async def connect(self) -> None:
        """Connect to Redis."""
        logger.debug("TaskStore connecting to Redis (prefix=%s)", self._prefix)
        await super().connect()

    def _key(self, task_id: str) -> str:
        return f"{self._prefix}{task_id}"

    @property
    def _index_key(self) -> str:
        return f"{self._prefix}index"

    async def set_status(self, task_id: str, status: TaskStatus, **kwargs: Any) -> None:
        """Update task state in a Redis hash.

        Extra keyword arguments are stored alongside the status (e.g.
        ``worker_id``, ``error``, ``result``, ``started_at``, ``completed_at``,
        ``retries``).
        """
        r = self._client()
        key = self._key(task_id)

        fields: dict[str, str] = {"task_id": task_id, "status": str(status)}
        for k, v in kwargs.items():
            if v is None:
                fields[k] = ""
            elif isinstance(v, dict):
                fields[k] = json.dumps(v)
            else:
                fields[k] = str(v)

        await r.hset(key, mapping=fields)  # type: ignore[misc]
        await r.expire(key, self._ttl_seconds)  # type: ignore[misc]

        # Maintain secondary index for list_tasks().
        await r.sadd(self._index_key, task_id)  # type: ignore[misc]
        logger.debug("TaskStore set_status task %s -> %s", task_id, status)

    async def get_status(self, task_id: str) -> TaskResult | None:
        """Retrieve current task state, or ``None`` if not found."""
        r = self._client()
        data = await r.hgetall(self._key(task_id))  # type: ignore[misc]
        if not data:
            return None
        return self._parse_result(data)

    async def list_tasks(
        self,
        status: TaskStatus | None = None,
        limit: int = 100,
    ) -> list[TaskResult]:
        """List tasks, optionally filtered by *status*.

        Uses ``SSCAN`` to iterate the index set in batches (avoids blocking
        the server with a full ``SMEMBERS`` on large indices) and fetches task
        hashes in pipelined batches.  Members whose hash no longer exists
        (TTL expired) are pruned from the index set to keep it bounded.
        """
        r = self._client()
        results: list[TaskResult] = []
        stale_members: list[str] = []

        scan_batch = 100
        cursor: int | bytes = 0
        while True:
            cursor, members = await r.sscan(  # type: ignore[misc]
                self._index_key, cursor=cursor, count=scan_batch
            )
            if members:
                # Batch-fetch all hashes in this scan page via a pipeline.
                pipe = r.pipeline(transaction=False)
                for tid in members:
                    pipe.hgetall(self._key(tid))  # type: ignore[misc]
                batch_data: list[dict[str, str]] = await pipe.execute()  # type: ignore[misc]

                for tid, data in zip(members, batch_data, strict=False):
                    if not data:
                        # Hash expired — queue for index pruning.
                        stale_members.append(tid)
                        continue
                    result = self._parse_result(data)
                    if status is not None and result.status != status:
                        continue
                    results.append(result)
                    if len(results) >= limit:
                        # Prune stale members collected so far before returning.
                        if stale_members:
                            await r.srem(self._index_key, *stale_members)  # type: ignore[misc]
                        return results

            if cursor == 0:
                break

        # Prune stale members found during this scan.
        if stale_members:
            await r.srem(self._index_key, *stale_members)  # type: ignore[misc]
            logger.debug(
                "TaskStore list_tasks pruned %d stale members from index", len(stale_members)
            )

        return results

    @staticmethod
    def _parse_result(data: dict[str, str]) -> TaskResult:
        """Convert a Redis hash dict into a ``TaskResult``."""
        parsed: dict[str, Any] = {}
        parsed["task_id"] = data.get("task_id", "")
        parsed["status"] = data.get("status", TaskStatus.PENDING)

        # result is JSON-encoded dict or empty.
        raw_result = data.get("result", "")
        parsed["result"] = json.loads(raw_result) if raw_result else None

        raw_error = data.get("error", "")
        parsed["error"] = raw_error if raw_error else None

        raw_started = data.get("started_at", "")
        parsed["started_at"] = float(raw_started) if raw_started else None

        raw_completed = data.get("completed_at", "")
        parsed["completed_at"] = float(raw_completed) if raw_completed else None

        raw_worker = data.get("worker_id", "")
        parsed["worker_id"] = raw_worker if raw_worker else None

        raw_retries = data.get("retries", "")
        parsed["retries"] = int(raw_retries) if raw_retries else 0

        return TaskResult(**parsed)
