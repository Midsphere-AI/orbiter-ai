"""Redis Streams-backed task broker for distributed agent execution."""

from __future__ import annotations

import json
import logging

import redis.asyncio as aioredis

from exo.distributed._redis_mixin import (  # pyright: ignore[reportMissingImports]
    RedisConnectionMixin,
)
from exo.distributed.metrics import (  # pyright: ignore[reportMissingImports]
    record_task_submitted,
)
from exo.distributed.models import (  # pyright: ignore[reportMissingImports]
    TaskPayload,
    TaskStatus,
)
from exo.types import ExoError  # pyright: ignore[reportMissingImports]

logger = logging.getLogger(__name__)


class TaskBroker(RedisConnectionMixin):
    """Enqueues and distributes agent execution tasks via Redis Streams.

    Uses Redis Streams (XADD/XREADGROUP/XACK) with consumer groups for
    durable, multi-worker task distribution.
    """

    def __init__(
        self,
        redis_url: str,
        *,
        queue_name: str = "exo:tasks",
        max_retries: int = 3,
        task_key_prefix: str = "exo:task:",
    ) -> None:
        super().__init__()
        self._redis_url = redis_url
        self._queue_name = queue_name
        self._max_retries = max_retries
        self._task_key_prefix = task_key_prefix
        self._group_name = f"{queue_name}:group"
        self._pending_ids: dict[str, str] = {}

    @property
    def max_retries(self) -> int:
        return self._max_retries

    async def connect(self) -> None:
        """Connect to Redis and ensure the consumer group exists."""
        await super().connect()
        r = self._client()
        try:
            # id="$" means "only deliver messages added after this point" — a
            # fresh group should not replay historical undelivered entries on
            # the first connect.  Use id="0" only when you intentionally want
            # to reprocess the full backlog (e.g. a replay consumer).
            await r.xgroup_create(self._queue_name, self._group_name, id="$", mkstream=True)
            logger.debug("TaskBroker created consumer group %s", self._group_name)
        except aioredis.ResponseError as exc:
            # Group already exists — safe to ignore.
            if "BUSYGROUP" not in str(exc):
                raise ExoError(
                    f"Failed to create consumer group {self._group_name!r} on stream {self._queue_name!r}: {exc}",
                    context={"stream": self._queue_name, "group": self._group_name},
                    hint=(
                        "Ensure Redis is reachable and the stream key is not occupied by a different type. "
                        "If the stream already exists as a different data type, delete it and reconnect."
                    ),
                ) from exc
            logger.debug("TaskBroker consumer group %s already exists", self._group_name)

    async def submit(self, task: TaskPayload) -> str:
        """Enqueue a task and return its task_id."""
        r = self._client()
        data = task.model_dump()
        await r.xadd(self._queue_name, {"payload": json.dumps(data)})
        record_task_submitted(task_id=task.task_id, queue_name=self._queue_name)
        logger.debug("TaskBroker submitted task %s to %s", task.task_id, self._queue_name)
        return task.task_id

    async def claim(self, worker_id: str, *, timeout: float = 5.0) -> TaskPayload | None:
        """Pop the next task from the queue (blocking with *timeout* seconds).

        Returns ``None`` if no task is available within the timeout.
        """
        r = self._client()
        block_ms = int(timeout * 1000)
        result = await r.xreadgroup(
            self._group_name,
            worker_id,
            {self._queue_name: ">"},
            count=1,
            block=block_ms,
        )
        if not result:
            return None

        _stream_name, messages = result[0]
        msg_id, fields = messages[0]
        payload_json: str = fields["payload"]
        payload = TaskPayload(**json.loads(payload_json))
        # Store the stream message id so ack/nack can reference it.
        self._pending_ids[payload.task_id] = msg_id
        logger.debug(
            "TaskBroker worker %s claimed task %s (msg_id=%s)",
            worker_id,
            payload.task_id,
            msg_id,
        )
        return payload

    async def ack(self, task_id: str) -> None:
        """Acknowledge successful processing of a task."""
        r = self._client()
        msg_id = self._pending_ids.pop(task_id, None)
        if msg_id is not None:
            await r.xack(self._queue_name, self._group_name, msg_id)
            logger.debug("TaskBroker acked task %s (msg_id=%s)", task_id, msg_id)
        else:
            logger.warning("TaskBroker ack called for unknown task %s (no pending msg_id)", task_id)

    async def nack(self, task_id: str) -> None:
        """Return a task to the queue for retry by another consumer.

        The message is acknowledged (removed from this consumer's PEL) and
        re-added to the stream so any consumer in the group can pick it up.
        """
        r = self._client()
        msg_id = self._pending_ids.pop(task_id, None)
        if msg_id is not None:
            # Read the original message data before acking.
            msgs = await r.xrange(self._queue_name, min=msg_id, max=msg_id)
            if msgs:
                # Happy path: ack the original entry then re-add it so any
                # consumer in the group can pick it up for retry.
                _mid, fields = msgs[0]
                try:
                    await r.xack(self._queue_name, self._group_name, msg_id)
                except Exception:
                    logger.error(
                        "TaskBroker nack for task %s: xack failed — message may remain stuck in PEL",
                        task_id,
                        exc_info=True,
                    )
                    return
                try:
                    await r.xadd(self._queue_name, {"payload": fields["payload"]})
                    logger.debug("TaskBroker nacked task %s — re-queued for retry", task_id)
                except Exception:
                    logger.error(
                        "TaskBroker nack for task %s: xadd re-queue failed — task may be lost",
                        task_id,
                        exc_info=True,
                    )
            else:
                # The original stream entry is gone (stream trimmed or Redis
                # restarted).  Ack it to clear the PEL so the worker is not
                # permanently blocked, but log at ERROR so the operator knows
                # the retry payload is unrecoverable.
                logger.error(
                    "TaskBroker nack for task %s: original message %s not found in stream — "
                    "acking to clear PEL; retry payload is unrecoverable",
                    task_id,
                    msg_id,
                )
                try:
                    await r.xack(self._queue_name, self._group_name, msg_id)
                except Exception:
                    logger.error(
                        "TaskBroker nack for task %s: xack (PEL-clear) also failed — "
                        "message stuck in PEL",
                        task_id,
                        exc_info=True,
                    )
        else:
            logger.warning(
                "TaskBroker nack called for unknown task %s (no pending msg_id)", task_id
            )

    async def cancel(self, task_id: str) -> None:
        """Cancel a running task.

        Publishes a cancel signal to ``exo:cancel:{task_id}`` Pub/Sub
        channel and sets the task status to CANCELLED in the task hash.
        """
        r = self._client()
        logger.info("TaskBroker cancelling task %s", task_id)
        # Publish cancel signal via Pub/Sub.
        await r.publish(  # type: ignore[misc]
            f"exo:cancel:{task_id}", "cancel"
        )
        # Update status in the task hash using the same prefix as TaskStore.
        key = f"{self._task_key_prefix}{task_id}"
        await r.hset(key, mapping={"status": str(TaskStatus.CANCELLED)})  # type: ignore[misc]
