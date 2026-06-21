"""High-level API for submitting agent execution to the distributed queue."""

from __future__ import annotations

import asyncio
import logging
import os
import time
from collections.abc import AsyncIterator
from typing import Any

from exo.distributed._propagation import (  # pyright: ignore[reportMissingImports]
    BaggagePropagator,
    DictCarrier,
)
from exo.distributed.broker import TaskBroker  # pyright: ignore[reportMissingImports]
from exo.distributed.events import EventSubscriber  # pyright: ignore[reportMissingImports]
from exo.distributed.models import (  # pyright: ignore[reportMissingImports]
    TaskPayload,
    TaskResult,
    TaskStatus,
)
from exo.distributed.store import TaskStore  # pyright: ignore[reportMissingImports]
from exo.observability.tracing import aspan  # pyright: ignore[reportMissingImports]
from exo.types import ExoError, StreamEvent  # pyright: ignore[reportMissingImports]

logger = logging.getLogger(__name__)

_DEFAULT_REDIS_ENV = "EXO_REDIS_URL"


class TaskHandle:
    """Handle returned by :func:`distributed` for monitoring a submitted task.

    Provides methods to wait for the result, stream live events, check
    status, and cancel the task.

    Args:
        task_id: The unique task identifier.
        broker: Connected :class:`TaskBroker` instance.
        store: Connected :class:`TaskStore` instance.
        subscriber: Connected :class:`EventSubscriber` instance.
    """

    def __init__(
        self,
        task_id: str,
        *,
        broker: TaskBroker,
        store: TaskStore,
        subscriber: EventSubscriber,
    ) -> None:
        self._task_id = task_id
        self._broker = broker
        self._store = store
        self._subscriber = subscriber

    @property
    def task_id(self) -> str:
        """The unique task identifier."""
        return self._task_id

    async def close(self) -> None:
        """Close underlying Redis connections for broker, store, and subscriber.

        Always call this (or use the async context manager) when the handle is
        no longer needed to avoid leaking Redis connections.
        """
        logger.debug("TaskHandle closing connections for task %s", self._task_id)
        for component in (self._broker, self._store, self._subscriber):
            try:
                await component.disconnect()
            except Exception:
                logger.warning(
                    "Error disconnecting %s for task %s",
                    type(component).__name__,
                    self._task_id,
                    exc_info=True,
                )

    async def __aenter__(self) -> TaskHandle:
        return self

    async def __aexit__(self, *exc_info: Any) -> None:
        await self.close()

    async def result(self, *, poll_interval: float = 0.5) -> dict[str, Any]:
        """Block until the task completes and return the result dict.

        Polls :class:`TaskStore` at *poll_interval* seconds until the task
        reaches a terminal status (``COMPLETED``, ``FAILED``, or
        ``CANCELLED``).

        Returns:
            The ``result`` dict from :class:`TaskResult` on success.

        Raises:
            RuntimeError: If the task failed or was cancelled.
        """
        logger.debug("TaskHandle waiting for result of task %s", self._task_id)
        terminal = {TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED}
        while True:
            task_result = await self._store.get_status(self._task_id)
            if task_result is not None and task_result.status in terminal:
                if task_result.status == TaskStatus.COMPLETED:
                    logger.info("Task %s result received (status=completed)", self._task_id)
                    return task_result.result or {}
                if task_result.status == TaskStatus.CANCELLED:
                    logger.info("Task %s was cancelled", self._task_id)
                    msg = f"Task {self._task_id} was cancelled"
                    raise ExoError(msg)
                # FAILED
                logger.error(
                    "Task %s failed: %s", self._task_id, task_result.error or "unknown error"
                )
                msg = f"Task {self._task_id} failed: {task_result.error or 'unknown error'}"
                raise ExoError(msg)
            await asyncio.sleep(poll_interval)

    async def stream(self) -> AsyncIterator[StreamEvent]:
        """Subscribe to live streaming events for this task.

        Yields deserialized :class:`StreamEvent` instances via Redis Pub/Sub.
        The iterator ends when a terminal ``StatusEvent`` is received.
        """
        async for event in self._subscriber.subscribe(self._task_id):
            yield event

    async def cancel(self) -> None:
        """Cancel the running task."""
        logger.info("TaskHandle requesting cancellation of task %s", self._task_id)
        await self._broker.cancel(self._task_id)

    async def status(self) -> TaskResult | None:
        """Return the current task status."""
        return await self._store.get_status(self._task_id)


async def distributed(
    agent: Any,
    input: str,
    *,
    messages: list[dict[str, Any]] | None = None,
    redis_url: str | None = None,
    detailed: bool = False,
    timeout: float = 300.0,
    metadata: dict[str, Any] | None = None,
) -> TaskHandle:
    """Submit agent execution to the distributed queue.

    Serializes the agent (or swarm) to a :class:`TaskPayload`, enqueues it
    via :class:`TaskBroker`, and returns a :class:`TaskHandle` for
    monitoring the result.

    Args:
        agent: An ``Agent`` or ``Swarm`` instance (must support ``to_dict()``).
        input: The input string for the agent.
        messages: Optional message history.
        redis_url: Redis connection URL.  Defaults to the
            ``EXO_REDIS_URL`` environment variable.
        detailed: Whether to enable rich streaming events.
        timeout: Task timeout in seconds.
        metadata: Arbitrary metadata dict attached to the task payload.

    Returns:
        A :class:`TaskHandle` for result retrieval, streaming, and cancellation.

    Raises:
        ValueError: If *redis_url* is not provided and ``EXO_REDIS_URL``
            is not set.
    """
    url = redis_url or os.environ.get(_DEFAULT_REDIS_ENV)
    if url is None:
        msg = "redis_url must be provided or EXO_REDIS_URL environment variable must be set"
        raise ExoError(msg)

    agent_name = getattr(agent, "name", "")
    logger.debug("distributed() submitting task (agent=%s)", agent_name)

    broker = TaskBroker(url)
    store = TaskStore(url)
    subscriber = EventSubscriber(url)

    await broker.connect()
    await store.connect()
    await subscriber.connect()

    # Build metadata with trace context propagation
    task_metadata = dict(metadata) if metadata else {}
    propagator = BaggagePropagator()
    carrier = DictCarrier()
    propagator.inject(carrier)
    if carrier.headers:
        task_metadata["trace_context"] = carrier.headers

    payload = TaskPayload(
        agent_config=agent.to_dict(),
        input=input,
        messages=messages or [],
        detailed=detailed,
        metadata=task_metadata,
        created_at=time.time(),
        timeout_seconds=timeout,
    )

    async with aspan(
        "exo.distributed.submit",
        attributes={
            "dist.task_id": payload.task_id,
            "dist.agent_name": agent_name,
        },
    ):
        await broker.submit(payload)

    logger.info(
        "distributed() task %s submitted (agent=%s, timeout=%.0fs)",
        payload.task_id,
        agent_name,
        timeout,
    )
    return TaskHandle(
        payload.task_id,
        broker=broker,
        store=store,
        subscriber=subscriber,
    )
