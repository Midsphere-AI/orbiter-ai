"""Distributed worker process — claims tasks from the queue and executes agents."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import random
import signal
import socket
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal

import redis.asyncio as aioredis

from exo.distributed._propagation import (  # pyright: ignore[reportMissingImports]
    BaggagePropagator,
    DictCarrier,
)
from exo.distributed._redis_config import RedisConfig  # pyright: ignore[reportMissingImports]
from exo.distributed.broker import TaskBroker  # pyright: ignore[reportMissingImports]
from exo.distributed.cancel import CancellationToken  # pyright: ignore[reportMissingImports]
from exo.distributed.events import EventPublisher  # pyright: ignore[reportMissingImports]
from exo.distributed.metrics import (  # pyright: ignore[reportMissingImports]
    record_task_cancelled,
    record_task_completed,
    record_task_failed,
)
from exo.distributed.models import (  # pyright: ignore[reportMissingImports]
    TaskPayload,
    TaskStatus,
)
from exo.distributed.store import TaskStore  # pyright: ignore[reportMissingImports]
from exo.distributed.temporal import HAS_TEMPORAL  # pyright: ignore[reportMissingImports]
from exo.observability.tracing import aspan  # pyright: ignore[reportMissingImports]
from exo.types import ExoError  # pyright: ignore[reportMissingImports]

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from exo.distributed.temporal import (  # pyright: ignore[reportMissingImports]
        TemporalExecutor,
    )


def _generate_worker_id() -> str:
    """Generate a unique worker ID from hostname, PID, and a random suffix."""
    hostname = socket.gethostname()
    pid = os.getpid()
    suffix = random.randbytes(4).hex()
    return f"{hostname}-{pid}-{suffix}"


def _deserialize_messages(raw: list[dict[str, Any]]) -> list[Any]:
    """Convert a list of message dicts to typed Message objects.

    Dispatches on the ``role`` field to create the appropriate Pydantic model.

    Args:
        raw: List of message dicts, each with a ``role`` key.

    Returns:
        List of typed Message objects (UserMessage, AssistantMessage, etc.).

    Raises:
        ValueError: If a message has an unknown role.
    """
    from exo.types import (  # pyright: ignore[reportMissingImports]
        AssistantMessage,
        SystemMessage,
        ToolResult,
        UserMessage,
    )

    _role_map = {
        "user": UserMessage,
        "assistant": AssistantMessage,
        "system": SystemMessage,
        "tool": ToolResult,
    }

    messages: list[Any] = []
    for msg in raw:
        role = msg.get("role", "")
        cls = _role_map.get(role)
        if cls is None:
            raise ExoError(f"Unknown message role: {role!r}")
        messages.append(cls(**msg))
    return messages


class Worker:
    """Claims tasks from a Redis queue, executes agents, and publishes events.

    The worker:
    1. Claims tasks from the queue via :class:`TaskBroker`.
    2. Reconstructs the agent from ``agent_config`` via ``Agent.from_dict()``.
    3. Runs ``run.stream()`` with ``detailed=task.detailed``.
    4. Publishes each streaming event via :class:`EventPublisher`.
    5. Updates task status in :class:`TaskStore`.
    6. Publishes a heartbeat to Redis periodically.

    **Grouped constructor** (preferred for new code)::

        from exo.distributed import Worker, RedisConfig

        worker = Worker(redis=RedisConfig(url="redis://localhost:6379"))

    **Flat constructor** (backward-compatible, always works)::

        worker = Worker("redis://localhost:6379", queue_name="exo:tasks")

    Args:
        redis_url: Redis connection URL.  Mutually exclusive with *redis=*.
        worker_id: Unique worker identifier. Auto-generated if not provided.
        concurrency: Number of concurrent task executions (default 1).
        queue_name: Redis Streams queue name (default ``"exo:tasks"``).
            Mutually exclusive with *redis=*.
        heartbeat_ttl: TTL in seconds for the heartbeat key (default 30).
            Mutually exclusive with *redis=*.
        executor: Execution backend — ``"local"`` for direct execution or
            ``"temporal"`` for durable Temporal workflows.
        redis: Grouped Redis connection config (:class:`RedisConfig`).
            When provided, supplies *redis_url*, *queue_name*, and
            *heartbeat_ttl* together.  Raises :exc:`ValueError` if any of
            the flat counterparts are also explicitly set.
    """

    def __init__(
        self,
        redis_url: str = "",
        *,
        worker_id: str | None = None,
        concurrency: int = 1,
        queue_name: str = "exo:tasks",
        heartbeat_ttl: int = 30,
        executor: Literal["local", "temporal"] = "local",
        provider_factory: Callable[[str], Any] | None = None,
        redis: RedisConfig | None = None,
    ) -> None:
        # --- grouped vs flat resolution ---
        if redis is not None:
            # Detect explicit conflicts with the flat counterparts.
            # We can only detect positional redis_url if it was given as non-empty;
            # keyword conflicts are checked individually.
            if redis_url:
                raise ValueError(
                    "Cannot specify both 'redis=' and 'redis_url='. "
                    "Use RedisConfig(url=...) or the flat 'redis_url=' kwarg, not both."
                )
            if queue_name != "exo:tasks":
                raise ValueError(
                    "Cannot specify both 'redis=' and 'queue_name='. "
                    "Set queue_name via RedisConfig(queue_name=...) instead."
                )
            if heartbeat_ttl != 30:
                raise ValueError(
                    "Cannot specify both 'redis=' and 'heartbeat_ttl='. "
                    "Set heartbeat_ttl via RedisConfig(heartbeat_ttl=...) instead."
                )
            resolved_url = redis.url
            resolved_queue = redis.queue_name
            resolved_ttl = redis.heartbeat_ttl
        else:
            if not redis_url:
                raise ValueError(
                    "A Redis URL is required. Pass redis_url='redis://...' "
                    "or redis=RedisConfig(url='redis://...')."
                )
            resolved_url = redis_url
            resolved_queue = queue_name
            resolved_ttl = heartbeat_ttl

        self._redis_url = resolved_url
        self._worker_id = worker_id or _generate_worker_id()
        self._concurrency = concurrency
        self._queue_name = resolved_queue
        self._heartbeat_ttl = resolved_ttl
        self._executor_type = executor
        self._provider_factory = provider_factory

        self._broker = TaskBroker(resolved_url, queue_name=resolved_queue)
        self._store = TaskStore(resolved_url)
        self._publisher = EventPublisher(resolved_url)
        self._temporal_executor: TemporalExecutor | None = None

        if executor == "temporal":
            if not HAS_TEMPORAL:
                msg = (
                    "Temporal executor requires temporalio to be installed. "
                    "Install it with: pip install exo-distributed[temporal]"
                )
                raise ImportError(msg)
            from exo.distributed.temporal import (  # pyright: ignore[reportMissingImports]
                TemporalExecutor,
            )

            self._temporal_executor = TemporalExecutor()

        self._shutdown_event = asyncio.Event()
        self._tasks_processed = 0
        self._tasks_failed = 0
        self._current_task_id: str | None = None
        self._started_at: float = 0.0

    @property
    def worker_id(self) -> str:
        return self._worker_id

    @property
    def tasks_processed(self) -> int:
        return self._tasks_processed

    @property
    def tasks_failed(self) -> int:
        return self._tasks_failed

    @property
    def redis_config(self) -> RedisConfig:
        """Return the active Redis connection settings as a :class:`RedisConfig`.

        Provides a grouped read-only view of the connection parameters
        regardless of whether the worker was constructed with flat kwargs or
        the ``redis=RedisConfig(...)`` shorthand.
        """
        return RedisConfig(
            url=self._redis_url,
            queue_name=self._queue_name,
            heartbeat_ttl=self._heartbeat_ttl,
        )

    async def start(self) -> None:
        """Enter the claim-execute loop until shutdown is signalled.

        Registers SIGINT/SIGTERM handlers for graceful shutdown.
        When ``executor="temporal"``, also connects the Temporal executor.
        """
        self._started_at = time.time()

        logger.info(
            "Worker %s starting (concurrency=%d, executor=%s, queue=%s)",
            self._worker_id,
            self._concurrency,
            self._executor_type,
            self._queue_name,
        )

        await self._broker.connect()
        await self._store.connect()
        await self._publisher.connect()

        if self._temporal_executor is not None:
            await self._temporal_executor.connect()
            logger.info("Worker %s connected to Temporal", self._worker_id)

        # Register signal handlers for graceful shutdown
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, self._handle_signal)

        logger.info("Worker %s ready, waiting for tasks", self._worker_id)

        try:
            # Start heartbeat in the background
            heartbeat_task = asyncio.create_task(self._heartbeat_loop())

            # Run claim-execute workers
            workers = [asyncio.create_task(self._claim_loop()) for _ in range(self._concurrency)]
            await asyncio.gather(*workers)
            heartbeat_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await heartbeat_task
        finally:
            logger.info(
                "Worker %s shutting down (processed=%d, failed=%d)",
                self._worker_id,
                self._tasks_processed,
                self._tasks_failed,
            )
            if self._temporal_executor is not None:
                await self._temporal_executor.disconnect()
            await self._broker.disconnect()
            await self._store.disconnect()
            await self._publisher.disconnect()

    async def stop(self) -> None:
        """Signal the worker to shut down gracefully."""
        self._shutdown_event.set()

    def _handle_signal(self) -> None:
        """Signal handler for SIGINT/SIGTERM."""
        logger.info("Worker %s received shutdown signal", self._worker_id)
        self._shutdown_event.set()

    async def _heartbeat_loop(self) -> None:
        """Publish a heartbeat to Redis periodically."""
        r: aioredis.Redis = aioredis.from_url(self._redis_url, decode_responses=True)
        key = f"exo:workers:{self._worker_id}"
        try:
            while not self._shutdown_event.is_set():
                try:
                    fields = {
                        "status": "running",
                        "tasks_processed": str(self._tasks_processed),
                        "tasks_failed": str(self._tasks_failed),
                        "current_task_id": self._current_task_id or "",
                        "started_at": str(self._started_at),
                        "last_heartbeat": str(time.time()),
                        "concurrency": str(self._concurrency),
                        "hostname": socket.gethostname(),
                    }
                    await r.hset(key, mapping=fields)  # type: ignore[misc]
                    await r.expire(key, self._heartbeat_ttl)  # type: ignore[misc]
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.warning(
                        "Worker %s heartbeat failed, will retry next interval",
                        self._worker_id,
                        exc_info=True,
                    )
                await asyncio.sleep(self._heartbeat_ttl / 3)
        finally:
            await r.aclose()

    async def _claim_loop(self) -> None:
        """Claim tasks and execute them until shutdown."""
        while not self._shutdown_event.is_set():
            try:
                task = await self._broker.claim(self._worker_id, timeout=2.0)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.error(
                    "Worker %s error claiming task, retrying in 2s",
                    self._worker_id,
                    exc_info=True,
                )
                await asyncio.sleep(2.0)
                continue
            if task is None:
                continue
            await self._execute_task(task)

    async def _execute_task(self, task: TaskPayload) -> None:
        """Execute a single task: reconstruct agent, stream, update status."""
        logger.info(
            "Worker %s claimed task %s (input=%.80s...)",
            self._worker_id,
            task.task_id,
            task.input,
        )
        self._current_task_id = task.task_id
        token = CancellationToken()

        # Extract trace context from task metadata (injected by client)
        trace_context = task.metadata.get("trace_context")
        if isinstance(trace_context, dict):
            carrier = DictCarrier(trace_context)
            propagator = BaggagePropagator()
            propagator.extract(carrier)

        # Start listening for cancel signals on exo:cancel:{task_id}
        cancel_task = asyncio.create_task(self._listen_for_cancel(task.task_id, token))

        started_at = time.time()
        wait_time = started_at - task.created_at if task.created_at > 0 else 0.0

        # Track final outcome for on_task_done
        final_status: TaskStatus = TaskStatus.FAILED
        final_result: str | None = None
        final_error: str | None = None

        # Memory hydration state
        persistence: Any = None
        mem_result: tuple[Any, Any] | None = None
        agent: Any = None

        async with aspan(
            "exo.distributed.execute",
            attributes={
                "dist.task_id": task.task_id,
                "dist.worker_id": self._worker_id,
            },
        ) as s:
            try:
                # Mark as RUNNING
                await self._store.set_status(
                    task.task_id,
                    TaskStatus.RUNNING,
                    worker_id=self._worker_id,
                    started_at=started_at,
                )

                if self._temporal_executor is not None:
                    # Delegate to Temporal for durable execution
                    result_text = await self._temporal_executor.execute_task(
                        task,
                        self._store,
                        self._publisher,
                        token,
                        self._worker_id,
                    )
                else:
                    # Local execution: reconstruct agent and stream directly
                    agent = self._reconstruct_agent(task.agent_config)
                    logger.debug(
                        "Task %s: reconstructed agent '%s' (model=%s)",
                        task.task_id,
                        getattr(agent, "name", "?"),
                        getattr(agent, "model", "?"),
                    )

                    # Memory hydration (Feature 4)
                    mem_config = task.metadata.get("memory")
                    if mem_config:
                        logger.info(
                            "Task %s: hydrating memory (backend=%s)",
                            task.task_id,
                            mem_config.get("backend", "short_term"),
                        )
                        from exo.distributed.memory import (  # pyright: ignore[reportMissingImports]
                            create_memory_store,
                        )

                        store, mem_metadata = await create_memory_store(mem_config, task.task_id)
                        agent.memory = store

                        _memory_persistence_cls: type | None = None
                        try:
                            from exo.memory.persistence import (  # pyright: ignore[reportMissingImports]
                                MemoryPersistence as _MemoryPersistence,
                            )

                            _memory_persistence_cls = _MemoryPersistence
                        except ImportError:
                            logger.warning(
                                "Task %s: exo-memory is not installed; "
                                "memory persistence will be skipped. "
                                "Install it with: pip install exo-memory",
                                task.task_id,
                            )

                        if _memory_persistence_cls is not None:
                            persistence = _memory_persistence_cls(store, metadata=mem_metadata)
                            # L-6: Guard against double hook registration. If the agent
                            # already has a MemoryPersistence attached (e.g. from
                            # Agent.__init__ with memory=), detach it first.
                            existing_persistence = getattr(agent, "_memory_persistence", None)
                            if existing_persistence is not None:
                                existing_persistence.detach(agent)
                                logger.debug(
                                    "Task %s: detached existing memory persistence before re-attaching",
                                    task.task_id,
                                )
                            persistence.attach(agent)
                            logger.debug(
                                "Task %s: memory persistence attached (scope=%s)",
                                task.task_id,
                                mem_metadata,
                            )

                        from exo.memory.base import (  # pyright: ignore[reportMissingImports]
                            HumanMemory,
                        )

                        await store.add(HumanMemory(content=task.input, metadata=mem_metadata))
                        mem_result = (store, mem_metadata)

                    result_text = await self._run_agent(agent, task, token)

                duration = time.time() - started_at

                if token.cancelled:
                    # Cancellation took effect during execution
                    logger.info("Task %s cancelled after %.2fs", task.task_id, duration)
                    await self._store.set_status(
                        task.task_id,
                        TaskStatus.CANCELLED,
                        completed_at=time.time(),
                    )
                    await self._broker.ack(task.task_id)
                    record_task_cancelled(
                        task_id=task.task_id,
                        worker_id=self._worker_id,
                    )
                    final_status = TaskStatus.CANCELLED
                else:
                    # Mark as COMPLETED
                    logger.info(
                        "Task %s completed in %.2fs (output_len=%d)",
                        task.task_id,
                        duration,
                        len(result_text),
                    )
                    await self._store.set_status(
                        task.task_id,
                        TaskStatus.COMPLETED,
                        completed_at=time.time(),
                        result={"output": result_text},
                    )
                    await self._broker.ack(task.task_id)
                    self._tasks_processed += 1
                    record_task_completed(
                        task_id=task.task_id,
                        worker_id=self._worker_id,
                        duration=duration,
                        wait_time=wait_time,
                    )
                    final_status = TaskStatus.COMPLETED
                    final_result = result_text

            except Exception as exc:
                s.record_exception(exc)
                self._tasks_failed += 1
                duration = time.time() - started_at
                final_error = str(exc)
                logger.error(
                    "Task %s failed after %.2fs: %s",
                    task.task_id,
                    duration,
                    exc,
                    exc_info=True,
                )
                await self._store.set_status(
                    task.task_id,
                    TaskStatus.FAILED,
                    completed_at=time.time(),
                    error=str(exc),
                )
                record_task_failed(
                    task_id=task.task_id,
                    worker_id=self._worker_id,
                    duration=duration,
                )

                # Check if retries remain
                status = await self._store.get_status(task.task_id)
                retries = status.retries if status else 0
                if retries < self._broker.max_retries:
                    logger.info(
                        "Task %s scheduling retry %d/%d",
                        task.task_id,
                        retries + 1,
                        self._broker.max_retries,
                    )
                    await self._store.set_status(
                        task.task_id,
                        TaskStatus.RETRYING,
                        retries=retries + 1,
                    )
                    await self._broker.nack(task.task_id)
                else:
                    logger.warning(
                        "Task %s exhausted all %d retries",
                        task.task_id,
                        self._broker.max_retries,
                    )
                    await self._broker.ack(task.task_id)

            finally:
                # Tear down memory persistence
                if persistence is not None and agent is not None:
                    persistence.detach(agent)
                    logger.debug("Task %s: memory persistence detached", task.task_id)
                if mem_result is not None:
                    from exo.distributed.memory import (  # pyright: ignore[reportMissingImports]
                        teardown_memory_store,
                    )

                    with contextlib.suppress(Exception):
                        await teardown_memory_store(mem_result[0])
                    logger.debug("Task %s: memory store torn down", task.task_id)

                self._current_task_id = None
                cancel_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await cancel_task

                try:
                    await self.on_task_done(task, final_status, final_result, final_error)
                except Exception:
                    logger.exception("on_task_done failed for task %s", task.task_id)

    async def _listen_for_cancel(self, task_id: str, token: CancellationToken) -> None:
        """Subscribe to ``exo:cancel:{task_id}`` and set the token on signal."""
        r: aioredis.Redis = aioredis.from_url(self._redis_url, decode_responses=True)
        channel_name = f"exo:cancel:{task_id}"
        pubsub = r.pubsub()
        try:
            await pubsub.subscribe(channel_name)  # type: ignore[misc]
            while True:
                msg = await pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
                if msg is not None and msg["type"] == "message":
                    logger.info("Task %s: cancel signal received", task_id)
                    token.cancel()
                    return
                await asyncio.sleep(0.01)
        finally:
            await pubsub.unsubscribe(channel_name)
            await pubsub.aclose()  # type: ignore[misc]
            await r.aclose()

    def _reconstruct_agent(self, agent_config: dict[str, Any]) -> Any:
        """Reconstruct an Agent or Swarm from the serialized config dict.

        .. warning:: **Tool-serialization gap — the most common remote-worker footgun.**

            ``Agent.to_dict()`` / ``Swarm.from_dict()`` serialize *configuration*
            (name, model, instructions, memory settings, etc.) but do **NOT**
            serialize ``@tool``-decorated Python functions.  When an agent is
            reconstructed here, any tools that were attached as Python callables
            on the submitting side will be silently absent on the worker side.

            **Required action:** the worker process must independently register
            every tool the agent needs before it receives tasks.  The canonical
            pattern is to pass a pre-built ``Agent`` subclass or factory to the
            worker, or to ensure the same tool-registration code runs on both
            the submitting process and the worker process.

            There is intentionally no automatic fix here — serializing arbitrary
            callables across process boundaries is unsafe.  This note exists so
            that the problem is visible at the call site rather than discovered
            as silent no-ops at runtime.
        """
        if "agents" in agent_config:
            # Swarm config
            from exo.swarm import Swarm  # pyright: ignore[reportMissingImports]

            return Swarm.from_dict(agent_config)
        else:
            from exo.agent import Agent  # pyright: ignore[reportMissingImports]

            return Agent.from_dict(agent_config)

    async def on_task_done(
        self,
        task: TaskPayload,
        status: TaskStatus,
        result: str | None,
        error: str | None,
    ) -> None:
        """Override in subclasses for post-task cleanup. Default is a no-op."""

    async def _run_agent(self, agent: Any, task: TaskPayload, token: CancellationToken) -> str:
        """Stream agent execution, publishing events and collecting output.

        Checks *token* between steps for cooperative cancellation.  When
        cancelled, emits a ``StatusEvent(status='cancelled')`` and stops.
        """
        from exo.runner import run  # pyright: ignore[reportMissingImports]
        from exo.types import StatusEvent, TextEvent  # pyright: ignore[reportMissingImports]

        messages = _deserialize_messages(task.messages) if task.messages else None

        # Feature 6: Load conversation history from memory
        if hasattr(agent, "memory") and agent.memory is not None:
            mem_config = task.metadata.get("memory", {})
            scope = mem_config.get("scope", {})
            try:
                from exo.distributed.memory import (  # pyright: ignore[reportMissingImports]
                    memory_items_to_messages,
                )
                from exo.memory.base import (  # pyright: ignore[reportMissingImports]
                    MemoryMetadata,
                )

                search_meta = MemoryMetadata(
                    user_id=scope.get("user_id"),
                    session_id=scope.get("session_id"),
                    agent_id=scope.get("agent_id"),
                )
                prior_items = await agent.memory.search(metadata=search_meta, limit=100)
                # Persistent backends return newest-first; ensure chronological
                if len(prior_items) > 1 and prior_items[0].created_at > prior_items[-1].created_at:
                    prior_items.reverse()
                # L-10: Exclude the last item if it matches the current user
                # input (saved by hydration). Use index comparison instead of
                # `is` identity, which breaks when items are reconstructed.
                if (
                    prior_items
                    and prior_items[-1].memory_type == "human"
                    and prior_items[-1].content == task.input
                ):
                    prior_items = prior_items[:-1]
                if prior_items:
                    history = memory_items_to_messages(prior_items)
                    messages = history + (messages or [])
                    logger.info(
                        "Task %s: loaded %d prior memory items as conversation history",
                        task.task_id,
                        len(prior_items),
                    )
                else:
                    logger.debug("Task %s: no prior memory items found", task.task_id)
            except Exception:
                logger.warning(
                    "Failed to load conversation history for task %s",
                    task.task_id,
                    exc_info=True,
                )

        # Feature 5: Provider factory
        provider = None
        if self._provider_factory is not None:
            model = getattr(agent, "model", None) or task.model
            if model:
                provider = self._provider_factory(model)
                logger.debug("Task %s: resolved provider for model '%s'", task.task_id, model)

        logger.debug("Task %s: starting agent stream", task.task_id)
        text_parts: list[str] = []

        async for event in run.stream(
            agent,
            task.input,
            messages=messages,
            provider=provider,
            detailed=task.detailed,
        ):
            # Check for cancellation between steps
            if token.cancelled:
                cancelled_event = StatusEvent(
                    status="cancelled",
                    agent_name=getattr(agent, "name", ""),
                    message=f"Task {task.task_id} cancelled",
                )
                await self._publisher.publish(task.task_id, cancelled_event)
                break

            # Publish every event to Redis
            await self._publisher.publish(task.task_id, event)

            # Collect text for final result
            if isinstance(event, TextEvent):
                text_parts.append(event.text)

        return "".join(text_parts)
