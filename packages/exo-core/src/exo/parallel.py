"""Programmatic parallel agent dispatch.

Provides :func:`run_parallel` and :func:`stream_parallel` — free functions
for running multiple agents concurrently and collecting their results.

Unlike :class:`~exo._internal.agent_group.ParallelGroup` (which is a static
Swarm node), these functions work with *any* agents at *runtime*, support
per-agent providers and timeouts, and return structured
:class:`SubAgentResult` objects rather than a single concatenated string.

Usage::

    from exo import run_parallel, stream_parallel
    from exo.parallel import SubAgentTask, SubAgentResult, SubAgentStatus, SubAgentError

    tasks = [
        SubAgentTask(agent=agent_a, input="Summarise X"),
        SubAgentTask(agent=agent_b, input="Summarise X", timeout=10.0),
    ]

    # Non-streaming — wait for all agents, return results
    results = await run_parallel(tasks, continue_on_error=True)
    for r in results:
        print(r.agent_name, r.status, r.output)

    # Streaming — multiplexed events from all agents in arrival order
    async for event in stream_parallel(tasks):
        print(event)
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from exo.types import ErrorEvent, ExoError, RunResult, StatusEvent, StreamEvent, TextEvent

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


class SubAgentStatus(StrEnum):
    """Terminal status of a sub-agent execution."""

    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"


@dataclass(frozen=True)
class SubAgentTask:
    """Specification for one parallel sub-agent invocation.

    Args:
        agent: An ``Agent``, ``Swarm``, or any object with a ``run()``
            coroutine and a ``stream()`` async generator.
        input: User query string or content blocks.
        name: Label override for event tagging; defaults to ``agent.name``.
        messages: Prior conversation history.  ``None`` means a fresh
            conversation with no history.
        provider: LLM provider override for this agent.
        timeout: Per-agent timeout in seconds.  ``None`` means no timeout.
    """

    agent: Any
    input: Any  # MessageContent
    name: str | None = None
    messages: Sequence[Any] | None = None
    provider: Any = None
    timeout: float | None = None


@dataclass(frozen=True)
class SubAgentResult:
    """Result from one parallel sub-agent.

    Always populated, even on failure.  On error, ``output`` is empty
    and ``error`` holds the original exception.

    Args:
        agent_name: Name of the agent that produced this result.
        status: Terminal status (success, failed, cancelled, timed_out).
        output: Agent's text output (empty on failure).
        result: Full ``RunResult`` on success, ``None`` on failure.
        error: Original exception on failure, ``None`` on success.
        elapsed_seconds: Wall-clock seconds the agent ran.
    """

    agent_name: str
    status: SubAgentStatus
    output: str = ""
    result: RunResult | None = None
    error: BaseException | None = None
    elapsed_seconds: float = 0.0


class SubAgentError(ExoError):
    """Raised when parallel sub-agent execution fails in fail-fast mode.

    Carries partial results so callers can inspect what completed
    before the failure.

    Attributes:
        results: All results (including failed ones) in task order.
        failed_agents: Names of agents that did not succeed.
    """

    def __init__(
        self,
        message: str,
        *,
        results: list[SubAgentResult],
        failed_agents: list[str],
    ) -> None:
        super().__init__(message)
        self.results = results
        self.failed_agents = failed_agents


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _label(task: SubAgentTask) -> str:
    return task.name or getattr(task.agent, "name", f"agent_{id(task.agent)}")


def _validate_tasks(tasks: list[SubAgentTask]) -> None:
    seen: set[str] = set()
    for task in tasks:
        lbl = _label(task)
        if lbl in seen:
            raise ValueError(f"Duplicate sub-agent name '{lbl}' in parallel tasks")
        seen.add(lbl)


# ---------------------------------------------------------------------------
# Internal sentinel for streaming completion tracking
# ---------------------------------------------------------------------------


@dataclass
class _Done:
    """Queue marker: a single agent has finished."""

    agent_name: str
    result: SubAgentResult
    error: BaseException | None = None


# ---------------------------------------------------------------------------
# run_parallel — non-streaming
# ---------------------------------------------------------------------------


async def run_parallel(
    tasks: list[SubAgentTask],
    *,
    provider: Any = None,
    continue_on_error: bool = False,
    max_concurrency: int | None = None,
) -> list[SubAgentResult]:
    """Run multiple agents concurrently and collect results.

    All agents receive their own input and run as concurrent asyncio tasks.
    Results are returned in the same order as *tasks*.

    Args:
        tasks: Agent tasks to execute in parallel.
        provider: Default LLM provider, used for tasks that don't specify one.
        continue_on_error: When ``False`` (default), raises
            :class:`SubAgentError` on the first agent failure.  When
            ``True``, all agents run to completion and results for failed
            agents have ``status=FAILED`` with the exception in ``error``.
        max_concurrency: Maximum number of agents running at the same time.
            ``None`` means unlimited.

    Returns:
        List of :class:`SubAgentResult` in the same order as *tasks*.

    Raises:
        SubAgentError: When ``continue_on_error=False`` and any agent fails.
    """
    if not tasks:
        return []
    _validate_tasks(tasks)

    from exo.runner import run as _run

    n = len(tasks)
    results: list[SubAgentResult | None] = [None] * n
    cancel_event = asyncio.Event()
    semaphore = asyncio.Semaphore(max_concurrency) if max_concurrency else None

    async def _run_one(idx: int) -> None:
        task = tasks[idx]
        label = _label(task)
        started = time.monotonic()
        resolved = task.provider or provider

        if cancel_event.is_set():
            results[idx] = SubAgentResult(
                agent_name=label,
                status=SubAgentStatus.CANCELLED,
                elapsed_seconds=0.0,
            )
            return

        if semaphore:
            await semaphore.acquire()
        try:
            coro = _run(task.agent, task.input, messages=task.messages, provider=resolved)
            if task.timeout is not None:
                run_result = await asyncio.wait_for(coro, timeout=task.timeout)
            else:
                run_result = await coro

            results[idx] = SubAgentResult(
                agent_name=label,
                status=SubAgentStatus.SUCCESS,
                output=run_result.output,
                result=run_result,
                elapsed_seconds=time.monotonic() - started,
            )
            logger.debug("run_parallel: '%s' completed", label)

        except TimeoutError:
            results[idx] = SubAgentResult(
                agent_name=label,
                status=SubAgentStatus.TIMED_OUT,
                elapsed_seconds=time.monotonic() - started,
            )
            logger.warning("run_parallel: '%s' timed out", label)
            if not continue_on_error:
                cancel_event.set()

        except asyncio.CancelledError:
            results[idx] = SubAgentResult(
                agent_name=label,
                status=SubAgentStatus.CANCELLED,
                elapsed_seconds=time.monotonic() - started,
            )

        except Exception as exc:
            results[idx] = SubAgentResult(
                agent_name=label,
                status=SubAgentStatus.FAILED,
                error=exc,
                elapsed_seconds=time.monotonic() - started,
            )
            logger.warning("run_parallel: '%s' failed: %s", label, exc)
            if not continue_on_error:
                cancel_event.set()

        finally:
            if semaphore:
                semaphore.release()

    agent_tasks = [asyncio.create_task(_run_one(i)) for i in range(n)]
    await asyncio.gather(*agent_tasks, return_exceptions=True)

    # Fill any None slots defensively
    final: list[SubAgentResult] = []
    for i in range(n):
        if results[i] is None:
            label = _label(tasks[i])
            final.append(
                SubAgentResult(
                    agent_name=label,
                    status=SubAgentStatus.FAILED,
                    error=RuntimeError("Agent wrapper exited without setting result"),
                )
            )
        else:
            final.append(results[i])  # type: ignore[arg-type]

    if not continue_on_error:
        non_success = [r for r in final if r.status != SubAgentStatus.SUCCESS]
        if non_success:
            failed_names = [r.agent_name for r in non_success]
            raise SubAgentError(
                f"Parallel sub-agents failed: {', '.join(failed_names)}",
                results=final,
                failed_agents=failed_names,
            )

    return final


# ---------------------------------------------------------------------------
# stream_parallel — streaming with event multiplexing
# ---------------------------------------------------------------------------


async def stream_parallel(
    tasks: list[SubAgentTask],
    *,
    provider: Any = None,
    continue_on_error: bool = False,
    max_concurrency: int | None = None,
    queue_size: int = 256,
) -> AsyncIterator[StreamEvent]:
    """Stream events from multiple agents running concurrently.

    Events from all agents are multiplexed into a single stream in
    arrival order.  Each event's ``agent_name`` field identifies which
    agent produced it.

    Args:
        tasks: Agent tasks to execute in parallel.
        provider: Default LLM provider, used for tasks that don't specify one.
        continue_on_error: When ``False`` (default), cancels all remaining
            agents on the first failure.  When ``True``, other agents
            continue and failed agents produce :class:`~exo.types.ErrorEvent`
            items in the stream.
        max_concurrency: Maximum number of agents running at the same time.
        queue_size: Bounded queue size for backpressure (default 256).

    Yields:
        :class:`~exo.types.StreamEvent` instances from all agents,
        interleaved in arrival order.
    """
    if not tasks:
        return
    _validate_tasks(tasks)

    from exo.runner import run as _run

    n = len(tasks)
    queue: asyncio.Queue[StreamEvent | _Done | None] = asyncio.Queue(maxsize=queue_size)
    cancel_event = asyncio.Event()
    semaphore = asyncio.Semaphore(max_concurrency) if max_concurrency else None

    async def _stream_worker(idx: int) -> None:
        task = tasks[idx]
        label = _label(task)
        started = time.monotonic()
        resolved = task.provider or provider
        text_parts: list[str] = []

        if semaphore:
            await semaphore.acquire()
        try:
            if cancel_event.is_set():
                await queue.put(
                    _Done(
                        agent_name=label,
                        result=SubAgentResult(
                            agent_name=label,
                            status=SubAgentStatus.CANCELLED,
                        ),
                        error=asyncio.CancelledError(),
                    )
                )
                return

            await queue.put(StatusEvent(status="starting", agent_name=label))

            async for event in _run.stream(
                task.agent,
                task.input,
                messages=task.messages,
                provider=resolved,
            ):
                if cancel_event.is_set():
                    break
                if isinstance(event, TextEvent):
                    text_parts.append(event.text)
                await queue.put(event)

            output = "".join(text_parts)
            await queue.put(StatusEvent(status="completed", agent_name=label))
            await queue.put(
                _Done(
                    agent_name=label,
                    result=SubAgentResult(
                        agent_name=label,
                        status=SubAgentStatus.SUCCESS,
                        output=output,
                        elapsed_seconds=time.monotonic() - started,
                    ),
                )
            )
            logger.debug("stream_parallel: '%s' completed", label)

        except TimeoutError:
            output = "".join(text_parts)
            await queue.put(
                ErrorEvent(
                    error=f"Agent '{label}' timed out",
                    error_type="TimeoutError",
                    agent_name=label,
                    recoverable=continue_on_error,
                )
            )
            await queue.put(
                _Done(
                    agent_name=label,
                    result=SubAgentResult(
                        agent_name=label,
                        status=SubAgentStatus.TIMED_OUT,
                        output=output,
                        elapsed_seconds=time.monotonic() - started,
                    ),
                    error=TimeoutError(),
                )
            )
            if not continue_on_error:
                cancel_event.set()

        except asyncio.CancelledError:
            output = "".join(text_parts)
            await queue.put(StatusEvent(status="cancelled", agent_name=label))
            await queue.put(
                _Done(
                    agent_name=label,
                    result=SubAgentResult(
                        agent_name=label,
                        status=SubAgentStatus.CANCELLED,
                        output=output,
                        elapsed_seconds=time.monotonic() - started,
                    ),
                    error=asyncio.CancelledError(),
                )
            )

        except Exception as exc:
            output = "".join(text_parts)
            await queue.put(
                ErrorEvent(
                    error=str(exc),
                    error_type=type(exc).__name__,
                    agent_name=label,
                    recoverable=continue_on_error,
                )
            )
            await queue.put(
                _Done(
                    agent_name=label,
                    result=SubAgentResult(
                        agent_name=label,
                        status=SubAgentStatus.FAILED,
                        error=exc,
                        output=output,
                        elapsed_seconds=time.monotonic() - started,
                    ),
                    error=exc,
                )
            )
            logger.warning("stream_parallel: '%s' failed: %s", label, exc)
            if not continue_on_error:
                cancel_event.set()

        finally:
            if semaphore:
                semaphore.release()

    agent_tasks: list[asyncio.Task[None]] = []

    async def _run_producers() -> None:
        nonlocal agent_tasks
        agent_tasks = [asyncio.create_task(_stream_worker(i)) for i in range(n)]
        await asyncio.gather(*agent_tasks, return_exceptions=True)
        await queue.put(None)  # All-done sentinel

    producer_task = asyncio.create_task(_run_producers())
    try:
        while True:
            item = await queue.get()

            if item is None:
                break  # All producers finished

            if isinstance(item, _Done):
                # Fail-fast: cancel remaining agents on error
                if item.error and not continue_on_error:
                    cancel_event.set()
                    for t in agent_tasks:
                        if not t.done():
                            t.cancel()
                continue

            yield item

    finally:
        if not producer_task.done():
            cancel_event.set()
            for t in agent_tasks:
                if not t.done():
                    t.cancel()
            producer_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await producer_task
