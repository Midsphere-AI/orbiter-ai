"""Parallel sub-agent execution for harness orchestration.

Provides :func:`run_parallel` and :func:`stream_parallel` — the
engines behind :meth:`HarnessContext.run_agents_parallel` and
:meth:`HarnessContext.stream_agents_parallel`.

**Output contract for sub-agents:**

1. Each sub-agent writes events to ``/tmp/exo_subagent_{name}_{id}.log``
   so the parent can monitor progress in real time.
2. When a sub-agent completes, its output is appended to the harness
   message list as an ``AssistantMessage``.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from exo.harness.types import (
    SessionState,
    SubAgentResult,
    SubAgentStatus,
    SubAgentTask,
)
from exo.types import AssistantMessage, ErrorEvent, ExoError, StatusEvent, StreamEvent, TextEvent

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from exo.harness.base import HarnessContext

logger = logging.getLogger(__name__)

LOG_DIR = Path("/tmp")


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class SubAgentError(ExoError):
    """Raised when parallel sub-agent execution fails in fail-fast mode.

    Carries partial results so callers can inspect what completed
    before the failure.
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
# State isolation
# ---------------------------------------------------------------------------


class _ForkedSessionState:
    """Write-local, read-through state fork for parallel agents.

    Reads search the local dict first, then fall through to the
    parent snapshot.  Writes only touch the local dict.  This gives
    O(1) fork cost and O(K) memory where K is the number of keys
    written by the agent.
    """

    __slots__ = ("_dirty", "_local", "_parent_snapshot")

    def __init__(self, parent: SessionState) -> None:
        self._parent_snapshot = parent.data
        self._local: dict[str, Any] = {}
        self._dirty = False

    def __getitem__(self, key: str) -> Any:
        if key in self._local:
            return self._local[key]
        return self._parent_snapshot[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self._local[key] = value
        self._dirty = True

    def __contains__(self, key: str) -> bool:
        return key in self._local or key in self._parent_snapshot

    def get(self, key: str, default: Any = None) -> Any:
        if key in self._local:
            return self._local[key]
        return self._parent_snapshot.get(key, default)

    @property
    def dirty(self) -> bool:
        return self._dirty


async def _merge_state(
    parent: SessionState,
    fork: _ForkedSessionState,
    lock: asyncio.Lock,
) -> None:
    """Merge a fork's local writes back into the parent under a lock."""
    if not fork.dirty:
        return
    async with lock:
        parent.data.update(fork._local)
        parent._dirty = True


# ---------------------------------------------------------------------------
# Internal sentinel
# ---------------------------------------------------------------------------


@dataclass
class _Sentinel:
    """Internal queue marker signalling a single agent has finished."""

    agent_name: str
    forked_state: _ForkedSessionState | None = None
    error: BaseException | None = None
    result: SubAgentResult | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_label(task: SubAgentTask) -> str:
    return task.name or getattr(task.agent, "name", f"agent_{id(task.agent)}")


def _validate_tasks(tasks: list[SubAgentTask]) -> None:
    names: set[str] = set()
    for task in tasks:
        label = _resolve_label(task)
        if label in names:
            raise ValueError(f"Duplicate sub-agent name '{label}' in parallel tasks")
        names.add(label)


def _make_log_path(label: str) -> Path:
    """Create a unique log path for a sub-agent."""
    short_id = uuid.uuid4().hex[:8]
    return LOG_DIR / f"exo_subagent_{label}_{short_id}.log"


def _format_event(event: StreamEvent) -> str:
    """Format a StreamEvent as a single log line."""
    if isinstance(event, TextEvent):
        return f"[text] {event.text}"
    if isinstance(event, StatusEvent):
        msg = f"[status] {event.status}"
        if event.message:
            msg += f" — {event.message}"
        return msg
    if isinstance(event, ErrorEvent):
        return f"[error] {event.error_type}: {event.error}"
    # Generic fallback
    return f"[{event.type}] {event!r}"


def _notify_started(ctx: HarnessContext, label: str, log_path: Path) -> None:
    """Tell the parent agent that a sub-agent is now running in the background."""
    ctx.messages.append(
        AssistantMessage(
            content=(
                f"[Sub-agent '{label}' is now executing in the background]\n"
                f"You can monitor its progress at: {log_path}\n"
                f"You will receive its output as a message when it completes."
            )
        )
    )


def _notify_completed(
    ctx: HarnessContext,
    label: str,
    output: str,
    status: SubAgentStatus,
) -> None:
    """Append the sub-agent's final answer to the parent's conversation."""
    if status == SubAgentStatus.SUCCESS:
        content = f"[Sub-agent '{label}' completed]:\n{output}"
    elif status == SubAgentStatus.TIMED_OUT:
        content = f"[Sub-agent '{label}' timed out]"
        if output:
            content += f"\nPartial output:\n{output}"
    elif status == SubAgentStatus.CANCELLED:
        content = f"[Sub-agent '{label}' was cancelled]"
        if output:
            content += f"\nPartial output:\n{output}"
    else:
        content = f"[Sub-agent '{label}' failed]"
        if output:
            content += f"\nPartial output:\n{output}"
    ctx.messages.append(AssistantMessage(content=content))


# ---------------------------------------------------------------------------
# run_parallel — non-streaming
# ---------------------------------------------------------------------------


async def run_parallel(
    ctx: HarnessContext,
    tasks: list[SubAgentTask],
    *,
    continue_on_error: bool = False,
    max_concurrency: int | None = None,
) -> list[SubAgentResult]:
    """Run multiple agents concurrently and collect results.

    Each sub-agent's output is:

    1. Written to a log file at ``/tmp/exo_subagent_{name}_{id}.log``.
    2. Appended to ``ctx.messages`` as an ``AssistantMessage``.

    Args:
        ctx: The harness runtime context.
        tasks: Agent tasks to execute in parallel.
        continue_on_error: When ``False`` (default), raises
            :class:`SubAgentError` on first failure.  When ``True``,
            captures errors per-agent and returns all results.
        max_concurrency: Limit concurrent agents via semaphore.

    Returns:
        Results in the same order as *tasks*.

    Raises:
        SubAgentError: When ``continue_on_error=False`` and any
            agent fails.
    """
    if not tasks:
        return []
    _validate_tasks(tasks)

    n = len(tasks)
    results: list[SubAgentResult | None] = [None] * n
    cancel_event = asyncio.Event()
    merge_lock = asyncio.Lock()
    semaphore = asyncio.Semaphore(max_concurrency) if max_concurrency else None

    async def _agent_wrapper(idx: int) -> None:
        task = tasks[idx]
        label = _resolve_label(task)
        log_path = _make_log_path(label)
        started = time.monotonic()

        if cancel_event.is_set():
            results[idx] = SubAgentResult(
                agent_name=label, status=SubAgentStatus.CANCELLED, log_path=str(log_path)
            )
            return

        forked = _ForkedSessionState(ctx.state)

        if semaphore:
            await semaphore.acquire()
        try:
            with open(log_path, "w") as f:
                ts = time.strftime("%H:%M:%S")
                f.write(f"[{ts}] [starting] Sub-agent '{label}' starting\n")

            # Tell the parent agent this sub-agent is running
            _notify_started(ctx, label, log_path)

            coro = ctx.run_agent(
                task.agent,
                task.input,
                messages=task.messages,
                provider=task.provider,
            )
            if task.timeout is not None:
                run_result = await asyncio.wait_for(coro, timeout=task.timeout)
            else:
                run_result = await coro

            # Log the result
            with open(log_path, "a") as f:
                ts = time.strftime("%H:%M:%S")
                f.write(f"[{ts}] [completed] output={run_result.output[:200]!r}\n")

            await _merge_state(ctx.state, forked, merge_lock)

            # Deliver the sub-agent's answer to the parent
            _notify_completed(ctx, label, run_result.output, SubAgentStatus.SUCCESS)

            results[idx] = SubAgentResult(
                agent_name=label,
                status=SubAgentStatus.SUCCESS,
                output=run_result.output,
                result=run_result,
                elapsed_seconds=time.monotonic() - started,
                log_path=str(log_path),
            )
            logger.debug("Sub-agent '%s' completed successfully", label)

        except TimeoutError:
            with open(log_path, "a") as f:
                ts = time.strftime("%H:%M:%S")
                f.write(f"[{ts}] [timed_out] after {task.timeout}s\n")

            _notify_completed(ctx, label, "", SubAgentStatus.TIMED_OUT)

            results[idx] = SubAgentResult(
                agent_name=label,
                status=SubAgentStatus.TIMED_OUT,
                elapsed_seconds=time.monotonic() - started,
                log_path=str(log_path),
            )
            logger.warning("Sub-agent '%s' timed out", label)
            if not continue_on_error:
                cancel_event.set()

        except asyncio.CancelledError:
            results[idx] = SubAgentResult(
                agent_name=label,
                status=SubAgentStatus.CANCELLED,
                elapsed_seconds=time.monotonic() - started,
                log_path=str(log_path),
            )

        except Exception as exc:
            with open(log_path, "a") as f:
                ts = time.strftime("%H:%M:%S")
                f.write(f"[{ts}] [failed] {type(exc).__name__}: {exc}\n")

            _notify_completed(ctx, label, "", SubAgentStatus.FAILED)

            results[idx] = SubAgentResult(
                agent_name=label,
                status=SubAgentStatus.FAILED,
                error=exc,
                elapsed_seconds=time.monotonic() - started,
                log_path=str(log_path),
            )
            logger.warning("Sub-agent '%s' failed: %s", label, exc)
            if not continue_on_error:
                cancel_event.set()

        finally:
            if semaphore:
                semaphore.release()

    agent_tasks = [asyncio.create_task(_agent_wrapper(i)) for i in range(n)]
    await asyncio.gather(*agent_tasks, return_exceptions=True)

    # Fill any None slots defensively
    final: list[SubAgentResult] = []
    for i in range(n):
        if results[i] is None:
            label = _resolve_label(tasks[i])
            final.append(
                SubAgentResult(
                    agent_name=label,
                    status=SubAgentStatus.FAILED,
                    error=RuntimeError("Agent wrapper exited without setting result"),
                )
            )
        else:
            final.append(results[i])  # type: ignore[arg-type]

    # In fail-fast mode, raise if any agent failed
    if not continue_on_error:
        failed = [r for r in final if r.status not in (SubAgentStatus.SUCCESS,)]
        if failed:
            failed_names = [r.agent_name for r in failed]
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
    ctx: HarnessContext,
    tasks: list[SubAgentTask],
    *,
    continue_on_error: bool = False,
    max_concurrency: int | None = None,
    queue_size: int = 256,
) -> AsyncIterator[StreamEvent]:
    """Stream events from multiple agents running concurrently.

    Each sub-agent's events are:

    1. Written to a log file at ``/tmp/exo_subagent_{name}_{id}.log``
       in real time, so the parent can ``cat`` the file mid-execution.
    2. Appended to ``ctx.messages`` as an ``AssistantMessage`` when the
       sub-agent completes.

    Args:
        ctx: The harness runtime context.
        tasks: Agent tasks to execute in parallel.
        continue_on_error: When ``False``, cancels all agents on
            first error.  When ``True``, other agents continue.
        max_concurrency: Limit concurrent agents via semaphore.
        queue_size: Bounded queue size for backpressure.

    Yields:
        ``StreamEvent`` instances from all agents, interleaved by
        arrival order.
    """
    if not tasks:
        return
    _validate_tasks(tasks)

    n = len(tasks)
    queue: asyncio.Queue[StreamEvent | _Sentinel | None] = asyncio.Queue(maxsize=queue_size)
    cancel_event = asyncio.Event()
    merge_lock = asyncio.Lock()
    semaphore = asyncio.Semaphore(max_concurrency) if max_concurrency else None
    cancel_agents: dict[str, asyncio.Event] = {}

    # Expose cancel_agents on context for per-agent cancellation
    ctx._cancel_agents = cancel_agents  # type: ignore[attr-defined]

    async def _stream_worker(idx: int) -> None:
        task = tasks[idx]
        label = _resolve_label(task)
        log_path = _make_log_path(label)
        agent_cancel = asyncio.Event()
        cancel_agents[label] = agent_cancel
        forked = _ForkedSessionState(ctx.state)
        text_parts: list[str] = []
        started = time.monotonic()

        # Open log file for this sub-agent
        log_file = open(log_path, "w")  # noqa: SIM115
        try:
            if semaphore:
                await semaphore.acquire()
            try:
                if cancel_event.is_set():
                    await queue.put(
                        _Sentinel(
                            agent_name=label,
                            error=asyncio.CancelledError(),
                            result=SubAgentResult(
                                agent_name=label,
                                status=SubAgentStatus.CANCELLED,
                                log_path=str(log_path),
                            ),
                        )
                    )
                    return

                ts = time.strftime("%H:%M:%S")
                log_file.write(f"[{ts}] [starting] Sub-agent '{label}' starting\n")
                log_file.flush()

                # Tell the parent agent this sub-agent is running
                _notify_started(ctx, label, log_path)

                await queue.put(StatusEvent(status="starting", agent_name=label))

                async for event in ctx.stream_agent(
                    task.agent,
                    task.input,
                    messages=task.messages,
                    provider=task.provider,
                ):
                    if cancel_event.is_set() or agent_cancel.is_set():
                        break
                    if ctx.cancelled:
                        break

                    # Accumulate text
                    if isinstance(event, TextEvent):
                        text_parts.append(event.text)

                    # Write to log file
                    ts = time.strftime("%H:%M:%S")
                    log_file.write(f"[{ts}] {_format_event(event)}\n")
                    log_file.flush()

                    await queue.put(event)

                output = "".join(text_parts)

                ts = time.strftime("%H:%M:%S")
                log_file.write(f"[{ts}] [completed] output={output[:200]!r}\n")
                log_file.flush()

                # Append AssistantMessage to parent's message list
                _notify_completed(ctx, label, output, SubAgentStatus.SUCCESS)

                await queue.put(StatusEvent(status="completed", agent_name=label))
                await queue.put(
                    _Sentinel(
                        agent_name=label,
                        forked_state=forked,
                        result=SubAgentResult(
                            agent_name=label,
                            status=SubAgentStatus.SUCCESS,
                            output=output,
                            elapsed_seconds=time.monotonic() - started,
                            log_path=str(log_path),
                        ),
                    )
                )
                logger.debug("Sub-agent stream '%s' completed", label)

            except TimeoutError:
                output = "".join(text_parts)
                ts = time.strftime("%H:%M:%S")
                log_file.write(f"[{ts}] [timed_out] partial_output={output[:200]!r}\n")
                log_file.flush()

                _notify_completed(ctx, label, output, SubAgentStatus.TIMED_OUT)

                await queue.put(
                    ErrorEvent(
                        error=f"Agent '{label}' timed out",
                        error_type="TimeoutError",
                        agent_name=label,
                        recoverable=continue_on_error,
                    )
                )
                await queue.put(
                    _Sentinel(
                        agent_name=label,
                        error=TimeoutError(),
                        result=SubAgentResult(
                            agent_name=label,
                            status=SubAgentStatus.TIMED_OUT,
                            output=output,
                            elapsed_seconds=time.monotonic() - started,
                            log_path=str(log_path),
                        ),
                    )
                )
                if not continue_on_error:
                    cancel_event.set()

            except asyncio.CancelledError:
                output = "".join(text_parts)
                ts = time.strftime("%H:%M:%S")
                log_file.write(f"[{ts}] [cancelled]\n")
                log_file.flush()

                _notify_completed(ctx, label, output, SubAgentStatus.CANCELLED)

                await queue.put(StatusEvent(status="cancelled", agent_name=label))
                await queue.put(
                    _Sentinel(
                        agent_name=label,
                        error=asyncio.CancelledError(),
                        result=SubAgentResult(
                            agent_name=label,
                            status=SubAgentStatus.CANCELLED,
                            output=output,
                            elapsed_seconds=time.monotonic() - started,
                            log_path=str(log_path),
                        ),
                    )
                )

            except Exception as exc:
                output = "".join(text_parts)
                ts = time.strftime("%H:%M:%S")
                log_file.write(f"[{ts}] [failed] {type(exc).__name__}: {exc}\n")
                log_file.flush()

                _notify_completed(ctx, label, output, SubAgentStatus.FAILED)

                await queue.put(
                    ErrorEvent(
                        error=str(exc),
                        error_type=type(exc).__name__,
                        agent_name=label,
                        recoverable=continue_on_error,
                    )
                )
                await queue.put(
                    _Sentinel(
                        agent_name=label,
                        error=exc,
                        result=SubAgentResult(
                            agent_name=label,
                            status=SubAgentStatus.FAILED,
                            error=exc,
                            output=output,
                            elapsed_seconds=time.monotonic() - started,
                            log_path=str(log_path),
                        ),
                    )
                )
                logger.warning("Sub-agent stream '%s' failed: %s", label, exc)
                if not continue_on_error:
                    cancel_event.set()

            finally:
                if semaphore:
                    semaphore.release()
        finally:
            log_file.close()

    # Track individual tasks for cancellation
    agent_tasks: list[asyncio.Task[None]] = []

    async def _run_producers() -> None:
        nonlocal agent_tasks
        agent_tasks = [asyncio.create_task(_stream_worker(i)) for i in range(n)]
        await asyncio.gather(*agent_tasks, return_exceptions=True)
        await queue.put(None)  # Final "all done" sentinel

    producer_task = asyncio.create_task(_run_producers())

    try:
        while True:
            item = await queue.get()

            # All producers finished
            if item is None:
                break

            # Per-agent completion sentinel
            if isinstance(item, _Sentinel):
                # Merge state on success
                if item.forked_state:
                    await _merge_state(ctx.state, item.forked_state, merge_lock)
                # Fail-fast: cancel remaining
                if item.error and not continue_on_error:
                    cancel_event.set()
                    for t in agent_tasks:
                        if not t.done():
                            t.cancel()
                continue

            # Yield the stream event to the consumer
            yield item

    finally:
        # Structured cleanup: cancel producers if consumer stops
        if not producer_task.done():
            cancel_event.set()
            for t in agent_tasks:
                if not t.done():
                    t.cancel()
            producer_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await producer_task
