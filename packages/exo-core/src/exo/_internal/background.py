"""Background task handler with hot-merge and wake-up-merge patterns.

Provides ``BackgroundTaskHandler`` for managing long-running background
tasks that produce results asynchronously.  Results can be merged into
the running execution (hot-merge) or queued for later retrieval
(wake-up-merge) when the main task has already completed.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from exo._internal.state import RunNodeStatus, RunState
from exo.observability.logging import get_logger  # pyright: ignore[reportMissingImports]
from exo.types import ExoError

_log = get_logger(__name__)

# Cap the rolling partial-output buffer so a chatty background child cannot
# grow a background task's progress unbounded while it streams.
_PARTIAL_OUTPUT_CAP = 4_000


class BackgroundTaskError(ExoError):
    """Raised for background task handler errors."""


class MergeMode(StrEnum):
    """How background results are merged into the main execution."""

    HOT = "hot"
    WAKEUP = "wakeup"


@dataclass
class BackgroundProgress:
    """Live, mutable snapshot of what a background task is doing right now.

    Updated in place by the stream consumer that drives the background
    child, so a parent can poll it at any time (via ``check_subagent``)
    to see the child's current step, latest tool call, partial output,
    and token usage — not just its terminal status.
    """

    step_number: int = 0
    step_status: str = ""
    current_tool: str = ""
    current_tool_args: str = ""
    last_tool_result: str = ""
    last_tool_error: str | None = None
    partial_output: str = ""
    tokens_used: int = 0


class BackgroundTask:
    """Tracks a single background task's lifecycle.

    Args:
        task_id: Unique identifier for this task.
        parent_task_id: The task that spawned this background task.
        payload: Arbitrary data associated with the task.
    """

    def __init__(
        self,
        task_id: str,
        parent_task_id: str,
        *,
        payload: Any = None,
    ) -> None:
        self.task_id = task_id
        self.parent_task_id = parent_task_id
        self.payload = payload
        self.result: Any = None
        self.error: str | None = None
        self.status: RunNodeStatus = RunNodeStatus.INIT
        self.merge_mode: MergeMode | None = None
        self.progress: BackgroundProgress = BackgroundProgress()

    def update_progress(self, **fields: Any) -> None:
        """Update one or more live-progress fields in place.

        Unknown field names are ignored so the stream consumer can pass
        whatever it has without guarding each event type.
        """
        for key, value in fields.items():
            if hasattr(self.progress, key):
                setattr(self.progress, key, value)

    def append_partial(self, text: str) -> None:
        """Append streamed text to the capped partial-output buffer."""
        if not text:
            return
        combined = self.progress.partial_output + text
        if len(combined) > _PARTIAL_OUTPUT_CAP:
            combined = combined[-_PARTIAL_OUTPUT_CAP:]
        self.progress.partial_output = combined

    def start(self) -> None:
        """Mark task as running."""
        self.status = RunNodeStatus.RUNNING

    def complete(self, result: Any) -> None:
        """Mark task as successfully completed with a result."""
        self.status = RunNodeStatus.SUCCESS
        self.result = result

    def fail(self, error: str) -> None:
        """Mark task as failed with an error message."""
        self.status = RunNodeStatus.FAILED
        self.error = error

    @property
    def is_complete(self) -> bool:
        """Whether the task has reached a terminal state."""
        return self.status in (RunNodeStatus.SUCCESS, RunNodeStatus.FAILED)


class PendingQueue:
    """Thread-safe queue for background results awaiting merge.

    Used in the wake-up-merge pattern: when the main task has
    already completed, background results are queued here for
    later retrieval and re-processing.
    """

    def __init__(self) -> None:
        self._items: list[BackgroundTask] = []
        self._event = asyncio.Event()

    def push(self, task: BackgroundTask) -> None:
        """Add a completed task to the pending queue."""
        self._items.append(task)
        self._event.set()

    def pop_all(self) -> list[BackgroundTask]:
        """Remove and return all pending tasks."""
        items = list(self._items)
        self._items.clear()
        self._event.clear()
        return items

    async def wait(self, timeout: float | None = None) -> bool:
        """Wait until at least one item is available.

        Args:
            timeout: Max seconds to wait, or None for no timeout.

        Returns:
            True if items are available, False on timeout.
        """
        try:
            await asyncio.wait_for(self._event.wait(), timeout=timeout)
            return True
        except TimeoutError:
            return False

    @property
    def size(self) -> int:
        """Number of pending items."""
        return len(self._items)

    @property
    def empty(self) -> bool:
        """Whether the queue has no pending items."""
        return len(self._items) == 0


class BackgroundTaskHandler:
    """Manages background tasks with hot-merge and wake-up-merge patterns.

    **Hot-merge**: When the main task is still running and a background
    task completes, the result is merged directly into the active
    execution state.

    **Wake-up-merge**: When the main task has already completed (or
    paused), background results are queued in a ``PendingQueue`` for
    later retrieval.  A checkpoint can be restored and the pending
    results merged in.

    Args:
        state: Optional ``RunState`` for tracking background task nodes.
    """

    def __init__(self, *, state: RunState | None = None) -> None:
        self._tasks: dict[str, BackgroundTask] = {}
        self._pending = PendingQueue()
        self._state = state
        self._merge_callbacks: list[Any] = []

    def submit(
        self,
        task_id: str,
        parent_task_id: str,
        *,
        payload: Any = None,
    ) -> BackgroundTask:
        """Register a new background task.

        Args:
            task_id: Unique identifier for the task.
            parent_task_id: The parent task that spawned this one.
            payload: Arbitrary data for the task.

        Returns:
            The created ``BackgroundTask``.

        Raises:
            BackgroundTaskError: If a task with this ID already exists.
        """
        if task_id in self._tasks:
            raise BackgroundTaskError(f"Background task '{task_id}' already exists")

        task = BackgroundTask(task_id, parent_task_id, payload=payload)
        task.start()
        self._tasks[task_id] = task
        _log.debug("Background task '%s' submitted (parent='%s')", task_id, parent_task_id)

        if self._state is not None:
            node = self._state.new_node(agent_name=f"bg:{task_id}")
            node.start()

        return task

    async def handle_result(
        self,
        task_id: str,
        result: Any,
        *,
        is_main_running: bool = True,
    ) -> MergeMode:
        """Handle a background task's completion.

        Routes to hot-merge or wake-up-merge based on whether the
        main task is still running.

        Args:
            task_id: The background task that completed.
            result: The task's result value.
            is_main_running: Whether the parent task is still active.

        Returns:
            The ``MergeMode`` that was applied.

        Raises:
            BackgroundTaskError: If the task ID is not found.
        """
        task = self._tasks.get(task_id)
        if task is None:
            raise BackgroundTaskError(f"Background task '{task_id}' not found")

        task.complete(result)
        _log.debug(
            "Background task '%s' completed → %s merge",
            task_id,
            "hot" if is_main_running else "wakeup",
        )

        if self._state is not None:
            self._complete_node(task_id)

        if is_main_running:
            task.merge_mode = MergeMode.HOT
            await self._hot_merge(task)
        else:
            task.merge_mode = MergeMode.WAKEUP
            self._pending.push(task)

        return task.merge_mode

    def handle_error(self, task_id: str, error: str) -> None:
        """Record a background task failure.

        Args:
            task_id: The task that failed.
            error: Error description.

        Raises:
            BackgroundTaskError: If the task ID is not found.
        """
        task = self._tasks.get(task_id)
        if task is None:
            raise BackgroundTaskError(f"Background task '{task_id}' not found")

        task.fail(error)
        _log.warning("Background task '%s' failed: %s", task_id, error)

        if self._state is not None:
            for node in reversed(self._state.nodes):
                if node.agent_name == f"bg:{task_id}":
                    node.fail(error)
                    break

    async def drain_pending(self) -> AsyncIterator[BackgroundTask]:
        """Yield all pending background tasks (wake-up-merge pattern).

        Used when restoring from a checkpoint to process any
        background results that arrived while the main task was
        paused.

        Yields:
            Each pending ``BackgroundTask`` with its result.
        """
        for task in self._pending.pop_all():
            yield task

    def on_merge(self, callback: Any) -> None:
        """Register a callback for merge events.

        The callback is called with ``(task, merge_mode)`` whenever
        a background result is merged (hot or wake-up).

        Args:
            callback: An async callable ``(BackgroundTask, MergeMode) -> None``.
        """
        self._merge_callbacks.append(callback)

    def get_task(self, task_id: str) -> BackgroundTask | None:
        """Retrieve a background task by ID."""
        return self._tasks.get(task_id)

    def list_tasks(self, *, status: RunNodeStatus | None = None) -> list[BackgroundTask]:
        """List background tasks, optionally filtered by status.

        Args:
            status: If given, only return tasks with this status.

        Returns:
            List of matching tasks.
        """
        if status is None:
            return list(self._tasks.values())
        return [t for t in self._tasks.values() if t.status == status]

    @property
    def pending_queue(self) -> PendingQueue:
        """Access the pending queue for wake-up-merge tasks."""
        return self._pending

    async def _hot_merge(self, task: BackgroundTask) -> None:
        """Merge a background result directly into the active execution.

        Fires registered merge callbacks.  Individual callback failures
        are logged but do not prevent subsequent callbacks from running.

        Args:
            task: The completed background task.
        """
        for cb in self._merge_callbacks:
            try:
                await cb(task, MergeMode.HOT)
            except Exception as exc:
                _log.warning(
                    "Merge callback failed for background task '%s': %s",
                    task.task_id,
                    exc,
                )

    def _complete_node(self, task_id: str) -> None:
        """Mark the RunNode for a background task as successful."""
        if self._state is None:
            return
        for node in reversed(self._state.nodes):
            if node.agent_name == f"bg:{task_id}":
                node.succeed()
                break
