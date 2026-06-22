"""Distributed task metrics recording helpers.

Records metrics for task submission, completion, failure, and cancellation
using the ``exo.observability.metrics`` infrastructure (both OTel and
in-memory fallback).
"""

from __future__ import annotations

from typing import Any

from exo.distributed._semconv import (  # pyright: ignore[reportMissingImports]
    DIST_QUEUE_NAME,
    DIST_TASK_ID,
    DIST_TASK_STATUS,
    DIST_WORKER_ID,
    METRIC_DIST_QUEUE_DEPTH,
    METRIC_DIST_TASK_DURATION,
    METRIC_DIST_TASK_WAIT_TIME,
    METRIC_DIST_TASKS_CANCELLED,
    METRIC_DIST_TASKS_COMPLETED,
    METRIC_DIST_TASKS_FAILED,
    METRIC_DIST_TASKS_SUBMITTED,
)
from exo.observability.metrics import (  # pyright: ignore[reportMissingImports]
    HAS_OTEL,
    _collector,
    _get_meter,
)

# ---------------------------------------------------------------------------
# Instrument cache — OTel instruments are safe to create once and reuse.
# Calling create_counter/create_histogram on every metric emission is
# wasteful (the meter caches by name, but it still acquires a lock).
# We pre-create (lazily on first use) and cache here.
# ---------------------------------------------------------------------------
_counter_cache: dict[str, Any] = {}
_histogram_cache: dict[str, Any] = {}
_gauge_cache: dict[str, Any] = {}


def _get_counter(name: str, *, description: str = "", unit: str = "1") -> Any:
    """Return (or lazily create and cache) an OTel counter instrument."""
    if name not in _counter_cache:
        _counter_cache[name] = _get_meter().create_counter(
            name=name, unit=unit, description=description
        )
    return _counter_cache[name]


def _get_histogram(name: str, *, description: str = "", unit: str = "s") -> Any:
    """Return (or lazily create and cache) an OTel histogram instrument."""
    if name not in _histogram_cache:
        _histogram_cache[name] = _get_meter().create_histogram(
            name=name, unit=unit, description=description
        )
    return _histogram_cache[name]


def _get_gauge(name: str, *, description: str = "", unit: str = "1") -> Any:
    """Return (or lazily create and cache) an OTel up-down-counter instrument."""
    if name not in _gauge_cache:
        _gauge_cache[name] = _get_meter().create_up_down_counter(
            name=name, unit=unit, description=description
        )
    return _gauge_cache[name]


def _build_attributes(
    *,
    task_id: str = "",
    worker_id: str = "",
    queue_name: str = "",
    status: str = "",
) -> dict[str, str]:
    """Build attribute dict for distributed task metrics."""
    attrs: dict[str, str] = {}
    if task_id:
        attrs[DIST_TASK_ID] = task_id
    if worker_id:
        attrs[DIST_WORKER_ID] = worker_id
    if queue_name:
        attrs[DIST_QUEUE_NAME] = queue_name
    if status:
        attrs[DIST_TASK_STATUS] = status
    return attrs


def increment_counter(
    name: str, attrs: dict[str, Any], *, description: str = "", unit: str = "1"
) -> None:
    """Increment a counter metric, routing to OTel or the in-memory collector."""
    if HAS_OTEL:
        _get_counter(name, description=description, unit=unit).add(1, attrs)
    else:
        _collector.add_counter(name, 1.0, attrs)


def record_histogram_value(
    name: str, value: float, attrs: dict[str, Any], *, description: str = "", unit: str = "s"
) -> None:
    """Record a histogram observation, routing to OTel or the in-memory collector."""
    if HAS_OTEL:
        _get_histogram(name, description=description, unit=unit).record(value, attrs)
    else:
        _collector.record_histogram(name, value, attrs)


def record_task_submitted(
    *,
    task_id: str = "",
    queue_name: str = "",
) -> None:
    """Record that a task was submitted to the distributed queue."""
    attrs = _build_attributes(task_id=task_id, queue_name=queue_name)
    increment_counter(
        METRIC_DIST_TASKS_SUBMITTED,
        attrs,
        description="Number of distributed tasks submitted",
    )


def record_task_completed(
    *,
    task_id: str = "",
    worker_id: str = "",
    duration: float = 0.0,
    wait_time: float = 0.0,
) -> None:
    """Record that a task completed successfully.

    Args:
        task_id: The task identifier.
        worker_id: The worker that executed the task.
        duration: Total execution duration in seconds.
        wait_time: Time the task waited in the queue in seconds.
    """
    attrs = _build_attributes(task_id=task_id, worker_id=worker_id)
    increment_counter(
        METRIC_DIST_TASKS_COMPLETED,
        attrs,
        description="Number of distributed tasks completed",
    )
    if duration > 0:
        record_histogram_value(
            METRIC_DIST_TASK_DURATION,
            duration,
            attrs,
            description="Distributed task execution duration",
        )
    if wait_time > 0:
        record_histogram_value(
            METRIC_DIST_TASK_WAIT_TIME,
            wait_time,
            attrs,
            description="Time task waited in queue before execution",
        )


def record_task_failed(
    *,
    task_id: str = "",
    worker_id: str = "",
    duration: float = 0.0,
) -> None:
    """Record that a task failed.

    Args:
        task_id: The task identifier.
        worker_id: The worker that attempted the task.
        duration: Execution duration before failure in seconds.
    """
    attrs = _build_attributes(task_id=task_id, worker_id=worker_id)
    increment_counter(
        METRIC_DIST_TASKS_FAILED,
        attrs,
        description="Number of distributed tasks failed",
    )
    if duration > 0:
        record_histogram_value(
            METRIC_DIST_TASK_DURATION,
            duration,
            attrs,
            description="Distributed task execution duration",
        )


def record_task_cancelled(
    *,
    task_id: str = "",
    worker_id: str = "",
) -> None:
    """Record that a task was cancelled."""
    attrs = _build_attributes(task_id=task_id, worker_id=worker_id)
    increment_counter(
        METRIC_DIST_TASKS_CANCELLED,
        attrs,
        description="Number of distributed tasks cancelled",
    )


def record_queue_depth(
    *,
    depth: int,
    queue_name: str = "",
) -> None:
    """Record the current queue depth."""
    attrs: dict[str, Any] = _build_attributes(queue_name=queue_name)
    if HAS_OTEL:
        _get_gauge(
            METRIC_DIST_QUEUE_DEPTH,
            description="Current distributed task queue depth",
            unit="1",
        ).add(depth, attrs)
    else:
        _collector.set_gauge(METRIC_DIST_QUEUE_DEPTH, float(depth), attrs)
