"""Tests for distributed task metrics recording."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from exo.distributed._semconv import (  # pyright: ignore[reportMissingImports]
    DIST_QUEUE_NAME,
    DIST_TASK_ID,
    DIST_WORKER_ID,
    METRIC_DIST_QUEUE_DEPTH,
    METRIC_DIST_TASK_DURATION,
    METRIC_DIST_TASK_WAIT_TIME,
    METRIC_DIST_TASKS_CANCELLED,
    METRIC_DIST_TASKS_COMPLETED,
    METRIC_DIST_TASKS_FAILED,
    METRIC_DIST_TASKS_SUBMITTED,
)
from exo.distributed.metrics import (  # pyright: ignore[reportMissingImports]
    _build_attributes,
    record_queue_depth,
    record_task_cancelled,
    record_task_completed,
    record_task_failed,
    record_task_submitted,
)
from exo.observability.metrics import (  # pyright: ignore[reportMissingImports]
    get_metrics_snapshot,
    reset_metrics,
)

# Patch HAS_OTEL to False so all recording helpers use the in-memory collector.
_NO_OTEL = patch("exo.distributed.metrics.HAS_OTEL", False)


@pytest.fixture(autouse=True)
def _reset() -> None:
    """Reset in-memory metrics and force in-memory path for each test."""
    reset_metrics()
    with _NO_OTEL:
        yield  # type: ignore[misc]


# ---------------------------------------------------------------------------
# _build_attributes
# ---------------------------------------------------------------------------


class TestBuildAttributes:
    def test_all_fields(self) -> None:
        attrs = _build_attributes(task_id="t1", worker_id="w1", queue_name="q1", status="running")
        assert attrs[DIST_TASK_ID] == "t1"
        assert attrs[DIST_WORKER_ID] == "w1"
        assert attrs[DIST_QUEUE_NAME] == "q1"
        assert attrs["exo.distributed.task_status"] == "running"

    def test_empty_fields_omitted(self) -> None:
        attrs = _build_attributes(task_id="t1")
        assert DIST_TASK_ID in attrs
        assert DIST_WORKER_ID not in attrs
        assert DIST_QUEUE_NAME not in attrs

    def test_no_fields(self) -> None:
        attrs = _build_attributes()
        assert attrs == {}


# ---------------------------------------------------------------------------
# record_task_submitted
# ---------------------------------------------------------------------------


class TestRecordTaskSubmitted:
    def test_increments_counter(self) -> None:
        record_task_submitted(task_id="t1", queue_name="exo:tasks")
        snap = get_metrics_snapshot()
        assert snap["counters"][METRIC_DIST_TASKS_SUBMITTED] == 1.0

    def test_multiple_submissions(self) -> None:
        record_task_submitted(task_id="t1")
        record_task_submitted(task_id="t2")
        record_task_submitted(task_id="t3")
        snap = get_metrics_snapshot()
        assert snap["counters"][METRIC_DIST_TASKS_SUBMITTED] == 3.0


# ---------------------------------------------------------------------------
# record_task_completed
# ---------------------------------------------------------------------------


class TestRecordTaskCompleted:
    def test_increments_counter(self) -> None:
        record_task_completed(task_id="t1", worker_id="w1")
        snap = get_metrics_snapshot()
        assert snap["counters"][METRIC_DIST_TASKS_COMPLETED] == 1.0

    def test_records_duration(self) -> None:
        record_task_completed(task_id="t1", worker_id="w1", duration=2.5)
        snap = get_metrics_snapshot()
        durations = snap["histograms"][METRIC_DIST_TASK_DURATION]
        assert len(durations) == 1
        assert durations[0]["value"] == 2.5

    def test_records_wait_time(self) -> None:
        record_task_completed(task_id="t1", worker_id="w1", wait_time=1.0)
        snap = get_metrics_snapshot()
        waits = snap["histograms"][METRIC_DIST_TASK_WAIT_TIME]
        assert len(waits) == 1
        assert waits[0]["value"] == 1.0

    def test_skips_zero_duration(self) -> None:
        record_task_completed(task_id="t1", worker_id="w1", duration=0.0)
        snap = get_metrics_snapshot()
        assert METRIC_DIST_TASK_DURATION not in snap["histograms"]

    def test_attributes_included(self) -> None:
        record_task_completed(task_id="t1", worker_id="w1", duration=1.0)
        snap = get_metrics_snapshot()
        durations = snap["histograms"][METRIC_DIST_TASK_DURATION]
        assert durations[0]["attributes"][DIST_TASK_ID] == "t1"
        assert durations[0]["attributes"][DIST_WORKER_ID] == "w1"


# ---------------------------------------------------------------------------
# record_task_failed
# ---------------------------------------------------------------------------


class TestRecordTaskFailed:
    def test_increments_counter(self) -> None:
        record_task_failed(task_id="t1", worker_id="w1")
        snap = get_metrics_snapshot()
        assert snap["counters"][METRIC_DIST_TASKS_FAILED] == 1.0

    def test_records_duration(self) -> None:
        record_task_failed(task_id="t1", worker_id="w1", duration=3.0)
        snap = get_metrics_snapshot()
        durations = snap["histograms"][METRIC_DIST_TASK_DURATION]
        assert len(durations) == 1
        assert durations[0]["value"] == 3.0

    def test_skips_zero_duration(self) -> None:
        record_task_failed(task_id="t1", duration=0.0)
        snap = get_metrics_snapshot()
        assert METRIC_DIST_TASK_DURATION not in snap["histograms"]


# ---------------------------------------------------------------------------
# record_task_cancelled
# ---------------------------------------------------------------------------


class TestRecordTaskCancelled:
    def test_increments_counter(self) -> None:
        record_task_cancelled(task_id="t1", worker_id="w1")
        snap = get_metrics_snapshot()
        assert snap["counters"][METRIC_DIST_TASKS_CANCELLED] == 1.0

    def test_attributes(self) -> None:
        record_task_cancelled(task_id="t1")
        record_task_cancelled(worker_id="w1")
        record_task_cancelled()
        snap = get_metrics_snapshot()
        assert snap["counters"][METRIC_DIST_TASKS_CANCELLED] == 3.0


# ---------------------------------------------------------------------------
# record_queue_depth
# ---------------------------------------------------------------------------


class TestRecordQueueDepth:
    def test_sets_gauge(self) -> None:
        record_queue_depth(depth=42, queue_name="exo:tasks")
        snap = get_metrics_snapshot()
        assert snap["gauges"][METRIC_DIST_QUEUE_DEPTH] == 42.0

    def test_gauge_updates(self) -> None:
        record_queue_depth(depth=10)
        record_queue_depth(depth=5)
        snap = get_metrics_snapshot()
        assert snap["gauges"][METRIC_DIST_QUEUE_DEPTH] == 5.0


# ---------------------------------------------------------------------------
# Worker integration — metrics recorded during task lifecycle
# ---------------------------------------------------------------------------


class TestWorkerMetricsIntegration:
    @pytest.mark.asyncio
    async def test_successful_task_records_completed_metric(self) -> None:
        from unittest.mock import AsyncMock, MagicMock
        from unittest.mock import patch as mock_patch

        from exo.distributed.cancel import (
            CancellationToken,  # pyright: ignore[reportMissingImports]
        )
        from exo.distributed.models import TaskPayload  # pyright: ignore[reportMissingImports]
        from exo.distributed.worker import Worker  # pyright: ignore[reportMissingImports]

        w = Worker("redis://localhost", worker_id="w1")
        w._broker = AsyncMock()
        w._store = AsyncMock()
        w._publisher = AsyncMock()

        task = TaskPayload(
            task_id="task-m1",
            agent_config={"name": "agent", "model": "openai:gpt-4o"},
            input="hello",
            detailed=False,
        )

        async def _fake_run_agent(agent: object, t: TaskPayload, token: CancellationToken) -> str:
            return "done"

        mock_agent = MagicMock()
        with (
            mock_patch.object(w, "_reconstruct_agent", return_value=mock_agent),
            mock_patch.object(w, "_run_agent", side_effect=_fake_run_agent),
            mock_patch.object(w, "_listen_for_cancel", new_callable=AsyncMock),
        ):
            await w._execute_task(task)

        snap = get_metrics_snapshot()
        assert snap["counters"][METRIC_DIST_TASKS_COMPLETED] == 1.0

    @pytest.mark.asyncio
    async def test_failed_task_records_failed_metric(self) -> None:
        from unittest.mock import AsyncMock
        from unittest.mock import patch as mock_patch

        from exo.distributed.models import (  # pyright: ignore[reportMissingImports]
            TaskPayload,
            TaskResult,
            TaskStatus,
        )
        from exo.distributed.worker import Worker  # pyright: ignore[reportMissingImports]

        w = Worker("redis://localhost", worker_id="w1")
        w._broker = AsyncMock()
        w._broker.max_retries = 3
        w._store = AsyncMock()
        w._store.get_status.return_value = TaskResult(
            task_id="task-m2", status=TaskStatus.RUNNING, retries=3
        )
        w._publisher = AsyncMock()

        task = TaskPayload(
            task_id="task-m2",
            agent_config={"name": "agent", "model": "openai:gpt-4o"},
            input="fail",
        )

        with (
            mock_patch.object(w, "_reconstruct_agent", side_effect=ValueError("bad")),
            mock_patch.object(w, "_listen_for_cancel", new_callable=AsyncMock),
        ):
            await w._execute_task(task)

        snap = get_metrics_snapshot()
        assert snap["counters"][METRIC_DIST_TASKS_FAILED] == 1.0

    @pytest.mark.asyncio
    async def test_cancelled_task_records_cancelled_metric(self) -> None:
        from unittest.mock import AsyncMock, MagicMock
        from unittest.mock import patch as mock_patch

        from exo.distributed.cancel import (
            CancellationToken,  # pyright: ignore[reportMissingImports]
        )
        from exo.distributed.models import TaskPayload  # pyright: ignore[reportMissingImports]
        from exo.distributed.worker import Worker  # pyright: ignore[reportMissingImports]

        w = Worker("redis://localhost", worker_id="w1")
        w._broker = AsyncMock()
        w._store = AsyncMock()
        w._publisher = AsyncMock()

        task = TaskPayload(
            task_id="task-m3",
            agent_config={"name": "agent", "model": "openai:gpt-4o"},
            input="hello",
        )

        async def _fake_run_cancelled(
            agent: object, t: TaskPayload, token: CancellationToken
        ) -> str:
            token.cancel()
            return ""

        mock_agent = MagicMock()
        with (
            mock_patch.object(w, "_reconstruct_agent", return_value=mock_agent),
            mock_patch.object(w, "_run_agent", side_effect=_fake_run_cancelled),
            mock_patch.object(w, "_listen_for_cancel", new_callable=AsyncMock),
        ):
            await w._execute_task(task)

        snap = get_metrics_snapshot()
        assert snap["counters"][METRIC_DIST_TASKS_CANCELLED] == 1.0
