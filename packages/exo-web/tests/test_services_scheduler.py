"""Tests for exo_web.services.scheduler.

Covers:
- compute_next_run: basic croniter integration, after= parameter, format of returned string
- _fire_schedule: workflow-not-found disabling, empty-nodes skip, normal run record creation
- start_scheduler / stop_scheduler: idempotency and task lifecycle
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from exo_web.services.scheduler import (
    compute_next_run,
    start_scheduler,
    stop_scheduler,
)

# ---------------------------------------------------------------------------
# compute_next_run
# ---------------------------------------------------------------------------


class TestComputeNextRun:
    def test_returns_string(self) -> None:
        result = compute_next_run("* * * * *")
        assert isinstance(result, str)

    def test_format_is_datetime_string(self) -> None:
        result = compute_next_run("* * * * *")
        # Must be parseable as a datetime
        parsed = datetime.strptime(result, "%Y-%m-%d %H:%M:%S")
        assert parsed is not None

    def test_next_run_is_in_the_future(self) -> None:
        now = datetime.now(UTC)
        result = compute_next_run("* * * * *")
        next_dt = datetime.strptime(result, "%Y-%m-%d %H:%M:%S")
        # next_dt is naive (UTC stripped), compare without tz
        assert next_dt > now.replace(tzinfo=None)

    def test_after_parameter_respected(self) -> None:
        # A specific fixed base time — every-minute cron fires 1 min later
        base = datetime(2025, 6, 15, 12, 0, 0, tzinfo=UTC)
        result = compute_next_run("* * * * *", after=base)
        next_dt = datetime.strptime(result, "%Y-%m-%d %H:%M:%S")
        assert next_dt >= datetime(2025, 6, 15, 12, 1, 0)

    def test_hourly_cron(self) -> None:
        base = datetime(2025, 1, 1, 10, 30, 0, tzinfo=UTC)
        result = compute_next_run("0 * * * *", after=base)
        next_dt = datetime.strptime(result, "%Y-%m-%d %H:%M:%S")
        # Next hour boundary after 10:30 is 11:00
        assert next_dt.hour == 11
        assert next_dt.minute == 0

    def test_daily_cron(self) -> None:
        base = datetime(2025, 3, 1, 5, 0, 0, tzinfo=UTC)
        result = compute_next_run("0 9 * * *", after=base)
        next_dt = datetime.strptime(result, "%Y-%m-%d %H:%M:%S")
        assert next_dt.hour == 9
        assert next_dt.minute == 0

    def test_weekly_cron_fires_once_per_week(self) -> None:
        # Monday noon
        base = datetime(2025, 1, 6, 12, 0, 0, tzinfo=UTC)  # Monday
        result = compute_next_run("0 12 * * 1", after=base)
        next_dt = datetime.strptime(result, "%Y-%m-%d %H:%M:%S")
        delta = next_dt - base.replace(tzinfo=None)
        # Should fire 7 days later (next Monday noon, since base IS the fire time)
        assert delta.days == 7

    def test_no_after_parameter_uses_current_time(self) -> None:
        before = datetime.now(UTC).replace(tzinfo=None)
        result = compute_next_run("* * * * *")
        next_dt = datetime.strptime(result, "%Y-%m-%d %H:%M:%S")
        # Should be roughly 1 minute from now
        assert next_dt > before
        assert (next_dt - before).total_seconds() < 120


# ---------------------------------------------------------------------------
# Helpers for _fire_schedule tests
# ---------------------------------------------------------------------------


def _make_db_mock(
    workflow_row: dict[str, Any] | None = None,
    *,
    include_workflow: bool = True,
) -> AsyncMock:
    """Build a mock async DB context manager."""
    db = AsyncMock()
    cursor = AsyncMock()

    if include_workflow and workflow_row is not None:
        fake_wf = MagicMock()
        fake_wf.__getitem__ = lambda self, k: workflow_row[k]
        fake_wf.get = lambda k, d=None: workflow_row.get(k, d)
        cursor.fetchone = AsyncMock(return_value=fake_wf)
    else:
        cursor.fetchone = AsyncMock(return_value=None)

    db.execute = AsyncMock(return_value=cursor)
    db.commit = AsyncMock()
    return db


# ---------------------------------------------------------------------------
# _fire_schedule
# ---------------------------------------------------------------------------


class TestFireSchedule:
    async def test_workflow_not_found_disables_schedule(self) -> None:
        """When the workflow row is missing the schedule should be disabled."""
        from exo_web.services.scheduler import _fire_schedule

        schedule = {
            "id": "sched-001",
            "workflow_id": "wf-missing",
            "user_id": "user-001",
            "cron_expression": "* * * * *",
        }

        mock_db = AsyncMock()
        cursor = AsyncMock()
        cursor.fetchone = AsyncMock(return_value=None)
        mock_db.execute = AsyncMock(return_value=cursor)
        mock_db.commit = AsyncMock()

        with patch("exo_web.services.scheduler.get_db") as mock_get_db:
            mock_get_db.return_value.__aenter__ = AsyncMock(return_value=mock_db)
            mock_get_db.return_value.__aexit__ = AsyncMock(return_value=False)

            await _fire_schedule(schedule)

        # Should have issued an UPDATE … SET enabled = 0 … SQL
        calls = [str(call) for call in mock_db.execute.call_args_list]
        assert any("enabled" in c and "0" in c for c in calls)
        mock_db.commit.assert_called_once()

    async def test_empty_nodes_skips_run_creation(self) -> None:
        """Workflow with empty nodes_json skips run record creation and advances schedule."""
        from exo_web.services.scheduler import _fire_schedule

        schedule = {
            "id": "sched-002",
            "workflow_id": "wf-empty",
            "user_id": "user-001",
            "cron_expression": "* * * * *",
        }

        # Simulate a DB where the workflow row has empty nodes_json
        mock_db = AsyncMock()
        wf_mock = MagicMock()
        wf_mock.__getitem__ = lambda self, k: {"nodes_json": "[]", "edges_json": "[]"}[k]
        cursor = AsyncMock()
        cursor.fetchone = AsyncMock(return_value=wf_mock)
        mock_db.execute = AsyncMock(return_value=cursor)
        mock_db.commit = AsyncMock()

        with patch("exo_web.services.scheduler.get_db") as mock_get_db:
            mock_get_db.return_value.__aenter__ = AsyncMock(return_value=mock_db)
            mock_get_db.return_value.__aexit__ = AsyncMock(return_value=False)

            await _fire_schedule(schedule)

        # commit was called (last_run_at / next_run_at updated)
        mock_db.commit.assert_called_once()
        # No INSERT INTO workflow_runs should have been issued
        insert_calls = [
            str(c) for c in mock_db.execute.call_args_list if "INSERT" in str(c).upper()
        ]
        assert insert_calls == []

    async def test_normal_run_creates_workflow_run_record(self) -> None:
        """With real nodes, a workflow_run INSERT and schedule UPDATE are both issued."""
        from exo_web.services.scheduler import _fire_schedule

        schedule = {
            "id": "sched-003",
            "workflow_id": "wf-real",
            "user_id": "user-001",
            "cron_expression": "* * * * *",
        }

        nodes = [{"id": "n1", "type": "agent"}]
        wf_mock = MagicMock()
        wf_mock.__getitem__ = lambda self, k: {
            "nodes_json": json.dumps(nodes),
            "edges_json": "[]",
        }[k]
        cursor = AsyncMock()
        cursor.fetchone = AsyncMock(return_value=wf_mock)
        mock_db = AsyncMock()
        mock_db.execute = AsyncMock(return_value=cursor)
        mock_db.commit = AsyncMock()

        with (
            patch("exo_web.services.scheduler.get_db") as mock_get_db,
            patch("exo_web.services.scheduler.execute_workflow", new_callable=AsyncMock),
        ):
            mock_get_db.return_value.__aenter__ = AsyncMock(return_value=mock_db)
            mock_get_db.return_value.__aexit__ = AsyncMock(return_value=False)

            await _fire_schedule(schedule)

        # An INSERT INTO workflow_runs must have been called
        all_calls = [str(c) for c in mock_db.execute.call_args_list]
        assert any("INSERT" in c.upper() and "workflow_runs" in c for c in all_calls)
        # Schedule must have been updated with new next_run_at
        assert any("UPDATE" in c.upper() and "schedules" in c for c in all_calls)
        mock_db.commit.assert_called_once()


# ---------------------------------------------------------------------------
# start_scheduler / stop_scheduler
# ---------------------------------------------------------------------------


class TestSchedulerLifecycle:
    async def test_start_creates_background_task(self) -> None:
        """start_scheduler launches a background asyncio task."""
        import exo_web.services.scheduler as sched_mod

        # Ensure clean state
        sched_mod._scheduler_task = None

        with patch("exo_web.services.scheduler._poll_loop", new_callable=AsyncMock):
            await start_scheduler()
            assert sched_mod._scheduler_task is not None

        # Clean up
        await stop_scheduler()

    async def test_start_is_idempotent(self) -> None:
        """Calling start_scheduler twice does not create two tasks."""
        import exo_web.services.scheduler as sched_mod

        sched_mod._scheduler_task = None

        with patch("exo_web.services.scheduler._poll_loop", new_callable=AsyncMock):
            await start_scheduler()
            first_task = sched_mod._scheduler_task
            await start_scheduler()  # second call — should be a no-op
            assert sched_mod._scheduler_task is first_task

        await stop_scheduler()

    async def test_stop_clears_task(self) -> None:
        """stop_scheduler cancels the task and sets _scheduler_task to None."""
        import exo_web.services.scheduler as sched_mod

        sched_mod._scheduler_task = None

        with patch("exo_web.services.scheduler._poll_loop", new_callable=AsyncMock):
            await start_scheduler()
            assert sched_mod._scheduler_task is not None

        await stop_scheduler()
        assert sched_mod._scheduler_task is None

    async def test_stop_when_not_started_is_noop(self) -> None:
        """stop_scheduler with no running task does not raise."""
        import exo_web.services.scheduler as sched_mod

        sched_mod._scheduler_task = None
        # Should not raise
        await stop_scheduler()
        assert sched_mod._scheduler_task is None
