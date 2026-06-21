"""Tests for exo.distributed.alerts — distributed alert rules."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from exo.distributed._alerts import (  # pyright: ignore[reportMissingImports]
    AlertRule,
    AlertSeverity,
    get_manager,
    reset,
)
from exo.distributed._semconv import (  # pyright: ignore[reportMissingImports]
    METRIC_DIST_QUEUE_DEPTH,
    METRIC_DIST_TASK_WAIT_TIME,
    METRIC_DIST_TASKS_FAILED,
)
from exo.distributed.alerts import (  # pyright: ignore[reportMissingImports]
    METRIC_WORKER_COUNT,
    RULE_FAILURE_RATE_WARNING,
    RULE_QUEUE_DEPTH_CRITICAL,
    RULE_QUEUE_DEPTH_WARNING,
    RULE_WAIT_TIME_WARNING,
    RULE_WORKER_COUNT_CRITICAL,
    register_distributed_alerts,
)


@pytest.fixture(autouse=True)
def _clean_alerts():
    """Reset global alert manager before and after each test."""
    reset()
    yield
    reset()


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


class TestRegisterDistributedAlerts:
    def test_returns_five_rules(self) -> None:
        rules = register_distributed_alerts()
        assert len(rules) == 5

    def test_rules_registered_with_global_manager(self) -> None:
        register_distributed_alerts()
        manager = get_manager()
        registered = manager.list_rules()
        assert len(registered) == 5

    def test_rule_names(self) -> None:
        rules = register_distributed_alerts()
        names = {r.name for r in rules}
        assert names == {
            RULE_QUEUE_DEPTH_WARNING,
            RULE_QUEUE_DEPTH_CRITICAL,
            RULE_FAILURE_RATE_WARNING,
            RULE_WORKER_COUNT_CRITICAL,
            RULE_WAIT_TIME_WARNING,
        }

    def test_idempotent_registration(self) -> None:
        """Calling twice doesn't duplicate rules (overwrite by name)."""
        register_distributed_alerts()
        register_distributed_alerts()
        manager = get_manager()
        assert len(manager.list_rules()) == 5

    def test_all_rules_are_alert_rule_instances(self) -> None:
        rules = register_distributed_alerts()
        for rule in rules:
            assert isinstance(rule, AlertRule)


# ---------------------------------------------------------------------------
# Queue depth rules
# ---------------------------------------------------------------------------


class TestQueueDepthRules:
    def test_warning_threshold(self) -> None:
        register_distributed_alerts()
        manager = get_manager()
        # Below threshold — no alert
        alerts = manager.evaluate(METRIC_DIST_QUEUE_DEPTH, 50.0)
        assert len(alerts) == 0

    def test_warning_fires(self) -> None:
        register_distributed_alerts()
        manager = get_manager()
        alerts = manager.evaluate(METRIC_DIST_QUEUE_DEPTH, 150.0)
        # Only warning fires (150 > 100 but not > 500)
        assert len(alerts) == 1
        assert alerts[0].rule_name == RULE_QUEUE_DEPTH_WARNING
        assert alerts[0].severity == AlertSeverity.WARNING

    def test_critical_fires(self) -> None:
        register_distributed_alerts()
        manager = get_manager()
        alerts = manager.evaluate(METRIC_DIST_QUEUE_DEPTH, 600.0)
        # Both warning and critical fire (600 > 100 and > 500)
        assert len(alerts) == 2
        severities = {a.severity for a in alerts}
        assert severities == {AlertSeverity.WARNING, AlertSeverity.CRITICAL}

    def test_queue_depth_at_boundary(self) -> None:
        register_distributed_alerts()
        manager = get_manager()
        # Exactly 100 — GT means not firing
        alerts = manager.evaluate(METRIC_DIST_QUEUE_DEPTH, 100.0)
        assert len(alerts) == 0


# ---------------------------------------------------------------------------
# Failure rate rule
# ---------------------------------------------------------------------------


class TestFailureRateRule:
    def test_below_threshold(self) -> None:
        register_distributed_alerts()
        manager = get_manager()
        alerts = manager.evaluate(METRIC_DIST_TASKS_FAILED, 0.05)
        assert len(alerts) == 0

    def test_above_threshold(self) -> None:
        register_distributed_alerts()
        manager = get_manager()
        alerts = manager.evaluate(METRIC_DIST_TASKS_FAILED, 0.15)
        assert len(alerts) == 1
        assert alerts[0].rule_name == RULE_FAILURE_RATE_WARNING
        assert alerts[0].severity == AlertSeverity.WARNING

    def test_at_boundary(self) -> None:
        register_distributed_alerts()
        manager = get_manager()
        # Exactly 0.1 — GT means not firing
        alerts = manager.evaluate(METRIC_DIST_TASKS_FAILED, 0.1)
        assert len(alerts) == 0


# ---------------------------------------------------------------------------
# Worker count rule
# ---------------------------------------------------------------------------


class TestWorkerCountRule:
    def test_no_workers_fires_critical(self) -> None:
        register_distributed_alerts()
        manager = get_manager()
        alerts = manager.evaluate(METRIC_WORKER_COUNT, 0.0)
        assert len(alerts) == 1
        assert alerts[0].rule_name == RULE_WORKER_COUNT_CRITICAL
        assert alerts[0].severity == AlertSeverity.CRITICAL

    def test_workers_available_no_alert(self) -> None:
        register_distributed_alerts()
        manager = get_manager()
        alerts = manager.evaluate(METRIC_WORKER_COUNT, 3.0)
        assert len(alerts) == 0


# ---------------------------------------------------------------------------
# Wait time rule
# ---------------------------------------------------------------------------


class TestWaitTimeRule:
    def test_below_threshold(self) -> None:
        register_distributed_alerts()
        manager = get_manager()
        alerts = manager.evaluate(METRIC_DIST_TASK_WAIT_TIME, 30.0)
        assert len(alerts) == 0

    def test_above_threshold(self) -> None:
        register_distributed_alerts()
        manager = get_manager()
        alerts = manager.evaluate(METRIC_DIST_TASK_WAIT_TIME, 90.0)
        assert len(alerts) == 1
        assert alerts[0].rule_name == RULE_WAIT_TIME_WARNING
        assert alerts[0].severity == AlertSeverity.WARNING

    def test_at_boundary(self) -> None:
        register_distributed_alerts()
        manager = get_manager()
        # Exactly 60 — GT means not firing
        alerts = manager.evaluate(METRIC_DIST_TASK_WAIT_TIME, 60.0)
        assert len(alerts) == 0


# ---------------------------------------------------------------------------
# Callback integration
# ---------------------------------------------------------------------------


class TestAlertCallbacks:
    def test_callback_fires_on_alert(self) -> None:
        register_distributed_alerts()
        manager = get_manager()
        cb = MagicMock()
        manager.register_callback(cb)

        manager.evaluate(METRIC_DIST_QUEUE_DEPTH, 600.0)
        # Both warning and critical fire
        assert cb.call_count == 2

    def test_callback_receives_correct_metadata(self) -> None:
        register_distributed_alerts()
        manager = get_manager()
        received = []
        manager.register_callback(lambda a: received.append(a))

        manager.evaluate(METRIC_WORKER_COUNT, 0.0)
        assert len(received) == 1
        assert received[0].rule_name == RULE_WORKER_COUNT_CRITICAL
        assert received[0].metric_value == 0.0
        assert received[0].threshold == 0.0
        assert received[0].severity == AlertSeverity.CRITICAL
