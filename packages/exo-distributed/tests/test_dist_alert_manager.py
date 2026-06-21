"""Tests for exo.distributed._alerts — alerting hooks system."""

from __future__ import annotations

import asyncio
import time
from datetime import datetime
from unittest.mock import MagicMock

import pytest

from exo.distributed._alerts import (  # pyright: ignore[reportMissingImports]
    Alert,
    AlertManager,
    AlertRule,
    AlertSeverity,
    Comparator,
    get_manager,
    reset,
)

# ---------------------------------------------------------------------------
# AlertSeverity
# ---------------------------------------------------------------------------


class TestAlertSeverity:
    def test_values(self) -> None:
        assert AlertSeverity.INFO == "info"
        assert AlertSeverity.WARNING == "warning"
        assert AlertSeverity.CRITICAL == "critical"

    def test_is_str(self) -> None:
        for s in AlertSeverity:
            assert isinstance(s, str)


# ---------------------------------------------------------------------------
# Comparator
# ---------------------------------------------------------------------------


class TestComparator:
    def test_values(self) -> None:
        assert Comparator.GT == "gt"
        assert Comparator.LT == "lt"
        assert Comparator.GTE == "gte"
        assert Comparator.LTE == "lte"
        assert Comparator.EQ == "eq"

    def test_is_str(self) -> None:
        for c in Comparator:
            assert isinstance(c, str)


# ---------------------------------------------------------------------------
# AlertRule
# ---------------------------------------------------------------------------


class TestAlertRule:
    def test_creation(self) -> None:
        rule = AlertRule(name="high_latency", metric_name="latency", threshold=1.0)
        assert rule.name == "high_latency"
        assert rule.metric_name == "latency"
        assert rule.threshold == 1.0
        assert rule.comparator == Comparator.GT
        assert rule.severity == AlertSeverity.WARNING
        assert rule.cooldown_seconds == 300.0

    def test_custom_fields(self) -> None:
        rule = AlertRule(
            name="low_success",
            metric_name="success_rate",
            threshold=0.95,
            comparator=Comparator.LT,
            severity=AlertSeverity.CRITICAL,
            cooldown_seconds=60.0,
        )
        assert rule.comparator == Comparator.LT
        assert rule.severity == AlertSeverity.CRITICAL
        assert rule.cooldown_seconds == 60.0

    def test_frozen(self) -> None:
        rule = AlertRule(name="r", metric_name="m", threshold=1.0)
        with pytest.raises(AttributeError):
            rule.name = "changed"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Alert
# ---------------------------------------------------------------------------


class TestAlert:
    def test_creation(self) -> None:
        alert = Alert(
            rule_name="high_latency",
            metric_value=2.5,
            threshold=1.0,
            severity=AlertSeverity.WARNING,
        )
        assert alert.rule_name == "high_latency"
        assert alert.metric_value == 2.5
        assert alert.threshold == 1.0
        assert alert.severity == AlertSeverity.WARNING
        assert isinstance(alert.timestamp, datetime)
        assert alert.metadata == {}

    def test_with_metadata(self) -> None:
        alert = Alert(
            rule_name="r",
            metric_value=1.0,
            threshold=0.5,
            severity=AlertSeverity.INFO,
            metadata={"agent": "test"},
        )
        assert alert.metadata == {"agent": "test"}

    def test_frozen(self) -> None:
        alert = Alert(rule_name="r", metric_value=1.0, threshold=0.5, severity=AlertSeverity.INFO)
        with pytest.raises(AttributeError):
            alert.rule_name = "changed"  # type: ignore[misc]

    def test_timestamp_is_utc(self) -> None:
        alert = Alert(rule_name="r", metric_value=1.0, threshold=0.5, severity=AlertSeverity.INFO)
        assert alert.timestamp.tzinfo is not None


# ---------------------------------------------------------------------------
# AlertManager — rule management
# ---------------------------------------------------------------------------


class TestAlertManagerRules:
    def test_register_and_list(self) -> None:
        mgr = AlertManager()
        rule = AlertRule(name="r1", metric_name="m", threshold=1.0)
        mgr.register_rule(rule)
        rules = mgr.list_rules()
        assert len(rules) == 1
        assert rules[0].name == "r1"

    def test_unregister(self) -> None:
        mgr = AlertManager()
        rule = AlertRule(name="r1", metric_name="m", threshold=1.0)
        mgr.register_rule(rule)
        mgr.unregister_rule("r1")
        assert mgr.list_rules() == []

    def test_unregister_unknown(self) -> None:
        mgr = AlertManager()
        # Should not raise
        mgr.unregister_rule("nonexistent")

    def test_overwrite_rule(self) -> None:
        mgr = AlertManager()
        r1 = AlertRule(name="r", metric_name="m", threshold=1.0)
        r2 = AlertRule(name="r", metric_name="m", threshold=2.0)
        mgr.register_rule(r1)
        mgr.register_rule(r2)
        rules = mgr.list_rules()
        assert len(rules) == 1
        assert rules[0].threshold == 2.0

    def test_clear_rules(self) -> None:
        mgr = AlertManager()
        mgr.register_rule(AlertRule(name="r1", metric_name="m", threshold=1.0))
        mgr.register_rule(AlertRule(name="r2", metric_name="m", threshold=2.0))
        mgr.clear_rules()
        assert mgr.list_rules() == []


# ---------------------------------------------------------------------------
# AlertManager — comparators
# ---------------------------------------------------------------------------


class TestAlertManagerComparators:
    def test_gt(self) -> None:
        mgr = AlertManager()
        mgr.register_rule(
            AlertRule(
                name="r",
                metric_name="m",
                threshold=10.0,
                comparator=Comparator.GT,
                cooldown_seconds=0,
            )
        )
        assert len(mgr.evaluate("m", 11.0)) == 1
        assert len(mgr.evaluate("m", 10.0)) == 0
        assert len(mgr.evaluate("m", 9.0)) == 0

    def test_lt(self) -> None:
        mgr = AlertManager()
        mgr.register_rule(
            AlertRule(
                name="r",
                metric_name="m",
                threshold=10.0,
                comparator=Comparator.LT,
                cooldown_seconds=0,
            )
        )
        assert len(mgr.evaluate("m", 9.0)) == 1
        assert len(mgr.evaluate("m", 10.0)) == 0
        assert len(mgr.evaluate("m", 11.0)) == 0

    def test_gte(self) -> None:
        mgr = AlertManager()
        mgr.register_rule(
            AlertRule(
                name="r",
                metric_name="m",
                threshold=10.0,
                comparator=Comparator.GTE,
                cooldown_seconds=0,
            )
        )
        assert len(mgr.evaluate("m", 10.0)) == 1
        assert len(mgr.evaluate("m", 11.0)) == 1
        assert len(mgr.evaluate("m", 9.0)) == 0

    def test_lte(self) -> None:
        mgr = AlertManager()
        mgr.register_rule(
            AlertRule(
                name="r",
                metric_name="m",
                threshold=10.0,
                comparator=Comparator.LTE,
                cooldown_seconds=0,
            )
        )
        assert len(mgr.evaluate("m", 10.0)) == 1
        assert len(mgr.evaluate("m", 9.0)) == 1
        assert len(mgr.evaluate("m", 11.0)) == 0

    def test_eq(self) -> None:
        mgr = AlertManager()
        mgr.register_rule(
            AlertRule(
                name="r",
                metric_name="m",
                threshold=10.0,
                comparator=Comparator.EQ,
                cooldown_seconds=0,
            )
        )
        assert len(mgr.evaluate("m", 10.0)) == 1
        assert len(mgr.evaluate("m", 10.1)) == 0


# ---------------------------------------------------------------------------
# AlertManager — evaluation
# ---------------------------------------------------------------------------


class TestAlertManagerEvaluate:
    def test_no_matching_metric(self) -> None:
        mgr = AlertManager()
        mgr.register_rule(
            AlertRule(name="r", metric_name="latency", threshold=1.0, cooldown_seconds=0)
        )
        assert mgr.evaluate("errors", 5.0) == []

    def test_threshold_not_breached(self) -> None:
        mgr = AlertManager()
        mgr.register_rule(AlertRule(name="r", metric_name="m", threshold=10.0, cooldown_seconds=0))
        assert mgr.evaluate("m", 5.0) == []

    def test_threshold_breached(self) -> None:
        mgr = AlertManager()
        mgr.register_rule(AlertRule(name="r", metric_name="m", threshold=10.0, cooldown_seconds=0))
        alerts = mgr.evaluate("m", 15.0)
        assert len(alerts) == 1
        assert alerts[0].rule_name == "r"
        assert alerts[0].metric_value == 15.0
        assert alerts[0].threshold == 10.0

    def test_alert_severity_from_rule(self) -> None:
        mgr = AlertManager()
        mgr.register_rule(
            AlertRule(
                name="r",
                metric_name="m",
                threshold=10.0,
                severity=AlertSeverity.CRITICAL,
                cooldown_seconds=0,
            )
        )
        alerts = mgr.evaluate("m", 15.0)
        assert alerts[0].severity == AlertSeverity.CRITICAL

    def test_multiple_rules_same_metric(self) -> None:
        mgr = AlertManager()
        mgr.register_rule(
            AlertRule(
                name="warn",
                metric_name="m",
                threshold=10.0,
                severity=AlertSeverity.WARNING,
                cooldown_seconds=0,
            )
        )
        mgr.register_rule(
            AlertRule(
                name="crit",
                metric_name="m",
                threshold=20.0,
                severity=AlertSeverity.CRITICAL,
                cooldown_seconds=0,
            )
        )
        # Only first threshold breached
        alerts = mgr.evaluate("m", 15.0)
        assert len(alerts) == 1
        assert alerts[0].rule_name == "warn"

        # Both thresholds breached
        alerts = mgr.evaluate("m", 25.0)
        assert len(alerts) == 2
        names = {a.rule_name for a in alerts}
        assert names == {"warn", "crit"}


# ---------------------------------------------------------------------------
# AlertManager — cooldown
# ---------------------------------------------------------------------------


class TestAlertManagerCooldown:
    def test_cooldown_prevents_refiring(self) -> None:
        mgr = AlertManager()
        mgr.register_rule(
            AlertRule(name="r", metric_name="m", threshold=10.0, cooldown_seconds=300)
        )
        # First evaluation fires
        alerts = mgr.evaluate("m", 15.0)
        assert len(alerts) == 1

        # Second evaluation within cooldown does not fire
        alerts = mgr.evaluate("m", 15.0)
        assert len(alerts) == 0

    def test_cooldown_expiry(self) -> None:
        mgr = AlertManager()
        mgr.register_rule(
            AlertRule(name="r", metric_name="m", threshold=10.0, cooldown_seconds=0.01)
        )
        alerts = mgr.evaluate("m", 15.0)
        assert len(alerts) == 1

        # Wait for cooldown to expire
        time.sleep(0.02)

        alerts = mgr.evaluate("m", 15.0)
        assert len(alerts) == 1

    def test_zero_cooldown(self) -> None:
        mgr = AlertManager()
        mgr.register_rule(AlertRule(name="r", metric_name="m", threshold=10.0, cooldown_seconds=0))
        assert len(mgr.evaluate("m", 15.0)) == 1
        assert len(mgr.evaluate("m", 15.0)) == 1
        assert len(mgr.evaluate("m", 15.0)) == 1

    def test_cooldown_per_rule(self) -> None:
        """Each rule has its own cooldown timer."""
        mgr = AlertManager()
        mgr.register_rule(
            AlertRule(name="r1", metric_name="m", threshold=10.0, cooldown_seconds=300)
        )
        mgr.register_rule(AlertRule(name="r2", metric_name="m", threshold=5.0, cooldown_seconds=0))
        # Both fire on first eval
        alerts = mgr.evaluate("m", 15.0)
        assert len(alerts) == 2

        # r1 in cooldown, r2 fires again
        alerts = mgr.evaluate("m", 15.0)
        assert len(alerts) == 1
        assert alerts[0].rule_name == "r2"

    def test_unregister_clears_cooldown(self) -> None:
        mgr = AlertManager()
        mgr.register_rule(
            AlertRule(name="r", metric_name="m", threshold=10.0, cooldown_seconds=300)
        )
        mgr.evaluate("m", 15.0)
        mgr.unregister_rule("r")

        # Re-register same rule — cooldown should be cleared
        mgr.register_rule(
            AlertRule(name="r", metric_name="m", threshold=10.0, cooldown_seconds=300)
        )
        alerts = mgr.evaluate("m", 15.0)
        assert len(alerts) == 1


# ---------------------------------------------------------------------------
# AlertManager — callbacks
# ---------------------------------------------------------------------------


class TestAlertManagerCallbacks:
    def test_sync_callback_invoked(self) -> None:
        mgr = AlertManager()
        mgr.register_rule(AlertRule(name="r", metric_name="m", threshold=10.0, cooldown_seconds=0))
        received: list[Alert] = []
        mgr.register_callback(lambda a: received.append(a))

        mgr.evaluate("m", 15.0)
        assert len(received) == 1
        assert received[0].rule_name == "r"

    def test_multiple_callbacks(self) -> None:
        mgr = AlertManager()
        mgr.register_rule(AlertRule(name="r", metric_name="m", threshold=10.0, cooldown_seconds=0))
        cb1 = MagicMock()
        cb2 = MagicMock()
        mgr.register_callback(cb1)
        mgr.register_callback(cb2)

        mgr.evaluate("m", 15.0)
        cb1.assert_called_once()
        cb2.assert_called_once()

    def test_callback_not_invoked_when_not_breached(self) -> None:
        mgr = AlertManager()
        mgr.register_rule(AlertRule(name="r", metric_name="m", threshold=10.0, cooldown_seconds=0))
        cb = MagicMock()
        mgr.register_callback(cb)

        mgr.evaluate("m", 5.0)
        cb.assert_not_called()

    def test_callback_not_invoked_during_cooldown(self) -> None:
        mgr = AlertManager()
        mgr.register_rule(
            AlertRule(name="r", metric_name="m", threshold=10.0, cooldown_seconds=300)
        )
        cb = MagicMock()
        mgr.register_callback(cb)

        mgr.evaluate("m", 15.0)
        assert cb.call_count == 1

        mgr.evaluate("m", 15.0)
        assert cb.call_count == 1  # Still 1, not invoked again

    async def test_async_callback_invoked(self) -> None:
        mgr = AlertManager()
        mgr.register_rule(AlertRule(name="r", metric_name="m", threshold=10.0, cooldown_seconds=0))
        received: list[Alert] = []

        async def async_cb(alert: Alert) -> None:
            received.append(alert)

        mgr.register_callback(async_cb)
        mgr.evaluate("m", 15.0)

        # Allow the scheduled task to run
        await asyncio.sleep(0.01)
        assert len(received) == 1

    def test_clear_callbacks(self) -> None:
        mgr = AlertManager()
        mgr.register_rule(AlertRule(name="r", metric_name="m", threshold=10.0, cooldown_seconds=0))
        cb = MagicMock()
        mgr.register_callback(cb)
        mgr.clear_callbacks()

        mgr.evaluate("m", 15.0)
        cb.assert_not_called()

    def test_callback_receives_correct_alert(self) -> None:
        mgr = AlertManager()
        mgr.register_rule(
            AlertRule(
                name="high_cpu",
                metric_name="cpu",
                threshold=90.0,
                severity=AlertSeverity.CRITICAL,
                cooldown_seconds=0,
            )
        )
        received: list[Alert] = []
        mgr.register_callback(lambda a: received.append(a))

        mgr.evaluate("cpu", 95.0)
        assert received[0].rule_name == "high_cpu"
        assert received[0].metric_value == 95.0
        assert received[0].threshold == 90.0
        assert received[0].severity == AlertSeverity.CRITICAL


# ---------------------------------------------------------------------------
# AlertManager — clear / management
# ---------------------------------------------------------------------------


class TestAlertManagerClear:
    def test_clear_all(self) -> None:
        mgr = AlertManager()
        mgr.register_rule(AlertRule(name="r", metric_name="m", threshold=1.0))
        mgr.register_callback(lambda a: None)
        mgr.clear()
        assert mgr.list_rules() == []
        # Callback was cleared — evaluate should not invoke it
        cb = MagicMock()
        # Re-register rule but not callback
        mgr.register_rule(AlertRule(name="r", metric_name="m", threshold=1.0, cooldown_seconds=0))
        mgr.evaluate("m", 5.0)
        cb.assert_not_called()


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------


class TestModuleFunctions:
    def setup_method(self) -> None:
        reset()

    def teardown_method(self) -> None:
        reset()

    def test_get_manager(self) -> None:
        mgr = get_manager()
        assert isinstance(mgr, AlertManager)

    def test_get_manager_singleton(self) -> None:
        assert get_manager() is get_manager()

    def test_reset_clears_global(self) -> None:
        mgr = get_manager()
        mgr.register_rule(AlertRule(name="r", metric_name="m", threshold=1.0))
        assert len(mgr.list_rules()) == 1
        reset()
        assert len(mgr.list_rules()) == 0

    def test_full_flow(self) -> None:
        """Integration test: register rule + callback, evaluate, check alert."""
        mgr = get_manager()
        mgr.register_rule(
            AlertRule(
                name="latency_warn",
                metric_name="latency",
                threshold=0.5,
                severity=AlertSeverity.WARNING,
                cooldown_seconds=0,
            )
        )
        received: list[Alert] = []
        mgr.register_callback(lambda a: received.append(a))

        # Below threshold — no alert
        mgr.evaluate("latency", 0.3)
        assert len(received) == 0

        # Above threshold — alert fires
        mgr.evaluate("latency", 0.8)
        assert len(received) == 1
        assert received[0].metric_value == 0.8
        assert received[0].severity == AlertSeverity.WARNING
