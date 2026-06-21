"""Alerting hooks for metric threshold monitoring.

Register alert rules and callbacks that fire when metric values exceed
configurable thresholds. Supports cooldown to prevent alert storms and
both sync and async callback invocation.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import threading
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

logger = logging.getLogger(__name__)


class AlertSeverity(StrEnum):
    """Severity levels for alerts."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class Comparator(StrEnum):
    """Threshold comparators for alert rules."""

    GT = "gt"
    LT = "lt"
    GTE = "gte"
    LTE = "lte"
    EQ = "eq"


@dataclass(frozen=True)
class AlertRule:
    """Definition of a single alert rule."""

    name: str
    metric_name: str
    threshold: float
    comparator: Comparator = Comparator.GT
    severity: AlertSeverity = AlertSeverity.WARNING
    cooldown_seconds: float = 300.0


@dataclass(frozen=True)
class Alert:
    """An alert event fired when a rule is triggered."""

    rule_name: str
    metric_value: float
    threshold: float
    severity: AlertSeverity
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = field(default_factory=dict)


# Callback type: sync or async callable taking an Alert.
AlertCallback = Callable[[Alert], None] | Callable[[Alert], Awaitable[None]]

# Comparator functions keyed by Comparator enum value.
_COMPARATORS: dict[Comparator, Callable[[float, float], bool]] = {
    Comparator.GT: lambda value, threshold: value > threshold,
    Comparator.LT: lambda value, threshold: value < threshold,
    Comparator.GTE: lambda value, threshold: value >= threshold,
    Comparator.LTE: lambda value, threshold: value <= threshold,
    Comparator.EQ: lambda value, threshold: value == threshold,
}


class AlertManager:
    """Manages alert rules and dispatches callbacks when thresholds are breached."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._rules: dict[str, AlertRule] = {}
        self._callbacks: list[AlertCallback] = []
        # Tracks last fire time (monotonic) per rule name for cooldown.
        self._last_fired: dict[str, float] = {}
        # Background task references (prevent GC of fire-and-forget tasks).
        self._background_tasks: set[asyncio.Task[None]] = set()

    # -- Rules ----------------------------------------------------------------

    def register_rule(self, rule: AlertRule) -> None:
        """Register an alert rule (overwrites if name exists)."""
        with self._lock:
            self._rules[rule.name] = rule
        logger.debug(
            "registered alert rule %r (metric=%s, %s %s)",
            rule.name,
            rule.metric_name,
            rule.comparator,
            rule.threshold,
        )

    def unregister_rule(self, name: str) -> None:
        """Remove an alert rule by name."""
        with self._lock:
            self._rules.pop(name, None)
            self._last_fired.pop(name, None)
        logger.debug("unregistered alert rule %r", name)

    def list_rules(self) -> list[AlertRule]:
        """Return all registered rules."""
        with self._lock:
            return list(self._rules.values())

    # -- Callbacks ------------------------------------------------------------

    def register_callback(self, callback: AlertCallback) -> None:
        """Register a callback to be invoked when any alert fires."""
        with self._lock:
            self._callbacks.append(callback)
        logger.debug("registered alert callback %r", callback)

    # -- Evaluation -----------------------------------------------------------

    def evaluate(self, metric_name: str, value: float) -> list[Alert]:
        """Check *value* against all rules matching *metric_name*.

        Returns a list of :class:`Alert` objects for rules that fired.
        Callbacks are invoked synchronously (async callbacks are scheduled
        on the running event loop if one exists, otherwise run via
        ``asyncio.run``).
        """
        now = time.monotonic()
        fired: list[Alert] = []

        with self._lock:
            matching = [r for r in self._rules.values() if r.metric_name == metric_name]
            callbacks = list(self._callbacks)

        for rule in matching:
            comparator_fn = _COMPARATORS[rule.comparator]
            if not comparator_fn(value, rule.threshold):
                continue

            # Cooldown check
            with self._lock:
                last = self._last_fired.get(rule.name, 0.0)
                if now - last < rule.cooldown_seconds:
                    logger.debug(
                        "alert rule %r in cooldown (%.1fs remaining)",
                        rule.name,
                        rule.cooldown_seconds - (now - last),
                    )
                    continue
                self._last_fired[rule.name] = now

            alert = Alert(
                rule_name=rule.name,
                metric_value=value,
                threshold=rule.threshold,
                severity=rule.severity,
            )
            fired.append(alert)
            logger.warning(
                "alert fired: %r severity=%s value=%s threshold=%s",
                rule.name,
                rule.severity,
                value,
                rule.threshold,
            )

            for cb in callbacks:
                self._invoke_callback(cb, alert)

        return fired

    # -- Management -----------------------------------------------------------

    def clear_rules(self) -> None:
        """Remove all rules and cooldown state."""
        with self._lock:
            self._rules.clear()
            self._last_fired.clear()

    def clear_callbacks(self) -> None:
        """Remove all callbacks."""
        with self._lock:
            self._callbacks.clear()

    def clear(self) -> None:
        """Remove all rules, callbacks, and cooldown state."""
        with self._lock:
            self._rules.clear()
            self._callbacks.clear()
            self._last_fired.clear()

    def _invoke_callback(self, cb: AlertCallback, alert: Alert) -> None:
        """Invoke a callback, handling both sync and async."""
        try:
            if inspect.iscoroutinefunction(cb):
                try:
                    loop = asyncio.get_running_loop()
                    task = loop.create_task(cb(alert))  # type: ignore[arg-type]
                    self._background_tasks.add(task)
                    task.add_done_callback(self._background_tasks.discard)
                except RuntimeError:
                    asyncio.run(cb(alert))  # type: ignore[arg-type]
            else:
                cb(alert)  # type: ignore[arg-type]
        except Exception:
            logger.error("alert callback %r failed for rule %r", cb, alert.rule_name, exc_info=True)


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------

_global_manager = AlertManager()


def get_manager() -> AlertManager:
    """Return the global alert manager."""
    return _global_manager


def reset() -> None:
    """Reset global alert manager — for testing only."""
    _global_manager.clear()
