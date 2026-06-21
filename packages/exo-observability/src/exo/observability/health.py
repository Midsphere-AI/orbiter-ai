"""Health check system for monitoring observability pipeline and agent health.

Provides a registry of health checks that can be evaluated individually or in
aggregate, producing a JSON-serializable health summary.
"""

from __future__ import annotations

import logging
import resource
import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


class HealthStatus(StrEnum):
    """Possible outcomes for a health check."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass(frozen=True)
class HealthResult:
    """The result of a single health check evaluation."""

    status: HealthStatus
    message: str
    metadata: dict[str, Any] = field(default_factory=dict)
    checked_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "status": str(self.status),
            "message": self.message,
            "metadata": self.metadata,
            "checked_at": self.checked_at.isoformat(),
        }


@runtime_checkable
class HealthCheck(Protocol):
    """Protocol for health checks."""

    @property
    def name(self) -> str: ...

    def check(self) -> HealthResult: ...


# ---------------------------------------------------------------------------
# Built-in checks
# ---------------------------------------------------------------------------


class MemoryUsageCheck:
    """Check process RSS memory against a threshold (in MB)."""

    def __init__(self, threshold_mb: float = 512.0) -> None:
        self._threshold_mb = threshold_mb

    @property
    def name(self) -> str:
        return "memory_usage"

    def check(self) -> HealthResult:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        # ru_maxrss is in KB on Linux, bytes on macOS
        rss_mb = usage.ru_maxrss / 1024.0
        metadata = {"rss_mb": round(rss_mb, 2), "threshold_mb": self._threshold_mb}

        if rss_mb >= self._threshold_mb:
            return HealthResult(
                status=HealthStatus.UNHEALTHY,
                message=f"RSS {rss_mb:.1f}MB exceeds threshold {self._threshold_mb:.1f}MB",
                metadata=metadata,
            )
        if rss_mb >= self._threshold_mb * 0.8:
            return HealthResult(
                status=HealthStatus.DEGRADED,
                message=f"RSS {rss_mb:.1f}MB approaching threshold {self._threshold_mb:.1f}MB",
                metadata=metadata,
            )
        return HealthResult(
            status=HealthStatus.HEALTHY,
            message=f"RSS {rss_mb:.1f}MB within threshold {self._threshold_mb:.1f}MB",
            metadata=metadata,
        )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_STATUS_PRIORITY = {
    HealthStatus.HEALTHY: 0,
    HealthStatus.DEGRADED: 1,
    HealthStatus.UNHEALTHY: 2,
}


class HealthRegistry:
    """Registry of health checks with aggregate evaluation."""

    def __init__(self) -> None:
        self._checks: dict[str, HealthCheck] = {}
        self._lock = threading.Lock()

    def register(self, check: HealthCheck) -> None:
        """Register a health check by its name."""
        with self._lock:
            self._checks[check.name] = check
        logger.debug("registered health check %r", check.name)

    def unregister(self, name: str) -> None:
        """Remove a health check by name."""
        with self._lock:
            self._checks.pop(name, None)

    def run_all(self) -> dict[str, HealthResult]:
        """Run all registered checks and return results keyed by name."""
        with self._lock:
            checks = dict(self._checks)
        results: dict[str, HealthResult] = {}
        for name, check in checks.items():
            try:
                results[name] = check.check()
            except Exception:
                logger.error("health check %r raised an exception", name, exc_info=True)
                results[name] = HealthResult(
                    status=HealthStatus.UNHEALTHY,
                    message=f"Check {name!r} raised an exception",
                )
        logger.debug("ran %d health checks", len(results))
        return results

    def run(self, name: str) -> HealthResult:
        """Run a single named check."""
        with self._lock:
            check = self._checks.get(name)
        if check is None:
            msg = f"Unknown health check: {name}"
            raise KeyError(msg)
        return check.check()

    def aggregate_status(self, results: dict[str, HealthResult] | None = None) -> HealthStatus:
        """Compute aggregate status from results.

        UNHEALTHY if any check is UNHEALTHY, DEGRADED if any is DEGRADED,
        else HEALTHY.
        """
        if results is None:
            results = self.run_all()
        if not results:
            return HealthStatus.HEALTHY
        worst = max(results.values(), key=lambda r: _STATUS_PRIORITY[r.status])
        return worst.status

    def list_checks(self) -> list[str]:
        """Return names of all registered checks."""
        with self._lock:
            return list(self._checks.keys())

    def clear(self) -> None:
        """Remove all registered checks."""
        with self._lock:
            self._checks.clear()


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------

_global_registry = HealthRegistry()


def get_registry() -> HealthRegistry:
    """Return the global health registry."""
    return _global_registry


def get_health_summary() -> dict[str, Any]:
    """Return a JSON-serializable health report from the global registry."""
    results = _global_registry.run_all()
    status = _global_registry.aggregate_status(results)
    return {
        "status": str(status),
        "checks": {name: result.to_dict() for name, result in results.items()},
        "timestamp": datetime.now(UTC).isoformat(),
    }


def reset() -> None:
    """Reset global registry — for testing only."""
    _global_registry.clear()
