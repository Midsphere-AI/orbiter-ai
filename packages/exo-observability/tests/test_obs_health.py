"""Tests for exo.observability.health — health check system."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from exo.observability.health import (  # pyright: ignore[reportMissingImports]
    HealthCheck,
    HealthRegistry,
    HealthResult,
    HealthStatus,
    MemoryUsageCheck,
    get_health_summary,
    get_registry,
    reset,
)
from exo.types import ExoError  # pyright: ignore[reportMissingImports]

# ---------------------------------------------------------------------------
# HealthStatus
# ---------------------------------------------------------------------------


class TestHealthStatus:
    def test_values(self) -> None:
        assert HealthStatus.HEALTHY == "healthy"
        assert HealthStatus.DEGRADED == "degraded"
        assert HealthStatus.UNHEALTHY == "unhealthy"

    def test_is_str(self) -> None:
        for status in HealthStatus:
            assert isinstance(status, str)


# ---------------------------------------------------------------------------
# HealthResult
# ---------------------------------------------------------------------------


class TestHealthResult:
    def test_creation(self) -> None:
        result = HealthResult(
            status=HealthStatus.HEALTHY,
            message="All good",
        )
        assert result.status == HealthStatus.HEALTHY
        assert result.message == "All good"
        assert result.metadata == {}
        assert isinstance(result.checked_at, datetime)

    def test_with_metadata(self) -> None:
        result = HealthResult(
            status=HealthStatus.DEGRADED,
            message="High load",
            metadata={"cpu": 85},
        )
        assert result.metadata == {"cpu": 85}

    def test_frozen(self) -> None:
        result = HealthResult(status=HealthStatus.HEALTHY, message="ok")
        with pytest.raises(AttributeError):
            result.message = "changed"  # type: ignore[misc]

    def test_to_dict(self) -> None:
        result = HealthResult(
            status=HealthStatus.UNHEALTHY,
            message="down",
            metadata={"reason": "timeout"},
        )
        d = result.to_dict()
        assert d["status"] == "unhealthy"
        assert d["message"] == "down"
        assert d["metadata"] == {"reason": "timeout"}
        assert "checked_at" in d
        # ISO format string
        datetime.fromisoformat(d["checked_at"])

    def test_checked_at_is_utc(self) -> None:
        result = HealthResult(status=HealthStatus.HEALTHY, message="ok")
        assert result.checked_at.tzinfo is not None


# ---------------------------------------------------------------------------
# HealthCheck protocol
# ---------------------------------------------------------------------------


class TestHealthCheckProtocol:
    def test_protocol_compliance(self) -> None:
        """A class with name property and check() method satisfies the protocol."""

        class MyCheck:
            @property
            def name(self) -> str:
                return "my_check"

            def check(self) -> HealthResult:
                return HealthResult(status=HealthStatus.HEALTHY, message="ok")

        instance = MyCheck()
        assert isinstance(instance, HealthCheck)

    def test_builtin_checks_satisfy_protocol(self) -> None:
        assert isinstance(MemoryUsageCheck(), HealthCheck)


# ---------------------------------------------------------------------------
# MemoryUsageCheck
# ---------------------------------------------------------------------------


class TestMemoryUsageCheck:
    def test_name(self) -> None:
        assert MemoryUsageCheck().name == "memory_usage"

    def test_healthy(self) -> None:
        # Use a very high threshold to ensure healthy
        check = MemoryUsageCheck(threshold_mb=999_999.0)
        result = check.check()
        assert result.status == HealthStatus.HEALTHY
        assert "rss_mb" in result.metadata
        assert "threshold_mb" in result.metadata

    def test_unhealthy(self) -> None:
        # Use a tiny threshold to trigger unhealthy
        check = MemoryUsageCheck(threshold_mb=0.001)
        result = check.check()
        assert result.status == HealthStatus.UNHEALTHY
        assert "exceeds" in result.message

    def test_degraded(self) -> None:
        """Degraded when RSS >= 80% of threshold."""
        # Mock resource to return a controlled value
        mock_usage = MagicMock()
        # Set ru_maxrss to 850 KB = ~0.83 MB (>80% of 1.0MB, <1.0MB)
        mock_usage.ru_maxrss = 850
        with patch("exo.observability.health.resource.getrusage", return_value=mock_usage):
            check = MemoryUsageCheck(threshold_mb=1.0)
            result = check.check()
            assert result.status == HealthStatus.DEGRADED
            assert "approaching" in result.message


# ---------------------------------------------------------------------------
# HealthRegistry
# ---------------------------------------------------------------------------


class _AlwaysHealthy:
    @property
    def name(self) -> str:
        return "always_healthy"

    def check(self) -> HealthResult:
        return HealthResult(status=HealthStatus.HEALTHY, message="ok")


class _AlwaysDegraded:
    @property
    def name(self) -> str:
        return "always_degraded"

    def check(self) -> HealthResult:
        return HealthResult(status=HealthStatus.DEGRADED, message="degraded")


class _AlwaysUnhealthy:
    @property
    def name(self) -> str:
        return "always_unhealthy"

    def check(self) -> HealthResult:
        return HealthResult(status=HealthStatus.UNHEALTHY, message="down")


class TestHealthRegistry:
    def test_register_and_list(self) -> None:
        registry = HealthRegistry()
        registry.register(_AlwaysHealthy())
        assert "always_healthy" in registry.list_checks()

    def test_unregister(self) -> None:
        registry = HealthRegistry()
        registry.register(_AlwaysHealthy())
        registry.unregister("always_healthy")
        assert "always_healthy" not in registry.list_checks()

    def test_unregister_unknown(self) -> None:
        registry = HealthRegistry()
        # Should not raise
        registry.unregister("nonexistent")

    def test_run_single(self) -> None:
        registry = HealthRegistry()
        registry.register(_AlwaysHealthy())
        result = registry.run("always_healthy")
        assert result.status == HealthStatus.HEALTHY

    def test_run_unknown_raises(self) -> None:
        registry = HealthRegistry()
        with pytest.raises(ExoError, match="Unknown health check"):
            registry.run("nonexistent")

    def test_run_unknown_raises_has_hint(self) -> None:
        registry = HealthRegistry()
        with pytest.raises(ExoError) as exc_info:
            registry.run("nonexistent")
        err = exc_info.value
        assert err.hint is not None
        assert "list_checks" in err.hint
        assert err.context.get("check") == "nonexistent"

    def test_run_all(self) -> None:
        registry = HealthRegistry()
        registry.register(_AlwaysHealthy())
        registry.register(_AlwaysDegraded())
        results = registry.run_all()
        assert len(results) == 2
        assert results["always_healthy"].status == HealthStatus.HEALTHY
        assert results["always_degraded"].status == HealthStatus.DEGRADED

    def test_run_all_empty(self) -> None:
        registry = HealthRegistry()
        results = registry.run_all()
        assert results == {}

    def test_clear(self) -> None:
        registry = HealthRegistry()
        registry.register(_AlwaysHealthy())
        registry.clear()
        assert registry.list_checks() == []

    def test_overwrite_check(self) -> None:
        """Registering a check with the same name replaces the old one."""
        registry = HealthRegistry()
        registry.register(_AlwaysHealthy())

        class _ReplacementHealthy:
            @property
            def name(self) -> str:
                return "always_healthy"

            def check(self) -> HealthResult:
                return HealthResult(status=HealthStatus.DEGRADED, message="replaced")

        registry.register(_ReplacementHealthy())
        result = registry.run("always_healthy")
        assert result.status == HealthStatus.DEGRADED


# ---------------------------------------------------------------------------
# Aggregate status
# ---------------------------------------------------------------------------


class TestAggregateStatus:
    def test_all_healthy(self) -> None:
        registry = HealthRegistry()
        registry.register(_AlwaysHealthy())
        assert registry.aggregate_status() == HealthStatus.HEALTHY

    def test_one_degraded(self) -> None:
        registry = HealthRegistry()
        registry.register(_AlwaysHealthy())
        registry.register(_AlwaysDegraded())
        assert registry.aggregate_status() == HealthStatus.DEGRADED

    def test_one_unhealthy(self) -> None:
        registry = HealthRegistry()
        registry.register(_AlwaysHealthy())
        registry.register(_AlwaysDegraded())
        registry.register(_AlwaysUnhealthy())
        assert registry.aggregate_status() == HealthStatus.UNHEALTHY

    def test_empty_registry_healthy(self) -> None:
        registry = HealthRegistry()
        assert registry.aggregate_status() == HealthStatus.HEALTHY

    def test_with_explicit_results(self) -> None:
        registry = HealthRegistry()
        results = {
            "a": HealthResult(status=HealthStatus.HEALTHY, message="ok"),
            "b": HealthResult(status=HealthStatus.DEGRADED, message="slow"),
        }
        assert registry.aggregate_status(results) == HealthStatus.DEGRADED

    def test_unhealthy_wins_over_degraded(self) -> None:
        registry = HealthRegistry()
        results = {
            "a": HealthResult(status=HealthStatus.DEGRADED, message="slow"),
            "b": HealthResult(status=HealthStatus.UNHEALTHY, message="down"),
        }
        assert registry.aggregate_status(results) == HealthStatus.UNHEALTHY


# ---------------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------------


class TestModuleFunctions:
    def setup_method(self) -> None:
        reset()

    def teardown_method(self) -> None:
        reset()

    def test_get_registry(self) -> None:
        reg = get_registry()
        assert isinstance(reg, HealthRegistry)

    def test_get_registry_singleton(self) -> None:
        assert get_registry() is get_registry()

    def test_get_health_summary_empty(self) -> None:
        summary = get_health_summary()
        assert summary["status"] == "healthy"
        assert summary["checks"] == {}
        assert "timestamp" in summary

    def test_get_health_summary_with_checks(self) -> None:
        get_registry().register(_AlwaysHealthy())
        get_registry().register(_AlwaysDegraded())
        summary = get_health_summary()
        assert summary["status"] == "degraded"
        assert "always_healthy" in summary["checks"]
        assert "always_degraded" in summary["checks"]
        assert summary["checks"]["always_healthy"]["status"] == "healthy"
        assert summary["checks"]["always_degraded"]["status"] == "degraded"

    def test_reset_clears_global_registry(self) -> None:
        get_registry().register(_AlwaysHealthy())
        assert len(get_registry().list_checks()) == 1
        reset()
        assert len(get_registry().list_checks()) == 0

    def test_health_summary_serializable(self) -> None:
        """Ensure the summary is JSON-serializable."""
        import json

        get_registry().register(_AlwaysHealthy())
        summary = get_health_summary()
        # Should not raise
        json.dumps(summary)
