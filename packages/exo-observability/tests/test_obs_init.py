"""Tests for exo.observability public API (__init__.py exports)."""

from __future__ import annotations

import importlib
import sys
from unittest.mock import patch

import pytest

# ── Eager import smoke tests ──────────────────────────────────────────────


class TestEagerImports:
    """Core symbols are importable directly from exo.observability."""

    def test_configure(self) -> None:
        from exo.observability import configure  # pyright: ignore[reportMissingImports]

        assert callable(configure)

    def test_get_logger(self) -> None:
        from exo.observability import get_logger  # pyright: ignore[reportMissingImports]

        assert callable(get_logger)

    def test_traced(self) -> None:
        from exo.observability import traced  # pyright: ignore[reportMissingImports]

        assert callable(traced)

    def test_span(self) -> None:
        from exo.observability import span  # pyright: ignore[reportMissingImports]

        assert callable(span)

    def test_aspan(self) -> None:
        from exo.observability import aspan  # pyright: ignore[reportMissingImports]

        assert callable(aspan)

    def test_get_config(self) -> None:
        from exo.observability import get_config  # pyright: ignore[reportMissingImports]

        assert callable(get_config)

    def test_configure_logging(self) -> None:
        from exo.observability import configure_logging  # pyright: ignore[reportMissingImports]

        assert callable(configure_logging)

    def test_log_context(self) -> None:
        from exo.observability import LogContext  # pyright: ignore[reportMissingImports]

        assert LogContext is not None

    def test_observability_config(self) -> None:
        from exo.observability import (  # pyright: ignore[reportMissingImports]
            ObservabilityConfig,
        )

        assert ObservabilityConfig is not None

    def test_trace_backend(self) -> None:
        from exo.observability import TraceBackend  # pyright: ignore[reportMissingImports]

        assert TraceBackend is not None


# ── Lazy import smoke tests ───────────────────────────────────────────────


class TestLazyImports:
    """Lazy-loaded symbols are accessible via exo.observability.<name>."""

    def test_metrics_collector(self) -> None:
        from exo.observability import MetricsCollector  # pyright: ignore[reportMissingImports]

        assert MetricsCollector is not None

    def test_timer(self) -> None:
        from exo.observability import Timer  # pyright: ignore[reportMissingImports]

        assert Timer is not None

    def test_null_span(self) -> None:
        from exo.observability import NullSpan  # pyright: ignore[reportMissingImports]

        assert NullSpan is not None

    def test_health_status(self) -> None:
        from exo.observability import HealthStatus  # pyright: ignore[reportMissingImports]

        assert HealthStatus is not None

    def test_semconv_constant(self) -> None:
        from exo.observability import AGENT_NAME  # pyright: ignore[reportMissingImports]

        assert AGENT_NAME == "exo.agent.name"

    def test_record_agent_run(self) -> None:
        from exo.observability import record_agent_run  # pyright: ignore[reportMissingImports]

        assert callable(record_agent_run)

    def test_health_registry(self) -> None:
        from exo.observability import HealthRegistry  # pyright: ignore[reportMissingImports]

        assert HealthRegistry is not None


# ── Sub-module imports ────────────────────────────────────────────────────


class TestSubModuleImports:
    """Sub-module imports work correctly."""

    def test_from_metrics(self) -> None:
        from exo.observability.metrics import (  # pyright: ignore[reportMissingImports]
            MetricsCollector,
        )

        assert MetricsCollector is not None

    def test_from_tracing(self) -> None:
        from exo.observability.tracing import traced  # pyright: ignore[reportMissingImports]

        assert callable(traced)

    def test_from_logging(self) -> None:
        from exo.observability.logging import (  # pyright: ignore[reportMissingImports]
            get_logger,
        )

        assert callable(get_logger)

    def test_from_semconv(self) -> None:
        from exo.observability.semconv import (  # pyright: ignore[reportMissingImports]
            AGENT_NAME,
        )

        assert isinstance(AGENT_NAME, str)

    def test_from_health(self) -> None:
        from exo.observability.health import (  # pyright: ignore[reportMissingImports]
            HealthStatus,
        )

        assert HealthStatus is not None

    def test_from_config(self) -> None:
        from exo.observability.config import (  # pyright: ignore[reportMissingImports]
            ObservabilityConfig,
        )

        assert ObservabilityConfig is not None


# ── __all__ completeness ──────────────────────────────────────────────────


class TestAllCompleteness:
    """__all__ is complete and matches available exports."""

    def test_all_is_list(self) -> None:
        import exo.observability as obs  # pyright: ignore[reportMissingImports]

        assert isinstance(obs.__all__, list)

    def test_all_not_empty(self) -> None:
        import exo.observability as obs  # pyright: ignore[reportMissingImports]

        assert len(obs.__all__) > 0

    def test_all_has_no_duplicates(self) -> None:
        import exo.observability as obs  # pyright: ignore[reportMissingImports]

        assert len(obs.__all__) == len(set(obs.__all__))

    def test_all_symbols_are_strings(self) -> None:
        import exo.observability as obs  # pyright: ignore[reportMissingImports]

        for name in obs.__all__:
            assert isinstance(name, str), f"{name!r} is not a string"

    def test_all_symbols_resolve(self) -> None:
        """Every name in __all__ is importable from the package."""
        import exo.observability as obs  # pyright: ignore[reportMissingImports]

        for name in obs.__all__:
            value = getattr(obs, name)
            assert value is not None, f"{name} resolved to None"

    def test_all_includes_core_symbols(self) -> None:
        import exo.observability as obs  # pyright: ignore[reportMissingImports]

        core = {
            "configure",
            "get_logger",
            "traced",
            "span",
            "aspan",
            "ObservabilityConfig",
            "LogContext",
            "get_config",
            "configure_logging",
            "TraceBackend",
        }
        assert core.issubset(set(obs.__all__))

    def test_all_includes_metrics_symbols(self) -> None:
        import exo.observability as obs  # pyright: ignore[reportMissingImports]

        metrics = {
            "MetricsCollector",
            "Timer",
            "timer",
            "record_agent_run",
            "record_tool_step",
            "get_metrics_snapshot",
            "reset_metrics",
        }
        assert metrics.issubset(set(obs.__all__))

    def test_all_includes_health_symbols(self) -> None:
        import exo.observability as obs  # pyright: ignore[reportMissingImports]

        health = {"HealthStatus", "HealthResult", "HealthCheck", "HealthRegistry"}
        assert health.issubset(set(obs.__all__))

    def test_all_includes_semconv_constants(self) -> None:
        import exo.observability as obs  # pyright: ignore[reportMissingImports]

        semconv = {"AGENT_NAME", "TOOL_NAME", "GEN_AI_SYSTEM", "COST_TOTAL_USD"}
        assert semconv.issubset(set(obs.__all__))

    def test_all_includes_tracing_symbols(self) -> None:
        import exo.observability as obs  # pyright: ignore[reportMissingImports]

        tracing = {"NullSpan", "SpanLike", "extract_metadata", "traced", "span", "aspan"}
        assert tracing.issubset(set(obs.__all__))


# ── Lazy loading behavior ────────────────────────────────────────────────


class TestLazyLoading:
    """Lazy imports are actually lazy (not loaded at module import time)."""

    def test_getattr_unknown_raises(self) -> None:
        import exo.observability as obs  # pyright: ignore[reportMissingImports]

        with pytest.raises(AttributeError, match="no_such_symbol"):
            _ = obs.no_such_symbol  # type: ignore[attr-defined]

    def test_lazy_symbol_cached_after_first_access(self) -> None:
        """After first access, symbol is in module globals (no repeated __getattr__)."""
        import exo.observability as obs  # pyright: ignore[reportMissingImports]

        # First access triggers __getattr__
        val1 = obs.MetricsCollector
        # Second access should come from globals (same object)
        val2 = obs.MetricsCollector
        assert val1 is val2

    def test_lazy_import_returns_correct_type(self) -> None:
        import exo.observability as obs  # pyright: ignore[reportMissingImports]

        # Check a few lazy symbols match their sub-module versions
        from exo.observability.health import (  # pyright: ignore[reportMissingImports]
            HealthStatus,
        )

        assert obs.HealthStatus is HealthStatus

    def test_semconv_lazy_loads(self) -> None:
        import exo.observability as obs  # pyright: ignore[reportMissingImports]
        from exo.observability.semconv import (  # pyright: ignore[reportMissingImports]
            GEN_AI_SYSTEM,
        )

        assert obs.GEN_AI_SYSTEM == GEN_AI_SYSTEM


# ── OTel optional behavior ───────────────────────────────────────────────


class TestOTelOptional:
    """Package imports work even without opentelemetry installed."""

    def test_import_without_otel(self) -> None:
        """Simulate OTel not installed by patching HAS_OTEL flags."""
        # The modules use try/except ImportError so they already handle it.
        # Verify the main package loads when OTel-dependent modules report
        # no OTel.
        with (
            patch("exo.observability.tracing.HAS_OTEL", False),
            patch("exo.observability.metrics.HAS_OTEL", False),
        ):
            # Re-import to verify no errors
            importlib.reload(sys.modules["exo.observability"])
            import exo.observability as obs  # pyright: ignore[reportMissingImports]

            assert callable(obs.configure)
            assert callable(obs.get_logger)
            assert callable(obs.traced)

    def test_traced_noop_without_otel(self) -> None:
        """When OTel is not installed, @traced() is a passthrough."""
        from exo.observability import traced  # pyright: ignore[reportMissingImports]

        with patch("exo.observability.tracing.HAS_OTEL", False):

            @traced()
            def example() -> str:
                return "hello"

            # Should be the original function (no wrapper)
            assert example() == "hello"


# ── Convenience import patterns ──────────────────────────────────────────


class TestConveniencePatterns:
    """Common import patterns work as expected."""

    def test_single_line_convenience(self) -> None:
        from exo.observability import (  # pyright: ignore[reportMissingImports]
            configure,
            get_logger,
            traced,
        )

        assert callable(configure)
        assert callable(get_logger)
        assert callable(traced)

    def test_star_import_uses_all(self) -> None:
        """Verify that a star import would pull in __all__ symbols."""
        import exo.observability as obs  # pyright: ignore[reportMissingImports]

        # Python uses __all__ for `from module import *`
        # We verify __all__ is set and non-empty
        assert hasattr(obs, "__all__")
        assert len(obs.__all__) > 50  # Core logging, tracing, metrics, semconv, health, alerts
