"""Tests for ObservabilityConfig namespace grouping (Tier 3 DX).

Covers:
- grouped → flat population (LoggingConfig, TracingConfig, MetricsConfig)
- flat fields still work unchanged
- conflict raises ValueError
- read-accessor properties return correct grouped views
- sub-models exported from package __init__
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from exo.observability import (  # pyright: ignore[reportMissingImports]
    LoggingConfig as _LoggingConfigInit,
)
from exo.observability import (
    MetricsConfig as _MetricsConfigInit,
)
from exo.observability import (
    TracingConfig as _TracingConfigInit,
)
from exo.observability.config import (  # pyright: ignore[reportMissingImports]
    LoggingConfig,
    MetricsConfig,
    ObservabilityConfig,
    TraceBackend,
    TracingConfig,
    reset,
)
from exo.types import ExoError  # pyright: ignore[reportMissingImports]


@pytest.fixture(autouse=True)
def _reset_config() -> None:
    reset()


# ---------------------------------------------------------------------------
# Sub-model public exports
# ---------------------------------------------------------------------------


class TestSubmodelExports:
    def test_exported_from_package_init(self) -> None:
        assert _LoggingConfigInit is LoggingConfig
        assert _TracingConfigInit is TracingConfig
        assert _MetricsConfigInit is MetricsConfig


# ---------------------------------------------------------------------------
# LoggingConfig → flat population
# ---------------------------------------------------------------------------


class TestLoggingConfigNamespace:
    def test_grouped_object_populates_flat_fields(self) -> None:
        cfg = ObservabilityConfig(logging=LoggingConfig(level="DEBUG", format="json"))
        assert cfg.log_level == "DEBUG"
        assert cfg.log_format == "json"

    def test_grouped_dict_populates_flat_fields(self) -> None:
        cfg = ObservabilityConfig(logging={"level": "INFO", "format": "json"})
        assert cfg.log_level == "INFO"
        assert cfg.log_format == "json"

    def test_grouped_partial_uses_submodel_defaults(self) -> None:
        # Only override level; format should use LoggingConfig's default
        cfg = ObservabilityConfig(logging=LoggingConfig(level="ERROR"))
        assert cfg.log_level == "ERROR"
        assert cfg.log_format == "text"

    def test_flat_fields_unchanged_without_grouped(self) -> None:
        cfg = ObservabilityConfig(log_level="CRITICAL", log_format="json")
        assert cfg.log_level == "CRITICAL"
        assert cfg.log_format == "json"

    def test_conflict_with_log_level_raises(self) -> None:
        with pytest.raises(ExoError, match="log_level"):
            ObservabilityConfig(logging=LoggingConfig(level="DEBUG"), log_level="INFO")

    def test_conflict_with_log_format_raises(self) -> None:
        with pytest.raises(ExoError, match="log_format"):
            ObservabilityConfig(logging=LoggingConfig(), log_format="json")

    def test_invalid_grouped_type_raises(self) -> None:
        with pytest.raises(ExoError, match="LoggingConfig"):
            ObservabilityConfig(logging="bad-value")  # type: ignore[arg-type]

    def test_logging_property_returns_view(self) -> None:
        cfg = ObservabilityConfig(log_level="DEBUG", log_format="json")
        view = cfg.logging
        assert isinstance(view, LoggingConfig)
        assert view.level == "DEBUG"
        assert view.format == "json"

    def test_logging_property_default(self) -> None:
        cfg = ObservabilityConfig()
        view = cfg.logging
        assert view.level == "WARNING"
        assert view.format == "text"


# ---------------------------------------------------------------------------
# TracingConfig → flat population
# ---------------------------------------------------------------------------


class TestTracingConfigNamespace:
    def test_grouped_object_populates_flat_fields(self) -> None:
        cfg = ObservabilityConfig(
            tracing=TracingConfig(
                enabled=True,
                backend=TraceBackend.CONSOLE,
                endpoint="http://localhost:4318",
                sample_rate=0.5,
            )
        )
        assert cfg.trace_enabled is True
        assert cfg.trace_backend == TraceBackend.CONSOLE
        assert cfg.trace_endpoint == "http://localhost:4318"
        assert cfg.sample_rate == 0.5

    def test_grouped_dict_populates_flat_fields(self) -> None:
        cfg = ObservabilityConfig(
            tracing={"enabled": True, "backend": "console", "sample_rate": 0.25}
        )
        assert cfg.trace_enabled is True
        assert cfg.trace_backend == TraceBackend.CONSOLE
        assert cfg.sample_rate == 0.25

    def test_grouped_partial_uses_submodel_defaults(self) -> None:
        cfg = ObservabilityConfig(tracing=TracingConfig(enabled=True))
        assert cfg.trace_enabled is True
        assert cfg.trace_backend == TraceBackend.OTLP
        assert cfg.trace_endpoint is None
        assert cfg.sample_rate == 1.0

    def test_flat_fields_unchanged_without_grouped(self) -> None:
        cfg = ObservabilityConfig(trace_enabled=True, sample_rate=0.75)
        assert cfg.trace_enabled is True
        assert cfg.sample_rate == 0.75

    def test_conflict_with_trace_enabled_raises(self) -> None:
        with pytest.raises(ExoError, match="trace_enabled"):
            ObservabilityConfig(tracing=TracingConfig(enabled=True), trace_enabled=False)

    def test_conflict_with_sample_rate_raises(self) -> None:
        with pytest.raises(ExoError, match="sample_rate"):
            ObservabilityConfig(tracing=TracingConfig(), sample_rate=0.5)

    def test_invalid_grouped_type_raises(self) -> None:
        with pytest.raises(ExoError, match="TracingConfig"):
            ObservabilityConfig(tracing=42)  # type: ignore[arg-type]

    def test_tracing_property_returns_view(self) -> None:
        cfg = ObservabilityConfig(
            trace_enabled=True,
            trace_backend=TraceBackend.MEMORY,
            trace_endpoint="http://otel:4318",
            sample_rate=0.3,
        )
        view = cfg.tracing
        assert isinstance(view, TracingConfig)
        assert view.enabled is True
        assert view.backend == TraceBackend.MEMORY
        assert view.endpoint == "http://otel:4318"
        assert view.sample_rate == 0.3

    def test_tracing_property_default(self) -> None:
        cfg = ObservabilityConfig()
        view = cfg.tracing
        assert view.enabled is False
        assert view.backend == TraceBackend.OTLP
        assert view.endpoint is None
        assert view.sample_rate == 1.0

    def test_sample_rate_validation_still_enforced(self) -> None:
        with pytest.raises(ValidationError):
            ObservabilityConfig(tracing=TracingConfig(sample_rate=2.0))


# ---------------------------------------------------------------------------
# MetricsConfig → flat population
# ---------------------------------------------------------------------------


class TestMetricsConfigNamespace:
    def test_grouped_object_populates_flat_field(self) -> None:
        cfg = ObservabilityConfig(metrics=MetricsConfig(enabled=True))
        assert cfg.metrics_enabled is True

    def test_grouped_dict_populates_flat_field(self) -> None:
        cfg = ObservabilityConfig(metrics={"enabled": True})
        assert cfg.metrics_enabled is True

    def test_flat_field_unchanged_without_grouped(self) -> None:
        cfg = ObservabilityConfig(metrics_enabled=True)
        assert cfg.metrics_enabled is True

    def test_conflict_raises(self) -> None:
        with pytest.raises(ExoError, match="metrics_enabled"):
            ObservabilityConfig(metrics=MetricsConfig(enabled=True), metrics_enabled=False)

    def test_invalid_grouped_type_raises(self) -> None:
        with pytest.raises(ExoError, match="MetricsConfig"):
            ObservabilityConfig(metrics="bad")  # type: ignore[arg-type]

    def test_metrics_property_returns_view(self) -> None:
        cfg = ObservabilityConfig(metrics_enabled=True)
        view = cfg.metrics
        assert isinstance(view, MetricsConfig)
        assert view.enabled is True

    def test_metrics_property_default(self) -> None:
        cfg = ObservabilityConfig()
        view = cfg.metrics
        assert view.enabled is False


# ---------------------------------------------------------------------------
# Combined namespace + non-grouped fields work together
# ---------------------------------------------------------------------------


class TestCombinedNamespaces:
    def test_all_three_namespaces_combined(self) -> None:
        cfg = ObservabilityConfig(
            logging=LoggingConfig(level="DEBUG", format="json"),
            tracing=TracingConfig(enabled=True, sample_rate=0.5),
            metrics=MetricsConfig(enabled=True),
            service_name="my-service",
            namespace="myapp",
        )
        assert cfg.log_level == "DEBUG"
        assert cfg.log_format == "json"
        assert cfg.trace_enabled is True
        assert cfg.sample_rate == 0.5
        assert cfg.metrics_enabled is True
        assert cfg.service_name == "my-service"
        assert cfg.namespace == "myapp"

    def test_grouped_and_non_overlapping_flat_coexist(self) -> None:
        # tracing namespace + unrelated flat fields — no conflict
        cfg = ObservabilityConfig(
            tracing=TracingConfig(enabled=True),
            log_level="INFO",
            service_name="svc",
        )
        assert cfg.trace_enabled is True
        assert cfg.log_level == "INFO"
        assert cfg.service_name == "svc"

    def test_config_is_still_frozen(self) -> None:
        cfg = ObservabilityConfig(logging=LoggingConfig(level="INFO"))
        with pytest.raises(ValidationError):
            cfg.log_level = "DEBUG"  # type: ignore[misc]
