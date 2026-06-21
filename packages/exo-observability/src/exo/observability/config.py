"""Observability configuration and top-level initialization."""

from __future__ import annotations

import logging
import threading
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

logger = logging.getLogger(__name__)


class TraceBackend(StrEnum):
    """Supported trace export backends."""

    OTLP = "otlp"
    MEMORY = "memory"
    CONSOLE = "console"


# ---------------------------------------------------------------------------
# Grouped namespace sub-models
# ---------------------------------------------------------------------------


class LoggingConfig(BaseModel):
    """Grouped logging configuration namespace.

    Args:
        level: Root log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        format: Log output format: ``'text'`` (ANSI) or ``'json'``.
    """

    model_config = ConfigDict(frozen=True)

    level: str = Field(
        default="WARNING",
        description="Root log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    )
    format: str = Field(
        default="text",
        description="Log output format: 'text' (ANSI) or 'json'",
        pattern=r"^(text|json)$",
    )


class TracingConfig(BaseModel):
    """Grouped tracing configuration namespace.

    Args:
        enabled: Enable distributed tracing.
        backend: Trace export backend.
        endpoint: OTLP collector endpoint (e.g. ``http://localhost:4318``).
        sample_rate: Trace sampling probability (0.0 = none, 1.0 = all).
    """

    model_config = ConfigDict(frozen=True)

    enabled: bool = Field(default=False, description="Enable distributed tracing")
    backend: TraceBackend = Field(
        default=TraceBackend.OTLP,
        description="Trace export backend",
    )
    endpoint: str | None = Field(
        default=None,
        description="OTLP collector endpoint (e.g. http://localhost:4318)",
    )
    sample_rate: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Trace sampling probability (0.0 = none, 1.0 = all)",
    )


class MetricsConfig(BaseModel):
    """Grouped metrics configuration namespace.

    Args:
        enabled: Enable metrics collection.
    """

    model_config = ConfigDict(frozen=True)

    enabled: bool = Field(default=False, description="Enable metrics collection")


# ---------------------------------------------------------------------------
# Main config
# ---------------------------------------------------------------------------


class ObservabilityConfig(BaseModel):
    model_config = ConfigDict(frozen=True)
    """Immutable configuration for the unified observability layer."""

    # Logging
    log_level: str = Field(
        default="WARNING",
        description="Root log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    )
    log_format: str = Field(
        default="text",
        description="Log output format: 'text' (ANSI) or 'json'",
        pattern=r"^(text|json)$",
    )

    # Tracing
    trace_enabled: bool = Field(
        default=False,
        description="Enable distributed tracing",
    )
    trace_backend: TraceBackend = Field(
        default=TraceBackend.OTLP,
        description="Trace export backend",
    )
    trace_endpoint: str | None = Field(
        default=None,
        description="OTLP collector endpoint (e.g. http://localhost:4318)",
    )

    # Service identity
    service_name: str = Field(
        default="exo",
        description="Service name for spans and metrics",
    )
    sample_rate: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Trace sampling probability (0.0 = none, 1.0 = all)",
    )

    # Metrics
    metrics_enabled: bool = Field(
        default=False,
        description="Enable metrics collection",
    )

    # Namespace
    namespace: str = Field(
        default="exo",
        description="Attribute namespace prefix (e.g. exo.agent.name)",
    )

    @model_validator(mode="before")
    @classmethod
    def _explode_namespace_configs(cls, data: Any) -> Any:
        """Explode grouped namespace configs (logging/tracing/metrics) into flat fields.

        Grouped params and their overlapping flat fields must not both be set —
        a clear ValueError is raised when they conflict.
        """
        if not isinstance(data, dict):
            return data

        data = dict(data)

        # --- logging namespace ---
        logging_cfg = data.pop("logging", None)
        if logging_cfg is not None:
            if isinstance(logging_cfg, LoggingConfig):
                lc = logging_cfg
            elif isinstance(logging_cfg, dict):
                lc = LoggingConfig(**logging_cfg)
            else:
                raise ValueError(
                    f"'logging' must be a LoggingConfig or dict, got {type(logging_cfg)}"
                )

            flat_conflicts = [f for f in ("log_level", "log_format") if f in data]
            if flat_conflicts:
                raise ValueError(
                    f"Provide either 'logging=LoggingConfig(...)' or the flat fields "
                    f"{flat_conflicts!r}, not both."
                )
            data["log_level"] = lc.level
            data["log_format"] = lc.format

        # --- tracing namespace ---
        tracing_cfg = data.pop("tracing", None)
        if tracing_cfg is not None:
            if isinstance(tracing_cfg, TracingConfig):
                tc = tracing_cfg
            elif isinstance(tracing_cfg, dict):
                tc = TracingConfig(**tracing_cfg)
            else:
                raise ValueError(
                    f"'tracing' must be a TracingConfig or dict, got {type(tracing_cfg)}"
                )

            flat_conflicts = [
                f
                for f in ("trace_enabled", "trace_backend", "trace_endpoint", "sample_rate")
                if f in data
            ]
            if flat_conflicts:
                raise ValueError(
                    f"Provide either 'tracing=TracingConfig(...)' or the flat fields "
                    f"{flat_conflicts!r}, not both."
                )
            data["trace_enabled"] = tc.enabled
            data["trace_backend"] = tc.backend
            data["trace_endpoint"] = tc.endpoint
            data["sample_rate"] = tc.sample_rate

        # --- metrics namespace ---
        metrics_cfg = data.pop("metrics", None)
        if metrics_cfg is not None:
            if isinstance(metrics_cfg, MetricsConfig):
                mc = metrics_cfg
            elif isinstance(metrics_cfg, dict):
                mc = MetricsConfig(**metrics_cfg)
            else:
                raise ValueError(
                    f"'metrics' must be a MetricsConfig or dict, got {type(metrics_cfg)}"
                )

            flat_conflicts = [f for f in ("metrics_enabled",) if f in data]
            if flat_conflicts:
                raise ValueError(
                    f"Provide either 'metrics=MetricsConfig(...)' or the flat field "
                    f"{flat_conflicts!r}, not both."
                )
            data["metrics_enabled"] = mc.enabled

        return data

    # -----------------------------------------------------------------------
    # Read-accessor views (grouped properties)
    # -----------------------------------------------------------------------

    @property
    def logging(self) -> LoggingConfig:
        """Return a grouped view of the logging settings."""
        return LoggingConfig(level=self.log_level, format=self.log_format)

    @property
    def tracing(self) -> TracingConfig:
        """Return a grouped view of the tracing settings."""
        return TracingConfig(
            enabled=self.trace_enabled,
            backend=self.trace_backend,
            endpoint=self.trace_endpoint,
            sample_rate=self.sample_rate,
        )

    @property
    def metrics(self) -> MetricsConfig:
        """Return a grouped view of the metrics settings."""
        return MetricsConfig(enabled=self.metrics_enabled)


# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

_lock = threading.Lock()
_configured = False
_current_config: ObservabilityConfig | None = None


def configure(config: ObservabilityConfig | None = None, **kwargs: object) -> ObservabilityConfig:
    """Initialize the observability subsystem.

    Idempotent: calling again after the first time is a no-op and returns
    the existing config. Pass ``force=True`` as a kwarg to re-initialize.

    Args:
        config: An explicit config object. If None, one is built from kwargs.
        **kwargs: Forwarded to ``ObservabilityConfig(**kwargs)`` when *config*
                  is not provided. Also accepts ``force=True`` to allow
                  re-initialization.

    Returns:
        The active ObservabilityConfig.
    """
    global _configured, _current_config
    force = bool(kwargs.pop("force", False))

    with _lock:
        if _configured and not force:
            assert _current_config is not None
            logger.debug("observability already configured, returning existing config")
            return _current_config

        if config is None:
            config = ObservabilityConfig(**kwargs)  # type: ignore[arg-type]

        _current_config = config
        _configured = True
        logger.debug(
            "observability configured: log_level=%s, trace_enabled=%s, service=%s",
            config.log_level,
            config.trace_enabled,
            config.service_name,
        )
        return config


def get_config() -> ObservabilityConfig:
    """Return the current config, or a default if not yet configured."""
    if _current_config is not None:
        return _current_config
    return ObservabilityConfig()


def reset() -> None:
    """Reset global state — for testing only."""
    global _configured, _current_config
    with _lock:
        _configured = False
        _current_config = None
