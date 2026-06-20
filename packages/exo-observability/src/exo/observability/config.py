"""Observability configuration and top-level initialization."""

from __future__ import annotations

import logging
import threading
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)


class TraceBackend(StrEnum):
    """Supported trace export backends."""

    OTLP = "otlp"
    MEMORY = "memory"
    CONSOLE = "console"


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
