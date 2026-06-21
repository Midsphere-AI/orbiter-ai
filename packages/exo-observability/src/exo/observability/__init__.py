"""Exo Observability: structured logging, tracing, metrics, health checks.

Convenience imports -- the most common symbols are available directly::

    from exo.observability import get_logger, configure, traced, span

All other symbols are loaded lazily on first access so that importing this
package does not pull in OTel or other heavy sub-modules until needed.
Sub-module imports always work::

    from exo.observability.metrics import MetricsCollector
"""

from __future__ import annotations

__version__: str = "0.1.0"

# ---------------------------------------------------------------------------
# Eager imports: core symbols that cover 90 % of use-cases
# ---------------------------------------------------------------------------
from exo.observability.config import (  # pyright: ignore[reportMissingImports]
    ObservabilityConfig,
    TraceBackend,
    configure,
    get_config,
)
from exo.observability.logging import (  # pyright: ignore[reportMissingImports]
    LogContext,
    configure_logging,
    get_logger,
)
from exo.observability.tracing import (  # pyright: ignore[reportMissingImports]
    aspan,
    span,
    traced,
)

# ---------------------------------------------------------------------------
# Lazy-loaded module mapping: attribute name -> (module_path, symbol_name)
# ---------------------------------------------------------------------------
_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    # config (reset/get_config already eager above)
    "reset": ("exo.observability.config", "reset"),
    # logging extras
    "TextFormatter": ("exo.observability.logging", "TextFormatter"),
    "JsonFormatter": ("exo.observability.logging", "JsonFormatter"),
    "reset_logging": ("exo.observability.logging", "reset_logging"),
    # tracing extras
    "SpanLike": ("exo.observability.tracing", "SpanLike"),
    "NullSpan": ("exo.observability.tracing", "NullSpan"),
    "is_user_code": ("exo.observability.tracing", "is_user_code"),
    "get_user_frame": ("exo.observability.tracing", "get_user_frame"),
    "extract_metadata": ("exo.observability.tracing", "extract_metadata"),
    # metrics
    "MetricsCollector": ("exo.observability.metrics", "MetricsCollector"),
    "METRIC_AGENT_RUN_DURATION": (
        "exo.observability.metrics",
        "METRIC_AGENT_RUN_DURATION",
    ),
    "METRIC_AGENT_RUN_COUNTER": (
        "exo.observability.metrics",
        "METRIC_AGENT_RUN_COUNTER",
    ),
    "METRIC_AGENT_TOKEN_USAGE": (
        "exo.observability.metrics",
        "METRIC_AGENT_TOKEN_USAGE",
    ),
    "METRIC_TOOL_STEP_DURATION": (
        "exo.observability.metrics",
        "METRIC_TOOL_STEP_DURATION",
    ),
    "METRIC_TOOL_STEP_COUNTER": (
        "exo.observability.metrics",
        "METRIC_TOOL_STEP_COUNTER",
    ),
    "get_collector": ("exo.observability.metrics", "get_collector"),
    "get_metrics_snapshot": (
        "exo.observability.metrics",
        "get_metrics_snapshot",
    ),
    "reset_metrics": ("exo.observability.metrics", "reset_metrics"),
    "create_counter": ("exo.observability.metrics", "create_counter"),
    "create_histogram": ("exo.observability.metrics", "create_histogram"),
    "create_gauge": ("exo.observability.metrics", "create_gauge"),
    "build_agent_attributes": (
        "exo.observability.metrics",
        "build_agent_attributes",
    ),
    "build_tool_attributes": (
        "exo.observability.metrics",
        "build_tool_attributes",
    ),
    "record_agent_run": ("exo.observability.metrics", "record_agent_run"),
    "record_tool_step": ("exo.observability.metrics", "record_tool_step"),
    "Timer": ("exo.observability.metrics", "Timer"),
    "timer": ("exo.observability.metrics", "timer"),
    # health
    "HealthStatus": ("exo.observability.health", "HealthStatus"),
    "HealthResult": ("exo.observability.health", "HealthResult"),
    "HealthCheck": ("exo.observability.health", "HealthCheck"),
    "MemoryUsageCheck": ("exo.observability.health", "MemoryUsageCheck"),
    "HealthRegistry": ("exo.observability.health", "HealthRegistry"),
    "get_registry": ("exo.observability.health", "get_registry"),
    "get_health_summary": (
        "exo.observability.health",
        "get_health_summary",
    ),
}

# Semconv constants — all lazy-loaded from semconv module
_SEMCONV_NAMES: list[str] = [
    "GEN_AI_SYSTEM",
    "GEN_AI_REQUEST_MODEL",
    "GEN_AI_REQUEST_MAX_TOKENS",
    "GEN_AI_REQUEST_TEMPERATURE",
    "GEN_AI_REQUEST_TOP_P",
    "GEN_AI_REQUEST_TOP_K",
    "GEN_AI_REQUEST_FREQUENCY_PENALTY",
    "GEN_AI_REQUEST_PRESENCE_PENALTY",
    "GEN_AI_REQUEST_STOP_SEQUENCES",
    "GEN_AI_REQUEST_STREAMING",
    "GEN_AI_PROMPT",
    "GEN_AI_COMPLETION",
    "GEN_AI_DURATION",
    "GEN_AI_RESPONSE_FINISH_REASONS",
    "GEN_AI_RESPONSE_ID",
    "GEN_AI_RESPONSE_MODEL",
    "GEN_AI_USAGE_INPUT_TOKENS",
    "GEN_AI_USAGE_OUTPUT_TOKENS",
    "GEN_AI_USAGE_TOTAL_TOKENS",
    "GEN_AI_OPERATION_NAME",
    "GEN_AI_SERVER_ADDRESS",
    "AGENT_ID",
    "AGENT_NAME",
    "AGENT_TYPE",
    "AGENT_MODEL",
    "AGENT_STEP",
    "AGENT_MAX_STEPS",
    "AGENT_RUN_SUCCESS",
    "TOOL_NAME",
    "TOOL_CALL_ID",
    "TOOL_ARGUMENTS",
    "TOOL_RESULT",
    "TOOL_ERROR",
    "TOOL_DURATION",
    "TOOL_STEP_SUCCESS",
    "TASK_ID",
    "TASK_INPUT",
    "SESSION_ID",
    "USER_ID",
    "TRACE_ID",
    "COST_INPUT_TOKENS",
    "COST_OUTPUT_TOKENS",
    "COST_TOTAL_USD",
    "SPAN_PREFIX_AGENT",
    "SPAN_PREFIX_TOOL",
    "SPAN_PREFIX_LLM",
    "SPAN_PREFIX_TASK",
    "METRIC_STREAM_EVENTS_EMITTED",
    "STREAM_EVENT_TYPE",
]
for _name in _SEMCONV_NAMES:
    _LAZY_IMPORTS[_name] = ("exo.observability.semconv", _name)

# ---------------------------------------------------------------------------
# __all__ — everything available from this package
# ---------------------------------------------------------------------------
__all__: list[str] = [
    "__version__",
    # config (eager)
    "ObservabilityConfig",
    "TraceBackend",
    "configure",
    "get_config",
    # logging (eager)
    "get_logger",
    "configure_logging",
    "LogContext",
    # tracing (eager)
    "traced",
    "span",
    "aspan",
    # lazy symbols
    *_LAZY_IMPORTS,
]


def __getattr__(name: str) -> object:
    """Lazy-load symbols on first access."""
    if name in _LAZY_IMPORTS:
        module_path, symbol_name = _LAZY_IMPORTS[name]
        import importlib

        module = importlib.import_module(module_path)
        value = getattr(module, symbol_name)
        # Cache on this module so __getattr__ is only called once per symbol
        globals()[name] = value
        return value
    msg = f"module 'exo.observability' has no attribute {name!r}"
    raise AttributeError(msg)
