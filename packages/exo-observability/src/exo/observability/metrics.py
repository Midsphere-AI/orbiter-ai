"""Metrics collection with optional OTel support.

When ``opentelemetry`` is installed, metrics are recorded via the OTel SDK.
When it is **not** installed, metrics are collected in-memory via
:class:`MetricsCollector` for inspection and testing.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from exo.observability.semconv import (  # pyright: ignore[reportMissingImports]
    AGENT_NAME,
    AGENT_RUN_SUCCESS,
    AGENT_STEP,
    GEN_AI_USAGE_INPUT_TOKENS,
    GEN_AI_USAGE_OUTPUT_TOKENS,
    GEN_AI_USAGE_TOTAL_TOKENS,
    SESSION_ID,
    TASK_ID,
    TOOL_NAME,
    TOOL_STEP_SUCCESS,
    USER_ID,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional OTel import
# ---------------------------------------------------------------------------

try:
    from opentelemetry import metrics as _otel_metrics

    HAS_OTEL = True
except ImportError:
    _otel_metrics = None  # type: ignore[assignment]
    HAS_OTEL = False

# ---------------------------------------------------------------------------
# Metric name constants
# ---------------------------------------------------------------------------

METRIC_AGENT_RUN_DURATION = "agent_run_duration"
METRIC_AGENT_RUN_COUNTER = "agent_run_counter"
METRIC_AGENT_TOKEN_USAGE = "agent_token_usage"
METRIC_TOOL_STEP_DURATION = "tool_step_duration"
METRIC_TOOL_STEP_COUNTER = "tool_step_counter"


# ---------------------------------------------------------------------------
# In-memory fallback: MetricsCollector
# ---------------------------------------------------------------------------


class MetricsCollector:
    """Thread-safe in-memory metric storage for use when OTel is not installed.

    Stores counters, histograms, and gauges as simple dict-based accumulators.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: dict[str, float] = {}
        self._histograms: dict[str, list[tuple[float, dict[str, Any]]]] = {}
        self._gauges: dict[str, float] = {}

    def add_counter(
        self, name: str, value: float = 1.0, attributes: dict[str, Any] | None = None
    ) -> None:
        """Increment a counter by *value*."""
        _ = attributes  # stored in snapshot via histograms; counters aggregate
        with self._lock:
            self._counters[name] = self._counters.get(name, 0.0) + value

    def record_histogram(
        self, name: str, value: float, attributes: dict[str, Any] | None = None
    ) -> None:
        """Record a histogram observation."""
        with self._lock:
            self._histograms.setdefault(name, []).append(
                (value, dict(attributes) if attributes else {})
            )

    def set_gauge(self, name: str, value: float, attributes: dict[str, Any] | None = None) -> None:
        """Set a gauge to *value*."""
        _ = attributes
        with self._lock:
            self._gauges[name] = value

    def get_snapshot(self) -> dict[str, Any]:
        """Return a JSON-serializable snapshot of all collected metrics."""
        with self._lock:
            return {
                "counters": dict(self._counters),
                "histograms": {
                    k: [{"value": v, "attributes": a} for v, a in vs]
                    for k, vs in self._histograms.items()
                },
                "gauges": dict(self._gauges),
            }

    def reset(self) -> None:
        """Clear all collected metrics."""
        with self._lock:
            self._counters.clear()
            self._histograms.clear()
            self._gauges.clear()


# Singleton in-memory collector (used when OTel is absent).
_collector = MetricsCollector()


def get_collector() -> MetricsCollector:
    """Return the global in-memory metrics collector."""
    return _collector


def get_metrics_snapshot() -> dict[str, Any]:
    """Return a snapshot of in-memory metrics (useful for tests and health checks)."""
    return _collector.get_snapshot()


def reset_metrics() -> None:
    """Reset the in-memory metrics collector — for testing only."""
    _collector.reset()


# ---------------------------------------------------------------------------
# Instrument factories (OTel path)
# ---------------------------------------------------------------------------


def _get_meter() -> Any:
    """Get the exo meter from the current global MeterProvider."""
    assert _otel_metrics is not None
    return _otel_metrics.get_meter("exo")


def create_counter(name: str, unit: str = "1", description: str = "") -> Any:
    """Create an OTel counter instrument (requires OTel installed)."""
    return _get_meter().create_counter(name=name, unit=unit, description=description)


def create_histogram(name: str, unit: str = "1", description: str = "") -> Any:
    """Create an OTel histogram instrument (requires OTel installed)."""
    return _get_meter().create_histogram(name=name, unit=unit, description=description)


def create_gauge(name: str, unit: str = "1", description: str = "") -> Any:
    """Create an OTel gauge instrument (requires OTel installed)."""
    return _get_meter().create_up_down_counter(name=name, unit=unit, description=description)


# ---------------------------------------------------------------------------
# Attribute builders
# ---------------------------------------------------------------------------


def _safe_str(value: Any) -> str:
    """Convert a value to string, returning empty string for None."""
    return str(value) if value is not None else ""


def build_agent_attributes(
    *,
    agent_name: str,
    task_id: str = "",
    session_id: str = "",
    user_id: str = "",
    step: int | None = None,
) -> dict[str, str | int]:
    """Build attribute dict for agent metrics."""
    attrs: dict[str, str | int] = {
        AGENT_NAME: _safe_str(agent_name),
        TASK_ID: _safe_str(task_id),
        SESSION_ID: _safe_str(session_id),
        USER_ID: _safe_str(user_id),
    }
    if step is not None:
        attrs[AGENT_STEP] = step
    return attrs


def build_tool_attributes(
    *,
    tool_name: str,
    agent_name: str = "",
    task_id: str = "",
) -> dict[str, str]:
    """Build attribute dict for tool metrics."""
    return {
        TOOL_NAME: _safe_str(tool_name),
        AGENT_NAME: _safe_str(agent_name),
        TASK_ID: _safe_str(task_id),
    }


# ---------------------------------------------------------------------------
# OTel instrument cache — create once per meter instance, reuse on every call
# ---------------------------------------------------------------------------

# Keyed by (meter_id, instrument_name) so the cache is invalidated automatically
# when the global MeterProvider changes (e.g. between tests).
_otel_counters: dict[tuple[int, str], Any] = {}
_otel_histograms: dict[tuple[int, str], Any] = {}


def _get_counter(name: str) -> Any:
    """Return a cached OTel counter for the current meter, creating it on first use."""
    meter = _get_meter()
    # Use the meter's MeterProvider identity as part of the key so the cache is
    # invalidated automatically when the global MeterProvider is replaced.
    provider = getattr(meter, "_meter_provider", meter)
    key = (id(provider), name)
    if key not in _otel_counters:
        _otel_counters[key] = meter.create_counter(name=name)
    return _otel_counters[key]


def _get_histogram(name: str) -> Any:
    """Return a cached OTel histogram for the current meter, creating it on first use."""
    meter = _get_meter()
    provider = getattr(meter, "_meter_provider", meter)
    key = (id(provider), name)
    if key not in _otel_histograms:
        _otel_histograms[key] = meter.create_histogram(name=name)
    return _otel_histograms[key]


# ---------------------------------------------------------------------------
# Branching helpers — encapsulate OTel vs in-memory dispatch
# ---------------------------------------------------------------------------


def _increment_counter(name: str, attrs: dict[str, Any], *, value: float = 1.0) -> None:
    """Increment a counter via OTel or the in-memory fallback."""
    if HAS_OTEL:
        _get_counter(name).add(value, attrs)
    else:
        _collector.add_counter(name, value, attrs)


def _record_histogram_value(name: str, value: float, attrs: dict[str, Any]) -> None:
    """Record a histogram observation via OTel or the in-memory fallback."""
    if HAS_OTEL:
        _get_histogram(name).record(value, attrs)
    else:
        _collector.record_histogram(name, value, attrs)


# ---------------------------------------------------------------------------
# Recording helpers
# ---------------------------------------------------------------------------


def record_agent_run(
    *,
    duration: float,
    success: bool,
    attributes: dict[str, Any] | None = None,
    input_tokens: int = 0,
    output_tokens: int = 0,
) -> None:
    """Record agent run metrics (duration, counter, token usage).

    Works with both OTel and the in-memory fallback.
    """
    attrs = dict(attributes) if attributes else {}
    attrs[AGENT_RUN_SUCCESS] = "1" if success else "0"
    logger.debug(
        "record_agent_run: duration=%.3fs success=%s tokens=%d+%d",
        duration,
        success,
        input_tokens,
        output_tokens,
    )

    _record_histogram_value(METRIC_AGENT_RUN_DURATION, duration, attrs)
    _increment_counter(METRIC_AGENT_RUN_COUNTER, attrs)
    total_tokens = input_tokens + output_tokens
    if total_tokens > 0:
        token_attrs = dict(attrs)
        token_attrs[GEN_AI_USAGE_INPUT_TOKENS] = input_tokens
        token_attrs[GEN_AI_USAGE_OUTPUT_TOKENS] = output_tokens
        token_attrs[GEN_AI_USAGE_TOTAL_TOKENS] = total_tokens
        _record_histogram_value(METRIC_AGENT_TOKEN_USAGE, float(total_tokens), token_attrs)


def record_tool_step(
    *,
    duration: float,
    success: bool,
    attributes: dict[str, Any] | None = None,
) -> None:
    """Record tool step metrics (duration, counter).

    Works with both OTel and the in-memory fallback.
    """
    attrs = dict(attributes) if attributes else {}
    attrs[TOOL_STEP_SUCCESS] = "1" if success else "0"
    logger.debug("record_tool_step: duration=%.3fs success=%s", duration, success)

    _record_histogram_value(METRIC_TOOL_STEP_DURATION, duration, attrs)
    _increment_counter(METRIC_TOOL_STEP_COUNTER, attrs)


# ---------------------------------------------------------------------------
# Timer
# ---------------------------------------------------------------------------


class Timer:
    """Simple timer for measuring durations.

    Can be used as a context manager::

        with Timer() as t:
            do_something()
        print(t.elapsed)
    """

    __slots__ = ("_elapsed", "_start")

    def __init__(self) -> None:
        self._start: float = 0.0
        self._elapsed: float = 0.0

    def start(self) -> Timer:
        """Start the timer. Returns self for chaining."""
        self._start = time.monotonic()
        return self

    def stop(self) -> float:
        """Stop the timer and return elapsed seconds."""
        self._elapsed = time.monotonic() - self._start
        return self._elapsed

    @property
    def elapsed(self) -> float:
        """Return the last recorded elapsed time."""
        return self._elapsed

    def __enter__(self) -> Timer:
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop()


@contextmanager
def timer() -> Iterator[Timer]:
    """Context manager that yields a started Timer."""
    t = Timer()
    t.start()
    yield t
    t.stop()
