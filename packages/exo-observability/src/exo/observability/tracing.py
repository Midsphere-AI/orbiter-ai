"""@traced decorator and span context managers with optional OTel support.

When ``opentelemetry`` is installed, spans are created via the OTel SDK.
When it is **not** installed, all instrumentation becomes a lightweight no-op:
``@traced`` passes through, and ``span()``/``aspan()`` yield a ``NullSpan``.
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import logging
import sys
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional OTel import
# ---------------------------------------------------------------------------

try:
    from opentelemetry import trace as _otel_trace

    HAS_OTEL = True
except ImportError:
    _otel_trace = None  # type: ignore[assignment]
    HAS_OTEL = False


# ---------------------------------------------------------------------------
# NullSpan — stub when OTel is absent
# ---------------------------------------------------------------------------


@runtime_checkable
class SpanLike(Protocol):
    """Minimal span interface used by exo instrumentation."""

    def set_attribute(self, key: str, value: Any) -> None: ...
    def record_exception(self, exception: BaseException) -> None: ...
    def set_status(self, status: Any, description: str | None = None) -> None: ...


class NullSpan:
    """No-op span stub returned when OpenTelemetry is not installed."""

    def set_attribute(self, key: str, value: Any) -> None:
        pass

    def record_exception(self, exception: BaseException) -> None:
        """Log the exception so it is visible without OTel installed."""
        logger.warning(
            "exception recorded on NullSpan (OTel not installed): %s: %s",
            type(exception).__name__,
            exception,
            exc_info=exception,
        )

    def set_status(self, status: Any, description: str | None = None) -> None:
        pass

    def __enter__(self) -> NullSpan:
        return self

    def __exit__(self, *args: Any) -> None:
        pass


_NULL_SPAN = NullSpan()


def _get_tracer(name: str = "exo") -> Any:
    """Return an OTel tracer.  Only called when HAS_OTEL is True."""
    assert _otel_trace is not None
    return _otel_trace.get_tracer(name)


# ---------------------------------------------------------------------------
# User-code filtering
# ---------------------------------------------------------------------------

_NON_USER_PREFIXES: tuple[str, ...] = (
    str(Path(inspect.__file__).parent),
    str(Path(__file__).parent),
)


def is_user_code(filename: str) -> bool:
    """Return True if *filename* belongs to user code (not stdlib/observability)."""
    if filename.startswith("<"):
        return False
    abs_path = str(Path(filename).resolve())
    return not any(abs_path.startswith(p) for p in _NON_USER_PREFIXES)


def get_user_frame() -> inspect.FrameInfo | None:
    """Walk the call stack to find the first user-code frame."""
    frame = sys._getframe(1)
    while frame is not None:
        if is_user_code(frame.f_code.co_filename):
            return inspect.FrameInfo(
                frame,
                frame.f_code.co_filename,
                frame.f_lineno,
                frame.f_code.co_qualname,
                None,
                None,
            )
        frame = frame.f_back
    return None


# ---------------------------------------------------------------------------
# Function metadata extraction
# ---------------------------------------------------------------------------


def extract_metadata(func: Any) -> dict[str, Any]:
    """Extract tracing metadata from a callable."""
    attrs: dict[str, Any] = {}
    code = getattr(func, "__code__", None)
    attrs["code.function"] = getattr(func, "__qualname__", getattr(func, "__name__", str(func)))
    attrs["code.module"] = getattr(func, "__module__", "")
    if code is not None:
        attrs["code.lineno"] = code.co_firstlineno
        filepath = code.co_filename
        try:
            attrs["code.filepath"] = str(Path(filepath).relative_to(Path.cwd()))
        except ValueError:
            attrs["code.filepath"] = filepath
    try:
        sig = inspect.signature(func)
        attrs["code.parameters"] = [p for p in sig.parameters if p != "self"]
    except (ValueError, TypeError):
        attrs["code.parameters"] = []
    return attrs


# ---------------------------------------------------------------------------
# Span context managers
# ---------------------------------------------------------------------------


@contextmanager
def span(name: str, attributes: dict[str, Any] | None = None) -> Iterator[Any]:
    """Synchronous span context manager.

    Yields a real OTel span when available, otherwise a :class:`NullSpan`.
    """
    if HAS_OTEL:
        tracer = _get_tracer()
        with tracer.start_as_current_span(name, attributes=attributes or {}) as s:
            yield s
    else:
        yield _NULL_SPAN


@asynccontextmanager
async def aspan(name: str, attributes: dict[str, Any] | None = None) -> AsyncIterator[Any]:
    """Asynchronous span context manager.

    Yields a real OTel span when available, otherwise a :class:`NullSpan`.
    """
    if HAS_OTEL:
        tracer = _get_tracer()
        with tracer.start_as_current_span(name, attributes=attributes or {}) as s:
            yield s
    else:
        yield _NULL_SPAN


# ---------------------------------------------------------------------------
# @traced decorator
# ---------------------------------------------------------------------------


def traced(
    name: str | None = None,
    *,
    attributes: dict[str, Any] | None = None,
    extract_args: bool = False,
) -> Any:
    """Decorator that wraps a function in a span.

    Supports sync functions, async functions, sync generators, and async
    generators.  Metadata (qualname, module, line number, parameters) is
    automatically recorded as span attributes.

    When OTel is not installed the decorator is a lightweight passthrough
    that still preserves ``functools.wraps`` metadata.

    Args:
        name: Span name override (defaults to ``func.__qualname__``).
        attributes: Extra attributes merged onto the span.
        extract_args: When *True*, record the function's call arguments.
    """

    def decorator(func: Any) -> Any:
        if not HAS_OTEL:
            return func

        meta = extract_metadata(func)
        span_name = name or meta["code.function"]

        def _build_attrs(args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
            merged = dict(meta)
            if attributes:
                merged.update(attributes)
            if extract_args:
                try:
                    bound = inspect.signature(func).bind(*args, **kwargs)
                    bound.apply_defaults()
                    for k, v in bound.arguments.items():
                        if k != "self":
                            merged[f"arg.{k}"] = str(v)
                except TypeError as exc:
                    logger.debug(
                        "traced: argument inspection failed for %r: %s",
                        getattr(func, "__qualname__", func),
                        exc,
                    )
            # Flatten list values to strings for OTel compatibility.
            return {k: (str(v) if isinstance(v, list) else v) for k, v in merged.items()}

        if inspect.isasyncgenfunction(func):

            @functools.wraps(func)
            async def async_gen_wrapper(*args: Any, **kwargs: Any) -> Any:
                tracer = _get_tracer()
                with tracer.start_as_current_span(
                    span_name, attributes=_build_attrs(args, kwargs)
                ) as s:
                    try:
                        async for item in func(*args, **kwargs):
                            yield item
                    except BaseException as exc:
                        s.record_exception(exc)
                        raise

            return async_gen_wrapper

        if inspect.isgeneratorfunction(func):

            @functools.wraps(func)
            def gen_wrapper(*args: Any, **kwargs: Any) -> Any:
                tracer = _get_tracer()
                with tracer.start_as_current_span(
                    span_name, attributes=_build_attrs(args, kwargs)
                ) as s:
                    try:
                        yield from func(*args, **kwargs)
                    except BaseException as exc:
                        s.record_exception(exc)
                        raise

            return gen_wrapper

        if asyncio.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                tracer = _get_tracer()
                with tracer.start_as_current_span(
                    span_name, attributes=_build_attrs(args, kwargs)
                ) as s:
                    try:
                        return await func(*args, **kwargs)
                    except BaseException as exc:
                        s.record_exception(exc)
                        raise

            return async_wrapper

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            tracer = _get_tracer()
            with tracer.start_as_current_span(
                span_name, attributes=_build_attrs(args, kwargs)
            ) as s:
                try:
                    return func(*args, **kwargs)
                except BaseException as exc:
                    s.record_exception(exc)
                    raise

        return sync_wrapper

    return decorator
