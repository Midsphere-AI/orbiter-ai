"""Tracing backend registry + spec resolution (with env auto-detect).

``resolve_backends(spec)`` turns a user-facing spec into a list of configured
:class:`TracingBackend` objects:

* ``"off"`` / ``False`` -> ``[]`` (hard disable, beats auto-detect)
* ``"langfuse"`` / ``["langfuse", "phoenix"]`` -> those named backends
* a :class:`TracingConfig` -> its configured backend
* ``None`` / ``True`` -> **env auto-detect** (the on-by-default path): honor
  ``EXO_TRACING`` if set, else enable a vendor whose API keys are present in the
  environment *and* whose optional extra is importable.

Vendor backends (langfuse/langsmith/phoenix/braintrust) live in sibling modules
and register themselves on import; ``_ensure_loaded`` imports them lazily so a
missing extra degrades to a skipped backend instead of an ImportError.
"""

from __future__ import annotations

import importlib
import logging
import os
import threading
from collections.abc import Callable
from typing import Any

from exo.observability.backends.base import (  # pyright: ignore[reportMissingImports]
    MemoryBackend,
    OTLPBackend,
    TracingBackend,
)
from exo.observability.config import (  # pyright: ignore[reportMissingImports]
    TraceBackend,
    TracingConfig,
)
from exo.types import ExoError  # pyright: ignore[reportMissingImports]

_log = logging.getLogger(__name__)

# name -> factory returning a configured TracingBackend
_REGISTRY: dict[str, Callable[[], TracingBackend]] = {}

# Env probes for auto-detect: backend name -> (env keys to look for).
# A backend auto-enables only if at least one of its keys is set AND the backend
# can be loaded + is_available(). Order is the auto-enable priority.
_AUTO_DETECT: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("langfuse", ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY")),
    ("langsmith", ("LANGSMITH_API_KEY", "LANGCHAIN_API_KEY")),
    ("phoenix", ("PHOENIX_API_KEY", "PHOENIX_COLLECTOR_ENDPOINT")),
    ("braintrust", ("BRAINTRUST_API_KEY",)),
    ("otlp", ("OTEL_EXPORTER_OTLP_ENDPOINT",)),
)

# One-shot guard so auto-enable only logs an info line the first time.
# Protected by a lock so concurrent resolve_backends() calls don't race.
_announced = False
_announced_lock = threading.Lock()


def register(name: str, factory: Callable[[], TracingBackend]) -> None:
    """Register a backend factory under *name* (idempotent — last wins)."""
    _REGISTRY[name] = factory


def _ensure_loaded(name: str) -> bool:
    """Ensure the module that registers *name* is imported.

    Returns True if *name* is registered afterwards. Vendor modules register
    themselves on import; a missing optional extra is swallowed so callers see a
    skipped backend rather than an ImportError.
    """
    if name in _REGISTRY:
        return True
    try:
        importlib.import_module(f"exo.observability.backends.{name}")
    except ImportError as exc:
        _log.debug("tracing backend %r unavailable: %s", name, exc)
    return name in _REGISTRY


def _build(name: str) -> TracingBackend | None:
    if not _ensure_loaded(name):
        return None
    backend = _REGISTRY[name]()
    if not backend.is_available():
        _log.debug("tracing backend %r loaded but not available (missing deps/config)", name)
        return None
    return backend


def _announce(name: str) -> None:
    global _announced
    with _announced_lock:
        if _announced:
            return
        _announced = True
    _log.info(
        "exo: tracing auto-enabled via %s (set EXO_TRACING=off to disable)",
        name,
    )


def _auto_detect() -> list[TracingBackend]:
    """Resolve backends from the environment (the on-by-default path)."""
    env_spec = os.environ.get("EXO_TRACING")
    if env_spec is not None:
        return _resolve_names(_split(env_spec), announce=True)

    backends: list[TracingBackend] = []
    for name, keys in _AUTO_DETECT:
        if not any(os.environ.get(k) for k in keys):
            continue
        backend = _build(name)
        if backend is not None:
            _announce(name)
            backends.append(backend)
            break  # one auto-detected backend is enough
    return backends


def _split(spec: str) -> list[str]:
    return [part.strip() for part in spec.split(",") if part.strip()]


def _resolve_names(names: list[str], *, announce: bool = False) -> list[TracingBackend]:
    if any(n.lower() == "off" for n in names):
        return []
    backends: list[TracingBackend] = []
    for name in names:
        backend = _build(name)
        if backend is not None:
            if announce:
                _announce(name)
            backends.append(backend)
        else:
            _log.warning("tracing backend %r could not be enabled (skipped)", name)
    return backends


def resolve_backends(
    spec: str | list[str] | bool | TracingConfig | None = None,
) -> list[TracingBackend]:
    """Resolve a tracing *spec* into configured backends. See module docstring."""
    if spec is False or spec == "off":
        return []
    if spec is None or spec is True:
        return _auto_detect()
    if isinstance(spec, TracingConfig):
        cfg: Any = spec  # TracingConfig is namespace-imported; help the type checker
        if not cfg.enabled:
            return []
        if cfg.backend == TraceBackend.OTLP:
            backend: TracingBackend = OTLPBackend(endpoint=cfg.endpoint)
            return [backend] if backend.is_available() else []
        return _resolve_names([cfg.backend.value])
    if isinstance(spec, str):
        return _resolve_names(_split(spec))
    if isinstance(spec, list):
        return _resolve_names([str(s) for s in spec])
    raise ExoError(
        f"Unsupported tracing spec: {spec!r}",
        context={"spec_type": type(spec).__name__},
        hint=("Pass a string ('langfuse', 'otlp'), list of strings, bool, TracingConfig, or None."),
    )


# ---------------------------------------------------------------------------
# Built-in backends
# ---------------------------------------------------------------------------

register("otlp", lambda: OTLPBackend())

# Shared in-process span sink (TraceBackend.MEMORY). A singleton so callers can
# read captured spans after a run via get_memory_backend().spans.
_MEMORY_BACKEND = MemoryBackend()
register("memory", lambda: _MEMORY_BACKEND)

# Console backend — prints spans to stdout via OTel's ConsoleSpanExporter.
# Registered eagerly (like otlp/memory) since it has no optional-SDK deps beyond
# opentelemetry-sdk itself.
from exo.observability.backends.console import (  # noqa: E402  # pyright: ignore[reportMissingImports]
    ConsoleBackend,
)

register("console", lambda: ConsoleBackend())


def get_memory_backend() -> MemoryBackend:
    """Return the shared in-process span sink used by the ``"memory"`` backend."""
    return _MEMORY_BACKEND


# Phase 3: vendor backends (langfuse / langsmith / phoenix / braintrust) register
# themselves when their module is imported via _ensure_loaded(). Do not import
# them here — that would pull optional deps into the base install.


def _reset_for_tests() -> None:
    """Reset module-level state for test isolation (tests only)."""
    global _announced
    with _announced_lock:
        _announced = False
    # Clear all vendor registrations so tests that import backends directly do
    # not leak registrations into unrelated test sessions.  Built-in backends
    # (otlp, memory, console) are re-registered immediately after the clear.
    _REGISTRY.clear()
    register("otlp", lambda: OTLPBackend())
    register("memory", lambda: _MEMORY_BACKEND)
    register("console", lambda: ConsoleBackend())


__all__ = ["_reset_for_tests", "get_memory_backend", "register", "resolve_backends"]
