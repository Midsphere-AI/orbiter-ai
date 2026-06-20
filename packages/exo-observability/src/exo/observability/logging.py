"""Structured logging for Exo — stdlib only.

Provides ANSI text and JSON formatters, logger namespace management,
and a ``LogContext`` context manager for binding structured key-value
pairs to all log records within a scope.

Usage::

    from exo.observability.logging import get_logger, configure_logging, LogContext

    log = get_logger("agent")          # -> exo.agent
    configure_logging(level="DEBUG")   # idempotent
    with LogContext(agent_name="alpha"):
        log.info("starting")
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import Any

# NOTE (D-5): _PREFIX is also defined in exo-core/src/exo/log.py.  That module
# is a backward-compat shim that re-exports symbols from this file and re-declares
# _PREFIX locally so it can be imported standalone without pulling in this package.
# The duplication is intentional — do not import _PREFIX from exo.log here, as
# that would create a circular-import risk (exo.log imports from this module).
_PREFIX = "exo"

# ---------------------------------------------------------------------------
# Context variable for structured log context
# ---------------------------------------------------------------------------

_log_context: ContextVar[dict[str, Any] | None] = ContextVar("_log_context", default=None)


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------

_COLORS: dict[int, str] = {
    logging.DEBUG: "\033[2m",  # dim
    logging.INFO: "\033[36m",  # cyan
    logging.WARNING: "\033[33m",  # yellow
    logging.ERROR: "\033[31m",  # red
    logging.CRITICAL: "\033[1;31m",  # bold red
}
_RESET = "\033[0m"
_DIM = "\033[2m"

_LEVEL_CHAR: dict[int, str] = {
    logging.DEBUG: "D",
    logging.INFO: "I",
    logging.WARNING: "W",
    logging.ERROR: "E",
    logging.CRITICAL: "C",
}


class TextFormatter(logging.Formatter):
    """Compact single-line ANSI formatter (ported from exo.log)."""

    def format(self, record: logging.LogRecord) -> str:
        ts = self.formatTime(record, "%H:%M:%S")
        lvl = _LEVEL_CHAR.get(record.levelno, "?")
        color = _COLORS.get(record.levelno, "")
        # Strip "exo." prefix for shorter output
        name = record.name
        if name.startswith(f"{_PREFIX}."):
            name = name[len(_PREFIX) + 1 :]
        msg = record.getMessage()

        # Append context vars if present
        ctx = _log_context.get()
        ctx_str = ""
        if ctx:
            ctx_str = " " + " ".join(f"{k}={v}" for k, v in ctx.items())

        line = f"{_DIM}{ts}{_RESET} {color}{lvl} {name}{_RESET}{ctx_str} \u25b8 {msg}"
        if record.exc_info and not record.exc_text:
            record.exc_text = self.formatException(record.exc_info)
        if record.exc_text:
            line += f"\n{color}{record.exc_text}{_RESET}"
        return line


class JsonFormatter(logging.Formatter):
    """Structured JSON formatter for production logging."""

    def format(self, record: logging.LogRecord) -> str:
        entry: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Merge context vars
        ctx = _log_context.get()
        if ctx:
            entry["extra"] = dict(ctx)

        if record.exc_info and record.exc_info[1] is not None:
            entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(entry, default=str)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_logger(name: str) -> logging.Logger:
    """Return a stdlib ``Logger`` under the ``exo.`` namespace.

    If *name* does not start with ``exo.``, it is auto-prefixed.
    """
    if not name.startswith(f"{_PREFIX}.") and name != _PREFIX:
        name = f"{_PREFIX}.{name}"
    return logging.getLogger(name)


_configure_lock = threading.Lock()
_configured = False


def configure_logging(
    level: str | int = "WARNING",
    fmt: str = "text",
    *,
    force: bool = False,
) -> None:
    """One-time handler setup on the ``exo`` root logger.

    Idempotent by default — calling twice is a no-op unless *force* is set.

    Args:
        level: Log level name or int (e.g. ``"DEBUG"``, ``logging.INFO``).
        fmt: ``"text"`` for compact ANSI output, ``"json"`` for structured JSON.
        force: If ``True``, remove existing handlers before adding a new one.
    """
    global _configured

    with _configure_lock:
        if _configured and not force:
            return
        if force:
            _configured = False

        root = logging.getLogger(_PREFIX)
        root.handlers.clear()

        handler = logging.StreamHandler(sys.stderr)
        if fmt == "json":
            handler.setFormatter(JsonFormatter())
        else:
            handler.setFormatter(TextFormatter())

        root.addHandler(handler)

        if isinstance(level, str):
            level = getattr(logging, level.upper(), logging.WARNING)
        root.setLevel(level)
        _configured = True


def reset_logging() -> None:
    """Reset logging state — for testing only."""
    global _configured
    with _configure_lock:
        _configured = False
        root = logging.getLogger(_PREFIX)
        root.handlers.clear()
        root.setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Environment-variable driven auto-configuration
# ---------------------------------------------------------------------------

_ENV_FORMAT = "%(asctime)s %(levelname)-8s %(name)s  %(message)s"
_VALID_ENV_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR"}


def _configure_from_env() -> None:
    """Apply EXO_DEBUG / EXO_LOG_LEVEL env vars to the root exo logger.

    Called automatically at module import time.  Safe to call again after the
    environment is mutated (e.g. in tests using ``monkeypatch.setenv``).

    - ``EXO_DEBUG=1`` forces level to DEBUG regardless of ``EXO_LOG_LEVEL``.
    - ``EXO_LOG_LEVEL`` accepts DEBUG / INFO / WARNING / ERROR (case-insensitive).
      Any unrecognised value falls back to WARNING.
    - When either variable is present, a :class:`logging.StreamHandler` with the
      standard format ``%(asctime)s %(levelname)-8s %(name)s  %(message)s`` is
      attached to the root ``exo`` logger (idempotent — only once per reset).
    """
    global _configured

    env_debug = os.environ.get("EXO_DEBUG", "")
    env_level_str = os.environ.get("EXO_LOG_LEVEL", "WARNING").upper()
    if env_level_str not in _VALID_ENV_LEVELS:
        env_level_str = "WARNING"
    if env_debug == "1":
        env_level_str = "DEBUG"

    level = getattr(logging, env_level_str, logging.WARNING)
    root = logging.getLogger(_PREFIX)
    root.setLevel(level)

    # Only add a handler when explicitly triggered by an env var
    if env_debug == "1" or "EXO_LOG_LEVEL" in os.environ:
        with _configure_lock:
            if not _configured:
                handler = logging.StreamHandler(sys.stderr)
                handler.setFormatter(logging.Formatter(_ENV_FORMAT))
                root.addHandler(handler)
                _configured = True


# Apply env-var config at import time so the level is in effect before any
# explicit configure_logging() call.
_configure_from_env()


# ---------------------------------------------------------------------------
# LogContext — bind structured key-value pairs to logs within a scope
# ---------------------------------------------------------------------------


class LogContext:
    """Context manager that binds key-value pairs to all log records in scope.

    Uses ``contextvars`` so it works correctly with asyncio concurrency.

    Usage::

        with LogContext(agent_name="alpha", task_id="t-1"):
            log.info("step completed")  # includes agent_name=alpha task_id=t-1
    """

    def __init__(self, **kwargs: Any) -> None:
        self._bindings = kwargs
        self._token: Any = None

    def __enter__(self) -> LogContext:
        current = _log_context.get()
        merged = dict(current) if current else {}
        merged.update(self._bindings)
        self._token = _log_context.set(merged)
        return self

    def __exit__(self, *_: object) -> None:
        if self._token is not None:
            _log_context.reset(self._token)
