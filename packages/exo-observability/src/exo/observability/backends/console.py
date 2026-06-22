"""Console tracing backend for Exo observability.

Exports OpenTelemetry spans to stdout via OTel's built-in ``ConsoleSpanExporter``.
Useful for local debugging without setting up an external collector.

Self-registers on import:
    import exo.observability.backends.console  # registers "console"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from exo.observability.backends import register  # pyright: ignore[reportMissingImports]
from exo.observability.backends.base import TracingBackend  # pyright: ignore[reportMissingImports]

_log = logging.getLogger(__name__)


@dataclass
class ConsoleBackend:
    """Console tracing backend — prints finished spans to stdout.

    Uses OTel's ``ConsoleSpanExporter`` (from ``opentelemetry-sdk``) which
    writes a JSON representation of each span to stdout when it finishes.
    Always available when ``opentelemetry-sdk`` is installed (no extra deps).
    """

    name: str = "console"

    def is_available(self) -> bool:
        """True when ``opentelemetry-sdk`` is installed."""
        try:
            import opentelemetry.sdk.trace  # noqa: F401  # pyright: ignore[reportMissingImports]
        except ImportError:
            return False
        return True

    def build_span_processor(self) -> Any | None:
        try:
            from opentelemetry.sdk.trace.export import (  # pyright: ignore[reportMissingImports]
                SimpleSpanProcessor,
            )
            from opentelemetry.sdk.trace.export.in_memory_span_exporter import (  # noqa: F401
                InMemorySpanExporter,
            )
        except ImportError:
            _log.debug(
                "ConsoleBackend: opentelemetry-sdk not installed; cannot build span processor"
            )
            return None

        try:
            # ConsoleSpanExporter lives in the same optional-sdk package.
            from opentelemetry.sdk.trace.export import (  # pyright: ignore[reportMissingImports]
                ConsoleSpanExporter,
            )
        except ImportError:
            _log.debug("ConsoleBackend: ConsoleSpanExporter not available; falling back to no-op")
            return None

        return SimpleSpanProcessor(ConsoleSpanExporter())

    def score_client(self) -> Any | None:
        return None

    def dataset_client(self) -> Any | None:
        return None


# Satisfy the TracingBackend structural protocol (mypy/pyright).
_: TracingBackend = ConsoleBackend()  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Self-registration
# ---------------------------------------------------------------------------

register("console", lambda: ConsoleBackend())
