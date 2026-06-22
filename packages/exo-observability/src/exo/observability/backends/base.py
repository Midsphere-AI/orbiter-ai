"""Vendor-neutral tracing backend protocol + generic OTLP backend.

A :class:`TracingBackend` knows how to build an OpenTelemetry ``SpanProcessor``
that exports spans to a particular destination (a generic OTLP collector,
Langfuse, LangSmith, Phoenix, Braintrust, ...). Vendor adapters live in sibling
modules and register themselves with the backend registry in
``exo.observability.backends`` — this module only defines the seam.

All OpenTelemetry imports are guarded so the package keeps working when neither
``opentelemetry-sdk`` nor the OTLP exporter is installed.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

_log = logging.getLogger(__name__)


@runtime_checkable
class TracingBackend(Protocol):
    """A destination spans are exported to.

    Phase 1 only needs ``build_span_processor``; the optional ``score_client``
    and ``dataset_client`` hooks are the seam later vendor adapters (Langfuse,
    LangSmith, Braintrust) fill in to publish eval scores / sync datasets.
    """

    name: str

    def is_available(self) -> bool:
        """Return True if this backend can actually export (deps + config present)."""
        ...

    def build_span_processor(self) -> Any | None:
        """Build an OTel ``SpanProcessor``, or None if it cannot be constructed."""
        ...

    def score_client(self) -> Any | None:  # pragma: no cover - default seam
        """Optional native client for publishing eval scores. None if unsupported."""
        ...

    def dataset_client(self) -> Any | None:  # pragma: no cover - default seam
        """Optional native client for dataset pull/push. None if unsupported."""
        ...


@dataclass
class OTLPBackend:
    """Generic OTLP/HTTP span exporter backend.

    Works with any OTLP-compatible collector. Langfuse, LangSmith, Phoenix and
    others all ingest OTLP, so vendor adapters typically subclass or wrap this,
    supplying their endpoint + auth ``headers``.

    Args:
        endpoint: OTLP traces endpoint (e.g. ``http://localhost:4318/v1/traces``).
            Falls back to ``OTEL_EXPORTER_OTLP_ENDPOINT`` when omitted.
        headers: Auth/extra headers for the exporter.
        name: Backend name used in the registry and de-dupe set.
    """

    endpoint: str | None = None
    headers: dict[str, str] = field(default_factory=dict)
    name: str = "otlp"
    export_timeout_millis: int = 5000

    def is_available(self) -> bool:
        try:
            import opentelemetry.exporter.otlp.proto.http.trace_exporter  # pyright: ignore[reportMissingImports]
            import opentelemetry.sdk.trace  # noqa: F401  # pyright: ignore[reportMissingImports]
        except ImportError:
            return False
        return True

    def build_span_processor(self) -> Any | None:
        try:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import (  # pyright: ignore[reportMissingImports]
                OTLPSpanExporter,
            )
            from opentelemetry.sdk.trace.export import BatchSpanProcessor
        except ImportError:
            _log.debug(
                "OTLP exporter unavailable for backend %r (install opentelemetry-exporter-otlp)",
                self.name,
            )
            return None

        endpoint = self.endpoint or os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
        kwargs: dict[str, Any] = {}
        if endpoint:
            kwargs["endpoint"] = endpoint
        if self.headers:
            kwargs["headers"] = dict(self.headers)
        exporter = OTLPSpanExporter(**kwargs)
        return BatchSpanProcessor(exporter, export_timeout_millis=self.export_timeout_millis)

    def score_client(self) -> Any | None:
        return None

    def dataset_client(self) -> Any | None:
        return None


@dataclass
class MemoryBackend:
    """In-process span sink (the ``TraceBackend.MEMORY`` backend).

    Captures finished spans in memory via OTel's ``InMemorySpanExporter`` so
    they can be inspected without an external collector — useful for tests and
    local debugging. ``spans`` returns the captured ReadableSpans.
    """

    name: str = "memory"

    def __post_init__(self) -> None:
        self._exporter: Any | None = None

    def is_available(self) -> bool:
        try:
            import opentelemetry.sdk.trace  # noqa: F401
        except ImportError:
            return False
        return True

    def build_span_processor(self) -> Any | None:
        try:
            from opentelemetry.sdk.trace.export import SimpleSpanProcessor
            from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
                InMemorySpanExporter,
            )
        except ImportError:
            return None
        self._exporter = InMemorySpanExporter()
        return SimpleSpanProcessor(self._exporter)

    @property
    def spans(self) -> tuple[Any, ...]:
        if self._exporter is None:
            return ()
        return tuple(self._exporter.get_finished_spans())

    def clear(self) -> None:
        if self._exporter is not None:
            self._exporter.clear()

    def score_client(self) -> Any | None:
        return None

    def dataset_client(self) -> Any | None:
        return None
