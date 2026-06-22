"""Braintrust tracing backend for Exo observability.

Exports OpenTelemetry spans to Braintrust via OTLP/HTTP. The backend auto-enables
when ``BRAINTRUST_API_KEY`` is set in the environment and the OTLP exporter package
is installed.

Env vars:
    ``BRAINTRUST_API_KEY``     — required for auth.
    ``BRAINTRUST_API_URL``     — base API URL (default ``https://api.braintrust.dev``).
    ``BRAINTRUST_PARENT``      — span-routing header value (default ``project_name:default``).

Self-registration: importing this module calls ``register("braintrust", ...)`` so
``_ensure_loaded`` in the registry can discover it without a direct import.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any

from exo.observability.backends import register  # pyright: ignore[reportMissingImports]
from exo.observability.backends.base import OTLPBackend  # pyright: ignore[reportMissingImports]

_log = logging.getLogger(__name__)

_DEFAULT_API_URL = "https://api.braintrust.dev"
_DEFAULT_PARENT = "project_name:default"


# ---------------------------------------------------------------------------
# Score client wrapper
# ---------------------------------------------------------------------------


class _BraintrustScoreClient:
    """Thin wrapper around the ``braintrust`` SDK for publishing eval scores.

    All braintrust SDK interactions are deferred to method call time so the
    wrapper can be constructed without importing the SDK at module level.

    Args:
        api_key: Braintrust API key used to authenticate SDK calls.
    """

    def __init__(self, api_key: str) -> None:
        # Note: SDK import is deferred to log_score() to match the test contract
        # that _BraintrustScoreClient can be constructed even without the SDK
        # installed (unlike Langfuse/LangSmith which raise at construction time).
        self._api_key = api_key

    def log_score(self, span_id_or_name: str, name: str, score: float) -> None:
        """Publish a single eval score to Braintrust.

        Args:
            span_id_or_name: The span ID or human-readable name to attach the score to.
            name:            Score metric name (e.g. ``"accuracy"``, ``"faithfulness"``).
            score:           Numeric score value (typically 0.0-1.0).
        """
        try:
            import braintrust  # type: ignore[import-untyped]  # pyright: ignore[reportMissingImports]
        except ImportError:
            _log.warning(
                "braintrust SDK not installed; cannot log score %r for span %r",
                name,
                span_id_or_name,
            )
            return

        try:
            span = braintrust.current_span()
            span.log(scores={name: score})
            _log.debug("logged score %r=%.4f for span %r", name, score, span_id_or_name)
        except Exception as exc:  # pragma: no cover — SDK runtime errors
            _log.warning("braintrust.log_score failed: %s", exc, exc_info=True)


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------


@dataclass
class BraintrustBackend(OTLPBackend):
    """OTLP tracing backend that ships spans to Braintrust.

    Braintrust accepts standard OTLP/HTTP traces at ``<api_url>/otel/v1/traces``
    and uses the ``Authorization: Bearer <api_key>`` header plus an optional
    ``x-bt-parent`` header for project routing.

    Args:
        api_key: Braintrust API key. Falls back to ``BRAINTRUST_API_KEY``.
        api_url: Base API URL. Falls back to ``BRAINTRUST_API_URL`` then
            ``https://api.braintrust.dev``.
        parent: Span-routing header (``project_name:<name>`` or ``experiment:<id>``).
            Falls back to ``BRAINTRUST_PARENT`` then ``project_name:default``.
    """

    # Braintrust-specific constructor args — resolved in __post_init__.
    api_key: str | None = field(default=None)
    api_url: str | None = field(default=None)
    parent: str | None = field(default=None)

    # OTLPBackend fields — set by __post_init__ after env resolution.
    # Declared here with None so the dataclass field ordering is valid; they
    # are always populated before any caller sees the instance.
    name: str = "braintrust"

    def __post_init__(self) -> None:
        # Resolve configuration from explicit args → env → defaults.
        resolved_key = self.api_key or os.environ.get("BRAINTRUST_API_KEY") or ""
        resolved_url = self.api_url or os.environ.get("BRAINTRUST_API_URL") or _DEFAULT_API_URL
        resolved_parent = self.parent or os.environ.get("BRAINTRUST_PARENT") or _DEFAULT_PARENT

        # Build OTLP endpoint.
        self.endpoint = f"{resolved_url.rstrip('/')}/otel/v1/traces"

        # Build headers — only include auth if key is present.
        self.headers = {}
        if resolved_key:
            self.headers["Authorization"] = f"Bearer {resolved_key}"
        self.headers["x-bt-parent"] = resolved_parent

        # Store resolved values for introspection / score_client.
        self._resolved_key = resolved_key
        self._resolved_url = resolved_url
        self._resolved_parent = resolved_parent

    def is_available(self) -> bool:
        """Return True only when an API key is present AND the OTLP exporter is importable."""
        if not self._resolved_key:
            _log.debug("braintrust backend unavailable: BRAINTRUST_API_KEY not set")
            return False
        return super().is_available()

    def score_client(self) -> _BraintrustScoreClient | None:
        """Return a score client if the braintrust SDK is installed and a key is set.

        Returns ``None`` gracefully when either condition is unmet — callers
        should treat ``None`` as "scoring unavailable" rather than an error.
        """
        if not self._resolved_key:
            return None
        try:
            import braintrust  # noqa: F401  # type: ignore[import-untyped]  # pyright: ignore[reportMissingImports]
        except ImportError:
            _log.debug("braintrust SDK not installed; score_client returning None")
            return None
        return _BraintrustScoreClient(api_key=self._resolved_key)

    def dataset_client(self) -> Any | None:
        """Dataset sync not yet implemented for Braintrust."""
        return None


# ---------------------------------------------------------------------------
# Self-registration (runs on module import)
# ---------------------------------------------------------------------------

register("braintrust", lambda: BraintrustBackend())
