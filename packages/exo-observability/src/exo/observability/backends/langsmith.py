"""LangSmith tracing backend for Exo observability.

Exports spans via OTLP/HTTP to the LangSmith ingestion endpoint and optionally
exposes a thin dataset/feedback client when the ``langsmith`` SDK is installed.

Self-registers on import — load via:

    importlib.import_module("exo.observability.backends.langsmith")

or by setting ``EXO_TRACING=langsmith``.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any

from exo.observability.backends import register  # pyright: ignore[reportMissingImports]
from exo.observability.backends.base import OTLPBackend  # pyright: ignore[reportMissingImports]
from exo.types import ExoError  # pyright: ignore[reportMissingImports]

_log = logging.getLogger(__name__)

_DEFAULT_ENDPOINT = "https://api.smith.langchain.com"


# ---------------------------------------------------------------------------
# Dataset / feedback client wrapper
# ---------------------------------------------------------------------------


class _LangSmithDatasetClient:
    """Thin wrapper around the langsmith SDK Client for dataset + feedback ops.

    Only instantiated when the ``langsmith`` package is available and an API
    key is configured. All heavy imports are deferred to construction time.
    """

    def __init__(self, api_key: str, endpoint: str) -> None:
        try:
            from langsmith import Client  # pyright: ignore[reportMissingImports]
        except ImportError as exc:
            raise ExoError(
                "langsmith SDK is not installed; cannot use dataset_client().",
                hint="Install it with: pip install langsmith",
                context={"operation": "dataset_client"},
            ) from exc

        self._client = Client(api_key=api_key, api_url=endpoint)

    def list_examples(self, dataset_name: str) -> list[dict[str, Any]]:
        """Return all examples in *dataset_name* as plain dicts."""
        examples = self._client.list_examples(dataset_name=dataset_name)
        return [
            {
                "id": str(ex.id),
                "inputs": ex.inputs,
                "outputs": ex.outputs,
            }
            for ex in examples
        ]

    def create_feedback(self, run_id: str, key: str, score: float) -> None:
        """Create a scalar feedback entry against *run_id*."""
        self._client.create_feedback(run_id=run_id, key=key, score=score)


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------


@dataclass
class LangSmithBackend(OTLPBackend):
    """OTLP tracing backend that targets the LangSmith collector.

    Configuration (constructor args override env vars):

    - ``api_key``: LangSmith API key. Env: ``LANGSMITH_API_KEY`` (fallback
      ``LANGCHAIN_API_KEY``).
    - ``endpoint``: LangSmith base URL (default ``https://api.smith.langchain.com``).
      Env: ``LANGSMITH_ENDPOINT``.  The OTLP traces path ``/otel/v1/traces`` is
      appended automatically.
    - ``project``: LangSmith project name (default ``"default"``). Env:
      ``LANGSMITH_PROJECT`` (fallback ``LANGCHAIN_PROJECT``).

    OTLP auth uses the ``x-api-key`` and ``Langsmith-Project`` headers per
    LangSmith's OTLP ingestion spec.
    """

    name: str = field(default="langsmith", init=False)

    # Explicit overrides (None means "read from env at construction time")
    api_key: str | None = field(default=None, repr=False)
    project: str | None = None
    # ``endpoint`` is inherited from OTLPBackend (stores the full OTLP URL)

    def __post_init__(self) -> None:
        resolved_key = self.api_key or _resolve_api_key()
        resolved_project = self.project or _resolve_project()
        resolved_base = self.endpoint or os.environ.get("LANGSMITH_ENDPOINT", _DEFAULT_ENDPOINT)

        # Store resolved values so callers can inspect them
        self.api_key = resolved_key
        self.project = resolved_project

        # Build the full OTLP endpoint and headers for OTLPBackend
        self.endpoint = f"{resolved_base.rstrip('/')}/otel/v1/traces"
        self.headers = {
            "x-api-key": resolved_key or "",
            "Langsmith-Project": resolved_project,
        }

        # Cached dataset/score client — built once on first call.
        # None = not yet built; False = build failed (SDK unavailable or no key).
        self._dataset_client_cache: _LangSmithDatasetClient | None | bool = None

    def is_available(self) -> bool:
        """True only when an API key is present AND the OTLP exporter is importable."""
        if not self.api_key:
            return False
        return super().is_available()

    def dataset_client(self) -> _LangSmithDatasetClient | None:
        """Return a cached dataset/feedback client if the langsmith SDK is available."""
        if not self.api_key:
            _log.debug("LangSmith dataset_client: no API key configured")
            return None
        # Return cached instance (avoids rebuilding a new SDK client each call).
        if self._dataset_client_cache is not None and self._dataset_client_cache is not False:
            return self._dataset_client_cache
        if self._dataset_client_cache is False:
            return None
        try:
            import langsmith  # noqa: F401  # pyright: ignore[reportMissingImports]
        except ImportError:
            _log.debug("LangSmith dataset_client: langsmith SDK not installed")
            self._dataset_client_cache = False
            return None

        base = os.environ.get("LANGSMITH_ENDPOINT", _DEFAULT_ENDPOINT)
        client = _LangSmithDatasetClient(api_key=self.api_key, endpoint=base)
        self._dataset_client_cache = client
        return client

    def score_client(self) -> Any | None:
        """Alias to dataset_client() — LangSmith's feedback API is on the same client."""
        return self.dataset_client()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_api_key() -> str | None:
    return os.environ.get("LANGSMITH_API_KEY") or os.environ.get("LANGCHAIN_API_KEY")


def _resolve_project() -> str:
    return os.environ.get("LANGSMITH_PROJECT") or os.environ.get("LANGCHAIN_PROJECT") or "default"


# ---------------------------------------------------------------------------
# Self-registration
# ---------------------------------------------------------------------------

register("langsmith", lambda: LangSmithBackend())
