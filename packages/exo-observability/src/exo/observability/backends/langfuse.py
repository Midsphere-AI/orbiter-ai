"""Langfuse tracing backend for Exo observability.

Exports OpenTelemetry spans to Langfuse via its OTLP/HTTP endpoint and
optionally exposes a thin score client backed by the langfuse SDK.

The module self-registers on import:
    import exo.observability.backends.langfuse  # registers "langfuse"
"""

from __future__ import annotations

import base64
import logging
import os
from dataclasses import dataclass, field
from typing import Any

from exo.observability.backends import register  # pyright: ignore[reportMissingImports]
from exo.observability.backends.base import OTLPBackend  # pyright: ignore[reportMissingImports]
from exo.types import ExoError  # pyright: ignore[reportMissingImports]

_log = logging.getLogger(__name__)

_DEFAULT_HOST = "https://cloud.langfuse.com"


# ---------------------------------------------------------------------------
# Optional langfuse SDK wrapper
# ---------------------------------------------------------------------------


class _LangfuseScoreClient:
    """Thin wrapper around the langfuse SDK for publishing eval scores.

    Only instantiated when the ``langfuse`` package is installed and keys are
    present. The eval bridge uses this seam to avoid a hard dependency.
    """

    def __init__(self, public_key: str, secret_key: str, host: str) -> None:
        try:
            import langfuse as _lf  # pyright: ignore[reportMissingImports]
        except ImportError as exc:
            raise ExoError(
                "langfuse SDK is not installed; cannot publish eval scores.",
                hint="Install it with: pip install langfuse  (or: pip install 'exo-ai[langfuse]')",
                context={"operation": "score_client"},
            ) from exc

        self._client = _lf.Langfuse(
            public_key=public_key,
            secret_key=secret_key,
            host=host,
        )

    def score(
        self,
        trace_id: str,
        name: str,
        value: float,
        comment: str | None = None,
    ) -> None:
        """Publish an eval score for *trace_id* to Langfuse."""
        kwargs: dict[str, Any] = {
            "trace_id": trace_id,
            "name": name,
            "value": value,
        }
        if comment is not None:
            kwargs["comment"] = comment
        try:
            self._client.score(**kwargs)
        except Exception as exc:
            _log.warning(
                "langfuse.score failed (score not recorded): %s",
                exc,
                exc_info=True,
            )


# ---------------------------------------------------------------------------
# LangfuseBackend
# ---------------------------------------------------------------------------


@dataclass
class LangfuseBackend(OTLPBackend):
    """Langfuse tracing backend — exports OTel spans to Langfuse's OTLP endpoint.

    Reads credentials from the environment by default:
    - ``LANGFUSE_PUBLIC_KEY``: Langfuse public key (required).
    - ``LANGFUSE_SECRET_KEY``: Langfuse secret key (required).
    - ``LANGFUSE_HOST``: Langfuse host (default ``https://cloud.langfuse.com``).

    Explicit constructor args override the corresponding env vars.

    Args:
        public_key: Langfuse public key.  Falls back to ``LANGFUSE_PUBLIC_KEY``.
        secret_key: Langfuse secret key.  Falls back to ``LANGFUSE_SECRET_KEY``.
        host: Langfuse host URL.  Falls back to ``LANGFUSE_HOST``.
        name: Backend registry name (default ``"langfuse"``).
    """

    public_key: str | None = field(default=None)
    secret_key: str | None = field(default=None)
    host: str | None = field(default=None)
    name: str = "langfuse"

    def __post_init__(self) -> None:
        # Resolve credentials from env if not supplied explicitly.
        pk = self.public_key or os.environ.get("LANGFUSE_PUBLIC_KEY") or ""
        sk = self.secret_key or os.environ.get("LANGFUSE_SECRET_KEY") or ""
        resolved_host = self.host or os.environ.get("LANGFUSE_HOST") or _DEFAULT_HOST

        # Store resolved values so callers can inspect them.
        self._resolved_public_key: str = pk
        self._resolved_secret_key: str = sk
        self._resolved_host: str = resolved_host

        # Build OTLP endpoint and auth header.
        endpoint = f"{resolved_host.rstrip('/')}/api/public/otel/v1/traces"
        headers: dict[str, str] = {}
        if pk and sk:
            token = base64.b64encode(f"{pk}:{sk}".encode()).decode()
            headers["Authorization"] = f"Basic {token}"

        # Initialise the OTLPBackend fields.
        self.endpoint = endpoint
        self.headers = headers

        # Cached score client — built once on first call (None = not yet built,
        # False = build was attempted but SDK unavailable/no keys).
        self._score_client_cache: _LangfuseScoreClient | None | bool = None

    def is_available(self) -> bool:
        """True only when both keys are present AND the OTLP exporter is importable."""
        if not (self._resolved_public_key and self._resolved_secret_key):
            return False
        return super().is_available()

    def score_client(self) -> Any | None:
        """Return a cached score client if the langfuse SDK is installed and keys are present."""
        if not (self._resolved_public_key and self._resolved_secret_key):
            return None
        # Return cached instance (avoids rebuilding a new SDK client each call).
        if self._score_client_cache is not None and self._score_client_cache is not False:
            return self._score_client_cache
        if self._score_client_cache is False:
            return None
        try:
            import langfuse  # noqa: F401  # pyright: ignore[reportMissingImports]
        except ImportError:
            _log.debug("langfuse SDK not installed; score_client() returning None")
            self._score_client_cache = False
            return None
        client = _LangfuseScoreClient(
            public_key=self._resolved_public_key,
            secret_key=self._resolved_secret_key,
            host=self._resolved_host,
        )
        self._score_client_cache = client
        return client

    def dataset_client(self) -> Any | None:
        return None


# ---------------------------------------------------------------------------
# Self-registration
# ---------------------------------------------------------------------------

register("langfuse", lambda: LangfuseBackend())
