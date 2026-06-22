"""Arize Phoenix tracing backend (OTLP/HTTP).

Wraps :class:`OTLPBackend` with Phoenix-specific endpoint/auth resolution so
Phoenix works out of the box — both self-hosted (localhost:6006) and Phoenix
Cloud (api_key required).

Environment variables
---------------------
PHOENIX_COLLECTOR_ENDPOINT
    Full base URL of the Phoenix OTLP collector (default ``http://localhost:6006``).
    The OTLP traces path ``/v1/traces`` is appended automatically.
PHOENIX_API_KEY
    API key for Phoenix Cloud.  When set, the ``Authorization: Bearer <key>``
    and ``api_key: <key>`` headers are added (Phoenix Cloud accepts both forms).
PHOENIX_CLIENT_HEADERS
    Extra headers as a comma-separated ``k=v,k2=v2`` string.  Merged after the
    API-key headers so they can override or extend them.

The module self-registers as ``"phoenix"`` on import so the backend registry
can load it lazily without explicit imports at the framework level.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

from exo.observability.backends import register  # pyright: ignore[reportMissingImports]
from exo.observability.backends.base import OTLPBackend  # pyright: ignore[reportMissingImports]

_log = logging.getLogger(__name__)

_DEFAULT_COLLECTOR = "http://localhost:6006"


def _parse_client_headers(raw: str) -> dict[str, str]:
    """Parse a ``k=v,k2=v2`` header string into a dict.

    Silently skips malformed pairs (no ``=`` separator).
    """
    result: dict[str, str] = {}
    for token in raw.split(","):
        token = token.strip()
        if "=" in token:
            k, _, v = token.partition("=")
            result[k.strip()] = v.strip()
        elif token:
            _log.debug("phoenix: skipping malformed header token %r", token)
    return result


@dataclass
class PhoenixBackend(OTLPBackend):
    """Arize Phoenix OTLP tracing backend.

    Args:
        collector_endpoint: Base collector URL (env ``PHOENIX_COLLECTOR_ENDPOINT``
            or ``http://localhost:6006`` by default). ``/v1/traces`` is appended.
        api_key: Phoenix Cloud API key (env ``PHOENIX_API_KEY``).  Self-hosted
            Phoenix does **not** require a key.
        headers: Extra headers merged on top of the auto-resolved auth headers.
            Explicit headers here are merged *after* API-key derivation but may
            override them if the same key appears.
    """

    # Override the parent's generic name.
    name: str = "phoenix"

    # Phoenix-specific init args — stored so is_available() / build_span_processor()
    # can defer construction until the parent dataclass fields are fully populated.
    _collector_endpoint: str = field(default="", init=False, repr=False)
    _api_key: str | None = field(default=None, init=False, repr=False)
    _extra_headers: dict[str, str] = field(default_factory=dict, init=False, repr=False)

    def __init__(
        self,
        *,
        collector_endpoint: str | None = None,
        api_key: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        # --- resolve collector endpoint ---
        resolved_collector = (
            collector_endpoint or os.environ.get("PHOENIX_COLLECTOR_ENDPOINT") or _DEFAULT_COLLECTOR
        ).rstrip("/")
        otlp_endpoint = f"{resolved_collector}/v1/traces"

        # --- resolve auth headers ---
        resolved_key = api_key or os.environ.get("PHOENIX_API_KEY") or None
        auth_headers: dict[str, str] = {}
        if resolved_key:
            auth_headers["Authorization"] = f"Bearer {resolved_key}"
            auth_headers["api_key"] = resolved_key

        # --- parse PHOENIX_CLIENT_HEADERS ---
        client_headers_raw = os.environ.get("PHOENIX_CLIENT_HEADERS", "")
        if client_headers_raw:
            auth_headers.update(_parse_client_headers(client_headers_raw))

        # --- apply caller-supplied header overrides ---
        if headers:
            auth_headers.update(headers)

        # Delegate to the parent OTLPBackend dataclass.
        super().__init__(
            endpoint=otlp_endpoint,
            headers=auth_headers,
            name="phoenix",
        )

        # Stash resolved values for introspection / repr.
        self._collector_endpoint = resolved_collector
        self._api_key = resolved_key
        self._extra_headers = dict(headers or {})

    # is_available() is inherited from OTLPBackend — returns True iff the OTLP
    # HTTP exporter is importable.  Phoenix self-hosted needs no API key.

    # build_span_processor() is fully inherited from OTLPBackend.

    # score_client() / dataset_client() return None (inherited).


# ---------------------------------------------------------------------------
# Self-registration
# ---------------------------------------------------------------------------

register("phoenix", lambda: PhoenixBackend())
