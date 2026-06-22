"""Google LLM provider implementation (Gemini API and Vertex AI).

A single ``GoogleProvider`` class supports both direct-Gemini auth (via
``GOOGLE_API_KEY`` / ``GEMINI_API_KEY``) and Vertex AI auth (project,
location, service-account).  The provider is registered under both the
``"gemini"`` and ``"vertex"`` prefixes so that model strings
``"gemini:..."`` and ``"vertex:..."`` both resolve correctly.

``GeminiProvider`` is exported as an alias for backward compatibility.
"""

from __future__ import annotations

import base64
import json
import logging
import os
from collections.abc import AsyncIterator
from typing import Any

from google import genai
from google.genai import errors as genai_errors

from exo.config import ModelConfig
from exo.models._google_common import (
    _build_config,
    _convert_tools,  # noqa: F401 — re-exported for backward compat
    _map_finish_reason,  # noqa: F401 — re-exported for backward compat
    _parse_response,
    _parse_stream_chunk,
    _to_google_contents,
)
from exo.models.provider import ModelProvider, model_registry
from exo.models.types import ModelError, ModelResponse, StreamChunk
from exo.types import Message

_log = logging.getLogger(__name__)

__all__ = [
    "GeminiProvider",
    "GoogleProvider",
]

_VERTEX_SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]


# ---------------------------------------------------------------------------
# Error helper
# ---------------------------------------------------------------------------


def _google_error_hint(exc: Any) -> tuple[str, str | None]:
    """Return (message, hint) for a Google genai API error."""
    msg = str(exc)
    code = getattr(exc, "code", None)
    status = str(code) if code else ""
    if "PERMISSION_DENIED" in status or "403" in msg:
        return msg, "Check GOOGLE_API_KEY or Vertex AI service account permissions."
    if "RESOURCE_EXHAUSTED" in status or "429" in msg or "quota" in msg.lower():
        return msg, "API quota exceeded — reduce request rate or check your quota limits."
    if "UNAUTHENTICATED" in status or "401" in msg:
        return msg, "Authentication failed — check GOOGLE_API_KEY or service account credentials."
    if "UNAVAILABLE" in status or "503" in msg:
        return msg, "Google API is temporarily unavailable — retry after a short delay."
    return msg, None


def _credentials_from_base64(encoded: str) -> Any:
    """Decode a base64 service-account JSON and return scoped credentials.

    Args:
        encoded: Base64-encoded service-account JSON string.

    Returns:
        A ``google.oauth2.service_account.Credentials`` instance scoped for
        Vertex AI.
    """
    from google.oauth2 import service_account

    raw_json = base64.b64decode(encoded)
    info = json.loads(raw_json)
    return service_account.Credentials.from_service_account_info(info, scopes=_VERTEX_SCOPES)


# ---------------------------------------------------------------------------
# Provider class
# ---------------------------------------------------------------------------


class GoogleProvider(ModelProvider):
    """Unified Google LLM provider (Gemini API + Vertex AI).

    Supports two authentication modes depending on which kwargs are present
    in ``ModelConfig``:

    **Gemini API mode** (default):
    Uses a Google AI API key.  Looks for the key in (priority order):

    1. ``config.api_key``
    2. ``GOOGLE_API_KEY`` environment variable
    3. ``GEMINI_API_KEY`` environment variable (legacy)

    Raises :class:`~exo.models.types.ModelError` at construction time if no
    key is found.

    **Vertex AI mode**:
    Activated when ``config.google_project`` or the ``GOOGLE_CLOUD_PROJECT``
    env-var is set.  Additional parameters:

    +-----------------------------------------+-------------------------------------+
    | Config kwarg                            | Env-var fallback                    |
    +=========================================+=====================================+
    | ``google_project``                      | ``GOOGLE_CLOUD_PROJECT``            |
    +-----------------------------------------+-------------------------------------+
    | ``google_location``                     | ``GOOGLE_CLOUD_LOCATION``           |
    +-----------------------------------------+-------------------------------------+
    | ``google_service_account_base64``       | ``GOOGLE_SERVICE_ACCOUNT_BASE64``   |
    +-----------------------------------------+-------------------------------------+

    Args:
        config: Provider connection configuration.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)

        # Determine auth mode: Vertex if a project is configured.
        project = getattr(config, "google_project", None) or os.environ.get(
            "GOOGLE_CLOUD_PROJECT", ""
        )

        if project or getattr(config, "google_location", None):
            # --- Vertex AI mode ---
            # Require a non-empty project; an empty string causes a cryptic auth error
            # deep in the Google SDK rather than a clear user-facing message.
            if not project:
                raise ModelError(
                    "Vertex AI mode requires a Google Cloud project ID. "
                    "Set GOOGLE_CLOUD_PROJECT or pass google_project= to get_provider().",
                    model=f"vertex:{config.model_name}",
                    hint=(
                        "Set the GOOGLE_CLOUD_PROJECT environment variable or pass "
                        "google_project='my-project' to get_provider()."
                    ),
                    context={"model": f"vertex:{config.model_name}"},
                )
            location = getattr(config, "google_location", None) or os.environ.get(
                "GOOGLE_CLOUD_LOCATION", "us-central1"
            )
            sa_b64 = getattr(config, "google_service_account_base64", None) or os.environ.get(
                "GOOGLE_SERVICE_ACCOUNT_BASE64"
            )
            credentials = _credentials_from_base64(sa_b64) if sa_b64 else None
            self._client = genai.Client(
                vertexai=True,
                project=project,
                location=location,
                credentials=credentials,
            )
            self._prefix = "vertex"
        else:
            # --- Gemini API mode ---
            api_key = (
                config.api_key
                or os.environ.get("GOOGLE_API_KEY", "")
                or os.environ.get("GEMINI_API_KEY", "")
            )
            if not api_key:
                raise ModelError(
                    "No API key found for Google provider. "
                    "Set GOOGLE_API_KEY (or GEMINI_API_KEY) or pass api_key= to get_provider().",
                    model=f"gemini:{config.model_name}",
                    hint="Set GOOGLE_API_KEY (or GEMINI_API_KEY) env var or pass api_key= to get_provider().",
                    context={"model": f"gemini:{config.model_name}"},
                )
            self._client = genai.Client(api_key=api_key)
            self._prefix = "gemini"

    async def complete(
        self,
        messages: list[Message],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> ModelResponse:
        """Send a completion request to Google (Gemini or Vertex AI).

        Args:
            messages: Conversation history.
            tools: JSON-schema tool definitions (OpenAI format, auto-converted).
            temperature: Sampling temperature override.
            max_tokens: Maximum output tokens override.

        Returns:
            Normalized model response.

        Raises:
            ModelError: If the API call fails.
        """
        _log.debug(
            "%s complete: model=%s, messages=%d, tools=%d",
            self._prefix,
            self.config.model_name,
            len(messages),
            len(tools or []),
        )
        contents, system_instruction = _to_google_contents(messages)
        config = _build_config(tools, temperature, max_tokens, system_instruction)
        try:
            response = await self._client.aio.models.generate_content(
                model=self.config.model_name,
                contents=contents,
                config=config,
            )
        except genai_errors.APIError as exc:
            _log.error(
                "%s complete failed: model=%s, error=%s",
                self._prefix,
                self.config.model_name,
                exc,
                exc_info=True,
            )
            msg, hint = _google_error_hint(exc)
            raise ModelError(
                msg,
                model=f"{self._prefix}:{self.config.model_name}",
                hint=hint,
                context={
                    "status_code": getattr(exc, "code", None),
                    "status": str(getattr(exc, "code", "")) or None,
                },
            ) from exc
        return _parse_response(response, self.config.model_name)

    async def stream(
        self,
        messages: list[Message],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """Stream a completion from Google (Gemini or Vertex AI).

        Args:
            messages: Conversation history.
            tools: JSON-schema tool definitions (OpenAI format, auto-converted).
            temperature: Sampling temperature override.
            max_tokens: Maximum output tokens override.

        Yields:
            Incremental response chunks.

        Raises:
            ModelError: If the API call fails.
        """
        _log.debug(
            "%s stream: model=%s, messages=%d, tools=%d",
            self._prefix,
            self.config.model_name,
            len(messages),
            len(tools or []),
        )
        contents, system_instruction = _to_google_contents(messages)
        config = _build_config(tools, temperature, max_tokens, system_instruction)
        try:
            async for chunk in await self._client.aio.models.generate_content_stream(
                model=self.config.model_name,
                contents=contents,
                config=config,
            ):
                yield _parse_stream_chunk(chunk)
        except ModelError:
            raise
        except genai_errors.APIError as exc:
            _log.error(
                "%s stream failed: model=%s, error=%s",
                self._prefix,
                self.config.model_name,
                exc,
                exc_info=True,
            )
            msg, hint = _google_error_hint(exc)
            raise ModelError(
                msg,
                model=f"{self._prefix}:{self.config.model_name}",
                hint=hint,
                context={
                    "status_code": getattr(exc, "code", None),
                    "status": str(getattr(exc, "code", "")) or None,
                },
            ) from exc


# Backward-compatible alias
GeminiProvider = GoogleProvider


# ---------------------------------------------------------------------------
# Registration — both prefixes map to the same class
# ---------------------------------------------------------------------------

model_registry.register("gemini", GoogleProvider)
