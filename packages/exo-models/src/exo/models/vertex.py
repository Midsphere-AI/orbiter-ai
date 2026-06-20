"""Google Vertex AI LLM provider implementation.

Wraps the ``google-genai`` SDK with Vertex AI authentication to implement
``ModelProvider.complete()`` and ``ModelProvider.stream()`` with normalized
response types.

Connection parameters (``google_project``, ``google_location``,
``google_service_account_base64``) can be supplied as ``ModelConfig``
extras — ideal for distributed workers — or via the corresponding
environment variables as a fallback.
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
    _convert_tools,
    _map_finish_reason,
    _parse_response,
    _parse_stream_chunk,
    _to_google_contents,
)
from exo.models.provider import ModelProvider, model_registry
from exo.models.types import ModelError, StreamChunk
from exo.types import Message

_log = logging.getLogger(__name__)

_VERTEX_SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]

# Re-export shared helpers so existing imports from exo.models.vertex still work.
__all__ = [
    "VertexProvider",
    "_build_config",
    "_convert_tools",
    "_credentials_from_base64",
    "_map_finish_reason",
    "_parse_response",
    "_parse_stream_chunk",
    "_to_google_contents",
]


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


class VertexProvider(ModelProvider):
    """Google Vertex AI LLM provider.

    Wraps the ``google.genai.Client`` with Vertex AI authentication.

    All connection parameters can be supplied directly via ``ModelConfig``
    extras (preferred for distributed workers) **or** via environment
    variables (fallback):

    +-----------------------------------------+-------------------------------------+
    | Config kwarg                            | Env-var fallback                    |
    +=========================================+=====================================+
    | ``google_project``                      | ``GOOGLE_CLOUD_PROJECT``            |
    +-----------------------------------------+-------------------------------------+
    | ``google_location``                     | ``GOOGLE_CLOUD_LOCATION``           |
    +-----------------------------------------+-------------------------------------+
    | ``google_service_account_base64``       | ``GOOGLE_SERVICE_ACCOUNT_BASE64``   |
    +-----------------------------------------+-------------------------------------+

    When a service-account base64 string is provided (via either path), it
    is decoded and used to build explicit credentials.  Otherwise,
    Application Default Credentials (ADC) are used.

    Args:
        config: Provider connection configuration.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        project = getattr(config, "google_project", None) or os.environ.get(
            "GOOGLE_CLOUD_PROJECT", ""
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

    async def complete(
        self,
        messages: list[Message],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Any:
        """Send a completion request to Vertex AI.

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
            "vertex complete: model=%s, messages=%d, tools=%d",
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
                "vertex complete failed: model=%s, error=%s",
                self.config.model_name,
                exc,
                exc_info=True,
            )
            raise ModelError(str(exc), model=f"vertex:{self.config.model_name}") from exc
        return _parse_response(response, self.config.model_name)

    async def stream(
        self,
        messages: list[Message],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """Stream a completion from Vertex AI.

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
            "vertex stream: model=%s, messages=%d, tools=%d",
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
                "vertex stream failed: model=%s, error=%s",
                self.config.model_name,
                exc,
                exc_info=True,
            )
            raise ModelError(str(exc), model=f"vertex:{self.config.model_name}") from exc


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

model_registry.register("vertex", VertexProvider)
