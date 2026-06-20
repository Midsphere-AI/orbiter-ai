"""Google Gemini LLM provider implementation.

Wraps the ``google-genai`` SDK to implement ``ModelProvider.complete()`` and
``ModelProvider.stream()`` with normalized response types.
"""

from __future__ import annotations

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

# Re-export shared helpers so existing imports from exo.models.gemini still work.
__all__ = [
    "GeminiProvider",
    "_build_config",
    "_convert_tools",
    "_map_finish_reason",
    "_parse_response",
    "_parse_stream_chunk",
    "_to_google_contents",
]


# ---------------------------------------------------------------------------
# Provider class
# ---------------------------------------------------------------------------


class GeminiProvider(ModelProvider):
    """Google Gemini LLM provider.

    Wraps the ``google.genai.Client`` for Gemini API completions.

    Args:
        config: Provider connection configuration.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        api_key = config.api_key or os.environ.get("GEMINI_API_KEY", "")
        self._client = genai.Client(api_key=api_key or "dummy")

    async def complete(
        self,
        messages: list[Message],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Any:
        """Send a completion request to Google Gemini.

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
            "gemini complete: model=%s, messages=%d, tools=%d",
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
                "gemini complete failed: model=%s, error=%s",
                self.config.model_name,
                exc,
                exc_info=True,
            )
            raise ModelError(str(exc), model=f"gemini:{self.config.model_name}") from exc
        return _parse_response(response, self.config.model_name)

    async def stream(
        self,
        messages: list[Message],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """Stream a completion from Google Gemini.

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
            "gemini stream: model=%s, messages=%d, tools=%d",
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
                "gemini stream failed: model=%s, error=%s",
                self.config.model_name,
                exc,
                exc_info=True,
            )
            raise ModelError(str(exc), model=f"gemini:{self.config.model_name}") from exc


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

model_registry.register("gemini", GeminiProvider)
