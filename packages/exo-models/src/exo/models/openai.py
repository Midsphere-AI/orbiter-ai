"""OpenAI LLM provider implementation.

Wraps the ``openai`` SDK to implement ``ModelProvider.complete()`` and
``ModelProvider.stream()`` with normalized response types.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import Any

import openai
from openai import AsyncOpenAI

from exo.config import ModelConfig
from exo.models.provider import ModelProvider, model_registry
from exo.models.types import (
    FinishReason,
    ModelError,
    ModelResponse,
    StreamChunk,
    ToolCallDelta,
)
from exo.types import (
    AssistantMessage,
    AudioBlock,
    ContentBlock,
    DocumentBlock,
    ImageDataBlock,
    ImageURLBlock,
    Message,
    MessageContent,
    SystemMessage,
    TextBlock,
    ToolCall,
    ToolResult,
    Usage,
    UserMessage,
    VideoBlock,
)

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Error helper
# ---------------------------------------------------------------------------


def _openai_error_hint(exc: openai.APIError) -> tuple[str, str | None]:
    """Return (message, hint) for an OpenAI API error."""
    status = getattr(exc, "status_code", None)
    msg = str(exc)
    if status == 401:
        return msg, "Check that OPENAI_API_KEY is set correctly and has not expired."
    if status == 429:
        return msg, "Rate limited — reduce request frequency or add a delay between calls."
    if status == 400:
        body = str(getattr(exc, "body", "") or "")
        if "context" in body.lower() or "too long" in body.lower() or "max_tokens" in body.lower():
            return msg, (
                "Reduce prompt length or increase max_tokens to fit within the model's context window."
            )
        return msg, None
    if status in (500, 502, 503, 529):
        return msg, "The OpenAI API is experiencing issues — retry after a short delay."
    return msg, None


# ---------------------------------------------------------------------------
# Finish reason mapping
# ---------------------------------------------------------------------------

_FINISH_REASON_MAP: dict[str | None, FinishReason] = {
    "stop": "stop",
    "tool_calls": "tool_calls",
    "length": "length",
    "content_filter": "content_filter",
    None: "stop",
}


def _map_finish_reason(raw: str | None) -> FinishReason:
    """Normalize an OpenAI finish reason to a ``FinishReason``."""
    return _FINISH_REASON_MAP.get(raw, "stop")


# ---------------------------------------------------------------------------
# Message conversion
# ---------------------------------------------------------------------------


def _content_blocks_to_openai(blocks: list[ContentBlock]) -> list[dict[str, Any]]:
    """Convert a list of ContentBlock objects to OpenAI content parts.

    Args:
        blocks: List of ContentBlock objects.

    Returns:
        List of OpenAI-format content part dicts.
    """
    parts: list[dict[str, Any]] = []
    for block in blocks:
        if isinstance(block, TextBlock):
            parts.append({"type": "text", "text": block.text})
        elif isinstance(block, ImageURLBlock):
            parts.append(
                {
                    "type": "image_url",
                    "image_url": {"url": block.url, "detail": block.detail},
                }
            )
        elif isinstance(block, ImageDataBlock):
            data_url = f"data:{block.media_type};base64,{block.data}"
            parts.append(
                {
                    "type": "image_url",
                    "image_url": {"url": data_url, "detail": "auto"},
                }
            )
        elif isinstance(block, AudioBlock):
            parts.append(
                {
                    "type": "input_audio",
                    "input_audio": {"data": block.data, "format": block.format},
                }
            )
        elif isinstance(block, VideoBlock):
            _log.warning("OpenAI does not support video input; skipping VideoBlock")
        elif isinstance(block, DocumentBlock):
            _log.warning("OpenAI does not support document input natively; skipping DocumentBlock")
    return parts


def _message_content_to_openai(content: MessageContent) -> str | list[dict[str, Any]]:
    """Convert MessageContent to an OpenAI-compatible content value.

    Args:
        content: A string or list of ContentBlock objects.

    Returns:
        A string (for plain text) or list of content part dicts.
    """
    if isinstance(content, str):
        return content
    return _content_blocks_to_openai(content)


def _to_openai_messages(messages: list[Message]) -> list[dict[str, Any]]:
    """Convert Exo messages to OpenAI chat message dicts.

    For ToolResult messages that contain media content blocks, OpenAI's
    ``role: "tool"`` only accepts strings, so media blocks are collected
    and injected as a synthetic user message after all tool results.

    Args:
        messages: Exo message sequence.

    Returns:
        List of dicts suitable for the OpenAI ``messages`` parameter.
    """
    result: list[dict[str, Any]] = []
    for msg in messages:
        if isinstance(msg, SystemMessage):
            result.append({"role": "system", "content": msg.content})
        elif isinstance(msg, UserMessage):
            result.append({"role": "user", "content": _message_content_to_openai(msg.content)})
        elif isinstance(msg, AssistantMessage):
            entry: dict[str, Any] = {"role": "assistant"}
            if msg.content:
                entry["content"] = _message_content_to_openai(msg.content)
            if msg.tool_calls:
                entry["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.name, "arguments": tc.arguments},
                    }
                    for tc in msg.tool_calls
                ]
            if not msg.content and not msg.tool_calls:
                entry["content"] = ""
            result.append(entry)
        elif isinstance(msg, ToolResult):
            if isinstance(msg.content, list):
                # OpenAI tool role only accepts str — extract text, then collect
                # non-text media blocks for a synthetic follow-up user message.
                if msg.error:
                    text_content = msg.error
                else:
                    # Extract text from any TextBlocks; fall back to placeholder
                    # only when there are no text blocks at all.
                    text_parts = [b.text for b in msg.content if isinstance(b, TextBlock)]
                    text_content = " ".join(text_parts) if text_parts else "[media result]"
                result.append(
                    {
                        "role": "tool",
                        "tool_call_id": msg.tool_call_id,
                        "content": text_content,
                    }
                )
                # Only inject a synthetic user turn for genuinely non-text blocks
                # (images, audio) — text is already carried in the tool role above.
                media_parts = _content_blocks_to_openai(
                    [b for b in msg.content if not isinstance(b, TextBlock)]
                )
                if media_parts:
                    result.append({"role": "user", "content": media_parts})
            else:
                result.append(
                    {
                        "role": "tool",
                        "tool_call_id": msg.tool_call_id,
                        "content": msg.error if msg.error else msg.content,
                    }
                )
    return result


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


def _parse_response(raw: Any, model_name: str) -> ModelResponse:
    """Convert an OpenAI ChatCompletion to a ``ModelResponse``.

    Args:
        raw: The raw OpenAI API response object.
        model_name: The model name for error context.

    Returns:
        A normalized ``ModelResponse``.
    """
    choice = raw.choices[0]
    message = choice.message

    content = message.content or ""
    tool_calls: list[ToolCall] = []
    if message.tool_calls:
        for tc in message.tool_calls:
            tool_calls.append(
                ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=tc.function.arguments,
                )
            )

    # Extract reasoning content from o1/o3 models
    reasoning_content = ""
    extras = getattr(message, "model_extra", None) or {}
    if isinstance(extras, dict) and extras.get("reasoning_content"):
        reasoning_content = extras["reasoning_content"]

    usage = Usage()
    if raw.usage:
        usage = Usage(
            input_tokens=raw.usage.prompt_tokens,
            output_tokens=raw.usage.completion_tokens,
            total_tokens=raw.usage.total_tokens,
        )

    return ModelResponse(
        id=raw.id or "",
        model=raw.model or model_name,
        content=content,
        tool_calls=tool_calls,
        usage=usage,
        finish_reason=_map_finish_reason(choice.finish_reason),
        reasoning_content=reasoning_content,
    )


def _parse_stream_chunk(chunk: Any) -> StreamChunk:
    """Convert a single OpenAI streaming chunk to a ``StreamChunk``.

    Args:
        chunk: A single chunk from the OpenAI streaming response.

    Returns:
        A normalized ``StreamChunk``.
    """
    # Usage-only final chunk (empty choices)
    if not chunk.choices:
        usage = Usage()
        if chunk.usage:
            usage = Usage(
                input_tokens=chunk.usage.prompt_tokens,
                output_tokens=chunk.usage.completion_tokens,
                total_tokens=chunk.usage.total_tokens,
            )
        return StreamChunk(usage=usage)

    choice = chunk.choices[0]
    delta = choice.delta

    text = delta.content or ""
    finish_reason = _map_finish_reason(choice.finish_reason) if choice.finish_reason else None

    tool_call_deltas: list[ToolCallDelta] = []
    if delta.tool_calls:
        for tc_delta in delta.tool_calls:
            tool_call_deltas.append(
                ToolCallDelta(
                    index=tc_delta.index,
                    id=tc_delta.id,
                    name=tc_delta.function.name
                    if tc_delta.function and tc_delta.function.name
                    else None,
                    arguments=tc_delta.function.arguments
                    if tc_delta.function and tc_delta.function.arguments
                    else "",
                )
            )

    return StreamChunk(
        delta=text,
        tool_call_deltas=tool_call_deltas,
        finish_reason=finish_reason,
    )


# ---------------------------------------------------------------------------
# Provider class
# ---------------------------------------------------------------------------


class OpenAIProvider(ModelProvider):
    """OpenAI LLM provider.

    Wraps the ``openai.AsyncOpenAI`` client for chat completions.

    Args:
        config: Provider connection configuration.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        self._client = AsyncOpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            max_retries=config.max_retries,
            timeout=config.timeout,
        )

    async def complete(
        self,
        messages: list[Message],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> ModelResponse:
        """Send a chat completion request to OpenAI.

        Args:
            messages: Conversation history.
            tools: JSON-schema tool definitions.
            temperature: Sampling temperature override.
            max_tokens: Maximum output tokens override.

        Returns:
            Normalized model response.

        Raises:
            ModelError: If the API call fails.
        """
        kwargs = self._build_kwargs(
            messages, tools=tools, temperature=temperature, max_tokens=max_tokens
        )
        _log.debug(
            "openai complete: model=%s, messages=%d, tools=%d",
            self.config.model_name,
            len(messages),
            len(tools or []),
        )
        try:
            response = await self._client.chat.completions.create(**kwargs)
        except openai.APIError as exc:
            _log.error(
                "openai complete failed: model=%s, error=%s",
                self.config.model_name,
                exc,
                exc_info=True,
            )
            msg, hint = _openai_error_hint(exc)
            raise ModelError(
                msg,
                model=f"openai:{self.config.model_name}",
                context={"status_code": getattr(exc, "status_code", None)},
                hint=hint,
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
        """Stream a chat completion from OpenAI.

        Args:
            messages: Conversation history.
            tools: JSON-schema tool definitions.
            temperature: Sampling temperature override.
            max_tokens: Maximum output tokens override.

        Yields:
            Incremental response chunks.

        Raises:
            ModelError: If the API call fails.
        """
        kwargs = self._build_kwargs(
            messages, tools=tools, temperature=temperature, max_tokens=max_tokens
        )
        kwargs["stream"] = True
        kwargs["stream_options"] = {"include_usage": True}
        _log.debug(
            "openai stream: model=%s, messages=%d, tools=%d",
            self.config.model_name,
            len(messages),
            len(tools or []),
        )
        try:
            response = await self._client.chat.completions.create(**kwargs)
            async for chunk in response:
                yield _parse_stream_chunk(chunk)
        except openai.APIError as exc:
            _log.error(
                "openai stream failed: model=%s, error=%s",
                self.config.model_name,
                exc,
                exc_info=True,
            )
            msg, hint = _openai_error_hint(exc)
            raise ModelError(
                msg,
                model=f"openai:{self.config.model_name}",
                context={"status_code": getattr(exc, "status_code", None)},
                hint=hint,
            ) from exc

    def _build_kwargs(
        self,
        messages: list[Message],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        """Build keyword arguments for the OpenAI API call.

        Args:
            messages: Conversation history.
            tools: JSON-schema tool definitions.
            temperature: Sampling temperature.
            max_tokens: Maximum output tokens.

        Returns:
            Dict of kwargs for ``chat.completions.create()``.
        """
        kwargs: dict[str, Any] = {
            "model": self.config.model_name,
            "messages": _to_openai_messages(messages),
        }
        if tools is not None:
            kwargs["tools"] = tools
        if temperature is not None:
            kwargs["temperature"] = temperature
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        # Generic escape hatch: ``extra_create_kwargs`` (an optional dict stored on
        # ModelConfig via ``get_provider(..., extra_create_kwargs={...})``) is sent
        # through the OpenAI SDK's ``extra_body``, so it reaches the wire even though
        # the SDK rejects unknown top-level kwargs. This lets callers pass
        # OpenAI-compatible-endpoint params the typed interface omits -- notably
        # OpenRouter's unified reasoning control, e.g.
        # ``{"reasoning": {"effort": "high"}}``. Merged into any existing extra_body.
        extra = getattr(self.config, "extra_create_kwargs", None)
        if extra:
            kwargs["extra_body"] = {**kwargs.get("extra_body", {}), **extra}
        return kwargs


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

model_registry.register("openai", OpenAIProvider)
