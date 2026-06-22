"""Anthropic LLM provider implementation.

Wraps the ``anthropic`` SDK to implement ``ModelProvider.complete()`` and
``ModelProvider.stream()`` with normalized response types.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import AsyncIterator
from typing import Any

import anthropic
from anthropic import AsyncAnthropic

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

_DEFAULT_MAX_TOKENS = 4096


# ---------------------------------------------------------------------------
# Error helper
# ---------------------------------------------------------------------------


def _anthropic_error_hint(exc: anthropic.APIError) -> tuple[str, str | None]:
    """Return (message, hint) for an Anthropic API error."""
    status = getattr(exc, "status_code", None)
    msg = str(exc)
    if status == 401:
        return msg, "Check that ANTHROPIC_API_KEY is set correctly and has not expired."
    if status == 429:
        return msg, "Rate limited — reduce request frequency or add a delay between calls."
    if status == 400:
        body = str(getattr(exc, "body", "") or "")
        if "token" in body.lower() or "context" in body.lower():
            return msg, "Reduce prompt length or increase max_tokens."
        return msg, None
    if status in (500, 502, 503, 529):
        return msg, "Anthropic API is overloaded — retry after a short delay."
    return msg, None


# ---------------------------------------------------------------------------
# Stop reason mapping
# ---------------------------------------------------------------------------

_STOP_REASON_MAP: dict[str | None, FinishReason] = {
    "end_turn": "stop",
    "tool_use": "tool_calls",
    "max_tokens": "length",
    "stop_sequence": "stop",
    None: "stop",
}


def _map_stop_reason(raw: str | None) -> FinishReason:
    """Normalize an Anthropic stop reason to a ``FinishReason``."""
    return _STOP_REASON_MAP.get(raw, "stop")


# ---------------------------------------------------------------------------
# Message conversion
# ---------------------------------------------------------------------------


def _content_blocks_to_anthropic(blocks: list[ContentBlock]) -> list[dict[str, Any]]:
    """Convert a list of ContentBlock objects to Anthropic content parts.

    Args:
        blocks: List of ContentBlock objects.

    Returns:
        List of Anthropic-format content part dicts.
    """
    parts: list[dict[str, Any]] = []
    for block in blocks:
        if isinstance(block, TextBlock):
            parts.append({"type": "text", "text": block.text})
        elif isinstance(block, ImageURLBlock):
            parts.append({"type": "image", "source": {"type": "url", "url": block.url}})
        elif isinstance(block, ImageDataBlock):
            parts.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": block.media_type,
                        "data": block.data,
                    },
                }
            )
        elif isinstance(block, DocumentBlock):
            doc: dict[str, Any] = {
                "type": "document",
                "source": {
                    "type": "base64",
                    "media_type": block.media_type,
                    "data": block.data,
                },
            }
            if block.title:
                doc["title"] = block.title
            parts.append(doc)
        elif isinstance(block, AudioBlock):
            raise ModelError(
                "Anthropic does not support audio input; remove AudioBlock before calling this provider",
                model="anthropic",
                hint="Convert to a text description or use a provider that supports audio (e.g. openai:gpt-4o-audio-preview).",
            )
        elif isinstance(block, VideoBlock):
            raise ModelError(
                "Anthropic does not support video input; remove VideoBlock before calling this provider",
                model="anthropic",
                hint="Convert to a text description or use a provider that supports video (e.g. gemini:gemini-2.0-flash).",
            )
    return parts


def _message_content_to_anthropic(content: MessageContent) -> str | list[dict[str, Any]]:
    """Convert MessageContent to an Anthropic-compatible content value.

    Args:
        content: A string or list of ContentBlock objects.

    Returns:
        A string (for plain text) or list of content part dicts.
    """
    if isinstance(content, str):
        return content
    return _content_blocks_to_anthropic(content)


def _merge_same_role(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge consecutive messages with the same role into one.

    Anthropic requires strict user/assistant alternation.  When two adjacent
    messages share a role (e.g. two ``user`` turns produced by a tool-result
    followed by a plain user message) their ``content`` lists are concatenated
    into a single message so the final sequence strictly alternates.

    Args:
        messages: Anthropic-format message list (no system messages).

    Returns:
        New list where no two adjacent messages share a role.
    """
    merged: list[dict[str, Any]] = []
    for msg in messages:
        if merged and merged[-1]["role"] == msg["role"]:
            prev = merged[-1]
            # Normalise the previous content to a list if it is still a string
            if isinstance(prev["content"], str):
                prev["content"] = [{"type": "text", "text": prev["content"]}]
            # Normalise the incoming content to a list
            incoming: list[dict[str, Any]]
            if isinstance(msg["content"], str):
                incoming = [{"type": "text", "text": msg["content"]}]
            else:
                incoming = list(msg["content"])
            prev["content"].extend(incoming)
        else:
            # Shallow-copy so callers don't share structure
            merged.append(dict(msg))
    return merged


def _build_messages(messages: list[Message]) -> tuple[str, list[dict[str, Any]]]:
    """Convert Exo messages to Anthropic format.

    Extracts system messages into a single system string (Anthropic takes
    ``system=`` as a separate kwarg). Consecutive tool results are merged
    into one ``user`` message to maintain strict alternation.  A final
    alternation pass ensures the output sequence never has two adjacent
    messages with the same role (which the Anthropic API rejects).

    Args:
        messages: Exo message sequence.

    Returns:
        A ``(system, messages)`` tuple for the Anthropic API.
    """
    system_parts: list[str] = []
    result: list[dict[str, Any]] = []

    for msg in messages:
        if isinstance(msg, SystemMessage):
            system_parts.append(msg.content)
        elif isinstance(msg, UserMessage):
            result.append({"role": "user", "content": _message_content_to_anthropic(msg.content)})
        elif isinstance(msg, AssistantMessage):
            content: list[dict[str, Any]] = []
            if msg.content:
                if isinstance(msg.content, str):
                    content.append({"type": "text", "text": msg.content})
                else:
                    content.extend(_content_blocks_to_anthropic(msg.content))
            for tc in msg.tool_calls:
                if tc.arguments:
                    try:
                        tc_input = json.loads(tc.arguments)
                    except json.JSONDecodeError as exc:
                        raise ModelError(
                            f"Failed to parse tool call arguments for '{tc.name}': {exc}",
                            model="anthropic",
                            hint="Ensure tool call arguments are valid JSON.",
                            context={"tool_name": tc.name, "arguments": tc.arguments},
                        ) from exc
                else:
                    tc_input = {}
                content.append(
                    {
                        "type": "tool_use",
                        "id": tc.id,
                        "name": tc.name,
                        "input": tc_input,
                    }
                )
            if not content:
                content.append({"type": "text", "text": ""})
            result.append({"role": "assistant", "content": content})
        elif isinstance(msg, ToolResult):
            # Anthropic natively supports image blocks inside tool_result content
            if isinstance(msg.content, list):
                tool_content: str | list[dict[str, Any]] = _content_blocks_to_anthropic(msg.content)
            else:
                tool_content = msg.error if msg.error else msg.content
            tool_block: dict[str, Any] = {
                "type": "tool_result",
                "tool_use_id": msg.tool_call_id,
                "content": tool_content,
            }
            if msg.error:
                tool_block["is_error"] = True
            # Merge consecutive tool results into one user message
            if result and result[-1]["role"] == "user" and isinstance(result[-1]["content"], list):
                result[-1]["content"].append(tool_block)
            else:
                result.append({"role": "user", "content": [tool_block]})

    # Enforce strict user/assistant alternation required by Anthropic API
    return "\n".join(system_parts), _merge_same_role(result)


def _convert_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert OpenAI-format tool schemas to Anthropic format.

    Args:
        tools: List of OpenAI-style tool definitions.

    Returns:
        List of Anthropic-style tool definitions.
    """
    converted: list[dict[str, Any]] = []
    for t in tools:
        fn = t.get("function", {})
        converted.append(
            {
                "name": fn.get("name", ""),
                "description": fn.get("description", ""),
                "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
            }
        )
    return converted


_CACHE_CONTROL = {"type": "ephemeral"}


def _apply_cache_breakpoints(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Add Anthropic prompt-caching ``cache_control`` markers to API kwargs.

    Places up to 4 breakpoints (the Anthropic maximum):
    1. System prompt (last content block)
    2. Last tool definition
    3. Last 2 user-turn messages (rolling cache frontier)

    Mutates *kwargs* in place and returns it.  Safe because
    ``_build_kwargs`` builds fresh dicts each call.
    """
    # 1. System prompt — convert plain string to content-block list
    system = kwargs.get("system")
    if system and isinstance(system, str):
        kwargs["system"] = [{"type": "text", "text": system, "cache_control": _CACHE_CONTROL}]
    elif isinstance(system, list) and system:
        system[-1]["cache_control"] = _CACHE_CONTROL

    # 2. Last tool definition
    tools = kwargs.get("tools")
    if tools:
        tools[-1]["cache_control"] = _CACHE_CONTROL

    # 3. Last 2 user-turn messages
    messages = kwargs.get("messages", [])
    user_marked = 0
    for msg in reversed(messages):
        if user_marked >= 2:
            break
        if msg.get("role") != "user":
            continue
        content = msg["content"]
        if isinstance(content, str):
            msg["content"] = [{"type": "text", "text": content, "cache_control": _CACHE_CONTROL}]
        elif isinstance(content, list) and content:
            content[-1]["cache_control"] = _CACHE_CONTROL
        user_marked += 1

    return kwargs


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


def _parse_response(raw: Any, model_name: str) -> ModelResponse:
    """Convert an Anthropic Message to a ``ModelResponse``.

    Args:
        raw: The raw Anthropic API response object.
        model_name: The model name for context.

    Returns:
        A normalized ``ModelResponse``.
    """
    content_parts: list[str] = []
    tool_calls: list[ToolCall] = []
    reasoning_content = ""

    for block in raw.content:
        block_type = getattr(block, "type", None)
        if block_type == "text":
            content_parts.append(block.text)
        elif block_type == "tool_use":
            tool_calls.append(
                ToolCall(
                    id=block.id,
                    name=block.name,
                    arguments=json.dumps(block.input) if block.input else "",
                )
            )
        elif block_type == "thinking":
            reasoning_content = getattr(block, "thinking", "")

    usage = Usage()
    if raw.usage:
        usage = Usage(
            input_tokens=raw.usage.input_tokens,
            output_tokens=raw.usage.output_tokens,
            total_tokens=raw.usage.input_tokens + raw.usage.output_tokens,
        )

    return ModelResponse(
        id=raw.id or "",
        model=raw.model or model_name,
        content="".join(content_parts),
        tool_calls=tool_calls,
        usage=usage,
        finish_reason=_map_stop_reason(raw.stop_reason),
        reasoning_content=reasoning_content,
    )


# ---------------------------------------------------------------------------
# Provider class
# ---------------------------------------------------------------------------


class AnthropicProvider(ModelProvider):
    """Anthropic LLM provider.

    Wraps the ``anthropic.AsyncAnthropic`` client for message completions.

    Args:
        config: Provider connection configuration.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        api_key = config.api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            raise ModelError(
                "No API key found for Anthropic provider. "
                "Set ANTHROPIC_API_KEY or pass api_key= to get_provider().",
                model=f"anthropic:{config.model_name}",
                hint="Set ANTHROPIC_API_KEY env var or pass api_key= to get_provider().",
                context={"model": f"anthropic:{config.model_name}"},
            )
        self._client = AsyncAnthropic(
            api_key=api_key,
            base_url=config.base_url,
            max_retries=config.max_retries,
            timeout=config.timeout,
        )
        # ``use_cache`` can be set to False to disable prompt-cache breakpoints.
        self._use_cache: bool = bool(getattr(config, "use_cache", True))

    async def complete(
        self,
        messages: list[Message],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> ModelResponse:
        """Send a message completion request to Anthropic.

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
        kwargs = self._build_kwargs(
            messages, tools=tools, temperature=temperature, max_tokens=max_tokens
        )
        _log.debug(
            "anthropic complete: model=%s, messages=%d, tools=%d",
            self.config.model_name,
            len(messages),
            len(tools or []),
        )
        try:
            response = await self._client.messages.create(**kwargs)
        except anthropic.APIError as exc:
            _log.error(
                "anthropic complete failed: model=%s, error=%s",
                self.config.model_name,
                exc,
                exc_info=True,
            )
            msg, hint = _anthropic_error_hint(exc)
            raise ModelError(
                msg,
                model=f"anthropic:{self.config.model_name}",
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
        """Stream a message completion from Anthropic.

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
        kwargs = self._build_kwargs(
            messages, tools=tools, temperature=temperature, max_tokens=max_tokens
        )
        kwargs["stream"] = True
        input_tokens = 0
        reasoning_parts: list[str] = []
        _log.debug(
            "anthropic stream: model=%s, messages=%d, tools=%d",
            self.config.model_name,
            len(messages),
            len(tools or []),
        )
        response = None
        try:
            response = await self._client.messages.create(**kwargs)
            async for event in response:
                event_type = event.type
                if event_type == "message_start":
                    if hasattr(event, "message") and hasattr(event.message, "usage"):
                        input_tokens = event.message.usage.input_tokens
                elif event_type == "content_block_start":
                    block = event.content_block
                    block_type = getattr(block, "type", None)
                    if block_type == "tool_use":
                        yield StreamChunk(
                            tool_call_deltas=[
                                ToolCallDelta(
                                    index=event.index,
                                    id=block.id,
                                    name=block.name,
                                )
                            ]
                        )
                    # thinking blocks: no delta to emit at start, just track
                elif event_type == "content_block_delta":
                    delta = event.delta
                    delta_type = getattr(delta, "type", None)
                    if delta_type == "text_delta":
                        yield StreamChunk(delta=delta.text)
                    elif delta_type == "input_json_delta":
                        yield StreamChunk(
                            tool_call_deltas=[
                                ToolCallDelta(
                                    index=event.index,
                                    arguments=delta.partial_json,
                                )
                            ]
                        )
                    elif delta_type == "thinking_delta":
                        # Accumulate reasoning content — emitted in the final chunk
                        reasoning_parts.append(getattr(delta, "thinking", "") or "")
                elif event_type == "message_delta":
                    output_tokens = 0
                    if hasattr(event, "usage") and event.usage:
                        output_tokens = event.usage.output_tokens
                    total = input_tokens + output_tokens
                    stop_reason = getattr(event.delta, "stop_reason", None)
                    yield StreamChunk(
                        finish_reason=_map_stop_reason(stop_reason),
                        usage=Usage(
                            input_tokens=input_tokens,
                            output_tokens=output_tokens,
                            total_tokens=total,
                        ),
                        reasoning_content="".join(reasoning_parts) if reasoning_parts else None,
                    )
        except anthropic.APIError as exc:
            _log.error(
                "anthropic stream failed: model=%s, error=%s",
                self.config.model_name,
                exc,
                exc_info=True,
            )
            msg, hint = _anthropic_error_hint(exc)
            raise ModelError(
                msg,
                model=f"anthropic:{self.config.model_name}",
                context={"status_code": getattr(exc, "status_code", None)},
                hint=hint,
            ) from exc
        finally:
            if response is not None and hasattr(response, "close"):
                await response.close()

    def _build_kwargs(
        self,
        messages: list[Message],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        """Build keyword arguments for the Anthropic API call.

        Args:
            messages: Conversation history.
            tools: JSON-schema tool definitions.
            temperature: Sampling temperature.
            max_tokens: Maximum output tokens.

        Returns:
            Dict of kwargs for ``messages.create()``.
        """
        system, converted = _build_messages(messages)
        kwargs: dict[str, Any] = {
            "model": self.config.model_name,
            "messages": converted,
            "max_tokens": max_tokens or _DEFAULT_MAX_TOKENS,
        }
        if system:
            kwargs["system"] = system
        if tools is not None:
            kwargs["tools"] = _convert_tools(tools)
        if temperature is not None:
            kwargs["temperature"] = temperature
        # Generic escape hatch: ``extra_create_kwargs`` (an optional dict stored on
        # ModelConfig via ``get_provider(..., extra_create_kwargs={...})``) is merged
        # verbatim into the ``messages.create()`` call. This lets callers reach
        # Anthropic-only params the typed interface does not expose -- notably
        # extended thinking, e.g. ``{"thinking": {"type": "enabled",
        # "budget_tokens": 8000}}``. Merged last, so the caller takes precedence;
        # the caller is responsible for honouring API constraints (e.g. thinking
        # requires ``max_tokens > budget_tokens`` and no custom ``temperature``).
        extra = getattr(self.config, "extra_create_kwargs", None)
        if extra:
            kwargs.update(extra)
        if self._use_cache:
            return _apply_cache_breakpoints(kwargs)
        return kwargs


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

model_registry.register("anthropic", AnthropicProvider)
