"""Shared helpers for Google Gemini and Vertex AI provider implementations.

Both ``GeminiProvider`` and ``VertexProvider`` use the same ``google-genai``
SDK under the hood.  This module centralises all of the conversion logic so
that ``gemini.py`` and ``vertex.py`` only contain provider-specific
initialisation code.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from exo.models._media import content_blocks_to_google
from exo.models.types import (
    FinishReason,
    ModelResponse,
    StreamChunk,
    ToolCallDelta,
)
from exo.types import (
    AssistantMessage,
    Message,
    MessageContent,
    SystemMessage,
    ToolCall,
    ToolResult,
    Usage,
    UserMessage,
)

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Finish reason mapping
# ---------------------------------------------------------------------------

_FINISH_REASON_MAP: dict[str | None, FinishReason] = {
    "STOP": "stop",
    "MAX_TOKENS": "length",
    "SAFETY": "content_filter",
    "RECITATION": "content_filter",
    "BLOCKLIST": "content_filter",
    "MALFORMED_FUNCTION_CALL": "stop",
    "OTHER": "stop",
    None: "stop",
}


def _map_finish_reason(raw: str | None) -> FinishReason:
    """Normalize a Google finish reason to a ``FinishReason``."""
    return _FINISH_REASON_MAP.get(raw, "stop")


# ---------------------------------------------------------------------------
# Message conversion
# ---------------------------------------------------------------------------


def _user_parts_from_content(content: MessageContent) -> list[dict[str, Any]]:
    """Convert MessageContent to Google API parts for a user message.

    Args:
        content: A string or list of ContentBlock objects.

    Returns:
        List of Google-format part dicts.
    """
    if isinstance(content, str):
        return [{"text": content}]
    return content_blocks_to_google(content)


def _to_google_contents(messages: list[Message]) -> tuple[list[dict[str, Any]], str]:
    """Convert Exo messages to Google API format.

    Extracts system messages into a separate system instruction string.

    Gemini requires that the number of ``function_response`` parts in a user
    turn exactly matches the number of ``function_call`` parts in the
    preceding model turn.  Consecutive :class:`~exo.types.ToolResult`
    messages are therefore merged into a single ``role: user`` entry so that
    all responses for one model turn travel together.

    Args:
        messages: Exo message sequence.

    Returns:
        A ``(contents, system_instruction)`` tuple.
    """
    system_parts: list[str] = []
    contents: list[dict[str, Any]] = []
    pending_tool_parts: list[dict[str, Any]] = []

    def _flush_tool_parts() -> None:
        """Emit accumulated function_response parts as one user turn."""
        if pending_tool_parts:
            contents.append({"role": "user", "parts": list(pending_tool_parts)})
            pending_tool_parts.clear()

    for msg in messages:
        if isinstance(msg, SystemMessage):
            _flush_tool_parts()
            system_parts.append(msg.content)
        elif isinstance(msg, UserMessage):
            _flush_tool_parts()
            contents.append({"role": "user", "parts": _user_parts_from_content(msg.content)})
        elif isinstance(msg, AssistantMessage):
            _flush_tool_parts()
            parts: list[dict[str, Any]] = []
            if msg.content:
                if isinstance(msg.content, str):
                    parts.append({"text": msg.content})
                else:
                    parts.extend(content_blocks_to_google(msg.content))
            for tc in msg.tool_calls:
                args = json.loads(tc.arguments) if tc.arguments else {}
                fc_part: dict[str, Any] = {"function_call": {"name": tc.name, "args": args}}
                if tc.thought_signature is not None:
                    fc_part["thought_signature"] = tc.thought_signature
                parts.append(fc_part)
            if not parts:
                parts.append({"text": ""})
            contents.append({"role": "model", "parts": parts})
        elif isinstance(msg, ToolResult):
            # Build function_response part; append media parts alongside it
            if isinstance(msg.content, list):
                response_data: Any = content_blocks_to_google(msg.content)
            else:
                response_data = msg.error if msg.error else msg.content
            function_response_part: dict[str, Any] = {
                "function_response": {
                    "name": msg.tool_name,
                    "response": {"content": response_data},
                },
            }
            media_parts: list[dict[str, Any]] = []
            if isinstance(msg.content, list):
                media_parts = content_blocks_to_google(msg.content)
            pending_tool_parts.extend([function_response_part, *media_parts])

    # Flush any trailing tool results
    _flush_tool_parts()

    return contents, "\n".join(system_parts)


# ---------------------------------------------------------------------------
# Tool conversion
# ---------------------------------------------------------------------------


def _convert_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert OpenAI-format tool schemas to Google format.

    Args:
        tools: List of OpenAI-style tool definitions.

    Returns:
        List of Google-style tool definitions with function_declarations.
    """
    declarations: list[dict[str, Any]] = []
    for t in tools:
        fn = t.get("function", {})
        declarations.append(
            {
                "name": fn.get("name", ""),
                "description": fn.get("description", ""),
                "parameters": fn.get("parameters", {"type": "object", "properties": {}}),
            }
        )
    return [{"function_declarations": declarations}] if declarations else []


# ---------------------------------------------------------------------------
# Config builder
# ---------------------------------------------------------------------------


def _build_config(
    tools: list[dict[str, Any]] | None,
    temperature: float | None,
    max_tokens: int | None,
    system_instruction: str,
) -> dict[str, Any]:
    """Build a config dict for ``generate_content()``.

    Args:
        tools: OpenAI-format tool definitions (will be converted).
        temperature: Sampling temperature.
        max_tokens: Maximum output tokens.
        system_instruction: System prompt text.

    Returns:
        Dict suitable for the ``config`` parameter.
    """
    config: dict[str, Any] = {}
    if system_instruction:
        config["system_instruction"] = system_instruction
    if temperature is not None:
        config["temperature"] = temperature
    if max_tokens is not None:
        config["max_output_tokens"] = max_tokens
    if tools is not None:
        config["tools"] = _convert_tools(tools)
    return config


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


def _parse_response(raw: Any, model_name: str) -> ModelResponse:
    """Convert a Google GenerateContentResponse to a ``ModelResponse``.

    Args:
        raw: The raw Google API response object.
        model_name: The model name for context.

    Returns:
        A normalized ``ModelResponse``.
    """
    candidate = raw.candidates[0]
    content_parts: list[str] = []
    tool_calls: list[ToolCall] = []

    for i, part in enumerate(candidate.content.parts):
        text = getattr(part, "text", None)
        if text and not getattr(part, "thought", False):
            content_parts.append(text)
        fc = getattr(part, "function_call", None)
        if fc:
            call_id = getattr(fc, "id", None) or f"call_{i}"
            thought_sig = getattr(part, "thought_signature", None)
            tool_calls.append(
                ToolCall(
                    id=call_id,
                    name=fc.name,
                    arguments=json.dumps(fc.args) if fc.args else "{}",
                    thought_signature=thought_sig,
                )
            )

    finish_reason_raw = str(candidate.finish_reason) if candidate.finish_reason else None

    usage = Usage()
    if raw.usage_metadata:
        usage = Usage(
            input_tokens=raw.usage_metadata.prompt_token_count or 0,
            output_tokens=raw.usage_metadata.candidates_token_count or 0,
            total_tokens=raw.usage_metadata.total_token_count or 0,
        )

    return ModelResponse(
        id="",
        model=model_name,
        content="".join(content_parts),
        tool_calls=tool_calls,
        usage=usage,
        finish_reason=_map_finish_reason(finish_reason_raw),
    )


def _parse_stream_chunk(chunk: Any) -> StreamChunk:
    """Convert a single Google streaming chunk to a ``StreamChunk``.

    Args:
        chunk: A single chunk from the streaming response.

    Returns:
        A normalized ``StreamChunk``.
    """
    if not chunk.candidates:
        usage = Usage()
        if chunk.usage_metadata:
            usage = Usage(
                input_tokens=chunk.usage_metadata.prompt_token_count or 0,
                output_tokens=chunk.usage_metadata.candidates_token_count or 0,
                total_tokens=chunk.usage_metadata.total_token_count or 0,
            )
        return StreamChunk(usage=usage)

    candidate = chunk.candidates[0]
    text_parts: list[str] = []
    tool_call_deltas: list[ToolCallDelta] = []

    parts = (candidate.content.parts if candidate.content else None) or []
    for i, part in enumerate(parts):
        text = getattr(part, "text", None)
        if text and not getattr(part, "thought", False):
            text_parts.append(text)
        fc = getattr(part, "function_call", None)
        if fc:
            call_id = getattr(fc, "id", None) or f"call_{i}"
            thought_sig = getattr(part, "thought_signature", None)
            tool_call_deltas.append(
                ToolCallDelta(
                    index=i,
                    id=call_id,
                    name=fc.name,
                    arguments=json.dumps(fc.args) if fc.args else "{}",
                    thought_signature=thought_sig,
                )
            )

    finish_reason_raw = str(candidate.finish_reason) if candidate.finish_reason else None
    finish = _map_finish_reason(finish_reason_raw) if finish_reason_raw else None

    usage = Usage()
    if chunk.usage_metadata:
        usage = Usage(
            input_tokens=chunk.usage_metadata.prompt_token_count or 0,
            output_tokens=chunk.usage_metadata.candidates_token_count or 0,
            total_tokens=chunk.usage_metadata.total_token_count or 0,
        )

    return StreamChunk(
        delta="".join(text_parts),
        tool_call_deltas=tool_call_deltas,
        finish_reason=finish,
        usage=usage,
    )
