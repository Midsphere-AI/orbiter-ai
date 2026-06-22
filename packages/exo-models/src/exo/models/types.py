"""Provider-agnostic model response types.

These types define the contract between LLM provider implementations
and the agent core. ``ModelResponse`` is returned by ``provider.complete()``,
and ``StreamChunk`` is yielded by ``provider.stream()``.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from exo.types import ExoError, ToolCall, Usage

# ---------------------------------------------------------------------------
# Error
# ---------------------------------------------------------------------------


class ModelError(ExoError):
    """Raised when an LLM provider call fails.

    Args:
        message: Human-readable error description.
        model: The model identifier that caused the error.
        code: Optional error code for classification (e.g. ``"context_length"``
            for context-window overflow, ``"rate_limit"`` for throttling).
        context: Optional dict of developer-facing context (merged with ``model``).
        hint: Optional actionable hint for the developer.
        doc: Optional documentation URL.
    """

    def __init__(
        self,
        message: str,
        *,
        model: str = "",
        code: str = "",
        context: dict | None = None,
        hint: str | None = None,
        doc: str | None = None,
    ) -> None:
        self.model = model
        self.code = code
        full = f"[{model}] {message}" if model else message
        merged_context = {"model": model} if model else {}
        if context:
            merged_context.update(context)
        super().__init__(full, context=merged_context or None, hint=hint, doc=doc)


# ---------------------------------------------------------------------------
# FinishReason
# ---------------------------------------------------------------------------

FinishReason = Literal["stop", "tool_calls", "length", "content_filter"]
"""Why the model stopped generating.

Providers normalize their native values to these four categories:
- ``"stop"``: Natural completion (Anthropic ``"end_turn"``).
- ``"tool_calls"``: Model wants to invoke tools (Anthropic ``"tool_use"``).
- ``"length"``: Hit max tokens limit.
- ``"content_filter"``: Content was filtered by the provider.
"""

# ---------------------------------------------------------------------------
# ModelResponse (return of complete())
# ---------------------------------------------------------------------------


class ModelResponse(BaseModel):
    """Response from a non-streaming LLM completion call.

    Args:
        id: Provider-assigned correlation ID.
        model: Which model produced this response.
        content: Text output from the model.
        tool_calls: Tool invocations requested by the model.
        usage: Token usage statistics.
        finish_reason: Why the model stopped generating.
        reasoning_content: Chain-of-thought content for reasoning models.
    """

    model_config = {"frozen": True}

    id: str = ""
    model: str = ""
    content: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)
    finish_reason: FinishReason = "stop"
    reasoning_content: str = ""


# ---------------------------------------------------------------------------
# Streaming types (yielded by stream())
# ---------------------------------------------------------------------------


class ToolCallDelta(BaseModel):
    """An incremental fragment of a streamed tool call.

    Args:
        index: Position in multi-tool-call responses.
        id: Tool call ID, present only in the first chunk.
        name: Tool name, present only in the first chunk.
        arguments: Incremental JSON fragment of arguments.
        thought_signature: Opaque signature from thinking models (e.g. Gemini)
            that must be echoed back in conversation history.
    """

    model_config = {"frozen": True}

    index: int = 0
    id: str | None = None
    name: str | None = None
    arguments: str = ""
    thought_signature: bytes | None = None


class StreamChunk(BaseModel):
    """A single chunk yielded during streaming LLM completion.

    Args:
        delta: Incremental text content.
        tool_call_deltas: Incremental tool call fragments.
        finish_reason: Non-None only on the final chunk.
        usage: Token usage, typically only on the final chunk.
        reasoning_content: Chain-of-thought content accumulated by the final
            chunk (non-None only when extended thinking is active).
    """

    model_config = {"frozen": True}

    delta: str = ""
    tool_call_deltas: list[ToolCallDelta] = Field(default_factory=list)
    finish_reason: FinishReason | None = None
    usage: Usage = Field(default_factory=Usage)
    reasoning_content: str | None = None
