"""Typed event input models for the agent rail system.

These Pydantic models provide structured, validated inputs for each
hook event in the agent lifecycle. They are intentionally **not frozen**
so that hooks/rails may mutate inputs before they reach the next handler.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from exo.hooks import HookPoint


class InvokeInputs(BaseModel):
    """Structured inputs for agent invocation events (START / FINISHED).

    Args:
        input: The user query or prompt string.
        messages: Prior conversation messages, if any.
        result: The final result of the invocation (populated on FINISHED).
    """

    input: str
    messages: list[Any] | None = None
    result: Any | None = None


class ModelCallInputs(BaseModel):
    """Structured inputs for LLM call events (PRE_LLM_CALL / POST_LLM_CALL).

    Args:
        messages: The message list sent to the model.
        tools: Tool definitions provided to the model, if any.
        response: The model response object (populated on POST_LLM_CALL).
        usage: Token usage statistics (populated on POST_LLM_CALL).
    """

    messages: list[Any]
    tools: list[dict[str, Any]] | None = None
    response: Any | None = None
    usage: Any | None = None


class ToolCallInputs(BaseModel):
    """Structured inputs for tool call events (PRE_TOOL_CALL / POST_TOOL_CALL).

    Args:
        tool_name: Name of the tool being invoked.
        arguments: The arguments passed to the tool.
        result: The tool execution result (populated on POST_TOOL_CALL).
        metadata: Additional metadata about the tool call.
    """

    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    result: Any | None = None
    metadata: Any | None = None


class GuardContext(BaseModel):
    """Context object passed to guard handlers.

    Bundles the agent reference, the lifecycle event, the typed inputs
    for that event, and an open-ended extra dict for cross-guard state.

    Args:
        agent: Reference to the agent instance.
        event: The lifecycle hook point that triggered this guard.
        inputs: Typed inputs specific to the event kind.
        extra: Open-ended dict for cross-guard state sharing.
    """

    agent: Any
    event: HookPoint
    inputs: InvokeInputs | ModelCallInputs | ToolCallInputs
    extra: dict[str, Any] = Field(default_factory=dict)


# Deprecated alias: use GuardContext
RailContext = GuardContext
