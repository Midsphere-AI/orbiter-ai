"""Core message types for the Exo framework."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field


class ExoError(Exception):
    """Base exception for all Exo errors."""


# ---------------------------------------------------------------------------
# Content block types for multimodal messages
# ---------------------------------------------------------------------------


class TextBlock(BaseModel):
    """A plain-text content block."""

    model_config = {"frozen": True}

    type: Literal["text"] = "text"
    text: str


class ImageURLBlock(BaseModel):
    """An image referenced by URL (https:// or data: URI)."""

    model_config = {"frozen": True}

    type: Literal["image_url"] = "image_url"
    url: str
    detail: Literal["auto", "low", "high"] = "auto"


class ImageDataBlock(BaseModel):
    """An image provided as raw base64-encoded bytes."""

    model_config = {"frozen": True}

    type: Literal["image_data"] = "image_data"
    data: str
    media_type: str = "image/jpeg"


class AudioBlock(BaseModel):
    """An audio clip provided as raw base64-encoded bytes."""

    model_config = {"frozen": True}

    type: Literal["audio"] = "audio"
    data: str
    format: str = "mp3"


class VideoBlock(BaseModel):
    """A video clip provided as base64 bytes or a URL."""

    model_config = {"frozen": True}

    type: Literal["video"] = "video"
    data: str | None = None
    url: str | None = None
    media_type: str = "video/mp4"


class DocumentBlock(BaseModel):
    """A document (e.g. PDF) provided as raw base64-encoded bytes."""

    model_config = {"frozen": True}

    type: Literal["document"] = "document"
    data: str
    media_type: str = "application/pdf"
    title: str | None = None


ContentBlock = Annotated[
    TextBlock | ImageURLBlock | ImageDataBlock | AudioBlock | VideoBlock | DocumentBlock,
    Field(discriminator="type"),
]

MessageContent = str | list[ContentBlock]


# ---------------------------------------------------------------------------
# Message types
# ---------------------------------------------------------------------------


class UserMessage(BaseModel):
    """A message from the user."""

    model_config = {"frozen": True}

    role: Literal["user"] = "user"
    content: MessageContent


class SystemMessage(BaseModel):
    """A system instruction message."""

    model_config = {"frozen": True}

    role: Literal["system"] = "system"
    content: str


class ToolCall(BaseModel):
    """A request from the LLM to invoke a tool.

    Args:
        id: Unique identifier for this tool call.
        name: Name of the tool to invoke.
        arguments: JSON-encoded string of the tool arguments.
        thought_signature: Opaque signature from thinking models (e.g. Gemini)
            that must be echoed back in conversation history.
    """

    model_config = {"frozen": True}

    id: str
    name: str
    arguments: str = ""
    thought_signature: bytes | None = None


class AssistantMessage(BaseModel):
    """A response from the LLM assistant.

    May contain text content, tool calls, or both.

    Args:
        content: Text content of the response.
        tool_calls: Tool invocations requested by the assistant.
    """

    model_config = {"frozen": True}

    role: Literal["assistant"] = "assistant"
    content: MessageContent = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)


class ToolResult(BaseModel):
    """The result of executing a tool call.

    Args:
        tool_call_id: The id of the ToolCall this responds to.
        tool_name: Name of the tool that was executed.
        content: The result from the tool (string or list of content blocks).
        error: Error message if the tool failed.
    """

    model_config = {"frozen": True}

    role: Literal["tool"] = "tool"
    tool_call_id: str
    tool_name: str
    content: MessageContent = ""
    error: str | None = None


Message = UserMessage | AssistantMessage | SystemMessage | ToolResult


class Usage(BaseModel):
    """Token usage statistics from an LLM call.

    Args:
        input_tokens: Number of tokens in the prompt.
        output_tokens: Number of tokens in the completion.
        total_tokens: Total tokens consumed.
    """

    model_config = {"frozen": True}

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


class AgentInput(BaseModel):
    """Normalized input for an agent run.

    Args:
        query: The user's query (string or list of content blocks).
        messages: Prior conversation messages for context.
    """

    model_config = {"frozen": True}

    query: MessageContent
    messages: list[Message] = Field(default_factory=list)


class AgentOutput(BaseModel):
    """Output from a single LLM call within a run.

    Args:
        text: Text content of the LLM response.
        tool_calls: Tool invocations requested by the LLM.
        usage: Token usage for this call.
    """

    model_config = {"frozen": True}

    text: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)

    @property
    def output(self) -> str:
        """Alias for :attr:`text` — mirrors :attr:`RunResult.output`.

        Lets the same attribute name work whether you hold an
        ``AgentOutput`` or a ``RunResult``.
        """
        return self.text


class ActionModel(BaseModel):
    """A parsed tool action ready for execution.

    Unlike ``ToolCall`` where arguments is a JSON string, here arguments
    is already parsed into a dict.

    Args:
        tool_call_id: Identifier linking back to the originating ToolCall.
        tool_name: Name of the tool to execute.
        arguments: Parsed keyword arguments for the tool.
    """

    model_config = {"frozen": True}

    tool_call_id: str
    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class RunResult(BaseModel):
    """Return type of ``run()`` — the final result of an agent execution.

    Args:
        output: Final text output from the agent.
        messages: Full message history of the run.
        usage: Aggregated token usage across all steps.
        steps: Number of LLM call steps taken.
    """

    model_config = {"frozen": True}

    output: str = ""
    messages: list[Message] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)
    steps: int = Field(default=0, ge=0)

    @property
    def text(self) -> str:
        """Alias for :attr:`output` — mirrors :attr:`AgentOutput.text`.

        Lets the same attribute name work whether you hold a
        ``RunResult`` or an ``AgentOutput``.
        """
        return self.output


class TextEvent(BaseModel):
    """Streaming event for a text delta.

    Args:
        type: Discriminator literal, always ``"text"``.
        text: The text chunk.
        agent_name: Name of the agent producing this event.
    """

    model_config = {"frozen": True}

    type: Literal["text"] = "text"
    text: str
    agent_name: str = ""


class ToolCallEvent(BaseModel):
    """Streaming event for a tool call notification.

    Emitted after all argument deltas for a tool call have been
    accumulated.  Acts as the "complete" signal.

    Args:
        type: Discriminator literal, always ``"tool_call"``.
        tool_name: Name of the tool being called.
        tool_call_id: Identifier for this tool call.
        arguments: The fully assembled JSON arguments string.
        agent_name: Name of the agent producing this event.
    """

    model_config = {"frozen": True}

    type: Literal["tool_call"] = "tool_call"
    tool_name: str
    tool_call_id: str
    arguments: str = ""
    agent_name: str = ""


class ToolCallDeltaEvent(BaseModel):
    """Streaming event for incremental tool call argument data.

    Emitted during the LLM stream as tool call arguments arrive
    token-by-token.  Consumers can use ``index`` to demux parallel
    tool calls.  The ``tool_call_id`` and ``tool_name`` fields are
    non-empty only on the first delta for a given index.

    Only emitted when ``detailed=True``.

    Args:
        type: Discriminator literal, always ``"tool_call_delta"``.
        index: Position in a multi-tool-call response (0-based).
        tool_call_id: Tool call ID, non-empty on the first delta only.
        tool_name: Tool name, non-empty on the first delta only.
        arguments_delta: Incremental JSON fragment of arguments.
        agent_name: Name of the agent producing this event.
    """

    model_config = {"frozen": True}

    type: Literal["tool_call_delta"] = "tool_call_delta"
    index: int = 0
    tool_call_id: str = ""
    tool_name: str = ""
    arguments_delta: str = ""
    agent_name: str = ""


class StepEvent(BaseModel):
    """Streaming event for step start/completion.

    Emitted at the start and end of each agent step so consumers
    can show step-by-step progress.

    Args:
        type: Discriminator literal, always ``"step"``.
        step_number: The step index (1-based).
        agent_name: Name of the agent executing this step.
        status: Whether the step is starting or has completed.
        started_at: Timestamp when the step started.
        completed_at: Timestamp when the step completed (None if still running).
        usage: Token usage for this step (None if not yet available).
    """

    model_config = {"frozen": True}

    type: Literal["step"] = "step"
    step_number: int
    agent_name: str
    status: Literal["started", "completed"]
    started_at: float
    completed_at: float | None = None
    usage: Usage | None = None


class ToolResultEvent(BaseModel):
    """Streaming event emitted after each tool execution.

    Args:
        type: Discriminator literal, always ``"tool_result"``.
        tool_name: Name of the tool that was executed.
        tool_call_id: Identifier linking back to the originating tool call.
        arguments: The arguments passed to the tool.
        result: The string result from the tool.
        error: Error message if the tool failed (None on success).
        success: Whether the tool execution succeeded.
        duration_ms: How long the tool execution took in milliseconds.
        agent_name: Name of the agent that invoked this tool.
    """

    model_config = {"frozen": True}

    type: Literal["tool_result"] = "tool_result"
    tool_name: str
    tool_call_id: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    result: MessageContent = ""
    error: str | None = None
    success: bool = True
    duration_ms: float = 0.0
    agent_name: str = ""


class ReasoningEvent(BaseModel):
    """Streaming event for reasoning content from the model.

    Args:
        type: Discriminator literal, always ``"reasoning"``.
        text: The reasoning text content.
        agent_name: Name of the agent producing this event.
    """

    model_config = {"frozen": True}

    type: Literal["reasoning"] = "reasoning"
    text: str
    agent_name: str = ""


class ErrorEvent(BaseModel):
    """Streaming event for errors during agent execution.

    Args:
        type: Discriminator literal, always ``"error"``.
        error: The error message.
        error_type: The type/class of the error.
        agent_name: Name of the agent that encountered the error.
        step_number: The step during which the error occurred (None if not step-specific).
        recoverable: Whether the error is recoverable.
    """

    model_config = {"frozen": True}

    type: Literal["error"] = "error"
    error: str
    error_type: str
    agent_name: str = ""
    step_number: int | None = None
    recoverable: bool = False


class StatusEvent(BaseModel):
    """Streaming event for agent status changes.

    Args:
        type: Discriminator literal, always ``"status"``.
        status: The current status of the agent.
        agent_name: Name of the agent whose status changed.
        message: Human-readable description of the status change.
    """

    model_config = {"frozen": True}

    type: Literal["status"] = "status"
    status: Literal["starting", "running", "waiting_for_tool", "completed", "cancelled", "error"]
    agent_name: str = ""
    message: str = ""


class UsageEvent(BaseModel):
    """Streaming event for per-step token usage.

    Args:
        type: Discriminator literal, always ``"usage"``.
        usage: Token usage statistics for this step.
        agent_name: Name of the agent that consumed the tokens.
        step_number: The step this usage is associated with.
        model: The model identifier used for this step.
    """

    model_config = {"frozen": True}

    type: Literal["usage"] = "usage"
    usage: Usage
    agent_name: str = ""
    step_number: int = 0
    model: str = ""


class MCPProgressEvent(BaseModel):
    """Streaming event for MCP tool progress notifications.

    Emitted when an MCP tool sends a progress notification during execution.
    These are never returned as part of the tool result — they are yielded
    only during ``agent.stream()`` as they arrive.

    Args:
        type: Discriminator literal, always ``"mcp_progress"``.
        tool_name: Name of the MCP tool sending progress.
        progress: Current progress value (converted to int from MCP float).
        total: Total progress value if known, else None.
        message: Human-readable progress message.
        agent_name: Name of the agent executing this tool.
    """

    model_config = {"frozen": True}

    type: Literal["mcp_progress"] = "mcp_progress"
    tool_name: str
    progress: int
    total: int | None = None
    message: str = ""
    agent_name: str = ""


class ContextEvent(BaseModel):
    """Streaming event emitted when the context engine takes action.

    Fired when the context engine performs summarization, offloading (aggressive
    trimming), history windowing, or token-budget-triggered compression so that
    consumers can observe context management in real time.

    Args:
        type: Discriminator literal, always ``"context"``.
        action: The context operation that was performed.  One of
            ``"offload"`` (aggressive trim past offload threshold),
            ``"summarize"`` (LLM-based conversation summarization),
            ``"window"`` (history rounds trimming), or
            ``"token_budget"`` (token fill ratio exceeded trigger).
        agent_name: Name of the agent whose context was modified.
        before_count: Number of non-system messages before the action.
        after_count: Number of non-system messages after the action.
        details: Action-specific metadata (thresholds, fill ratio, etc.).
    """

    model_config = {"frozen": True}

    type: Literal["context"] = "context"
    action: Literal["offload", "summarize", "window", "token_budget"]
    agent_name: str = ""
    before_count: int = 0
    after_count: int = 0
    details: dict[str, Any] = Field(default_factory=dict)


class MessageInjectedEvent(BaseModel):
    """Streaming event emitted when a message is injected into a running agent."""

    model_config = {"frozen": True}

    type: Literal["message_injected"] = "message_injected"
    content: str
    agent_name: str = ""


class RalphIterationEvent(BaseModel):
    """Emitted at the start/end of each Ralph loop iteration."""

    model_config = {"frozen": True}

    type: Literal["ralph_iteration"] = "ralph_iteration"
    iteration: int
    status: Literal["started", "completed", "failed"]
    scores: dict[str, float] = Field(default_factory=dict)
    agent_name: str = ""


class RalphStopEvent(BaseModel):
    """Emitted when the Ralph loop terminates."""

    model_config = {"frozen": True}

    type: Literal["ralph_stop"] = "ralph_stop"
    stop_type: str
    reason: str
    iterations: int
    final_scores: dict[str, float] = Field(default_factory=dict)
    agent_name: str = ""


class HITLApprovalEvent(BaseModel):
    """Emitted when a HITL tool's approval prompt is resolved.

    Records whether the human approved or denied execution of a tool
    that was listed in the agent's ``hitl_tools``.

    Args:
        type: Discriminator literal, always ``"hitl_approval"``.
        tool_name: Name of the tool that required approval.
        tool_call_id: Identifier linking back to the originating tool call.
        arguments: JSON-encoded string of the tool arguments.
        approved: Whether the human approved execution.
        agent_name: Name of the agent that requested approval.
    """

    model_config = {"frozen": True}

    type: Literal["hitl_approval"] = "hitl_approval"
    tool_name: str
    tool_call_id: str
    arguments: str = ""
    approved: bool
    agent_name: str = ""


StreamEvent = (
    TextEvent
    | ToolCallEvent
    | ToolCallDeltaEvent
    | StepEvent
    | ToolResultEvent
    | ReasoningEvent
    | ErrorEvent
    | StatusEvent
    | UsageEvent
    | MCPProgressEvent
    | ContextEvent
    | MessageInjectedEvent
    | RalphIterationEvent
    | RalphStopEvent
    | HITLApprovalEvent
)


class EventType(StrEnum):
    """Discriminator values for every :data:`StreamEvent` type.

    Use these for the ``event_types`` filter of ``run.stream`` so a typo
    fails loudly instead of silently yielding nothing::

        async for ev in run.stream(agent, "hi", event_types={EventType.TEXT}):
            ...

    Plain strings still work; this enum just makes the valid set
    discoverable via autocomplete.
    """

    TEXT = "text"
    TOOL_CALL = "tool_call"
    TOOL_CALL_DELTA = "tool_call_delta"
    STEP = "step"
    TOOL_RESULT = "tool_result"
    REASONING = "reasoning"
    ERROR = "error"
    STATUS = "status"
    USAGE = "usage"
    MCP_PROGRESS = "mcp_progress"
    CONTEXT = "context"
    MESSAGE_INJECTED = "message_injected"
    RALPH_ITERATION = "ralph_iteration"
    RALPH_STOP = "ralph_stop"
    HITL_APPROVAL = "hitl_approval"


#: The set of all valid ``event_types`` filter values (string form).
EVENT_TYPES: frozenset[str] = frozenset(e.value for e in EventType)
