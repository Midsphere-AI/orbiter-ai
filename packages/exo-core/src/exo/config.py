"""Configuration types for the Exo framework."""

from __future__ import annotations

from collections.abc import Mapping

from pydantic import BaseModel, Field, field_validator


def parse_model_string(model: str) -> tuple[str, str]:
    """Split a model string into provider and model name.

    Parses the ``"provider:model_name"`` format. If no colon is present,
    defaults the provider to ``"openai"``.

    Args:
        model: Model string, e.g. ``"openai:gpt-4o-mini"`` or ``"gpt-4o-mini"``.

    Returns:
        A ``(provider, model_name)`` tuple.

    Raises:
        ValueError: If *model* is not a string.
    """
    if not isinstance(model, str):
        raise ValueError(
            f"model must be a string like 'openai:gpt-4o-mini', got {type(model).__name__!r}"
        )
    if ":" in model:
        provider, _, model_name = model.partition(":")
        return provider, model_name
    return "openai", model


def validate_planning_model(model: str | None) -> str | None:
    """Validate a planner model override.

    Args:
        model: Planner model override in the normal Exo model format.

    Returns:
        The normalized model string, or ``None`` when planning uses the
        executor model.

    Raises:
        ValueError: If the model string is empty or omits a model name.
    """
    if model is None:
        return None

    normalized = model.strip()
    if not normalized:
        raise ValueError("planning_model must be a non-empty model string")

    _, model_name = parse_model_string(normalized)
    if not model_name.strip():
        raise ValueError("planning_model must include a model name")

    return normalized


def validate_budget_awareness(value: str | None) -> str | None:
    """Validate the configured budget-awareness mode.

    Args:
        value: Budget-awareness mode string or ``None`` to disable it.

    Returns:
        The normalized budget-awareness mode, or ``None`` when disabled.

    Raises:
        ValueError: If the value is not ``per-message`` or ``limit:<0-100>``.
    """
    if value is None:
        return None

    normalized = value.strip()
    if normalized == "per-message":
        return normalized

    if normalized.startswith("limit:"):
        limit_text = normalized.split(":", 1)[1]
        if limit_text.isdigit():
            limit = int(limit_text)
            if 0 <= limit <= 100:
                return normalized

    raise ValueError("budget_awareness must be 'per-message' or 'limit:<0-100>'")


def validate_injected_tool_args(value: Mapping[str, str] | None) -> dict[str, str]:
    """Validate schema-only injected tool arguments.

    Args:
        value: Mapping of injected argument name to description.

    Returns:
        A shallow copy of the validated mapping.

    Raises:
        ValueError: If a key is empty or a description is not a string.
    """
    if value is None:
        return {}

    normalized: dict[str, str] = {}
    for key, description in value.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError("injected_tool_args keys must be non-empty strings")
        if not isinstance(description, str):
            raise ValueError("injected_tool_args values must be strings")
        normalized[key] = description

    return normalized


def validate_max_parallel_subagents(value: int) -> int:
    """Validate the per-call parallel sub-agent cap.

    Args:
        value: Maximum number of child jobs allowed in one parallel call.

    Returns:
        The validated limit.

    Raises:
        ValueError: If the limit falls outside ``1..7``.
    """
    if 1 <= value <= 7:
        return value
    raise ValueError("max_parallel_subagents must be between 1 and 7")


def validate_max_spawn_children(value: int) -> int:
    """Validate the per-call spawn children cap.

    Args:
        value: Maximum number of child agents spawned in one spawn_self call.

    Returns:
        The validated limit.

    Raises:
        ValueError: If the limit falls outside ``1..8``.
    """
    if 1 <= value <= 8:
        return value
    raise ValueError("max_spawn_children must be between 1 and 8")


class ModelConfig(BaseModel):
    """Configuration for an LLM provider connection.

    The core fields cover the common case. Provider-specific options
    (e.g. ``google_project``, ``google_service_account_base64``) can be
    passed as extra keyword arguments and will be stored on the instance.

    Args:
        provider: Provider name, e.g. ``"openai"`` or ``"anthropic"``.
        model_name: Model identifier within the provider.
        api_key: API key for authentication.
        base_url: Custom API base URL.
        max_retries: Maximum number of retries on transient failures.
        timeout: Request timeout in seconds.
    """

    model_config = {"frozen": True, "extra": "allow"}

    provider: str = "openai"
    model_name: str = "gpt-4o"
    api_key: str | None = None
    base_url: str | None = None
    max_retries: int = Field(default=3, ge=0)
    timeout: float = Field(default=30.0, gt=0)
    context_window_tokens: int | None = None


class AgentConfig(BaseModel):
    """Configuration for an Agent.

    Args:
        name: Unique identifier for the agent.
        model: Model string in ``"provider:model_name"`` format.
        instructions: System prompt for the agent.
        temperature: LLM sampling temperature.
        max_tokens: Maximum output tokens per LLM call.
        max_steps: Maximum LLM-tool round-trips.
        planning_enabled: Whether to run a planner phase before execution.
        planning_model: Optional planner model override.
        planning_instructions: Optional planner-only instructions.
        budget_awareness: Context-budget handling mode.
        hitl_tools: Tool names that require human approval before execution.
        bare_tools: Suppress auto-registered helper tools (retrieve_artifact, context tools).
        emit_mcp_progress: Whether MCP progress events are emitted.
        injected_tool_args: Schema-only tool arguments exposed to the model.
        allow_parallel_subagents: Whether the parallel-subagent tool is enabled.
        max_parallel_subagents: Maximum child jobs per parallel-subagent call.
    """

    model_config = {"frozen": True}

    name: str
    model: str = "openai:gpt-4o-mini"
    instructions: str = ""
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    max_tokens: int | None = None
    max_steps: int = Field(default=10, ge=1)
    planning_enabled: bool = False
    planning_model: str | None = None
    planning_instructions: str = ""
    budget_awareness: str | None = None
    hitl_tools: list[str] = Field(default_factory=list)
    bare_tools: bool = False
    emit_mcp_progress: bool = True
    injected_tool_args: dict[str, str] = Field(default_factory=dict)
    allow_parallel_subagents: bool = False
    max_parallel_subagents: int = 3

    @field_validator("planning_model")
    @classmethod
    def _validate_planning_model(cls, value: str | None) -> str | None:
        return validate_planning_model(value)

    @field_validator("budget_awareness")
    @classmethod
    def _validate_budget_awareness(cls, value: str | None) -> str | None:
        return validate_budget_awareness(value)

    @field_validator("hitl_tools")
    @classmethod
    def _validate_hitl_tools(cls, value: list[str]) -> list[str]:
        for tool_name in value:
            if not isinstance(tool_name, str) or not tool_name.strip():
                raise ValueError("hitl_tools entries must be non-empty strings")
        return list(value)

    @field_validator("injected_tool_args")
    @classmethod
    def _validate_injected_tool_args(cls, value: dict[str, str]) -> dict[str, str]:
        return validate_injected_tool_args(value)

    @field_validator("max_parallel_subagents")
    @classmethod
    def _validate_max_parallel_subagents(cls, value: int) -> int:
        return validate_max_parallel_subagents(value)


class TaskConfig(BaseModel):
    """Configuration for a task.

    Args:
        name: Unique identifier for the task.
        description: Human-readable description of what the task does.
    """

    model_config = {"frozen": True}

    name: str
    description: str = ""


class RunConfig(BaseModel):
    """Configuration for a single run invocation.

    Args:
        max_steps: Maximum LLM-tool round-trips for this run.
        timeout: Overall timeout in seconds for the run.
        stream: Whether to enable streaming output.
        verbose: Whether to enable verbose logging.
    """

    model_config = {"frozen": True}

    max_steps: int = Field(default=10, ge=1)
    timeout: float | None = None
    stream: bool = False
    verbose: bool = False
