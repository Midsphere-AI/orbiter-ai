"""Configuration types for the Exo framework."""

from __future__ import annotations

from collections.abc import Mapping

from pydantic import BaseModel, Field

from exo.types import ExoError


class ConfigError(ExoError, ValueError):
    """Raised for invalid Agent/config values.

    Inherits ``ValueError`` as well as ``ExoError`` so existing
    ``except ValueError`` call sites keep working while the message gains an
    actionable ``hint=`` and structured ``context=``.
    """


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
        raise ConfigError(
            f"model must be a string, got {type(model).__name__!r}.",
            context={"model": model},
            hint="Pass a 'provider:model' string, e.g. model='openai:gpt-4o-mini'.",
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
        raise ConfigError(
            "planning_model must be a non-empty model string.",
            hint="Pass a 'provider:model' string, e.g. planning_model='openai:gpt-4o', "
            "or omit it to plan with the executor model.",
        )

    _, model_name = parse_model_string(normalized)
    if not model_name.strip():
        raise ConfigError(
            "planning_model must include a model name.",
            context={"planning_model": model},
            hint="Use the 'provider:model' format, e.g. planning_model='openai:gpt-4o'.",
        )

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

    raise ConfigError(
        "budget_awareness must be 'per-message' or 'limit:<0-100>'.",
        context={"budget_awareness": value},
        hint="Use budget_awareness='per-message' for every-turn awareness, or "
        "'limit:80' to start warning at 80% of the token budget.",
    )


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
            raise ConfigError(
                "injected_tool_args keys must be non-empty strings.",
                context={"key": key},
                hint="Map each injected arg name to a description string, e.g. "
                "injected_tool_args={'user_id': 'the caller's id'}.",
            )
        if not isinstance(description, str):
            raise ConfigError(
                f"injected_tool_args['{key}'] description must be a string, "
                f"got {type(description).__name__!r}.",
                context={"key": key},
                hint="Each value is the schema description shown to the LLM — pass a string.",
            )
        normalized[key] = description

    return normalized


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
    raise ConfigError(
        f"max_spawn_children must be between 1 and 8, got {value}.",
        context={"max_spawn_children": value},
        hint="Choose a value in 1..8; the cap bounds fan-out per spawn_self call.",
    )


class ModelConfig(BaseModel):
    """Configuration for an LLM provider connection.

    The core fields cover the common case. Provider-specific options
    (e.g. ``google_project``, ``google_service_account_base64``) can be
    passed as extra keyword arguments and will be stored on the instance.

    Args:
        provider: Provider name, e.g. ``"openai"`` or ``"anthropic"``.
        model_name: Model identifier within the provider.
        api_key: API key for authentication.  The raw string value is
            accessible as ``config.api_key`` for provider SDKs; the value
            is redacted in ``repr()`` / ``str()`` to prevent leaking into
            logs and tracebacks.
        base_url: Custom API base URL.
        max_retries: Maximum number of retries on transient failures.
        timeout: Request timeout in seconds.
    """

    model_config = {"frozen": True, "extra": "allow"}

    provider: str = "openai"
    model_name: str = "gpt-4o"
    api_key: str | None = Field(default=None, repr=False)
    base_url: str | None = None
    max_retries: int = Field(default=3, ge=0)
    timeout: float = Field(default=30.0, gt=0)
    context_window_tokens: int | None = None

    def __repr__(self) -> str:
        # Manually build repr so api_key is always redacted regardless of
        # how Pydantic's repr is configured (frozen + extra="allow" complicates it).
        masked_key = "**redacted**" if self.api_key is not None else None
        extra_fields = self.__pydantic_extra__ or {}
        parts = [
            f"provider={self.provider!r}",
            f"model_name={self.model_name!r}",
            f"api_key={masked_key!r}",
        ]
        if self.base_url is not None:
            parts.append(f"base_url={self.base_url!r}")
        parts += [
            f"max_retries={self.max_retries!r}",
            f"timeout={self.timeout!r}",
        ]
        if self.context_window_tokens is not None:
            parts.append(f"context_window_tokens={self.context_window_tokens!r}")
        for k, v in extra_fields.items():
            parts.append(f"{k}={v!r}")
        return f"ModelConfig({', '.join(parts)})"


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
