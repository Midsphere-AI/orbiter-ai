"""Planner pre-pass helpers for agent execution."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any

from exo.config import parse_model_string
from exo.observability.logging import get_logger  # pyright: ignore[reportMissingImports]
from exo.types import Message, MessageContent, SystemMessage

_log = get_logger(__name__)

_PLANNER_INPUT_TEMPLATE = (
    "Original task:\n{task}\n\n"
    "Planner output:\n{plan}\n\n"
    "Use the planner output while completing the task."
)
_PLANNER_CONTEXT_TEMPLATE = (
    "Planner output:\n{plan}\n\nUse the planner output while responding to the next user task."
)


async def prepare_planned_execution(
    agent: Any,
    input: MessageContent,
    messages: Sequence[Message] | None,
    provider: Any,
    *,
    max_retries: int,
) -> tuple[MessageContent, list[Message] | None]:
    """Run an ephemeral planner phase and return executor context.

    The planner phase is enabled only when ``agent.planning_enabled`` is true.
    Its transcript remains isolated from the executor run; only the final plan
    text is injected into the executor context.
    """
    if not getattr(agent, "planning_enabled", False) or provider is None:
        return input, list(messages) if messages else None

    planner_model = getattr(agent, "planning_model", None) or getattr(agent, "model", "")
    executor_instructions = await _resolve_instructions(
        getattr(agent, "instructions", ""), getattr(agent, "name", "")
    )
    planner_instructions = getattr(agent, "planning_instructions", "").strip()
    if not planner_instructions:
        planner_instructions = executor_instructions

    planner_agent = _build_planner_agent(agent, planner_model, planner_instructions)
    planner_provider = _resolve_planner_provider(
        planner_model=planner_model,
        executor_model=getattr(agent, "model", ""),
        executor_provider=provider,
    )
    if planner_provider is None:
        from exo.agent import AgentError

        raise AgentError(
            f"Planner-enabled agent '{agent.name}' could not resolve provider "
            f"for planning model '{planner_model}'"
        )

    _log.debug(
        "planner pre-pass starting: agent=%s planner_model=%s",
        getattr(agent, "name", ""),
        planner_model,
    )
    planner_output = await planner_agent._run(
        input,
        messages=messages,
        provider=planner_provider,
        max_retries=max_retries,
    )
    plan_text = planner_output.text.strip()
    if not plan_text:
        return input, list(messages) if messages else None

    _log.debug(
        "planner pre-pass completed: agent=%s plan_chars=%d",
        getattr(agent, "name", ""),
        len(plan_text),
    )
    return _inject_plan_into_context(input, messages, plan_text)


async def _resolve_instructions(raw_instructions: Any, agent_name: str) -> str:
    """Resolve string or callable instructions to plain text."""
    if callable(raw_instructions):
        if asyncio.iscoroutinefunction(raw_instructions):
            return str(await raw_instructions(agent_name))
        return str(raw_instructions(agent_name))
    if raw_instructions:
        return str(raw_instructions)
    return ""


def _build_planner_agent(agent: Any, planner_model: str, planner_instructions: str) -> Any:
    """Create an ephemeral planner agent with the executor's tool set.

    Excludes the parent's ``__exo_ptc__`` tool (bound to the parent
    instance) and propagates the executor's PTC settings so the planner
    re-registers its own ``PTCTool`` — otherwise PTC-eligible tools
    would leak as direct schemas on the planner and the planner could
    call tools that should only be reachable through PTC.
    """
    from exo.agent import Agent

    planner_tools = []
    allow_self_spawn = bool(getattr(agent, "allow_self_spawn", False))
    for tool_name, tool in getattr(agent, "tools", {}).items():
        if tool_name == "retrieve_artifact":
            continue
        if getattr(tool, "_is_context_tool", False):
            continue
        if getattr(tool, "_is_ptc_tool", False):
            # Parent's PTCTool is bound to the parent agent; the planner
            # constructor below will register a fresh PTCTool on the
            # planner when ``ptc`` is propagated.
            continue
        if allow_self_spawn and tool_name == "spawn_self":
            continue
        planner_tools.append(tool)

    return Agent(
        name=f"{getattr(agent, 'name', 'agent')}_planner",
        model=planner_model,
        instructions=planner_instructions,
        tools=planner_tools,
        max_steps=getattr(agent, "max_steps", 10),
        temperature=getattr(agent, "temperature", 1.0),
        max_tokens=getattr(agent, "max_tokens", None),
        budget_awareness=getattr(agent, "budget_awareness", None),
        hitl_tools=list(getattr(agent, "hitl_tools", [])),
        human_input_handler=getattr(agent, "_human_input_handler", None),
        emit_mcp_progress=getattr(agent, "emit_mcp_progress", True),
        injected_tool_args=dict(getattr(agent, "injected_tool_args", {})),
        store=False,
        context=getattr(agent, "context", None),
        subagents=allow_self_spawn,
        max_spawn_depth=getattr(agent, "max_spawn_depth", 3),
        max_spawn_children=getattr(agent, "max_spawn_children", 4),
        # Propagate PTC settings so the planner sees the same filtered
        # schema view as the executor (PTC-eligible tools absorbed into
        # ``__exo_ptc__``, not leaked as direct schemas).
        ptc=bool(getattr(agent, "ptc", False)),
        ptc_timeout=int(getattr(agent, "ptc_timeout", 60)),
        ptc_max_output_bytes=int(getattr(agent, "ptc_max_output_bytes", 200_000)),
        ptc_max_tool_calls=int(getattr(agent, "ptc_max_tool_calls", 200)),
        ptc_extra_args=dict(getattr(agent, "ptc_extra_args", {}) or {}) or None,
    )


def _resolve_planner_provider(
    *,
    planner_model: str,
    executor_model: str,
    executor_provider: Any,
) -> Any | None:
    """Resolve the provider instance used for the planner phase."""
    if planner_model == executor_model:
        return executor_provider

    cloned_provider = _clone_provider_for_model(executor_provider, planner_model)
    if cloned_provider is not None:
        return cloned_provider

    return _resolve_provider_for_model(planner_model)


def _clone_provider_for_model(provider: Any, model: str) -> Any | None:
    """Clone a provider for another model when it shares the same backend."""
    config = getattr(provider, "config", None)
    if config is None or not hasattr(config, "model_copy"):
        return None

    provider_name, model_name = parse_model_string(model)
    if getattr(config, "provider", None) != provider_name:
        return None

    update_fields: dict[str, Any] = {"provider": provider_name, "model_name": model_name}
    try:
        from exo.models.context_windows import (  # pyright: ignore[reportMissingImports]
            MODEL_CONTEXT_WINDOWS,
        )

        update_fields["context_window_tokens"] = MODEL_CONTEXT_WINDOWS.get(model_name)
    except Exception:
        pass

    try:
        return type(provider)(config.model_copy(update=update_fields))
    except Exception as exc:
        _log.warning(
            "Failed to clone planner provider for model '%s': %s",
            model,
            exc,
        )
        return None


def _resolve_provider_for_model(model: str) -> Any | None:
    """Resolve a fresh provider from a model string."""
    try:
        from exo.models.provider import get_provider  # pyright: ignore[reportMissingImports]

        return get_provider(model)
    except Exception as exc:
        _log.warning("Failed to resolve planner provider for model '%s': %s", model, exc)
        return None


def _inject_plan_into_context(
    input: MessageContent,
    messages: Sequence[Message] | None,
    plan_text: str,
) -> tuple[MessageContent, list[Message] | None]:
    """Inject the planner text into the executor context."""
    if isinstance(input, str):
        return _PLANNER_INPUT_TEMPLATE.format(task=input, plan=plan_text), list(
            messages
        ) if messages else None

    prepared_messages = list(messages) if messages else []
    prepared_messages.append(
        SystemMessage(content=_PLANNER_CONTEXT_TEMPLATE.format(plan=plan_text))
    )
    return input, prepared_messages
