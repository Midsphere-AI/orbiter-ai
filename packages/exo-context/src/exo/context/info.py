"""Rich context snapshot for custom context windowing hooks."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ContextWindowInfo:
    """Read-only snapshot of context state at windowing time.

    Passed to ``CONTEXT_WINDOW`` hooks alongside the mutable *messages*
    list so hooks can make informed decisions about what to keep, compress,
    or discard.

    Example::

        async def my_hook(*, messages, info, **_):
            if info.fill_ratio > 0.7:
                # ... compress ...
    """

    # ── Where we are in the run ──────────────────────────────────
    step: int
    """0-based tool-loop index. ``-1`` for initial (pre-first-LLM) windowing."""

    max_steps: int
    """Agent's ``max_steps`` — the loop budget."""

    is_initial: bool
    """``True`` for the pre-first-LLM windowing pass, ``False`` mid-run."""

    # ── Message counts (at hook entry, before mutations) ─────────
    total_messages: int
    system_count: int
    user_count: int
    assistant_count: int
    tool_result_count: int

    # ── Token pressure ───────────────────────────────────────────
    context_window: int | None
    """Model's max context tokens. ``None`` if unknown."""

    input_tokens: int
    """Prompt tokens from last LLM call (0 on initial windowing)."""

    output_tokens: int
    """Output tokens from last LLM call (0 on initial windowing)."""

    fill_ratio: float
    """``input_tokens / context_window``. 0.0 when unknown or initial."""

    token_pressure_threshold: float
    """The ``token_pressure`` config value that triggers forced windowing."""

    force: bool
    """``True`` when token budget exceeded and forced windowing was triggered."""

    # ── Cumulative trajectory ────────────────────────────────────
    cumulative_input_tokens: int
    """Total prompt tokens across all steps so far."""

    cumulative_output_tokens: int
    """Total output tokens across all steps so far."""

    trajectory: tuple[Any, ...] = field(default=())
    """Per-step token history as ``TokenStep`` objects (empty if unavailable)."""

    # ── Config (for reference) ───────────────────────────────────
    overflow: str = "summarize"
    """The overflow strategy that was configured."""

    limit: int = 20
    """``config.limit`` / ``history_rounds``."""

    keep_recent: int = 5
    """``config.keep_recent``."""

    # ── Agent identity ───────────────────────────────────────────
    agent_name: str = ""
    model: str = ""


def build_context_window_info(
    msg_list: list[Any],
    config: Any,
    *,
    step: int = -1,
    max_steps: int = 0,
    agent_name: str = "",
    model: str = "",
    context_window_tokens: int | None = None,
    last_usage: Any | None = None,
    token_tracker: Any | None = None,
    force: bool = False,
) -> ContextWindowInfo:
    """Build a :class:`ContextWindowInfo` from data available at a windowing call site."""
    try:
        from exo.types import (  # pyright: ignore[reportMissingImports]
            AssistantMessage,
            SystemMessage,
            ToolResult,
            UserMessage,
        )

        _msg_types: tuple[type, ...] = (SystemMessage, UserMessage, AssistantMessage, ToolResult)
    except ImportError:
        logger.debug("exo.types unavailable in build_context_window_info; message counts will be 0")
        _msg_types = (type(None), type(None), type(None), type(None))

    # Single pass — accumulate all four counts together to avoid 4x iteration.
    system_count = user_count = assistant_count = tool_result_count = 0
    sys_t, usr_t, asst_t, tool_t = _msg_types
    for m in msg_list:
        if isinstance(m, sys_t):
            system_count += 1
        elif isinstance(m, usr_t):
            user_count += 1
        elif isinstance(m, asst_t):
            assistant_count += 1
        elif isinstance(m, tool_t):
            tool_result_count += 1

    # Token usage from last LLM call (duck-typed)
    input_tokens = getattr(last_usage, "input_tokens", 0) or 0
    output_tokens = getattr(last_usage, "output_tokens", 0) or 0

    # Fill ratio
    fill_ratio = 0.0
    if context_window_tokens and input_tokens > 0:
        fill_ratio = input_tokens / context_window_tokens

    # Trajectory from token tracker
    trajectory: tuple[Any, ...] = ()
    cumulative_input = 0
    cumulative_output = 0
    if token_tracker is not None:
        try:
            traj_list = token_tracker.get_trajectory(agent_name)
            trajectory = tuple(traj_list)
            cumulative_input = sum(getattr(s, "prompt_tokens", 0) for s in traj_list)
            cumulative_output = sum(getattr(s, "output_tokens", 0) for s in traj_list)
        except Exception as exc:
            logger.debug(
                "trajectory unavailable for %r: %s",
                agent_name,
                exc,
                exc_info=True,
            )

    # Config fields (duck-typed for both ContextConfig and bare objects)
    overflow = str(getattr(config, "overflow", "summarize"))
    limit = getattr(config, "limit", None) or getattr(config, "history_rounds", 20)
    keep_recent = getattr(config, "keep_recent", 5)
    token_pressure = getattr(config, "token_pressure", getattr(config, "token_budget_trigger", 0.8))

    return ContextWindowInfo(
        step=step,
        max_steps=max_steps,
        is_initial=(step == -1),
        total_messages=len(msg_list),
        system_count=system_count,
        user_count=user_count,
        assistant_count=assistant_count,
        tool_result_count=tool_result_count,
        context_window=context_window_tokens,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        fill_ratio=fill_ratio,
        token_pressure_threshold=token_pressure,
        force=force,
        cumulative_input_tokens=cumulative_input,
        cumulative_output_tokens=cumulative_output,
        trajectory=trajectory,
        overflow=overflow,
        limit=limit,
        keep_recent=keep_recent,
        agent_name=agent_name,
        model=model,
    )
