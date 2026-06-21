"""Shared helpers for the non-streaming and streaming agent run loops.

These functions capture logic that was previously duplicated between
``exo.agent.Agent._run_inner`` (non-streaming) and
``exo.runner._stream`` (streaming).  Both call-sites import from here.

Design constraint: no imports that create circular dependencies.
``agent.py`` is imported by ``runner.py`` so helpers that need Agent
must accept it as ``Any``.
"""

from __future__ import annotations

import asyncio
from typing import Any

from exo.observability.logging import get_logger  # pyright: ignore[reportMissingImports]
from exo.types import Message, UserMessage  # pyright: ignore[reportMissingImports]

_log = get_logger(__name__)


# ---------------------------------------------------------------------------
# 1. Instructions resolution
# ---------------------------------------------------------------------------


async def resolve_instructions(agent: Any) -> str:
    """Resolve an agent's instructions, awaiting async callables as needed.

    Handles three cases:
    - ``instructions`` is a coroutine function → ``await`` it with ``agent.name``
    - ``instructions`` is a regular callable → call it with ``agent.name``
    - ``instructions`` is a string (or ``None``) → convert to str

    Args:
        agent: An ``Agent`` instance with an ``instructions`` attribute.

    Returns:
        The resolved instruction string (may be empty).
    """
    raw = agent.instructions
    if callable(raw):
        if asyncio.iscoroutinefunction(raw):
            return str(await raw(agent.name))
        return str(raw(agent.name))
    return str(raw) if raw else ""


# ---------------------------------------------------------------------------
# 2. Ephemeral message drain
# ---------------------------------------------------------------------------


def drain_ephemeral_messages(agent: Any) -> list[Message]:
    """Drain all queued ephemeral messages from *agent._ephemeral_messages*.

    Ephemeral messages are visible for ONE LLM call only.  They are
    collected here into a batch so the caller can concatenate them with
    the persistent message list for that single call without mutating
    ``msg_list`` (which preserves the KV-cache prefix).

    Args:
        agent: An ``Agent`` instance with an ``_ephemeral_messages`` queue.

    Returns:
        A (possibly empty) list of :class:`Message` objects to append for
        the current LLM call only.
    """
    batch: list[Message] = []
    while not agent._ephemeral_messages.empty():
        try:
            batch.append(agent._ephemeral_messages.get_nowait())
        except asyncio.QueueEmpty:
            break
    return batch


# ---------------------------------------------------------------------------
# 3. Injected-message drain (non-streaming — no event emission)
# ---------------------------------------------------------------------------


def drain_injected_messages(agent: Any, msg_list: list[Message]) -> int:
    """Drain all queued injected messages into *msg_list*.

    Injected messages (pushed via ``agent.inject_message()``) are
    appended to the persistent message history so they persist across
    steps.

    The streaming path emits a ``MessageInjectedEvent`` per message and
    therefore handles injection inline; this helper is only for the
    non-streaming path.

    Args:
        agent: An ``Agent`` instance with an ``_injected_messages`` queue.
        msg_list: The message list to append injected messages to (mutated
            in place).

    Returns:
        Number of messages injected.
    """
    count = 0
    while not agent._injected_messages.empty():
        try:
            content = agent._injected_messages.get_nowait()
            msg_list.append(UserMessage(content=content))
            count += 1
        except asyncio.QueueEmpty:
            break
    return count


# ---------------------------------------------------------------------------
# 4. Token-budget pressure check
# ---------------------------------------------------------------------------


def check_token_budget_pressure(
    input_tokens: int,
    context_window: int | None,
    context: Any,
) -> tuple[bool, float, float]:
    """Check whether the token fill ratio has crossed the pressure threshold.

    Used by both the non-streaming and streaming run loops to decide
    whether to trigger aggressive context summarization.

    The canonical config field is ``token_pressure`` (on
    ``ContextConfig``); ``_token_budget_trigger`` is an alias property
    that resolves to the same value.  We try both to stay compatible with
    any plain-dict or duck-typed context configs in tests.

    Args:
        input_tokens: ``input_tokens`` from the last LLM call's usage.
        context_window: Context window capacity in tokens, or ``None`` if
            unknown (in which case pressure is never triggered).
        context: The agent's ``context`` object (a ``Context`` or
            ``ContextConfig``).

    Returns:
        A ``(force_summarize, fill_ratio, trigger)`` tuple where:
        - ``force_summarize`` is ``True`` when fill_ratio > trigger.
        - ``fill_ratio`` is ``input_tokens / context_window`` (or 0.0).
        - ``trigger`` is the configured threshold (default 0.8).
    """
    if not context_window or input_tokens <= 0:
        return False, 0.0, 0.8

    fill_ratio = input_tokens / context_window
    _cfg = getattr(context, "config", context)
    # Try canonical name first, then alias, then fallback default.
    trigger: float = getattr(
        _cfg, "token_pressure", getattr(_cfg, "_token_budget_trigger", getattr(_cfg, "token_budget_trigger", 0.8))
    )

    return fill_ratio > trigger, fill_ratio, trigger
