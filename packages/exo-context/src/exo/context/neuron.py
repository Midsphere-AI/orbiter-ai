"""Neuron — modular prompt composition components.

A Neuron is a composable unit that produces a prompt fragment from context.
Neurons are prioritised: lower priority numbers execute first and appear
earlier in the assembled prompt.

Core neurons:
- SystemNeuron    (priority 100) — date, time, platform info
- TaskNeuron      (priority 1)   — task ID, input, plan
- HistoryNeuron   (priority 10)  — conversation history with windowing
"""

from __future__ import annotations

import logging
import platform
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from typing import Any

logger = logging.getLogger(__name__)

from exo.context.context import Context  # pyright: ignore[reportMissingImports]
from exo.registry import Registry  # pyright: ignore[reportMissingImports]

# ── Registry ─────────────────────────────────────────────────────────

neuron_registry: Registry[Neuron] = Registry("neuron_registry")

# ── ABC ──────────────────────────────────────────────────────────────


class Neuron(ABC):
    """Abstract base for prompt neurons.

    Subclasses implement :meth:`format` to produce a prompt fragment
    from the given context.  The :attr:`priority` controls ordering
    when multiple neurons are composed: lower values appear first.

    Parameters
    ----------
    name:
        Human-readable name for registry and debugging.
    priority:
        Ordering priority (lower = earlier in prompt). Default 50.
    """

    __slots__ = ("_name", "_priority")

    def __init__(self, name: str, *, priority: int = 50) -> None:
        self._name = name
        self._priority = priority

    @property
    def name(self) -> str:
        return self._name

    @property
    def priority(self) -> int:
        return self._priority

    @abstractmethod
    async def format(self, ctx: Context, **kwargs: Any) -> str:
        """Produce a prompt fragment from *ctx*.

        Returns an empty string to signal "nothing to contribute".
        """

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self._name!r}, priority={self._priority})"


# ── Built-in neurons ────────────────────────────────────────────────


class SystemNeuron(Neuron):
    """Provides dynamic system variables: date, time, platform.

    Priority 100 (low) — appended near the end of system prompts.
    """

    def __init__(self, name: str = "system", *, priority: int = 100) -> None:
        super().__init__(name, priority=priority)

    async def format(self, ctx: Context, **kwargs: Any) -> str:
        now = datetime.now(tz=UTC)
        lines = [
            "<system_info>",
            f"Current date: {now.strftime('%Y-%m-%d')}",
            f"Current time: {now.strftime('%H:%M:%S UTC')}",
            f"Platform: {platform.system()} {platform.release()}",
            "</system_info>",
        ]
        return "\n".join(lines)


class TaskNeuron(Neuron):
    """Provides task context: task ID, input, output, subtask plan.

    Reads from ``ctx.state``:
    - ``task_input``  — the current task input text
    - ``task_output`` — any partial output so far
    - ``subtasks``    — list of subtask descriptions (plan)

    Priority 1 (high) — appears first in the prompt.
    """

    def __init__(self, name: str = "task", *, priority: int = 1) -> None:
        super().__init__(name, priority=priority)

    async def format(self, ctx: Context, **kwargs: Any) -> str:
        parts: list[str] = ["<task_info>", f"Task ID: {ctx.task_id}"]

        task_input = ctx.state.get("task_input")
        if task_input:
            parts.append(f"Input: {task_input}")

        task_output = ctx.state.get("task_output")
        if task_output:
            parts.append(f"Output: {task_output}")

        subtasks: list[str] | None = ctx.state.get("subtasks")
        if subtasks:
            parts.append("Plan:")
            for i, step in enumerate(subtasks, 1):
                parts.append(f"  <step{i}>{step}</step{i}>")

        parts.append("</task_info>")
        return "\n".join(parts)


class HistoryNeuron(Neuron):
    """Provides windowed conversation history.

    Reads ``history`` from ``ctx.state`` — expected to be a list of
    message dicts (``[{"role": ..., "content": ...}, ...]``).

    Uses ``ctx.config._history_rounds`` (or ``limit``) to limit the number of rounds
    included.

    Priority 10 — appears early in the prompt, after task info.
    """

    def __init__(self, name: str = "history", *, priority: int = 10) -> None:
        super().__init__(name, priority=priority)

    async def format(self, ctx: Context, **kwargs: Any) -> str:
        history: list[dict[str, Any]] | None = ctx.state.get("history")
        if not history:
            return ""

        # Window to last N rounds (each round = user + assistant = 2 messages)
        _cfg = ctx.config
        _hr = getattr(_cfg, "_history_rounds", getattr(_cfg, "history_rounds", getattr(_cfg, "limit", 20)))
        max_messages = _hr * 2
        windowed = history[-max_messages:]

        lines = ["<conversation_history>"]
        for msg in windowed:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            lines.append(f"[{role}]: {content}")
        lines.append("</conversation_history>")
        return "\n".join(lines)


# ── Register built-ins ──────────────────────────────────────────────

neuron_registry.register("system", SystemNeuron())
neuron_registry.register("task", TaskNeuron())
neuron_registry.register("history", HistoryNeuron())
