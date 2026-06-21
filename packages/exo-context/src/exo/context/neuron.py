"""PromptSection — modular prompt composition components.

A PromptSection is a composable unit that produces a prompt fragment from context.
Sections are prioritised: lower priority numbers execute first and appear
earlier in the assembled prompt.

Core sections:
- SystemSection   (priority 100) — date, time, platform info
- TaskSection     (priority 1)   — task ID, input, plan
- HistorySection  (priority 10)  — conversation history with windowing
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

section_registry: Registry[PromptSection] = Registry("section_registry")

# ── ABC ──────────────────────────────────────────────────────────────


class PromptSection(ABC):
    """Abstract base for prompt sections.

    Subclasses implement :meth:`format` to produce a prompt fragment
    from the given context.  The :attr:`priority` controls ordering
    when multiple sections are composed: lower values appear first.

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


# Deprecated alias — use PromptSection
Neuron = PromptSection  # type: ignore[misc]


# ── Built-in sections ────────────────────────────────────────────────


class SystemSection(PromptSection):
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


# Deprecated alias — use SystemSection
SystemNeuron = SystemSection


class TaskSection(PromptSection):
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


# Deprecated alias — use TaskSection
TaskNeuron = TaskSection


class HistorySection(PromptSection):
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
        _hr = getattr(
            _cfg, "_history_rounds", getattr(_cfg, "history_rounds", getattr(_cfg, "limit", 20))
        )
        max_messages = _hr * 2
        windowed = history[-max_messages:]

        lines = ["<conversation_history>"]
        for msg in windowed:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            lines.append(f"[{role}]: {content}")
        lines.append("</conversation_history>")
        return "\n".join(lines)


# Deprecated alias — use HistorySection
HistoryNeuron = HistorySection


# ── Register built-ins ──────────────────────────────────────────────

section_registry.register("system", SystemSection())
section_registry.register("task", TaskSection())
section_registry.register("history", HistorySection())

# Deprecated alias — use section_registry
neuron_registry = section_registry
