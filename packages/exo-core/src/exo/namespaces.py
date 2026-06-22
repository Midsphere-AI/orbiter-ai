"""Grouped configuration namespaces for :class:`~exo.agent.Agent`.

These small frozen dataclasses group the agent's many constructor params by
concern so that power users configure a capability through one obvious,
discoverable handle instead of memorizing a wall of flat keyword arguments::

    agent = Agent(
        name="researcher",
        model="openai:gpt-4o-mini",
        tools=[search],
        planner=PlannerConfig(enabled=True, model="openai:gpt-4o-mini"),
        subagents=SubagentsConfig(max_depth=2, max_children=3),
        batch_tools=ToolBatchConfig(enabled=True, timeout=120),
        guardrails=GuardrailsConfig(approval_tools=["deploy"]),
    )

Every namespace is **additive grouping, not a breaking rename** — the
equivalent flat kwargs (``planning_enabled``, ``max_spawn_depth``,
``batch_tools_timeout``, ``approval_tools``, …) keep working exactly as before.
Pass a config object *or* the flat shorthands, never both for the same concern.

The resolved configs are also exposed as read attributes after construction
(``agent.planner``, ``agent.subagents``, ``agent.guardrails``) for inspection
and serialization.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from exo.guardrail import Guard  # pyright: ignore[reportMissingImports]
    from exo.human import HumanInputHandler


@dataclass(frozen=True)
class PlannerConfig:
    """Planner phase configuration.

    Args:
        enabled: Run a planning pre-pass before the main executor phase.
        model: Optional planner model override (defaults to the agent model).
        instructions: Optional planner-only instructions.
    """

    enabled: bool = False
    model: str | None = None
    instructions: str = ""


@dataclass(frozen=True)
class SubagentsConfig:
    """Self-spawn (sub-agent) configuration.

    Args:
        enabled: Register the ``spawn_self`` tool so the agent can spin up
            copies of itself for parallel sub-tasks.
        max_depth: Maximum recursive spawn depth.
        max_children: Maximum number of children spawned in one call (1..8).
        background: Also register the fire-and-forget background sub-agent
            tools (``spawn_background``/``check_subagent``/``list_subagents``)
            so the agent can launch a child, keep working, and have the
            result injected back when it completes.  Gated behind *enabled*.
        background_timeout: Per-child wall-clock timeout in seconds for
            background sub-agents (``None`` means no timeout).
        background_max: Maximum number of concurrently running background
            sub-agents (back-pressure cap).
    """

    enabled: bool = True
    max_depth: int = 3
    max_children: int = 4
    background: bool = True
    background_timeout: float | None = None
    background_max: int = 8


@dataclass(frozen=True)
class ToolBatchConfig:
    """Programmatic Tool Calling (batch tools) configuration.

    When enabled, the LLM may write code that batches several tool calls in a
    single step, reducing round-trips.

    Args:
        enabled: Turn on Programmatic Tool Calling.
        timeout: Per-batch execution timeout in seconds.
        max_output_bytes: Cap on captured batch output.
        max_tool_calls: Cap on tool calls inside a single batch.
        extra_args: Schema-only args exposed to the executing batch code.
    """

    enabled: bool = False
    timeout: int = 60
    max_output_bytes: int = 200_000
    max_tool_calls: int = 200
    extra_args: dict[str, str] | None = None


@dataclass(frozen=True)
class WorkspaceConfig:
    """Workspace-backed knowledge/file tool configuration (opt-in).

    These tools are **off by default** — a bare agent does not advertise them.
    Pass ``workspace=WorkspaceConfig(enabled=True)`` to give the agent a
    :class:`~exo.context.workspace.Workspace` plus a ``KnowledgeStore`` wired
    into its context, which turns on the knowledge tools (``search_knowledge``,
    ``get_knowledge``, ``grep_knowledge``).  Artifacts the agent offloads are
    auto-indexed into the same store, so the knowledge tools can search them.

    ``read_file`` is registered only when *working_dir* is set, and is scoped
    to that directory (path traversal outside it is rejected).  Leaving
    *working_dir* unset keeps the host filesystem off-limits.

    Args:
        enabled: Register the workspace-backed knowledge tools.
        working_dir: If set, also register ``read_file`` scoped to this directory.
        storage_path: Optional filesystem root to persist workspace artifacts.
        chunk_size: Maximum characters per knowledge-store chunk.
        chunk_overlap: Overlap between consecutive chunks.
    """

    enabled: bool = False
    working_dir: str | None = None
    storage_path: str | None = None
    chunk_size: int = 512
    chunk_overlap: int = 64


@dataclass(frozen=True)
class GuardrailsConfig:
    """Guardrail / human-approval configuration.

    Args:
        guards: Input/output guards run around LLM and tool calls.
        approval_tools: Tool names that require human approval before running.
        human_input_handler: Handler invoked for approval prompts.
    """

    guards: list[Guard] = field(default_factory=list)
    approval_tools: list[str] = field(default_factory=list)
    human_input_handler: HumanInputHandler | None = None


# Sentinel + small extractors so Agent.__init__ can "explode" a namespace config
# back into the flat locals its existing resolution logic already understands,
# while rejecting the ambiguous "config AND flat kwarg" case.

_NS_UNSET: Any = object()
