"""Agent class: the core autonomous unit in Exo."""

from __future__ import annotations

import asyncio
import importlib
import json
import uuid
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from exo.context.config import ContextConfig  # pyright: ignore[reportMissingImports]
    from exo.context.context import Context  # pyright: ignore[reportMissingImports]
    from exo.mcp.client import MCPServerConfig  # pyright: ignore[reportMissingImports]
    from exo.memory.base import AgentMemory, MemoryStore  # pyright: ignore[reportMissingImports]

from pydantic import BaseModel

from exo._internal.errors import unwrap_exception_group
from exo._internal.message_builder import build_messages
from exo._internal.output_parser import OutputParseError, parse_response, parse_tool_arguments
from exo._internal.run_helpers import (
    check_token_budget_pressure,
    drain_ephemeral_messages,
    drain_injected_messages,
    resolve_instructions,
)
from exo.config import (
    parse_model_string,
    validate_budget_awareness,
    validate_injected_tool_args,
    validate_max_spawn_children,
    validate_planning_model,
)
from exo.hooks import Hook, HookManager, HookPoint
from exo.human import HumanInputHandler
from exo.namespaces import (
    GuardrailsConfig,
    PlannerConfig,
    SubagentsConfig,
    ToolBatchConfig,
    WorkspaceConfig,
)
from exo.observability.logging import get_logger  # pyright: ignore[reportMissingImports]
from exo.rail import Guard, GuardAbortError, GuardManager
from exo.skills import DictToolResolver, SkillError, SkillRegistry, ToolResolver
from exo.task_controller import TaskLoopEvent, TaskLoopEventType, TaskLoopQueue
from exo.tool import FunctionTool, Tool, ToolError
from exo.tool_context import ToolContext
from exo.tool_result import tool_error, tool_ok
from exo.types import (
    AgentOutput,
    AssistantMessage,
    ExoError,
    Message,
    MessageContent,
    SystemMessage,
    ToolResult,
    UserMessage,
)

_log = get_logger(__name__)

# Sentinels: distinguish "not provided" (auto-create) from explicit None (disable)
_MEMORY_UNSET: Any = object()
_CONTEXT_UNSET: Any = object()
_SPAWN_UNSET: Any = object()
_RENAMED_UNSET: Any = object()

# Parent-bound sub-agent tools that must never be cloned onto a child agent
# (they capture the parent instance in their closures).
_SUBAGENT_TOOL_NAMES = frozenset(
    {"spawn_self", "spawn_background", "check_subagent", "list_subagents"}
)


def _resolve_renamed(
    new: Any,
    old: Any,
    *,
    default: Any,
    new_name: str,
    old_name: str,
) -> Any:
    """Resolve a renamed keyword argument with a deprecated alias.

    ``new`` is the canonical kwarg (sentinel-defaulted to ``_RENAMED_UNSET``);
    ``old`` is the deprecated alias carrying its original concrete *default*.
    The canonical value wins; supplying the deprecated alias emits a
    :class:`DeprecationWarning`; passing both (with conflicting values) raises.
    """
    new_given = new is not _RENAMED_UNSET
    old_given = old is not default and old != default
    if new_given:
        if old_given:
            raise AgentError(
                f"Cannot combine {new_name}= and {old_name}= — they are the same "
                f"setting. Use {new_name}= (it replaces {old_name}=)."
            )
        return new
    if old_given:
        import warnings

        warnings.warn(
            f"Agent({old_name}=...) is deprecated; use {new_name}= instead.",
            DeprecationWarning,
            stacklevel=3,
        )
        return old
    return default


# ---------------------------------------------------------------------------
# spawn_self helpers — build per-child memory and context
# ---------------------------------------------------------------------------


def _build_child_memory(parent: Any) -> Any:
    """Build memory for a spawned child: fresh short-term, shared long-term."""
    child_memory: Any = _MEMORY_UNSET
    if parent.memory is None:
        child_memory = None
    elif parent.memory is not _MEMORY_UNSET:
        try:
            from exo.memory.base import AgentMemory  # pyright: ignore[reportMissingImports]
            from exo.memory.short_term import (
                ShortTermMemory,  # pyright: ignore[reportMissingImports]
            )

            long_term = getattr(parent.memory, "long_term", None)
            child_memory = AgentMemory(
                short_term=ShortTermMemory(),
                long_term=long_term,
            )
        except ImportError:
            child_memory = None
    return child_memory


def _build_child_context(parent: Any, child_name: str) -> Any:
    """Fork or share context for a spawned child."""
    child_context: Any = _CONTEXT_UNSET
    if parent.context is not None:
        try:
            child_context = parent.context.fork(child_name)
        except Exception:
            child_context = parent.context
    else:
        child_context = None
    return child_context


class TaskLoopAbort(ExoError):
    """Raised when a task loop queue contains an ABORT event."""


def _drain_task_loop_queue(queue: TaskLoopQueue, messages: list) -> None:  # type: ignore[type-arg]
    """Drain all events from a :class:`TaskLoopQueue` and process them.

    Events are sorted by priority (abort first). Processing rules:

    - **ABORT** events raise :class:`TaskLoopAbort` immediately.
    - **STEER** events append a ``UserMessage`` with ``[STEER] {content}``.
    - **FOLLOWUP** events append a ``UserMessage`` with ``[FOLLOWUP] {content}``.

    Args:
        queue: The task loop queue to drain.
        messages: The message list to append steering/followup messages to.

    Raises:
        TaskLoopAbort: If the queue contains any ABORT event.
    """
    events: list[TaskLoopEvent] = []
    while queue:
        evt = queue.pop()
        if evt is not None:
            events.append(evt)

    # Sort by priority (abort < steer < followup) preserving FIFO within same type
    events.sort()

    for evt in events:
        if evt.type == TaskLoopEventType.ABORT:
            raise TaskLoopAbort(evt.content)
        elif evt.type == TaskLoopEventType.STEER:
            messages.append(UserMessage(content=f"[STEER] {evt.content}"))
        elif evt.type == TaskLoopEventType.FOLLOWUP:
            messages.append(UserMessage(content=f"[FOLLOWUP] {evt.content}"))


def _make_default_long_term(db_path: str | None = None) -> Any:
    """Create the default long-term memory store.

    Returns a :class:`SQLiteMemoryStore` using a per-call temp file when
    *db_path* is ``None``, or the given path for durable on-disk storage.
    The SQLite file is created lazily on first use — no disk I/O at
    construction time.

    Args:
        db_path: Explicit path for the SQLite database.  When ``None``
            a fresh temp file path is generated (the file is created on
            first use, not here).
    """
    import tempfile

    from exo.memory.backends.sqlite import (  # pyright: ignore[reportMissingImports]
        SQLiteMemoryStore,
    )

    if db_path is None:
        # Build a path inside a new temp directory; the file itself is not
        # created here — SQLiteMemoryStore opens the connection on first use.
        tmp_dir = tempfile.mkdtemp(prefix="exo_agent_")
        db_path = f"{tmp_dir}/memory.db"
    return SQLiteMemoryStore(db_path)


def _make_default_memory(db_path: str | None = None) -> Any:
    """Try to create a default AgentMemory. Returns None if exo-memory is not installed.

    Args:
        db_path: Optional SQLite database path for durable long-term memory.
            When ``None``, a temp-dir path is used (file created on first use).
    """
    try:
        from exo.memory.base import AgentMemory  # pyright: ignore[reportMissingImports]
        from exo.memory.short_term import ShortTermMemory  # pyright: ignore[reportMissingImports]

        return AgentMemory(short_term=ShortTermMemory(), long_term=_make_default_long_term(db_path))
    except ImportError:
        return None


# Shared context helpers — moved to _internal/context_helpers.py so that
# runner.py can import them without coupling to agent.py internals.
from exo._internal.context_helpers import (
    _ContextAction,
    _ProviderSummarizer,
    _apply_context_windowing,
    _get_context_window_tokens,
    _inject_long_term_knowledge,
    _update_system_token_info,
)


def _make_default_context() -> Any:
    """Try to create a default Context. Returns None if exo-context is not installed."""
    try:
        from exo.context.config import ContextConfig  # pyright: ignore[reportMissingImports]
        from exo.context.context import Context as CtxClass  # pyright: ignore[reportMissingImports]

        cfg = ContextConfig()
        return CtxClass(task_id="__default__", config=cfg)
    except ImportError:
        return None


# Canonical context-size modes → their deprecated aviation-metaphor aliases.
_CONTEXT_MODE_ALIASES: dict[str, str] = {
    "pilot": "large",
    "copilot": "balanced",
    "navigator": "compact",
}


def _make_context_from_mode(mode: Any) -> Any:
    """Create a Context from a mode string.

    Accepted modes: "large", "balanced", "compact". The legacy aviation
    metaphors ("pilot", "copilot", "navigator") remain accepted as
    deprecated aliases. Returns None if exo-context is not installed.
    """
    try:
        from exo.context.config import ContextConfig  # pyright: ignore[reportMissingImports]
        from exo.context.context import Context as CtxClass  # pyright: ignore[reportMissingImports]

        mode_str = str(mode).lower()
        if mode_str in _CONTEXT_MODE_ALIASES:
            import warnings

            canonical = _CONTEXT_MODE_ALIASES[mode_str]
            warnings.warn(
                f"context_mode={mode_str!r} is deprecated; use {canonical!r} instead.",
                DeprecationWarning,
                stacklevel=3,
            )
            mode_str = canonical

        if mode_str == "large":
            cfg = ContextConfig(limit=100, overflow="summarize")
        elif mode_str == "compact":
            cfg = ContextConfig(limit=10, overflow="summarize", cache=True)
        elif mode_str == "balanced":
            cfg = ContextConfig()
        else:
            raise AgentError(
                f"Unknown context_mode {mode!r}. Valid modes are 'large', 'balanced', or 'compact'."
            )
        return CtxClass(task_id="__default__", config=cfg)
    except ImportError:
        return None



class AgentError(ExoError):
    """Raised for agent-level errors (duplicate tools, invalid config, etc.)."""


def _normalize_hitl_tools(hitl_tools: list[str] | None) -> list[str]:
    """Validate and normalize HITL tool names.

    Args:
        hitl_tools: Tool names that should require approval.

    Returns:
        A shallow copy of the configured tool names.

    Raises:
        AgentError: If any entry is empty or not a string.
    """
    if hitl_tools is None:
        return []

    normalized: list[str] = []
    for tool_name in hitl_tools:
        if not isinstance(tool_name, str) or not tool_name.strip():
            raise AgentError("hitl_tools entries must be non-empty strings")
        normalized.append(tool_name)
    return normalized


# Env keys that auto-enable tracing when ``tracing`` is left at its default.
_TRACING_ENV_KEYS: tuple[str, ...] = (
    "EXO_TRACING",
    "LANGFUSE_PUBLIC_KEY",
    "LANGSMITH_API_KEY",
    "LANGCHAIN_API_KEY",
    "PHOENIX_API_KEY",
    "PHOENIX_COLLECTOR_ENDPOINT",
    "BRAINTRUST_API_KEY",
    "OTEL_EXPORTER_OTLP_ENDPOINT",
)


def _tracing_requested(tracing: bool | str | list[str] | None) -> bool:
    """Decide whether to attempt tracing wiring without importing observability.

    Explicit truthy specs always request it; ``False``/``"off"`` always disable.
    A ``None`` default falls through to cheap env auto-detect (on-by-default when
    a backend's keys are present), so a bare ``Agent(...)`` stays zero-overhead.
    """
    import os

    if tracing is False or tracing == "off":
        return False
    if tracing is not None:
        return bool(tracing)
    env = os.environ.get("EXO_TRACING")
    if env is not None:
        return env.strip().lower() != "off"
    return any(os.environ.get(k) for k in _TRACING_ENV_KEYS)


class Agent:
    """An autonomous LLM-powered agent with tools and lifecycle hooks.

    Agents are the core building block in Exo. Each agent wraps an LLM
    model, a set of tools, optional handoff targets, and lifecycle hooks.
    The ``run()`` method (added in a later session) executes the agent's
    tool loop.

    All parameters are keyword-only; only ``name`` is required.

    Related params can be configured through one grouped namespace object
    instead of the flat kwargs (additive — the flat kwargs still work)::

        Agent(
            name="researcher",
            planner=PlannerConfig(enabled=True, model="openai:gpt-4o-mini"),
            subagents=SubagentsConfig(max_depth=2, max_children=3),
            batch_tools=ToolBatchConfig(enabled=True, timeout=120),
            guardrails=GuardrailsConfig(approval_tools=["deploy"]),
            context=ContextConfig(limit=50, overflow="summarize"),
        )

    The resolved configs are exposed as read attributes (``agent.planner``,
    ``agent.subagents``, ``agent.guardrails``). Passing both a namespace config
    and a conflicting flat kwarg for the same concern raises ``AgentError``.

    Args:
        name: Unique identifier for this agent.
        model: Model string in ``"provider:model_name"`` format.
        instructions: System prompt. Can be a string or an async callable
            that receives a context dict and returns a string.
        tools: Tools available to this agent.
        transfers: Other agents this agent can hand control to. (Deprecated
            alias: ``handoffs``.)
        guards: Input/output guards run around LLM and tool calls. (Deprecated
            alias: ``rails``.)
        hooks: Lifecycle hooks as ``(HookPoint, Hook)`` tuples.
        output_type: Pydantic model class for structured output validation.
        max_steps: Maximum LLM-tool round-trips before stopping.
        temperature: LLM sampling temperature.
        max_tokens: Maximum output tokens per LLM call.
        planning_enabled: When ``True``, the runtime may execute a planner
            phase before the main executor phase.
        planning_model: Optional planner model override. When unset, planning
            uses the main agent model.
        planning_instructions: Optional planner-only instructions.
        context_pressure: Optional context-pressure mode that nudges the agent
            as its context fills. Valid values are ``"per-message"`` and
            ``"limit:<0-100>"``. (Deprecated alias: ``budget_awareness``.)
        approval_tools: Tool names that require human approval before execution.
            (Deprecated alias: ``hitl_tools``.)
        bare_tools: When ``True``, suppress auto-registered helper tools
            (``retrieve_artifact``, context tools). ``activate_skill``,
            ``spawn_self``, and batch-tools (PTC) tools are **not** affected.
        human_input_handler: Handler for approval prompts. When set
            alongside ``approval_tools``, tools in that list will block for
            human approval before executing. Defaults to ``None`` (no gate).
        emit_mcp_progress: Whether MCP progress events should be emitted.
        injected_tool_args: Schema-only tool arguments exposed to the LLM.
        store: Memory store shorthand (batteries-included convenience).  Takes
            precedence over *memory* when both are supplied (raises if both are
            non-sentinel values).  Accepted values:

            - ``"sqlite"`` (default): SQLite in a per-agent temp directory;
              lazily created on first use — no disk I/O at construction time.
            - ``"<path>.db"`` / any string ending in ``.db``: durable SQLite at
              that path.
            - ``"memory"``: in-process :class:`ShortTermMemory` only (no file).
            - a :class:`MemoryStore` instance: use it directly as the long-term
              store.
            - ``False`` / ``None``: disable memory entirely.
        memory: Legacy memory kwarg.  Accepts an :class:`AgentMemory` (or any
            memory store), ``None`` to disable, or the sentinel default to
            auto-create.  When *store* is also provided they must not conflict.
        context: Optional context engine for hierarchical state and prompt building.
        allow_self_spawn: When ``True`` (default), automatically adds a
            ``spawn_self(tasks)`` tool that lets the agent spin up copies of
            itself for parallel sub-tasks.  Set ``subagents=False`` to disable.
        subagents: Convenience flag that overrides *allow_self_spawn*.  Pass
            ``subagents=False`` to disable the ``spawn_self`` tool.
        max_spawn_depth: Maximum recursive spawn depth (default 3). When a spawned
            agent's depth equals or exceeds this value, ``spawn_self`` returns an
            error string instead of spawning.
        tool_gate: Conditional tool injection that preserves the LLM KV cache.
            Maps a trigger tool name to a list of tools that become available
            after the trigger tool executes. Gated tools are **appended** to the
            tool list (never reordered) so the cached prefix stays valid.
    """

    def __init__(
        self,
        *,
        name: str,
        model: str = "openai:gpt-4o-mini",
        instructions: str | Callable[..., Any] = "",
        tools: list[Tool] | None = None,
        transfers: list[Agent] | None = _RENAMED_UNSET,
        handoffs: list[Agent] | None = None,
        hooks: list[tuple[HookPoint, Hook]] | None = None,
        guards: list[Guard] | None = _RENAMED_UNSET,
        rails: list[Guard] | None = None,
        output_type: type[BaseModel] | None = None,
        max_steps: int = 10,
        temperature: float = 1.0,
        max_tokens: int | None = None,
        planner: PlannerConfig | None = None,
        planning_enabled: bool = False,
        planning_model: str | None = None,
        planning_instructions: str = "",
        context_pressure: str | None = _RENAMED_UNSET,
        budget_awareness: str | None = None,
        guardrails: GuardrailsConfig | None = None,
        approval_tools: list[str] | None = _RENAMED_UNSET,
        hitl_tools: list[str] | None = None,
        bare_tools: bool = False,
        human_input_handler: HumanInputHandler | None = None,
        emit_mcp_progress: bool = True,
        injected_tool_args: dict[str, str] | None = None,
        store: MemoryStore | AgentMemory | str | bool | None = _MEMORY_UNSET,
        memory: AgentMemory | MemoryStore | None = _MEMORY_UNSET,
        context_mode: Literal["large", "balanced", "compact", "pilot", "copilot", "navigator"]
        | None = _CONTEXT_UNSET,
        context: Context | ContextConfig | None = _CONTEXT_UNSET,
        context_limit: int | None = None,
        overflow: str | None = None,
        cache: bool | None = None,
        subagents: bool | SubagentsConfig | None = None,
        allow_self_spawn: bool = _SPAWN_UNSET,
        max_spawn_depth: int = 3,
        max_spawn_children: int = 4,
        background_subagents: bool = True,
        background_timeout: float | None = None,
        background_max: int = 8,
        batch_tools: bool | ToolBatchConfig = _RENAMED_UNSET,
        batch_tools_timeout: int = _RENAMED_UNSET,
        batch_tools_max_output_bytes: int = _RENAMED_UNSET,
        batch_tools_max_tool_calls: int = _RENAMED_UNSET,
        batch_tools_extra_args: dict[str, str] | None = _RENAMED_UNSET,
        ptc: bool = False,
        ptc_timeout: int = 60,
        ptc_max_output_bytes: int = 200_000,
        ptc_max_tool_calls: int = 200,
        ptc_extra_args: dict[str, str] | None = None,
        skills: SkillRegistry | None = None,
        tool_resolver: ToolResolver | dict[str, Tool | list[Tool]] | None = None,
        tool_gate: dict[str, list[Tool]] | None = None,
        workspace: WorkspaceConfig | None = None,
        tracing: bool | str | list[str] | None = None,
    ) -> None:
        if max_steps < 1:
            raise AgentError(f"max_steps must be >= 1, got {max_steps}")

        # ---- Namespace configs → flat locals --------------------------------
        # The grouped *Config objects (planner=, subagents=, batch_tools=,
        # guardrails=) are additive sugar over the flat kwargs. Each explodes
        # into the flat locals the existing resolution logic already understands;
        # passing both a config and a conflicting flat kwarg for the same concern
        # is rejected so there is exactly one source of truth.
        if planner is not None:
            if planning_enabled or planning_model is not None or planning_instructions:
                raise AgentError(
                    "Cannot combine planner=PlannerConfig(...) with planning_enabled/"
                    "planning_model/planning_instructions. Pass one or the other."
                )
            planning_enabled = planner.enabled
            planning_model = planner.model
            planning_instructions = planner.instructions

        if isinstance(subagents, SubagentsConfig):
            if allow_self_spawn is not _SPAWN_UNSET:
                raise AgentError(
                    "Cannot combine subagents=SubagentsConfig(...) with allow_self_spawn=. "
                    "Pass one or the other."
                )
            _sub_cfg = subagents
            subagents = _sub_cfg.enabled
            max_spawn_depth = _sub_cfg.max_depth
            max_spawn_children = _sub_cfg.max_children
            background_subagents = _sub_cfg.background
            background_timeout = _sub_cfg.background_timeout
            background_max = _sub_cfg.background_max

        if isinstance(batch_tools, ToolBatchConfig):
            if ptc or any(
                x is not _RENAMED_UNSET
                for x in (
                    batch_tools_timeout,
                    batch_tools_max_output_bytes,
                    batch_tools_max_tool_calls,
                    batch_tools_extra_args,
                )
            ):
                raise AgentError(
                    "Cannot combine batch_tools=ToolBatchConfig(...) with the flat "
                    "ptc*/batch_tools_* kwargs. Pass one or the other."
                )
            _batch_cfg = batch_tools
            batch_tools = _batch_cfg.enabled
            batch_tools_timeout = _batch_cfg.timeout
            batch_tools_max_output_bytes = _batch_cfg.max_output_bytes
            batch_tools_max_tool_calls = _batch_cfg.max_tool_calls
            batch_tools_extra_args = _batch_cfg.extra_args

        if guardrails is not None:
            if (
                guards is not _RENAMED_UNSET
                or rails is not None
                or approval_tools is not _RENAMED_UNSET
                or hitl_tools is not None
                or human_input_handler is not None
            ):
                raise AgentError(
                    "Cannot combine guardrails=GuardrailsConfig(...) with guards/rails/"
                    "approval_tools/hitl_tools/human_input_handler. Pass one or the other."
                )
            guards = list(guardrails.guards)
            approval_tools = list(guardrails.approval_tools)
            human_input_handler = guardrails.human_input_handler

        self.name = name
        self.model = model
        self.provider_name, self.model_name = parse_model_string(model)
        self.instructions = instructions
        self.output_type = output_type
        self.max_steps = max_steps
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.planning_enabled = planning_enabled
        self.planning_model = validate_planning_model(planning_model)
        self.planning_instructions = planning_instructions
        # ``context_pressure=`` is the canonical knob; ``budget_awareness=`` is
        # a deprecated alias.
        _context_pressure = _resolve_renamed(
            context_pressure,
            budget_awareness,
            default=None,
            new_name="context_pressure",
            old_name="budget_awareness",
        )
        self.budget_awareness = validate_budget_awareness(_context_pressure)
        self.emit_mcp_progress = emit_mcp_progress
        self.injected_tool_args = validate_injected_tool_args(injected_tool_args)
        # ``approval_tools=`` is the canonical knob; ``hitl_tools=`` is a
        # deprecated alias.
        _approval_tools = _resolve_renamed(
            approval_tools,
            hitl_tools,
            default=None,
            new_name="approval_tools",
            old_name="hitl_tools",
        )
        normalized_hitl_tools = _normalize_hitl_tools(_approval_tools)
        self.bare_tools: bool = bare_tools
        # Self-spawn: on by default. ``subagents=`` is the canonical knob;
        # ``allow_self_spawn=`` is a deprecated alias kept for backward compat.
        if allow_self_spawn is not _SPAWN_UNSET:
            import warnings

            warnings.warn(
                "Agent(allow_self_spawn=...) is deprecated; use subagents= instead "
                "(subagents=False disables the spawn_self tool).",
                DeprecationWarning,
                stacklevel=2,
            )
        if subagents is not None:
            _effective_self_spawn = bool(subagents)
        elif allow_self_spawn is not _SPAWN_UNSET:
            _effective_self_spawn = bool(allow_self_spawn)
        else:
            _effective_self_spawn = True
        self.allow_self_spawn: bool = _effective_self_spawn
        self.max_spawn_depth: int = max_spawn_depth
        self.max_spawn_children: int = validate_max_spawn_children(max_spawn_children)
        # ``batch_tools*`` are the canonical knobs for Programmatic Tool Calling;
        # the ``ptc*`` spellings remain as deprecated aliases. Internally the
        # state is still tracked under the ``self.ptc*`` attributes.
        self.ptc: bool = _resolve_renamed(
            batch_tools, ptc, default=False, new_name="batch_tools", old_name="ptc"
        )
        self.ptc_timeout: int = _resolve_renamed(
            batch_tools_timeout,
            ptc_timeout,
            default=60,
            new_name="batch_tools_timeout",
            old_name="ptc_timeout",
        )
        self.ptc_max_output_bytes: int = _resolve_renamed(
            batch_tools_max_output_bytes,
            ptc_max_output_bytes,
            default=200_000,
            new_name="batch_tools_max_output_bytes",
            old_name="ptc_max_output_bytes",
        )
        self.ptc_max_tool_calls: int = _resolve_renamed(
            batch_tools_max_tool_calls,
            ptc_max_tool_calls,
            default=200,
            new_name="batch_tools_max_tool_calls",
            old_name="ptc_max_tool_calls",
        )
        # Schema-only args added to the outer ``__exo_ptc__`` tool call.
        # The LLM fills them alongside ``code``; they are NOT propagated
        # to inner tools (unlike ``injected_tool_args``) and are instead
        # exposed to the executing PTC code as a ``ptc_args`` dict.
        _batch_extra_args = _resolve_renamed(
            batch_tools_extra_args,
            ptc_extra_args,
            default=None,
            new_name="batch_tools_extra_args",
            old_name="ptc_extra_args",
        )
        self.ptc_extra_args: dict[str, str] = validate_injected_tool_args(_batch_extra_args)
        # Internal: spawn depth (0 for top-level agents; incremented for each spawn level)
        self._spawn_depth: int = 0
        # Internal: provider reference stored during run() for use by spawn_self tool
        self._current_provider: Any = None
        # ---- Background sub-agents (fire-and-forget) ----
        self._background_subagents: bool = bool(background_subagents)
        self._bg_timeout: float | None = background_timeout
        self._bg_max: int = background_max
        # Lazily constructed on first spawn_background call.
        self._bg_handler: Any = None
        # Live registry of detached background tasks (owner for cancel/cleanup).
        self._bg_tasks: dict[str, asyncio.Task] = {}
        # True while a run() is in flight — drives HOT (inject live) vs
        # WAKEUP (queue for next run) routing of background results.
        self._is_running: bool = False
        # Skills: lazy activation via activate_skill tool
        self._skill_registry: SkillRegistry | None = skills
        self._tool_resolver: ToolResolver | None = None
        if tool_resolver is not None:
            if isinstance(tool_resolver, dict):
                self._tool_resolver = DictToolResolver(tool_resolver)
            else:
                self._tool_resolver = tool_resolver
        # ---- Memory resolution ------------------------------------------------
        # Priority: store= > memory= > auto-create
        # `store=` is the single recommended knob; `memory=` is a deprecated alias.
        if store is not _MEMORY_UNSET and memory is not _MEMORY_UNSET:
            raise AgentError(
                "Cannot combine 'store' and 'memory' kwargs. "
                "Use 'store=' — it accepts everything 'memory=' did "
                "(an AgentMemory/MemoryStore instance, 'sqlite', 'memory', a path, "
                "or False to disable)."
            )
        if memory is not _MEMORY_UNSET:
            import warnings

            warnings.warn(
                "Agent(memory=...) is deprecated; use store= instead. "
                "store= accepts an AgentMemory/MemoryStore instance, 'sqlite', "
                "'memory', a '.db' path, or False to disable.",
                DeprecationWarning,
                stacklevel=2,
            )
        if store is not _MEMORY_UNSET:
            # Resolve store= shorthand
            resolved_memory: Any
            if store is False or store is None:
                resolved_memory = None
            elif store == "sqlite":
                resolved_memory = _make_default_memory()
            elif store == "memory":
                # In-process only: ShortTermMemory as both short and long term
                try:
                    from exo.memory.base import AgentMemory  # pyright: ignore[reportMissingImports]
                    from exo.memory.short_term import (
                        ShortTermMemory,  # pyright: ignore[reportMissingImports]
                    )

                    resolved_memory = AgentMemory(
                        short_term=ShortTermMemory(),
                        long_term=ShortTermMemory(),
                    )
                except ImportError:
                    resolved_memory = None
            elif isinstance(store, str):
                # A string store must be a filesystem path to a SQLite db, not a
                # backend name. Reject bare words (e.g. store="redis") that would
                # otherwise be silently written as a file of that name.
                if "/" not in store and "\\" not in store and "." not in store:
                    raise AgentError(
                        f"Unknown store {store!r}. Use store='sqlite' (temp file), "
                        f"store='memory' (in-process), a '.db' path for a durable "
                        f"SQLite file, or pass a MemoryStore/AgentMemory instance. "
                        f"To disable memory use store=False."
                    )
                # Treat as a durable SQLite path
                resolved_memory = _make_default_memory(db_path=store)
            elif hasattr(store, "short_term") and hasattr(store, "long_term"):
                # Already an AgentMemory — use it directly (memory= passthrough).
                resolved_memory = store
            else:
                # Assume it's a MemoryStore instance — wrap in AgentMemory
                try:
                    from exo.memory.base import AgentMemory  # pyright: ignore[reportMissingImports]
                    from exo.memory.short_term import (
                        ShortTermMemory,  # pyright: ignore[reportMissingImports]
                    )

                    resolved_memory = AgentMemory(short_term=ShortTermMemory(), long_term=store)
                except ImportError:
                    resolved_memory = None
            memory = resolved_memory
            self._memory_is_auto: bool = False
        elif memory is _MEMORY_UNSET:
            # Auto-create: SQLite in a temp dir (lazy — no I/O at construction)
            memory = _make_default_memory()
            self._memory_is_auto = True
        else:
            self._memory_is_auto = False
        # -----------------------------------------------------------------------
        self.memory: AgentMemory | None = memory
        self.conversation_id: str | None = None
        # Resolve context: new shorthand params → context_mode → context → default.
        _has_new_ctx = any(x is not None for x in (context_limit, overflow, cache))
        if _has_new_ctx and context is not _CONTEXT_UNSET:
            raise AgentError(
                "Cannot combine 'context' with 'context_limit'/'overflow'/'cache'. "
                "Use either context= or the shorthand params."
            )
        if _has_new_ctx and context_mode is not _CONTEXT_UNSET:
            raise AgentError(
                "Cannot combine 'context_mode' with 'context_limit'/'overflow'/'cache'. "
                "Use either context_mode= or the shorthand params."
            )

        if _has_new_ctx:
            try:
                from exo.context.config import (  # pyright: ignore[reportMissingImports]
                    ContextConfig as _CtxConfig,
                )
                from exo.context.context import (  # pyright: ignore[reportMissingImports]
                    Context as _CtxClass,
                )
            except ImportError:
                self.context: Context | None = None
                self._context_is_auto: bool = True
            else:
                _kw: dict[str, Any] = {}
                if context_limit is not None:
                    _kw["limit"] = context_limit
                if overflow is not None:
                    _kw["overflow"] = overflow
                if cache is not None:
                    _kw["cache"] = cache
                self.context = _CtxClass(task_id="__default__", config=_CtxConfig(**_kw))
                self._context_is_auto = False
        elif context is not _CONTEXT_UNSET:
            # Accept a ready Context, a ContextConfig (wrapped into a Context),
            # or None (context disabled).
            try:
                from exo.context.config import (  # pyright: ignore[reportMissingImports]
                    ContextConfig as _CtxConfig,
                )
                from exo.context.context import (  # pyright: ignore[reportMissingImports]
                    Context as _CtxClass,
                )
            except ImportError:
                self.context = context
            else:
                if isinstance(context, _CtxConfig):
                    self.context = _CtxClass(task_id="__default__", config=context)
                else:
                    self.context = context
            self._context_is_auto = False
        elif context_mode is not _CONTEXT_UNSET:
            self.context = None if context_mode is None else _make_context_from_mode(context_mode)
            self._context_is_auto = False
        else:
            self.context = _make_default_context()
            self._context_is_auto = True
        self._memory_persistence: Any = None
        # Workspace for large-output tool result offloading (lazy-created on first use).
        # When workspace=WorkspaceConfig(enabled=True) it is created eagerly with an
        # attached KnowledgeStore so the opt-in knowledge tools can search artifacts.
        self._workspace: Any = None
        self._knowledge_store: Any = None
        # Opt-in workspace-backed knowledge/file tools (off by default).
        self.workspace = workspace if workspace is not None else WorkspaceConfig()
        self._working_dir = self.workspace.working_dir

        # Tools indexed by name for O(1) lookup during execution
        self.tools: dict[str, Tool] = {}
        self._cached_tool_schemas: list[dict[str, Any]] | None = None
        if tools is not None:
            if not isinstance(tools, list):
                raise AgentError(
                    f"'tools' must be a list of Tool objects, e.g. tools=[my_tool]; "
                    f"got {type(tools).__name__!r}"
                )
            for i, t in enumerate(tools):
                if isinstance(t, (str, int, bytes)):
                    raise AgentError(
                        f"'tools[{i}]' is a {type(t).__name__!r}, not a Tool. "
                        f"Decorate your function with @tool to make it a Tool."
                    )
                if not hasattr(t, "name") and not hasattr(t, "execute"):
                    raise AgentError(
                        f"'tools[{i}]' ({t!r}) is not a Tool. "
                        f"Decorate your function with @tool to make it a Tool."
                    )
        if tools:
            for t in tools:
                self._register_tool(t)

        # Auto-register activate_skill tool when skills are provided
        if self._skill_registry is not None:
            self._register_tool(self._make_activate_skill_tool())

        if not self.bare_tools:
            # Register retrieve_artifact when the context mode supports
            # workspace offloading (disabled for context=None / overflow=hook).
            if self._should_enable_artifact_offloading() and "retrieve_artifact" not in self.tools:
                self._register_retrieve_artifact()

            # Auto-load context tools (planning, knowledge, file) when context is available
            self._auto_load_context_tools()

        # Transfer targets indexed by name. ``transfers=`` is the canonical
        # knob; ``handoffs=`` is a deprecated alias. Stored under
        # ``self.handoffs`` for backward compat.
        _transfers = _resolve_renamed(
            transfers, handoffs, default=None, new_name="transfers", old_name="handoffs"
        )
        self.handoffs: dict[str, Agent] = {}
        if _transfers:
            for agent in _transfers:
                self._register_handoff(agent)

        # Lock for asyncio-safe runtime mutations (add_tool, add_mcp_server, add_handoff)
        self._tools_lock: asyncio.Lock = asyncio.Lock()

        # Task loop queue: external callers push ABORT/STEER/FOLLOWUP events here;
        # the run loop drains this queue before each LLM call.
        self.task_loop_queue: TaskLoopQueue = TaskLoopQueue()

        # Queue for live message injection into a running agent
        self._injected_messages: asyncio.Queue[str] = asyncio.Queue()

        # Queue for ephemeral messages: visible for ONE LLM call, then auto-removed
        self._ephemeral_messages: asyncio.Queue[Message] = asyncio.Queue()

        # Queue for tool-emitted streaming events (drained by run.stream())
        self._event_queue: asyncio.Queue = asyncio.Queue()

        # Auto-register spawn_self tool when self-spawn is enabled
        if _effective_self_spawn:
            self._register_tool(self._make_spawn_self_tool())
            # Background (fire-and-forget) sub-agent tools ride on self-spawn.
            if self._background_subagents:
                self._register_tool(self._make_spawn_background_tool())
                self._register_tool(self._make_check_subagent_tool())
                self._register_tool(self._make_list_subagents_tool())

        self.hitl_tools = normalized_hitl_tools
        self._human_input_handler: HumanInputHandler | None = human_input_handler
        self._validate_hitl_tools()

        # PTC: register synthetic PTC tool when programmatic tool calling is on
        if self.ptc:
            from exo.ptc import PTC_TOOL_NAME, PTCTool

            if PTC_TOOL_NAME in self.tools:
                raise AgentError(
                    f"Cannot enable ptc: a tool named '{PTC_TOOL_NAME}' is already registered"
                )
            self.tools[PTC_TOOL_NAME] = PTCTool(
                agent=self,
                timeout=self.ptc_timeout,
                max_output_bytes=self.ptc_max_output_bytes,
                max_tool_calls=self.ptc_max_tool_calls,
                extra_args=self.ptc_extra_args or None,
            )
            self._cached_tool_schemas = None

        # Lifecycle hooks
        self.hook_manager = HookManager()
        self._has_user_hooks: bool = bool(hooks)  # tracks explicitly-provided hooks only
        if hooks:
            for point, hook in hooks:
                self.hook_manager.add(point, hook)

        # Guards integration: create GuardManager and register hooks for all
        # points. ``guards=`` is the canonical knob; ``rails=`` is a deprecated
        # alias. State is stored under ``self.rail_manager`` for backward compat.
        _guards = _resolve_renamed(guards, rails, default=None, new_name="guards", old_name="rails")
        if _guards:
            self.rail_manager: GuardManager | None = GuardManager()
            for guard in _guards:
                self.rail_manager.add(guard)
            for point in HookPoint:
                if point == HookPoint.CONTEXT_WINDOW:
                    continue  # context windowing is not a guardrail concern
                self.hook_manager.add(point, self.rail_manager.hook_for(point))
        else:
            self.rail_manager = None

        # Tool gate: conditional tool injection (append-only for KV-cache safety)
        self._tool_gate: dict[str, list[Tool]] = dict(tool_gate) if tool_gate else {}
        self._unlocked_gates: set[str] = set()
        if self._tool_gate:
            # Validate that trigger names actually refer to registered tools
            for trigger_name in self._tool_gate:
                if trigger_name not in self.tools:
                    raise AgentError(f"tool_gate trigger '{trigger_name}' is not a registered tool")
            self.hook_manager.add(HookPoint.POST_TOOL_CALL, self._tool_gate_hook)

        # Auto-attach memory persistence hooks when a MemoryStore is provided
        if memory is not None:
            self._attach_memory_persistence(memory)

        # ---- Inspectable namespace configs (Tier-3) -------------------------
        # Expose the resolved per-concern config as a read attribute so callers
        # can introspect/serialize (``agent.planner.enabled``,
        # ``agent.subagents.max_depth``, ``agent.guardrails.approval_tools``).
        self.planner = PlannerConfig(
            enabled=self.planning_enabled,
            model=self.planning_model,
            instructions=self.planning_instructions,
        )
        self.subagents = SubagentsConfig(
            enabled=self.allow_self_spawn,
            max_depth=self.max_spawn_depth,
            max_children=self.max_spawn_children,
            background=self._background_subagents,
            background_timeout=self._bg_timeout,
            background_max=self._bg_max,
        )
        self.guardrails = GuardrailsConfig(
            guards=list(_guards) if _guards else [],
            approval_tools=list(self.hitl_tools),
            human_input_handler=self._human_input_handler,
        )

        # ---- Optional tracing wiring (exo-observability, lazy) --------------
        # Registered last so the root span opens first (START runs in order) and
        # closes last (FINISHED runs in order) — guard/memory hooks nest inside.
        self.tracing = tracing
        self._tracer: Any = None
        if _tracing_requested(tracing):
            try:
                from exo.observability.tracer import install_tracing

                self._tracer = install_tracing(self, tracing)
            except ImportError:
                _log.debug(
                    "tracing requested on agent %r but exo-observability is not installed",
                    self.name,
                )

    # ---- Canonical-name read aliases (Tier-2 vocabulary) -------------------
    # The constructor accepts modern names (``transfers``/``approval_tools``/
    # ``context_pressure``/``batch_tools``/``guards``); internally the state is
    # stored under the historical attribute names. These read-only properties
    # expose the modern spelling for inspection without duplicating state.

    @property
    def transfers(self) -> dict[str, Agent]:
        """Transfer (handoff) targets keyed by agent name."""
        return self.handoffs

    @property
    def approval_tools(self) -> list[str]:
        """Tool names that require human approval before executing."""
        return list(self.hitl_tools)

    @property
    def context_pressure(self) -> str | None:
        """Context-pressure mode (``"per-message"`` / ``"limit:<0-100>"``)."""
        return self.budget_awareness

    @property
    def batch_tools(self) -> bool:
        """Whether Programmatic Tool Calling (batch tools) is enabled."""
        return self.ptc

    def _register_tool(self, t: Tool) -> None:
        """Add a tool, raising on duplicate names.

        When a ``large_output=True`` tool is registered, the context mode
        supports offloading, and ``retrieve_artifact`` is not yet present,
        auto-registers the ``retrieve_artifact`` tool so the LLM can access
        offloaded results.

        Args:
            t: The tool to register.

        Raises:
            AgentError: If a tool with the same name is already registered.
        """
        if t.name in self.tools:
            raise AgentError(f"Duplicate tool name '{t.name}' on agent '{self.name}'")
        self.tools[t.name] = t
        self._cached_tool_schemas = None
        # Auto-register retrieve_artifact when the first large_output=True tool is added
        # and the context mode supports workspace offloading.
        if (
            not self.bare_tools
            and getattr(t, "large_output", False)
            and self._should_enable_artifact_offloading()
            and "retrieve_artifact" not in self.tools
        ):
            self._register_retrieve_artifact()

    def _register_handoff(self, agent: Agent) -> None:
        """Add a handoff target, raising on duplicate names.

        Args:
            agent: The target agent.

        Raises:
            AgentError: If a handoff with the same name is already registered.
        """
        if agent.name in self.handoffs:
            raise AgentError(f"Duplicate handoff agent '{agent.name}' on agent '{self.name}'")
        self.handoffs[agent.name] = agent
        # Invalidate schema cache: ``get_ptc_eligible_tools`` uses
        # ``self.handoffs`` keys to exclude tools whose name matches a
        # handoff target.  Without invalidation, a runtime handoff add on
        # a ``ptc=True`` agent would leave a stale schema where a tool is
        # still listed that should now be excluded.
        self._cached_tool_schemas = None

    def _should_enable_artifact_offloading(self) -> bool:
        """Return True only when the context mode supports workspace offloading.

        Disabled when context is ``None`` or overflow is ``hook``-based — the
        ``retrieve_artifact`` tool cannot function without a managed context.
        """
        if self.context is None:
            return False
        try:
            from exo.context.config import OverflowStrategy  # pyright: ignore[reportMissingImports]

            if self.context.config.overflow == OverflowStrategy.HOOK:
                return False
        except (ImportError, AttributeError):
            pass
        return True

    def _register_retrieve_artifact(self) -> None:
        """Auto-register the ``retrieve_artifact`` tool for workspace access.

        Called automatically by :meth:`_register_tool` when the first
        ``large_output=True`` tool is registered on this agent and the
        context mode supports offloading.
        """
        agent_ref = self

        async def retrieve_artifact(id: str) -> str:
            """Retrieve the content of a large tool result stored as an artifact.

            Args:
                id: The artifact ID returned in the pointer string from a
                    large_output tool.

            Returns:
                The full content of the stored artifact, or a structured error
                with recovery hint if retrieval fails.
            """
            try:
                if agent_ref._workspace is None:
                    return tool_error(
                        "No workspace available",
                        hint=(
                            "No artifacts have been stored yet. Use a large_output "
                            "tool first, then call retrieve_artifact with the "
                            "returned artifact ID."
                        ),
                    )
                content = agent_ref._workspace.read(id)
                if content is None:
                    return tool_error(
                        f"Artifact '{id}' not found in workspace",
                        hint=(
                            "Check the artifact ID — use the exact string "
                            "returned in the pointer message from the "
                            "large_output tool."
                        ),
                    )
                return content
            except Exception as exc:
                return tool_error(
                    f"Failed to read artifact: {exc}",
                    hint=(
                        "Retry the retrieve_artifact call. If the error "
                        "persists, re-run the original tool that produced "
                        "the artifact."
                    ),
                )

        # Direct dict insertion avoids triggering the duplicate check in _register_tool
        # and the large_output auto-registration loop.
        self.tools["retrieve_artifact"] = FunctionTool(retrieve_artifact, name="retrieve_artifact")
        self._cached_tool_schemas = None

    def _validate_hitl_tools(self) -> None:
        """Ensure all HITL tool names reference registered tools."""
        missing = sorted(
            {tool_name for tool_name in self.hitl_tools if tool_name not in self.tools}
        )
        if missing:
            raise AgentError(
                f"hitl_tools contains unknown tool names for agent '{self.name}': {', '.join(missing)}"
            )

    def _auto_load_context_tools(self) -> None:
        """Auto-load, bind, and register context tools when exo-context is installed.

        Called by ``__init__`` after context resolution. Skipped when
        ``self.context`` is ``None``, exo-context is not installed, or the
        overflow strategy is ``hook`` (context management fully delegated to hooks).
        Context tools are fresh instances per agent to avoid shared mutable state.

        Planning tools (``add_todo``/``complete_todo``/``get_todo``) are always
        registered — they manage their own state and have no backing store.  The
        workspace-backed knowledge tools (``search_knowledge``/``get_knowledge``/
        ``grep_knowledge``) and the ``read_file`` tool are **opt-in** via
        ``workspace=WorkspaceConfig(...)``; without it they are not advertised to
        the LLM (otherwise they would always error for lack of a backing store).
        """
        if self.context is None:
            return
        # When overflow is "hook", context management is fully delegated to
        # user-provided hooks — don't inject built-in context tools.
        try:
            from exo.context.config import OverflowStrategy  # pyright: ignore[reportMissingImports]

            if self.context.config.overflow == OverflowStrategy.HOOK:
                _log.debug("skipping context tools for agent %r (overflow=hook)", self.name)
                return
        except (ImportError, AttributeError):
            pass
        try:
            from exo.context.tools import (  # pyright: ignore[reportMissingImports]
                get_file_tools,
                get_knowledge_tools,
                get_planning_tools,
            )

            # Planning tools are always on (no backing store required).
            tools_to_load = get_planning_tools()

            # Knowledge tools require a workspace + knowledge store wired into the
            # context state — opt in via WorkspaceConfig(enabled=True).
            if self.workspace.enabled:
                self._ensure_workspace()
                tools_to_load += get_knowledge_tools()

            # read_file is registered only when a working_dir is configured, and
            # is scoped to that directory.
            if self._working_dir is not None:
                self.context.state.set("working_dir", self._working_dir)
                tools_to_load += get_file_tools()

            added = False
            for t in tools_to_load:
                t.bind(self.context)
                # Skip if user already registered a tool with the same name
                if t.name not in self.tools:
                    self.tools[t.name] = t
                    added = True
            # Defensive: invalidate the schema cache if any context tool
            # was injected.  Currently this method runs only during
            # ``__init__`` (before the cache is populated), but an explicit
            # invalidation protects against future call sites that might
            # run it after the cache has been built.
            if added:
                self._cached_tool_schemas = None
            _log.debug(
                "auto-loaded context tools for agent %r (%d tools)",
                self.name,
                len([t for t in self.tools.values() if getattr(t, "_is_context_tool", False)]),
            )
        except ImportError:
            pass

    def _ensure_workspace(self) -> Any:
        """Get-or-create the agent's workspace, with a KnowledgeStore attached.

        Creates a :class:`~exo.context.workspace.Workspace` backed by a
        ``KnowledgeStore`` (so every artifact write auto-indexes) and wires both
        into the context state under ``"workspace"`` / ``"knowledge_store"`` so
        the knowledge tools can reach them.  Idempotent — returns the existing
        workspace on subsequent calls.  Returns ``None`` when exo-context is not
        installed.
        """
        if self._workspace is not None:
            return self._workspace
        try:
            from exo.context._internal.knowledge import (  # pyright: ignore[reportMissingImports]
                KnowledgeStore,
            )
            from exo.context.workspace import Workspace  # pyright: ignore[reportMissingImports]
        except ImportError:
            return None

        self._knowledge_store = KnowledgeStore(
            chunk_size=self.workspace.chunk_size,
            chunk_overlap=self.workspace.chunk_overlap,
        )
        self._workspace = Workspace(
            workspace_id=f"agent_{self.name}",
            storage_path=self.workspace.storage_path,
            knowledge_store=self._knowledge_store,
        )
        if self.context is not None:
            self.context.state.set("workspace", self._workspace)
            self.context.state.set("knowledge_store", self._knowledge_store)
        return self._workspace

    async def _offload_large_result(self, tool_name: str, content: str) -> str:
        """Store a large tool result in the workspace and return a pointer string.

        Lazily creates the agent's :class:`~exo.context.workspace.Workspace`
        on first use.  Falls back to returning the content unchanged when
        ``exo-context`` is not installed.

        Args:
            tool_name: Name of the tool that produced the result (used in the artifact ID).
            content: The full tool result string to offload.

        Returns:
            A pointer string referencing the stored artifact, or the original
            *content* when the workspace is unavailable.
        """
        artifact_id = f"tool_result_{tool_name}_{uuid.uuid4().hex[:8]}"

        # Lazy-create workspace when first needed (shares the same instance and
        # KnowledgeStore wiring as the opt-in knowledge tools).
        if self._workspace is None and self._ensure_workspace() is None:
            _log.debug(
                "ToolResultOffloader: exo-context not installed, skipping offload for %s",
                tool_name,
            )
            return content

        await self._workspace.write(artifact_id, content)
        _log.debug(
            "ToolResultOffloader: offloading %s result size=%d bytes artifact_id=%s",
            tool_name,
            len(content),
            artifact_id,
        )
        return (
            f"[Result stored as artifact '{artifact_id}'. "
            f"Call retrieve_artifact('{artifact_id}') to access.]"
        )

    def _make_spawn_self_tool(self) -> Tool:
        """Create the ``spawn_self`` FunctionTool closure for this agent.

        The returned tool captures ``self`` as *parent* so it can access
        ``_current_provider``, ``_spawn_depth``, ``max_spawn_depth``,
        ``max_spawn_children``, and the agent configuration needed to create
        child agents.
        """
        parent = self

        async def spawn_self(tasks: list[str]) -> str:
            """Spawn copies of the current agent to handle parallel sub-tasks.

            Creates one new agent per task, all running concurrently.  Each
            child gets the same model, instructions, and tools (but fresh
            short-term memory) and shares the parent's long-term memory store
            so knowledge accumulates across spawns.

            Args:
                tasks: List of sub-task prompts, one per child agent to spawn.

            Returns:
                The text results of the spawned agents' runs, or a structured
                error with recovery hint if spawning fails.
            """
            try:
                if not tasks:
                    return tool_error(
                        "Empty tasks list",
                        hint=(
                            "Provide at least one task string in the tasks "
                            "list. Each task should describe a sub-problem "
                            "to solve in parallel."
                        ),
                    )

                if len(tasks) > parent.max_spawn_children:
                    return tool_error(
                        f"Too many tasks ({len(tasks)})",
                        hint=(
                            f"Reduce the tasks list to "
                            f"{parent.max_spawn_children} or fewer items. "
                            f"Split into multiple spawn_self calls if needed."
                        ),
                        max_children=parent.max_spawn_children,
                    )

                if parent._spawn_depth >= parent.max_spawn_depth:
                    return tool_error(
                        f"Maximum spawn depth ({parent.max_spawn_depth}) reached",
                        hint=(
                            "Cannot spawn further sub-agents. Handle the "
                            "remaining tasks directly without spawning."
                        ),
                    )

                provider = parent._current_provider
                if provider is None:
                    return tool_error(
                        "No provider available for spawned agent",
                        hint=(
                            "The agent has no active provider. Handle the "
                            "tasks directly without spawning."
                        ),
                    )

                # Build tools list once — exclude spawn_self, context tools,
                # and the parent's PTC tool (which is bound to the parent
                # agent instance).  If parent has ``ptc=True`` the child
                # agent will re-register its own PTCTool via its ``ptc``
                # init flag below, so PTC-eligible tools are absorbed into
                # the child's ``__exo_ptc__`` instead of leaking as direct
                # schemas on the child.
                child_tools = [
                    t
                    for name, t in parent.tools.items()
                    if name not in _SUBAGENT_TOOL_NAMES
                    and not getattr(t, "_is_context_tool", False)
                    and not getattr(t, "_is_ptc_tool", False)
                ]

                results: list[str] = [""] * len(tasks)

                async def _run_child(idx: int) -> None:
                    try:
                        task = tasks[idx]
                        child_memory = _build_child_memory(parent)
                        child_name = f"{parent.name}_spawn_{uuid.uuid4().hex[:8]}"
                        child_context = _build_child_context(parent, child_name)

                        child_agent = Agent(
                            name=child_name,
                            model=parent.model,
                            instructions=parent.instructions,
                            tools=child_tools,
                            max_steps=parent.max_steps,
                            temperature=parent.temperature,
                            max_tokens=parent.max_tokens,
                            store=child_memory,
                            context=child_context,
                            subagents=False,
                            # Inherit batch-tools (PTC) settings so the child
                            # re-registers its own batch-tools tool and the
                            # schema filter hides batch-eligible tools from the
                            # child's LLM call.
                            batch_tools=parent.ptc,
                            batch_tools_timeout=parent.ptc_timeout,
                            batch_tools_max_output_bytes=parent.ptc_max_output_bytes,
                            batch_tools_max_tool_calls=parent.ptc_max_tool_calls,
                            batch_tools_extra_args=dict(parent.ptc_extra_args) or None,
                        )
                        child_agent._spawn_depth = parent._spawn_depth + 1

                        _log.info(
                            "spawn_self: parent=%s child=%s depth=%d task_idx=%d/%d task_len=%d",
                            parent.name,
                            child_agent.name,
                            child_agent._spawn_depth,
                            idx + 1,
                            len(tasks),
                            len(task),
                        )

                        result = await child_agent.run(task, provider=provider)
                        results[idx] = result.text or ""
                    except Exception as exc:
                        results[idx] = f"[child {idx + 1} error] {exc}"

                try:
                    async with asyncio.TaskGroup() as tg:
                        for i in range(len(tasks)):
                            tg.create_task(_run_child(i))
                except asyncio.CancelledError:
                    raise  # Cancellation must propagate — never swallow into a tool result
                except BaseException as exc:
                    return tool_error(
                        f"Spawn execution failed: {unwrap_exception_group(exc)}",
                        hint=(
                            "One or more spawned agents failed. Handle the "
                            "tasks directly without spawning."
                        ),
                    )

                if len(tasks) == 1:
                    return results[0]

                parts = []
                for i, result in enumerate(results):
                    parts.append(f"[Task {i + 1}]: {result}")
                return "\n\n".join(parts)
            except Exception as exc:
                return tool_error(
                    f"spawn_self failed: {exc}",
                    hint=(
                        "Handle the tasks directly without spawning. "
                        "Break the work into sequential steps if needed."
                    ),
                )

        return FunctionTool(spawn_self, name="spawn_self")

    # ------------------------------------------------------------------
    # Background (fire-and-forget) sub-agents
    # ------------------------------------------------------------------

    def _ensure_bg_handler(self) -> Any:
        """Lazily build the per-agent background task handler.

        The handler is created on first ``spawn_background`` use and the
        hot-merge callback (which injects completed results) is wired once.
        """
        if self._bg_handler is None:
            from exo._internal.background import BackgroundTaskHandler

            handler = BackgroundTaskHandler(state=None)
            handler.on_merge(self._on_bg_merge)
            self._bg_handler = handler
        return self._bg_handler

    async def _on_bg_merge(self, task: Any, mode: Any) -> None:
        """Hot-merge callback: inject a finished child's result into this run."""
        self.inject_message(self._format_bg_result(task))

    def _format_bg_result(self, task: Any) -> str:
        """Format a completed background task for injection into the parent."""
        from exo._internal.state import RunNodeStatus

        if task.status == RunNodeStatus.FAILED:
            return f"[background subagent {task.task_id} failed] {task.error or 'unknown error'}"
        return f"[background subagent {task.task_id} completed]\n{task.result or ''}"

    def _format_bg_status(self, task: Any, *, brief: bool = False) -> str:
        """Format a background task's live status for check/list tools."""
        from exo._internal.state import RunNodeStatus

        if task.status == RunNodeStatus.SUCCESS:
            head = f"{task.task_id}: completed"
            return head if brief else f"{head}\nresult: {task.result or ''}"
        if task.status == RunNodeStatus.FAILED:
            return f"{task.task_id}: failed ({task.error or 'unknown error'})"

        p = task.progress
        head = f"{task.task_id}: running (step {p.step_number}"
        if p.current_tool:
            head += f", tool={p.current_tool}"
        head += ")"
        if brief or not p.partial_output:
            return head
        return f"{head}\npartial: {p.partial_output[-500:]}"

    async def _run_bg_child(
        self,
        task_id: str,
        task: str,
        child_tools: list[Tool],
        provider: Any,
        handler: Any,
        task_obj: Any,
    ) -> None:
        """Detached coroutine driving one background sub-agent.

        Builds a child agent (same contract as ``spawn_self``), streams it
        for live progress, then routes the final result through the handler
        (HOT merge if the parent is still running, WAKEUP otherwise).
        """
        parent = self
        try:
            from exo.runner import run
            from exo.types import (
                StepEvent,
                TextEvent,
                ToolCallEvent,
                ToolResultEvent,
                UsageEvent,
            )

            child_memory = _build_child_memory(parent)
            child_name = f"{parent.name}_bg_{task_id}"
            child_context = _build_child_context(parent, child_name)
            child_agent = Agent(
                name=child_name,
                model=parent.model,
                instructions=parent.instructions,
                tools=child_tools,
                max_steps=parent.max_steps,
                temperature=parent.temperature,
                max_tokens=parent.max_tokens,
                store=child_memory,
                context=child_context,
                subagents=False,
                batch_tools=parent.ptc,
                batch_tools_timeout=parent.ptc_timeout,
                batch_tools_max_output_bytes=parent.ptc_max_output_bytes,
                batch_tools_max_tool_calls=parent.ptc_max_tool_calls,
                batch_tools_extra_args=dict(parent.ptc_extra_args) or None,
            )
            child_agent._spawn_depth = parent._spawn_depth + 1

            final_text = ""

            async def _drive() -> str:
                nonlocal final_text
                async for ev in run.stream(child_agent, task, provider=provider, detailed=True):
                    if isinstance(ev, StepEvent):
                        task_obj.update_progress(step_number=ev.step_number, step_status=ev.status)
                    elif isinstance(ev, ToolCallEvent):
                        task_obj.update_progress(
                            current_tool=ev.tool_name, current_tool_args=ev.arguments
                        )
                        # Text emitted before a tool call is reasoning, not the
                        # final answer — reset so only post-last-tool text remains.
                        final_text = ""
                    elif isinstance(ev, ToolResultEvent):
                        task_obj.update_progress(
                            last_tool_result=str(ev.result)[:500],
                            last_tool_error=ev.error,
                        )
                    elif isinstance(ev, TextEvent):
                        final_text += ev.text
                        task_obj.append_partial(ev.text)
                    elif isinstance(ev, UsageEvent):
                        task_obj.update_progress(tokens_used=ev.usage.total_tokens)
                return final_text

            if parent._bg_timeout:
                result_text = await asyncio.wait_for(_drive(), timeout=parent._bg_timeout)
            else:
                result_text = await _drive()

            # Snapshot _is_running immediately after the child finishes (before
            # the next await) so the routing decision is race-safe: the parent
            # cannot clear the flag between the snapshot and handle_result.
            _parent_still_running = parent._is_running
            await handler.handle_result(task_id, result_text, is_main_running=_parent_still_running)
        except asyncio.CancelledError:
            handler.handle_error(task_id, "cancelled")
            raise
        except TimeoutError:
            handler.handle_error(task_id, f"timed out after {parent._bg_timeout}s")
        except Exception as exc:
            handler.handle_error(task_id, str(exc))
        finally:
            parent._bg_tasks.pop(task_id, None)

    async def _cancel_background(self) -> None:
        """Cancel and reap every still-running background sub-agent."""
        if not self._bg_tasks:
            return
        items = list(self._bg_tasks.items())
        for _tid, t in items:
            t.cancel()
        await asyncio.gather(*(t for _, t in items), return_exceptions=True)
        # A task cancelled before it reached its first await never runs its
        # own CancelledError handler, so finalize any still-running records.
        if self._bg_handler is not None:
            for tid, _t in items:
                bg = self._bg_handler.get_task(tid)
                if bg is not None and not bg.is_complete:
                    self._bg_handler.handle_error(tid, "cancelled")
        self._bg_tasks.clear()

    async def aclose(self) -> None:
        """Cancel and reap any detached background sub-agents.

        Background sub-agents are intentionally left running across
        ``run()`` calls (fire-and-forget).  Call this for deterministic
        cleanup when you are done with the agent.
        """
        await self._cancel_background()

    def _make_spawn_background_tool(self) -> Tool:
        """Create the ``spawn_background`` FunctionTool closure for this agent."""
        parent = self

        async def spawn_background(task: str) -> str:
            """Spawn a sub-agent that runs in the background (fire-and-forget).

            Returns immediately with a ``task_id`` — the sub-agent runs
            concurrently while you keep working.  When it finishes, its
            result is injected back into your conversation automatically.
            Call ``check_subagent(task_id)`` to see what it is doing at any
            time, or ``list_subagents()`` to see them all.

            Args:
                task: The sub-task prompt for the background agent to work on.

            Returns:
                A ``task_id`` string (e.g. ``"bg_ab12cd34"``), or a structured
                error with a recovery hint if it could not be launched.
            """
            try:
                if not task or not task.strip():
                    return tool_error(
                        "Empty task",
                        hint="Provide a non-empty task describing the sub-problem to run.",
                    )
                if parent._spawn_depth >= parent.max_spawn_depth:
                    return tool_error(
                        f"Maximum spawn depth ({parent.max_spawn_depth}) reached",
                        hint=(
                            "Cannot spawn further sub-agents. Handle this "
                            "task directly without spawning."
                        ),
                    )
                if len(parent._bg_tasks) >= parent._bg_max:
                    return tool_error(
                        f"Too many background sub-agents running ({len(parent._bg_tasks)})",
                        hint=(
                            f"Wait for some to finish (max {parent._bg_max}). "
                            "Use list_subagents() to check their status."
                        ),
                        background_max=parent._bg_max,
                    )
                provider = parent._current_provider
                if provider is None:
                    return tool_error(
                        "No provider available for background sub-agent",
                        hint=(
                            "The agent has no active provider. Handle this "
                            "task directly without spawning."
                        ),
                    )

                # Snapshot the provider into the closure NOW: _current_provider
                # is cleared when run() returns, but this child outlives the run.
                child_tools = [
                    t
                    for name, t in parent.tools.items()
                    if name not in _SUBAGENT_TOOL_NAMES
                    and not getattr(t, "_is_context_tool", False)
                    and not getattr(t, "_is_ptc_tool", False)
                ]

                handler = parent._ensure_bg_handler()
                task_id = f"bg_{uuid.uuid4().hex[:8]}"
                task_obj = handler.submit(task_id, parent.name, payload={"task": task})

                bg_task = asyncio.create_task(
                    parent._run_bg_child(task_id, task, child_tools, provider, handler, task_obj),
                    name=f"bg:{parent.name}:{task_id}",
                )
                parent._bg_tasks[task_id] = bg_task
                _log.info(
                    "spawn_background: parent=%s task_id=%s depth=%d",
                    parent.name,
                    task_id,
                    parent._spawn_depth,
                )
                return task_id
            except Exception as exc:
                return tool_error(
                    f"spawn_background failed: {exc}",
                    hint="Handle the task directly without spawning.",
                )

        return FunctionTool(spawn_background, name="spawn_background")

    def _make_check_subagent_tool(self) -> Tool:
        """Create the ``check_subagent`` FunctionTool closure for this agent."""
        parent = self

        async def check_subagent(task_id: str) -> str:
            """Check the live status of a background sub-agent.

            Reports whether the sub-agent is still running (and which step /
            tool it is currently on, plus partial output) or — once it has
            finished — its result or error.

            Args:
                task_id: The id returned by ``spawn_background``.
            """
            handler = parent._bg_handler
            bg = handler.get_task(task_id) if handler is not None else None
            if bg is None:
                return tool_error(
                    f"No background sub-agent '{task_id}'",
                    hint="Use list_subagents() to see active sub-agents.",
                )
            return parent._format_bg_status(bg)

        return FunctionTool(check_subagent, name="check_subagent")

    def _make_list_subagents_tool(self) -> Tool:
        """Create the ``list_subagents`` FunctionTool closure for this agent."""
        parent = self

        async def list_subagents() -> str:
            """List all background sub-agents spawned this session and their status."""
            handler = parent._bg_handler
            tasks = handler.list_tasks() if handler is not None else []
            if not tasks:
                return "No background sub-agents."
            return "\n".join(parent._format_bg_status(t, brief=True) for t in tasks)

        return FunctionTool(list_subagents, name="list_subagents")

    def _make_activate_skill_tool(self) -> Tool:
        """Create the ``activate_skill`` FunctionTool for lazy skill loading.

        The returned tool captures ``self`` so it can look up skills in the
        registry, resolve their tools, and add them to the agent's toolset.
        """
        agent_ref = self

        async def activate_skill(name: str) -> str:
            """Activate a skill by name, loading its tools and returning instructions.

            Args:
                name: The name of the skill to activate.
            """
            try:
                registry = agent_ref._skill_registry
                if registry is None:
                    return tool_error(
                        "No skill registry configured",
                        hint=(
                            "This agent was not initialized with skills. "
                            "Skills must be provided when creating the Agent."
                        ),
                    )

                try:
                    skill = registry.get(name)
                except SkillError:
                    available = registry.list_names()
                    return tool_error(
                        f"Skill '{name}' not found",
                        hint=(
                            "Choose one of the available skills and call "
                            "activate_skill with that name."
                        ),
                        available_skills=available,
                    )

                # Resolve and add tools (skip duplicates). Invalidate the
                # schema cache so the next LLM call includes the newly
                # activated skill's tools (and, when ``ptc=True``, so the
                # PTCTool description is rebuilt with the new eligible
                # set — a stale cache would hide the new tools entirely).
                if agent_ref._tool_resolver is not None and skill.tool_list:
                    tools = agent_ref._tool_resolver.resolve(skill)
                    async with agent_ref._tools_lock:
                        added = False
                        for t in tools:
                            if t.name not in agent_ref.tools:
                                agent_ref.tools[t.name] = t
                                added = True
                        if added:
                            agent_ref._cached_tool_schemas = None

                return skill.usage or tool_ok(f"Skill '{name}' activated (no usage instructions)")
            except Exception as exc:
                available: list[str] = []
                try:
                    if agent_ref._skill_registry is not None:
                        available = agent_ref._skill_registry.list_names()
                except Exception:
                    pass
                return tool_error(
                    f"Failed to activate skill '{name}': {exc}",
                    hint=(
                        "Retry the activate_skill call. If the error "
                        "persists, continue without this skill."
                    ),
                    available_skills=available,
                )

        return FunctionTool(activate_skill, name="activate_skill")

    # -----------------------------------------------------------------------
    # Context snapshot persistence
    # -----------------------------------------------------------------------

    async def _save_snapshot_if_enabled(
        self,
        conversation_id: str | None,
        msg_list: list[Message],
        output: Any = None,
    ) -> None:
        """Save a context snapshot at end of run if enabled.

        Wrapped in try/except so a snapshot failure never breaks the run.
        """
        if self._memory_persistence is None or conversation_id is None or self.context is None:
            return
        _cfg = getattr(self.context, "config", self.context)
        if not getattr(_cfg, "_enable_snapshots", getattr(_cfg, "enable_snapshots", False)):
            return
        try:
            # Append the final assistant message to snapshot if available.
            snap_list = list(msg_list)
            if output is not None and hasattr(output, "text"):
                snap_list.append(
                    AssistantMessage(content=output.text, tool_calls=output.tool_calls or [])
                )
            await self._memory_persistence.save_snapshot(
                agent_name=self.name,
                conversation_id=conversation_id,
                msg_list=snap_list,
                context_config=_cfg,
            )
        except Exception:
            _log.warning("snapshot save failed", exc_info=True)

    async def clear_snapshot(self, conversation_id: str | None = None) -> bool:
        """Discard the context snapshot, forcing next run to rebuild from raw.

        Args:
            conversation_id: Conversation scope.  Defaults to
                ``self.conversation_id``.

        Returns:
            ``True`` if a snapshot was found and removed, ``False`` otherwise.
        """
        if self._memory_persistence is None:
            return False
        cid = conversation_id or self.conversation_id
        if cid is None:
            return False
        snap = await self._memory_persistence.load_snapshot(self.name, cid)
        if snap is None:
            return False
        from exo.memory.base import MemoryMetadata  # pyright: ignore[reportMissingImports]

        meta = MemoryMetadata(agent_id=self.name, task_id=cid)
        # Clear only snapshot items for this scope.
        items = await self._memory_persistence.store.search(
            metadata=meta,
            memory_type="snapshot",
            limit=10,
        )
        removed = 0
        for item in items:
            # For backends that support soft-delete, use clear with metadata.
            # For ShortTermMemory, removing from the list works.
            try:
                item.transition(
                    __import__("exo.memory.base", fromlist=["MemoryStatus"]).MemoryStatus.DISCARD
                )
                removed += 1
            except Exception:
                pass
        if removed == 0:
            # Fallback: clear all snapshots for this conversation.
            await self._memory_persistence.store.clear(metadata=meta)
        _log.debug("clear_snapshot: agent=%s conversation=%s removed=%d", self.name, cid, removed)
        return removed > 0

    # -----------------------------------------------------------------------
    # Runtime mutation API — asyncio-safe via _tools_lock
    # -----------------------------------------------------------------------

    def inject_message(self, content: str) -> None:
        """Push a user message into the running agent's context.

        Picked up before the next LLM call. Safe to call from any coroutine.

        Args:
            content: The message text to inject.

        Raises:
            ValueError: If *content* is empty.
        """
        if not content:
            raise ValueError("inject_message content must be non-empty")
        self._injected_messages.put_nowait(content)

    def inject_ephemeral(self, content: str | Message) -> None:
        """Queue a message visible to the NEXT LLM call only.

        Unlike :meth:`inject_message`, ephemeral messages are automatically
        removed from the message list after the LLM call completes.  They
        do not persist in history, snapshots, or memory.

        Safe to call from any coroutine (hooks, tools, external code).

        Args:
            content: A string (wrapped as UserMessage) or a Message object.

        Raises:
            ValueError: If *content* is an empty string.
        """
        if isinstance(content, str):
            if not content:
                raise ValueError("inject_ephemeral content must be non-empty")
            self._ephemeral_messages.put_nowait(UserMessage(content=content))
        else:
            self._ephemeral_messages.put_nowait(content)

    async def add_tool(self, tool: Tool) -> None:
        """Append a single tool at runtime, asyncio-safe.

        Uses ``_tools_lock`` to prevent concurrent registrations from
        interfering with each other.

        Args:
            tool: The tool to register.

        Raises:
            AgentError: If a tool with the same name is already registered.
        """
        async with self._tools_lock:
            self._register_tool(tool)

    def remove_tool(self, tool_name: str) -> None:
        """Unregister a tool by name.

        Uses a single dict.pop() for atomic check-and-remove (no gap
        between the existence check and the deletion).

        Args:
            tool_name: Name of the tool to remove.

        Raises:
            AgentError: If no tool with that name is registered.
        """
        _sentinel = object()
        if self.tools.pop(tool_name, _sentinel) is _sentinel:
            raise AgentError(f"Tool '{tool_name}' is not registered on agent '{self.name}'")
        self._cached_tool_schemas = None

    def _apply_ptc_setting(self, enabled: bool) -> None:
        """Toggle ``ptc`` state at runtime, safely.

        Unlike a bare ``self.ptc = True`` assignment, this:
        - Registers the synthetic ``__exo_ptc__`` tool on enable (if not
          already present), bound to *this* agent instance.
        - Removes ``__exo_ptc__`` on disable (if present).
        - Invalidates ``_cached_tool_schemas`` in both cases so the next
          ``get_tool_schemas()`` call re-applies (or drops) the PTC filter.

        Used by :class:`Swarm` when propagating ``ptc`` to member agents,
        and by :meth:`spawn_self` children so they inherit parent PTC
        settings without leaking PTC-eligible tools as direct schemas.
        """
        from exo.ptc import PTC_TOOL_NAME, PTCTool

        if enabled:
            self.ptc = True
            if PTC_TOOL_NAME not in self.tools:
                self.tools[PTC_TOOL_NAME] = PTCTool(
                    agent=self,
                    timeout=self.ptc_timeout,
                    max_output_bytes=self.ptc_max_output_bytes,
                    max_tool_calls=self.ptc_max_tool_calls,
                    extra_args=self.ptc_extra_args or None,
                )
            self._cached_tool_schemas = None
        else:
            self.ptc = False
            if PTC_TOOL_NAME in self.tools and getattr(
                self.tools[PTC_TOOL_NAME], "_is_ptc_tool", False
            ):
                del self.tools[PTC_TOOL_NAME]
            self._cached_tool_schemas = None

    async def _tool_gate_hook(self, **kwargs: Any) -> None:
        """POST_TOOL_CALL hook that unlocks gated tools when a trigger fires.

        Gated tools are appended (never reordered) so the LLM provider's
        KV-cache prefix remains valid.
        """
        tool_name: str = kwargs.get("tool_name", "")
        if tool_name not in self._tool_gate or tool_name in self._unlocked_gates:
            return
        self._unlocked_gates.add(tool_name)
        for t in self._tool_gate[tool_name]:
            if t.name not in self.tools:
                await self.add_tool(t)

    async def add_handoff(self, target: Agent) -> None:
        """Register a target agent as a handoff destination at runtime.

        Args:
            target: The agent to delegate to.

        Raises:
            AgentError: If a handoff with the same name is already registered.
        """
        async with self._tools_lock:
            self._register_handoff(target)

    async def add_mcp_server(self, config: MCPServerConfig) -> None:
        """Connect an MCP server and append its tools to this agent at runtime.

        Requires the ``exo-mcp`` package.  Creates a new
        ``MCPServerConnection``, connects to the server, lists its tools,
        and registers each one via :meth:`add_tool`.

        Args:
            config: An ``MCPServerConfig`` instance describing the server.

        Raises:
            AgentError: If ``exo-mcp`` is not installed or the connection
                fails.
        """
        try:
            from exo.mcp.client import MCPServerConnection  # pyright: ignore[reportMissingImports]
            from exo.mcp.tools import (
                load_tools_from_connection,  # pyright: ignore[reportMissingImports]
            )
        except ImportError as exc:
            raise AgentError("exo-mcp is required for add_mcp_server()") from exc

        try:
            conn = MCPServerConnection(config)
            await conn.connect()
        except Exception as exc:
            raise AgentError(
                f"Failed to connect MCP server '{getattr(config, 'name', config)}': {exc}"
            ) from exc

        mcp_tools = await load_tools_from_connection(conn)
        async with self._tools_lock:
            for tool in mcp_tools:
                self._register_tool(tool)

        _log.info(
            "add_mcp_server: agent=%s server=%s tools_added=%d",
            self.name,
            getattr(config, "name", config),
            len(mcp_tools),
        )

    # -----------------------------------------------------------------------

    def _attach_memory_persistence(self, memory: Any) -> None:
        """Auto-attach MemoryPersistence hooks if exo-memory is installed.

        Handles both ``AgentMemory`` (uses ``short_term`` store) and plain
        ``MemoryStore`` objects.  If the exo-memory package is not
        installed, this is a no-op.
        """
        try:
            from exo.memory.base import (  # pyright: ignore[reportMissingImports]
                AgentMemory,
                MemoryStore,
            )
            from exo.memory.persistence import (  # pyright: ignore[reportMissingImports]
                MemoryPersistence,
            )
        except ImportError:
            return

        if isinstance(memory, AgentMemory):
            persistence = MemoryPersistence(memory.short_term)
            persistence.attach(self)
            self._memory_persistence = persistence
        elif isinstance(memory, MemoryStore):
            persistence = MemoryPersistence(memory)
            persistence.attach(self)
            self._memory_persistence = persistence

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        """Return OpenAI-format tool schemas for all registered tools.

        When ``ptc=True``, PTC-eligible tools are excluded from the schema
        list — they are available as functions inside the PTC tool
        instead.  Returns cached schemas when available; rebuilds after
        tool mutations.

        When ``injected_tool_args`` is configured, each schema is deep-copied
        and augmented with the injected fields as optional string properties.
        The underlying ``Tool.parameters`` object is never mutated.
        """
        if self._cached_tool_schemas is None:
            if self.ptc:
                from exo.ptc import get_ptc_eligible_tools

                # Single pass: tools NOT in the PTC-eligible set go to direct schemas.
                ptc_eligible = get_ptc_eligible_tools(self)
                schemas = [
                    t.to_schema() for name, t in self.tools.items() if name not in ptc_eligible
                ]
            else:
                schemas = [t.to_schema() for t in self.tools.values()]
            if self.injected_tool_args:
                schemas = [self._augment_schema(s) for s in schemas]
            self._cached_tool_schemas = schemas
        return self._cached_tool_schemas

    def _augment_schema(self, schema: dict[str, Any]) -> dict[str, Any]:
        """Deep-copy *schema* and merge ``injected_tool_args`` as optional properties."""
        import copy

        schema = copy.deepcopy(schema)
        params = schema.get("function", {}).get("parameters")
        if params is None:
            return schema
        props = params.setdefault("properties", {})
        for arg_name, description in self.injected_tool_args.items():
            if arg_name not in props:
                props[arg_name] = {"type": "string", "description": description}
        return schema

    def describe(self) -> dict[str, Any]:
        """Return a summary of the agent's capabilities.

        Useful for debugging, logging, and capability advertisement
        in multi-agent systems.

        Returns:
            A dict with the agent's name, model, tools, and configuration.
        """
        info = {
            "name": self.name,
            "model": self.model,
            "tools": list(self.tools.keys()),
            "handoffs": list(self.handoffs.keys()),
            "max_steps": self.max_steps,
            "output_type": (self.output_type.__name__ if self.output_type else None),
            "planning_enabled": self.planning_enabled,
            "budget_awareness": self.budget_awareness,
            "emit_mcp_progress": self.emit_mcp_progress,
            "ptc": self.ptc,
        }
        if self._skill_registry is not None:
            info["skills"] = self._skill_registry.list_names()
        return info

    async def run(
        self,
        input: MessageContent,
        *,
        messages: Sequence[Message] | None = None,
        provider: Any = None,
        max_retries: int = 3,
        conversation_id: str | None = None,
    ) -> AgentOutput:
        """Deprecated. Use the top-level ``run(agent, ...)`` instead.

        ``run(agent, "...")`` (from ``exo``) is now the single, recommended
        way to execute an agent.  It returns a :class:`RunResult` with the
        full message history, aggregated usage, and step count — whereas this
        method returns only the final :class:`AgentOutput`.

        This shim is kept for backward compatibility and delegates to the
        internal :meth:`_run` engine.  It emits a :class:`DeprecationWarning`.
        """
        import warnings

        warnings.warn(
            "Agent.run() is deprecated; use the top-level run(agent, ...) "
            "from `exo` instead (it returns a RunResult with history, usage, "
            "and step count).",
            DeprecationWarning,
            stacklevel=2,
        )
        return await self._run(
            input,
            messages=messages,
            provider=provider,
            max_retries=max_retries,
            conversation_id=conversation_id,
        )

    async def _run(
        self,
        input: MessageContent,
        *,
        messages: Sequence[Message] | None = None,
        provider: Any = None,
        max_retries: int = 3,
        conversation_id: str | None = None,
    ) -> AgentOutput:
        """Execute the agent's LLM-tool loop with retry logic.

        Internal engine driving a single agent run.  Builds the message
        list, calls the LLM, and if tool calls are returned, executes them
        in parallel, feeds results back, and re-calls the LLM.  The loop
        continues until a text-only response is produced or ``max_steps``
        is reached.

        Public callers should use the top-level ``run(agent, ...)`` which
        wraps this with state tracking and returns a :class:`RunResult`.

        Args:
            input: User query — a string or list of ContentBlock objects.
            messages: Prior conversation history.
            provider: An object with an ``async complete()`` method
                (e.g. a ``ModelProvider`` instance).
            max_retries: Maximum retry attempts for transient errors.
            conversation_id: Conversation scope override for this call only.
                When omitted, the agent's ``conversation_id`` attribute is
                used (auto-assigned UUID4 on first run if memory is set).

        Returns:
            Parsed ``AgentOutput`` from the final LLM response.

        Raises:
            AgentError: If no provider is supplied or all retries are exhausted.
        """
        if provider is None:
            raise AgentError(f"Agent '{self.name}' requires a provider for run()")

        # Store provider reference so spawn_self tool can access it during execution.
        # Always cleaned up in the finally block below.
        self._current_provider = provider
        self._is_running = True
        try:
            return await self._run_inner(
                input,
                messages=messages,
                provider=provider,
                max_retries=max_retries,
                conversation_id=conversation_id,
            )
        except TaskLoopAbort:
            # An aborted run reaps its detached background children; a normal
            # run intentionally leaves them running (fire-and-forget).
            await self._cancel_background()
            raise
        finally:
            self._current_provider = None
            self._is_running = False

    async def _run_inner(
        self,
        input: str,
        *,
        messages: Sequence[Message] | None = None,
        provider: Any = None,
        max_retries: int = 3,
        conversation_id: str | None = None,
    ) -> AgentOutput:
        """Inner run implementation; called by :meth:`run` after provider setup."""
        # Resolve instructions (may be async callable)
        instructions = await resolve_instructions(self)

        # ---- Skills: inject catalog of available skills ----
        if self._skill_registry is not None:
            active_skills = self._skill_registry.search(active_only=True)
            if active_skills:
                skill_lines = [
                    "\n\n## Available Skills",
                    "You can activate any of these skills using the `activate_skill` tool:",
                ]
                for sk in active_skills:
                    desc = f": {sk.description}" if sk.description else ""
                    skill_lines.append(f"- **{sk.name}**{desc}")
                instructions = (instructions or "") + "\n".join(skill_lines)
        # ---- end Skills ----

        # ---- Memory: load history and persist user input before LLM call ----
        history: list[Message] = list(messages) if messages else []
        _active_conv: str | None = None
        _snapshot_loaded = False
        if self._memory_persistence is not None:
            _active_conv = conversation_id or self.conversation_id
            if _active_conv is None:
                _active_conv = str(uuid.uuid4())
                if conversation_id is None:
                    self.conversation_id = _active_conv
            from exo.memory.base import (  # pyright: ignore[reportMissingImports]
                HumanMemory,
                MemoryMetadata,
            )

            self._memory_persistence.metadata = MemoryMetadata(
                agent_id=self.name,
                task_id=_active_conv,
            )

            # ---- Snapshot load: try to use persisted processed context ----
            _ctx_cfg = getattr(self.context, "config", self.context) if self.context else None
            if (
                _ctx_cfg is not None
                and getattr(
                    _ctx_cfg, "_enable_snapshots", getattr(_ctx_cfg, "enable_snapshots", False)
                )
                and not messages  # external messages invalidate snapshot
            ):
                try:
                    _snap = await self._memory_persistence.load_snapshot(
                        agent_name=self.name,
                        conversation_id=_active_conv,
                    )
                    if _snap is not None and await self._memory_persistence.is_snapshot_fresh(
                        _snap, self.name, _active_conv, context_config=_ctx_cfg
                    ):
                        from exo.memory.snapshot import (  # pyright: ignore[reportMissingImports]
                            deserialize_msg_list,
                        )

                        history = deserialize_msg_list(_snap.content)
                        _snapshot_loaded = True
                        _log.debug(
                            "snapshot loaded: agent=%s conversation=%s",
                            self.name,
                            _active_conv,
                        )
                except Exception:
                    _log.warning(
                        "snapshot load failed, falling back to raw history",
                        exc_info=True,
                    )
            # ---- end Snapshot load ----

            if not _snapshot_loaded:
                _db_history = await self._memory_persistence.load_history(
                    agent_name=self.name,
                    conversation_id=_active_conv,
                    rounds=self.max_steps,
                )
                history = list(_db_history) + history

            # Always persist the user input.
            await self._memory_persistence.store.add(
                HumanMemory(
                    content=input,
                    metadata=self._memory_persistence.metadata,
                )
            )
            _log.debug(
                "memory pre-run: agent=%s conversation=%s snapshot=%s",
                self.name,
                _active_conv,
                _snapshot_loaded,
            )
        # ---- end Memory ----

        # Build initial message list
        history.append(UserMessage(content=input))
        msg_list = build_messages(instructions, history)

        # ---- Token tracking: look up context window (needed by windowing hooks) ----
        _context_window_tokens = _get_context_window_tokens(self.model_name)

        # ---- Context: apply windowing and summarization ----
        # Skip initial windowing when loaded from snapshot — it IS the
        # already-windowed state.  Mid-run budget triggers still fire.
        if self.context is not None and not _snapshot_loaded:
            msg_list, _ = await _apply_context_windowing(
                msg_list,
                self.context,
                provider,
                hook_manager=self.hook_manager,
                agent=self,
                step=-1,
                max_steps=self.max_steps,
                agent_name=self.name,
                model_name=self.model_name,
                context_window_tokens=_context_window_tokens,
            )
        # ---- end Context ----

        # ---- Long-term memory: inject relevant knowledge into system message ----
        if self.memory is not None:
            msg_list = await _inject_long_term_knowledge(self.memory, input, msg_list)
        # ---- end Long-term memory ----

        # ---- Token tracking: init per-run tracker ----
        _token_tracker: Any = None
        if self.context is not None:
            try:
                from exo.context.token_tracker import (
                    TokenTracker,  # pyright: ignore[reportMissingImports]
                )

                _token_tracker = TokenTracker()
            except ImportError:
                pass
        # ---- end Token tracking init ----

        # ---- Background sub-agents: flush results that completed while idle ----
        # Children that finished after a previous run ended were queued
        # (WAKEUP); inject them now so they surface on this run's first call.
        if self._bg_handler is not None and not self._bg_handler.pending_queue.empty:
            async for _bg in self._bg_handler.drain_pending():
                self.inject_message(self._format_bg_result(_bg))

        # Tool loop — iterate up to max_steps
        for _step in range(self.max_steps):
            # Re-enumerate tool schemas each step so dynamically added/removed
            # tools (via add_tool/remove_tool) take effect without restarting.
            tool_schemas = self.get_tool_schemas() or None

            # Augment system message with token context info from previous step
            if _token_tracker is not None and _context_window_tokens:
                _trajectory = _token_tracker.get_trajectory(self.name)
                if _trajectory:
                    _last_input = _trajectory[-1].prompt_tokens
                    msg_list = _update_system_token_info(
                        msg_list, _last_input, _context_window_tokens
                    )

            # ---- Drain TaskLoopQueue (ABORT / STEER / FOLLOWUP events) ----
            if self.task_loop_queue:
                _drain_task_loop_queue(self.task_loop_queue, msg_list)

            # ---- Drain injected messages ----
            drain_injected_messages(self, msg_list)

            # ---- Drain ephemeral messages (visible for this call only) ----
            # Ephemerals are collected into a separate batch and concatenated
            # for the LLM call.  msg_list itself is never mutated, keeping the
            # message history append-only (preserves KV-cache prefix).
            _ephemeral_batch = drain_ephemeral_messages(self)
            _call_messages = msg_list + _ephemeral_batch if _ephemeral_batch else msg_list
            output = await self._call_llm(_call_messages, tool_schemas, provider, max_retries)

            # Record token usage in tracker
            if _token_tracker is not None and output.usage.total_tokens > 0:
                _token_tracker.add_usage(self.name, output.usage)

            # Normalize ``default_api.<name>`` tool calls that some models
            # emit directly when they misread the PTC description.  This
            # rewrites the name to the bare tool name so dispatch succeeds
            # and the clean name flows through downstream events.
            if output.tool_calls:
                from exo.ptc import normalize_default_api_tool_calls

                normalize_default_api_tool_calls(output.tool_calls, self)

            # No tool calls — save snapshot and return the final text response
            if not output.tool_calls:
                await self._save_snapshot_if_enabled(
                    _active_conv,
                    msg_list,
                    output,
                )
                return output

            # Execute tool calls and collect results
            try:
                actions = parse_tool_arguments(output.tool_calls)
            except OutputParseError as exc:
                _log.warning("Failed to parse tool arguments on '%s': %s", self.name, exc)
                tool_results = [
                    ToolResult(
                        tool_call_id=tc.id,
                        tool_name=tc.name,
                        error=f"Tool '{tc.name}' error: invalid arguments: {exc}",
                    )
                    for tc in output.tool_calls
                ]
            else:
                tool_results = await self._execute_tools(actions)

            # Drain PTC/ToolContext events — non-streaming path has no consumer.
            # Without this, events accumulate (memory leak) and may leak into
            # a subsequent run.stream() call on the same agent instance.
            while not self._event_queue.empty():
                try:
                    self._event_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break

            # Append assistant message (with tool calls) and results to history
            msg_list.append(AssistantMessage(content=output.text, tool_calls=output.tool_calls))
            msg_list.extend(tool_results)

            # Apply context windowing every step (CONTEXT_WINDOW hook fires each turn).
            # Token budget check sets force_summarize for aggressive compression.
            if self.context is not None:
                _force_summarize, _fill_ratio, _trigger = check_token_budget_pressure(
                    output.usage.input_tokens if _token_tracker is not None else 0,
                    _context_window_tokens,
                    self.context,
                )
                if _force_summarize:
                    _log.info(
                        "token budget trigger: %.0f%% full (%d/%d tokens), forcing context reduction on '%s'",
                        100.0 * _fill_ratio,
                        output.usage.input_tokens,
                        _context_window_tokens,
                        self.name,
                    )
                msg_list, _ = await _apply_context_windowing(
                    msg_list,
                    self.context,
                    provider,
                    force_summarize=_force_summarize,
                    hook_manager=self.hook_manager,
                    agent=self,
                    step=_step,
                    max_steps=self.max_steps,
                    agent_name=self.name,
                    model_name=self.model_name,
                    context_window_tokens=_context_window_tokens,
                    last_usage=output.usage,
                    token_tracker=_token_tracker,
                )

        # max_steps exhausted — save snapshot and return last output as-is
        await self._save_snapshot_if_enabled(_active_conv, msg_list, output)
        return output

    async def branch(self, from_message_id: str) -> str:
        """Branch the conversation at *from_message_id*.

        Creates a new conversation that inherits all messages up to and
        including the message identified by *from_message_id*.  The branch is
        independent of the parent — activity in the branch does not affect the
        original conversation.

        ``Context.fork()`` is used internally to create an isolated child
        context so summarisation is tracked per-branch and not shared with the
        parent.

        Args:
            from_message_id: The ``MemoryItem.id`` of the last message to
                include in the branch.  All messages up to and including this
                item are copied to the new conversation scope.

        Returns:
            A new conversation_id (UUID4 string) for the branched conversation.
            Pass it to ``agent.run(input, conversation_id=branch_id)`` to
            continue on the branch.

        Raises:
            AgentError: If memory is not configured, no active conversation
                exists, or *from_message_id* is not found in the current
                conversation.
        """
        if self._memory_persistence is None:
            raise AgentError(f"Agent '{self.name}' requires memory to be set for branch()")
        if self.conversation_id is None:
            raise AgentError(
                f"Agent '{self.name}' has no active conversation; "
                "run the agent at least once before calling branch()"
            )

        store = self._memory_persistence.store

        # Collect all raw items for the current conversation.
        # Access _items directly (for ShortTermMemory) to bypass windowing and
        # incomplete-pair filtering — we want the full unfiltered history.
        raw_items: list[Any] = []
        store_internal = getattr(store, "_items", None)
        if store_internal is not None:
            for item in store_internal:
                item_meta = item.metadata
                if (
                    getattr(item_meta, "agent_id", None) == self.name
                    and getattr(item_meta, "task_id", None) == self.conversation_id
                ):
                    raw_items.append(item)
        else:
            try:
                from exo.memory.base import MemoryMetadata  # pyright: ignore[reportMissingImports]
            except ImportError as exc:
                raise AgentError("exo-memory is required for branch()") from exc
            _meta_filter = MemoryMetadata(agent_id=self.name, task_id=self.conversation_id)
            raw_items = await store.search(metadata=_meta_filter, limit=10000)
            raw_items = sorted(raw_items, key=lambda x: x.created_at)

        # Find the cutoff index
        cutoff_idx: int | None = None
        for i, item in enumerate(raw_items):
            if item.id == from_message_id:
                cutoff_idx = i
                break

        if cutoff_idx is None:
            raise AgentError(
                f"Message ID {from_message_id!r} not found in conversation "
                f"{self.conversation_id!r} on agent '{self.name}'"
            )

        # Create new conversation_id for the branch
        branch_conv_id = str(uuid.uuid4())

        # Copy messages up to and including the cutoff to the new conversation scope
        try:
            from exo.memory.base import MemoryMetadata  # pyright: ignore[reportMissingImports]
        except ImportError as exc:
            raise AgentError("exo-memory is required for branch()") from exc

        branch_meta = MemoryMetadata(agent_id=self.name, task_id=branch_conv_id)
        # Exclude snapshot items from branch — the branch rebuilds its own.
        items_to_copy = [
            item for item in raw_items[: cutoff_idx + 1] if item.memory_type != "snapshot"
        ]
        for item in items_to_copy:
            copied = item.model_copy(update={"id": uuid.uuid4().hex, "metadata": branch_meta})
            await store.add(copied)

        # Use Context.fork() to create an isolated child context so that
        # summarisation is tracked per-branch and not shared with the parent.
        if self.context is not None:
            try:
                from exo.context.context import Context  # pyright: ignore[reportMissingImports]

                _parent_ctx = Context(task_id=self.conversation_id, config=self.context)
                _parent_ctx.fork(branch_conv_id)
            except ImportError:
                pass

        _log.info(
            "branched conversation: agent=%s parent=%s branch=%s messages_copied=%d at_id=%s",
            self.name,
            self.conversation_id,
            branch_conv_id,
            len(items_to_copy),
            from_message_id,
        )
        return branch_conv_id

    async def _call_llm(
        self,
        msg_list: list[Message],
        tool_schemas: list[dict[str, Any]] | None,
        provider: Any,
        max_retries: int,
    ) -> AgentOutput:
        """Single LLM call with retry logic and lifecycle hooks."""
        last_error: Exception | None = None
        for attempt in range(max_retries):
            try:
                await self.hook_manager.run(HookPoint.PRE_LLM_CALL, agent=self, messages=msg_list)

                response = await provider.complete(
                    msg_list,
                    tools=tool_schemas,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )

                await self.hook_manager.run(HookPoint.POST_LLM_CALL, agent=self, response=response)

                return parse_response(
                    content=response.content,
                    tool_calls=response.tool_calls,
                    usage=response.usage,
                )

            except GuardAbortError:
                raise

            except Exception as exc:
                # GuardrailError (from exo-guardrail) is a deliberate security
                # block, not a transient failure — never retry it.
                if hasattr(exc, "risk_level"):
                    raise
                if _is_context_length_error(exc):
                    _log.error("Context length exceeded on '%s'", self.name)
                    raise AgentError(
                        f"Context length exceeded on agent '{self.name}': {exc}"
                    ) from exc

                # Auth errors (HTTP 401 / AuthenticationError) are non-retryable —
                # retrying will never fix a bad API key. Surface immediately.
                _status = getattr(exc, "status_code", None) or getattr(
                    getattr(exc, "response", None), "status_code", None
                )
                if _status == 401 or "Authentication" in type(exc).__name__:
                    raise AgentError(
                        f"Authentication failed for agent '{self.name}': {exc}\n"
                        "Check your API key / environment variable."
                    ) from exc

                last_error = exc
                if attempt < max_retries - 1:
                    _log.warning(
                        "Retry %d/%d for '%s': %s", attempt + 1, max_retries, self.name, exc
                    )
                    delay = 2**attempt
                    await asyncio.sleep(delay)

        _log.error("Agent '%s' failed after %d retries", self.name, max_retries)
        raise AgentError(
            f"Agent '{self.name}' failed after {max_retries} retries: {last_error}"
        ) from last_error

    async def _execute_tools(
        self,
        actions: list[Any],
    ) -> list[ToolResult]:
        """Execute tool calls in parallel, catching errors per-tool."""
        results: list[ToolResult] = [
            ToolResult(tool_call_id="", tool_name="") for _ in range(len(actions))
        ]

        def _tool_error(tool_name: str, tool_call_id: str, error: str) -> ToolResult:
            """Build a consistently-formatted error ToolResult."""
            return ToolResult(
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                error=f"Tool '{tool_name}' error: {error}",
            )

        async def _run_one(idx: int) -> None:
            action = actions[idx]
            result: ToolResult
            try:
                tool = self.tools.get(action.tool_name)

                # PRE_TOOL_CALL hook
                await self.hook_manager.run(
                    HookPoint.PRE_TOOL_CALL,
                    agent=self,
                    tool_name=action.tool_name,
                    arguments=action.arguments,
                )

                if tool is None:
                    result = _tool_error(
                        action.tool_name,
                        action.tool_call_id,
                        f"unknown tool '{action.tool_name}'",
                    )
                else:
                    try:
                        kwargs = dict(action.arguments)
                        # Strip injected_tool_args — schema-only fields the LLM
                        # fills in but the tool must never receive.
                        if self.injected_tool_args:
                            for key in self.injected_tool_args:
                                kwargs.pop(key, None)
                        # Inject ToolContext if the tool declares one
                        if isinstance(tool, FunctionTool) and tool._tool_context_param:
                            kwargs[tool._tool_context_param] = ToolContext(
                                agent_name=self.name,
                                queue=self._event_queue,
                                human_input_handler=self._human_input_handler,
                            )
                        output = await tool.execute(**kwargs)
                        content: MessageContent
                        if isinstance(output, list):
                            content = output  # list[ContentBlock] from tool
                        elif isinstance(output, str):
                            content = output
                        else:
                            content = (
                                json.dumps(output) if isinstance(output, dict) else str(output)
                            )
                        # Large-output offloading: store in workspace and inject pointer.
                        # Only fires for tools with explicit large_output=True and when
                        # the agent's context mode supports workspace offloading.
                        if (
                            isinstance(content, str)
                            and getattr(tool, "large_output", False)
                            and self._should_enable_artifact_offloading()
                        ):
                            content = await self._offload_large_result(action.tool_name, content)
                        result = ToolResult(
                            tool_call_id=action.tool_call_id,
                            tool_name=action.tool_name,
                            content=content,
                        )
                    except Exception as exc:
                        _log.warning(
                            "Tool '%s' failed on '%s': %s",
                            action.tool_name,
                            self.name,
                            exc,
                        )
                        result = _tool_error(action.tool_name, action.tool_call_id, str(exc))

                # POST_TOOL_CALL hook
                await self.hook_manager.run(
                    HookPoint.POST_TOOL_CALL,
                    agent=self,
                    tool_name=action.tool_name,
                    result=result,
                )
            except GuardAbortError:
                raise  # Security blocks must propagate — never swallow
            except asyncio.CancelledError:
                raise  # Cancellation must propagate — never convert it to a tool result
            except BaseException as exc:
                _log.warning("Tool '%s' failed on '%s': %s", action.tool_name, self.name, exc)
                result = _tool_error(action.tool_name, action.tool_call_id, str(exc))

            results[idx] = result

        try:
            async with asyncio.TaskGroup() as tg:
                for i in range(len(actions)):
                    tg.create_task(_run_one(i))
        except* GuardAbortError as eg:
            # Re-raise the first GuardAbortError directly so security blocks
            # propagate as their own type instead of wrapped in ExceptionGroup.
            raise eg.exceptions[0]

        return results

    def to_dict(self) -> dict[str, Any]:
        """Serialize the agent configuration to a dict.

        Tools are serialized as importable dotted paths. Callable instructions,
        hooks, memory, and context cannot be serialized and will raise ValueError.

        Returns:
            A dict suitable for JSON serialization and later reconstruction
            via ``Agent.from_dict()``.

        Raises:
            ValueError: If the agent contains non-serializable components
                (callable instructions, hooks, closure-based tools, memory, context).
        """
        if callable(self.instructions):
            raise ValueError(
                f"Agent '{self.name}' has callable instructions which cannot be serialized. "
                "Use a string instruction instead."
            )
        if self.memory is not None and not self._memory_is_auto:
            raise ValueError(f"Agent '{self.name}' has a memory store which cannot be serialized.")
        if self._has_user_hooks:
            raise ValueError(f"Agent '{self.name}' has hooks which cannot be serialized.")
        if self.context is not None and not self._context_is_auto:
            raise ValueError(
                f"Agent '{self.name}' has a context engine which cannot be serialized."
            )
        if self._skill_registry is not None:
            raise ValueError(
                f"Agent '{self.name}' has a skill registry which cannot be serialized."
            )
        if self._human_input_handler is not None:
            raise ValueError(
                f"Agent '{self.name}' has a human_input_handler which cannot be serialized."
            )

        data: dict[str, Any] = {
            "name": self.name,
            "model": self.model,
            "instructions": self.instructions,
            "max_steps": self.max_steps,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "planning_enabled": self.planning_enabled,
            "planning_model": self.planning_model,
            "planning_instructions": self.planning_instructions,
            "budget_awareness": self.budget_awareness,
            "hitl_tools": list(self.hitl_tools),
            "emit_mcp_progress": self.emit_mcp_progress,
            "injected_tool_args": dict(self.injected_tool_args),
            "allow_self_spawn": self.allow_self_spawn,
            "max_spawn_depth": self.max_spawn_depth,
            "max_spawn_children": self.max_spawn_children,
            "ptc": self.ptc,
            "ptc_timeout": self.ptc_timeout,
            "ptc_max_output_bytes": self.ptc_max_output_bytes,
            "ptc_max_tool_calls": self.ptc_max_tool_calls,
            "ptc_extra_args": dict(self.ptc_extra_args),
            "bare_tools": self.bare_tools,
        }

        # Serialize tools as importable dotted paths.
        # Skip retrieve_artifact (auto-registered), the spawn_self / background
        # sub-agent tools (auto-registered closures), activate_skill, context
        # tools (auto-loaded), and __exo_ptc__ (PTC auto-registered).
        user_tools = [
            t
            for name, t in self.tools.items()
            if name not in _SUBAGENT_TOOL_NAMES
            and name not in ("retrieve_artifact", "activate_skill")
            and not getattr(t, "_is_context_tool", False)
            and not getattr(t, "_is_ptc_tool", False)
        ]
        if user_tools:
            data["tools"] = [_serialize_tool(t) for t in user_tools]

        # Serialize handoffs recursively
        if self.handoffs:
            data["handoffs"] = [agent.to_dict() for agent in self.handoffs.values()]

        # Serialize output_type as importable dotted path
        if self.output_type is not None:
            data["output_type"] = f"{self.output_type.__module__}.{self.output_type.__qualname__}"

        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Agent:
        """Reconstruct an Agent from a dict produced by ``to_dict()``.

        Tools are resolved by importing dotted paths. Handoff agents are
        reconstructed recursively.

        Args:
            data: Dict as produced by ``Agent.to_dict()``.

        Returns:
            A reconstructed ``Agent`` instance.

        Raises:
            ValueError: If a tool or output_type path cannot be imported.
        """
        tools: list[Tool] | None = None
        if "tools" in data:
            tools = [_deserialize_tool(t) for t in data["tools"]]

        handoffs: list[Agent] | None = None
        if "handoffs" in data:
            handoffs = [Agent.from_dict(h) for h in data["handoffs"]]

        output_type: type[BaseModel] | None = None
        if "output_type" in data:
            output_type = _import_object(data["output_type"])

        return cls(
            name=data["name"],
            model=data.get("model", "openai:gpt-4o"),
            instructions=data.get("instructions", ""),
            tools=tools,
            transfers=handoffs,
            output_type=output_type,
            max_steps=data.get("max_steps", 10),
            temperature=data.get("temperature", 1.0),
            max_tokens=data.get("max_tokens"),
            planning_enabled=data.get("planning_enabled", False),
            planning_model=data.get("planning_model"),
            planning_instructions=data.get("planning_instructions", ""),
            context_pressure=data.get("budget_awareness"),
            approval_tools=data.get("hitl_tools"),
            emit_mcp_progress=data.get("emit_mcp_progress", True),
            injected_tool_args=data.get("injected_tool_args"),
            subagents=data.get("allow_self_spawn", True),
            max_spawn_depth=data.get("max_spawn_depth", 3),
            max_spawn_children=data.get("max_spawn_children", 4),
            batch_tools=data.get("ptc", False),
            batch_tools_timeout=data.get("ptc_timeout", 60),
            batch_tools_max_output_bytes=data.get("ptc_max_output_bytes", 200_000),
            batch_tools_max_tool_calls=data.get("ptc_max_tool_calls", 200),
            batch_tools_extra_args=data.get("ptc_extra_args"),
            bare_tools=data.get("bare_tools", False),
        )

    def __repr__(self) -> str:
        parts = [f"name={self.name!r}", f"model={self.model!r}"]
        if self.tools:
            parts.append(f"tools={list(self.tools.keys())}")
        if self.handoffs:
            parts.append(f"handoffs={list(self.handoffs.keys())}")
        return f"Agent({', '.join(parts)})"


def _is_context_length_error(exc: Exception) -> bool:
    """Check if an exception represents a context-length overflow.

    Detects errors with a ``code`` attribute of ``"context_length"``
    (set by ``ModelError``) or common provider error messages.
    """
    code = getattr(exc, "code", "")
    if code == "context_length":
        return True
    msg = str(exc).lower()
    return "context_length" in msg or "context length" in msg


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------


def _serialize_tool(t: Tool) -> str | dict[str, Any]:
    """Serialize a tool to an importable dotted path or a dict.

    For ``MCPToolWrapper``, returns a dict with an ``__mcp_tool__`` marker.
    For ``FunctionTool``, uses the wrapped function's module and qualname.
    For custom ``Tool`` subclasses, uses the class's module and qualname.

    Raises:
        ValueError: If the tool cannot be serialized (e.g., closures, lambdas).
    """
    # MCPToolWrapper — serialize as a dict with server config
    try:
        from exo.mcp.tools import MCPToolWrapper  # pyright: ignore[reportMissingImports]

        if isinstance(t, MCPToolWrapper):
            mcp_tool: Any = t
            return mcp_tool.to_dict()
    except ImportError:
        pass

    from exo.tool import FunctionTool

    if isinstance(t, FunctionTool):
        fn = t._fn
        module = getattr(fn, "__module__", None)
        qualname = getattr(fn, "__qualname__", None)
        if not module or not qualname:
            raise ValueError(
                f"Tool '{t.name}' wraps a function without __module__ or __qualname__ "
                "and cannot be serialized."
            )
        # Detect closures/lambdas (qualname contains '<')
        if "<" in qualname:
            raise ValueError(
                f"Tool '{t.name}' wraps a closure or lambda ({qualname}) "
                "which cannot be serialized. Use a module-level function instead."
            )
        return f"{module}.{qualname}"

    # Custom Tool subclass — serialize the class itself
    cls = type(t)
    module = cls.__module__
    qualname = cls.__qualname__
    if "<" in qualname:
        raise ValueError(
            f"Tool '{t.name}' is a locally-defined class ({qualname}) which cannot be serialized."
        )
    return f"{module}.{qualname}"


def _deserialize_tool(path: str | dict[str, Any]) -> Tool:
    """Deserialize a tool from an importable dotted path or a dict.

    If ``path`` is a dict with an ``__mcp_tool__`` marker, reconstructs an
    ``MCPToolWrapper`` via ``from_dict()``.

    If the imported object is a callable (function), wraps it as a FunctionTool.
    If it's already a Tool instance, returns it directly.
    If it's a Tool subclass, instantiates it.

    Raises:
        ValueError: If the path cannot be imported or doesn't resolve to a tool.
    """
    if isinstance(path, dict):
        if path.get("__mcp_tool__"):
            from exo.mcp.tools import (  # pyright: ignore[reportMissingImports]
                MCPToolWrapper,
            )

            return MCPToolWrapper.from_dict(path)
        raise ValueError(f"Unknown tool dict format: {path!r}")

    from exo.tool import FunctionTool

    obj = _import_object(path)

    # Already a Tool instance (e.g., @tool decorated at module level)
    if isinstance(obj, Tool):
        return obj

    # A Tool subclass — instantiate it
    if isinstance(obj, type) and issubclass(obj, Tool):
        return obj()

    # A plain callable — wrap it
    if callable(obj):
        return FunctionTool(obj)

    raise ValueError(f"Imported '{path}' is not a callable or Tool instance: {type(obj)}")


def _import_object(dotted_path: str) -> Any:
    """Import an object from a dotted path like 'package.module.ClassName'.

    Tries progressively shorter module paths, resolving the remainder
    via getattr.

    Raises:
        ValueError: If the path cannot be resolved.
    """
    parts = dotted_path.rsplit(".", 1)
    if len(parts) < 2:
        raise ValueError(f"Invalid dotted path: {dotted_path!r}")

    module_path, attr_name = parts
    try:
        module = importlib.import_module(module_path)
        return getattr(module, attr_name)
    except (ImportError, AttributeError):
        pass

    # Try splitting further for nested attributes (e.g., module.Class.method)
    parts = dotted_path.split(".")
    for i in range(len(parts) - 1, 0, -1):
        module_path = ".".join(parts[:i])
        try:
            obj = importlib.import_module(module_path)
            for attr in parts[i:]:
                obj = getattr(obj, attr)
            return obj
        except (ImportError, AttributeError):
            continue

    raise ValueError(f"Cannot import '{dotted_path}'")
