"""Swarm: multi-agent orchestration with flow DSL.

A ``Swarm`` groups multiple agents and defines their execution
topology using a simple DSL (``"a >> b >> c"``).  Supports
``mode='workflow'`` (sequential pipeline), ``mode='handoff'``
(agent-driven delegation), and ``mode='team'`` (lead-worker
delegation).

Usage::

    swarm = Swarm(
        agents=[agent_a, agent_b, agent_c],
        flow="a >> b >> c",
    )
    result = await run(swarm, "Hello!")
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import Any, Literal

from exo._internal.call_runner import call_runner
from exo._internal.graph import GraphError, parse_flow_dsl, topological_sort
from exo._internal.loop_node import BREAK_SENTINEL
from exo._internal.workflow_checkpoint import WorkflowCheckpoint, WorkflowCheckpointStore
from exo.observability.logging import get_logger  # pyright: ignore[reportMissingImports]
from exo.tool import Tool, ToolError
from exo.types import ExoError, Message, RunResult, StatusEvent, StreamEvent

_log = get_logger(__name__)

SwarmMode = Literal["workflow", "handoff", "team"]
_VALID_SWARM_MODES: frozenset[str] = frozenset(("workflow", "handoff", "team"))

# Sentinel to distinguish "not provided" from explicit None
_SWARM_CONTEXT_UNSET: object = object()


class SwarmError(ExoError):
    """Raised for swarm-level errors (invalid flow, missing agents, etc.)."""


class Swarm:
    """Multi-agent orchestration container.

    Groups agents and defines their execution topology via a flow DSL.
    In workflow mode, agents run sequentially with output→input chaining.
    In handoff mode, agents delegate dynamically via handoff targets.
    In team mode, the first agent is the lead and others are workers;
    the lead can delegate to workers via auto-generated tools.

    Args:
        agents: List of ``Agent`` instances to include in the swarm.
        flow: Flow DSL string defining execution order
            (e.g., ``"a >> b >> c"``).  If not provided, agents
            run in the order they are given.
        mode: Execution mode — ``"workflow"``, ``"handoff"``, or ``"team"``.
        max_handoffs: Maximum number of handoff transitions before
            raising an error (handoff mode only).
    """

    def __init__(
        self,
        *,
        agents: list[Any],
        flow: str | None = None,
        mode: SwarmMode = "workflow",
        max_handoffs: int = 10,
        context_mode: Any = _SWARM_CONTEXT_UNSET,
        context_limit: int | None = None,
        overflow: str | None = None,
        cache: bool | None = None,
        batch_tools: bool | None = None,
        ptc: bool | None = None,
    ) -> None:
        if not agents:
            raise SwarmError("Swarm requires at least one agent")

        if mode not in _VALID_SWARM_MODES:
            raise SwarmError(
                f"Unknown swarm mode {mode!r}. Valid modes are 'workflow', 'handoff', or 'team'."
            )

        self.mode = mode
        self.max_handoffs = max_handoffs

        # Index agents by name for O(1) lookup
        self.agents: dict[str, Any] = {}
        for agent in agents:
            name = agent.name
            if name in self.agents:
                _log.error("Duplicate agent name '%s' in swarm", name)
                raise SwarmError(f"Duplicate agent name '{name}' in swarm")
            self.agents[name] = agent

        # Resolve execution order from flow DSL or agent list order
        if flow is not None:
            try:
                graph = parse_flow_dsl(flow)
            except GraphError as exc:
                raise SwarmError(f"Invalid flow DSL: {exc}") from exc

            # Validate all flow nodes are known agents
            available = list(self.agents.keys())
            for node_name in graph.nodes:
                if node_name not in self.agents:
                    hint = ""
                    if "->" in node_name or "," in node_name:
                        hint = " (did you mean to use '>>' instead of '->'?)"
                    raise SwarmError(
                        f"Flow references unknown agent '{node_name}'{hint}. "
                        f"Available agents: {available}"
                    )

            try:
                self.flow_order = topological_sort(graph)
            except GraphError as exc:
                _log.error("Cycle detected in flow DSL: %s", exc)
                raise SwarmError(f"Cycle in flow DSL: {exc}") from exc
        else:
            # Default: run in the order agents were provided
            self.flow_order = [a.name for a in agents]

        self.flow = flow

        # Propagate context to all member agents when explicitly provided.
        _has_new_ctx = any(x is not None for x in (context_limit, overflow, cache))
        if _has_new_ctx and context_mode is not _SWARM_CONTEXT_UNSET:
            raise SwarmError(
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

                _kw: dict = {}
                if context_limit is not None:
                    _kw["limit"] = context_limit
                if overflow is not None:
                    _kw["overflow"] = overflow
                if cache is not None:
                    _kw["cache"] = cache
                ctx = _CtxClass(task_id="__default__", config=_CtxConfig(**_kw))
            except ImportError:
                ctx = None
            for agent in self.agents.values():
                agent.context = ctx
                agent._context_is_auto = False
        elif context_mode is not _SWARM_CONTEXT_UNSET:
            from exo.agent import _make_context_from_mode  # avoid circular at module level

            ctx = None if context_mode is None else _make_context_from_mode(context_mode)
            for agent in self.agents.values():
                agent.context = ctx
                agent._context_is_auto = False

        # Propagate batch-tools (PTC) to all member agents when explicitly
        # provided. ``batch_tools=`` is the canonical knob; ``ptc=`` is a
        # deprecated alias. Use ``_apply_ptc_setting`` so the synthetic
        # ``__exo_ptc__`` tool is registered (or removed) and each agent's
        # schema cache is invalidated — a bare ``agent.ptc = ptc`` assignment
        # would leave stale schemas and no PTC tool, causing PTC-eligible
        # tools to leak as direct schemas on the next ``get_tool_schemas()``.
        if batch_tools is not None and ptc is not None:
            raise SwarmError(
                "Cannot combine batch_tools= and ptc= — they are the same "
                "setting. Use batch_tools= (it replaces ptc=)."
            )
        _batch_tools = batch_tools
        if _batch_tools is None and ptc is not None:
            import warnings

            warnings.warn(
                "Swarm(ptc=...) is deprecated; use batch_tools= instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            _batch_tools = ptc
        if _batch_tools is not None:
            for agent in self.agents.values():
                agent._apply_ptc_setting(_batch_tools)

        # Set name from the first agent for compatibility with runner
        self.name = f"swarm({self.flow_order[0]}...)"

    async def run(
        self,
        input: str,
        *,
        messages: Sequence[Message] | None = None,
        provider: Any = None,
        max_retries: int = 3,
        checkpoint: WorkflowCheckpointStore | bool | None = None,
    ) -> RunResult:
        """Execute the swarm according to its mode.

        In workflow mode, agents execute in topological order.
        In handoff mode, the first agent runs and can hand off
        to other agents dynamically.

        Args:
            input: User query string.
            messages: Prior conversation history.
            provider: LLM provider for all agents.
            max_retries: Retry attempts for transient errors.
            checkpoint: When a ``WorkflowCheckpointStore`` is passed,
                checkpoints are saved before each node.  When ``True``,
                an internal store is created (useful for testing).

        Returns:
            ``RunResult`` from the final agent in the chain.

        Raises:
            SwarmError: If mode is unsupported or an agent fails.
        """
        # Resolve checkpoint store
        cp_store: WorkflowCheckpointStore | None = None
        if checkpoint is True:
            cp_store = WorkflowCheckpointStore()
        elif isinstance(checkpoint, WorkflowCheckpointStore):
            cp_store = checkpoint

        if self.mode == "workflow":
            return await self._run_workflow(
                input,
                messages=messages,
                provider=provider,
                max_retries=max_retries,
                checkpoint=cp_store,
            )
        if self.mode == "handoff":
            return await self._run_handoff(
                input, messages=messages, provider=provider, max_retries=max_retries
            )
        if self.mode == "team":
            return await self._run_team(
                input, messages=messages, provider=provider, max_retries=max_retries
            )
        raise SwarmError(f"Unsupported swarm mode: {self.mode!r}")

    async def stream(
        self,
        input: str,
        *,
        messages: Sequence[Message] | None = None,
        provider: Any = None,
        detailed: bool = False,
        max_steps: int | None = None,
        event_types: set[str] | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """Stream swarm execution, yielding events from each sub-agent.

        Each event includes the correct ``agent_name`` of the sub-agent
        that produced it.  ``StatusEvent`` is emitted for agent handoffs
        and delegations.

        Args:
            input: User query string.
            messages: Prior conversation history.
            provider: LLM provider for all agents.
            detailed: When ``True``, emit rich event types.
            max_steps: Maximum LLM-tool round-trips per agent.
            event_types: When provided, only events whose ``type`` field
                matches one of the given strings are yielded.

        Yields:
            ``StreamEvent`` instances from sub-agent execution.
        """
        if self.mode == "workflow":
            async for event in self._stream_workflow(
                input,
                messages=messages,
                provider=provider,
                detailed=detailed,
                max_steps=max_steps,
                event_types=event_types,
            ):
                yield event
        elif self.mode == "handoff":
            async for event in self._stream_handoff(
                input,
                messages=messages,
                provider=provider,
                detailed=detailed,
                max_steps=max_steps,
                event_types=event_types,
            ):
                yield event
        elif self.mode == "team":
            async for event in self._stream_team(
                input,
                messages=messages,
                provider=provider,
                detailed=detailed,
                max_steps=max_steps,
                event_types=event_types,
            ):
                yield event
        else:
            raise SwarmError(f"Unsupported swarm mode: {self.mode!r}")

    async def _stream_workflow(
        self,
        input: str,
        *,
        messages: Sequence[Message] | None = None,
        provider: Any = None,
        detailed: bool = False,
        max_steps: int | None = None,
        event_types: set[str] | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """Stream workflow mode: agents run sequentially, chaining output→input.

        Collects text from ``TextEvent`` objects during streaming to build
        the output for chaining to the next agent — avoiding double execution.
        Handles BranchNode (conditional routing) and LoopNode (iteration).
        """
        from exo.runner import run
        from exo.types import TextEvent

        def _passes_filter(event: StreamEvent) -> bool:
            return event_types is None or event.type in event_types

        current_input = input
        state: dict[str, Any] = {"input": input}
        flow_list = list(self.flow_order)
        idx = 0

        while idx < len(flow_list):
            agent_name = flow_list[idx]
            agent = self.agents[agent_name]

            # --- BranchNode handling ---
            if getattr(agent, "is_branch", False):
                state["input"] = current_input
                target = agent.evaluate(state)
                if target not in self.agents:
                    raise SwarmError(f"Branch '{agent_name}' targets unknown agent '{target}'")
                if detailed:
                    _ev = StatusEvent(
                        status="running",
                        agent_name=agent_name,
                        message=f"Branch '{agent_name}' routing to '{target}'",
                    )
                    if _passes_filter(_ev):
                        yield _ev
                state[agent_name] = target
                if target in flow_list:
                    idx = flow_list.index(target)
                else:
                    # Target not in flow — stream it directly, then stop
                    async for event in self._stream_agent_node(
                        target,
                        current_input,
                        run_fn=run,
                        messages=messages,
                        provider=provider,
                        detailed=detailed,
                        max_steps=max_steps,
                        event_types=event_types,
                        passes_filter=_passes_filter,
                        state=state,
                    ):
                        yield event
                    idx = len(flow_list)
                continue

            # --- LoopNode handling ---
            if getattr(agent, "is_loop", False):
                async for event in self._stream_loop(
                    agent,
                    current_input,
                    run_fn=run,
                    state=state,
                    messages=messages,
                    provider=provider,
                    detailed=detailed,
                    max_steps=max_steps,
                    event_types=event_types,
                    passes_filter=_passes_filter,
                ):
                    yield event
                current_input = state.get(agent_name, current_input)
                idx += 1
                continue

            if detailed:
                _ev = StatusEvent(
                    status="running",
                    agent_name=agent_name,
                    message=f"Workflow executing agent '{agent_name}'",
                )
                if _passes_filter(_ev):
                    yield _ev

            # For groups/nested swarms, delegate to their stream if available
            if getattr(agent, "is_group", False) or getattr(agent, "is_swarm", False):
                if hasattr(agent, "stream"):
                    text_parts: list[str] = []
                    async for event in agent.stream(
                        current_input,
                        messages=messages,
                        provider=provider,
                        detailed=detailed,
                        max_steps=max_steps,
                    ):
                        if isinstance(event, TextEvent):
                            text_parts.append(event.text)
                        if _passes_filter(event):
                            yield event
                    current_input = "".join(text_parts)
                else:
                    result = await agent.run(
                        current_input,
                        messages=messages,
                        provider=provider,
                    )
                    current_input = result.output
                state[agent_name] = current_input
                idx += 1
                continue

            # Stream the agent and collect text for chaining
            text_parts = []
            async for event in run.stream(
                agent,
                current_input,
                messages=messages,
                provider=provider,
                detailed=detailed,
                max_steps=max_steps,
            ):
                if isinstance(event, TextEvent):
                    text_parts.append(event.text)
                if _passes_filter(event):
                    yield event

            current_input = "".join(text_parts)
            state[agent_name] = current_input
            idx += 1

    async def _stream_agent_node(
        self,
        agent_name: str,
        input: str,
        *,
        run_fn: Any,
        messages: Sequence[Message] | None = None,
        provider: Any = None,
        detailed: bool = False,
        max_steps: int | None = None,
        event_types: set[str] | None = None,
        passes_filter: Any = None,
        state: dict[str, Any] | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """Stream a single agent node, collecting text for state tracking."""
        from exo.types import TextEvent

        agent = self.agents[agent_name]
        text_parts: list[str] = []

        if getattr(agent, "is_group", False) or getattr(agent, "is_swarm", False):
            if hasattr(agent, "stream"):
                async for event in agent.stream(
                    input,
                    messages=messages,
                    provider=provider,
                    detailed=detailed,
                    max_steps=max_steps,
                ):
                    if isinstance(event, TextEvent):
                        text_parts.append(event.text)
                    if passes_filter is None or passes_filter(event):
                        yield event
            else:
                result = await agent.run(input, messages=messages, provider=provider)
                text_parts.append(result.output)
        else:
            async for event in run_fn.stream(
                agent,
                input,
                messages=messages,
                provider=provider,
                detailed=detailed,
                max_steps=max_steps,
            ):
                if isinstance(event, TextEvent):
                    text_parts.append(event.text)
                if passes_filter is None or passes_filter(event):
                    yield event

        output = "".join(text_parts)
        if state is not None:
            state[agent_name] = output

    async def _stream_loop(
        self,
        loop_node: Any,
        input: str,
        *,
        run_fn: Any,
        state: dict[str, Any],
        messages: Sequence[Message] | None = None,
        provider: Any = None,
        detailed: bool = False,
        max_steps: int | None = None,
        event_types: set[str] | None = None,
        passes_filter: Any = None,
    ) -> AsyncIterator[StreamEvent]:
        """Stream a LoopNode's body agents through iterations."""
        for body_name in loop_node.body:
            if body_name not in self.agents:
                raise SwarmError(f"Loop '{loop_node.name}' references unknown agent '{body_name}'")

        current_input = input
        iteration = 0
        n_iters = loop_node._resolve_iterations(state)

        while True:
            if iteration >= loop_node.max_iterations:
                break
            if n_iters is not None and iteration >= n_iters:
                break
            if loop_node.condition is not None:
                state["input"] = current_input
                if not loop_node._check_condition(state):
                    break

            broke = False
            for body_name in loop_node.body:
                async for event in self._stream_agent_node(
                    body_name,
                    current_input,
                    run_fn=run_fn,
                    messages=messages,
                    provider=provider,
                    detailed=detailed,
                    max_steps=max_steps,
                    event_types=event_types,
                    passes_filter=passes_filter,
                    state=state,
                ):
                    yield event
                current_input = state.get(body_name, current_input)
                if BREAK_SENTINEL in current_input:
                    broke = True
                    break

            if broke:
                break
            iteration += 1

        state[loop_node.name] = current_input

    async def _stream_handoff(
        self,
        input: str,
        *,
        messages: Sequence[Message] | None = None,
        provider: Any = None,
        detailed: bool = False,
        max_steps: int | None = None,
        event_types: set[str] | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """Stream handoff mode: agents delegate dynamically.

        Collects text from ``TextEvent`` objects during streaming to detect
        handoff targets without double execution.
        """
        from exo.runner import run
        from exo.types import TextEvent

        def _passes_filter(event: StreamEvent) -> bool:
            return event_types is None or event.type in event_types

        current_agent_name = self.flow_order[0]
        current_input = input
        handoff_count = 0

        while True:
            agent = self.agents[current_agent_name]

            if detailed:
                _ev = StatusEvent(
                    status="running",
                    agent_name=current_agent_name,
                    message=f"Handoff executing agent '{current_agent_name}'",
                )
                if _passes_filter(_ev):
                    yield _ev

            # Stream the agent's execution and collect text
            text_parts: list[str] = []
            async for event in run.stream(
                agent,
                current_input,
                messages=messages,
                provider=provider,
                detailed=detailed,
                max_steps=max_steps,
            ):
                if isinstance(event, TextEvent):
                    text_parts.append(event.text)
                if _passes_filter(event):
                    yield event

            output_text = "".join(text_parts)

            # Check for handoff using a lightweight RunResult
            from exo.types import RunResult

            fake_result = RunResult(output=output_text)
            next_agent = self._detect_handoff(agent, fake_result)
            if next_agent is None:
                return

            handoff_count += 1
            if handoff_count > self.max_handoffs:
                raise SwarmError(f"Max handoffs ({self.max_handoffs}) exceeded in swarm")

            if detailed:
                _ev = StatusEvent(
                    status="running",
                    agent_name=next_agent,
                    message=f"Handoff from '{current_agent_name}' to '{next_agent}'",
                )
                if _passes_filter(_ev):
                    yield _ev

            current_agent_name = next_agent
            current_input = output_text

    async def _stream_team(
        self,
        input: str,
        *,
        messages: Sequence[Message] | None = None,
        provider: Any = None,
        detailed: bool = False,
        max_steps: int | None = None,
        event_types: set[str] | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """Stream team mode: lead delegates to workers via tools."""
        from exo.runner import run

        def _passes_filter(event: StreamEvent) -> bool:
            return event_types is None or event.type in event_types

        if len(self.agents) < 2:
            raise SwarmError("Team mode requires at least two agents (lead + workers)")

        lead_name = self.flow_order[0]
        lead = self.agents[lead_name]
        worker_names = [n for n in self.flow_order if n != lead_name]

        # Create delegate tools for each worker and add to lead
        delegate_tools: list[Tool] = []
        for worker_name in worker_names:
            worker = self.agents[worker_name]
            dtool = _DelegateTool(
                worker=worker,
                provider=provider,
                max_retries=3,
            )
            delegate_tools.append(dtool)

        # Temporarily add delegate tools to the lead agent. Invalidate
        # the schema cache so ``get_tool_schemas()`` rebuilds with the
        # delegate tools included — otherwise a previously-populated
        # cache would hide them from the LLM.
        original_tools = dict(lead.tools)
        for dtool in delegate_tools:
            lead.tools[dtool.name] = dtool
        lead._cached_tool_schemas = None

        try:
            if detailed:
                _ev = StatusEvent(
                    status="running",
                    agent_name=lead_name,
                    message=f"Team lead '{lead_name}' starting execution",
                )
                if _passes_filter(_ev):
                    yield _ev

            async for event in run.stream(
                lead,
                input,
                messages=messages,
                provider=provider,
                detailed=detailed,
                max_steps=max_steps,
                event_types=event_types,
            ):
                yield event
        finally:
            # Restore original tools and invalidate cache so the next
            # run does not see the stale delegate schemas.
            lead.tools = original_tools
            lead._cached_tool_schemas = None

    async def _run_workflow(
        self,
        input: str,
        *,
        messages: Sequence[Message] | None = None,
        provider: Any = None,
        max_retries: int = 3,
        checkpoint: WorkflowCheckpointStore | None = None,
        skip_completed: set[str] | None = None,
        restored_state: dict[str, Any] | None = None,
    ) -> RunResult:
        """Execute agents sequentially, chaining output→input.

        Supports regular agents, group nodes (``ParallelGroup``,
        ``ParallelGroup``), ``BranchNode`` (conditional routing), and
        ``LoopNode`` (iteration).  Groups have an ``is_group`` attribute
        and their own ``run()`` method.

        Args:
            input: User query string.
            messages: Prior conversation history.
            provider: LLM provider.
            max_retries: Retry attempts for transient errors.
            checkpoint: Optional checkpoint store.
            skip_completed: Set of agent names to skip (for resume).
            restored_state: State dict from a checkpoint (for resume).

        Returns the ``RunResult`` from the last agent in the flow.
        """
        current_input = input
        last_result: RunResult | None = None
        completed_nodes: list[str] = []
        state: dict[str, Any] = {"input": input}

        # Restore state from checkpoint if resuming
        if restored_state is not None:
            state.update(restored_state)
        if skip_completed:
            completed_nodes = list(skip_completed)
            # Use the last completed agent's output as input for the next
            for name in reversed(self.flow_order):
                if name in skip_completed and name in state:
                    current_input = state[name]
                    break

        # Build an ordered list of flow nodes; we may need to jump
        # forward (branch) or repeat (loop), so use an index.
        flow_list = list(self.flow_order)
        idx = 0
        while idx < len(flow_list):
            agent_name = flow_list[idx]

            if skip_completed and agent_name in skip_completed:
                idx += 1
                continue

            agent = self.agents[agent_name]

            # Save checkpoint before executing this node
            if checkpoint is not None:
                cp = WorkflowCheckpoint(
                    node_name=agent_name,
                    state=dict(state),
                    completed_nodes=list(completed_nodes),
                )
                checkpoint.save(cp)

            # --- BranchNode handling ---
            if getattr(agent, "is_branch", False):
                state["input"] = current_input
                target = agent.evaluate(state)
                if target not in self.agents:
                    raise SwarmError(f"Branch '{agent_name}' targets unknown agent '{target}'")
                _log.debug("Branch '%s' routing to '%s'", agent_name, target)
                completed_nodes.append(agent_name)
                state[agent_name] = target

                # Jump to the target agent in the flow order
                if target in flow_list:
                    idx = flow_list.index(target)
                else:
                    # Target is not in the flow order (e.g. only in agents
                    # dict).  Execute it directly, then stop.
                    last_result = await self._run_agent_node(
                        target,
                        current_input,
                        messages=messages,
                        provider=provider,
                        max_retries=max_retries,
                    )
                    current_input = last_result.output
                    completed_nodes.append(target)
                    state[target] = last_result.output
                    idx = len(flow_list)  # end
                continue

            # --- LoopNode handling ---
            if getattr(agent, "is_loop", False):
                last_result = await self._execute_loop(
                    agent,
                    current_input,
                    state=state,
                    messages=messages,
                    provider=provider,
                    max_retries=max_retries,
                )
                current_input = last_result.output
                completed_nodes.append(agent_name)
                state[agent_name] = last_result.output
                idx += 1
                continue

            # --- Group / SwarmNode handling ---
            if getattr(agent, "is_group", False) or getattr(agent, "is_swarm", False):
                last_result = await agent.run(
                    current_input,
                    messages=messages,
                    provider=provider,
                    max_retries=max_retries,
                )
            else:
                # --- Regular agent ---
                last_result = await call_runner(
                    agent,
                    current_input,
                    messages=messages,
                    provider=provider,
                    max_retries=max_retries,
                )
            current_input = last_result.output
            completed_nodes.append(agent_name)
            state[agent_name] = last_result.output
            idx += 1

        assert last_result is not None  # guaranteed since agents is non-empty
        return last_result

    async def _run_agent_node(
        self,
        agent_name: str,
        input: str,
        *,
        messages: Sequence[Message] | None = None,
        provider: Any = None,
        max_retries: int = 3,
    ) -> RunResult:
        """Execute a single agent node by name (regular, group, or swarm).

        Used internally when routing to an agent outside the flow order.
        """
        agent = self.agents[agent_name]
        if getattr(agent, "is_group", False) or getattr(agent, "is_swarm", False):
            return await agent.run(
                input,
                messages=messages,
                provider=provider,
                max_retries=max_retries,
            )
        return await call_runner(
            agent,
            input,
            messages=messages,
            provider=provider,
            max_retries=max_retries,
        )

    async def _execute_loop(
        self,
        loop_node: Any,
        input: str,
        *,
        state: dict[str, Any],
        messages: Sequence[Message] | None = None,
        provider: Any = None,
        max_retries: int = 3,
    ) -> RunResult:
        """Execute a LoopNode — iterate body agents according to mode.

        Handles count, items, and condition modes, plus break sentinels
        and max_iterations safety.

        Returns a ``RunResult`` with the last body agent's output.
        """
        # Validate body agents exist
        for body_name in loop_node.body:
            if body_name not in self.agents:
                raise SwarmError(f"Loop '{loop_node.name}' references unknown agent '{body_name}'")

        current_input = input
        last_result = RunResult(output=current_input)
        iteration = 0

        # Determine iteration limit for count/items modes
        n_iters = loop_node._resolve_iterations(state)

        while True:
            # Check iteration limits
            if iteration >= loop_node.max_iterations:
                _log.debug(
                    "Loop '%s' hit max_iterations (%d)",
                    loop_node.name,
                    loop_node.max_iterations,
                )
                break

            if n_iters is not None and iteration >= n_iters:
                break

            # For condition mode, check before each iteration
            if loop_node.condition is not None:
                state["input"] = current_input
                if not loop_node._check_condition(state):
                    break

            # Execute all body agents sequentially
            broke = False
            for body_name in loop_node.body:
                last_result = await self._run_agent_node(
                    body_name,
                    current_input,
                    messages=messages,
                    provider=provider,
                    max_retries=max_retries,
                )
                current_input = last_result.output
                state[body_name] = last_result.output

                # Check for break sentinel
                if BREAK_SENTINEL in current_input:
                    broke = True
                    break

            if broke:
                break

            iteration += 1

        return last_result

    async def resume(
        self,
        store: WorkflowCheckpointStore,
        *,
        provider: Any = None,
        max_retries: int = 3,
        checkpoint: WorkflowCheckpointStore | None = None,
    ) -> RunResult:
        """Resume a workflow from the latest checkpoint.

        Skips nodes that were already completed according to the checkpoint
        and continues from the next pending node.

        Args:
            store: Checkpoint store from a previous (failed) run.
            provider: LLM provider for all agents.
            max_retries: Retry attempts for transient errors.
            checkpoint: Optional new checkpoint store for this resumed run.

        Returns:
            ``RunResult`` from the final agent in the chain.

        Raises:
            SwarmError: If mode is not ``"workflow"`` or the store is empty.
        """
        if self.mode != "workflow":
            raise SwarmError("resume() is only supported in workflow mode")

        latest = store.latest()
        if latest is None:
            raise SwarmError("No checkpoints available to resume from")

        skip = set(latest.completed_nodes)
        restored_state = dict(latest.state)

        return await self._run_workflow(
            restored_state.get("input", ""),
            provider=provider,
            max_retries=max_retries,
            checkpoint=checkpoint,
            skip_completed=skip,
            restored_state=restored_state,
        )

    async def _run_handoff(
        self,
        input: str,
        *,
        messages: Sequence[Message] | None = None,
        provider: Any = None,
        max_retries: int = 3,
    ) -> RunResult:
        """Execute agents following handoff chains.

        Starts with the first agent in flow_order.  If an agent's
        output matches a handoff target name (declared on the agent),
        control transfers to that target with the full conversation
        history.  Stops when an agent produces output that is not a
        handoff target, or when ``max_handoffs`` is exceeded.

        Returns the ``RunResult`` from the last agent that ran.
        """
        current_agent_name = self.flow_order[0]
        current_input = input
        all_messages: list[Message] = list(messages) if messages else []
        handoff_count = 0

        while True:
            agent = self.agents[current_agent_name]
            result = await call_runner(
                agent,
                current_input,
                messages=all_messages,
                provider=provider,
                max_retries=max_retries,
            )

            # Accumulate conversation history from this agent's run
            all_messages = list(result.messages)

            # Check for handoff
            next_agent = self._detect_handoff(agent, result)
            if next_agent is None:
                return result

            handoff_count += 1
            if handoff_count > self.max_handoffs:
                raise SwarmError(f"Max handoffs ({self.max_handoffs}) exceeded in swarm")

            current_agent_name = next_agent
            current_input = result.output

    def _detect_handoff(self, agent: Any, result: RunResult) -> str | None:
        """Check if an agent's result indicates a handoff.

        Matches the agent's output (stripped) against its declared
        handoff target names.  The target must also exist in the
        swarm's agents dict.

        Returns:
            Target agent name, or ``None`` if no handoff detected.
        """
        handoffs: dict[str, Any] = getattr(agent, "handoffs", {})
        if not handoffs:
            return None

        output = result.output.strip()
        for target_name in handoffs:
            if target_name in self.agents and output == target_name:
                return target_name

        return None

    async def _run_team(
        self,
        input: str,
        *,
        messages: Sequence[Message] | None = None,
        provider: Any = None,
        max_retries: int = 3,
    ) -> RunResult:
        """Execute team mode: lead delegates to workers via tools.

        The first agent in flow_order is the lead.  Other agents are
        workers.  The lead receives auto-generated ``delegate_to_{name}``
        tools that invoke worker agents.  When the lead calls a delegate
        tool, the worker runs and its output is returned as the tool
        result.  The lead then synthesizes the final output.

        Returns the ``RunResult`` from the lead agent.
        """
        if len(self.agents) < 2:
            raise SwarmError("Team mode requires at least two agents (lead + workers)")

        lead_name = self.flow_order[0]
        lead = self.agents[lead_name]
        worker_names = [n for n in self.flow_order if n != lead_name]

        # Create delegate tools for each worker and add to lead
        delegate_tools: list[Tool] = []
        for worker_name in worker_names:
            worker = self.agents[worker_name]
            dtool = _DelegateTool(
                worker=worker,
                provider=provider,
                max_retries=max_retries,
            )
            delegate_tools.append(dtool)

        # Temporarily add delegate tools to the lead agent. Invalidate
        # the schema cache so the rebuild includes the new delegate
        # tools (a stale cache would hide them from the LLM).
        original_tools = dict(lead.tools)
        for dtool in delegate_tools:
            lead.tools[dtool.name] = dtool
        lead._cached_tool_schemas = None

        try:
            result = await call_runner(
                lead,
                input,
                messages=messages,
                provider=provider,
                max_retries=max_retries,
            )
        finally:
            # Restore original tools and invalidate again so subsequent
            # runs of the lead agent do not see stale delegate schemas.
            lead.tools = original_tools
            lead._cached_tool_schemas = None

        return result

    def to_dict(self) -> dict[str, Any]:
        """Serialize the swarm configuration to a dict.

        All agents are serialized via ``Agent.to_dict()``.

        Returns:
            A dict suitable for JSON serialization and later reconstruction
            via ``Swarm.from_dict()``.
        """
        return {
            "agents": [agent.to_dict() for agent in self.agents.values()],
            "flow": self.flow,
            "mode": self.mode,
            "max_handoffs": self.max_handoffs,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Swarm:
        """Reconstruct a Swarm from a dict produced by ``to_dict()``.

        Args:
            data: Dict as produced by ``Swarm.to_dict()``.

        Returns:
            A reconstructed ``Swarm`` instance.
        """
        from exo.agent import Agent

        agents = [Agent.from_dict(a) for a in data["agents"]]
        return cls(
            agents=agents,
            flow=data.get("flow"),
            mode=data.get("mode", "workflow"),
            max_handoffs=data.get("max_handoffs", 10),
        )

    def to_mermaid(self) -> str:
        """Generate a Mermaid flowchart of this swarm's topology.

        Convenience wrapper around
        :func:`exo._internal.visualization.to_mermaid`.

        Returns:
            A string containing valid ``graph TD`` Mermaid syntax.
        """
        from exo._internal.visualization import to_mermaid

        return to_mermaid(self)

    def describe(self) -> dict[str, Any]:
        """Return a summary of the swarm's configuration.

        Returns:
            Dict with mode, flow order, and agent descriptions.
        """
        return {
            "mode": self.mode,
            "flow": self.flow,
            "flow_order": self.flow_order,
            "agents": {name: agent.describe() for name, agent in self.agents.items()},
        }

    def __repr__(self) -> str:
        return f"Swarm(mode={self.mode!r}, agents={list(self.agents.keys())}, flow={self.flow!r})"


class _DelegateTool(Tool):
    """Auto-generated tool that delegates work to a worker agent.

    When the lead agent calls this tool, the worker agent runs with
    the provided task description and its output is returned as the
    tool result.

    ``_ptc_exclude=True`` keeps delegate tools as *direct* schemas even
    when the lead agent has ``ptc=True`` — routing to another agent is
    not a PTC-eligible orchestration step, it must be a distinct
    tool_call the LLM can see at the top level.
    """

    _ptc_exclude: bool = True

    def __init__(
        self,
        *,
        worker: Any,
        provider: Any = None,
        max_retries: int = 3,
    ) -> None:
        worker_name: str = worker.name
        self.name = f"delegate_to_{worker_name}"
        self.description = f"Delegate a task to the '{worker_name}' worker agent."
        self.parameters = {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": f"The task description to send to '{worker_name}'.",
                },
            },
            "required": ["task"],
        }
        self._worker = worker
        self._provider = provider
        self._max_retries = max_retries

    async def execute(self, **kwargs: Any) -> str:
        """Run the worker agent with the given task.

        Args:
            **kwargs: Must include ``task`` (str).

        Returns:
            The worker agent's output text.
        """
        task: str = kwargs.get("task", "")
        try:
            result = await call_runner(
                self._worker,
                task,
                provider=self._provider,
                max_retries=self._max_retries,
            )
        except Exception as exc:
            raise ToolError(f"Delegation to '{self._worker.name}' failed: {exc}") from exc
        return result.output
