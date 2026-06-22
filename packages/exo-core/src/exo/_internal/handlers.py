"""Handler abstractions for composable agent execution.

Provides ``Handler[IN, OUT]`` as the base abstraction for processing
units that transform inputs to outputs via async generators, and
concrete handlers for agent routing, tool execution, and group
orchestration in multi-agent swarms.
"""

from __future__ import annotations

import abc
import asyncio
import json
from collections import deque
from collections.abc import AsyncGenerator, AsyncIterator, Sequence
from enum import StrEnum
from typing import Any, Generic, TypeVar

from exo._internal.call_runner import call_runner
from exo._internal.state import RunState
from exo.observability.logging import get_logger  # pyright: ignore[reportMissingImports]
from exo.tool import Tool, ToolError
from exo.types import ExoError, Message, RunResult, ToolResult

_log = get_logger(__name__)

IN = TypeVar("IN")
OUT = TypeVar("OUT")


class HandlerError(ExoError):
    """Raised for handler-level errors (routing, dispatch, stop checks)."""


class SwarmMode(StrEnum):
    """Swarm topology modes for agent orchestration."""

    WORKFLOW = "workflow"
    HANDOFF = "handoff"
    TEAM = "team"


class Handler(abc.ABC, Generic[IN, OUT]):
    """Abstract base for composable processing units.

    Handlers receive an input and yield zero or more outputs via
    an async generator.  This enables streaming, backpressure,
    and composable pipelines.
    """

    @abc.abstractmethod
    def handle(self, input: IN, **kwargs: Any) -> AsyncIterator[OUT]:
        """Process input and yield outputs.

        Args:
            input: The input to process.
            **kwargs: Additional context passed through the pipeline.

        Yields:
            Processed output items.
        """
        ...


class AgentHandler(Handler[str, RunResult]):
    """Routes execution between agents in a swarm with topology-aware stops.

    Manages agent dispatch, handoff detection, and stop condition
    checks for workflow, handoff, and team modes.

    Args:
        agents: Dict mapping agent name to agent instance.
        mode: Swarm topology mode.
        flow_order: Ordered list of agent names for workflow mode.
        provider: LLM provider for agent execution.
        max_handoffs: Maximum handoff count before stopping (handoff mode).
    """

    def __init__(
        self,
        *,
        agents: dict[str, Any],
        mode: SwarmMode = SwarmMode.WORKFLOW,
        flow_order: list[str] | None = None,
        provider: Any = None,
        max_handoffs: int = 10,
    ) -> None:
        self.agents = agents
        self.mode = mode
        self.flow_order = flow_order or list(agents.keys())
        self.provider = provider
        self.max_handoffs = max_handoffs

    async def handle(self, input: str, **kwargs: Any) -> AsyncGenerator[RunResult, None]:
        """Execute agents according to the swarm topology.

        For workflow mode, runs agents in flow_order sequentially.
        For handoff mode, starts with the first agent and follows
        handoff chains.  For team mode, runs the lead agent which
        can delegate to workers.

        Args:
            input: User query string.
            **kwargs: Additional context (messages, state, etc.).

        Yields:
            ``RunResult`` from each agent execution.
        """
        messages: list[Message] = list(kwargs.get("messages", []))
        state = RunState(agent_name=self.flow_order[0] if self.flow_order else "")

        if self.mode == SwarmMode.WORKFLOW:
            async for result in self._run_workflow(input, messages, state):
                yield result
        elif self.mode == SwarmMode.HANDOFF:
            async for result in self._run_handoff(input, messages, state):
                yield result
        elif self.mode == SwarmMode.TEAM:
            async for result in self._run_team(input, messages, state):
                yield result

    async def _run_workflow(
        self, input: str, messages: list[Message], state: RunState
    ) -> AsyncGenerator[RunResult, None]:
        """Execute agents sequentially in flow order.

        Output of each agent becomes input for the next.
        """
        _log.debug("Workflow starting: agents=%s", self.flow_order)
        current_input = input
        for agent_name in self.flow_order:
            agent = self.agents.get(agent_name)
            if agent is None:
                raise HandlerError(f"Agent '{agent_name}' not found in swarm")
            result = await call_runner(
                agent, current_input, messages=messages, provider=self.provider
            )
            yield result
            current_input = result.output

    async def _run_handoff(
        self, input: str, messages: list[Message], state: RunState
    ) -> AsyncGenerator[RunResult, None]:
        """Execute agents following handoff chains.

        Starts with the first agent; if the agent's output references
        a handoff target, control transfers to that agent.
        """
        current_agent_name = self.flow_order[0] if self.flow_order else ""
        current_input = input
        handoff_count = 0
        _log.debug("Handoff chain starting from '%s'", current_agent_name)

        while current_agent_name:
            agent = self.agents.get(current_agent_name)
            if agent is None:
                raise HandlerError(f"Agent '{current_agent_name}' not found in swarm")

            result = await call_runner(
                agent, current_input, messages=messages, provider=self.provider
            )
            yield result

            # Check for handoff in the result
            next_agent = self._detect_handoff(agent, result)
            if next_agent is None:
                break

            handoff_count += 1
            if handoff_count >= self.max_handoffs:
                raise HandlerError(f"Max handoffs ({self.max_handoffs}) exceeded in swarm")

            current_agent_name = next_agent
            current_input = result.output

    async def _run_team(
        self, input: str, messages: list[Message], state: RunState
    ) -> AsyncGenerator[RunResult, None]:
        """Execute team mode: lead agent delegates to workers.

        The first agent in flow_order is the lead.  Workers are
        available as the lead's handoff targets.
        """
        if not self.flow_order:
            raise HandlerError("Team mode requires at least one agent")

        lead_name = self.flow_order[0]
        _log.debug("Team mode: lead='%s'", lead_name)
        lead = self.agents.get(lead_name)
        if lead is None:
            raise HandlerError(f"Lead agent '{lead_name}' not found in swarm")

        # Run the lead agent — it can delegate to workers via handoffs
        result = await call_runner(lead, input, messages=messages, provider=self.provider)
        yield result

    def _detect_handoff(self, agent: Any, result: RunResult) -> str | None:
        """Check if the agent's result indicates a handoff.

        Looks for a handoff target name in the result output that
        matches one of the agent's declared handoff targets.

        Args:
            agent: The agent that produced the result.
            result: The run result to check.

        Returns:
            The target agent name, or None if no handoff detected.
        """
        handoffs: dict[str, Any] = getattr(agent, "handoffs", {})
        if not handoffs:
            return None

        output = result.output.strip()
        # Check if the output matches a handoff target name
        for target_name in handoffs:
            if target_name in self.agents and output == target_name:
                return target_name

        return None

    def _check_workflow_stop(self, agent_name: str) -> bool:
        """Check if the workflow should stop after this agent.

        Returns True if agent_name is the last in flow_order.
        """
        if not self.flow_order:
            return True
        return agent_name == self.flow_order[-1]

    def _check_handoff_stop(self, result: RunResult, agent: Any) -> bool:
        """Check if the handoff chain should stop.

        Returns True if no handoff target is detected.
        """
        return self._detect_handoff(agent, result) is None

    def _check_team_stop(self, agent_name: str) -> bool:
        """Check if team execution should stop.

        In team mode, execution stops after the lead agent completes.
        """
        if not self.flow_order:
            return True
        return agent_name == self.flow_order[0]


class ToolHandler(Handler[dict[str, Any], ToolResult]):
    """Handles dynamic tool loading, execution, and result aggregation.

    Accepts a dict of tool arguments keyed by tool call ID, resolves
    tools from a registry, executes them (optionally in parallel),
    and yields ``ToolResult`` objects.

    Args:
        tools: Dict mapping tool name to ``Tool`` instance.
    """

    def __init__(self, *, tools: dict[str, Tool] | None = None) -> None:
        self.tools: dict[str, Tool] = tools or {}

    def register(self, tool: Tool) -> None:
        """Register a tool for execution.

        Args:
            tool: The tool to register.

        Raises:
            HandlerError: If a tool with the same name already exists.
        """
        if tool.name in self.tools:
            raise HandlerError(f"Duplicate tool '{tool.name}' in ToolHandler")
        self.tools[tool.name] = tool

    def register_many(self, tools: Sequence[Tool]) -> None:
        """Register multiple tools at once.

        Args:
            tools: Sequence of tools to register.
        """
        for t in tools:
            self.register(t)

    async def handle(self, input: dict[str, Any], **kwargs: Any) -> AsyncIterator[ToolResult]:
        """Execute tool calls described by the input dict.

        The input dict maps tool_call_id to a dict with ``"name"``
        and ``"arguments"`` keys.  Tools are executed in parallel via
        ``asyncio.TaskGroup``.

        Args:
            input: Mapping of tool_call_id -> {"name": str, "arguments": dict}.
            **kwargs: Ignored.

        Yields:
            ``ToolResult`` for each tool call (in order of tool_call_ids).
        """
        if not input:
            return

        call_ids = list(input.keys())
        results: list[ToolResult] = [
            ToolResult(tool_call_id="", tool_name="") for _ in range(len(call_ids))
        ]

        async def _run_one(idx: int) -> None:
            call_id = call_ids[idx]
            call_info = input[call_id]
            tool_name: str = call_info.get("name", "")
            arguments: dict[str, Any] = call_info.get("arguments", {})

            tool = self.tools.get(tool_name)
            if tool is None:
                results[idx] = ToolResult(
                    tool_call_id=call_id,
                    tool_name=tool_name,
                    error=f"Tool '{tool_name}' error: unknown tool '{tool_name}'",
                )
                return

            try:
                _log.debug("Executing tool '%s' (call_id=%s)", tool_name, call_id)
                output = await tool.execute(**arguments)
                content = (
                    output
                    if isinstance(output, str)
                    else json.dumps(output) if isinstance(output, dict) else str(output)
                )
                results[idx] = ToolResult(
                    tool_call_id=call_id,
                    tool_name=tool_name,
                    content=content,
                )
            except Exception as exc:
                results[idx] = ToolResult(
                    tool_call_id=call_id,
                    tool_name=tool_name,
                    error=f"Tool '{tool_name}' error: {exc}",
                )

        async with asyncio.TaskGroup() as tg:
            for i in range(len(call_ids)):
                tg.create_task(_run_one(i))

        for result in results:
            yield result

    def aggregate(self, results: Sequence[ToolResult]) -> dict[str, str]:
        """Aggregate tool results into a summary dict.

        Args:
            results: Sequence of tool results.

        Returns:
            Mapping of tool_call_id to content or error string.
        """
        out: dict[str, str] = {}
        for r in results:
            out[r.tool_call_id] = r.error if r.error else r.content
        return out


class GroupHandler(Handler[str, RunResult]):
    """Orchestrates parallel and sequential agent/tool group execution.

    Groups can be run in parallel (all at once) or serial (with
    output→input chaining and dependency resolution).

    Args:
        agents: Dict mapping agent name to agent instance.
        provider: LLM provider.
        parallel: If True, run agents concurrently; otherwise serially.
        dependencies: Mapping of agent name to list of agent names it
            depends on (serial mode only).
    """

    def __init__(
        self,
        *,
        agents: dict[str, Any],
        provider: Any = None,
        parallel: bool = True,
        dependencies: dict[str, list[str]] | None = None,
    ) -> None:
        self.agents = agents
        self.provider = provider
        self.parallel = parallel
        self.dependencies = dependencies or {}

    async def handle(self, input: str, **kwargs: Any) -> AsyncGenerator[RunResult, None]:
        """Execute agent group in parallel or serial mode.

        Args:
            input: User query string.
            **kwargs: Additional context (messages, etc.).

        Yields:
            ``RunResult`` from each agent execution.
        """
        messages: list[Message] = list(kwargs.get("messages", []))

        if self.parallel:
            async for result in self._run_parallel(input, messages):
                yield result
        else:
            async for result in self._run_serial(input, messages):
                yield result

    async def _run_parallel(
        self, input: str, messages: list[Message]
    ) -> AsyncGenerator[RunResult, None]:
        """Run all agents concurrently via asyncio.TaskGroup.

        All agents receive the same input.  Results are yielded
        in the order agents are registered.
        """
        agent_names = list(self.agents.keys())
        results: list[RunResult] = [RunResult() for _ in range(len(agent_names))]

        async def _run_one(idx: int) -> None:
            name = agent_names[idx]
            try:
                agent = self.agents[name]
                results[idx] = await call_runner(
                    agent, input, messages=messages, provider=self.provider
                )
            except Exception as exc:
                raise HandlerError(f"Agent '{name}' failed in parallel group: {exc}") from exc

        _log.debug("Running %d agents in parallel: %s", len(agent_names), agent_names)
        try:
            async with asyncio.TaskGroup() as tg:
                for i in range(len(agent_names)):
                    tg.create_task(_run_one(i))
        except* HandlerError as eg:
            msgs = "; ".join(str(e) for e in eg.exceptions)
            raise HandlerError(f"Parallel agent group failed: {msgs}") from eg
        _log.debug("Parallel group completed (%d agents)", len(agent_names))

        for result in results:
            yield result

    async def _run_serial(
        self, input: str, messages: list[Message]
    ) -> AsyncGenerator[RunResult, None]:
        """Run agents in dependency-resolved order with output chaining.

        Uses topological sort on the dependency graph to determine
        execution order.  Each agent receives its predecessor's output
        as input (or the original input if no predecessor).
        """
        order = self._resolve_order()
        outputs: dict[str, str] = {}

        for i, agent_name in enumerate(order):
            _log.debug("Running agent '%s' (serial step %d/%d)", agent_name, i + 1, len(order))
            agent = self.agents.get(agent_name)
            if agent is None:
                raise HandlerError(f"Agent '{agent_name}' not found in group")

            # Determine input: use output of last dependency, or original input
            deps = self.dependencies.get(agent_name, [])
            agent_input = outputs[deps[-1]] if deps else input

            result = await call_runner(
                agent, agent_input, messages=messages, provider=self.provider
            )
            outputs[agent_name] = result.output
            yield result

    def _resolve_order(self) -> list[str]:
        """Topological sort of agents based on dependencies.

        Uses Kahn's algorithm for deterministic ordering.

        Returns:
            Ordered list of agent names.

        Raises:
            HandlerError: If a dependency cycle is detected.
        """
        all_names = list(self.agents.keys())
        if not self.dependencies:
            return all_names

        # Build in-degree counts and adjacency list
        in_degree: dict[str, int] = {name: 0 for name in all_names}
        successors: dict[str, list[str]] = {name: [] for name in all_names}

        for name, deps in self.dependencies.items():
            if name not in in_degree:
                continue
            for dep in deps:
                if dep in successors:
                    successors[dep].append(name)
                    in_degree[name] += 1

        # Kahn's algorithm — use a sorted deque for O(n log n) overall instead
        # of O(n²) from list.sort() + list.pop(0) on each iteration.
        ready = deque(sorted(name for name in all_names if in_degree[name] == 0))
        order: list[str] = []

        while ready:
            current = ready.popleft()
            order.append(current)

            new_ready: list[str] = []
            for succ in successors[current]:
                in_degree[succ] -= 1
                if in_degree[succ] == 0:
                    new_ready.append(succ)
            # Keep deterministic ordering: insert new ready nodes sorted
            for name in sorted(new_ready):
                ready.append(name)

        if len(order) != len(all_names):
            raise HandlerError("Dependency cycle detected in group")

        return order
