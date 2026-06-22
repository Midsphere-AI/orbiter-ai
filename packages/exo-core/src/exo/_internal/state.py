"""Internal run state tracking for agent execution."""

from __future__ import annotations

import time
from collections.abc import Sequence
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from exo.observability.logging import get_logger  # pyright: ignore[reportMissingImports]
from exo.types import Message, ToolCall, Usage

_log = get_logger(__name__)


class RunNodeStatus(StrEnum):
    """Status of an execution node (agent step or tool call)."""

    INIT = "init"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"


class RunNode(BaseModel):
    """A single execution step within a run.

    Tracks one LLM call or tool execution with timing and status.

    Args:
        agent_name: Name of the agent that owns this step.
        step_index: Zero-based step index within the run.
        status: Current execution status.
        group_id: Optional group identifier for parallel/serial groups.
        created_at: Timestamp when the node was created.
        started_at: Timestamp when execution started.
        ended_at: Timestamp when execution finished.
        tool_calls: Tool calls produced during this step.
        usage: Token usage for this step.
        error: Error message if the step failed.
        metadata: Arbitrary key-value metadata.
    """

    agent_name: str
    step_index: int = 0
    status: RunNodeStatus = RunNodeStatus.INIT
    group_id: str | None = None
    created_at: float = Field(default_factory=time.time)
    started_at: float | None = None
    ended_at: float | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    def start(self) -> None:
        """Transition to RUNNING."""
        self.status = RunNodeStatus.RUNNING
        self.started_at = time.time()
        _log.debug("Node[%d] '%s' → RUNNING", self.step_index, self.agent_name)

    def succeed(self, usage: Usage | None = None) -> None:
        """Transition to SUCCESS with optional usage stats."""
        self.status = RunNodeStatus.SUCCESS
        self.ended_at = time.time()
        if usage is not None:
            self.usage = usage
        _log.debug(
            "Node[%d] '%s' → SUCCESS (duration=%.3fs)",
            self.step_index,
            self.agent_name,
            self.duration or 0,
        )

    def fail(self, error: str) -> None:
        """Transition to FAILED with an error message."""
        self.status = RunNodeStatus.FAILED
        self.ended_at = time.time()
        self.error = error
        _log.warning("Node[%d] '%s' → FAILED: %s", self.step_index, self.agent_name, error)

    def timeout(self) -> None:
        """Transition to TIMEOUT."""
        self.status = RunNodeStatus.TIMEOUT
        self.ended_at = time.time()
        _log.warning("Node[%d] '%s' → TIMEOUT", self.step_index, self.agent_name)

    @property
    def duration(self) -> float | None:
        """Elapsed time in seconds, or None if not yet finished."""
        if self.started_at is not None and self.ended_at is not None:
            return self.ended_at - self.started_at
        return None


class RunState:
    """Mutable execution state for a single run.

    Tracks the full message history, tool calls, iteration count,
    current agent, and per-step nodes.

    Args:
        agent_name: Name of the initial agent.
        max_nodes: Maximum number of ``RunNode`` entries to retain in
            ``self.nodes``.  When the list exceeds this cap the oldest
            entries are dropped to bound memory.  Loop-detection only
            needs the most-recent history, so a generous cap (default
            500) has no practical effect on correctness.
    """

    #: Default cap for the retained node history.
    DEFAULT_MAX_NODES: int = 500

    def __init__(self, agent_name: str, *, max_nodes: int = DEFAULT_MAX_NODES) -> None:
        self.agent_name: str = agent_name
        self.status: RunNodeStatus = RunNodeStatus.INIT
        self.messages: list[Message] = []
        self.nodes: list[RunNode] = []
        self.iterations: int = 0
        self.total_usage: Usage = Usage()
        self._max_nodes = max_nodes

    def start(self) -> None:
        """Mark the run as started."""
        self.status = RunNodeStatus.RUNNING
        _log.debug("Run '%s' started", self.agent_name)

    def add_message(self, message: Message) -> None:
        """Append a message to the run history."""
        self.messages.append(message)

    def add_messages(self, messages: Sequence[Message]) -> None:
        """Append multiple messages to the run history."""
        self.messages.extend(messages)

    def new_node(self, agent_name: str | None = None, group_id: str | None = None) -> RunNode:
        """Create and track a new execution node.

        Args:
            agent_name: Agent name for this node (defaults to run agent).
            group_id: Optional group identifier.

        Returns:
            The newly created RunNode.
        """
        node = RunNode(
            agent_name=agent_name or self.agent_name,
            # Use the global iteration counter for step_index so it stays
            # monotonically increasing even after the rolling window evicts
            # old entries from self.nodes.
            step_index=self.iterations,
            group_id=group_id,
        )
        self.nodes.append(node)
        self.iterations += 1
        # Evict the oldest nodes when the retained history exceeds the cap.
        # Loop-detection only needs recent history; a 500-node window is
        # more than sufficient for any realistic agent run.
        if len(self.nodes) > self._max_nodes:
            excess = len(self.nodes) - self._max_nodes
            del self.nodes[:excess]
            _log.debug(
                "RunState for '%s': evicted %d old nodes (retained=%d)",
                self.agent_name,
                excess,
                len(self.nodes),
            )
        return node

    def record_usage(self, usage: Usage) -> None:
        """Accumulate token usage into the run total."""
        self.total_usage = Usage(
            input_tokens=self.total_usage.input_tokens + usage.input_tokens,
            output_tokens=self.total_usage.output_tokens + usage.output_tokens,
            total_tokens=self.total_usage.total_tokens + usage.total_tokens,
        )

    def succeed(self) -> None:
        """Mark the run as successful."""
        self.status = RunNodeStatus.SUCCESS
        _log.debug(
            "Run '%s' succeeded (%d steps, %d tokens)",
            self.agent_name,
            len(self.nodes),
            self.total_usage.total_tokens,
        )

    def fail(self, error: str | None = None) -> None:
        """Mark the run as failed."""
        self.status = RunNodeStatus.FAILED
        _log.warning("Run '%s' failed: %s", self.agent_name, error or "(no message)")

    def timeout(self) -> None:
        """Mark the run as timed out."""
        self.status = RunNodeStatus.TIMEOUT
        _log.warning("Run '%s' timed out", self.agent_name)

    @property
    def current_node(self) -> RunNode | None:
        """The most recently created node, or None."""
        return self.nodes[-1] if self.nodes else None

    @property
    def is_running(self) -> bool:
        """Whether the run is currently in progress."""
        return self.status == RunNodeStatus.RUNNING

    @property
    def is_terminal(self) -> bool:
        """Whether the run has reached a terminal state."""
        return self.status in (RunNodeStatus.SUCCESS, RunNodeStatus.FAILED, RunNodeStatus.TIMEOUT)
