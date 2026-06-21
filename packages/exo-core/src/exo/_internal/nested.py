"""Support for nested swarms — using a Swarm as a node in another Swarm.

A :class:`SwarmNode` wraps a :class:`~exo.swarm.Swarm` so it can be
placed in another Swarm's agent list.  The outer Swarm treats it like
any other agent node: it has a ``name``, an ``is_swarm`` marker, and a
``run()`` method that delegates to the inner Swarm.

Context isolation is achieved by never sharing mutable state between
the inner and outer swarms — each execution creates its own
``RunState`` and message history.

Usage::

    inner = Swarm(agents=[a, b], flow="a >> b")
    node = SwarmNode(swarm=inner, name="inner_pipeline")
    outer = Swarm(agents=[c, node, d], flow="c >> inner_pipeline >> d")
    result = await run(outer, "Hello!")
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import Any

from exo.types import ExoError, Message, RunResult, StreamEvent


class NestedSwarmError(ExoError):
    """Raised for nested swarm errors."""


class SwarmNode:
    """Wraps a Swarm so it can be used as a node in another Swarm.

    Provides the same interface that :class:`~exo.swarm.Swarm`
    expects from agent nodes — ``name`` attribute and ``run()`` method.
    An ``is_swarm`` marker allows the outer Swarm to detect nested
    swarms via duck-typing.

    Context isolation: the inner Swarm creates its own ``RunState``
    and message history on each ``run()`` invocation.  No mutable
    state leaks between inner and outer executions.

    Args:
        swarm: The inner Swarm to wrap.
        name: Node name for the outer Swarm's flow DSL.
            Defaults to the inner Swarm's ``name`` attribute.
    """

    def __init__(
        self,
        *,
        swarm: Any,
        name: str | None = None,
    ) -> None:
        if not hasattr(swarm, "flow_order"):
            raise NestedSwarmError("SwarmNode requires a Swarm instance (object with flow_order)")

        self._swarm = swarm
        self.name = name or swarm.name

        # Marker for outer Swarm detection (duck-typing)
        self.is_swarm = True

    async def run(
        self,
        input: str,
        *,
        messages: Sequence[Message] | None = None,
        provider: Any = None,
        max_retries: int = 3,
    ) -> RunResult:
        """Execute the inner swarm with context isolation.

        Each call creates a fresh execution context — the inner
        Swarm builds its own ``RunState`` and does not share mutable
        state with the outer Swarm.

        Args:
            input: User query string.
            messages: Not forwarded to inner swarm (context isolation).
                The inner swarm starts with a clean message history.
            provider: LLM provider, forwarded to inner swarm.
            max_retries: Retry attempts, forwarded to inner swarm.

        Returns:
            ``RunResult`` from the inner swarm's execution.

        Raises:
            NestedSwarmError: If the inner swarm execution fails.
        """
        # Context isolation: inner swarm gets fresh message history.
        # The outer swarm's messages are NOT forwarded — each swarm
        # maintains its own conversation context.
        try:
            return await self._swarm.run(
                input,
                provider=provider,
                max_retries=max_retries,
            )
        except Exception as exc:
            raise NestedSwarmError(f"SwarmNode '{self.name}' failed: {exc}") from exc

    async def stream(
        self,
        input: str,
        *,
        messages: Sequence[Message] | None = None,
        provider: Any = None,
        detailed: bool = False,
        max_steps: int | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """Stream inner swarm events with context isolation.

        Each call creates a fresh execution context — the inner
        Swarm builds its own stream and does not share mutable
        state with the outer Swarm.

        Args:
            input: User query string.
            messages: Not forwarded to inner swarm (context isolation).
            provider: LLM provider, forwarded to inner swarm.
            detailed: When ``True``, emit rich event types.
            max_steps: Maximum LLM-tool round-trips per agent.

        Yields:
            ``StreamEvent`` instances from the inner swarm's execution.

        Raises:
            NestedSwarmError: If the inner swarm execution fails.
        """
        try:
            async for event in self._swarm.stream(
                input,
                provider=provider,
                detailed=detailed,
                max_steps=max_steps,
            ):
                yield event
        except Exception as exc:
            raise NestedSwarmError(f"SwarmNode '{self.name}' stream failed: {exc}") from exc

    def describe(self) -> dict[str, Any]:
        """Return a summary including the inner swarm's description."""
        return {
            "type": "nested_swarm",
            "name": self.name,
            "inner": self._swarm.describe(),
        }

    def __repr__(self) -> str:
        return f"SwarmNode(name={self.name!r}, inner={self._swarm!r})"


class RefinementNode:
    """Wraps a RefinementLoop so it can be used as a node in a Swarm.

    The ``is_group = True`` marker makes the Swarm's duck-typing check
    (``getattr(agent, "is_group", False)``) route to ``.stream()``
    during streaming and ``.run()`` during non-streaming execution.

    Args:
        runner: The RefinementLoop to wrap.
        name: Node name for the outer Swarm's flow DSL.
    """

    def __init__(self, *, runner: Any, name: str = "ralph") -> None:
        self._runner = runner
        self.name = name
        self.is_group = True  # triggers Swarm's .stream() delegation path

    async def run(
        self,
        input: str,
        *,
        messages: Sequence[Message] | None = None,
        provider: Any = None,
        max_retries: int = 3,
    ) -> RunResult:
        """Execute the refinement loop and return the final result.

        Args:
            input: User query string.
            messages: Ignored (RefinementLoop manages its own context).
            provider: Ignored (RefinementLoop uses its own execute_fn).
            max_retries: Ignored (RefinementLoop has its own retry logic).

        Returns:
            ``RunResult`` with the refinement loop's final output.

        Raises:
            NestedSwarmError: If the refinement loop fails.
        """
        try:
            result = await self._runner.run(input)
            return RunResult(output=result.output)
        except Exception as exc:
            raise NestedSwarmError(f"RefinementNode '{self.name}' failed: {exc}") from exc

    async def stream(
        self,
        input: str,
        *,
        messages: Sequence[Message] | None = None,
        provider: Any = None,
        detailed: bool = False,
        max_steps: int | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """Stream refinement loop events including inner agent events.

        Args:
            input: User query string.
            messages: Ignored (RefinementLoop manages its own context).
            provider: Ignored (RefinementLoop uses its own execute_fn).
            detailed: Passed through for compatibility but not used by RefinementLoop.
            max_steps: Ignored (RefinementLoop has its own stop conditions).

        Yields:
            ``StreamEvent`` instances from the refinement loop execution.

        Raises:
            NestedSwarmError: If the refinement loop fails.
        """
        try:
            async for event in self._runner.stream(input, name=self.name):
                yield event
        except Exception as exc:
            raise NestedSwarmError(f"RefinementNode '{self.name}' stream failed: {exc}") from exc

    def __repr__(self) -> str:
        # repr keeps the deprecated alias name for backward compatibility
        return f"RalphNode(name={self.name!r})"


# Deprecated alias: use RefinementNode
RalphNode = RefinementNode
