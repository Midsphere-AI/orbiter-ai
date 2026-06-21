"""Swarm integration tests via the public run() API.

End-to-end tests covering all swarm modes (workflow, handoff, team),
parallel agent groups, nested swarms, and public API exports.
"""

from __future__ import annotations

from typing import Any

from exo.agent import Agent
from exo.runner import run
from exo.swarm import Swarm
from exo.tool import tool
from exo.types import (
    AgentOutput,
    RunResult,
    ToolCall,
    Usage,
)

# ---------------------------------------------------------------------------
# Shared fixtures: mock LLM provider
# ---------------------------------------------------------------------------

_DEFAULT_USAGE = Usage(input_tokens=10, output_tokens=5, total_tokens=15)


def _make_provider(responses: list[AgentOutput]) -> Any:
    """Create a mock provider returning pre-defined AgentOutput values."""
    call_count = 0

    async def complete(messages: Any, **kwargs: Any) -> Any:
        nonlocal call_count
        resp = responses[min(call_count, len(responses) - 1)]
        call_count += 1

        class FakeResponse:
            content = resp.text
            tool_calls = resp.tool_calls
            usage = resp.usage

        return FakeResponse()

    class Provider:
        pass

    p = Provider()
    p.complete = complete  # type: ignore[attr-defined]
    return p


def _text(text: str) -> AgentOutput:
    """Helper: text-only AgentOutput with default usage."""
    return AgentOutput(text=text, usage=_DEFAULT_USAGE)


def _tc(text: str, calls: list[ToolCall]) -> AgentOutput:
    """Helper: AgentOutput with tool calls and default usage."""
    return AgentOutput(text=text, tool_calls=calls, usage=_DEFAULT_USAGE)


# ---------------------------------------------------------------------------
# Workflow mode integration via run()
# ---------------------------------------------------------------------------


class TestWorkflowIntegration:
    """run(swarm, input) with mode='workflow'."""

    async def test_two_agent_workflow(self) -> None:
        """Two agents in sequence, output->input chaining."""
        a = Agent(name="a", instructions="Step 1")
        b = Agent(name="b", instructions="Step 2")

        swarm = Swarm(agents=[a, b], flow="a >> b")
        provider = _make_provider([_text("step1-result"), _text("final-result")])

        result = await run(swarm, "start", provider=provider)

        assert isinstance(result, RunResult)
        assert result.output == "final-result"

    async def test_three_agent_pipeline(self) -> None:
        """Three agents in a pipeline via flow DSL."""
        a = Agent(name="a")
        b = Agent(name="b")
        c = Agent(name="c")

        swarm = Swarm(agents=[a, b, c], flow="a >> b >> c")
        provider = _make_provider([_text("out-a"), _text("out-b"), _text("out-c")])

        result = await run(swarm, "input", provider=provider)

        assert result.output == "out-c"

    async def test_workflow_without_flow_dsl(self) -> None:
        """Without flow DSL, agents run in list order."""
        a = Agent(name="a")
        b = Agent(name="b")

        swarm = Swarm(agents=[a, b])
        provider = _make_provider([_text("first"), _text("second")])

        result = await run(swarm, "go", provider=provider)

        assert result.output == "second"

    def test_workflow_sync(self) -> None:
        """run.sync() works with swarms."""
        a = Agent(name="a")
        b = Agent(name="b")

        swarm = Swarm(agents=[a, b], flow="a >> b")
        provider = _make_provider([_text("sync-a"), _text("sync-b")])

        result = run.sync(swarm, "hello", provider=provider)

        assert result.output == "sync-b"


# ---------------------------------------------------------------------------
# Handoff mode integration via run()
# ---------------------------------------------------------------------------


class TestHandoffIntegration:
    """run(swarm, input) with mode='handoff'."""

    async def test_simple_handoff(self) -> None:
        """Agent A hands off to agent B, which returns final text."""
        b = Agent(name="b", instructions="Handler B")
        a = Agent(name="a", instructions="Triage", handoffs=[b])

        swarm = Swarm(agents=[a, b], mode="handoff")
        # Agent A outputs "b" (triggers handoff), agent B outputs final text
        provider = _make_provider([_text("b"), _text("B handled it.")])

        result = await run(swarm, "help me", provider=provider)

        assert result.output == "B handled it."

    async def test_handoff_chain(self) -> None:
        """A -> B -> C handoff chain."""
        c = Agent(name="c")
        b = Agent(name="b", handoffs=[c])
        a = Agent(name="a", handoffs=[b])

        swarm = Swarm(agents=[a, b, c], mode="handoff")
        provider = _make_provider([_text("b"), _text("c"), _text("Final from C.")])

        result = await run(swarm, "start", provider=provider)

        assert result.output == "Final from C."

    async def test_no_handoff_returns_directly(self) -> None:
        """Agent with no matching handoff target returns its output."""
        b = Agent(name="b")
        a = Agent(name="a", handoffs=[b])

        swarm = Swarm(agents=[a, b], mode="handoff")
        # Agent A outputs something that isn't "b" → no handoff
        provider = _make_provider([_text("I handled it myself.")])

        result = await run(swarm, "query", provider=provider)

        assert result.output == "I handled it myself."


# ---------------------------------------------------------------------------
# Team mode integration via run()
# ---------------------------------------------------------------------------


class TestTeamIntegration:
    """run(swarm, input) with mode='team'."""

    async def test_lead_delegates_to_worker(self) -> None:
        """Lead calls delegate_to_worker tool, worker runs, lead synthesizes."""
        lead = Agent(name="lead", instructions="Coordinate work")
        worker = Agent(name="worker", instructions="Do the work")

        swarm = Swarm(agents=[lead, worker], mode="team")

        # Lead's first call: delegate to worker
        # Worker's call: returns worker result
        # Lead's second call (after tool result): final synthesis
        provider = _make_provider(
            [
                _tc(
                    "",
                    [
                        ToolCall(
                            id="tc1",
                            name="delegate_to_worker",
                            arguments='{"task":"analyze data"}',
                        )
                    ],
                ),
                _text("Worker analysis complete."),  # worker agent
                _text("Based on the analysis: all good."),  # lead final
            ]
        )

        result = await run(swarm, "analyze the data", provider=provider)

        assert result.output == "Based on the analysis: all good."

    async def test_lead_no_delegation(self) -> None:
        """Lead can respond directly without delegating."""
        lead = Agent(name="lead")
        worker = Agent(name="worker")

        swarm = Swarm(agents=[lead, worker], mode="team")
        provider = _make_provider([_text("I can handle this directly.")])

        result = await run(swarm, "simple task", provider=provider)

        assert result.output == "I can handle this directly."


# ---------------------------------------------------------------------------
# Parallel/Serial groups via run()
# ---------------------------------------------------------------------------


class TestGroupIntegration:
    """run(swarm, input) with ParallelGroup nodes."""

    async def test_parallel_group_in_workflow(self) -> None:
        """ParallelGroup as a node in a swarm workflow."""
        from exo._internal.agent_group import ParallelGroup

        a1 = Agent(name="a1")
        a2 = Agent(name="a2")
        pg = ParallelGroup(name="parallel", agents=[a1, a2])
        final = Agent(name="final")

        swarm = Swarm(agents=[pg, final], flow="parallel >> final")
        # a1 and a2 run in parallel (2 calls), then final (1 call)
        provider = _make_provider(
            [_text("result-a1"), _text("result-a2"), _text("combined result")]
        )

        result = await run(swarm, "go", provider=provider)

        assert result.output == "combined result"

    async def test_parallel_group_custom_aggregation(self) -> None:
        """ParallelGroup with custom aggregation function."""
        from exo._internal.agent_group import ParallelGroup

        a1 = Agent(name="a1")
        a2 = Agent(name="a2")

        def agg(results: list[RunResult]) -> str:
            return " + ".join(r.output for r in results)

        pg = ParallelGroup(name="pg", agents=[a1, a2], aggregate_fn=agg)

        swarm = Swarm(agents=[pg])
        provider = _make_provider([_text("alpha"), _text("beta")])

        result = await run(swarm, "go", provider=provider)

        assert result.output == "alpha + beta"


# ---------------------------------------------------------------------------
# Nested swarms via run()
# ---------------------------------------------------------------------------


class TestNestedSwarmIntegration:
    """run(swarm, input) with nested swarms via SwarmNode."""

    async def test_nested_workflow(self) -> None:
        """SwarmNode wraps an inner workflow swarm in an outer swarm."""
        from exo._internal.nested import SwarmNode

        inner_a = Agent(name="ia")
        inner_b = Agent(name="ib")
        inner_swarm = Swarm(agents=[inner_a, inner_b], flow="ia >> ib")
        node = SwarmNode(swarm=inner_swarm, name="inner")

        outer_start = Agent(name="start")
        outer_end = Agent(name="end")

        outer = Swarm(agents=[outer_start, node, outer_end], flow="start >> inner >> end")
        # outer_start (1), inner_a (1), inner_b (1), outer_end (1)
        provider = _make_provider(
            [_text("from-start"), _text("inner-a"), _text("inner-b"), _text("final")]
        )

        result = await run(outer, "go", provider=provider)

        assert result.output == "final"

    async def test_nested_handoff_inner(self) -> None:
        """Inner swarm uses handoff mode within an outer workflow."""
        from exo._internal.nested import SwarmNode

        hb = Agent(name="hb")
        ha = Agent(name="ha", handoffs=[hb])
        inner = Swarm(agents=[ha, hb], mode="handoff")
        node = SwarmNode(swarm=inner, name="handoff_inner")

        outer = Swarm(agents=[node])
        # ha outputs "hb" (handoff), then hb returns final
        provider = _make_provider([_text("hb"), _text("Handoff result.")])

        result = await run(outer, "test", provider=provider)

        assert result.output == "Handoff result."


# ---------------------------------------------------------------------------
# Public API export tests
# ---------------------------------------------------------------------------


class TestSwarmPublicAPI:
    """Verify swarm-related types are importable from exo."""

    def test_swarm_importable(self) -> None:
        from exo import Swarm as SwarmImport

        assert SwarmImport is Swarm

    def test_swarm_node_importable(self) -> None:
        from exo import SwarmNode
        from exo._internal.nested import SwarmNode as Direct

        assert SwarmNode is Direct

    def test_parallel_group_importable(self) -> None:
        from exo import ParallelGroup
        from exo._internal.agent_group import ParallelGroup as Direct

        assert ParallelGroup is Direct

    async def test_run_detects_swarm(self) -> None:
        """run() detects Swarm via flow_order attribute and delegates."""
        a = Agent(name="a")
        swarm = Swarm(agents=[a])
        provider = _make_provider([_text("ok")])

        result = await run(swarm, "test", provider=provider)

        assert result.output == "ok"

    async def test_run_detects_agent(self) -> None:
        """run() executes plain Agent without swarm delegation."""
        a = Agent(name="a")
        provider = _make_provider([_text("direct")])

        result = await run(a, "test", provider=provider)

        assert result.output == "direct"


# ---------------------------------------------------------------------------
# Swarm with tools integration
# ---------------------------------------------------------------------------


class TestSwarmWithTools:
    """Swarm workflows where agents use tools."""

    async def test_workflow_agent_with_tool(self) -> None:
        """Agent in a workflow uses a @tool then passes output to next agent."""

        @tool
        def reverse(text: str) -> str:
            """Reverse text."""
            return str(text)[::-1]

        a = Agent(name="a", tools=[reverse])
        b = Agent(name="b")

        swarm = Swarm(agents=[a, b], flow="a >> b")
        provider = _make_provider(
            [
                _tc(
                    "",
                    [ToolCall(id="tc1", name="reverse", arguments='{"text":"hello"}')],
                ),
                _text("Reversed: olleh"),  # agent a final response
                _text("Processed: olleh"),  # agent b
            ]
        )

        result = await run(swarm, "reverse hello", provider=provider)

        assert result.output == "Processed: olleh"
