"""Tests for exo._internal.agent_group — ParallelGroup."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest

from exo._internal.agent_group import (
    GroupError,
    ParallelGroup,
)
from exo.agent import Agent
from exo.runner import run
from exo.swarm import Swarm
from exo.types import AgentOutput, RunResult, Usage

# ---------------------------------------------------------------------------
# Fixtures: mock provider
# ---------------------------------------------------------------------------


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

    mock = AsyncMock()
    mock.complete = complete
    return mock


# ---------------------------------------------------------------------------
# ParallelGroup construction
# ---------------------------------------------------------------------------


class TestParallelGroupConstruction:
    def test_minimal_parallel_group(self) -> None:
        """ParallelGroup can be created with one agent."""
        a = Agent(name="a")
        group = ParallelGroup(name="g", agents=[a])

        assert group.name == "g"
        assert group.is_group is True
        assert group.agent_order == ["a"]

    def test_parallel_group_multiple_agents(self) -> None:
        """ParallelGroup tracks all agents in order."""
        a = Agent(name="a")
        b = Agent(name="b")
        c = Agent(name="c")
        group = ParallelGroup(name="g", agents=[a, b, c])

        assert group.agent_order == ["a", "b", "c"]
        assert "a" in group.agents
        assert "b" in group.agents
        assert "c" in group.agents

    def test_parallel_group_empty_raises(self) -> None:
        """ParallelGroup with no agents raises GroupError."""
        with pytest.raises(GroupError, match="at least one agent"):
            ParallelGroup(name="g", agents=[])

    def test_parallel_group_describe(self) -> None:
        """ParallelGroup.describe() returns type and agents."""
        a = Agent(name="a")
        b = Agent(name="b")
        group = ParallelGroup(name="g", agents=[a, b])

        desc = group.describe()

        assert desc["type"] == "parallel"
        assert desc["name"] == "g"
        assert "a" in desc["agents"]
        assert "b" in desc["agents"]

    def test_parallel_group_repr(self) -> None:
        """ParallelGroup.__repr__() includes name and agents."""
        a = Agent(name="a")
        group = ParallelGroup(name="g", agents=[a])

        r = repr(group)

        assert "ParallelGroup" in r
        assert "g" in r


# ---------------------------------------------------------------------------
# ParallelGroup execution
# ---------------------------------------------------------------------------


class TestParallelGroupExecution:
    async def test_parallel_single_agent(self) -> None:
        """ParallelGroup with one agent returns its output."""
        a = Agent(name="a")
        group = ParallelGroup(name="g", agents=[a])
        provider = _make_provider([AgentOutput(text="output_a")])

        result = await group.run("Hello", provider=provider)

        assert result.output == "output_a"

    async def test_parallel_two_agents_joined(self) -> None:
        """ParallelGroup with two agents joins outputs with separator."""
        a = Agent(name="a")
        b = Agent(name="b")
        group = ParallelGroup(name="g", agents=[a, b])

        provider = _make_provider(
            [
                AgentOutput(
                    text="from_a",
                    usage=Usage(input_tokens=10, output_tokens=5, total_tokens=15),
                ),
                AgentOutput(
                    text="from_b",
                    usage=Usage(input_tokens=20, output_tokens=10, total_tokens=30),
                ),
            ]
        )

        result = await group.run("Hello", provider=provider)

        assert "from_a" in result.output
        assert "from_b" in result.output
        # Default separator is "\n\n"
        assert "\n\n" in result.output

    async def test_parallel_usage_aggregation(self) -> None:
        """ParallelGroup sums usage from all agents."""
        a = Agent(name="a")
        b = Agent(name="b")
        group = ParallelGroup(name="g", agents=[a, b])

        provider = _make_provider(
            [
                AgentOutput(
                    text="a",
                    usage=Usage(input_tokens=10, output_tokens=5, total_tokens=15),
                ),
                AgentOutput(
                    text="b",
                    usage=Usage(input_tokens=20, output_tokens=10, total_tokens=30),
                ),
            ]
        )

        result = await group.run("Hello", provider=provider)

        assert result.usage.input_tokens == 30
        assert result.usage.output_tokens == 15
        assert result.usage.total_tokens == 45

    async def test_parallel_custom_separator(self) -> None:
        """ParallelGroup uses custom separator."""
        a = Agent(name="a")
        b = Agent(name="b")
        group = ParallelGroup(name="g", agents=[a, b], separator=" | ")

        provider = _make_provider(
            [
                AgentOutput(text="A"),
                AgentOutput(text="B"),
            ]
        )

        result = await group.run("Hello", provider=provider)

        assert result.output == "A | B"

    async def test_parallel_custom_aggregate_fn(self) -> None:
        """ParallelGroup uses custom aggregation function."""

        def pick_longest(results: list[RunResult]) -> str:
            return max((r.output for r in results), key=len)

        a = Agent(name="a")
        b = Agent(name="b")
        group = ParallelGroup(name="g", agents=[a, b], aggregate_fn=pick_longest)

        provider = _make_provider(
            [
                AgentOutput(text="short"),
                AgentOutput(text="much longer output"),
            ]
        )

        result = await group.run("Hello", provider=provider)

        assert result.output == "much longer output"

    async def test_parallel_three_agents(self) -> None:
        """ParallelGroup with three agents runs all concurrently."""
        a = Agent(name="a")
        b = Agent(name="b")
        c = Agent(name="c")
        group = ParallelGroup(name="g", agents=[a, b, c])

        provider = _make_provider(
            [
                AgentOutput(text="A"),
                AgentOutput(text="B"),
                AgentOutput(text="C"),
            ]
        )

        result = await group.run("Hello", provider=provider)

        assert "A" in result.output
        assert "B" in result.output
        assert "C" in result.output


# ---------------------------------------------------------------------------
# Groups in Swarm flow DSL
# ---------------------------------------------------------------------------


class TestGroupsInSwarm:
    async def test_parallel_group_in_workflow(self) -> None:
        """ParallelGroup works as a node in Swarm workflow."""
        a = Agent(name="a")
        b = Agent(name="b")
        group = ParallelGroup(name="group_ab", agents=[a, b])
        c = Agent(name="c")

        # Swarm: group_ab >> c
        swarm = Swarm(
            agents=[group, c],
            flow="group_ab >> c",
        )

        provider = _make_provider(
            [
                AgentOutput(text="A_out"),  # agent a in parallel group
                AgentOutput(text="B_out"),  # agent b in parallel group
                AgentOutput(text="C_final"),  # agent c
            ]
        )

        result = await swarm.run("Hello", provider=provider)

        assert result.output == "C_final"

    async def test_agent_then_parallel_group(self) -> None:
        """Regular agent followed by ParallelGroup in workflow."""
        a = Agent(name="a")
        b = Agent(name="b")
        c = Agent(name="c")
        group = ParallelGroup(name="bc_group", agents=[b, c])

        swarm = Swarm(
            agents=[a, group],
            flow="a >> bc_group",
        )

        provider = _make_provider(
            [
                AgentOutput(text="from_a"),
                AgentOutput(text="from_b"),
                AgentOutput(text="from_c"),
            ]
        )

        result = await swarm.run("Hello", provider=provider)

        # ParallelGroup output is joined
        assert "from_b" in result.output
        assert "from_c" in result.output

    async def test_mixed_topology(self) -> None:
        """Mixed topology: agent >> parallel_group >> agent."""
        a = Agent(name="a")
        b = Agent(name="b")
        c = Agent(name="c")
        d = Agent(name="d")

        group = ParallelGroup(name="bc_group", agents=[b, c])

        swarm = Swarm(
            agents=[a, group, d],
            flow="a >> bc_group >> d",
        )

        provider = _make_provider(
            [
                AgentOutput(text="from_a"),
                AgentOutput(text="from_b"),
                AgentOutput(text="from_c"),
                AgentOutput(text="final_d"),
            ]
        )

        result = await swarm.run("Hello", provider=provider)

        assert result.output == "final_d"

    async def test_group_via_run_public_api(self) -> None:
        """Groups work with the public run() API."""
        a = Agent(name="a")
        b = Agent(name="b")
        group = ParallelGroup(name="ab_group", agents=[a, b])

        swarm = Swarm(agents=[group])

        provider = _make_provider(
            [
                AgentOutput(text="A"),
                AgentOutput(text="B"),
            ]
        )

        result = await run(swarm, "Hello", provider=provider)

        assert isinstance(result, RunResult)
        assert "A" in result.output
        assert "B" in result.output

    async def test_parallel_group_standalone(self) -> None:
        """ParallelGroup can run standalone without Swarm."""
        a = Agent(name="a")
        b = Agent(name="b")
        group = ParallelGroup(name="g", agents=[a, b])

        provider = _make_provider(
            [
                AgentOutput(text="A"),
                AgentOutput(text="B"),
            ]
        )

        result = await group.run("test", provider=provider)

        assert "A" in result.output
        assert "B" in result.output

