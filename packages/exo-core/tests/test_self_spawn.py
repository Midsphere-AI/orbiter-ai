"""Tests for Agent self-spawn capability — US-025."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest

from exo.agent import Agent, AgentError
from exo.models.types import ModelResponse  # pyright: ignore[reportMissingImports]
from exo.tool import FunctionTool, tool
from exo.types import Usage

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_provider(content: str = "done") -> AsyncMock:
    """Create a mock provider that returns a single text response."""
    resp = ModelResponse(
        id="r1",
        model="test",
        content=content,
        tool_calls=[],
        usage=Usage(input_tokens=5, output_tokens=3, total_tokens=8),
    )
    provider = AsyncMock()
    provider.complete = AsyncMock(return_value=resp)
    return provider


# ---------------------------------------------------------------------------
# Init / attribute tests
# ---------------------------------------------------------------------------


class TestSelfSpawnInit:
    def test_allow_self_spawn_true_by_default(self) -> None:
        """Default agent has spawn_self tool (allow_self_spawn=True by default)."""
        agent = Agent(name="bot", memory=None, context=None)
        assert "spawn_self" in agent.tools
        assert agent.allow_self_spawn is True

    def test_allow_self_spawn_false_disables_tool(self) -> None:
        """allow_self_spawn=False removes spawn_self tool."""
        agent = Agent(name="bot", memory=None, context=None, allow_self_spawn=False)
        assert "spawn_self" not in agent.tools
        assert agent.allow_self_spawn is False

    def test_subagents_false_disables_spawn_self(self) -> None:
        """subagents=False convenience flag disables spawn_self."""
        agent = Agent(name="bot", memory=None, context=None, subagents=False)
        assert "spawn_self" not in agent.tools
        assert agent.allow_self_spawn is False

    def test_subagents_true_enables_spawn_self(self) -> None:
        """subagents=True convenience flag enables spawn_self."""
        agent = Agent(name="bot", memory=None, context=None, subagents=True)
        assert "spawn_self" in agent.tools
        assert agent.allow_self_spawn is True

    def test_allow_self_spawn_explicit_true_adds_tool(self) -> None:
        """allow_self_spawn=True registers a spawn_self tool."""
        agent = Agent(name="bot", memory=None, context=None, allow_self_spawn=True)
        assert "spawn_self" in agent.tools
        assert agent.allow_self_spawn is True

    def test_spawn_depth_starts_at_zero(self) -> None:
        """New agents start at spawn depth 0."""
        agent = Agent(name="bot", memory=None, context=None)
        assert agent._spawn_depth == 0

    def test_max_spawn_depth_default_is_three(self) -> None:
        """Default max_spawn_depth is 3."""
        agent = Agent(name="bot", memory=None, context=None)
        assert agent.max_spawn_depth == 3

    def test_max_spawn_depth_custom(self) -> None:
        """Custom max_spawn_depth is stored."""
        agent = Agent(name="bot", memory=None, context=None, max_spawn_depth=5)
        assert agent.max_spawn_depth == 5

    def test_current_provider_starts_none(self) -> None:
        """_current_provider is None before any run()."""
        agent = Agent(name="bot", memory=None, context=None)
        assert agent._current_provider is None

    def test_spawn_self_is_function_tool(self) -> None:
        """The spawn_self tool is a FunctionTool instance."""
        agent = Agent(name="bot", memory=None, context=None, allow_self_spawn=True)
        assert isinstance(agent.tools["spawn_self"], FunctionTool)

    def test_spawn_self_tool_has_tasks_parameter(self) -> None:
        """spawn_self tool schema has a 'tasks' parameter."""
        agent = Agent(name="bot", memory=None, context=None, allow_self_spawn=True)
        schema = agent.tools["spawn_self"].parameters
        assert "tasks" in schema["properties"]

    def test_max_spawn_children_default_is_four(self) -> None:
        agent = Agent(name="bot", memory=None, context=None)
        assert agent.max_spawn_children == 4

    def test_max_spawn_children_custom(self) -> None:
        agent = Agent(name="bot", memory=None, context=None, max_spawn_children=2)
        assert agent.max_spawn_children == 2


# ---------------------------------------------------------------------------
# spawn_self tool behaviour — depth guard and provider guard
# ---------------------------------------------------------------------------


class TestSpawnSelfGuards:
    @pytest.mark.asyncio
    async def test_spawn_self_returns_error_at_max_depth(self) -> None:
        """Returns error string when spawn depth is at max_spawn_depth."""
        agent = Agent(
            name="bot",
            memory=None,
            context=None,
            allow_self_spawn=True,
            max_spawn_depth=3,
        )
        agent._spawn_depth = 3  # at the limit
        agent._current_provider = _mock_provider()

        result = await agent.tools["spawn_self"].execute(tasks=["some task"])
        assert "Maximum spawn depth" in str(result)
        assert "3" in str(result)

    @pytest.mark.asyncio
    async def test_spawn_self_returns_error_when_depth_exceeds_max(self) -> None:
        """Returns error string when spawn depth exceeds max_spawn_depth."""
        agent = Agent(
            name="bot",
            memory=None,
            context=None,
            allow_self_spawn=True,
            max_spawn_depth=2,
        )
        agent._spawn_depth = 5  # well above limit
        agent._current_provider = _mock_provider()

        result = await agent.tools["spawn_self"].execute(tasks=["task"])
        assert "Maximum spawn depth" in str(result)

    @pytest.mark.asyncio
    async def test_spawn_self_returns_error_no_provider(self) -> None:
        """Returns error string when _current_provider is None."""
        agent = Agent(name="bot", memory=None, context=None, allow_self_spawn=True)
        # _current_provider is None by default (outside of run())

        result = await agent.tools["spawn_self"].execute(tasks=["task"])
        assert "No provider available" in str(result)

    @pytest.mark.asyncio
    async def test_spawn_self_allowed_at_depth_less_than_max(self) -> None:
        """Does NOT return error string when depth < max_spawn_depth."""
        provider = _mock_provider(content="child result")
        agent = Agent(
            name="bot",
            memory=None,
            context=None,
            allow_self_spawn=True,
            max_spawn_depth=3,
        )
        agent._spawn_depth = 2  # below limit
        agent._current_provider = provider

        result = await agent.tools["spawn_self"].execute(tasks=["sub task"])
        # Should succeed and return the provider's content
        assert result == "child result"

    @pytest.mark.asyncio
    async def test_spawn_self_returns_error_when_tasks_exceed_max_children(self) -> None:
        """Returns error string when tasks list exceeds max_spawn_children."""
        agent = Agent(
            name="bot",
            memory=None,
            context=None,
            allow_self_spawn=True,
            max_spawn_children=2,
        )
        agent._current_provider = _mock_provider()
        result = await agent.tools["spawn_self"].execute(tasks=["a", "b", "c"])
        assert "Too many tasks" in str(result)
        assert "3" in str(result)
        assert "2" in str(result)

    @pytest.mark.asyncio
    async def test_spawn_self_returns_error_on_empty_tasks(self) -> None:
        """Returns error string when tasks list is empty."""
        agent = Agent(name="bot", memory=None, context=None, allow_self_spawn=True)
        agent._current_provider = _mock_provider()
        result = await agent.tools["spawn_self"].execute(tasks=[])
        assert "Empty tasks list" in str(result)


# ---------------------------------------------------------------------------
# spawn_self tool — spawned agent properties
# ---------------------------------------------------------------------------


class TestSpawnedAgentProperties:
    @pytest.mark.asyncio
    async def test_spawned_agent_has_allow_self_spawn_false(self) -> None:
        """Spawned agent does not have a spawn_self tool (allow_self_spawn=False)."""
        captured_agents: list[Any] = []

        async def capturing_complete(messages: Any, **kwargs: Any) -> Any:
            return ModelResponse(
                id="r",
                model="m",
                content="ok",
                tool_calls=[],
                usage=Usage(input_tokens=1, output_tokens=1, total_tokens=2),
            )

        provider = AsyncMock()
        provider.complete = AsyncMock(side_effect=capturing_complete)

        parent = Agent(
            name="parent",
            memory=None,
            context=None,
            allow_self_spawn=True,
            max_spawn_depth=3,
        )
        parent._current_provider = provider

        # Patch Agent.__init__ to capture the child agent
        original_run_inner = Agent._run_inner

        spawned_agents: list[Agent] = []

        async def capturing_run_inner(self_agent: Any, *args: Any, **kwargs: Any) -> Any:
            if self_agent.name != "parent":
                spawned_agents.append(self_agent)
            return await original_run_inner(self_agent, *args, **kwargs)

        import unittest.mock as mock

        with mock.patch.object(Agent, "_run_inner", capturing_run_inner):
            await parent.tools["spawn_self"].execute(tasks=["do something"])

        assert len(spawned_agents) == 1
        child = spawned_agents[0]
        assert "spawn_self" not in child.tools
        assert child.allow_self_spawn is False

    @pytest.mark.asyncio
    async def test_spawned_agent_depth_incremented(self) -> None:
        """Spawned agent has _spawn_depth = parent._spawn_depth + 1."""
        provider = _mock_provider(content="result")

        parent = Agent(
            name="parent",
            memory=None,
            context=None,
            allow_self_spawn=True,
        )
        parent._spawn_depth = 1
        parent._current_provider = provider

        spawned_agents: list[Agent] = []
        original_run_inner = Agent._run_inner

        async def capturing_run_inner(self_agent: Any, *args: Any, **kwargs: Any) -> Any:
            if self_agent.name != "parent":
                spawned_agents.append(self_agent)
            return await original_run_inner(self_agent, *args, **kwargs)

        import unittest.mock as mock

        with mock.patch.object(Agent, "_run_inner", capturing_run_inner):
            await parent.tools["spawn_self"].execute(tasks=["sub task"])

        assert len(spawned_agents) == 1
        assert spawned_agents[0]._spawn_depth == 2  # parent was 1 → child is 2

    @pytest.mark.asyncio
    async def test_spawned_agent_shares_long_term_memory(self) -> None:
        """Spawned agent shares the parent's long-term memory instance."""
        try:
            from exo.memory.base import AgentMemory  # pyright: ignore[reportMissingImports]
            from exo.memory.short_term import (
                ShortTermMemory,  # pyright: ignore[reportMissingImports]
            )
        except ImportError:
            pytest.skip("exo-memory not installed")

        class FakeLongTerm:
            async def add(self, item: Any) -> None:
                pass

            async def search(self, **kwargs: Any) -> list[Any]:
                return []

            async def clear(self, **kwargs: Any) -> None:
                pass

            async def get_recent(self, *args: Any, **kwargs: Any) -> list[Any]:
                return []

        long_term = FakeLongTerm()
        parent_memory = AgentMemory(short_term=ShortTermMemory(), long_term=long_term)

        provider = _mock_provider(content="ok")

        parent = Agent(
            name="parent",
            memory=parent_memory,
            context=None,
            allow_self_spawn=True,
        )
        parent._current_provider = provider

        spawned_agents: list[Agent] = []
        original_run_inner = Agent._run_inner

        async def capturing_run_inner(self_agent: Any, *args: Any, **kwargs: Any) -> Any:
            if self_agent.name != "parent":
                spawned_agents.append(self_agent)
            return await original_run_inner(self_agent, *args, **kwargs)

        import unittest.mock as mock

        with mock.patch.object(Agent, "_run_inner", capturing_run_inner):
            await parent.tools["spawn_self"].execute(tasks=["task"])

        assert len(spawned_agents) == 1
        child = spawned_agents[0]
        assert child.memory is not None
        child_long_term = getattr(child.memory, "long_term", None)
        # Spawned agent shares the SAME long_term object
        assert child_long_term is long_term

    @pytest.mark.asyncio
    async def test_spawned_agent_fresh_short_term_memory(self) -> None:
        """Spawned agent gets a fresh ShortTermMemory, not the parent's."""
        try:
            from exo.memory.base import AgentMemory  # pyright: ignore[reportMissingImports]
            from exo.memory.short_term import (
                ShortTermMemory,  # pyright: ignore[reportMissingImports]
            )
        except ImportError:
            pytest.skip("exo-memory not installed")

        parent_short_term = ShortTermMemory()
        parent_memory = AgentMemory(
            short_term=parent_short_term,
            long_term=ShortTermMemory(),  # use as fake long_term
        )

        provider = _mock_provider(content="ok")

        parent = Agent(
            name="parent",
            memory=parent_memory,
            context=None,
            allow_self_spawn=True,
        )
        parent._current_provider = provider

        spawned_agents: list[Agent] = []
        original_run_inner = Agent._run_inner

        async def capturing_run_inner(self_agent: Any, *args: Any, **kwargs: Any) -> Any:
            if self_agent.name != "parent":
                spawned_agents.append(self_agent)
            return await original_run_inner(self_agent, *args, **kwargs)

        import unittest.mock as mock

        with mock.patch.object(Agent, "_run_inner", capturing_run_inner):
            await parent.tools["spawn_self"].execute(tasks=["task"])

        assert len(spawned_agents) == 1
        child = spawned_agents[0]
        child_short_term = getattr(child.memory, "short_term", None)
        # Spawned agent gets a DIFFERENT short_term instance
        assert child_short_term is not parent_short_term

    @pytest.mark.asyncio
    async def test_spawned_agent_uses_parent_model(self) -> None:
        """Spawned agent uses the same model as the parent."""
        provider = _mock_provider(content="result")
        parent = Agent(
            name="parent",
            model="openai:gpt-4o-mini",
            memory=None,
            context=None,
            allow_self_spawn=True,
        )
        parent._current_provider = provider

        spawned_agents: list[Agent] = []
        original_run_inner = Agent._run_inner

        async def capturing_run_inner(self_agent: Any, *args: Any, **kwargs: Any) -> Any:
            if self_agent.name != "parent":
                spawned_agents.append(self_agent)
            return await original_run_inner(self_agent, *args, **kwargs)

        import unittest.mock as mock

        with mock.patch.object(Agent, "_run_inner", capturing_run_inner):
            await parent.tools["spawn_self"].execute(tasks=["task"])

        assert len(spawned_agents) == 1
        assert spawned_agents[0].model == "openai:gpt-4o-mini"

    @pytest.mark.asyncio
    async def test_spawned_agent_tools_exclude_spawn_self(self) -> None:
        """Spawned agent has parent's tools but NOT spawn_self."""

        @tool
        def helper(x: str) -> str:
            """A helper tool."""
            return x

        provider = _mock_provider(content="result")
        parent = Agent(
            name="parent",
            memory=None,
            context=None,
            tools=[helper],
            allow_self_spawn=True,
        )
        parent._current_provider = provider

        spawned_agents: list[Agent] = []
        original_run_inner = Agent._run_inner

        async def capturing_run_inner(self_agent: Any, *args: Any, **kwargs: Any) -> Any:
            if self_agent.name != "parent":
                spawned_agents.append(self_agent)
            return await original_run_inner(self_agent, *args, **kwargs)

        import unittest.mock as mock

        with mock.patch.object(Agent, "_run_inner", capturing_run_inner):
            await parent.tools["spawn_self"].execute(tasks=["task"])

        assert len(spawned_agents) == 1
        child = spawned_agents[0]
        assert "helper" in child.tools
        assert "spawn_self" not in child.tools


# ---------------------------------------------------------------------------
# Provider lifecycle — _current_provider set/cleared during run()
# ---------------------------------------------------------------------------


class TestProviderLifecycle:
    @pytest.mark.asyncio
    async def test_current_provider_set_during_run(self) -> None:
        """_current_provider is non-None inside the run() tool loop."""
        provider_during_run: list[Any] = []

        @tool
        async def capture_provider_ref(dummy: str = "") -> str:
            """Captures parent._current_provider."""
            return "captured"

        provider = _mock_provider(content="done")
        agent = Agent(
            name="bot",
            memory=None,
            context=None,
            allow_self_spawn=True,
        )

        # Verify _current_provider is None before
        assert agent._current_provider is None
        await agent.run("hello", provider=provider)
        # Verify _current_provider is None after
        assert agent._current_provider is None

    @pytest.mark.asyncio
    async def test_current_provider_cleared_after_exception(self) -> None:
        """_current_provider is cleared even if run() raises."""

        broken_provider = AsyncMock()
        broken_provider.complete = AsyncMock(side_effect=RuntimeError("boom"))

        agent = Agent(name="bot", memory=None, context=None, allow_self_spawn=True)

        with pytest.raises((AgentError, RuntimeError)):
            await agent.run("hello", provider=broken_provider)

        # Must be cleaned up even after exception
        assert agent._current_provider is None

    @pytest.mark.asyncio
    async def test_spawn_self_returns_task_result(self) -> None:
        """spawn_self executes the spawned agent and returns its text output."""
        provider = _mock_provider(content="task completed successfully")

        agent = Agent(name="parent", memory=None, context=None, allow_self_spawn=True)
        agent._current_provider = provider

        result = await agent.tools["spawn_self"].execute(tasks=["do something"])
        assert result == "task completed successfully"


# ---------------------------------------------------------------------------
# spawn_self — parallel execution
# ---------------------------------------------------------------------------


class TestSpawnSelfParallel:
    @pytest.mark.asyncio
    async def test_spawn_self_parallel_two_children(self) -> None:
        """Two tasks spawn two children concurrently."""
        provider = _mock_provider(content="result")
        parent = Agent(
            name="parent",
            memory=None,
            context=None,
            allow_self_spawn=True,
        )
        parent._current_provider = provider

        spawned_agents: list[Any] = []
        original_run_inner = Agent._run_inner

        async def capturing_run_inner(self_agent: Any, *args: Any, **kwargs: Any) -> Any:
            if self_agent.name != "parent":
                spawned_agents.append(self_agent)
            return await original_run_inner(self_agent, *args, **kwargs)

        import unittest.mock as mock

        with mock.patch.object(Agent, "_run_inner", capturing_run_inner):
            await parent.tools["spawn_self"].execute(tasks=["task A", "task B"])

        assert len(spawned_agents) == 2
        names = {a.name for a in spawned_agents}
        assert len(names) == 2  # unique names
        for child in spawned_agents:
            assert child._spawn_depth == 1

    @pytest.mark.asyncio
    async def test_spawn_self_parallel_results_formatted(self) -> None:
        """Multiple tasks return [Task N] formatted results."""
        call_count = 0

        async def counting_complete(messages: Any, **kwargs: Any) -> Any:
            nonlocal call_count
            call_count += 1
            return ModelResponse(
                id=f"r{call_count}",
                model="test",
                content=f"result {call_count}",
                tool_calls=[],
                usage=Usage(input_tokens=1, output_tokens=1, total_tokens=2),
            )

        provider = AsyncMock()
        provider.complete = AsyncMock(side_effect=counting_complete)

        parent = Agent(
            name="parent",
            memory=None,
            context=None,
            allow_self_spawn=True,
        )
        parent._current_provider = provider

        result = await parent.tools["spawn_self"].execute(tasks=["a", "b"])
        assert "[Task 1]:" in result
        assert "[Task 2]:" in result

    @pytest.mark.asyncio
    async def test_spawn_self_single_task_no_prefix(self) -> None:
        """Single task returns raw result without [Task N] prefix."""
        provider = _mock_provider(content="the answer")
        parent = Agent(
            name="parent",
            memory=None,
            context=None,
            allow_self_spawn=True,
        )
        parent._current_provider = provider

        result = await parent.tools["spawn_self"].execute(tasks=["only task"])
        assert result == "the answer"
        assert "[Task" not in result
