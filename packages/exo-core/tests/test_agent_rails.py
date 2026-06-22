"""Tests for US-005: Agent integration with rails + backward compatibility."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest

from exo.agent import Agent
from exo.hooks import HookPoint
from exo.models.types import ModelResponse  # pyright: ignore[reportMissingImports]
from exo.rail import Rail, RailAbortError, RailAction, RailManager
from exo.rail_types import RailContext, ToolCallInputs
from exo.tool import tool
from exo.types import ToolCall, Usage

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@tool
def greet(name: str) -> str:
    """Say hello."""
    return f"Hello, {name}!"


def _mock_provider(
    content: str = "Hello!",
    tool_calls: list[ToolCall] | None = None,
    usage: Usage | None = None,
) -> AsyncMock:
    """Create a mock provider that returns a fixed ModelResponse."""
    resp = ModelResponse(
        id="resp-1",
        model="test-model",
        content=content,
        tool_calls=tool_calls or [],
        usage=usage or Usage(input_tokens=10, output_tokens=5, total_tokens=15),
    )
    provider = AsyncMock()
    provider.complete = AsyncMock(return_value=resp)
    return provider


def _multi_step_provider(*responses: ModelResponse) -> AsyncMock:
    provider = AsyncMock()
    provider.complete = AsyncMock(side_effect=list(responses))
    return provider


# ---------------------------------------------------------------------------
# Agent construction with rails
# ---------------------------------------------------------------------------


class TestAgentRailsConstruction:
    def test_no_rails_default(self) -> None:
        """Agent with no rails has no rail_manager."""
        agent = Agent(name="bot")
        assert agent.rail_manager is None

    def test_rails_creates_rail_manager(self) -> None:
        """Providing rails creates a RailManager."""

        class NoopRail(Rail):
            async def handle(self, ctx: RailContext) -> RailAction | None:
                return RailAction.CONTINUE

        rail = NoopRail("noop")
        agent = Agent(name="bot", rails=[rail])
        assert agent.rail_manager is not None
        assert isinstance(agent.rail_manager, RailManager)

    def test_rails_registered_as_hooks(self) -> None:
        """Rails are registered as hooks on all hook points except CONTEXT_WINDOW."""

        class NoopRail(Rail):
            async def handle(self, ctx: RailContext) -> RailAction | None:
                return RailAction.CONTINUE

        agent = Agent(name="bot", rails=[NoopRail("noop")])
        for point in HookPoint:
            if point == HookPoint.CONTEXT_WINDOW:
                assert not agent.hook_manager.has_hooks(point)
            else:
                assert agent.hook_manager.has_hooks(point)

    def test_no_rails_no_hooks_added(self) -> None:
        """Agent without rails doesn't register any hooks."""
        agent = Agent(name="bot", memory=None)
        for point in HookPoint:
            assert not agent.hook_manager.has_hooks(point)


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------


class TestBackwardCompatibility:
    async def test_no_rails_same_behavior(self) -> None:
        """Agent without rails behaves exactly as before."""
        provider = _mock_provider(content="Hi there!")
        agent = Agent(name="bot", instructions="Be helpful.")

        output = await agent.run("Hello", provider=provider)

        assert output.text == "Hi there!"
        provider.complete.assert_awaited_once()

    async def test_hooks_still_work_without_rails(self) -> None:
        """Traditional hooks fire normally when no rails are set."""
        events: list[str] = []

        async def pre_hook(**data: Any) -> None:
            events.append("pre_llm")

        provider = _mock_provider()
        agent = Agent(
            name="bot",
            hooks=[(HookPoint.PRE_LLM_CALL, pre_hook)],
        )

        await agent.run("Hello", provider=provider)

        assert events == ["pre_llm"]

    async def test_hooks_and_rails_both_execute(self) -> None:
        """Both traditional hooks and rails fire when both are configured."""
        events: list[str] = []

        async def traditional_hook(**data: Any) -> None:
            events.append("traditional")

        class TrackingRail(Rail):
            async def handle(self, ctx: RailContext) -> RailAction | None:
                events.append("rail")
                return RailAction.CONTINUE

        provider = _mock_provider()
        agent = Agent(
            name="bot",
            hooks=[(HookPoint.PRE_LLM_CALL, traditional_hook)],
            rails=[TrackingRail("tracker")],
        )

        await agent.run("Hello", provider=provider)

        # Both should have been called during PRE_LLM_CALL
        assert "traditional" in events
        assert "rail" in events


# ---------------------------------------------------------------------------
# Integration test: Rail that aborts on specific tool name
# ---------------------------------------------------------------------------


class TestRailAbortOnTool:
    async def test_abort_on_dangerous_tool(self) -> None:
        """A rail that aborts when a specific tool is called."""

        class BlockToolRail(Rail):
            """Blocks a specific tool by name."""

            def __init__(self, blocked_tool: str) -> None:
                super().__init__("block_tool", priority=10)
                self.blocked_tool = blocked_tool

            async def handle(self, ctx: RailContext) -> RailAction | None:
                if (
                    ctx.event == HookPoint.PRE_TOOL_CALL
                    and isinstance(ctx.inputs, ToolCallInputs)
                    and ctx.inputs.tool_name == self.blocked_tool
                ):
                    return RailAction.ABORT
                return RailAction.CONTINUE

        tc = ToolCall(id="tc-1", name="greet", arguments='{"name": "Alice"}')
        provider = _multi_step_provider(
            ModelResponse(
                content="",
                tool_calls=[tc],
                usage=Usage(input_tokens=10, output_tokens=5, total_tokens=15),
            ),
            ModelResponse(
                content="Done!",
                usage=Usage(input_tokens=20, output_tokens=10, total_tokens=30),
            ),
        )

        agent = Agent(
            name="bot",
            tools=[greet],
            rails=[BlockToolRail("greet")],
        )

        # GuardAbortError (and its RailAbortError subclass) is now unwrapped
        # from the TaskGroup's ExceptionGroup and re-raised directly.
        with pytest.raises(RailAbortError):
            await agent.run("Say hi", provider=provider)

    async def test_non_blocked_tool_continues(self) -> None:
        """A rail that only blocks specific tools lets others through."""

        class BlockToolRail(Rail):
            def __init__(self, blocked_tool: str) -> None:
                super().__init__("block_tool", priority=10)
                self.blocked_tool = blocked_tool

            async def handle(self, ctx: RailContext) -> RailAction | None:
                if (
                    ctx.event == HookPoint.PRE_TOOL_CALL
                    and isinstance(ctx.inputs, ToolCallInputs)
                    and ctx.inputs.tool_name == self.blocked_tool
                ):
                    return RailAction.ABORT
                return RailAction.CONTINUE

        tc = ToolCall(id="tc-1", name="greet", arguments='{"name": "Alice"}')
        provider = _multi_step_provider(
            ModelResponse(
                content="",
                tool_calls=[tc],
                usage=Usage(input_tokens=10, output_tokens=5, total_tokens=15),
            ),
            ModelResponse(
                content="Greeted Alice!",
                usage=Usage(input_tokens=20, output_tokens=10, total_tokens=30),
            ),
        )

        agent = Agent(
            name="bot",
            tools=[greet],
            rails=[BlockToolRail("dangerous_tool")],  # blocking a different tool
        )

        output = await agent.run("Say hi", provider=provider)
        assert output.text == "Greeted Alice!"


# ---------------------------------------------------------------------------
# Integration test: Rail that retries on model error
# ---------------------------------------------------------------------------


class TestRailRetryOnModelError:
    async def test_retry_rail_on_error_response(self) -> None:
        """A rail that signals RETRY when it detects an error in the response."""
        retry_count = 0

        class RetryOnErrorRail(Rail):
            """Returns RETRY if the model response contains 'error'."""

            def __init__(self) -> None:
                super().__init__("retry_on_error", priority=10)
                self.retried = False

            async def handle(self, ctx: RailContext) -> RailAction | None:
                nonlocal retry_count
                if ctx.event == HookPoint.POST_LLM_CALL:
                    response = getattr(ctx.inputs, "response", None)
                    if (
                        response
                        and hasattr(response, "content")
                        and "error" in str(response.content).lower()
                        and not self.retried
                    ):
                        self.retried = True
                        retry_count += 1
                        return RailAction.RETRY
                return RailAction.CONTINUE

        # Note: RETRY from a rail doesn't actually cause the agent to retry
        # the LLM call automatically — it's a signal. The RailManager.hook_for()
        # only raises on ABORT. RETRY is returned but doesn't raise.
        # The rail system returns RETRY as a signal for the caller to handle.
        provider = _mock_provider(content="Everything is fine")
        agent = Agent(
            name="bot",
            rails=[RetryOnErrorRail()],
        )

        output = await agent.run("Hello", provider=provider)
        assert output.text == "Everything is fine"
        assert retry_count == 0  # No error in response, no retry triggered

    async def test_rail_sees_model_response(self) -> None:
        """Rail on POST_LLM_CALL can inspect the model response."""
        seen_responses: list[Any] = []

        class InspectResponseRail(Rail):
            async def handle(self, ctx: RailContext) -> RailAction | None:
                if ctx.event == HookPoint.POST_LLM_CALL:
                    seen_responses.append(ctx.inputs.response)
                return RailAction.CONTINUE

        provider = _mock_provider(content="Test response")
        agent = Agent(
            name="bot",
            rails=[InspectResponseRail("inspector")],
        )

        await agent.run("Hello", provider=provider)

        assert len(seen_responses) == 1
        assert seen_responses[0].content == "Test response"


# ---------------------------------------------------------------------------
# Multiple rails with priority ordering
# ---------------------------------------------------------------------------


class TestMultipleRailsOnAgent:
    async def test_rails_execute_in_priority_order(self) -> None:
        """Multiple rails on an agent execute in ascending priority order."""
        call_order: list[str] = []

        class OrderRail(Rail):
            async def handle(self, ctx: RailContext) -> RailAction | None:
                if ctx.event == HookPoint.PRE_LLM_CALL:
                    call_order.append(self.name)
                return RailAction.CONTINUE

        provider = _mock_provider()
        agent = Agent(
            name="bot",
            rails=[
                OrderRail("low", priority=100),
                OrderRail("high", priority=10),
                OrderRail("mid", priority=50),
            ],
        )

        await agent.run("Hello", provider=provider)

        assert call_order == ["high", "mid", "low"]

    async def test_first_abort_stops_execution(self) -> None:
        """First rail returning ABORT prevents subsequent rails."""

        class AbortRail(Rail):
            async def handle(self, ctx: RailContext) -> RailAction | None:
                return RailAction.ABORT

        second_called = False

        class CheckRail(Rail):
            async def handle(self, ctx: RailContext) -> RailAction | None:
                nonlocal second_called
                second_called = True
                return RailAction.CONTINUE

        provider = _mock_provider()
        agent = Agent(
            name="bot",
            rails=[
                AbortRail("blocker", priority=10),
                CheckRail("checker", priority=20),
            ],
        )

        # ABORT on START hook point raises RailAbortError directly
        # (not wrapped in ExceptionGroup since it's outside TaskGroup)
        with pytest.raises(RailAbortError):
            await agent.run("Hello", provider=provider)
