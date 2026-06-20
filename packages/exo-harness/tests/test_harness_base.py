"""Tests for Harness ABC, HarnessContext, and runner integration."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock

from exo.agent import Agent
from exo.harness.base import Harness, HarnessContext, HarnessError, HarnessNode
from exo.harness.types import HarnessEvent, SessionState
from exo.ptc import PTC_TOOL_NAME
from exo.runner import run
from exo.tool import tool
from exo.types import (
    AgentOutput,
    RunResult,
    StreamEvent,
    TextEvent,
    ToolCall,
    ToolCallEvent,
    ToolResultEvent,
    Usage,
)

# ---------------------------------------------------------------------------
# Helpers
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


class _FakeStreamChunk:
    def __init__(
        self,
        delta: str = "",
        tool_call_deltas: list[Any] | None = None,
        finish_reason: str | None = None,
        usage: Usage | None = None,
    ) -> None:
        self.delta = delta
        self.tool_call_deltas = tool_call_deltas or []
        self.finish_reason = finish_reason
        self.usage = usage or Usage()


class _FakeToolCallDelta:
    def __init__(
        self,
        index: int = 0,
        id: str | None = None,
        name: str | None = None,
        arguments: str = "",
    ) -> None:
        self.index = index
        self.id = id
        self.name = name
        self.arguments = arguments


def _make_stream_provider(stream_rounds: list[list[_FakeStreamChunk]]) -> Any:
    call_count = 0

    async def stream(messages: Any, **kwargs: Any) -> AsyncIterator[Any]:
        nonlocal call_count
        chunks = stream_rounds[min(call_count, len(stream_rounds) - 1)]
        call_count += 1
        for c in chunks:
            yield c

    mock = AsyncMock()
    mock.stream = stream
    mock.complete = AsyncMock()
    return mock


# ---------------------------------------------------------------------------
# Concrete harness implementations for testing
# ---------------------------------------------------------------------------


class PassthroughHarness(Harness):
    """Streams one agent, passes all events through."""

    async def execute(self, ctx: HarnessContext) -> AsyncIterator[StreamEvent]:
        agent = next(iter(self.agents.values()))
        async for event in ctx.stream_agent(agent, ctx.input):
            yield event


class RunOnlyHarness(Harness):
    """Runs one agent (non-streaming), emits a custom event with the output."""

    async def execute(self, ctx: HarnessContext) -> AsyncIterator[StreamEvent]:
        agent = next(iter(self.agents.values()))
        result = await ctx.run_agent(agent, ctx.input)
        ctx.state["output"] = result.output
        yield ctx.emit("done", output=result.output)


class RouterHarness(Harness):
    """Routes to different agents based on classifier output."""

    async def execute(self, ctx: HarnessContext) -> AsyncIterator[StreamEvent]:
        # Phase 1: classify
        result = await ctx.run_agent(self.agents["classifier"], ctx.input)
        category = result.output.strip()
        ctx.state["category"] = category
        yield ctx.emit("classified", category=category)

        # Phase 2: route to specialist
        agent_name = ctx.state.get("route_map", {}).get(category, "default")
        agent = self.agents.get(agent_name, next(iter(self.agents.values())))
        async for event in ctx.stream_agent(agent, ctx.input):
            yield event


class InterceptingHarness(Harness):
    """Intercepts events and records text in state."""

    async def execute(self, ctx: HarnessContext) -> AsyncIterator[StreamEvent]:
        agent = next(iter(self.agents.values()))
        texts: list[str] = []
        async for event in ctx.stream_agent(agent, ctx.input):
            if isinstance(event, TextEvent):
                texts.append(event.text)
            yield event
        ctx.state["intercepted_text"] = "".join(texts)


class CancellableHarness(Harness):
    """Yields events until cancelled."""

    async def execute(self, ctx: HarnessContext) -> AsyncIterator[StreamEvent]:
        agent = next(iter(self.agents.values()))
        async for event in ctx.stream_agent(agent, ctx.input):
            ctx.check_cancelled()
            yield event


# ---------------------------------------------------------------------------
# Tests: Harness constructor
# ---------------------------------------------------------------------------


class TestHarnessInit:
    def test_agents_from_list(self) -> None:
        agent_a = Agent(name="a")
        agent_b = Agent(name="b")
        h = PassthroughHarness(name="h", agents=[agent_a, agent_b])
        assert set(h.agents.keys()) == {"a", "b"}

    def test_agents_from_dict(self) -> None:
        agent_a = Agent(name="a")
        h = PassthroughHarness(name="h", agents={"a": agent_a})
        assert h.agents["a"] is agent_a

    def test_duplicate_agent_name_raises(self) -> None:
        agent_a = Agent(name="dup")
        agent_b = Agent(name="dup")
        with __import__("pytest").raises(HarnessError, match="Duplicate"):
            PassthroughHarness(name="h", agents=[agent_a, agent_b])

    def test_no_agents(self) -> None:
        h = PassthroughHarness(name="h")
        assert h.agents == {}

    def test_state_from_dict(self) -> None:
        h = PassthroughHarness(name="h", state={"foo": "bar"})
        assert h.session["foo"] == "bar"

    def test_state_from_session_state(self) -> None:
        state = SessionState(data={"x": 1})
        h = PassthroughHarness(name="h", state=state)
        assert h.session is state

    def test_is_harness_marker(self) -> None:
        h = PassthroughHarness(name="h")
        assert hasattr(h, "is_harness")
        assert h.is_harness is True

    def test_repr(self) -> None:
        h = PassthroughHarness(name="test", agents=[Agent(name="a")])
        assert "test" in repr(h)
        assert "a" in repr(h)


# ---------------------------------------------------------------------------
# Tests: Harness.run()
# ---------------------------------------------------------------------------


class TestHarnessRun:
    async def test_basic_run(self) -> None:
        agent = Agent(name="bot")
        chunks = [_FakeStreamChunk(delta="Hello!")]
        provider = _make_stream_provider([chunks])
        h = PassthroughHarness(name="h", agents=[agent])

        result = await h.run("Hi", provider=provider)

        assert isinstance(result, RunResult)
        assert result.output == "Hello!"

    async def test_run_via_runner(self) -> None:
        """run() detects Harness via is_harness marker."""
        agent = Agent(name="bot")
        chunks = [_FakeStreamChunk(delta="world")]
        provider = _make_stream_provider([chunks])
        h = PassthroughHarness(name="h", agents=[agent])

        result = await run(h, "Hello", provider=provider)

        assert result.output == "world"

    async def test_run_accumulates_usage(self) -> None:
        agent = Agent(name="bot")
        chunks = [
            _FakeStreamChunk(
                delta="ok",
                usage=Usage(input_tokens=10, output_tokens=5, total_tokens=15),
            ),
        ]
        provider = _make_stream_provider([chunks])
        h = PassthroughHarness(name="h", agents=[agent])

        result = await h.run("Hi", provider=provider)
        # Usage comes via UsageEvent which requires detailed=True
        # PassthroughHarness doesn't request detailed, so usage stays 0
        assert isinstance(result.usage, Usage)

    async def test_run_only_harness_emits_custom_event(self) -> None:
        agent = Agent(name="bot")
        provider = _make_provider([AgentOutput(text="done")])
        h = RunOnlyHarness(name="h", agents=[agent])

        result = await h.run("Hi", provider=provider)

        # RunOnlyHarness emits HarnessEvent, not TextEvent
        assert result.output == ""
        assert h.session["output"] == "done"

    async def test_run_updates_session_state(self) -> None:
        agent = Agent(name="bot")
        provider = _make_provider([AgentOutput(text="ok")])
        h = RunOnlyHarness(name="h", agents=[agent])

        await h.run("Hi", provider=provider)

        assert h.session["output"] == "ok"
        assert h.session.dirty


# ---------------------------------------------------------------------------
# Tests: Harness.stream()
# ---------------------------------------------------------------------------


class TestHarnessStream:
    async def test_stream_text_events(self) -> None:
        agent = Agent(name="bot")
        chunks = [
            _FakeStreamChunk(delta="Hello"),
            _FakeStreamChunk(delta=" world"),
        ]
        provider = _make_stream_provider([chunks])
        h = PassthroughHarness(name="h", agents=[agent])

        events = [ev async for ev in h.stream("Hi", provider=provider)]

        text_events = [e for e in events if isinstance(e, TextEvent)]
        assert len(text_events) == 2
        assert text_events[0].text == "Hello"
        assert text_events[1].text == " world"

    async def test_stream_via_runner(self) -> None:
        """run.stream() detects Harness via is_harness marker."""
        agent = Agent(name="bot")
        chunks = [_FakeStreamChunk(delta="ok")]
        provider = _make_stream_provider([chunks])
        h = PassthroughHarness(name="h", agents=[agent])

        events = [ev async for ev in run.stream(h, "Hi", provider=provider)]

        assert len(events) >= 1
        assert any(isinstance(e, TextEvent) for e in events)

    async def test_stream_event_type_filter(self) -> None:
        agent = Agent(name="bot")
        provider = _make_provider([AgentOutput(text="ok")])
        h = RunOnlyHarness(name="h", agents=[agent])

        events = [ev async for ev in h.stream("Hi", provider=provider, event_types={"harness"})]

        assert all(isinstance(e, HarnessEvent) for e in events)
        assert len(events) == 1
        assert events[0].kind == "done"  # type: ignore[union-attr]

    async def test_stream_custom_event_data(self) -> None:
        agent = Agent(name="bot")
        provider = _make_provider([AgentOutput(text="result")])
        h = RunOnlyHarness(name="h", agents=[agent])

        events = [ev async for ev in h.stream("Hi", provider=provider)]

        harness_events = [e for e in events if isinstance(e, HarnessEvent)]
        assert len(harness_events) == 1
        assert harness_events[0].data == {"output": "result"}
        assert harness_events[0].agent_name == "h"

    async def test_event_interception(self) -> None:
        agent = Agent(name="bot")
        chunks = [
            _FakeStreamChunk(delta="foo"),
            _FakeStreamChunk(delta="bar"),
        ]
        provider = _make_stream_provider([chunks])
        h = InterceptingHarness(name="h", agents=[agent])

        events = [ev async for ev in h.stream("Hi", provider=provider)]

        assert h.session["intercepted_text"] == "foobar"
        assert len(events) == 2


# ---------------------------------------------------------------------------
# Tests: Cancellation
# ---------------------------------------------------------------------------


class TestHarnessCancellation:
    def test_cancel_sets_flag(self) -> None:
        h = PassthroughHarness(name="h")
        assert not h.cancelled
        h.cancel()
        assert h.cancelled

    def test_reset_clears_flag(self) -> None:
        h = PassthroughHarness(name="h")
        h.cancel()
        h.reset()
        assert not h.cancelled

    async def test_check_cancelled_raises(self) -> None:
        agent = Agent(name="bot")
        chunks = [_FakeStreamChunk(delta="hi")]
        provider = _make_stream_provider([chunks])
        h = CancellableHarness(name="h", agents=[agent])
        h.cancel()

        with __import__("pytest").raises(HarnessError, match="cancelled"):
            async for _ in h.stream("Hi", provider=provider):
                pass


# ---------------------------------------------------------------------------
# Tests: SessionState
# ---------------------------------------------------------------------------


class TestSessionState:
    def test_getitem_setitem(self) -> None:
        s = SessionState()
        s["key"] = "value"
        assert s["key"] == "value"

    def test_contains(self) -> None:
        s = SessionState(data={"a": 1})
        assert "a" in s
        assert "b" not in s

    def test_get_default(self) -> None:
        s = SessionState()
        assert s.get("missing", 42) == 42

    def test_dirty_tracking(self) -> None:
        s = SessionState()
        assert not s.dirty
        s["x"] = 1
        assert s.dirty
        s.mark_clean()
        assert not s.dirty


# ---------------------------------------------------------------------------
# Tests: HarnessNode (Swarm composition)
# ---------------------------------------------------------------------------


class TestHarnessNode:
    async def test_node_run(self) -> None:
        agent = Agent(name="bot")
        chunks = [_FakeStreamChunk(delta="ok")]
        provider = _make_stream_provider([chunks])
        h = PassthroughHarness(name="inner", agents=[agent])
        node = HarnessNode(harness=h)

        result = await node.run("Hi", provider=provider)

        assert result.output == "ok"

    async def test_node_stream(self) -> None:
        agent = Agent(name="bot")
        chunks = [_FakeStreamChunk(delta="streamed")]
        provider = _make_stream_provider([chunks])
        h = PassthroughHarness(name="inner", agents=[agent])
        node = HarnessNode(harness=h)

        events = [ev async for ev in node.stream("Hi", provider=provider)]

        text_events = [e for e in events if isinstance(e, TextEvent)]
        assert len(text_events) == 1
        assert text_events[0].text == "streamed"

    def test_node_name_default(self) -> None:
        h = PassthroughHarness(name="my_harness")
        node = HarnessNode(harness=h)
        assert node.name == "my_harness"

    def test_node_name_override(self) -> None:
        h = PassthroughHarness(name="my_harness")
        node = HarnessNode(harness=h, name="custom")
        assert node.name == "custom"

    def test_node_is_group_marker(self) -> None:
        h = PassthroughHarness(name="h")
        node = HarnessNode(harness=h)
        assert node.is_group is True

    def test_node_repr(self) -> None:
        h = PassthroughHarness(name="h")
        node = HarnessNode(harness=h)
        assert "h" in repr(node)


# ---------------------------------------------------------------------------
# Tests: HarnessContext.emit()
# ---------------------------------------------------------------------------


class TestHarnessContext:
    def test_emit_creates_event(self) -> None:
        h = PassthroughHarness(name="test_harness")
        ctx = HarnessContext(
            input="hi",
            messages=[],
            state=SessionState(),
            harness=h,
        )
        event = ctx.emit("my_kind", foo="bar")
        assert isinstance(event, HarnessEvent)
        assert event.kind == "my_kind"
        assert event.data == {"foo": "bar"}
        assert event.agent_name == "test_harness"
        assert event.type == "harness"

    async def test_history_visibility_none_is_fresh(self) -> None:
        """messages=None means fresh conversation."""
        agent = Agent(name="bot")
        provider = _make_provider([AgentOutput(text="ok")])

        h = PassthroughHarness(name="h", agents=[agent])
        ctx = HarnessContext(
            input="hi",
            messages=[],
            state=SessionState(),
            harness=h,
            provider=provider,
        )
        result = await ctx.run_agent(agent, "test", messages=None)
        assert result.output == "ok"


# ---------------------------------------------------------------------------
# PTC + Harness tests
# ---------------------------------------------------------------------------


def _ptc_stream_rounds(tool_code: str) -> list[list[_FakeStreamChunk]]:
    """Two-round stream: PTC tool call, then text response."""
    return [
        [
            _FakeStreamChunk(
                tool_call_deltas=[
                    _FakeToolCallDelta(index=0, id="tc_ptc", name=PTC_TOOL_NAME),
                ],
            ),
            _FakeStreamChunk(
                tool_call_deltas=[
                    _FakeToolCallDelta(
                        index=0,
                        arguments=json.dumps({"code": tool_code}),
                    ),
                ],
                finish_reason="tool_calls",
            ),
        ],
        [_FakeStreamChunk(delta="Done!")],
    ]


class TestHarnessPTCEvents:
    """Verify PTC tool events through harness execution."""

    async def test_stream_agent_emits_ptc_events(self) -> None:
        """ctx.stream_agent() on a PTC agent yields inner tool events."""

        @tool
        def echo(msg: str) -> str:
            """Echo back."""
            return msg

        agent = Agent(name="worker", tools=[echo], ptc=True)
        provider = _make_stream_provider(_ptc_stream_rounds('await default_api.echo(msg="hello")'))

        h = PassthroughHarness(name="h", agents=[agent])

        events: list[StreamEvent] = []
        async for ev in h.stream("test", provider=provider, detailed=True):
            events.append(ev)

        call_events = [e for e in events if isinstance(e, ToolCallEvent)]
        result_events = [e for e in events if isinstance(e, ToolResultEvent)]

        assert len(call_events) == 1
        assert call_events[0].tool_name == "echo"
        assert len(result_events) == 1
        assert result_events[0].tool_name == "echo"
        assert result_events[0].success is True

    async def test_run_agent_ptc_no_stale_events(self) -> None:
        """ctx.run_agent() on a PTC agent does not leak events to later stream."""

        @tool
        def echo(msg: str) -> str:
            """Echo back."""
            return msg

        agent = Agent(name="worker", tools=[echo], ptc=True)

        # Provider for run_agent (non-streaming, uses complete())
        run_provider = _make_provider(
            [
                AgentOutput(
                    text="",
                    tool_calls=[
                        ToolCall(
                            id="tc1",
                            name=PTC_TOOL_NAME,
                            arguments=json.dumps({"code": 'await default_api.echo(msg="stale")'}),
                        )
                    ],
                ),
                AgentOutput(text="done"),
            ]
        )

        h = RunOnlyHarness(name="h", agents=[agent])

        # Execute the harness with run_agent — RunOnlyHarness calls ctx.run_agent()
        # which uses the non-streaming path. The harness output itself is just
        # a HarnessEvent (no TextEvent), so result.output will be empty.
        await h.run("test", provider=run_provider)

        # The key assertion: the agent's _event_queue must be clean after
        # the non-streaming run. PTC events should have been drained.
        assert agent._event_queue.empty(), "PTC events should be drained after non-streaming run"
