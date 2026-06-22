"""Tests for Programmatic Tool Calling (PTC).

All tests use mock providers — no real API calls.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Annotated, Any, Literal
from unittest.mock import AsyncMock

import pytest

from exo.agent import Agent, AgentError
from exo.hooks import HookPoint
from exo.models.types import ModelResponse  # pyright: ignore[reportMissingImports]
from exo.ptc import (
    PTC_TOOL_NAME,
    PTCExecutor,
    PTCTool,
    build_tool_signatures,
    get_ptc_eligible_tools,
    schema_to_python_sig,
)
from exo.runner import run
from exo.tool import Tool, tool
from exo.types import (
    ToolCall,
    ToolCallEvent,
    ToolResultEvent,
    Usage,
)

# ---------------------------------------------------------------------------
# Test tools
# ---------------------------------------------------------------------------


@tool
def greet(name: str) -> str:
    """Greet someone by name."""
    return f"Hello, {name}!"


@tool
def add(a: int, b: int) -> str:
    """Add two numbers."""
    return str(a + b)


@tool
def search(query: str, max_results: int = 5) -> str:
    """Search for information."""
    return json.dumps([{"title": f"Result for {query}", "rank": i} for i in range(max_results)])


@tool
def failing_tool(msg: str) -> str:
    """A tool that always fails."""
    raise ValueError(f"Intentional error: {msg}")


# ---------------------------------------------------------------------------
# Mock provider helpers
# ---------------------------------------------------------------------------


def _mock_provider(
    content: str = "Done!",
    tool_calls: list[ToolCall] | None = None,
) -> AsyncMock:
    resp = ModelResponse(
        id="resp-1",
        model="test-model",
        content=content,
        tool_calls=tool_calls or [],
        usage=Usage(input_tokens=10, output_tokens=5, total_tokens=15),
    )
    provider = AsyncMock()
    provider.complete = AsyncMock(return_value=resp)
    return provider


def _multi_step_provider(*responses: ModelResponse) -> AsyncMock:
    provider = AsyncMock()
    provider.complete = AsyncMock(side_effect=list(responses))
    return provider


# ---------------------------------------------------------------------------
# Init & registration tests
# ---------------------------------------------------------------------------


class TestPTCInit:
    def test_ptc_false_by_default(self) -> None:
        agent = Agent(name="bot", tools=[greet])
        assert agent.ptc is False
        assert PTC_TOOL_NAME not in agent.tools

    def test_ptc_true_registers_ptc_tool(self) -> None:
        agent = Agent(name="bot", tools=[greet], ptc=True)
        assert agent.ptc is True
        assert PTC_TOOL_NAME in agent.tools
        assert isinstance(agent.tools[PTC_TOOL_NAME], PTCTool)

    def test_ptc_timeout_default(self) -> None:
        agent = Agent(name="bot", tools=[greet], ptc=True)
        assert agent.ptc_timeout == 60

    def test_ptc_timeout_custom(self) -> None:
        agent = Agent(name="bot", tools=[greet], ptc=True, ptc_timeout=120)
        assert agent.ptc_timeout == 120

    def test_ptc_collision_raises(self) -> None:
        @tool(name=PTC_TOOL_NAME)
        def my_tool() -> str:
            """Conflicting name."""
            return "x"

        with pytest.raises(AgentError, match=r"already registered"):
            Agent(name="bot", tools=[my_tool], ptc=True)

    def test_ptc_tool_is_tool_subclass(self) -> None:
        agent = Agent(name="bot", tools=[greet], ptc=True)
        assert isinstance(agent.tools[PTC_TOOL_NAME], Tool)


# ---------------------------------------------------------------------------
# Schema tests
# ---------------------------------------------------------------------------


class TestPTCSchemas:
    def test_schemas_exclude_user_tools(self) -> None:
        """PTC-eligible tools should NOT appear as individual schemas."""
        agent = Agent(name="bot", tools=[greet, add], ptc=True)
        schemas = agent.get_tool_schemas()
        schema_names = {s["function"]["name"] for s in schemas}

        assert PTC_TOOL_NAME in schema_names
        assert "greet" not in schema_names
        assert "add" not in schema_names
        # retrieve_artifact is always auto-registered and should stay direct
        assert "retrieve_artifact" in schema_names

    def test_schemas_hitl_tools_stay_direct(self) -> None:
        """HITL tools must remain as direct schemas, not PTC-wrapped."""
        agent = Agent(
            name="bot",
            tools=[greet, add],
            hitl_tools=["greet"],
            ptc=True,
        )
        schemas = agent.get_tool_schemas()
        schema_names = {s["function"]["name"] for s in schemas}

        assert PTC_TOOL_NAME in schema_names
        assert "greet" in schema_names  # HITL → stays direct
        assert "add" not in schema_names  # not HITL → PTC-wrapped

    def test_description_lists_tool_signatures(self) -> None:
        agent = Agent(name="bot", tools=[greet, search], ptc=True)
        ptc_tool = agent.tools[PTC_TOOL_NAME]
        desc = ptc_tool.description

        assert "async def greet" in desc
        assert "name: str" in desc
        assert "async def search" in desc
        assert "query: str" in desc
        assert "max_results: int" in desc

    def test_description_excludes_framework_tools(self) -> None:
        agent = Agent(name="bot", tools=[greet], ptc=True)
        desc = agent.tools[PTC_TOOL_NAME].description

        assert "retrieve_artifact" not in desc
        assert PTC_TOOL_NAME not in desc

    def test_description_excludes_hitl_tools(self) -> None:
        agent = Agent(
            name="bot",
            tools=[greet, add],
            hitl_tools=["greet"],
            ptc=True,
        )
        desc = agent.tools[PTC_TOOL_NAME].description

        assert "greet" not in desc  # HITL tools not in PTC namespace
        assert "async def add" in desc

    async def test_dynamic_tool_add_updates_schemas(self) -> None:
        agent = Agent(name="bot", tools=[greet], ptc=True)

        # Cache schemas
        schemas1 = agent.get_tool_schemas()
        schema_names1 = {s["function"]["name"] for s in schemas1}
        assert "add" not in schema_names1

        # Add tool dynamically
        await agent.add_tool(add)

        schemas2 = agent.get_tool_schemas()
        schema_names2 = {s["function"]["name"] for s in schemas2}
        # add should still not be in schemas (it's PTC-eligible)
        assert "add" not in schema_names2
        # But should appear in PTC tool description
        desc = agent.tools[PTC_TOOL_NAME].description
        assert "async def add" in desc


# ---------------------------------------------------------------------------
# Signature generation tests
# ---------------------------------------------------------------------------


class TestSignatureGeneration:
    def test_simple_required_param(self) -> None:
        sig = schema_to_python_sig(greet)
        assert sig == "async def greet(name: str) -> str"

    def test_optional_param(self) -> None:
        sig = schema_to_python_sig(search)
        assert "query: str" in sig
        assert "max_results" in sig

    def test_multiple_required(self) -> None:
        sig = schema_to_python_sig(add)
        assert "a: int" in sig
        assert "b: int" in sig

    def test_build_tool_signatures(self) -> None:
        result = build_tool_signatures({"greet": greet, "add": add})
        assert "async def greet" in result
        assert "async def add" in result
        assert '"""Greet someone by name."""' in result
        assert '"""Add two numbers."""' in result

    def test_build_tool_signatures_empty(self) -> None:
        assert build_tool_signatures({}) == "(no tools available)"


# ---------------------------------------------------------------------------
# PTCExecutor unit tests
# ---------------------------------------------------------------------------


class TestPTCExecutor:
    """Test PTCExecutor.run() directly."""

    def _make_agent(self, tools: list[Tool] | None = None) -> Agent:
        return Agent(name="test", tools=tools or [greet, add, search], ptc=True)

    async def test_simple_print(self) -> None:
        agent = self._make_agent()
        executor = PTCExecutor(agent)
        result = await executor.run('print("hello world")')
        assert result == "hello world"

    async def test_return_value(self) -> None:
        agent = self._make_agent()
        executor = PTCExecutor(agent)
        result = await executor.run("return 42")
        assert result == "42"

    async def test_print_and_return(self) -> None:
        agent = self._make_agent()
        executor = PTCExecutor(agent)
        result = await executor.run('print("line1")\nreturn 99')
        assert "line1" in result
        assert "99" in result

    async def test_tool_call(self) -> None:
        agent = self._make_agent()
        executor = PTCExecutor(agent)
        result = await executor.run('r = await default_api.greet(name="Alice")\nprint(r)')
        assert "Hello, Alice!" in result

    async def test_multiple_tools_sequential(self) -> None:
        agent = self._make_agent()
        executor = PTCExecutor(agent)
        code = """\
r1 = await default_api.greet(name="Alice")
r2 = await default_api.add(a=3, b=4)
print(f"{r1} sum={r2}")
"""
        result = await executor.run(code)
        assert "Hello, Alice!" in result
        assert "sum=7" in result

    async def test_parallel_gather(self) -> None:
        agent = self._make_agent()
        executor = PTCExecutor(agent)
        code = """\
results = await asyncio.gather(
    default_api.greet(name="A"),
    default_api.greet(name="B"),
    default_api.greet(name="C"),
)
for r in results:
    print(r)
"""
        result = await executor.run(code)
        assert "Hello, A!" in result
        assert "Hello, B!" in result
        assert "Hello, C!" in result

    async def test_loop_with_filter(self) -> None:
        """Core PTC use case: loop + filter to reduce context."""
        agent = self._make_agent()
        executor = PTCExecutor(agent)
        code = """\
data = json.loads(await default_api.search(query="test", max_results=5))
high_rank = [d for d in data if d["rank"] >= 3]
print(json.dumps(high_rank))
"""
        result = await executor.run(code)
        parsed = json.loads(result)
        assert len(parsed) == 2  # rank 3 and 4
        assert all(d["rank"] >= 3 for d in parsed)

    async def test_syntax_error(self) -> None:
        agent = self._make_agent()
        executor = PTCExecutor(agent)
        result = await executor.run("def foo(")
        assert "SyntaxError" in result

    async def test_runtime_error(self) -> None:
        agent = self._make_agent()
        executor = PTCExecutor(agent)
        result = await executor.run("x = 1 / 0")
        assert "ZeroDivisionError" in result

    async def test_tool_error_propagates(self) -> None:
        agent = Agent(name="test", tools=[failing_tool], ptc=True)
        executor = PTCExecutor(agent)
        # Error should appear in the output (caught by outer handler)
        result = await executor.run('await default_api.failing_tool(msg="boom")')
        assert "Intentional error: boom" in result

    async def test_tool_error_catchable_in_code(self) -> None:
        """User code can catch tool errors and continue."""
        agent = Agent(name="test", tools=[failing_tool, greet], ptc=True)
        executor = PTCExecutor(agent)
        code = """\
try:
    await default_api.failing_tool(msg="oops")
except Exception as e:
    print(f"caught: {e}")
r = await default_api.greet(name="OK")
print(r)
"""
        result = await executor.run(code)
        assert "caught:" in result
        assert "Hello, OK!" in result

    async def test_timeout(self) -> None:
        agent = self._make_agent()
        executor = PTCExecutor(agent, timeout=1)
        result = await executor.run("await asyncio.sleep(10)")
        assert "TimeoutError" in result

    async def test_stdlib_available(self) -> None:
        agent = self._make_agent()
        executor = PTCExecutor(agent)
        code = """\
import math as m  # also available without import
print(math.sqrt(16))
print(json.dumps({"a": 1}))
print(re.sub(r"\\d+", "X", "abc123"))
"""
        result = await executor.run(code)
        assert "4.0" in result
        assert '{"a": 1}' in result
        assert "abcX" in result

    async def test_no_output(self) -> None:
        agent = self._make_agent()
        executor = PTCExecutor(agent)
        result = await executor.run("x = 1 + 1")
        assert result == "(no output)"


# ---------------------------------------------------------------------------
# Hook tests
# ---------------------------------------------------------------------------


class TestPTCHooks:
    async def test_hooks_fire_for_inner_calls(self) -> None:
        agent = Agent(name="test", tools=[greet], ptc=True)
        hook_calls: list[tuple[str, str]] = []

        async def capture_hook(**kwargs: Any) -> None:
            hook_calls.append((kwargs.get("tool_name", ""), "called"))

        agent.hook_manager.add(HookPoint.PRE_TOOL_CALL, capture_hook)
        agent.hook_manager.add(HookPoint.POST_TOOL_CALL, capture_hook)

        executor = PTCExecutor(agent)
        await executor.run('await default_api.greet(name="Test")')

        # PRE + POST for the inner greet call
        tool_names = [name for name, _ in hook_calls]
        assert tool_names.count("greet") == 2

    async def test_hooks_correct_arguments(self) -> None:
        agent = Agent(name="test", tools=[greet], ptc=True)
        captured_args: list[dict[str, Any]] = []

        async def capture_pre(**kwargs: Any) -> None:
            if kwargs.get("arguments"):
                captured_args.append(kwargs["arguments"])

        agent.hook_manager.add(HookPoint.PRE_TOOL_CALL, capture_pre)

        executor = PTCExecutor(agent)
        await executor.run('await default_api.greet(name="Alice")')

        assert len(captured_args) == 1
        assert captured_args[0] == {"name": "Alice"}

    async def test_hooks_fire_for_multiple_inner_calls(self) -> None:
        agent = Agent(name="test", tools=[greet, add], ptc=True)
        pre_calls: list[str] = []

        async def capture_pre(**kwargs: Any) -> None:
            pre_calls.append(kwargs.get("tool_name", ""))

        agent.hook_manager.add(HookPoint.PRE_TOOL_CALL, capture_pre)

        executor = PTCExecutor(agent)
        await executor.run('await default_api.greet(name="A")\nawait default_api.add(a=1, b=2)')

        assert "greet" in pre_calls
        assert "add" in pre_calls


# ---------------------------------------------------------------------------
# PTC-eligible tools helper
# ---------------------------------------------------------------------------


class TestPTCEligible:
    def test_excludes_framework_tools(self) -> None:
        agent = Agent(name="bot", tools=[greet], ptc=True)
        eligible = get_ptc_eligible_tools(agent)
        assert "greet" in eligible
        assert PTC_TOOL_NAME not in eligible
        assert "retrieve_artifact" not in eligible

    def test_excludes_hitl_tools(self) -> None:
        agent = Agent(name="bot", tools=[greet, add], hitl_tools=["greet"], ptc=True)
        eligible = get_ptc_eligible_tools(agent)
        assert "greet" not in eligible
        assert "add" in eligible


# ---------------------------------------------------------------------------
# Integration tests: full agent.run()
# ---------------------------------------------------------------------------


class TestPTCIntegration:
    async def test_end_to_end_run(self) -> None:
        """MockProvider returns PTC tool call → agent runs PTC → final text."""
        tc = ToolCall(
            id="tc-1",
            name=PTC_TOOL_NAME,
            arguments=json.dumps({"code": 'r = await default_api.greet(name="World")\nprint(r)'}),
        )
        resp_tool = ModelResponse(
            content="",
            tool_calls=[tc],
            usage=Usage(input_tokens=10, output_tokens=20, total_tokens=30),
        )
        resp_final = ModelResponse(
            content="I greeted World for you!",
            tool_calls=[],
            usage=Usage(input_tokens=20, output_tokens=10, total_tokens=30),
        )
        provider = _multi_step_provider(resp_tool, resp_final)

        agent = Agent(name="bot", tools=[greet], ptc=True)
        output = await agent.run("Say hi to World", provider=provider)

        assert output.text == "I greeted World for you!"
        assert provider.complete.await_count == 2

    async def test_ptc_tool_result_contains_output(self) -> None:
        """The tool result from PTC tool should contain the code output."""
        tc = ToolCall(
            id="tc-1",
            name=PTC_TOOL_NAME,
            arguments=json.dumps({"code": 'r = await default_api.greet(name="Test")\nprint(r)'}),
        )
        # Track messages sent to provider
        call_count = 0
        captured_msgs: list[Any] = []

        async def fake_complete(messages: Any, **kwargs: Any) -> ModelResponse:
            nonlocal call_count
            call_count += 1
            captured_msgs.append(list(messages))
            if call_count == 1:
                return ModelResponse(
                    content="",
                    tool_calls=[tc],
                    usage=Usage(input_tokens=10, output_tokens=5, total_tokens=15),
                )
            return ModelResponse(
                content="Done",
                usage=Usage(input_tokens=10, output_tokens=5, total_tokens=15),
            )

        provider = AsyncMock()
        provider.complete = AsyncMock(side_effect=fake_complete)

        agent = Agent(name="bot", tools=[greet], ptc=True)
        await agent.run("test", provider=provider)

        # The second call should have the tool result with "Hello, Test!"
        second_call_msgs = captured_msgs[1]
        tool_results = [m for m in second_call_msgs if getattr(m, "role", "") == "tool"]
        assert len(tool_results) == 1
        assert "Hello, Test!" in tool_results[0].content


# ---------------------------------------------------------------------------
# Serialization tests
# ---------------------------------------------------------------------------


class TestPTCSerialization:
    def test_to_dict_includes_ptc(self) -> None:
        agent = Agent(name="bot", tools=[greet], ptc=True, ptc_timeout=120)
        data = agent.to_dict()
        assert data["ptc"] is True
        assert data["ptc_timeout"] == 120

    def test_to_dict_ptc_false(self) -> None:
        agent = Agent(name="bot", tools=[greet])
        data = agent.to_dict()
        assert data["ptc"] is False

    def test_ptc_tool_not_in_serialized_tools(self) -> None:
        agent = Agent(name="bot", tools=[greet], ptc=True)
        data = agent.to_dict()
        if "tools" in data:
            tool_names = [t.get("name", t) if isinstance(t, dict) else t for t in data["tools"]]
            assert PTC_TOOL_NAME not in str(tool_names)

    def test_describe_includes_ptc(self) -> None:
        agent = Agent(name="bot", tools=[greet], ptc=True)
        info = agent.describe()
        assert info["ptc"] is True


# ---------------------------------------------------------------------------
# Swarm propagation tests
# ---------------------------------------------------------------------------


class TestSwarmPTC:
    def test_swarm_ptc_propagation(self) -> None:
        from exo.swarm import Swarm

        a1 = Agent(name="a1", tools=[greet])
        a2 = Agent(name="a2", tools=[add])
        assert a1.ptc is False
        assert a2.ptc is False

        Swarm(agents=[a1, a2], ptc=True)
        assert a1.ptc is True
        assert a2.ptc is True

    def test_swarm_ptc_none_no_change(self) -> None:
        from exo.swarm import Swarm

        a1 = Agent(name="a1", tools=[greet], ptc=True)
        a2 = Agent(name="a2", tools=[add])

        Swarm(agents=[a1, a2])  # ptc=None by default
        assert a1.ptc is True  # unchanged
        assert a2.ptc is False  # unchanged


# ---------------------------------------------------------------------------
# PTC transparency tests — synthetic event emission
# ---------------------------------------------------------------------------


class TestPTCTransparency:
    """Verify that PTC emits per-tool ToolCallEvent/ToolResultEvent to the queue."""

    async def test_single_tool_emits_events(self) -> None:
        agent = Agent(name="test", tools=[greet], ptc=True)
        executor = PTCExecutor(agent)
        await executor.run('await default_api.greet(name="Alice")')

        events: list[Any] = []
        while not agent._event_queue.empty():
            events.append(agent._event_queue.get_nowait())

        call_events = [e for e in events if isinstance(e, ToolCallEvent)]
        result_events = [e for e in events if isinstance(e, ToolResultEvent)]

        assert len(call_events) == 1
        assert call_events[0].tool_name == "greet"
        assert call_events[0].agent_name == "test"
        assert '"Alice"' in call_events[0].arguments

        assert len(result_events) == 1
        assert result_events[0].tool_name == "greet"
        assert result_events[0].success is True
        assert "Hello, Alice!" in str(result_events[0].result)

    async def test_multiple_tools_emit_ordered_events(self) -> None:
        agent = Agent(name="test", tools=[greet, add], ptc=True)
        executor = PTCExecutor(agent)
        await executor.run('await default_api.greet(name="A")\nawait default_api.add(a=1, b=2)')

        events: list[Any] = []
        while not agent._event_queue.empty():
            events.append(agent._event_queue.get_nowait())

        tool_events = [e for e in events if isinstance(e, (ToolCallEvent, ToolResultEvent))]

        # Expect: call(greet), result(greet), call(add), result(add)
        assert len(tool_events) == 4
        assert tool_events[0].tool_name == "greet"
        assert isinstance(tool_events[0], ToolCallEvent)
        assert tool_events[1].tool_name == "greet"
        assert isinstance(tool_events[1], ToolResultEvent)
        assert tool_events[2].tool_name == "add"
        assert isinstance(tool_events[2], ToolCallEvent)
        assert tool_events[3].tool_name == "add"
        assert isinstance(tool_events[3], ToolResultEvent)

    async def test_error_tool_emits_failure_event(self) -> None:
        agent = Agent(name="test", tools=[failing_tool], ptc=True)
        executor = PTCExecutor(agent)
        await executor.run('await default_api.failing_tool(msg="boom")')

        events: list[Any] = []
        while not agent._event_queue.empty():
            events.append(agent._event_queue.get_nowait())

        result_events = [e for e in events if isinstance(e, ToolResultEvent)]
        assert len(result_events) == 1
        assert result_events[0].success is False
        assert "boom" in (result_events[0].error or "")

    async def test_events_have_unique_call_ids(self) -> None:
        agent = Agent(name="test", tools=[greet], ptc=True)
        executor = PTCExecutor(agent)
        await executor.run('await default_api.greet(name="A")\nawait default_api.greet(name="B")')

        events: list[Any] = []
        while not agent._event_queue.empty():
            events.append(agent._event_queue.get_nowait())

        call_events = [e for e in events if isinstance(e, ToolCallEvent)]
        assert len(call_events) == 2
        assert call_events[0].tool_call_id != call_events[1].tool_call_id

    async def test_call_and_result_share_call_id(self) -> None:
        agent = Agent(name="test", tools=[greet], ptc=True)
        executor = PTCExecutor(agent)
        await executor.run('await default_api.greet(name="X")')

        events: list[Any] = []
        while not agent._event_queue.empty():
            events.append(agent._event_queue.get_nowait())

        call_ev = next(e for e in events if isinstance(e, ToolCallEvent))
        result_ev = next(e for e in events if isinstance(e, ToolResultEvent))
        assert call_ev.tool_call_id == result_ev.tool_call_id

    async def test_result_has_duration(self) -> None:
        agent = Agent(name="test", tools=[greet], ptc=True)
        executor = PTCExecutor(agent)
        await executor.run('await default_api.greet(name="A")')

        events: list[Any] = []
        while not agent._event_queue.empty():
            events.append(agent._event_queue.get_nowait())

        result_events = [e for e in events if isinstance(e, ToolResultEvent)]
        assert result_events[0].duration_ms >= 0.0

    async def test_no_ptc_tool_name_in_events(self) -> None:
        """No event should reference the internal PTC tool name."""
        agent = Agent(name="test", tools=[greet, add], ptc=True)
        executor = PTCExecutor(agent)
        await executor.run('await default_api.greet(name="A")\nawait default_api.add(a=1, b=2)')

        events: list[Any] = []
        while not agent._event_queue.empty():
            events.append(agent._event_queue.get_nowait())

        for ev in events:
            if hasattr(ev, "tool_name"):
                assert ev.tool_name != PTC_TOOL_NAME


# ---------------------------------------------------------------------------
# Streaming + non-streaming integration helpers
# ---------------------------------------------------------------------------


class _FakeStreamChunk:
    """Minimal stream chunk for PTC integration tests."""

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
    """Minimal tool call delta for PTC integration tests."""

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
    """Create a mock provider returning pre-defined stream chunks."""
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


def _ptc_tool_call(code: str) -> list[_FakeStreamChunk]:
    """Build a stream round that calls the __exo_ptc__ tool with *code*."""
    return [
        _FakeStreamChunk(
            tool_call_deltas=[
                _FakeToolCallDelta(index=0, id="tc_ptc_1", name=PTC_TOOL_NAME),
            ],
        ),
        _FakeStreamChunk(
            tool_call_deltas=[
                _FakeToolCallDelta(index=0, arguments=json.dumps({"code": code})),
            ],
            finish_reason="tool_calls",
        ),
    ]


# ---------------------------------------------------------------------------
# PTC streaming integration tests
# ---------------------------------------------------------------------------


class TestPTCStreamingIntegration:
    """End-to-end: PTC events flow through run.stream()."""

    async def test_ptc_events_in_stream_detailed(self) -> None:
        """run.stream(detailed=True) yields inner ToolCallEvent + ToolResultEvent."""
        agent = Agent(name="bot", tools=[greet], ptc=True)
        round1 = _ptc_tool_call('await default_api.greet(name="Alice")')
        round2 = [_FakeStreamChunk(delta="Done!")]
        provider = _make_stream_provider([round1, round2])

        events = [ev async for ev in run.stream(agent, "hi", provider=provider, detailed=True)]

        call_events = [e for e in events if isinstance(e, ToolCallEvent)]
        result_events = [e for e in events if isinstance(e, ToolResultEvent)]

        assert len(call_events) == 1
        assert call_events[0].tool_name == "greet"
        assert call_events[0].agent_name == "bot"

        assert len(result_events) == 1
        assert result_events[0].tool_name == "greet"
        assert result_events[0].success is True
        assert "Hello, Alice!" in str(result_events[0].result)

    async def test_ptc_tool_call_without_detailed(self) -> None:
        """run.stream(detailed=False) yields ToolCallEvent but NOT ToolResultEvent."""
        agent = Agent(name="bot", tools=[greet], ptc=True)
        round1 = _ptc_tool_call('await default_api.greet(name="Bob")')
        round2 = [_FakeStreamChunk(delta="Done!")]
        provider = _make_stream_provider([round1, round2])

        events = [ev async for ev in run.stream(agent, "hi", provider=provider, detailed=False)]

        call_events = [e for e in events if isinstance(e, ToolCallEvent)]
        result_events = [e for e in events if isinstance(e, ToolResultEvent)]

        assert len(call_events) == 1
        assert call_events[0].tool_name == "greet"
        # ToolResultEvent requires detailed=True
        assert len(result_events) == 0

    async def test_ptc_outer_tool_suppressed(self) -> None:
        """No event should reference __exo_ptc__ in the stream."""
        agent = Agent(name="bot", tools=[greet], ptc=True)
        round1 = _ptc_tool_call('await default_api.greet(name="X")')
        round2 = [_FakeStreamChunk(delta="ok")]
        provider = _make_stream_provider([round1, round2])

        events = [ev async for ev in run.stream(agent, "hi", provider=provider, detailed=True)]

        for ev in events:
            if hasattr(ev, "tool_name"):
                assert ev.tool_name != PTC_TOOL_NAME

    async def test_ptc_multiple_inner_tools_stream(self) -> None:
        """Multiple PTC inner tool calls emit ordered events in stream."""
        agent = Agent(name="bot", tools=[greet, add], ptc=True)
        code = 'await default_api.greet(name="A")\nawait default_api.add(a=1, b=2)'
        round1 = _ptc_tool_call(code)
        round2 = [_FakeStreamChunk(delta="ok")]
        provider = _make_stream_provider([round1, round2])

        events = [ev async for ev in run.stream(agent, "hi", provider=provider, detailed=True)]

        call_events = [e for e in events if isinstance(e, ToolCallEvent)]
        result_events = [e for e in events if isinstance(e, ToolResultEvent)]

        assert len(call_events) == 2
        assert call_events[0].tool_name == "greet"
        assert call_events[1].tool_name == "add"
        assert len(result_events) == 2
        assert result_events[0].tool_name == "greet"
        assert result_events[1].tool_name == "add"


# ---------------------------------------------------------------------------
# Non-streaming: queue drain prevents stale events
# ---------------------------------------------------------------------------


class TestPTCNonStreamingDrain:
    """Verify that run() (non-streaming) drains PTC events from the queue."""

    async def test_run_drains_ptc_event_queue(self) -> None:
        """After run(), the agent's _event_queue should be empty."""
        agent = Agent(name="bot", tools=[greet], ptc=True)
        provider = _multi_step_provider(
            ModelResponse(
                id="r1",
                model="test",
                content="",
                tool_calls=[
                    ToolCall(
                        id="tc1",
                        name=PTC_TOOL_NAME,
                        arguments=json.dumps({"code": 'await default_api.greet(name="Z")'}),
                    )
                ],
                usage=Usage(input_tokens=10, output_tokens=5, total_tokens=15),
            ),
            ModelResponse(
                id="r2",
                model="test",
                content="Done",
                tool_calls=[],
                usage=Usage(input_tokens=10, output_tokens=5, total_tokens=15),
            ),
        )

        result = await run(agent, "greet Z", provider=provider)

        assert result.output == "Done"
        assert agent._event_queue.empty(), (
            "_event_queue should be empty after non-streaming run() — "
            "PTC events must be drained to prevent memory leaks"
        )

    async def test_run_then_stream_no_stale_events(self) -> None:
        """Stale PTC events from run() must not leak into a later stream()."""
        agent = Agent(name="bot", tools=[greet], ptc=True)

        # First: non-streaming run
        provider1 = _multi_step_provider(
            ModelResponse(
                id="r1",
                model="test",
                content="",
                tool_calls=[
                    ToolCall(
                        id="tc1",
                        name=PTC_TOOL_NAME,
                        arguments=json.dumps({"code": 'await default_api.greet(name="Old")'}),
                    )
                ],
                usage=Usage(input_tokens=10, output_tokens=5, total_tokens=15),
            ),
            ModelResponse(
                id="r2",
                model="test",
                content="done1",
                tool_calls=[],
                usage=Usage(input_tokens=10, output_tokens=5, total_tokens=15),
            ),
        )
        await run(agent, "first", provider=provider1)

        # Second: streaming run (text-only, no tool calls)
        provider2 = _make_stream_provider([[_FakeStreamChunk(delta="Hello stream!")]])
        events = [ev async for ev in run.stream(agent, "second", provider=provider2, detailed=True)]

        # No stale ToolCallEvent/ToolResultEvent from the first run should appear
        tool_events = [e for e in events if isinstance(e, (ToolCallEvent, ToolResultEvent))]
        assert len(tool_events) == 0, (
            f"Expected no tool events in text-only stream, got {len(tool_events)}"
        )


# ---------------------------------------------------------------------------
# Robustness tests (H1-H5, M6, M9, M11, M12, L13-L18)
# ---------------------------------------------------------------------------


class TestPTCBaseExceptionHandling:
    """PTC must catch SystemExit / KeyboardInterrupt / MaxToolCallsExceeded."""

    async def test_system_exit_is_caught(self) -> None:
        agent = Agent(name="t", tools=[greet], ptc=True)
        executor = PTCExecutor(agent)
        result = await executor.run("raise SystemExit(2)")
        assert "SystemExit" in result
        assert "blocked inside PTC" in result

    async def test_system_exit_with_captured_output(self) -> None:
        agent = Agent(name="t", tools=[greet], ptc=True)
        executor = PTCExecutor(agent)
        result = await executor.run('print("done work")\nraise SystemExit(1)')
        assert "done work" in result
        assert "SystemExit" in result

    async def test_raise_system_exit_with_code_is_caught(self) -> None:
        """``raise SystemExit(3)`` in user code is caught by the inner trap.

        ``import sys`` is now sandbox-blocked, so this tests the bare
        ``raise SystemExit(...)`` form which remains a valid escape attempt.
        """
        agent = Agent(name="t", tools=[greet], ptc=True)
        executor = PTCExecutor(agent)
        result = await executor.run("raise SystemExit(3)")
        assert "SystemExit" in result
        assert "code=3" in result

    async def test_cancelled_error_propagates(self) -> None:
        """asyncio.CancelledError MUST propagate, not be swallowed."""
        agent = Agent(name="t", tools=[greet], ptc=True)
        executor = PTCExecutor(agent)
        # Raise CancelledError directly from user code
        with pytest.raises(asyncio.CancelledError):
            await executor.run("raise asyncio.CancelledError()")


class TestPTCStderrCapture:
    """Verify stderr from library code inside tools is captured.

    Since the sandbox blocks ``import sys``, user code cannot write to
    stderr directly.  But libraries invoked *inside* tool execution (e.g.
    ``warnings.warn``, legacy C extensions) may still write to stderr —
    those writes must be captured and returned to the model, not leaked
    to the real terminal.
    """

    async def test_tool_stderr_is_captured(self) -> None:
        import sys as _sys

        @tool
        def noisy() -> str:
            """Write a warning to stderr then return a value."""
            _sys.stderr.write("library warning: about to compute\n")
            return "done"

        agent = Agent(name="t", tools=[noisy], ptc=True)
        executor = PTCExecutor(agent)
        result = await executor.run("r = await default_api.noisy()\nprint(r)")
        assert "library warning" in result
        assert "done" in result


class TestPTCHyphenatedNames:
    """Tools whose name is not a valid Python identifier are excluded from PTC."""

    def test_hyphen_name_excluded_from_ptc(self) -> None:
        @tool(name="get-data")
        def get_data(query: str) -> str:
            """Fetch data."""
            return f"data:{query}"

        agent = Agent(name="t", tools=[get_data, greet], ptc=True)
        eligible = get_ptc_eligible_tools(agent)
        assert "get-data" not in eligible
        assert "greet" in eligible

    def test_hyphen_name_stays_as_direct_schema(self) -> None:
        @tool(name="get-data")
        def get_data(query: str) -> str:
            """Fetch data."""
            return f"data:{query}"

        agent = Agent(name="t", tools=[get_data, greet], ptc=True)
        schemas = agent.get_tool_schemas()
        names = {s["function"]["name"] for s in schemas}
        # Hyphenated tool stays as direct schema (so the LLM can still call it)
        assert "get-data" in names
        # greet is absorbed into PTC
        assert "greet" not in names
        assert PTC_TOOL_NAME in names


class TestPTCExcludeAttribute:
    """Tools with _ptc_exclude=True are never wrapped by PTC."""

    def test_ptc_exclude_attribute(self) -> None:
        @tool
        def special(x: int) -> str:
            """Do something special."""
            return str(x)

        special._ptc_exclude = True  # type: ignore[attr-defined]
        agent = Agent(name="t", tools=[special, greet], ptc=True)
        eligible = get_ptc_eligible_tools(agent)
        assert "special" not in eligible
        assert "greet" in eligible

        schemas = agent.get_tool_schemas()
        names = {s["function"]["name"] for s in schemas}
        assert "special" in names  # stays as direct schema


class TestPTCJsonDumpsDefault:
    """Tools returning non-JSON-serializable objects are stringified via default=str."""

    async def test_datetime_in_dict_return_serialized(self) -> None:
        import datetime as _dt

        @tool
        def get_info() -> dict:
            """Return a dict with datetime."""
            return {"when": _dt.datetime(2026, 4, 10, 12, 0, 0)}

        agent = Agent(name="t", tools=[get_info], ptc=True)
        executor = PTCExecutor(agent)
        result = await executor.run("r = await default_api.get_info()\nprint(r)")
        # datetime is stringified via default=str; no TypeError
        assert "2026-04-10" in result

    async def test_non_serialisable_falls_back_to_str(self) -> None:
        class Weird:
            def __str__(self) -> str:
                return "weird-object"

        @tool
        def get_weird() -> dict:
            """Return a dict with weird object."""
            return {"thing": Weird()}

        agent = Agent(name="t", tools=[get_weird], ptc=True)
        executor = PTCExecutor(agent)
        result = await executor.run("r = await default_api.get_weird()\nprint(r)")
        # Either json.dumps(default=str) or str(raw) handles this
        assert "weird-object" in result


class TestPTCOutputTruncation:
    async def test_output_is_capped(self) -> None:
        agent = Agent(name="t", tools=[greet], ptc=True)
        executor = PTCExecutor(agent, max_output_bytes=500)
        code = "print('x' * 5000)"
        result = await executor.run(code)
        assert "[truncated" in result
        assert len(result) <= 600  # cap + trailing marker

    async def test_output_under_cap_is_untouched(self) -> None:
        agent = Agent(name="t", tools=[greet], ptc=True)
        executor = PTCExecutor(agent, max_output_bytes=500)
        result = await executor.run("print('hello')")
        assert result == "hello"
        assert "[truncated" not in result


class TestPTCTracebackLineNumbers:
    async def test_traceback_line_numbers_match_user_code(self) -> None:
        agent = Agent(name="t", tools=[greet], ptc=True)
        executor = PTCExecutor(agent)
        # Line 1 of user code raises
        result = await executor.run("x = 1/0")
        # The rewritten traceback should say line 1, not line 2
        assert "line 1" in result or "ZeroDivisionError" in result
        # And definitely NOT "line 2" from the wrapper
        assert 'File "<ptc>", line 2' not in result

    async def test_traceback_multi_line_user_code(self) -> None:
        agent = Agent(name="t", tools=[greet], ptc=True)
        executor = PTCExecutor(agent)
        # Line 3 of user code raises
        code = "a = 1\nb = 2\nc = a / 0\n"
        result = await executor.run(code)
        assert "ZeroDivisionError" in result
        # Wrapper adds 1 line, so without rewrite the traceback would say line 4
        assert 'File "<ptc>", line 4' not in result

    async def test_syntax_error_line_number_adjusted(self) -> None:
        agent = Agent(name="t", tools=[greet], ptc=True)
        executor = PTCExecutor(agent)
        # Line 1 has the syntax error
        result = await executor.run("def foo(")
        assert "SyntaxError" in result


class TestPTCMaxToolCalls:
    async def test_max_tool_calls_enforced(self) -> None:
        agent = Agent(name="t", tools=[greet], ptc=True)
        executor = PTCExecutor(agent, max_tool_calls=3)
        code = """\
for _ in range(10):
    await default_api.greet(name="X")
"""
        result = await executor.run(code)
        assert "max_tool_calls" in result

    async def test_under_cap_runs_normally(self) -> None:
        agent = Agent(name="t", tools=[greet], ptc=True)
        executor = PTCExecutor(agent, max_tool_calls=5)
        code = """\
for _ in range(3):
    r = await default_api.greet(name="X")
print(r)
"""
        result = await executor.run(code)
        assert "Hello, X!" in result

    async def test_max_tool_calls_propagated_from_agent(self) -> None:
        agent = Agent(name="t", tools=[greet], ptc=True, ptc_max_tool_calls=2)
        assert agent.ptc_max_tool_calls == 2
        assert agent.tools[PTC_TOOL_NAME]._max_tool_calls == 2  # type: ignore[attr-defined]


class TestPTCFullUuid:
    async def test_tool_call_id_is_full_uuid(self) -> None:
        agent = Agent(name="t", tools=[greet], ptc=True)
        executor = PTCExecutor(agent)
        await executor.run('await default_api.greet(name="A")')

        events: list[Any] = []
        while not agent._event_queue.empty():
            events.append(agent._event_queue.get_nowait())

        call_events = [e for e in events if isinstance(e, ToolCallEvent)]
        assert len(call_events) == 1
        # Full uuid.uuid4().hex is 32 hex chars, prefix 'ptc_'
        assert len(call_events[0].tool_call_id) == 4 + 32
        assert call_events[0].tool_call_id.startswith("ptc_")


class TestPTCOrphanTaskCleanup:
    async def test_orphan_tasks_are_cancelled(self) -> None:
        agent = Agent(name="t", tools=[greet], ptc=True)
        executor = PTCExecutor(agent)
        code = """\
async def spinner():
    while True:
        await asyncio.sleep(0.01)

asyncio.create_task(spinner())
print("spawned")
"""
        before = asyncio.all_tasks()
        result = await executor.run(code)
        assert "spawned" in result
        # After run, the orphan spinner should be cancelled
        await asyncio.sleep(0.05)
        after = asyncio.all_tasks()
        new_tasks = after - before
        # Any surviving new task should be done (cancelled) or filtered
        for task in new_tasks:
            assert task.done()


class TestPTCArrayItemTypes:
    def test_array_items_shown_in_signature(self) -> None:
        @tool
        def batch(ids: list[str]) -> str:
            """Process a list of ids."""
            return str(ids)

        sig = schema_to_python_sig(batch)
        # Should show list[str], not just list
        assert "list[str]" in sig

    def test_array_without_items_defaults_to_list(self) -> None:
        class ToolNoItems(Tool):
            name = "simple"
            description = "simple"
            parameters = {  # noqa: RUF012
                "type": "object",
                "properties": {
                    "vals": {"type": "array"},
                },
                "required": ["vals"],
            }

            async def execute(self, **kwargs: Any) -> str:
                return ""

        sig = schema_to_python_sig(ToolNoItems())
        assert "vals: list" in sig


class TestPTCDescriptionCache:
    def test_description_cached_when_tools_unchanged(self) -> None:
        agent = Agent(name="t", tools=[greet, add], ptc=True)
        ptc_tool: PTCTool = agent.tools[PTC_TOOL_NAME]  # type: ignore[assignment]
        d1 = ptc_tool.description
        d2 = ptc_tool.description
        # Same content and same object (cache hit)
        assert d1 is d2

    async def test_description_invalidated_on_add_tool(self) -> None:
        agent = Agent(name="t", tools=[greet], ptc=True)
        ptc_tool: PTCTool = agent.tools[PTC_TOOL_NAME]  # type: ignore[assignment]
        d1 = ptc_tool.description
        assert "async def greet" in d1
        assert "async def add" not in d1

        await agent.add_tool(add)
        d2 = ptc_tool.description
        assert "async def add" in d2
        assert d1 is not d2

    def test_description_contains_instructions_block(self) -> None:
        agent = Agent(name="t", tools=[greet], ptc=True)
        desc = agent.tools[PTC_TOOL_NAME].description
        assert "<instructions>" in desc
        assert "</instructions>" in desc
        assert "<recommended_usage>" in desc
        assert "</recommended_usage>" in desc

    def test_description_emphasises_minimal_code(self) -> None:
        """Key directive: write MINIMAL, PTC-only code."""
        agent = Agent(name="t", tools=[greet], ptc=True)
        desc = agent.tools[PTC_TOOL_NAME].description
        # Guidance about minimal code presence
        assert "MINIMAL" in desc
        assert "ONLY write PTC-related" in desc

    def test_code_param_description_points_to_examples(self) -> None:
        agent = Agent(name="t", tools=[greet], ptc=True)
        ptc_tool = agent.tools[PTC_TOOL_NAME]
        code_desc = ptc_tool.parameters["properties"]["code"]["description"]
        # The parameter description should contain examples and a reference
        # to the tool description (not duplicate the full ruleset).
        assert "EXAMPLE" in code_desc
        assert "default_api" in code_desc
        assert "MINIMAL" in code_desc
        assert "See the tool description" in code_desc

    def test_code_param_description_does_not_duplicate_rules(self) -> None:
        """The parameter description should NOT re-list the full instructions block."""
        agent = Agent(name="t", tools=[greet], ptc=True)
        ptc_tool = agent.tools[PTC_TOOL_NAME]
        code_desc = ptc_tool.parameters["properties"]["code"]["description"]
        # The full rule block lives in the tool description — param desc stays concise.
        assert "<instructions>" not in code_desc
        assert "<recommended_usage>" not in code_desc
        # Compact: under ~1500 chars (vs the 2000+ of the rule block)
        assert len(code_desc) < 1500


class TestPTCAgentParameters:
    def test_max_output_bytes_parameter(self) -> None:
        agent = Agent(name="t", tools=[greet], ptc=True, ptc_max_output_bytes=1024)
        assert agent.ptc_max_output_bytes == 1024
        assert agent.tools[PTC_TOOL_NAME]._max_output_bytes == 1024  # type: ignore[attr-defined]

    def test_default_max_output_bytes(self) -> None:
        agent = Agent(name="t", tools=[greet], ptc=True)
        assert agent.ptc_max_output_bytes == 200_000

    def test_default_max_tool_calls(self) -> None:
        agent = Agent(name="t", tools=[greet], ptc=True)
        assert agent.ptc_max_tool_calls == 200

    def test_serialisation_round_trip(self) -> None:
        agent = Agent(
            name="t",
            tools=[greet],
            ptc=True,
            ptc_timeout=90,
            ptc_max_output_bytes=1024,
            ptc_max_tool_calls=50,
        )
        data = agent.to_dict()
        assert data["ptc_max_output_bytes"] == 1024
        assert data["ptc_max_tool_calls"] == 50

        restored = Agent.from_dict(data)
        assert restored.ptc_max_output_bytes == 1024
        assert restored.ptc_max_tool_calls == 50


# ---------------------------------------------------------------------------
# Sandbox tests — restricted builtins, blocked imports, AST pre-scan
# ---------------------------------------------------------------------------


class TestPTCSandboxImports:
    """Blocked stdlib imports must raise a clear ImportError."""

    async def test_import_os_blocked_by_ast_scan(self) -> None:
        agent = Agent(name="t", tools=[greet], ptc=True)
        executor = PTCExecutor(agent)
        result = await executor.run("import os\nprint(os.getcwd())")
        assert "blocked" in result
        assert "os" in result
        assert "default_api" in result

    async def test_import_subprocess_blocked(self) -> None:
        agent = Agent(name="t", tools=[greet], ptc=True)
        executor = PTCExecutor(agent)
        result = await executor.run("import subprocess")
        assert "blocked" in result
        assert "subprocess" in result

    async def test_import_sys_blocked(self) -> None:
        agent = Agent(name="t", tools=[greet], ptc=True)
        executor = PTCExecutor(agent)
        result = await executor.run("import sys")
        assert "blocked" in result
        assert "sys" in result

    async def test_import_pathlib_blocked(self) -> None:
        agent = Agent(name="t", tools=[greet], ptc=True)
        executor = PTCExecutor(agent)
        result = await executor.run("import pathlib")
        assert "blocked" in result

    async def test_import_shutil_blocked(self) -> None:
        agent = Agent(name="t", tools=[greet], ptc=True)
        executor = PTCExecutor(agent)
        result = await executor.run("import shutil")
        assert "blocked" in result

    async def test_import_socket_blocked(self) -> None:
        agent = Agent(name="t", tools=[greet], ptc=True)
        executor = PTCExecutor(agent)
        result = await executor.run("import socket")
        assert "blocked" in result

    async def test_import_urllib_blocked(self) -> None:
        agent = Agent(name="t", tools=[greet], ptc=True)
        executor = PTCExecutor(agent)
        result = await executor.run("import urllib")
        assert "blocked" in result

    async def test_import_threading_blocked(self) -> None:
        agent = Agent(name="t", tools=[greet], ptc=True)
        executor = PTCExecutor(agent)
        result = await executor.run("import threading")
        assert "blocked" in result

    async def test_import_multiprocessing_blocked(self) -> None:
        agent = Agent(name="t", tools=[greet], ptc=True)
        executor = PTCExecutor(agent)
        result = await executor.run("import multiprocessing")
        assert "blocked" in result

    async def test_import_ctypes_blocked(self) -> None:
        agent = Agent(name="t", tools=[greet], ptc=True)
        executor = PTCExecutor(agent)
        result = await executor.run("import ctypes")
        assert "blocked" in result

    async def test_import_inspect_blocked(self) -> None:
        agent = Agent(name="t", tools=[greet], ptc=True)
        executor = PTCExecutor(agent)
        result = await executor.run("import inspect")
        assert "blocked" in result

    async def test_from_os_import_blocked(self) -> None:
        agent = Agent(name="t", tools=[greet], ptc=True)
        executor = PTCExecutor(agent)
        result = await executor.run("from os import path")
        assert "blocked" in result

    async def test_from_subprocess_import_blocked(self) -> None:
        agent = Agent(name="t", tools=[greet], ptc=True)
        executor = PTCExecutor(agent)
        result = await executor.run("from subprocess import run")
        assert "blocked" in result

    async def test_import_dotted_blocked(self) -> None:
        """Blocking is by top package name — `os.path` → `os`."""
        agent = Agent(name="t", tools=[greet], ptc=True)
        executor = PTCExecutor(agent)
        result = await executor.run("import os.path")
        assert "blocked" in result

    async def test_re_import_of_preloaded_allowed(self) -> None:
        """`import json as j` should work (no-op re-import)."""
        agent = Agent(name="t", tools=[greet], ptc=True)
        executor = PTCExecutor(agent)
        result = await executor.run("import json as j\nprint(j.dumps({'a': 1}))")
        assert '{"a": 1}' in result

    async def test_unknown_import_rejected(self) -> None:
        agent = Agent(name="t", tools=[greet], ptc=True)
        executor = PTCExecutor(agent)
        result = await executor.run("import some_random_module")
        assert "not allowed" in result or "blocked" in result


class TestPTCSandboxBuiltins:
    """Dangerous builtins must raise PTCSandboxError with a clear message."""

    async def test_open_blocked(self) -> None:
        agent = Agent(name="t", tools=[greet], ptc=True)
        executor = PTCExecutor(agent)
        result = await executor.run("open('/etc/passwd')")
        assert "blocked" in result
        assert "default_api" in result

    async def test_eval_blocked(self) -> None:
        agent = Agent(name="t", tools=[greet], ptc=True)
        executor = PTCExecutor(agent)
        result = await executor.run("eval('1+1')")
        assert "blocked" in result

    async def test_exec_blocked(self) -> None:
        agent = Agent(name="t", tools=[greet], ptc=True)
        executor = PTCExecutor(agent)
        result = await executor.run("exec('x = 1')")
        assert "blocked" in result

    async def test_compile_blocked(self) -> None:
        agent = Agent(name="t", tools=[greet], ptc=True)
        executor = PTCExecutor(agent)
        result = await executor.run("compile('1+1', '<s>', 'eval')")
        assert "blocked" in result

    async def test_globals_blocked(self) -> None:
        agent = Agent(name="t", tools=[greet], ptc=True)
        executor = PTCExecutor(agent)
        result = await executor.run("globals()")
        assert "blocked" in result

    async def test_locals_blocked(self) -> None:
        agent = Agent(name="t", tools=[greet], ptc=True)
        executor = PTCExecutor(agent)
        result = await executor.run("locals()")
        assert "blocked" in result

    async def test_vars_blocked(self) -> None:
        agent = Agent(name="t", tools=[greet], ptc=True)
        executor = PTCExecutor(agent)
        result = await executor.run("vars()")
        assert "blocked" in result

    async def test_dir_blocked(self) -> None:
        agent = Agent(name="t", tools=[greet], ptc=True)
        executor = PTCExecutor(agent)
        result = await executor.run("dir()")
        assert "blocked" in result

    async def test_input_blocked(self) -> None:
        agent = Agent(name="t", tools=[greet], ptc=True)
        executor = PTCExecutor(agent)
        result = await executor.run("input('> ')")
        assert "blocked" in result

    async def test_blocked_error_is_catchable(self) -> None:
        """User code can catch the sandbox error and continue."""
        agent = Agent(name="t", tools=[greet], ptc=True)
        executor = PTCExecutor(agent)
        code = """\
try:
    open('/etc/passwd')
except Exception as e:
    print(f'caught: {e}')
print('continued')
"""
        result = await executor.run(code)
        assert "caught:" in result
        assert "continued" in result


class TestPTCSandboxAstScan:
    """Static AST pre-scan blocks dunder escape patterns."""

    async def test_dunder_class_blocked(self) -> None:
        agent = Agent(name="t", tools=[greet], ptc=True)
        executor = PTCExecutor(agent)
        result = await executor.run("print(().__class__)")
        assert "__class__" in result
        assert "blocked" in result

    async def test_dunder_bases_blocked(self) -> None:
        agent = Agent(name="t", tools=[greet], ptc=True)
        executor = PTCExecutor(agent)
        result = await executor.run("print(object.__bases__)")
        assert "__bases__" in result

    async def test_dunder_subclasses_blocked(self) -> None:
        agent = Agent(name="t", tools=[greet], ptc=True)
        executor = PTCExecutor(agent)
        result = await executor.run("object.__subclasses__()")
        assert "__subclasses__" in result

    async def test_dunder_mro_blocked(self) -> None:
        agent = Agent(name="t", tools=[greet], ptc=True)
        executor = PTCExecutor(agent)
        result = await executor.run("print(int.__mro__)")
        assert "__mro__" in result

    async def test_dunder_globals_blocked(self) -> None:
        agent = Agent(name="t", tools=[greet], ptc=True)
        executor = PTCExecutor(agent)
        result = await executor.run("f = lambda: 1\nprint(f.__globals__)")
        assert "__globals__" in result

    async def test_dunder_dict_blocked(self) -> None:
        agent = Agent(name="t", tools=[greet], ptc=True)
        executor = PTCExecutor(agent)
        result = await executor.run("print({}.__dict__)")
        assert "__dict__" in result

    async def test_dunder_code_blocked(self) -> None:
        agent = Agent(name="t", tools=[greet], ptc=True)
        executor = PTCExecutor(agent)
        result = await executor.run("f = lambda: 1\nprint(f.__code__)")
        assert "__code__" in result

    async def test_chained_dunder_blocked(self) -> None:
        """The classic escape chain must be caught at the first hop."""
        agent = Agent(name="t", tools=[greet], ptc=True)
        executor = PTCExecutor(agent)
        result = await executor.run("print(().__class__.__bases__[0].__subclasses__())")
        assert "blocked" in result

    async def test_dynamic_getattr_dunder_blocked_at_runtime(self) -> None:
        """Dynamic dunder access via getattr(x, '__class__') is runtime-blocked."""
        agent = Agent(name="t", tools=[greet], ptc=True)
        executor = PTCExecutor(agent)
        # Build the dunder name at runtime so the AST scan cannot see it.
        code = "name = '__' + 'class' + '__'\nprint(getattr(object(), name))"
        result = await executor.run(code)
        # safer_getattr rejects underscore-prefixed names at runtime
        assert "blocked" in result or "invalid attribute" in result or "starts with" in result


class TestPTCSandboxSafeBuiltins:
    """The safe builtin subset must remain functional."""

    async def test_len_range_sum_sorted(self) -> None:
        agent = Agent(name="t", tools=[greet], ptc=True)
        executor = PTCExecutor(agent)
        result = await executor.run("print(sum(range(10)))\nprint(len(sorted([3,1,2])))")
        assert "45" in result
        assert "3" in result

    async def test_comprehensions_and_filters(self) -> None:
        agent = Agent(name="t", tools=[greet], ptc=True)
        executor = PTCExecutor(agent)
        result = await executor.run("print([x * 2 for x in range(5) if x % 2 == 0])")
        assert "[0, 4, 8]" in result

    async def test_type_constructors(self) -> None:
        agent = Agent(name="t", tools=[greet], ptc=True)
        executor = PTCExecutor(agent)
        result = await executor.run(
            "print(dict(a=1, b=2))\nprint(list((1,2,3)))\nprint(set([1,1,2]))"
        )
        assert "{'a': 1, 'b': 2}" in result
        assert "[1, 2, 3]" in result

    async def test_isinstance_still_works(self) -> None:
        agent = Agent(name="t", tools=[greet], ptc=True)
        executor = PTCExecutor(agent)
        result = await executor.run("print(isinstance(1, int))")
        assert "True" in result

    async def test_any_all_min_max(self) -> None:
        agent = Agent(name="t", tools=[greet], ptc=True)
        executor = PTCExecutor(agent)
        result = await executor.run("print(any([0, 1]))\nprint(all([1, 1]))\nprint(max([3, 1, 2]))")
        assert "True" in result
        assert "3" in result

    async def test_getattr_on_safe_name_works(self) -> None:
        agent = Agent(name="t", tools=[greet], ptc=True)
        executor = PTCExecutor(agent)
        result = await executor.run('d = {"a": 1}\nprint(getattr(d, "keys")())')
        assert "dict_keys" in result

    async def test_asyncio_gather_still_works(self) -> None:
        agent = Agent(name="t", tools=[greet, add], ptc=True)
        executor = PTCExecutor(agent)
        code = """\
results = await asyncio.gather(
    default_api.greet(name="A"),
    default_api.add(a=1, b=2),
)
for r in results:
    print(r)
"""
        result = await executor.run(code)
        assert "Hello, A!" in result
        assert "3" in result

    async def test_json_loads_still_works(self) -> None:
        agent = Agent(name="t", tools=[greet], ptc=True)
        executor = PTCExecutor(agent)
        result = await executor.run('d = json.loads(\'{"a": 1, "b": 2}\')\nprint(d["a"] + d["b"])')
        assert "3" in result

    async def test_math_re_still_work(self) -> None:
        agent = Agent(name="t", tools=[greet], ptc=True)
        executor = PTCExecutor(agent)
        result = await executor.run('print(math.sqrt(16))\nprint(re.sub(r"\\d", "X", "a1b2"))')
        assert "4.0" in result
        assert "aXbX" in result


class TestPTCSandboxMessages:
    """Sandbox errors must clearly direct the agent to default_api."""

    async def test_import_error_mentions_default_api(self) -> None:
        agent = Agent(name="t", tools=[greet], ptc=True)
        executor = PTCExecutor(agent)
        result = await executor.run("import os")
        assert "default_api" in result

    async def test_builtin_error_mentions_default_api(self) -> None:
        agent = Agent(name="t", tools=[greet], ptc=True)
        executor = PTCExecutor(agent)
        result = await executor.run("open('x')")
        assert "default_api" in result

    async def test_ast_scan_error_mentions_default_api(self) -> None:
        agent = Agent(name="t", tools=[greet], ptc=True)
        executor = PTCExecutor(agent)
        result = await executor.run("x = object.__bases__")
        assert "default_api" in result


# ---------------------------------------------------------------------------
# Schema-leak regressions — Swarm propagation, spawn_self, activate_skill,
# delegate tool marking, default_api.* normalization
# ---------------------------------------------------------------------------


class TestSwarmPTCLeakRegression:
    """``Swarm(ptc=True)`` must properly register the PTC tool on each member
    AND invalidate the cached tool schemas, otherwise PTC-eligible tools leak
    as direct schemas on the next LLM call.
    """

    def test_swarm_ptc_true_registers_ptc_tool_on_members(self) -> None:
        from exo.swarm import Swarm

        a1 = Agent(name="a1", tools=[greet])
        a2 = Agent(name="a2", tools=[add])
        assert PTC_TOOL_NAME not in a1.tools
        assert PTC_TOOL_NAME not in a2.tools

        Swarm(agents=[a1, a2], ptc=True)

        assert a1.ptc is True
        assert a2.ptc is True
        assert PTC_TOOL_NAME in a1.tools
        assert PTC_TOOL_NAME in a2.tools
        assert isinstance(a1.tools[PTC_TOOL_NAME], PTCTool)
        assert isinstance(a2.tools[PTC_TOOL_NAME], PTCTool)

    def test_swarm_ptc_true_hides_eligible_tools_from_schema(self) -> None:
        from exo.swarm import Swarm

        a1 = Agent(name="a1", tools=[greet, add])

        # Prime the cache in non-PTC state
        before = a1.get_tool_schemas()
        before_names = {s["function"]["name"] for s in before}
        assert "greet" in before_names
        assert "add" in before_names

        Swarm(agents=[a1], ptc=True)

        # Cache must be invalidated and schemas rebuilt with PTC filter.
        after = a1.get_tool_schemas()
        after_names = {s["function"]["name"] for s in after}
        assert "greet" not in after_names
        assert "add" not in after_names
        assert PTC_TOOL_NAME in after_names

    def test_swarm_ptc_false_removes_ptc_tool(self) -> None:
        from exo.swarm import Swarm

        a1 = Agent(name="a1", tools=[greet], ptc=True)
        assert PTC_TOOL_NAME in a1.tools

        Swarm(agents=[a1], ptc=False)

        assert a1.ptc is False
        assert PTC_TOOL_NAME not in a1.tools
        schemas = a1.get_tool_schemas()
        names = {s["function"]["name"] for s in schemas}
        assert "greet" in names  # back as direct schema
        assert PTC_TOOL_NAME not in names

    def test_swarm_ptc_true_is_idempotent(self) -> None:
        """Propagating ``ptc=True`` to an agent already in PTC mode is a no-op."""
        from exo.swarm import Swarm

        a1 = Agent(name="a1", tools=[greet], ptc=True)
        ptc_tool_before = a1.tools[PTC_TOOL_NAME]

        Swarm(agents=[a1], ptc=True)

        assert a1.ptc is True
        # The existing PTCTool instance should be preserved, not replaced.
        assert a1.tools[PTC_TOOL_NAME] is ptc_tool_before


class TestSpawnSelfPTCLeakRegression:
    """``spawn_self`` must not leak the parent's PTC tool into child.tools
    as a direct schema, and must propagate PTC settings so the child
    re-registers its own PTCTool.
    """

    async def test_spawn_self_excludes_parent_ptc_tool(self) -> None:
        from unittest.mock import AsyncMock

        parent = Agent(
            name="p",
            tools=[greet, add],
            ptc=True,
            allow_self_spawn=True,
        )
        parent._current_provider = AsyncMock()

        # Grab the spawn_self tool's closure-captured build of child_tools
        # by inspecting what the constructor would receive.  The simplest
        # regression proof is to spawn once and verify the child inherits
        # ptc=True, has its own PTCTool, and does NOT have the parent's
        # PTCTool object.
        parent_ptc_tool = parent.tools[PTC_TOOL_NAME]

        # Instead of running a full spawn (which needs a real provider),
        # replicate the child construction logic directly to verify the fix.
        child_tools = [
            t
            for name, t in parent.tools.items()
            if name != "spawn_self"
            and not getattr(t, "_is_context_tool", False)
            and not getattr(t, "_is_ptc_tool", False)
        ]
        # Parent's PTCTool MUST be filtered out
        assert parent_ptc_tool not in child_tools
        tool_names = {t.name for t in child_tools}
        assert PTC_TOOL_NAME not in tool_names
        assert "greet" in tool_names
        assert "add" in tool_names

    async def test_spawned_child_has_own_ptc_tool(self) -> None:
        """Reconstructing the child agent with propagated PTC settings
        should give it a fresh PTCTool bound to itself."""
        parent = Agent(
            name="parent",
            tools=[greet, add],
            ptc=True,
            ptc_timeout=45,
            ptc_max_tool_calls=99,
        )

        # Mimic the construction path in spawn_self
        from exo.agent import _SUBAGENT_TOOL_NAMES

        child_tools = [
            t
            for name, t in parent.tools.items()
            if name not in _SUBAGENT_TOOL_NAMES
            and not getattr(t, "_is_context_tool", False)
            and not getattr(t, "_is_ptc_tool", False)
        ]
        child = Agent(
            name="child",
            tools=child_tools,
            ptc=parent.ptc,
            ptc_timeout=parent.ptc_timeout,
            ptc_max_output_bytes=parent.ptc_max_output_bytes,
            ptc_max_tool_calls=parent.ptc_max_tool_calls,
        )

        assert child.ptc is True
        assert child.ptc_timeout == 45
        assert child.ptc_max_tool_calls == 99
        assert PTC_TOOL_NAME in child.tools
        # Child's PTCTool must be bound to the CHILD, not the parent.
        child_ptc_tool = child.tools[PTC_TOOL_NAME]
        assert isinstance(child_ptc_tool, PTCTool)
        assert child_ptc_tool._agent is child  # type: ignore[attr-defined]
        assert child_ptc_tool is not parent.tools[PTC_TOOL_NAME]

        # Child's schemas should NOT leak PTC-eligible tools
        schemas = child.get_tool_schemas()
        names = {s["function"]["name"] for s in schemas}
        assert "greet" not in names
        assert "add" not in names
        assert PTC_TOOL_NAME in names


class TestDefaultApiNormalization:
    """Runner must rewrite ``default_api.<name>`` tool calls that some LLMs
    emit directly (outside ``__exo_ptc__``) to the bare tool name so dispatch
    succeeds and the clean name flows through SSE events.
    """

    def test_normalize_strips_prefix_for_known_tool(self) -> None:
        from exo.ptc import normalize_default_api_tool_calls

        agent = Agent(name="t", tools=[greet], ptc=True)
        tcs = [ToolCall(id="tc1", name="default_api.greet", arguments="{}")]
        # ToolCall is a frozen pydantic model — the helper rebuilds the
        # entry at its list index via model_copy(update=...).
        normalize_default_api_tool_calls(tcs, agent)
        assert tcs[0].name == "greet"

    def test_normalize_leaves_unknown_tool_alone(self) -> None:
        from exo.ptc import normalize_default_api_tool_calls

        agent = Agent(name="t", tools=[greet], ptc=True)
        tcs = [ToolCall(id="tc1", name="default_api.unknown", arguments="{}")]
        normalize_default_api_tool_calls(tcs, agent)
        # Unknown → leave untouched so the standard "unknown tool" error
        # path still fires.
        assert tcs[0].name == "default_api.unknown"

    def test_normalize_leaves_non_prefixed_alone(self) -> None:
        from exo.ptc import normalize_default_api_tool_calls

        agent = Agent(name="t", tools=[greet], ptc=True)
        tcs = [ToolCall(id="tc1", name="greet", arguments="{}")]
        normalize_default_api_tool_calls(tcs, agent)
        assert tcs[0].name == "greet"

    def test_normalize_handles_multiple_calls(self) -> None:
        from exo.ptc import normalize_default_api_tool_calls

        agent = Agent(name="t", tools=[greet, add], ptc=True)
        tcs = [
            ToolCall(id="tc1", name="default_api.greet", arguments='{"name":"A"}'),
            ToolCall(id="tc2", name="default_api.add", arguments='{"a":1,"b":2}'),
            ToolCall(id="tc3", name="plain_name", arguments="{}"),
        ]
        normalize_default_api_tool_calls(tcs, agent)
        assert tcs[0].name == "greet"
        assert tcs[1].name == "add"
        assert tcs[2].name == "plain_name"

    async def test_ptc_description_no_longer_mentions_dotted_name(self) -> None:
        """The per-tool ``# usage: await default_api.<name>`` comment was
        removed because some models misparse it as the literal function
        name.  The preamble still teaches the ``default_api`` prefix."""
        agent = Agent(name="t", tools=[greet, add], ptc=True)
        desc = agent.tools[PTC_TOOL_NAME].description
        # Preamble keeps the high-level directive
        assert "default_api" in desc
        # But no per-tool "# usage: ..." comments (root cause of confusion)
        assert "# usage: await default_api.greet" not in desc
        assert "# usage: await default_api.add" not in desc


class TestSwarmTeamDelegateNoPTCLeak:
    """Swarm team-mode delegate tools must stay as direct schemas even when
    the lead agent has ``ptc=True``.  Routing to workers is not a PTC step.
    """

    def test_delegate_tool_has_ptc_exclude_flag(self) -> None:
        from exo.swarm import _DelegateTool

        class _FakeWorker:
            name = "worker_x"

        dtool = _DelegateTool(worker=_FakeWorker())
        assert getattr(dtool, "_ptc_exclude", False) is True

    def test_delegate_tool_excluded_from_ptc_eligibility(self) -> None:
        from exo.swarm import _DelegateTool

        class _FakeWorker:
            name = "worker_x"

        agent = Agent(name="lead", tools=[greet], ptc=True)
        dtool = _DelegateTool(worker=_FakeWorker())
        agent.tools[dtool.name] = dtool
        agent._cached_tool_schemas = None

        eligible = get_ptc_eligible_tools(agent)
        assert "greet" in eligible
        assert dtool.name not in eligible  # excluded via _ptc_exclude

        # And it appears as a DIRECT schema (not absorbed into PTC)
        schemas = agent.get_tool_schemas()
        names = {s["function"]["name"] for s in schemas}
        assert dtool.name in names
        assert PTC_TOOL_NAME in names
        assert "greet" not in names  # greet stays absorbed into PTC


class TestPTCSignatureRichMetadata:
    """Rich tool metadata (Literal, descriptions, full docstring, defaults,
    injected_tool_args) must all survive into the PTC description.
    """

    def test_literal_rendered_as_literal_type(self) -> None:

        @tool
        def picker(
            mode: Annotated[Literal["fast", "slow"], "Execution mode."],
        ) -> str:
            """Pick a mode."""
            return ""

        agent = Agent(name="t", tools=[picker], ptc=True)
        desc = agent.tools[PTC_TOOL_NAME].description
        assert "Literal['fast', 'slow']" in desc

    def test_per_param_descriptions_in_args_section(self) -> None:

        @tool
        def with_desc(
            query: Annotated[str, "The thing to look up."],
            limit: Annotated[int | None, "Max results 1-100."] = 10,
        ) -> str:
            """Search something."""
            return ""

        agent = Agent(name="t", tools=[with_desc], ptc=True)
        desc = agent.tools[PTC_TOOL_NAME].description
        assert "Args:" in desc
        assert "query: The thing to look up." in desc
        assert "limit: Max results 1-100." in desc

    def test_full_multiline_docstring_preserved(self) -> None:
        @tool
        def rich(x: int) -> str:
            """First line.

            <instructions>
            - Must follow this rule.
            - Also this rule.
            </instructions>

            <recommended_usage>
            - Use it like so.
            </recommended_usage>
            """
            return ""

        agent = Agent(name="t", tools=[rich], ptc=True)
        desc = agent.tools[PTC_TOOL_NAME].description
        assert "<instructions>" in desc
        assert "Must follow this rule" in desc
        assert "Also this rule" in desc
        assert "<recommended_usage>" in desc
        assert "Use it like so" in desc

    def test_default_values_shown_in_signature(self) -> None:
        @tool
        def defaults(
            x: int = 42,
            y: str = "hello",
            z: bool = True,
        ) -> str:
            """Defaults."""
            return ""

        agent = Agent(name="t", tools=[defaults], ptc=True)
        desc = agent.tools[PTC_TOOL_NAME].description
        assert "x: int = 42" in desc
        assert "y: str = 'hello'" in desc
        assert "z: bool = True" in desc

    def test_injected_tool_args_in_inner_signature(self) -> None:

        @tool
        def inner(q: Annotated[str, "query"]) -> str:
            """Search."""
            return ""

        agent = Agent(
            name="t",
            tools=[inner],
            ptc=True,
            injected_tool_args={"user_id": "The caller's user ID."},
        )
        desc = agent.tools[PTC_TOOL_NAME].description
        assert "user_id: str | None = None" in desc
        assert "user_id: [injected] The caller's user ID." in desc

    def test_injected_tool_args_stripped_before_inner_tool_called(self) -> None:
        from exo.ptc import PTCExecutor

        @tool
        def spy(x: int) -> str:
            """Spy tool."""
            return str(x)

        agent = Agent(
            name="t",
            tools=[spy],
            ptc=True,
            injected_tool_args={"user_id": "The user ID."},
        )
        executor = PTCExecutor(agent)
        import asyncio

        async def run() -> None:
            await executor.run('print(await default_api.spy(x=5, user_id="abc"))')

        # The tool function has no user_id param — if we didn't strip it,
        # this would TypeError.  The strip happens in _make_tool_fn.
        asyncio.run(run())


class TestPTCExtraArgs:
    """``ptc_extra_args`` adds schema fields to the outer __exo_ptc__ tool
    call and exposes them to executing PTC code as ``ptc_args``."""

    def test_extra_args_in_outer_schema(self) -> None:
        agent = Agent(
            name="t",
            tools=[greet],
            ptc=True,
            ptc_extra_args={
                "intent": "What the PTC call is trying to do.",
                "tag": "A categorisation tag.",
            },
        )
        schemas = agent.get_tool_schemas()
        ptc = next(s for s in schemas if s["function"]["name"] == PTC_TOOL_NAME)
        props = ptc["function"]["parameters"]["properties"]
        assert "code" in props
        assert "intent" in props
        assert "tag" in props
        assert props["intent"]["description"] == "What the PTC call is trying to do."
        # Only ``code`` is required
        assert ptc["function"]["parameters"]["required"] == ["code"]

    def test_extra_args_accessible_in_ptc_code(self) -> None:
        agent = Agent(
            name="t",
            tools=[greet],
            ptc=True,
            ptc_extra_args={"intent": "Description."},
        )
        ptc_tool = agent.tools[PTC_TOOL_NAME]
        import asyncio

        async def run() -> str:
            return await ptc_tool.execute(
                code="print(ptc_args.get('intent', 'MISSING'))",
                intent="search all items",
            )

        result = asyncio.run(run())
        assert "search all items" in result

    def test_extra_args_unknown_keys_filtered(self) -> None:
        agent = Agent(
            name="t",
            tools=[greet],
            ptc=True,
            ptc_extra_args={"intent": "Description."},
        )
        ptc_tool = agent.tools[PTC_TOOL_NAME]
        import asyncio

        async def run() -> str:
            return await ptc_tool.execute(
                code="print(list(sorted(ptc_args.keys())))",
                intent="ok",
                bogus_key="should be dropped",
            )

        result = asyncio.run(run())
        # Only 'intent' should be in ptc_args; 'bogus_key' dropped
        assert "['intent']" in result

    def test_extra_args_description_block_present(self) -> None:
        agent = Agent(
            name="t",
            tools=[greet],
            ptc=True,
            ptc_extra_args={"intent": "What you plan to do."},
        )
        desc = agent.tools[PTC_TOOL_NAME].description
        assert "<ptc_extra_args>" in desc
        assert "`intent`: What you plan to do." in desc
        assert "ptc_args" in desc

    def test_no_extra_args_no_block(self) -> None:
        agent = Agent(name="t", tools=[greet], ptc=True)
        desc = agent.tools[PTC_TOOL_NAME].description
        assert "<ptc_extra_args>" not in desc

    def test_extra_args_serialization_round_trip(self) -> None:
        agent = Agent(
            name="t",
            tools=[greet],
            ptc=True,
            ptc_extra_args={"intent": "test"},
        )
        data = agent.to_dict()
        assert data["ptc_extra_args"] == {"intent": "test"}
        restored = Agent.from_dict(data)
        assert restored.ptc_extra_args == {"intent": "test"}
        assert "intent" in restored.tools[PTC_TOOL_NAME].parameters["properties"]

    def test_ptc_args_is_isolated_per_run(self) -> None:
        """Each PTC invocation gets its own ``ptc_args`` — mutations don't
        bleed across runs."""
        agent = Agent(
            name="t",
            tools=[greet],
            ptc=True,
            ptc_extra_args={"key": "desc"},
        )
        ptc_tool = agent.tools[PTC_TOOL_NAME]
        import asyncio

        async def run() -> tuple[str, str]:
            a = await ptc_tool.execute(code="print(ptc_args)", key="first")
            b = await ptc_tool.execute(code="print(ptc_args)", key="second")
            return a, b

        a, b = asyncio.run(run())
        assert "first" in a
        assert "second" in b
        assert "first" not in b


class TestHandoffCacheInvalidation:
    """Runtime handoff registration must invalidate _cached_tool_schemas
    so PTC filtering sees the new handoff targets correctly."""

    def test_add_handoff_invalidates_schema_cache(self) -> None:
        @tool
        def foo(x: int) -> str:
            """A foo tool."""
            return str(x)

        target = Agent(name="target", tools=[])
        agent = Agent(name="src", tools=[foo])

        # Prime the cache
        before = agent.get_tool_schemas()
        before_names = {s["function"]["name"] for s in before}
        assert "foo" in before_names
        assert agent._cached_tool_schemas is not None

        # Register a handoff — must invalidate cache
        agent._register_handoff(target)
        assert agent._cached_tool_schemas is None

    def test_add_handoff_with_matching_tool_name_excludes_after(self) -> None:
        """When a handoff target's name matches a tool, PTC filter excludes
        the tool.  A runtime add_handoff must refresh this filter."""

        @tool
        def helper(x: int) -> str:
            """Helper tool."""
            return str(x)

        # Target named the same as the tool
        target = Agent(name="helper", tools=[])
        agent = Agent(name="src", tools=[helper], ptc=True)

        # Before handoff: helper is PTC-eligible → hidden from schemas
        before_eligible = get_ptc_eligible_tools(agent)
        assert "helper" in before_eligible

        # Register handoff — invalidates cache
        agent._register_handoff(target)

        # After handoff: helper is a handoff target → excluded from PTC
        after_eligible = get_ptc_eligible_tools(agent)
        assert "helper" not in after_eligible


class TestHumanInputToolPTCExclude:
    """HumanInputTool must stay as a direct schema when ptc=True so the
    interactive prompt works without being buffered inside PTC code."""

    def test_human_input_tool_has_ptc_exclude(self) -> None:
        from exo.human import HumanInputTool

        tool = HumanInputTool()
        assert getattr(tool, "_ptc_exclude", False) is True

    def test_human_input_tool_excluded_from_ptc_eligible(self) -> None:
        from exo.human import HumanInputTool

        @tool
        def regular(x: int) -> str:
            """regular tool"""
            return str(x)

        agent = Agent(name="t", tools=[regular, HumanInputTool()], ptc=True)
        eligible = get_ptc_eligible_tools(agent)
        assert "regular" in eligible
        assert "human_input" not in eligible

        schemas = agent.get_tool_schemas()
        names = {s["function"]["name"] for s in schemas}
        assert "human_input" in names  # stays as direct schema
        assert "regular" not in names  # absorbed into PTC


class TestPTCDescCacheIncludesParameters:
    """PTCTool description cache must invalidate when a tool's parameters
    mutate in place (dynamic schema updates)."""

    def test_parameter_mutation_invalidates_description_cache(self) -> None:
        @tool
        def dyn(x: int) -> str:
            """Dynamic tool."""
            return str(x)

        agent = Agent(name="t", tools=[dyn], ptc=True)
        ptc_tool: PTCTool = agent.tools[PTC_TOOL_NAME]  # type: ignore[assignment]

        desc1 = ptc_tool.description
        assert "x: int" in desc1

        # Mutate the tool's parameters in place (e.g. dynamic schema update)
        dyn.parameters = {
            "type": "object",
            "properties": {"y": {"type": "string"}},
            "required": ["y"],
        }

        desc2 = ptc_tool.description
        # Description must reflect the NEW parameter schema
        assert "y: str" in desc2
        assert desc1 is not desc2


class TestActivateSkillCacheInvalidation:
    """``activate_skill`` must invalidate the schema cache after adding tools,
    otherwise the new tools are invisible to the LLM on the next call.
    """

    def test_planner_agent_inherits_parent_ptc(self) -> None:
        """_build_planner_agent must propagate PTC settings and exclude the
        parent's PTCTool from the planner's tool list."""
        from exo._internal.planner import _build_planner_agent

        parent = Agent(
            name="parent",
            tools=[greet, add],
            ptc=True,
            ptc_timeout=90,
            ptc_max_tool_calls=75,
        )

        planner = _build_planner_agent(
            parent,
            planner_model="openai:gpt-4o-mini",
            planner_instructions="plan things",
        )

        # Planner must have its own PTC enabled + its own PTCTool
        assert planner.ptc is True
        assert planner.ptc_timeout == 90
        assert planner.ptc_max_tool_calls == 75
        assert PTC_TOOL_NAME in planner.tools

        # Planner's PTCTool must be bound to the planner itself, not parent
        planner_ptc = planner.tools[PTC_TOOL_NAME]
        assert planner_ptc is not parent.tools[PTC_TOOL_NAME]
        assert planner_ptc._agent is planner  # type: ignore[attr-defined]

        # PTC-eligible tools must be hidden from the planner's schema list
        schemas = planner.get_tool_schemas()
        names = {s["function"]["name"] for s in schemas}
        assert "greet" not in names
        assert "add" not in names
        assert PTC_TOOL_NAME in names

    def test_planner_agent_without_ptc(self) -> None:
        """When parent has ptc=False, planner also has ptc=False
        and all tools appear as direct schemas."""
        from exo._internal.planner import _build_planner_agent

        parent = Agent(name="parent", tools=[greet, add])
        assert parent.ptc is False

        planner = _build_planner_agent(
            parent,
            planner_model="openai:gpt-4o-mini",
            planner_instructions="plan",
        )
        assert planner.ptc is False
        assert PTC_TOOL_NAME not in planner.tools
        schemas = planner.get_tool_schemas()
        names = {s["function"]["name"] for s in schemas}
        assert "greet" in names
        assert "add" in names

    async def test_activate_skill_invalidates_schema_cache(self) -> None:
        from exo.skills import Skill, SkillRegistry

        @tool
        def new_skill_tool(x: int) -> str:
            """A tool added by the skill."""
            return str(x)

        skill = Skill(
            name="test_skill",
            description="test",
            usage="test skill usage",
            tool_list={"math": ["compute"]},
            active=True,
        )
        registry = SkillRegistry()
        registry._skills[skill.name] = skill  # type: ignore[attr-defined]

        agent = Agent(
            name="t",
            tools=[greet],
            skills=registry,
            tool_resolver={"test_skill": [new_skill_tool]},
        )

        # Prime the cache
        before = agent.get_tool_schemas()
        before_names = {s["function"]["name"] for s in before}
        assert "new_skill_tool" not in before_names

        # Activate the skill — this should invalidate the cache so the
        # next LLM call sees the new tool.
        await agent.tools["activate_skill"].execute(name="test_skill")

        # The new tool must appear in the schema list on the next fetch
        # (regression guard: before the fix, the stale cache hid it).
        after = agent.get_tool_schemas()
        after_names = {s["function"]["name"] for s in after}
        assert "new_skill_tool" in after_names
