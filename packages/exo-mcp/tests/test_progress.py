"""Tests for MCP progress notification capture (US-028) and stream wiring (US-029)."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock

import pytest
from mcp.types import CallToolResult, TextContent  # pyright: ignore[reportMissingImports]
from mcp.types import Tool as MCPTool  # pyright: ignore[reportMissingImports]

from exo.mcp.client import (  # pyright: ignore[reportMissingImports]
    MCPClientError,
    MCPServerConfig,
    MCPServerConnection,
    MCPTransport,
)
from exo.mcp.tools import (  # pyright: ignore[reportMissingImports]
    MCPToolWrapper,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mcp_tool(
    name: str = "search",
    description: str = "Search tool",
) -> MCPTool:
    return MCPTool(
        name=name,
        description=description,
        inputSchema={"type": "object", "properties": {"q": {"type": "string"}}},
    )


def _make_call_result(text: str = "result") -> CallToolResult:
    return CallToolResult(content=[TextContent(type="text", text=text)], isError=False)


# ---------------------------------------------------------------------------
# MCPProgressEvent type tests
# ---------------------------------------------------------------------------


class TestMCPProgressEvent:
    """MCPProgressEvent is a proper Pydantic model with the right fields."""

    def test_basic_construction(self) -> None:
        from exo.types import MCPProgressEvent

        evt = MCPProgressEvent(tool_name="my_tool", progress=5, total=10, message="halfway")
        assert evt.tool_name == "my_tool"
        assert evt.progress == 5
        assert evt.total == 10
        assert evt.message == "halfway"
        assert evt.type == "mcp_progress"

    def test_default_fields(self) -> None:
        from exo.types import MCPProgressEvent

        evt = MCPProgressEvent(tool_name="t", progress=3)
        assert evt.total is None
        assert evt.message == ""
        assert evt.agent_name == ""

    def test_is_in_stream_event_union(self) -> None:
        """MCPProgressEvent must be part of the StreamEvent type union."""
        # StreamEvent is a Union type alias; verify the annotation includes MCPProgressEvent
        import typing

        from exo.types import MCPProgressEvent, StreamEvent

        args = typing.get_args(StreamEvent)
        assert MCPProgressEvent in args

    def test_frozen_model(self) -> None:
        from pydantic import ValidationError

        from exo.types import MCPProgressEvent

        evt = MCPProgressEvent(tool_name="t", progress=1)
        with pytest.raises(ValidationError):
            evt.progress = 99  # type: ignore[misc]

    def test_progress_is_int(self) -> None:
        from exo.types import MCPProgressEvent

        evt = MCPProgressEvent(tool_name="t", progress=7, total=20)
        assert isinstance(evt.progress, int)
        assert isinstance(evt.total, int)


# ---------------------------------------------------------------------------
# MCPServerConnection.call_tool progress_callback tests
# ---------------------------------------------------------------------------


class TestMCPServerConnectionProgressCallback:
    """MCPServerConnection.call_tool() passes progress_callback to the session."""

    @pytest.mark.asyncio
    async def test_progress_callback_passed_to_session(self) -> None:
        config = MCPServerConfig(
            name="test-server",
            transport=MCPTransport.STDIO,
            command="echo",
        )
        conn = MCPServerConnection(config)

        mock_session = AsyncMock()
        mock_session.call_tool = AsyncMock(return_value=_make_call_result("ok"))
        conn._session = mock_session

        received: list[Any] = []

        async def my_callback(progress: float, total: float | None, message: str | None) -> None:
            received.append((progress, total, message))

        await conn.call_tool("search", {"q": "test"}, progress_callback=my_callback)

        mock_session.call_tool.assert_awaited_once()
        call_kwargs = mock_session.call_tool.call_args
        assert call_kwargs.kwargs.get("progress_callback") is my_callback

    @pytest.mark.asyncio
    async def test_no_progress_callback_default_none(self) -> None:
        config = MCPServerConfig(name="s", transport=MCPTransport.STDIO, command="echo")
        conn = MCPServerConnection(config)

        mock_session = AsyncMock()
        mock_session.call_tool = AsyncMock(return_value=_make_call_result("ok"))
        conn._session = mock_session

        await conn.call_tool("t", {"a": 1})

        call_kwargs = mock_session.call_tool.call_args
        # progress_callback should be None (default)
        assert call_kwargs.kwargs.get("progress_callback") is None

    @pytest.mark.asyncio
    async def test_not_connected_raises(self) -> None:
        config = MCPServerConfig(name="s", transport=MCPTransport.STDIO, command="echo")
        conn = MCPServerConnection(config)
        with pytest.raises(MCPClientError, match="not connected"):
            await conn.call_tool("t", None, progress_callback=None)


# ---------------------------------------------------------------------------
# MCPToolWrapper.progress_queue tests
# ---------------------------------------------------------------------------


class TestMCPToolWrapperProgressQueue:
    """MCPToolWrapper has a progress_queue and populates it during execute()."""

    def test_progress_queue_created_in_init(self) -> None:
        mcp_tool = _make_mcp_tool()
        call_fn = AsyncMock(return_value=_make_call_result("ok"))
        wrapper = MCPToolWrapper(mcp_tool, "srv", call_fn)
        assert hasattr(wrapper, "progress_queue")
        assert isinstance(wrapper.progress_queue, asyncio.Queue)

    def test_progress_queue_created_in_from_dict(self) -> None:
        mcp_tool = _make_mcp_tool()
        call_fn = AsyncMock(return_value=_make_call_result("ok"))
        wrapper = MCPToolWrapper(mcp_tool, "srv", call_fn)
        # to_dict() is exercised implicitly; from_dict is tested with an explicit payload
        _ = wrapper.to_dict()
        reconstructed = MCPToolWrapper.from_dict(
            {
                "__mcp_tool__": True,
                "name": "mcp__srv__search",
                "description": "Search tool",
                "parameters": {"type": "object", "properties": {}},
                "original_name": "search",
                "server_name": "srv",
                "large_output": False,
            }
        )
        assert hasattr(reconstructed, "progress_queue")
        assert isinstance(reconstructed.progress_queue, asyncio.Queue)

    @pytest.mark.asyncio
    async def test_execute_populates_progress_queue(self) -> None:
        """When call_fn triggers the progress callback, MCPProgressEvent items go in the queue."""
        from exo.types import MCPProgressEvent

        mcp_tool = _make_mcp_tool()
        captured_callbacks: list[Any] = []

        async def mock_call_fn(
            name: str, args: Any, *, progress_callback: Any = None
        ) -> CallToolResult:
            captured_callbacks.append(progress_callback)
            # Simulate the server firing two progress notifications
            if progress_callback is not None:
                await progress_callback(1.0, 3.0, "step 1")
                await progress_callback(2.0, 3.0, "step 2")
            return _make_call_result("final result")

        wrapper = MCPToolWrapper(mcp_tool, "srv", mock_call_fn)
        result = await wrapper.execute(q="hello")

        # Final result is returned, NOT the progress events
        assert result == "final result"

        # Progress queue contains the two events
        assert wrapper.progress_queue.qsize() == 2

        evt1: MCPProgressEvent = wrapper.progress_queue.get_nowait()
        assert evt1.progress == 1
        assert evt1.total == 3
        assert evt1.message == "step 1"

        evt2: MCPProgressEvent = wrapper.progress_queue.get_nowait()
        assert evt2.progress == 2
        assert evt2.total == 3
        assert evt2.message == "step 2"

    @pytest.mark.asyncio
    async def test_execute_no_progress_queue_empty(self) -> None:
        """When no progress notifications fire, progress_queue stays empty."""
        mcp_tool = _make_mcp_tool()
        call_fn = AsyncMock(return_value=_make_call_result("ok"))
        wrapper = MCPToolWrapper(mcp_tool, "srv", call_fn)
        result = await wrapper.execute(q="hi")
        assert result == "ok"
        assert wrapper.progress_queue.empty()

    @pytest.mark.asyncio
    async def test_execute_result_not_in_progress_queue(self) -> None:
        """The tool result string is never placed in the progress_queue."""
        from exo.types import MCPProgressEvent

        mcp_tool = _make_mcp_tool()

        async def mock_call_fn(
            name: str, args: Any, *, progress_callback: Any = None
        ) -> CallToolResult:
            if progress_callback is not None:
                await progress_callback(50.0, 100.0, "halfway")
            return _make_call_result("big result")

        wrapper = MCPToolWrapper(mcp_tool, "srv", mock_call_fn)
        result = await wrapper.execute(q="x")

        assert result == "big result"
        assert wrapper.progress_queue.qsize() == 1
        # The queued item is MCPProgressEvent, not a string
        item = wrapper.progress_queue.get_nowait()
        assert isinstance(item, MCPProgressEvent)
        assert item.message == "halfway"

    @pytest.mark.asyncio
    async def test_progress_event_total_none_when_unknown(self) -> None:
        """When MCP total is None, MCPProgressEvent.total is also None."""
        from exo.types import MCPProgressEvent

        mcp_tool = _make_mcp_tool()

        async def mock_call_fn(
            name: str, args: Any, *, progress_callback: Any = None
        ) -> CallToolResult:
            if progress_callback is not None:
                await progress_callback(5.0, None, "processing...")
            return _make_call_result("done")

        wrapper = MCPToolWrapper(mcp_tool, "srv", mock_call_fn)
        await wrapper.execute()

        evt: MCPProgressEvent = wrapper.progress_queue.get_nowait()
        assert evt.progress == 5
        assert evt.total is None
        assert evt.message == "processing..."

    @pytest.mark.asyncio
    async def test_progress_event_message_none_becomes_empty_string(self) -> None:
        """When MCP message is None, MCPProgressEvent.message is ''."""
        from exo.types import MCPProgressEvent

        mcp_tool = _make_mcp_tool()

        async def mock_call_fn(
            name: str, args: Any, *, progress_callback: Any = None
        ) -> CallToolResult:
            if progress_callback is not None:
                await progress_callback(1.0, 5.0, None)
            return _make_call_result("done")

        wrapper = MCPToolWrapper(mcp_tool, "srv", mock_call_fn)
        await wrapper.execute()

        evt: MCPProgressEvent = wrapper.progress_queue.get_nowait()
        assert evt.message == ""

    @pytest.mark.asyncio
    async def test_progress_queue_accumulates_across_calls(self) -> None:
        """Progress queue accumulates items; runner must drain between calls."""

        mcp_tool = _make_mcp_tool()

        async def mock_call_fn(
            name: str, args: Any, *, progress_callback: Any = None
        ) -> CallToolResult:
            if progress_callback is not None:
                await progress_callback(1.0, 2.0, "first")
            return _make_call_result("done")

        wrapper = MCPToolWrapper(mcp_tool, "srv", mock_call_fn)
        await wrapper.execute()
        await wrapper.execute()

        # Both calls contribute to the same queue
        assert wrapper.progress_queue.qsize() == 2

    @pytest.mark.asyncio
    async def test_progress_callback_not_passed_to_session_when_import_fails(self) -> None:
        """When exo-core MCPProgressEvent is not importable, no callback is passed."""
        mcp_tool = _make_mcp_tool()
        captured: list[Any] = []

        async def mock_call_fn(
            name: str, args: Any, *, progress_callback: Any = None
        ) -> CallToolResult:
            captured.append(progress_callback)
            return _make_call_result("ok")

        wrapper = MCPToolWrapper(mcp_tool, "srv", mock_call_fn)

        import sys

        orig = sys.modules.get("exo.types")
        sys.modules["exo.types"] = None  # type: ignore[assignment]
        try:
            result = await wrapper.execute(q="test")
        finally:
            if orig is None:
                del sys.modules["exo.types"]
            else:
                sys.modules["exo.types"] = orig

        assert result == "ok"
        # Callback should be None when import fails
        assert captured[0] is None


# ---------------------------------------------------------------------------
# US-029: agent.stream() yields MCPProgressEvent from tool progress queues
# ---------------------------------------------------------------------------


class _FakeStreamChunk:
    """Minimal stream chunk for testing."""

    def __init__(
        self,
        delta: str = "",
        tool_call_deltas: list[Any] | None = None,
        finish_reason: str | None = None,
        usage: Any = None,
    ) -> None:
        self.delta = delta
        self.tool_call_deltas = tool_call_deltas or []
        self.finish_reason = finish_reason

        class _U:
            input_tokens = 0
            output_tokens = 0
            total_tokens = 0

        self.usage = usage or _U()


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


class TestMCPProgressEventInStream:
    """agent.stream() drains MCPToolWrapper progress queues and yields MCPProgressEvents."""

    @pytest.mark.asyncio
    async def test_stream_yields_progress_events_from_mcp_tool(self) -> None:
        """MCPProgressEvent items appear in the stream after tool execution."""
        from exo.agent import Agent
        from exo.runner import run
        from exo.types import MCPProgressEvent, TextEvent, ToolCallEvent

        # Build an MCPToolWrapper whose call_fn fires progress notifications
        mcp_tool = _make_mcp_tool(name="search")

        async def mock_call_fn(
            name: str, args: Any, *, progress_callback: Any = None
        ) -> CallToolResult:
            if progress_callback is not None:
                await progress_callback(1.0, 2.0, "step 1")
                await progress_callback(2.0, 2.0, "step 2")
            return _make_call_result("search results")

        wrapper = MCPToolWrapper(mcp_tool, "srv", mock_call_fn)
        # wrapper.name is "mcp__srv__search"

        agent = Agent(name="searcher", tools=[wrapper], memory=None, context=None)

        # Round 1: LLM calls the mcp tool
        tool_name = wrapper.name  # "mcp__srv__search"
        round1 = [
            _FakeStreamChunk(
                tool_call_deltas=[
                    _FakeToolCallDelta(index=0, id="tc1", name=tool_name),
                ]
            ),
            _FakeStreamChunk(
                tool_call_deltas=[
                    _FakeToolCallDelta(index=0, arguments='{"q": "hello"}'),
                ],
                finish_reason="tool_calls",
            ),
        ]
        # Round 2: text response after tool
        round2 = [_FakeStreamChunk(delta="Done!")]
        provider = _make_stream_provider([round1, round2])

        events = [ev async for ev in run.stream(agent, "search for hello", provider=provider)]

        progress_events = [e for e in events if isinstance(e, MCPProgressEvent)]
        text_events = [e for e in events if isinstance(e, TextEvent)]
        tool_call_events = [e for e in events if isinstance(e, ToolCallEvent)]

        assert len(tool_call_events) == 1
        assert tool_call_events[0].tool_name == tool_name

        assert len(progress_events) == 2
        assert progress_events[0].progress == 1
        assert progress_events[0].total == 2
        assert progress_events[0].message == "step 1"
        assert progress_events[1].progress == 2
        assert progress_events[1].total == 2
        assert progress_events[1].message == "step 2"

        assert len(text_events) == 1
        assert text_events[0].text == "Done!"

    @pytest.mark.asyncio
    async def test_stream_progress_events_have_agent_name(self) -> None:
        """MCPProgressEvent items yielded from stream carry the agent's name."""
        from exo.agent import Agent
        from exo.runner import run
        from exo.types import MCPProgressEvent

        mcp_tool = _make_mcp_tool(name="lookup")

        async def mock_call_fn(
            name: str, args: Any, *, progress_callback: Any = None
        ) -> CallToolResult:
            if progress_callback is not None:
                await progress_callback(5.0, None, "processing")
            return _make_call_result("data")

        wrapper = MCPToolWrapper(mcp_tool, "srv", mock_call_fn)
        agent = Agent(name="myagent", tools=[wrapper], memory=None, context=None)

        tool_name = wrapper.name
        round1 = [
            _FakeStreamChunk(
                tool_call_deltas=[_FakeToolCallDelta(index=0, id="tc1", name=tool_name)]
            ),
            _FakeStreamChunk(
                tool_call_deltas=[_FakeToolCallDelta(index=0, arguments="{}")],
                finish_reason="tool_calls",
            ),
        ]
        round2 = [_FakeStreamChunk(delta="ok")]
        provider = _make_stream_provider([round1, round2])

        events = [ev async for ev in run.stream(agent, "lookup", provider=provider)]

        progress_events = [e for e in events if isinstance(e, MCPProgressEvent)]
        assert len(progress_events) == 1
        assert progress_events[0].agent_name == "myagent"

    @pytest.mark.asyncio
    async def test_stream_no_progress_when_tool_emits_none(self) -> None:
        """When MCP tool emits no progress, stream has no MCPProgressEvent items."""
        from exo.agent import Agent
        from exo.runner import run
        from exo.types import MCPProgressEvent, TextEvent

        mcp_tool = _make_mcp_tool(name="search")
        call_fn = AsyncMock(return_value=_make_call_result("result"))
        wrapper = MCPToolWrapper(mcp_tool, "srv", call_fn)
        agent = Agent(name="bot", tools=[wrapper], memory=None, context=None)

        tool_name = wrapper.name
        round1 = [
            _FakeStreamChunk(
                tool_call_deltas=[_FakeToolCallDelta(index=0, id="tc1", name=tool_name)]
            ),
            _FakeStreamChunk(
                tool_call_deltas=[_FakeToolCallDelta(index=0, arguments='{"q":"x"}')],
                finish_reason="tool_calls",
            ),
        ]
        round2 = [_FakeStreamChunk(delta="done")]
        provider = _make_stream_provider([round1, round2])

        events = [ev async for ev in run.stream(agent, "go", provider=provider)]

        progress_events = [e for e in events if isinstance(e, MCPProgressEvent)]
        assert len(progress_events) == 0
        assert any(isinstance(e, TextEvent) for e in events)

    @pytest.mark.asyncio
    async def test_run_does_not_yield_progress_events(self) -> None:
        """agent.run() (non-streaming) never surfaces MCPProgressEvent — no yield mechanism."""
        from exo.agent import Agent
        from exo.runner import run
        from exo.types import AgentOutput, MCPProgressEvent, ToolCall, Usage

        mcp_tool = _make_mcp_tool(name="search")

        async def mock_call_fn(
            name: str, args: Any, *, progress_callback: Any = None
        ) -> CallToolResult:
            if progress_callback is not None:
                await progress_callback(1.0, 1.0, "done")
            return _make_call_result("result")

        wrapper = MCPToolWrapper(mcp_tool, "srv", mock_call_fn)
        # run() returns RunResult (not a stream) so MCPProgressEvent cannot appear in it
        agent = Agent(name="bot", tools=[wrapper], memory=None, context=None)

        tool_name = wrapper.name
        call_count = 0

        async def complete(messages: Any, **kwargs: Any) -> Any:
            nonlocal call_count

            if call_count == 0:
                call_count += 1
                resp = AgentOutput(
                    text="",
                    tool_calls=[ToolCall(id="tc1", name=tool_name, arguments='{"q":"x"}')],
                    usage=Usage(),
                )
            else:
                resp = AgentOutput(text="done!", tool_calls=[], usage=Usage())

            class FakeResponse:
                content = resp.text
                tool_calls = resp.tool_calls
                usage = resp.usage

            return FakeResponse()

        mock_provider = AsyncMock()
        mock_provider.complete = complete

        result = await run(agent, "search something", provider=mock_provider)

        # run() returns a RunResult, not a stream — MCPProgressEvent items are
        # internal to the tool execution and never appear in the result
        assert result.output == "done!"
        # Progress items are accumulated in wrapper.progress_queue (not surfaced)
        # They won't have been drained since we're in run(), not stream()
        # The queue holds the event; it was captured but not yielded
        assert isinstance(wrapper.progress_queue.get_nowait(), MCPProgressEvent)

    @pytest.mark.asyncio
    async def test_progress_events_come_before_final_text(self) -> None:
        """MCPProgressEvent items appear before the final TextEvent in the stream."""
        from exo.agent import Agent
        from exo.runner import run
        from exo.types import MCPProgressEvent, TextEvent

        mcp_tool = _make_mcp_tool(name="search")

        async def mock_call_fn(
            name: str, args: Any, *, progress_callback: Any = None
        ) -> CallToolResult:
            if progress_callback is not None:
                await progress_callback(1.0, 1.0, "searching")
            return _make_call_result("done")

        wrapper = MCPToolWrapper(mcp_tool, "srv", mock_call_fn)
        agent = Agent(name="bot", tools=[wrapper], memory=None, context=None)

        tool_name = wrapper.name
        round1 = [
            _FakeStreamChunk(
                tool_call_deltas=[_FakeToolCallDelta(index=0, id="tc1", name=tool_name)]
            ),
            _FakeStreamChunk(
                tool_call_deltas=[_FakeToolCallDelta(index=0, arguments='{"q":"y"}')],
                finish_reason="tool_calls",
            ),
        ]
        round2 = [_FakeStreamChunk(delta="Final answer")]
        provider = _make_stream_provider([round1, round2])

        events = [ev async for ev in run.stream(agent, "search", provider=provider)]

        # Find indices of progress and text events
        progress_idx = [i for i, e in enumerate(events) if isinstance(e, MCPProgressEvent)]
        text_idx = [i for i, e in enumerate(events) if isinstance(e, TextEvent)]

        assert len(progress_idx) == 1
        assert len(text_idx) == 1
        # Progress event must appear before the text response
        assert progress_idx[0] < text_idx[0]

    @pytest.mark.asyncio
    async def test_stream_progress_queue_drained_after_each_tool_call_round(self) -> None:
        """After each tool-call round, the progress queue is fully drained."""
        from exo.agent import Agent
        from exo.runner import run
        from exo.types import MCPProgressEvent

        mcp_tool = _make_mcp_tool(name="search")
        call_count = 0

        async def mock_call_fn(
            name: str, args: Any, *, progress_callback: Any = None
        ) -> CallToolResult:
            nonlocal call_count
            call_count += 1
            if progress_callback is not None:
                await progress_callback(float(call_count), 2.0, f"call {call_count}")
            return _make_call_result(f"result {call_count}")

        wrapper = MCPToolWrapper(mcp_tool, "srv", mock_call_fn)
        agent = Agent(name="bot", tools=[wrapper], memory=None, context=None)

        tool_name = wrapper.name

        def _tool_round(args: str) -> list[_FakeStreamChunk]:
            return [
                _FakeStreamChunk(
                    tool_call_deltas=[_FakeToolCallDelta(index=0, id="tc1", name=tool_name)]
                ),
                _FakeStreamChunk(
                    tool_call_deltas=[_FakeToolCallDelta(index=0, arguments=args)],
                    finish_reason="tool_calls",
                ),
            ]

        rounds = [
            _tool_round('{"q":"a"}'),
            _tool_round('{"q":"b"}'),
            [_FakeStreamChunk(delta="final")],
        ]
        provider = _make_stream_provider(rounds)

        events = [ev async for ev in run.stream(agent, "go", provider=provider)]

        progress_events = [e for e in events if isinstance(e, MCPProgressEvent)]
        # Two tool rounds → two progress notifications
        assert len(progress_events) == 2
        assert progress_events[0].message == "call 1"
        assert progress_events[1].message == "call 2"
        # Queue should be fully drained
        assert wrapper.progress_queue.empty()
