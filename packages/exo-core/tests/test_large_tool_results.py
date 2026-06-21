"""Tests for @tool(large_output=True), context-mode gating, and MCP large_output_tools — US-026/US-027."""

from __future__ import annotations

import logging
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from exo.agent import Agent
from exo.models.types import ModelResponse  # pyright: ignore[reportMissingImports]
from exo.tool import FunctionTool, tool
from exo.types import ToolCall, Usage

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_provider(content: str = "done", tool_calls: list[ToolCall] | None = None) -> AsyncMock:
    """Create a mock provider that returns a response, then a final text response."""
    if tool_calls:
        # First call returns tool_calls; second call returns plain text
        tool_resp = ModelResponse(
            content="",
            tool_calls=tool_calls,
            usage=Usage(input_tokens=5, output_tokens=3, total_tokens=8),
        )
        final_resp = ModelResponse(
            content=content,
            tool_calls=[],
            usage=Usage(input_tokens=10, output_tokens=5, total_tokens=15),
        )
        provider = AsyncMock()
        provider.complete = AsyncMock(side_effect=[tool_resp, final_resp])
    else:
        resp = ModelResponse(
            content=content,
            tool_calls=[],
            usage=Usage(input_tokens=5, output_tokens=3, total_tokens=8),
        )
        provider = AsyncMock()
        provider.complete = AsyncMock(return_value=resp)
    return provider


def _make_default_context() -> Any:
    """Create a default context (summarize overflow) that supports offloading."""
    try:
        from exo.context.config import ContextConfig  # pyright: ignore[reportMissingImports]
        from exo.context.context import Context  # pyright: ignore[reportMissingImports]

        config = ContextConfig()
        return Context(task_id="test", config=config)
    except ImportError:
        pytest.skip("exo-context not installed")


def _make_hook_context() -> Any:
    """Create a hook-based context (overflow=hook) that does NOT support offloading."""
    try:
        from exo.context.config import ContextConfig  # pyright: ignore[reportMissingImports]
        from exo.context.context import Context  # pyright: ignore[reportMissingImports]

        config = ContextConfig(overflow="hook")
        return Context(task_id="test", config=config)
    except ImportError:
        pytest.skip("exo-context not installed")


# ---------------------------------------------------------------------------
# FunctionTool large_output attribute
# ---------------------------------------------------------------------------


class TestFunctionToolLargeOutput:
    def test_large_output_false_by_default(self) -> None:
        """FunctionTool.large_output is False when not specified."""

        @tool
        def my_tool() -> str:
            """A tool."""
            return "result"

        assert my_tool.large_output is False

    def test_large_output_true_via_decorator_kwarg(self) -> None:
        """@tool(large_output=True) sets FunctionTool.large_output=True."""

        @tool(large_output=True)
        def big_tool() -> str:
            """Returns a lot of data."""
            return "x" * 9999

        assert big_tool.large_output is True

    def test_large_output_false_explicit(self) -> None:
        """@tool(large_output=False) is the same as default."""

        @tool(large_output=False)
        def small_tool() -> str:
            """Small tool."""
            return "tiny"

        assert small_tool.large_output is False

    def test_function_tool_direct_large_output(self) -> None:
        """FunctionTool(fn, large_output=True) stores the flag."""

        def fn() -> str:
            return "data"

        ft = FunctionTool(fn, large_output=True)
        assert ft.large_output is True

    def test_function_tool_default_no_large_output_attr(self) -> None:
        """FunctionTool without large_output has large_output=False."""

        def fn() -> str:
            return "data"

        ft = FunctionTool(fn)
        assert ft.large_output is False


# ---------------------------------------------------------------------------
# Context-mode gating: _should_enable_artifact_offloading
# ---------------------------------------------------------------------------


class TestShouldEnableArtifactOffloading:
    def test_disabled_when_context_is_none(self) -> None:
        """Offloading is disabled when agent has no context."""
        agent = Agent(name="bot", memory=None, context=None)
        assert agent._should_enable_artifact_offloading() is False

    def test_disabled_when_overflow_is_hook(self) -> None:
        """Offloading is disabled when overflow strategy is hook-based."""
        ctx = _make_hook_context()
        agent = Agent(name="bot", memory=None, context=ctx)
        assert agent._should_enable_artifact_offloading() is False

    def test_enabled_with_default_context(self) -> None:
        """Offloading is enabled with a default (summarize) context."""
        ctx = _make_default_context()
        agent = Agent(name="bot", memory=None, context=ctx)
        assert agent._should_enable_artifact_offloading() is True


# ---------------------------------------------------------------------------
# retrieve_artifact auto-registration (context-mode gated)
# ---------------------------------------------------------------------------


class TestRetrieveArtifactAutoRegister:
    def test_retrieve_artifact_not_registered_when_context_is_none(self) -> None:
        """retrieve_artifact is NOT registered when context is None."""

        @tool
        def normal() -> str:
            """Normal tool."""
            return "ok"

        agent = Agent(name="bot", memory=None, context=None, tools=[normal])
        assert "retrieve_artifact" not in agent.tools

    def test_retrieve_artifact_not_registered_when_overflow_is_hook(self) -> None:
        """retrieve_artifact is NOT registered when overflow=hook."""
        ctx = _make_hook_context()

        @tool
        def normal() -> str:
            """Normal tool."""
            return "ok"

        agent = Agent(name="bot", memory=None, context=ctx, tools=[normal])
        assert "retrieve_artifact" not in agent.tools

    def test_retrieve_artifact_registered_with_default_context(self) -> None:
        """retrieve_artifact IS registered with a default (summarize) context."""
        ctx = _make_default_context()

        @tool
        def normal() -> str:
            """Normal tool."""
            return "ok"

        agent = Agent(name="bot", memory=None, context=ctx, tools=[normal])
        assert "retrieve_artifact" in agent.tools

    def test_retrieve_artifact_is_function_tool(self) -> None:
        """retrieve_artifact is a FunctionTool."""
        ctx = _make_default_context()

        @tool(large_output=True)
        def big() -> str:
            """Big tool."""
            return "x" * 50000

        agent = Agent(name="bot", memory=None, context=ctx, tools=[big])
        ra = agent.tools["retrieve_artifact"]
        assert isinstance(ra, FunctionTool)

    def test_retrieve_artifact_registered_once_for_multiple_large_tools(self) -> None:
        """Only one retrieve_artifact is registered even with multiple large_output tools."""
        ctx = _make_default_context()

        @tool(large_output=True)
        def big1() -> str:
            """Big tool 1."""
            return "a" * 50000

        @tool(large_output=True)
        def big2() -> str:
            """Big tool 2."""
            return "b" * 50000

        agent = Agent(name="bot", memory=None, context=ctx, tools=[big1, big2])
        # Only one retrieve_artifact, not duplicated
        assert list(agent.tools.keys()).count("retrieve_artifact") == 1

    @pytest.mark.asyncio
    async def test_add_tool_does_not_duplicate_retrieve_artifact(self) -> None:
        """add_tool does not create a second retrieve_artifact."""
        ctx = _make_default_context()

        @tool(large_output=True)
        def big() -> str:
            """Big tool."""
            return "x" * 50000

        agent = Agent(name="bot", memory=None, context=ctx)
        assert "retrieve_artifact" in agent.tools
        await agent.add_tool(big)
        # Still only one retrieve_artifact
        assert list(agent.tools.keys()).count("retrieve_artifact") == 1


# ---------------------------------------------------------------------------
# _offload_large_result / workspace usage
# ---------------------------------------------------------------------------


class TestOffloadLargeResult:
    @pytest.mark.asyncio
    async def test_offload_creates_workspace_lazily(self) -> None:
        """_offload_large_result lazily creates _workspace on first call."""
        agent = Agent(name="bot", memory=None, context=None)
        assert agent._workspace is None
        await agent._offload_large_result("my_tool", "some content")
        assert agent._workspace is not None

    @pytest.mark.asyncio
    async def test_offload_returns_pointer_string(self) -> None:
        """_offload_large_result returns the expected pointer string."""
        agent = Agent(name="bot", memory=None, context=None)
        result = await agent._offload_large_result("my_tool", "large data")
        assert "retrieve_artifact(" in result
        assert "tool_result_my_tool_" in result
        assert "Result stored as artifact" in result

    @pytest.mark.asyncio
    async def test_offload_stores_content_in_workspace(self) -> None:
        """_offload_large_result stores the full content in the workspace."""
        agent = Agent(name="bot", memory=None, context=None)
        content = "x" * 20000
        pointer = await agent._offload_large_result("big_tool", content)
        # Extract artifact_id from pointer string
        # Format: [Result stored as artifact 'ARTIFACT_ID'. Call retrieve_artifact(...)]
        import re

        match = re.search(r"artifact '([^']+)'", pointer)
        assert match is not None
        artifact_id = match.group(1)
        assert agent._workspace is not None
        stored = agent._workspace.read(artifact_id)
        assert stored == content

    @pytest.mark.asyncio
    async def test_offload_emits_debug_log(self, caplog: pytest.LogCaptureFixture) -> None:
        """_offload_large_result emits the expected debug log."""
        agent = Agent(name="bot", memory=None, context=None)
        content = "hello world"
        with caplog.at_level(logging.DEBUG, logger="exo.agent"):
            await agent._offload_large_result("test_tool", content)
        assert any(
            "ToolResultOffloader: offloading test_tool result" in record.message
            for record in caplog.records
        )
        assert any(
            "artifact_id=" in record.message or "tool_result_test_tool_" in record.message
            for record in caplog.records
        )

    @pytest.mark.asyncio
    async def test_offload_fallback_when_context_not_installed(self) -> None:
        """When exo-context is not importable, returns content unchanged."""
        import sys

        agent = Agent(name="bot", memory=None, context=None)
        original = sys.modules.get("exo.context.workspace")
        sys.modules["exo.context.workspace"] = None  # type: ignore[assignment]
        try:
            content = "some data"
            result = await agent._offload_large_result("tool", content)
            assert result == content
        finally:
            if original is None:
                sys.modules.pop("exo.context.workspace", None)
            else:
                sys.modules["exo.context.workspace"] = original


# ---------------------------------------------------------------------------
# End-to-end: large_output tool in agent run
# ---------------------------------------------------------------------------


class TestLargeOutputEndToEnd:
    @pytest.mark.asyncio
    async def test_large_output_tool_result_is_pointer_in_context(self) -> None:
        """When a large_output=True tool is called, the LLM sees the pointer, not the content."""
        ctx = _make_default_context()
        large_content = "X" * 20000

        @tool(large_output=True)
        def fetch_data() -> str:
            """Fetch a large dataset."""
            return large_content

        tool_calls = [ToolCall(id="tc1", name="fetch_data", arguments="{}")]
        provider = _mock_provider(content="Analysis complete.", tool_calls=tool_calls)

        agent = Agent(name="bot", memory=None, context=ctx, tools=[fetch_data])
        await agent.run("fetch the data", provider=provider)

        # The second LLM call receives the pointer, not the raw large content
        second_call_messages = provider.complete.call_args_list[1][0][0]
        from exo.types import ToolResult

        tool_result_msgs = [m for m in second_call_messages if isinstance(m, ToolResult)]
        assert len(tool_result_msgs) == 1
        assert "Result stored as artifact" in tool_result_msgs[0].content
        assert large_content not in tool_result_msgs[0].content

    @pytest.mark.asyncio
    async def test_retrieve_artifact_returns_full_content(self) -> None:
        """retrieve_artifact tool retrieves the full offloaded content."""
        ctx = _make_default_context()
        large_content = "big data " * 5000

        @tool(large_output=True)
        def produce() -> str:
            """Produce large data."""
            return large_content

        # First run: store the artifact
        tool_calls = [ToolCall(id="tc1", name="produce", arguments="{}")]
        provider = _mock_provider(content="done", tool_calls=tool_calls)
        agent = Agent(name="bot", memory=None, context=ctx, tools=[produce])
        await agent.run("produce data", provider=provider)

        # Now call retrieve_artifact directly to check content
        import re

        second_call_messages = provider.complete.call_args_list[1][0][0]
        from exo.types import ToolResult

        tool_result_msgs = [m for m in second_call_messages if isinstance(m, ToolResult)]
        pointer = tool_result_msgs[0].content
        match = re.search(r"artifact '([^']+)'", pointer)
        assert match is not None
        artifact_id = match.group(1)

        # Retrieve via the tool
        ra_tool = agent.tools["retrieve_artifact"]
        retrieved = await ra_tool.execute(id=artifact_id)
        assert retrieved == large_content

    @pytest.mark.asyncio
    async def test_retrieve_artifact_unknown_id(self) -> None:
        """retrieve_artifact returns error string for unknown artifact ID."""
        ctx = _make_default_context()

        @tool(large_output=True)
        def big() -> str:
            """Big tool."""
            return "x"

        agent = Agent(name="bot", memory=None, context=ctx, tools=[big])
        # Pre-create workspace so retrieve_artifact doesn't get "no workspace" error
        from exo.context.workspace import Workspace  # pyright: ignore[reportMissingImports]

        agent._workspace = Workspace(workspace_id="test")
        ra_tool = agent.tools["retrieve_artifact"]
        result = await ra_tool.execute(id="nonexistent_id")
        assert "not found" in result

    @pytest.mark.asyncio
    async def test_retrieve_artifact_no_workspace_yet(self) -> None:
        """retrieve_artifact returns graceful error when no workspace has been created."""
        ctx = _make_default_context()

        @tool(large_output=True)
        def big() -> str:
            """Big tool."""
            return "x"

        agent = Agent(name="bot", memory=None, context=ctx, tools=[big])
        # Workspace not yet created
        assert agent._workspace is None
        ra_tool = agent.tools["retrieve_artifact"]
        result = await ra_tool.execute(id="any_id")
        assert "No workspace available" in result

    @pytest.mark.asyncio
    async def test_normal_tool_not_offloaded(self) -> None:
        """A tool without large_output=True passes its result directly to LLM."""
        normal_content = "short result"

        @tool
        def simple() -> str:
            """Simple tool."""
            return normal_content

        tool_calls = [ToolCall(id="tc1", name="simple", arguments="{}")]
        provider = _mock_provider(content="ok", tool_calls=tool_calls)
        agent = Agent(name="bot", memory=None, context=None, tools=[simple])
        await agent.run("run simple", provider=provider)

        second_call_messages = provider.complete.call_args_list[1][0][0]
        from exo.types import ToolResult

        tool_result_msgs = [m for m in second_call_messages if isinstance(m, ToolResult)]
        assert len(tool_result_msgs) == 1
        assert tool_result_msgs[0].content == normal_content

    @pytest.mark.asyncio
    async def test_large_output_not_offloaded_when_context_is_none(self) -> None:
        """large_output=True tool result is NOT offloaded when context=None."""
        large_content = "X" * 20000

        @tool(large_output=True)
        def fetch_data() -> str:
            """Fetch a large dataset."""
            return large_content

        tool_calls = [ToolCall(id="tc1", name="fetch_data", arguments="{}")]
        provider = _mock_provider(content="done", tool_calls=tool_calls)

        agent = Agent(name="bot", memory=None, context=None, tools=[fetch_data])
        await agent.run("fetch the data", provider=provider)

        second_call_messages = provider.complete.call_args_list[1][0][0]
        from exo.types import ToolResult

        tool_result_msgs = [m for m in second_call_messages if isinstance(m, ToolResult)]
        assert len(tool_result_msgs) == 1
        # Content should pass through unchanged — no offloading
        assert tool_result_msgs[0].content == large_content

    @pytest.mark.asyncio
    async def test_large_output_not_offloaded_when_overflow_is_hook(self) -> None:
        """large_output=True tool result is NOT offloaded when overflow=hook."""
        ctx = _make_hook_context()
        large_content = "X" * 20000

        @tool(large_output=True)
        def fetch_data() -> str:
            """Fetch a large dataset."""
            return large_content

        tool_calls = [ToolCall(id="tc1", name="fetch_data", arguments="{}")]
        provider = _mock_provider(content="done", tool_calls=tool_calls)

        agent = Agent(name="bot", memory=None, context=ctx, tools=[fetch_data])
        await agent.run("fetch the data", provider=provider)

        second_call_messages = provider.complete.call_args_list[1][0][0]
        from exo.types import ToolResult

        tool_result_msgs = [m for m in second_call_messages if isinstance(m, ToolResult)]
        assert len(tool_result_msgs) == 1
        # Content should pass through unchanged — no offloading
        assert tool_result_msgs[0].content == large_content


# ---------------------------------------------------------------------------
# ToolResultOffloader large_output flag (processor.py)
# ---------------------------------------------------------------------------


class TestToolResultOffloaderLargeOutputFlag:
    @pytest.mark.asyncio
    async def test_offloader_fires_on_large_output_flag_regardless_of_size(self) -> None:
        """ToolResultOffloader offloads when large_output=True even if content is small."""
        from exo.context.config import ContextConfig  # pyright: ignore[reportMissingImports]
        from exo.context.context import Context  # pyright: ignore[reportMissingImports]
        from exo.context.processor import (
            ToolResultOffloader,  # pyright: ignore[reportMissingImports]
        )

        config = ContextConfig()
        ctx = Context(task_id="t1", config=config)
        offloader = ToolResultOffloader(max_size=50000)

        payload: dict[str, Any] = {
            "tool_result": "tiny result",  # well under max_size
            "tool_name": "my_tool",
            "tool_call_id": "tc1",
            "large_output": True,
        }
        await offloader.process(ctx, payload)
        # Result should be replaced with a reference
        assert payload["tool_result"] != "tiny result"
        assert (
            "offloaded" in payload["tool_result"].lower()
            or "Result stored" in payload["tool_result"]
            or "truncated" in payload["tool_result"].lower()
        )

    @pytest.mark.asyncio
    async def test_offloader_skips_small_non_large_output(self) -> None:
        """ToolResultOffloader does not fire for small results without large_output=True."""
        from exo.context.config import ContextConfig  # pyright: ignore[reportMissingImports]
        from exo.context.context import Context  # pyright: ignore[reportMissingImports]
        from exo.context.processor import (
            ToolResultOffloader,  # pyright: ignore[reportMissingImports]
        )

        config = ContextConfig()
        ctx = Context(task_id="t1", config=config)
        offloader = ToolResultOffloader(max_size=5000)

        payload: dict[str, Any] = {
            "tool_result": "small result",
            "tool_name": "my_tool",
            "tool_call_id": "tc1",
            "large_output": False,
        }
        await offloader.process(ctx, payload)
        # Result should be unchanged
        assert payload["tool_result"] == "small result"

    @pytest.mark.asyncio
    async def test_offloader_with_artifact_id_uses_pointer_format(self) -> None:
        """When payload has artifact_id, ToolResultOffloader uses pointer format."""
        from exo.context.config import ContextConfig  # pyright: ignore[reportMissingImports]
        from exo.context.context import Context  # pyright: ignore[reportMissingImports]
        from exo.context.processor import (
            ToolResultOffloader,  # pyright: ignore[reportMissingImports]
        )

        config = ContextConfig()
        ctx = Context(task_id="t1", config=config)
        offloader = ToolResultOffloader(max_size=5)

        payload: dict[str, Any] = {
            "tool_result": "hello world big content",
            "tool_name": "my_tool",
            "tool_call_id": "tc1",
            "artifact_id": "artifact_abc123",
        }
        await offloader.process(ctx, payload)
        assert "artifact_abc123" in payload["tool_result"]
        assert "retrieve_artifact" in payload["tool_result"]

    @pytest.mark.asyncio
    async def test_offloader_logs_with_artifact_id(self, caplog: pytest.LogCaptureFixture) -> None:
        """ToolResultOffloader logs the correct format including artifact_id."""
        from exo.context.config import ContextConfig  # pyright: ignore[reportMissingImports]
        from exo.context.context import Context  # pyright: ignore[reportMissingImports]
        from exo.context.processor import (
            ToolResultOffloader,  # pyright: ignore[reportMissingImports]
        )

        config = ContextConfig()
        ctx = Context(task_id="t1", config=config)
        offloader = ToolResultOffloader(max_size=5)

        payload: dict[str, Any] = {
            "tool_result": "hello world",
            "tool_name": "search_tool",
            "tool_call_id": "tc1",
            "artifact_id": "my_artifact_id",
        }
        with caplog.at_level(logging.DEBUG, logger="exo.context.processor"):
            await offloader.process(ctx, payload)

        assert any(
            "ToolResultOffloader" in r.message and "search_tool" in r.message
            for r in caplog.records
        )
        assert any("artifact_id" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# US-027: MCPServerConfig.large_output_tools + MCPToolWrapper.large_output
# ---------------------------------------------------------------------------


class TestMCPLargeOutputTools:
    def test_mcp_server_config_large_output_tools_default(self) -> None:
        """MCPServerConfig.large_output_tools defaults to empty list."""
        try:
            from exo.mcp.client import MCPServerConfig  # pyright: ignore[reportMissingImports]
        except ImportError:
            pytest.skip("exo-mcp not installed")

        config = MCPServerConfig(name="test", command="echo")
        assert config.large_output_tools == []

    def test_mcp_server_config_large_output_tools_set(self) -> None:
        """MCPServerConfig accepts large_output_tools list."""
        try:
            from exo.mcp.client import MCPServerConfig  # pyright: ignore[reportMissingImports]
        except ImportError:
            pytest.skip("exo-mcp not installed")

        config = MCPServerConfig(
            name="test", command="echo", large_output_tools=["search", "read_file"]
        )
        assert config.large_output_tools == ["search", "read_file"]

    def test_mcp_server_config_serializes_large_output_tools(self) -> None:
        """MCPServerConfig.to_dict() includes large_output_tools."""
        try:
            from exo.mcp.client import MCPServerConfig  # pyright: ignore[reportMissingImports]
        except ImportError:
            pytest.skip("exo-mcp not installed")

        config = MCPServerConfig(name="test", command="echo", large_output_tools=["tool1"])
        d = config.to_dict()
        assert d["large_output_tools"] == ["tool1"]

    def test_mcp_server_config_round_trips_large_output_tools(self) -> None:
        """MCPServerConfig from_dict() restores large_output_tools."""
        try:
            from exo.mcp.client import MCPServerConfig  # pyright: ignore[reportMissingImports]
        except ImportError:
            pytest.skip("exo-mcp not installed")

        config = MCPServerConfig(
            name="test", command="echo", large_output_tools=["tool1", "tool2"]
        )
        restored = MCPServerConfig.from_dict(config.to_dict())
        assert restored.large_output_tools == ["tool1", "tool2"]

    def test_mcp_tool_wrapper_large_output_from_config(self) -> None:
        """MCPToolWrapper.large_output is True when tool name is in large_output_tools."""
        try:
            from exo.mcp.client import MCPServerConfig  # pyright: ignore[reportMissingImports]
            from exo.mcp.tools import MCPToolWrapper  # pyright: ignore[reportMissingImports]
        except ImportError:
            pytest.skip("exo-mcp not installed")

        mcp_tool = MagicMock()
        mcp_tool.name = "search"
        mcp_tool.description = "Search tool"
        mcp_tool.inputSchema = {"type": "object", "properties": {}}

        config = MCPServerConfig(name="srv", command="echo", large_output_tools=["search"])
        wrapper = MCPToolWrapper(mcp_tool, "srv", MagicMock(), server_config=config)
        assert wrapper.large_output is True

    def test_mcp_tool_wrapper_large_output_false_when_not_listed(self) -> None:
        """MCPToolWrapper.large_output is False when tool name is not in large_output_tools."""
        try:
            from exo.mcp.client import MCPServerConfig  # pyright: ignore[reportMissingImports]
            from exo.mcp.tools import MCPToolWrapper  # pyright: ignore[reportMissingImports]
        except ImportError:
            pytest.skip("exo-mcp not installed")

        mcp_tool = MagicMock()
        mcp_tool.name = "list_files"
        mcp_tool.description = "List files"
        mcp_tool.inputSchema = {"type": "object", "properties": {}}

        config = MCPServerConfig(name="srv", command="echo", large_output_tools=["search"])
        wrapper = MCPToolWrapper(mcp_tool, "srv", MagicMock(), server_config=config)
        assert wrapper.large_output is False

    def test_mcp_tool_wrapper_large_output_false_when_no_config(self) -> None:
        """MCPToolWrapper.large_output is False when no server_config is provided."""
        try:
            from exo.mcp.tools import MCPToolWrapper  # pyright: ignore[reportMissingImports]
        except ImportError:
            pytest.skip("exo-mcp not installed")

        mcp_tool = MagicMock()
        mcp_tool.name = "search"
        mcp_tool.description = "Search"
        mcp_tool.inputSchema = {"type": "object", "properties": {}}

        wrapper = MCPToolWrapper(mcp_tool, "srv", MagicMock())
        assert wrapper.large_output is False

    def test_mcp_tool_wrapper_large_output_round_trips(self) -> None:
        """MCPToolWrapper.large_output=True round-trips through to_dict/from_dict."""
        try:
            from exo.mcp.client import MCPServerConfig  # pyright: ignore[reportMissingImports]
            from exo.mcp.tools import MCPToolWrapper  # pyright: ignore[reportMissingImports]
        except ImportError:
            pytest.skip("exo-mcp not installed")

        mcp_tool = MagicMock()
        mcp_tool.name = "fetch"
        mcp_tool.description = "Fetch data"
        mcp_tool.inputSchema = {"type": "object", "properties": {}}

        config = MCPServerConfig(name="srv", command="echo", large_output_tools=["fetch"])
        wrapper = MCPToolWrapper(mcp_tool, "srv", MagicMock(), server_config=config)
        assert wrapper.large_output is True

        restored = MCPToolWrapper.from_dict(wrapper.to_dict())
        assert restored.large_output is True
