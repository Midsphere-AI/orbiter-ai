"""Tests for Agent and Swarm serialization/deserialization (US-015)."""

from __future__ import annotations

import json
from typing import Any

import pytest
from pydantic import BaseModel

from exo.agent import Agent, AgentError, _deserialize_tool, _import_object, _serialize_tool
from exo.swarm import Swarm
from exo.tool import FunctionTool, Tool, tool

# ---------------------------------------------------------------------------
# Module-level fixtures (importable for serialization round-trips)
# ---------------------------------------------------------------------------


@tool
def search(query: str) -> str:
    """Search for something."""
    return f"Results for: {query}"


@tool
def calculate(expression: str) -> str:
    """Evaluate a math expression."""
    return f"Result: {expression}"


class SummaryOutput(BaseModel):
    title: str
    body: str


# A custom Tool subclass at module level
class CustomSearchTool(Tool):
    def __init__(self) -> None:
        self.name = "custom_search"
        self.description = "Custom search tool"
        self.parameters = {"type": "object", "properties": {}}

    async def execute(self, **kwargs: Any) -> str:
        return "custom result"


# ---------------------------------------------------------------------------
# Agent.to_dict() tests
# ---------------------------------------------------------------------------


class TestAgentToDict:
    def test_minimal_agent(self) -> None:
        """Minimal agent serializes with default values."""
        agent = Agent(name="bot", model="openai:gpt-4o")
        data = agent.to_dict()
        assert data["name"] == "bot"
        assert data["model"] == "openai:gpt-4o"
        assert data["instructions"] == ""
        assert data["max_steps"] == 10
        assert data["temperature"] == 1.0
        assert data["max_tokens"] is None
        assert data["planning_enabled"] is False
        assert data["planning_model"] is None
        assert data["planning_instructions"] == ""
        assert data["budget_awareness"] is None
        assert data["hitl_tools"] == []
        assert data["emit_mcp_progress"] is True
        assert data["injected_tool_args"] == {}
        assert "allow_parallel_subagents" not in data
        assert "max_parallel_subagents" not in data
        assert "tools" not in data
        assert "handoffs" not in data
        assert "output_type" not in data

    def test_with_runtime_controls(self) -> None:
        """New runtime-control fields serialize without losing values."""
        agent = Agent(
            name="planner",
            tools=[search, calculate],
            planning_enabled=True,
            planning_model="openai:gpt-4o-mini",
            planning_instructions="Return a short numbered plan.",
            budget_awareness="limit:70",
            hitl_tools=["search"],
            emit_mcp_progress=False,
            injected_tool_args={"ui_request_id": "Opaque request id"},
        )
        data = agent.to_dict()
        assert data["planning_enabled"] is True
        assert data["planning_model"] == "openai:gpt-4o-mini"
        assert data["planning_instructions"] == "Return a short numbered plan."
        assert data["budget_awareness"] == "limit:70"
        assert data["hitl_tools"] == ["search"]
        assert data["emit_mcp_progress"] is False
        assert data["injected_tool_args"] == {"ui_request_id": "Opaque request id"}

    def test_with_tools(self) -> None:
        """Agent with tools serializes tool paths."""
        agent = Agent(name="bot", tools=[search, calculate])
        data = agent.to_dict()
        assert "tools" in data
        assert len(data["tools"]) == 2
        # Tools should be importable dotted paths
        for path in data["tools"]:
            assert isinstance(path, str)
            assert "." in path

    def test_with_handoffs(self) -> None:
        """Agent with handoffs serializes them recursively."""
        target = Agent(name="target", model="openai:gpt-4o-mini")
        agent = Agent(name="router", handoffs=[target])
        data = agent.to_dict()
        assert "handoffs" in data
        assert len(data["handoffs"]) == 1
        assert data["handoffs"][0]["name"] == "target"
        assert data["handoffs"][0]["model"] == "openai:gpt-4o-mini"

    def test_with_output_type(self) -> None:
        """Agent with output_type serializes it as importable path."""
        agent = Agent(name="bot", output_type=SummaryOutput)
        data = agent.to_dict()
        assert "output_type" in data
        assert "SummaryOutput" in data["output_type"]

    def test_with_custom_tool_subclass(self) -> None:
        """Agent with a custom Tool subclass serializes its class path."""
        custom = CustomSearchTool()
        agent = Agent(name="bot", tools=[custom])
        data = agent.to_dict()
        assert len(data["tools"]) == 1
        assert "CustomSearchTool" in data["tools"][0]

    def test_callable_instructions_raises(self) -> None:
        """Callable instructions cannot be serialized."""
        agent = Agent(name="bot", instructions=lambda name: "You are helpful")
        with pytest.raises(ValueError, match="callable instructions"):
            agent.to_dict()

    def test_hooks_raises(self) -> None:
        """Agent with hooks cannot be serialized."""

        async def my_hook(**kwargs: Any) -> None:
            pass

        from exo.hooks import HookPoint

        agent = Agent(name="bot", hooks=[(HookPoint.START, my_hook)])
        with pytest.raises(ValueError, match="hooks"):
            agent.to_dict()

    def test_memory_raises(self) -> None:
        """Agent with memory cannot be serialized."""
        agent = Agent(name="bot", memory="some_memory")
        with pytest.raises(ValueError, match="memory"):
            agent.to_dict()

    def test_context_raises(self) -> None:
        """Agent with context cannot be serialized."""
        agent = Agent(name="bot", context="some_context")
        with pytest.raises(ValueError, match="context"):
            agent.to_dict()

    def test_closure_tool_raises(self) -> None:
        """Closure-based tools cannot be serialized."""

        def make_tool() -> FunctionTool:
            x = 42

            def my_closure(query: str) -> str:
                return f"{x}: {query}"

            return FunctionTool(my_closure)

        agent = Agent(name="bot", tools=[make_tool()])
        with pytest.raises(ValueError, match="closure or lambda"):
            agent.to_dict()

    def test_json_serializable(self) -> None:
        """to_dict() output is JSON-serializable."""
        agent = Agent(name="bot", tools=[search], instructions="Be helpful")
        data = agent.to_dict()
        json_str = json.dumps(data)
        assert isinstance(json_str, str)
        parsed = json.loads(json_str)
        assert parsed["name"] == "bot"


# ---------------------------------------------------------------------------
# Agent.from_dict() tests
# ---------------------------------------------------------------------------


class TestAgentFromDict:
    def test_minimal_reconstruction(self) -> None:
        """from_dict() reconstructs a minimal agent."""
        data = {"name": "bot", "model": "openai:gpt-4o"}
        agent = Agent.from_dict(data)
        assert agent.name == "bot"
        assert agent.model == "openai:gpt-4o"
        assert agent.max_steps == 10
        assert agent.planning_enabled is False
        assert agent.planning_model is None
        assert agent.planning_instructions == ""
        assert agent.budget_awareness is None
        assert agent.hitl_tools == []
        assert agent.emit_mcp_progress is True
        assert agent.injected_tool_args == {}
        assert not hasattr(agent, "allow_parallel_subagents")

    def test_with_all_fields(self) -> None:
        """from_dict() reconstructs agent with all scalar fields."""
        data = {
            "name": "assistant",
            "model": "anthropic:claude-3",
            "instructions": "You are a helpful assistant",
            "max_steps": 5,
            "temperature": 0.7,
            "max_tokens": 1000,
        }
        agent = Agent.from_dict(data)
        assert agent.name == "assistant"
        assert agent.model == "anthropic:claude-3"
        assert agent.instructions == "You are a helpful assistant"
        assert agent.max_steps == 5
        assert agent.temperature == 0.7
        assert agent.max_tokens == 1000

    def test_with_runtime_controls(self) -> None:
        """from_dict() reconstructs the new runtime-control fields."""
        path = f"{search._fn.__module__}.{search._fn.__qualname__}"
        data = {
            "name": "planner",
            "tools": [path],
            "planning_enabled": True,
            "planning_model": "openai:gpt-4o-mini",
            "planning_instructions": "Return a short numbered plan.",
            "budget_awareness": "per-message",
            "hitl_tools": ["search"],
            "emit_mcp_progress": False,
            "injected_tool_args": {"ui_request_id": "Opaque request id"},
            # Legacy no-op keys must be tolerated (silently ignored) by from_dict.
            "allow_parallel_subagents": True,
            "max_parallel_subagents": 4,
        }
        agent = Agent.from_dict(data)
        assert agent.planning_enabled is True
        assert agent.planning_model == "openai:gpt-4o-mini"
        assert agent.planning_instructions == "Return a short numbered plan."
        assert agent.budget_awareness == "per-message"
        assert agent.hitl_tools == ["search"]
        assert agent.emit_mcp_progress is False
        assert agent.injected_tool_args == {"ui_request_id": "Opaque request id"}
        assert not hasattr(agent, "allow_parallel_subagents")

    def test_with_tools_from_path(self) -> None:
        """from_dict() resolves tool paths to actual tools."""
        path = f"{search._fn.__module__}.{search._fn.__qualname__}"
        data = {"name": "bot", "tools": [path]}
        agent = Agent.from_dict(data)
        # search + retrieve_artifact + spawn_self + 3 background + 3 planning tools
        assert len(agent.tools) == 9
        assert "search" in agent.tools

    def test_with_handoffs(self) -> None:
        """from_dict() reconstructs handoff agents recursively."""
        data = {
            "name": "router",
            "handoffs": [{"name": "target", "model": "openai:gpt-4o-mini"}],
        }
        agent = Agent.from_dict(data)
        assert len(agent.handoffs) == 1
        assert "target" in agent.handoffs
        assert agent.handoffs["target"].model == "openai:gpt-4o-mini"

    def test_with_output_type(self) -> None:
        """from_dict() resolves output_type from importable path."""
        path = f"{SummaryOutput.__module__}.{SummaryOutput.__qualname__}"
        data = {"name": "bot", "output_type": path}
        agent = Agent.from_dict(data)
        assert agent.output_type is SummaryOutput

    def test_invalid_tool_path_raises(self) -> None:
        """from_dict() raises on unresolvable tool path."""
        data = {"name": "bot", "tools": ["nonexistent.module.func"]}
        with pytest.raises(ValueError, match="Cannot import"):
            Agent.from_dict(data)

    def test_invalid_budget_awareness_raises(self) -> None:
        """Malformed budget-awareness strings fail during deserialization."""
        data = {"name": "bot", "budget_awareness": "limit:abc"}
        with pytest.raises(ValueError, match="budget_awareness"):
            Agent.from_dict(data)

    def test_unknown_hitl_tool_raises(self) -> None:
        """Unknown HITL tool names fail during deserialization."""
        path = f"{search._fn.__module__}.{search._fn.__qualname__}"
        data = {"name": "bot", "tools": [path], "hitl_tools": ["deploy_service"]}
        with pytest.raises(AgentError, match="hitl_tools contains unknown tool names"):
            Agent.from_dict(data)


# ---------------------------------------------------------------------------
# Round-trip tests
# ---------------------------------------------------------------------------


class TestAgentRoundTrip:
    def test_minimal_round_trip(self) -> None:
        """Minimal agent survives to_dict/from_dict round-trip."""
        original = Agent(name="bot", model="openai:gpt-4o")
        reconstructed = Agent.from_dict(original.to_dict())
        assert reconstructed.name == original.name
        assert reconstructed.model == original.model
        assert reconstructed.max_steps == original.max_steps

    def test_full_round_trip(self) -> None:
        """Agent with tools, handoffs, and output_type round-trips correctly."""
        target = Agent(name="helper", model="openai:gpt-4o-mini")
        original = Agent(
            name="main",
            model="anthropic:claude-3",
            instructions="Be helpful",
            tools=[search, calculate],
            handoffs=[target],
            output_type=SummaryOutput,
            max_steps=5,
            temperature=0.7,
            max_tokens=2000,
            planning_enabled=True,
            planning_model="openai:gpt-4o-mini",
            planning_instructions="Return a short numbered plan.",
            budget_awareness="limit:70",
            hitl_tools=["search"],
            emit_mcp_progress=False,
            injected_tool_args={"ui_request_id": "Opaque request id"},
        )
        data = original.to_dict()
        reconstructed = Agent.from_dict(data)

        assert reconstructed.name == original.name
        assert reconstructed.model == original.model
        assert reconstructed.instructions == original.instructions
        assert reconstructed.max_steps == original.max_steps
        assert reconstructed.temperature == original.temperature
        assert reconstructed.max_tokens == original.max_tokens
        assert reconstructed.planning_enabled is original.planning_enabled
        assert reconstructed.planning_model == original.planning_model
        assert reconstructed.planning_instructions == original.planning_instructions
        assert reconstructed.budget_awareness == original.budget_awareness
        assert reconstructed.hitl_tools == original.hitl_tools
        assert reconstructed.emit_mcp_progress is original.emit_mcp_progress
        assert reconstructed.injected_tool_args == original.injected_tool_args
        assert set(reconstructed.tools.keys()) == set(original.tools.keys())
        assert set(reconstructed.handoffs.keys()) == set(original.handoffs.keys())
        assert reconstructed.output_type is original.output_type

    def test_json_round_trip(self) -> None:
        """Agent survives JSON serialization/deserialization round-trip."""
        original = Agent(
            name="bot",
            model="openai:gpt-4o",
            tools=[search],
            instructions="Be concise",
        )
        json_str = json.dumps(original.to_dict())
        data = json.loads(json_str)
        reconstructed = Agent.from_dict(data)
        assert reconstructed.name == original.name
        assert "search" in reconstructed.tools

    def test_custom_tool_round_trip(self) -> None:
        """Custom Tool subclass round-trips via class path."""
        custom = CustomSearchTool()
        original = Agent(name="bot", tools=[custom])
        data = original.to_dict()
        reconstructed = Agent.from_dict(data)
        assert "custom_search" in reconstructed.tools


# ---------------------------------------------------------------------------
# Swarm serialization tests
# ---------------------------------------------------------------------------


class TestSwarmToDict:
    def test_basic_swarm(self) -> None:
        """Basic swarm serializes agents, flow, and mode."""
        a = Agent(name="a", model="openai:gpt-4o")
        b = Agent(name="b", model="openai:gpt-4o")
        swarm = Swarm(agents=[a, b], flow="a >> b")
        data = swarm.to_dict()
        assert data["mode"] == "workflow"
        assert data["flow"] == "a >> b"
        assert data["max_handoffs"] == 10
        assert len(data["agents"]) == 2

    def test_handoff_mode(self) -> None:
        """Handoff mode swarm serializes correctly."""
        a = Agent(name="a", handoffs=[Agent(name="b")])
        b = Agent(name="b")
        swarm = Swarm(agents=[a, b], flow="a >> b", mode="handoff", max_handoffs=5)
        data = swarm.to_dict()
        assert data["mode"] == "handoff"
        assert data["max_handoffs"] == 5

    def test_no_flow(self) -> None:
        """Swarm without explicit flow serializes flow as None."""
        a = Agent(name="a")
        b = Agent(name="b")
        swarm = Swarm(agents=[a, b])
        data = swarm.to_dict()
        assert data["flow"] is None


class TestSwarmFromDict:
    def test_basic_reconstruction(self) -> None:
        """from_dict() reconstructs a basic swarm."""
        data = {
            "agents": [
                {"name": "a", "model": "openai:gpt-4o"},
                {"name": "b", "model": "openai:gpt-4o"},
            ],
            "flow": "a >> b",
            "mode": "workflow",
            "max_handoffs": 10,
        }
        swarm = Swarm.from_dict(data)
        assert swarm.mode == "workflow"
        assert swarm.flow == "a >> b"
        assert len(swarm.agents) == 2
        assert "a" in swarm.agents
        assert "b" in swarm.agents

    def test_no_flow_reconstruction(self) -> None:
        """from_dict() handles None flow (uses agent list order)."""
        data = {
            "agents": [
                {"name": "x"},
                {"name": "y"},
            ],
            "flow": None,
            "mode": "workflow",
        }
        swarm = Swarm.from_dict(data)
        assert swarm.flow_order == ["x", "y"]


class TestSwarmRoundTrip:
    def test_workflow_round_trip(self) -> None:
        """Workflow swarm round-trips correctly."""
        a = Agent(name="a", model="openai:gpt-4o")
        b = Agent(name="b", model="openai:gpt-4o-mini")
        original = Swarm(agents=[a, b], flow="a >> b")
        reconstructed = Swarm.from_dict(original.to_dict())
        assert reconstructed.mode == original.mode
        assert reconstructed.flow == original.flow
        assert reconstructed.flow_order == original.flow_order
        assert set(reconstructed.agents.keys()) == set(original.agents.keys())

    def test_json_round_trip(self) -> None:
        """Swarm survives JSON round-trip."""
        a = Agent(name="a", tools=[search])
        b = Agent(name="b")
        original = Swarm(agents=[a, b], flow="a >> b", mode="workflow")
        json_str = json.dumps(original.to_dict())
        data = json.loads(json_str)
        reconstructed = Swarm.from_dict(data)
        assert reconstructed.mode == "workflow"
        assert "search" in reconstructed.agents["a"].tools

    def test_team_mode_round_trip(self) -> None:
        """Team mode swarm round-trips correctly."""
        lead = Agent(name="lead")
        worker = Agent(name="worker")
        original = Swarm(agents=[lead, worker], mode="team")
        reconstructed = Swarm.from_dict(original.to_dict())
        assert reconstructed.mode == "team"
        assert "lead" in reconstructed.agents
        assert "worker" in reconstructed.agents


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------


class TestSerializationHelpers:
    def test_serialize_function_tool(self) -> None:
        """FunctionTool serializes to its wrapped function's dotted path."""
        path = _serialize_tool(search)
        assert "search" in path
        assert "." in path

    def test_serialize_custom_tool(self) -> None:
        """Custom Tool subclass serializes to its class dotted path."""
        custom = CustomSearchTool()
        path = _serialize_tool(custom)
        assert "CustomSearchTool" in path

    def test_deserialize_function_tool(self) -> None:
        """Deserializing a FunctionTool path returns a Tool."""
        path = _serialize_tool(search)
        result = _deserialize_tool(path)
        assert isinstance(result, Tool)

    def test_deserialize_custom_tool_class(self) -> None:
        """Deserializing a custom Tool subclass path returns an instance."""
        custom = CustomSearchTool()
        path = _serialize_tool(custom)
        result = _deserialize_tool(path)
        assert isinstance(result, Tool)
        assert result.name == "custom_search"

    def test_import_object_valid(self) -> None:
        """_import_object resolves a valid dotted path."""
        obj = _import_object("exo.agent.Agent")
        assert obj is Agent

    def test_import_object_invalid(self) -> None:
        """_import_object raises on invalid path."""
        with pytest.raises(ValueError, match="Cannot import"):
            _import_object("nonexistent.module.Thing")

    def test_lambda_tool_raises(self) -> None:
        """Lambda tools cannot be serialized."""
        lam = FunctionTool(lambda x: x, name="lam")
        with pytest.raises(ValueError, match="closure or lambda"):
            _serialize_tool(lam)

    def test_deserialize_unknown_dict_raises(self) -> None:
        """Deserializing an unknown dict format raises ValueError."""
        with pytest.raises(ValueError, match="Unknown tool dict format"):
            _deserialize_tool({"unknown": True})


# ---------------------------------------------------------------------------
# MCPToolWrapper serialization tests
# ---------------------------------------------------------------------------


class TestMCPToolWrapperSerialization:
    def test_serialize_mcp_tool_returns_dict(self) -> None:
        """MCPToolWrapper serializes to a dict with __mcp_tool__ marker."""
        from unittest.mock import AsyncMock

        from mcp.types import Tool as MCPTool

        from exo.mcp.tools import MCPToolWrapper

        mcp_tool = MCPTool(
            name="search",
            description="Search the web",
            inputSchema={"type": "object", "properties": {"q": {"type": "string"}}},
        )
        wrapper = MCPToolWrapper(mcp_tool, "my_server", AsyncMock())
        result = _serialize_tool(wrapper)
        assert isinstance(result, dict)
        assert result["__mcp_tool__"] is True
        assert result["original_name"] == "search"
        assert result["server_name"] == "my_server"

    def test_mcp_tool_round_trip(self) -> None:
        """MCPToolWrapper survives _serialize_tool / _deserialize_tool round-trip."""
        from unittest.mock import AsyncMock

        from mcp.types import Tool as MCPTool

        from exo.mcp.tools import MCPToolWrapper

        mcp_tool = MCPTool(
            name="read_file",
            description="Read a file",
            inputSchema={
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        )
        original = MCPToolWrapper(mcp_tool, "fs_server", AsyncMock())
        serialized = _serialize_tool(original)
        restored = _deserialize_tool(serialized)

        assert isinstance(restored, MCPToolWrapper)
        assert restored.name == original.name
        assert restored.description == original.description
        assert restored.parameters == original.parameters
        assert restored.original_name == original.original_name
        assert restored.server_name == original.server_name

    def test_mcp_tool_json_round_trip(self) -> None:
        """MCPToolWrapper survives JSON serialization round-trip."""
        from unittest.mock import AsyncMock

        from mcp.types import Tool as MCPTool

        from exo.mcp.client import MCPServerConfig
        from exo.mcp.tools import MCPToolWrapper

        config = MCPServerConfig(name="srv", transport="sse", url="http://localhost:8080")
        mcp_tool = MCPTool(
            name="query",
            description="Run a query",
            inputSchema={"type": "object", "properties": {}},
        )
        original = MCPToolWrapper(mcp_tool, "srv", AsyncMock(), server_config=config)
        serialized = _serialize_tool(original)
        json_str = json.dumps(serialized)
        data = json.loads(json_str)
        restored = _deserialize_tool(data)

        assert isinstance(restored, MCPToolWrapper)
        assert restored.name == original.name
        assert restored._server_config is not None
        assert restored._server_config.name == "srv"
        assert restored._server_config.url == "http://localhost:8080"

    def test_agent_with_mcp_tools_round_trip(self) -> None:
        """Agent with MCPToolWrapper tools survives to_dict/from_dict."""
        from unittest.mock import AsyncMock

        from mcp.types import Tool as MCPTool

        from exo.mcp.tools import MCPToolWrapper

        mcp_tool = MCPTool(
            name="search",
            description="Search",
            inputSchema={"type": "object", "properties": {}},
        )
        wrapper = MCPToolWrapper(mcp_tool, "srv", AsyncMock())
        agent = Agent(name="mcp-agent", tools=[wrapper])
        data = agent.to_dict()

        # Verify the tool is serialized as a dict
        assert isinstance(data["tools"][0], dict)
        assert data["tools"][0]["__mcp_tool__"] is True

        # Reconstruct
        restored = Agent.from_dict(data)
        # mcp search tool + retrieve_artifact + spawn_self + 3 background + 3 planning tools
        assert len(restored.tools) == 9
        assert any("search" in name for name in restored.tools)

    def test_agent_with_mixed_tools_round_trip(self) -> None:
        """Agent with both regular and MCP tools round-trips correctly."""
        from unittest.mock import AsyncMock

        from mcp.types import Tool as MCPTool

        from exo.mcp.tools import MCPToolWrapper

        mcp_tool = MCPTool(
            name="mcp_search",
            description="MCP Search",
            inputSchema={"type": "object", "properties": {}},
        )
        mcp_wrapper = MCPToolWrapper(mcp_tool, "srv", AsyncMock())
        agent = Agent(name="mixed", tools=[search, mcp_wrapper])
        data = agent.to_dict()

        # One should be a string (FunctionTool), one a dict (MCPToolWrapper)
        types = {type(t) for t in data["tools"]}
        assert str in types
        assert dict in types

        restored = Agent.from_dict(data)
        # search + mcp_search + retrieve_artifact + spawn_self + 3 background + 3 planning tools
        assert len(restored.tools) == 10


# ---------------------------------------------------------------------------
# Swarm + MCPToolWrapper serialization tests (SSE on distributed workers)
# ---------------------------------------------------------------------------


class TestSwarmMCPToolWrapperSerialization:
    """Test that Swarms with SSE MCP tools survive the distributed serialization path.

    When a Swarm is submitted to a distributed worker via ``distributed()``, the
    full path is: Swarm.to_dict() → JSON → Swarm.from_dict() on the worker.
    Each agent's MCPToolWrapper must preserve its server_config so the worker
    can lazily reconnect to SSE MCP servers.
    """

    def _make_sse_config(self, name: str, port: int = 3001) -> Any:
        from exo.mcp.client import MCPServerConfig

        return MCPServerConfig(
            name=name,
            transport="sse",
            url=f"http://localhost:{port}/sse",
            headers={"Authorization": "Bearer test-token"},
            timeout=30.0,
            sse_read_timeout=300.0,
        )

    def _make_mcp_wrapper(self, tool_name: str, server_name: str, server_config: Any = None) -> Any:
        from unittest.mock import AsyncMock

        from mcp.types import Tool as MCPTool

        from exo.mcp.tools import MCPToolWrapper

        mcp_tool = MCPTool(
            name=tool_name,
            description=f"MCP tool: {tool_name}",
            inputSchema={
                "type": "object",
                "properties": {"input": {"type": "string"}},
                "required": ["input"],
            },
        )
        return MCPToolWrapper(mcp_tool, server_name, AsyncMock(), server_config=server_config)

    def test_workflow_swarm_with_sse_mcp_tools(self) -> None:
        """Workflow swarm with SSE MCP tools round-trips through to_dict/from_dict."""
        from exo.mcp.tools import MCPToolWrapper

        cfg = self._make_sse_config("tools-server")
        researcher = Agent(
            name="researcher",
            model="openai:gpt-4o-mini",
            tools=[self._make_mcp_wrapper("web_search", "tools-server", cfg)],
        )
        writer = Agent(
            name="writer",
            model="openai:gpt-4o-mini",
            instructions="Write a summary from the research.",
        )
        swarm = Swarm(agents=[researcher, writer], flow="researcher >> writer")

        data = swarm.to_dict()
        restored = Swarm.from_dict(data)

        assert restored.mode == "workflow"
        assert restored.flow == "researcher >> writer"
        assert len(restored.agents) == 2

        # Researcher should have the MCP tool with SSE config preserved
        # (plus retrieve_artifact always auto-registered)
        r_tools = [
            t for t in restored.agents["researcher"].tools.values() if isinstance(t, MCPToolWrapper)
        ]
        assert len(r_tools) == 1
        assert r_tools[0]._server_config is not None
        assert r_tools[0]._server_config.transport.value == "sse"
        assert r_tools[0]._server_config.url == "http://localhost:3001/sse"
        assert r_tools[0]._server_config.headers == {"Authorization": "Bearer test-token"}
        assert r_tools[0]._call_fn is None  # lazy reconnection

    def test_swarm_multiple_agents_with_different_sse_servers(self) -> None:
        """Each agent's MCP tools preserve their own SSE server config."""
        from exo.mcp.tools import MCPToolWrapper

        cfg_a = self._make_sse_config("server-a", port=3001)
        cfg_b = self._make_sse_config("server-b", port=3002)

        agent_a = Agent(
            name="fetcher",
            tools=[self._make_mcp_wrapper("fetch", "server-a", cfg_a)],
        )
        agent_b = Agent(
            name="analyzer",
            tools=[self._make_mcp_wrapper("analyze", "server-b", cfg_b)],
        )
        swarm = Swarm(agents=[agent_a, agent_b], flow="fetcher >> analyzer")

        data = swarm.to_dict()
        restored = Swarm.from_dict(data)

        tool_a = next(iter(restored.agents["fetcher"].tools.values()))
        tool_b = next(iter(restored.agents["analyzer"].tools.values()))

        assert isinstance(tool_a, MCPToolWrapper)
        assert isinstance(tool_b, MCPToolWrapper)
        assert tool_a._server_config.url == "http://localhost:3001/sse"
        assert tool_b._server_config.url == "http://localhost:3002/sse"
        assert tool_a._server_config.name == "server-a"
        assert tool_b._server_config.name == "server-b"

    def test_swarm_mixed_local_and_sse_mcp_tools(self) -> None:
        """Agents with both local @tool functions and SSE MCP tools round-trip."""
        from exo.mcp.tools import MCPToolWrapper

        cfg = self._make_sse_config("sse-srv")
        agent_with_both = Agent(
            name="hybrid",
            tools=[search, self._make_mcp_wrapper("remote_search", "sse-srv", cfg)],
        )
        agent_plain = Agent(name="summarizer", tools=[calculate])
        swarm = Swarm(agents=[agent_with_both, agent_plain], flow="hybrid >> summarizer")

        data = swarm.to_dict()

        # Verify serialized structure
        hybrid_tools = data["agents"][0]["tools"]
        assert len(hybrid_tools) == 2
        types = {type(t) for t in hybrid_tools}
        assert str in types  # local tool
        assert dict in types  # MCP tool

        restored = Swarm.from_dict(data)
        hybrid = restored.agents["hybrid"]
        # search + MCPToolWrapper + retrieve_artifact + spawn_self + 3 background + 3 planning tools
        assert len(hybrid.tools) == 10

        mcp_tools = [t for t in hybrid.tools.values() if isinstance(t, MCPToolWrapper)]
        assert len(mcp_tools) == 1
        assert mcp_tools[0]._server_config.transport.value == "sse"

        # Plain agent's local tool should also survive
        assert "calculate" in restored.agents["summarizer"].tools

    def test_swarm_json_wire_round_trip(self) -> None:
        """Swarm with SSE MCP tools survives full JSON wire round-trip.

        This simulates what actually happens in distributed execution:
        Swarm.to_dict() → json.dumps() → wire → json.loads() → Swarm.from_dict()
        """
        from exo.mcp.tools import MCPToolWrapper

        cfg = self._make_sse_config("remote-tools", port=8080)
        agent = Agent(
            name="worker-agent",
            model="openai:gpt-4o-mini",
            tools=[
                search,
                self._make_mcp_wrapper("query_db", "remote-tools", cfg),
                self._make_mcp_wrapper("read_file", "remote-tools", cfg),
            ],
        )
        swarm = Swarm(agents=[agent, Agent(name="reviewer")], flow="worker-agent >> reviewer")

        # Full JSON wire simulation
        wire = json.dumps(swarm.to_dict())
        restored = Swarm.from_dict(json.loads(wire))

        worker = restored.agents["worker-agent"]
        # search + 2 MCP tools + retrieve_artifact + spawn_self + 3 background + 3 planning tools
        assert len(worker.tools) == 11

        mcp_tools = [t for t in worker.tools.values() if isinstance(t, MCPToolWrapper)]
        assert len(mcp_tools) == 2
        for t in mcp_tools:
            assert t._server_config is not None
            assert t._server_config.url == "http://localhost:8080/sse"
            assert t._server_config.transport.value == "sse"
            assert t._call_fn is None  # ready for lazy reconnect on worker

    def test_handoff_swarm_with_sse_mcp_tools(self) -> None:
        """Handoff-mode swarm with SSE MCP tools round-trips correctly."""
        from exo.mcp.tools import MCPToolWrapper

        cfg = self._make_sse_config("sse-tools")
        router = Agent(
            name="router",
            model="openai:gpt-4o-mini",
            handoffs=[Agent(name="specialist")],
        )
        specialist = Agent(
            name="specialist",
            tools=[self._make_mcp_wrapper("deep_search", "sse-tools", cfg)],
        )
        swarm = Swarm(
            agents=[router, specialist],
            flow="router >> specialist",
            mode="handoff",
            max_handoffs=5,
        )

        data = swarm.to_dict()
        restored = Swarm.from_dict(data)

        assert restored.mode == "handoff"
        assert restored.max_handoffs == 5

        spec_tools = [
            t for t in restored.agents["specialist"].tools.values() if isinstance(t, MCPToolWrapper)
        ]
        assert len(spec_tools) == 1
        assert isinstance(spec_tools[0], MCPToolWrapper)
        assert spec_tools[0]._server_config.transport.value == "sse"

    def test_team_swarm_with_sse_mcp_tools(self) -> None:
        """Team-mode swarm with SSE MCP tools on workers round-trips correctly."""
        from exo.mcp.tools import MCPToolWrapper

        cfg = self._make_sse_config("team-tools")
        lead = Agent(name="lead", model="openai:gpt-4o-mini")
        worker_a = Agent(
            name="data-worker",
            tools=[self._make_mcp_wrapper("query", "team-tools", cfg)],
        )
        worker_b = Agent(
            name="file-worker",
            tools=[self._make_mcp_wrapper("read_file", "team-tools", cfg)],
        )
        swarm = Swarm(
            agents=[lead, worker_a, worker_b],
            flow="lead >> data-worker >> file-worker",
            mode="team",
        )

        wire = json.dumps(swarm.to_dict())
        restored = Swarm.from_dict(json.loads(wire))

        assert restored.mode == "team"
        assert len(restored.agents) == 3

        for name in ("data-worker", "file-worker"):
            mcp_tools_only = [
                t for t in restored.agents[name].tools.values() if isinstance(t, MCPToolWrapper)
            ]
            assert len(mcp_tools_only) == 1
            assert isinstance(mcp_tools_only[0], MCPToolWrapper)
            assert mcp_tools_only[0]._server_config.transport.value == "sse"
            assert mcp_tools_only[0]._call_fn is None

    def test_sse_config_fields_preserved(self) -> None:
        """All SSE-specific config fields survive the swarm round-trip."""
        from exo.mcp.client import MCPServerConfig

        cfg = MCPServerConfig(
            name="full-config",
            transport="sse",
            url="http://mcp.internal:9090/sse",
            headers={"X-API-Key": "secret", "X-Org": "test"},
            timeout=60.0,
            sse_read_timeout=600.0,
            cache_tools=True,
            session_timeout=240.0,
        )
        agent = Agent(
            name="agent",
            tools=[self._make_mcp_wrapper("tool", "full-config", cfg)],
        )
        swarm = Swarm(agents=[agent], flow=None)

        wire = json.dumps(swarm.to_dict())
        restored = Swarm.from_dict(json.loads(wire))

        rc = next(iter(restored.agents["agent"].tools.values()))._server_config
        assert rc.url == "http://mcp.internal:9090/sse"
        assert rc.headers == {"X-API-Key": "secret", "X-Org": "test"}
        assert rc.timeout == 60.0
        assert rc.sse_read_timeout == 600.0
        assert rc.cache_tools is True
        assert rc.session_timeout == 240.0
