"""Tests for exo_server.agents — agent listing and workspace routes."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from httpx import ASGITransport, AsyncClient

from exo_server.agents import (
    AgentInfo,
    WorkspaceFile,
    WorkspaceFileContent,
    _agent_info,
    _get_workspace,
    agent_router,
)
from exo_server.app import create_app, register_agent

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_agent(
    name: str = "test-agent",
    model: str = "openai:gpt-4",
    *,
    tools: dict[str, Any] | None = None,
    handoffs: dict[str, Any] | None = None,
    max_steps: int = 10,
    temperature: float = 0.7,
    max_tokens: int | None = None,
    context: Any = None,
) -> Any:
    """Create a mock agent with configurable attributes."""
    agent = MagicMock()
    agent.name = name
    agent.model = model
    agent.tools = tools if tools is not None else {}
    agent.handoffs = handoffs if handoffs is not None else {}
    agent.max_steps = max_steps
    agent.temperature = temperature
    agent.max_tokens = max_tokens
    agent.context = context
    return agent


def _mock_workspace(artifacts: list[Any] | None = None) -> Any:
    """Create a mock workspace with optional artifacts."""
    ws = MagicMock()
    ws.list.return_value = artifacts if artifacts is not None else []
    return ws


def _mock_artifact(
    name: str = "readme.md",
    content: str = "# Hello",
    artifact_type: str = "text",
    versions: int = 1,
) -> Any:
    """Create a mock workspace artifact."""
    a = MagicMock()
    a.name = name
    a.content = content
    a.artifact_type = artifact_type
    a.versions = [MagicMock()] * versions
    return a


def _build_client(app: Any) -> AsyncClient:
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


# ---------------------------------------------------------------------------
# Response model tests
# ---------------------------------------------------------------------------


class TestAgentInfo:
    def test_defaults(self) -> None:
        info = AgentInfo(name="a")
        assert info.name == "a"
        assert info.model == ""
        assert info.is_default is False
        assert info.tools == []
        assert info.handoffs == []

    def test_with_data(self) -> None:
        info = AgentInfo(
            name="bot",
            model="openai:gpt-4",
            is_default=True,
            tools=["search", "read"],
            handoffs=["reviewer"],
            max_steps=5,
            temperature=0.5,
            max_tokens=1024,
        )
        assert info.is_default is True
        assert info.tools == ["search", "read"]
        assert info.max_tokens == 1024


class TestWorkspaceFile:
    def test_defaults(self) -> None:
        f = WorkspaceFile(name="test.txt")
        assert f.artifact_type == "text"
        assert f.version_count == 1

    def test_with_data(self) -> None:
        f = WorkspaceFile(name="code.py", artifact_type="code", version_count=3)
        assert f.artifact_type == "code"


class TestWorkspaceFileContent:
    def test_full(self) -> None:
        f = WorkspaceFileContent(
            name="x.md", content="# X", artifact_type="markdown", version_count=2
        )
        assert f.content == "# X"


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_agent_info_builder(self) -> None:
        agent = _mock_agent("bot", "openai:gpt-4", tools={"search": MagicMock()})
        info = _agent_info(agent, is_default=True)
        assert info.name == "bot"
        assert info.model == "openai:gpt-4"
        assert info.is_default is True
        assert info.tools == ["search"]

    def test_agent_info_no_tools_dict(self) -> None:
        agent = _mock_agent("bot")
        agent.tools = "not-a-dict"
        info = _agent_info(agent, is_default=False)
        assert info.tools == []

    def test_get_workspace_none_context(self) -> None:
        agent = _mock_agent()
        agent.context = None
        assert _get_workspace(agent) is None

    def test_get_workspace_no_workspace_attr(self) -> None:
        agent = _mock_agent()
        ctx = MagicMock(spec=[])  # no workspace attr
        agent.context = ctx
        assert _get_workspace(agent) is None

    def test_get_workspace_present(self) -> None:
        ws = _mock_workspace()
        ctx = MagicMock()
        ctx.workspace = ws
        agent = _mock_agent(context=ctx)
        assert _get_workspace(agent) is ws


# ---------------------------------------------------------------------------
# GET /agents — list agents
# ---------------------------------------------------------------------------


class TestListAgents:
    async def test_empty(self) -> None:
        app = create_app()
        async with _build_client(app) as client:
            resp = await client.get("/agents")
        assert resp.status_code == 200
        assert resp.json() == []

    async def test_single_agent(self) -> None:
        app = create_app()
        register_agent(app, _mock_agent("alpha"))
        async with _build_client(app) as client:
            resp = await client.get("/agents")
        data = resp.json()
        assert len(data) == 1
        assert data[0]["name"] == "alpha"
        assert data[0]["is_default"] is True

    async def test_multiple_agents(self) -> None:
        app = create_app()
        register_agent(app, _mock_agent("a"))
        register_agent(app, _mock_agent("b"))
        async with _build_client(app) as client:
            resp = await client.get("/agents")
        data = resp.json()
        assert len(data) == 2
        names = {d["name"] for d in data}
        assert names == {"a", "b"}
        # First registered is default
        defaults = [d for d in data if d["is_default"]]
        assert len(defaults) == 1
        assert defaults[0]["name"] == "a"


# ---------------------------------------------------------------------------
# GET /agents/{name} — agent detail
# ---------------------------------------------------------------------------


class TestGetAgent:
    async def test_found(self) -> None:
        app = create_app()
        register_agent(app, _mock_agent("bot", tools={"search": MagicMock()}))
        async with _build_client(app) as client:
            resp = await client.get("/agents/bot")
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "bot"
        assert "search" in data["tools"]

    async def test_not_found(self) -> None:
        app = create_app()
        async with _build_client(app) as client:
            resp = await client.get("/agents/nope")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# GET /agents/{name}/workspace — list workspace files
# ---------------------------------------------------------------------------


class TestListWorkspace:
    async def test_no_workspace(self) -> None:
        app = create_app()
        register_agent(app, _mock_agent("bot"))
        async with _build_client(app) as client:
            resp = await client.get("/agents/bot/workspace")
        assert resp.status_code == 200
        assert resp.json() == []

    async def test_with_artifacts(self) -> None:
        ws = _mock_workspace(
            [_mock_artifact("a.txt"), _mock_artifact("b.py", artifact_type="code")]
        )
        ctx = MagicMock()
        ctx.workspace = ws
        app = create_app()
        register_agent(app, _mock_agent("bot", context=ctx))
        async with _build_client(app) as client:
            resp = await client.get("/agents/bot/workspace")
        data = resp.json()
        assert len(data) == 2
        names = {d["name"] for d in data}
        assert names == {"a.txt", "b.py"}

    async def test_agent_not_found(self) -> None:
        app = create_app()
        async with _build_client(app) as client:
            resp = await client.get("/agents/nope/workspace")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# GET /agents/{name}/workspace/{file} — read workspace file
# ---------------------------------------------------------------------------


class TestReadWorkspaceFile:
    async def test_read_file(self) -> None:
        artifact = _mock_artifact("notes.md", "# Notes", "markdown", versions=2)
        ws = _mock_workspace()
        ws.get.return_value = artifact
        ctx = MagicMock()
        ctx.workspace = ws
        app = create_app()
        register_agent(app, _mock_agent("bot", context=ctx))
        async with _build_client(app) as client:
            resp = await client.get("/agents/bot/workspace/notes.md")
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "notes.md"
        assert data["content"] == "# Notes"
        assert data["version_count"] == 2

    async def test_agent_not_found(self) -> None:
        app = create_app()
        async with _build_client(app) as client:
            resp = await client.get("/agents/nope/workspace/x.txt")
        assert resp.status_code == 404

    async def test_no_workspace(self) -> None:
        app = create_app()
        register_agent(app, _mock_agent("bot"))
        async with _build_client(app) as client:
            resp = await client.get("/agents/bot/workspace/x.txt")
        assert resp.status_code == 404
        assert "no workspace" in resp.json()["detail"].lower()

    async def test_file_not_found(self) -> None:
        ws = _mock_workspace()
        ws.get.return_value = None
        ctx = MagicMock()
        ctx.workspace = ws
        app = create_app()
        register_agent(app, _mock_agent("bot", context=ctx))
        async with _build_client(app) as client:
            resp = await client.get("/agents/bot/workspace/missing.txt")
        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"].lower()


# ---------------------------------------------------------------------------
# Router registration
# ---------------------------------------------------------------------------


def _all_paths(app) -> list[str]:
    """Collect route paths, recursing into included routers.

    FastAPI >=0.130 / Starlette >=1.0 keep ``include_router`` routes nested
    under an ``_IncludedRouter`` mount instead of flattening them into
    ``app.routes``, so a flat walk no longer sees them.
    """
    paths: list[str] = []

    def walk(routes) -> None:
        for r in routes:
            if getattr(r, "path", None) is not None:
                paths.append(r.path)
            # _IncludedRouter mounts expose nested routes via original_router;
            # plain Mounts via .routes.
            nested = getattr(r, "original_router", None)
            if nested is not None:
                walk(nested.routes)
            elif hasattr(r, "routes"):
                walk(r.routes)

    walk(app.routes)
    return paths


class TestRouterRegistration:
    def test_agent_router_prefix(self) -> None:
        assert agent_router.prefix == "/agents"

    def test_app_has_agent_routes(self) -> None:
        app = create_app()
        paths = _all_paths(app)
        assert "/agents" in paths
        assert "/agents/{agent_name}" in paths
        assert "/agents/{agent_name}/workspace" in paths
