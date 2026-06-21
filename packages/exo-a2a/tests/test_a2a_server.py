"""Tests for exo.a2a.server — A2A server with agent card discovery."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from httpx import ASGITransport, AsyncClient

from exo.a2a.server import (  # pyright: ignore[reportMissingImports]
    A2AServer,
    A2ATaskStore,
    AgentExecutor,
    InMemoryTaskStore,
)
from exo.a2a.types import (  # pyright: ignore[reportMissingImports]
    AgentSkill,
    ServingConfig,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_agent(name: str = "test-agent", run_result: str = "hello") -> Any:
    """Create a mock agent with async run()."""
    agent = MagicMock()
    agent.name = name
    result = MagicMock()
    result.text = run_result
    agent.run = AsyncMock(return_value=result)
    return agent


def _build_client(server: A2AServer) -> AsyncClient:
    """Build an httpx async test client for the server's FastAPI app."""
    app = server.build_app()
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


# ---------------------------------------------------------------------------
# InMemoryTaskStore
# ---------------------------------------------------------------------------


class TestInMemoryTaskStore:
    async def test_save_and_get(self) -> None:
        store = InMemoryTaskStore()
        await store.save("t1", {"status": "working"})
        assert await store.get("t1") == {"status": "working"}

    async def test_get_missing(self) -> None:
        store = InMemoryTaskStore()
        assert await store.get("nope") is None

    async def test_delete(self) -> None:
        store = InMemoryTaskStore()
        await store.save("t1", {"x": 1})
        await store.delete("t1")
        assert await store.get("t1") is None

    async def test_delete_missing(self) -> None:
        store = InMemoryTaskStore()
        await store.delete("nope")  # no error

    async def test_overwrite(self) -> None:
        store = InMemoryTaskStore()
        await store.save("t1", {"v": 1})
        await store.save("t1", {"v": 2})
        assert await store.get("t1") == {"v": 2}

    def test_protocol_conformance(self) -> None:
        assert isinstance(InMemoryTaskStore(), A2ATaskStore)

    def test_repr(self) -> None:
        store = InMemoryTaskStore()
        assert "InMemoryTaskStore" in repr(store)
        assert "tasks=0" in repr(store)


# ---------------------------------------------------------------------------
# AgentExecutor
# ---------------------------------------------------------------------------


class TestAgentExecutorInit:
    def test_default(self) -> None:
        agent = _mock_agent()
        executor = AgentExecutor(agent)
        assert executor.agent_name == "test-agent"

    def test_streaming(self) -> None:
        executor = AgentExecutor(_mock_agent(), streaming=True)
        assert "streaming=True" in repr(executor)

    def test_agent_name_fallback(self) -> None:
        agent = MagicMock(spec=[])  # no 'name' attr
        executor = AgentExecutor(agent)
        assert executor.agent_name == "unknown"


class TestAgentExecutorExecute:
    async def test_basic(self) -> None:
        agent = _mock_agent(run_result="world")
        executor = AgentExecutor(agent)
        result = await executor.execute("hello")
        assert result == "world"
        agent.run.assert_called_once_with("hello")

    async def test_with_provider(self) -> None:
        agent = _mock_agent()
        executor = AgentExecutor(agent)
        provider = MagicMock()
        await executor.execute("hi", provider=provider)
        agent.run.assert_called_once_with("hi", provider=provider)

    async def test_empty_text(self) -> None:
        agent = _mock_agent(run_result="")
        executor = AgentExecutor(agent)
        result = await executor.execute("test")
        assert result == ""

    async def test_none_text_returns_empty(self) -> None:
        agent = _mock_agent()
        result_mock = MagicMock()
        result_mock.text = None
        agent.run = AsyncMock(return_value=result_mock)
        executor = AgentExecutor(agent)
        result = await executor.execute("test")
        assert result == ""


class TestAgentExecutorRepr:
    def test_repr(self) -> None:
        executor = AgentExecutor(_mock_agent("my-agent"))
        r = repr(executor)
        assert "my-agent" in r
        assert "AgentExecutor" in r


# ---------------------------------------------------------------------------
# A2AServer init & agent card
# ---------------------------------------------------------------------------


class TestA2AServerInit:
    def test_defaults(self) -> None:
        server = A2AServer(AgentExecutor(_mock_agent()))
        assert server.agent_card.name == "test-agent"
        assert "A2AServer" in repr(server)

    def test_custom_config(self) -> None:
        cfg = ServingConfig(host="0.0.0.0", port=8080, version="1.2.3")
        server = A2AServer(AgentExecutor(_mock_agent()), cfg)
        assert server.agent_card.version == "1.2.3"
        assert "8080" in server.agent_card.url

    def test_with_skills(self) -> None:
        skill = AgentSkill(id="s1", name="Search")
        cfg = ServingConfig(skills=(skill,))
        server = A2AServer(AgentExecutor(_mock_agent()), cfg)
        assert len(server.agent_card.skills) == 1
        assert server.agent_card.skills[0].name == "Search"

    def test_custom_task_store(self) -> None:
        store = InMemoryTaskStore()
        server = A2AServer(AgentExecutor(_mock_agent()), task_store=store)
        assert server.task_store is store


# ---------------------------------------------------------------------------
# Agent card endpoint
# ---------------------------------------------------------------------------


class TestAgentCardEndpoint:
    async def test_get_agent_card(self) -> None:
        server = A2AServer(AgentExecutor(_mock_agent("card-agent")))
        async with _build_client(server) as client:
            resp = await client.get("/.well-known/agent-card")
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "card-agent"
        assert "version" in data
        assert "capabilities" in data

    async def test_agent_card_has_skills(self) -> None:
        skill = AgentSkill(id="s1", name="Code")
        cfg = ServingConfig(skills=(skill,))
        server = A2AServer(AgentExecutor(_mock_agent()), cfg)
        async with _build_client(server) as client:
            resp = await client.get("/.well-known/agent-card")
        data = resp.json()
        assert len(data["skills"]) == 1
        assert data["skills"][0]["name"] == "Code"


# ---------------------------------------------------------------------------
# Task execution endpoint
# ---------------------------------------------------------------------------


class TestTaskExecution:
    async def test_execute_success(self) -> None:
        server = A2AServer(AgentExecutor(_mock_agent(run_result="done")))
        async with _build_client(server) as client:
            resp = await client.post("/", json={"text": "hello"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"]["state"] == "completed"
        assert data["artifact"]["text"] == "done"

    async def test_execute_with_task_id(self) -> None:
        server = A2AServer(AgentExecutor(_mock_agent()))
        async with _build_client(server) as client:
            resp = await client.post("/", json={"text": "hi", "task_id": "custom-123"})
        data = resp.json()
        assert data["task_id"] == "custom-123"

    async def test_execute_auto_task_id(self) -> None:
        server = A2AServer(AgentExecutor(_mock_agent()))
        async with _build_client(server) as client:
            resp = await client.post("/", json={"text": "hi"})
        data = resp.json()
        assert "task_id" in data
        assert len(data["task_id"]) > 0

    async def test_execute_failure(self) -> None:
        agent = _mock_agent()
        agent.run = AsyncMock(side_effect=RuntimeError("boom"))
        server = A2AServer(AgentExecutor(agent))
        async with _build_client(server) as client:
            resp = await client.post("/", json={"text": "fail"})
        assert resp.status_code == 500
        data = resp.json()
        assert data["status"]["state"] == "failed"
        assert "boom" in data["status"]["reason"]

    async def test_execute_stores_task(self) -> None:
        store = InMemoryTaskStore()
        server = A2AServer(
            AgentExecutor(_mock_agent(run_result="result")),
            task_store=store,
        )
        async with _build_client(server) as client:
            await client.post("/", json={"text": "x", "task_id": "t1"})
        stored = await store.get("t1")
        assert stored is not None
        assert stored["result"] == "result"


# ---------------------------------------------------------------------------
# Task lookup endpoint
# ---------------------------------------------------------------------------


class TestTaskLookup:
    async def test_get_existing_task(self) -> None:
        store = InMemoryTaskStore()
        server = A2AServer(AgentExecutor(_mock_agent()), task_store=store)
        async with _build_client(server) as client:
            await client.post("/", json={"text": "x", "task_id": "t1"})
            resp = await client.get("/tasks/t1")
        assert resp.status_code == 200
        assert resp.json()["task_id"] == "t1"

    async def test_get_missing_task(self) -> None:
        server = A2AServer(AgentExecutor(_mock_agent()))
        async with _build_client(server) as client:
            resp = await client.get("/tasks/nope")
        assert resp.status_code == 404
        assert "not found" in resp.json()["error"]


# ---------------------------------------------------------------------------
# Streaming endpoint
# ---------------------------------------------------------------------------


class TestStreamingEndpoint:
    async def test_streaming_disabled(self) -> None:
        cfg = ServingConfig(streaming=False)
        server = A2AServer(AgentExecutor(_mock_agent()), cfg)
        async with _build_client(server) as client:
            resp = await client.post("/stream", json={"text": "hi"})
        # /stream should not exist when streaming is disabled
        assert resp.status_code in (404, 405)

    async def test_streaming_success(self) -> None:
        cfg = ServingConfig(streaming=True)
        server = A2AServer(
            AgentExecutor(_mock_agent(run_result="streamed"), streaming=True),
            cfg,
        )
        async with _build_client(server) as client:
            resp = await client.post("/stream", json={"text": "hi"})
        assert resp.status_code == 200
        lines = [json.loads(line) for line in resp.text.strip().split("\n") if line.strip()]
        # Should have: working status, artifact, completed status
        assert len(lines) == 3
        assert lines[0]["status"]["state"] == "working"
        assert lines[1]["text"] == "streamed"
        assert lines[1]["last_chunk"] is True
        assert lines[2]["status"]["state"] == "completed"

    async def test_streaming_failure(self) -> None:
        agent = _mock_agent()
        agent.run = AsyncMock(side_effect=RuntimeError("stream-err"))
        cfg = ServingConfig(streaming=True)
        server = A2AServer(AgentExecutor(agent), cfg)
        async with _build_client(server) as client:
            resp = await client.post("/stream", json={"text": "x"})
        lines = [json.loads(line) for line in resp.text.strip().split("\n") if line.strip()]
        # working + failed
        assert len(lines) == 2
        assert lines[1]["status"]["state"] == "failed"
        assert "stream-err" in lines[1]["status"]["reason"]


# ---------------------------------------------------------------------------
# build_app
# ---------------------------------------------------------------------------


class TestBuildApp:
    def test_returns_fastapi_app(self) -> None:
        server = A2AServer(AgentExecutor(_mock_agent()))
        app = server.build_app()
        assert app is not None
        # FastAPI subclasses Starlette, has .routes
        assert hasattr(app, "routes")

    def test_idempotent(self) -> None:
        server = A2AServer(AgentExecutor(_mock_agent()))
        app1 = server.build_app()
        app2 = server.build_app()
        # Each call creates a new app (not cached)
        assert app1 is not app2


# ---------------------------------------------------------------------------
# Repr
# ---------------------------------------------------------------------------


class TestRepr:
    def test_server_repr(self) -> None:
        server = A2AServer(AgentExecutor(_mock_agent("rp")))
        r = repr(server)
        assert "rp" in r
        assert "A2AServer" in r
