"""Tests for exo_server.streaming — WebSocket + SSE streaming."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

from httpx import ASGITransport, AsyncClient
from starlette.testclient import TestClient

from exo_server.app import create_app, register_agent
from exo_server.streaming import _iter_events, _resolve_agent

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_agent(name: str = "test-agent") -> Any:
    agent = MagicMock()
    agent.name = name
    return agent


def _build_client(app: Any) -> AsyncClient:
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


# ---------------------------------------------------------------------------
# _resolve_agent
# ---------------------------------------------------------------------------


class TestResolveAgent:
    def test_no_agents(self) -> None:
        state = MagicMock(spec=[])
        assert _resolve_agent(state, None) is None

    def test_by_name(self) -> None:
        agent = _mock_agent("alpha")
        state = MagicMock()
        state.exo_agents = {"alpha": agent}
        assert _resolve_agent(state, "alpha") is agent

    def test_by_name_not_found(self) -> None:
        state = MagicMock()
        state.exo_agents = {"alpha": _mock_agent("alpha")}
        assert _resolve_agent(state, "nope") is None

    def test_default(self) -> None:
        agent = _mock_agent("alpha")
        state = MagicMock()
        state.exo_agents = {"alpha": agent}
        state.exo_default_agent = "alpha"
        assert _resolve_agent(state, None) is agent

    def test_no_default(self) -> None:
        state = MagicMock()
        state.exo_agents = {"a": _mock_agent("a")}
        state.exo_default_agent = None
        assert _resolve_agent(state, None) is None


# ---------------------------------------------------------------------------
# _iter_events
# ---------------------------------------------------------------------------


class TestIterEvents:
    async def test_text_events(self) -> None:
        agent = _mock_agent()

        async def _fake_stream(a: Any, msg: str, **kw: Any) -> Any:
            yield MagicMock(type="text", text="hello")

        mock_run = MagicMock()
        mock_run.stream = _fake_stream

        with patch("exo_server.streaming._run_agent", mock_run):
            events = [e async for e in _iter_events(agent, "hi")]

        assert len(events) == 1
        assert events[0] == {"type": "text", "text": "hello"}

    async def test_tool_call_events(self) -> None:
        agent = _mock_agent()

        async def _fake_stream(a: Any, msg: str, **kw: Any) -> Any:
            yield MagicMock(type="tool_call", tool_name="search", tool_call_id="tc1")

        mock_run = MagicMock()
        mock_run.stream = _fake_stream

        with patch("exo_server.streaming._run_agent", mock_run):
            events = [e async for e in _iter_events(agent, "hi")]

        assert events[0]["type"] == "tool_call"
        assert events[0]["tool_name"] == "search"

    async def test_stream_error(self) -> None:
        agent = _mock_agent()

        async def _fake_stream(a: Any, msg: str, **kw: Any) -> Any:
            raise RuntimeError("boom")
            yield  # pragma: no cover

        mock_run = MagicMock()
        mock_run.stream = _fake_stream

        with patch("exo_server.streaming._run_agent", mock_run):
            events = [e async for e in _iter_events(agent, "hi")]

        assert events[0]["type"] == "error"
        assert "boom" in events[0]["error"]

    async def test_no_stream_fn(self) -> None:
        agent = _mock_agent()
        mock_run = MagicMock(spec=[])  # no .stream attribute

        with patch("exo_server.streaming._run_agent", mock_run):
            events = [e async for e in _iter_events(agent, "hi")]

        assert events[0]["type"] == "error"
        assert "not available" in events[0]["error"].lower()


# ---------------------------------------------------------------------------
# WebSocket /ws/chat
# ---------------------------------------------------------------------------


class TestWebSocket:
    async def test_stream_text(self) -> None:
        app = create_app()
        register_agent(app, _mock_agent("bot"))

        async def _fake_stream(a: Any, msg: str, **kw: Any) -> Any:
            yield MagicMock(type="text", text="Hello ")
            yield MagicMock(type="text", text="world!")

        mock_run = MagicMock()
        mock_run.stream = _fake_stream

        with patch("exo_server.streaming._run_agent", mock_run):
            tc = TestClient(app)
            with tc.websocket_connect("/ws/chat") as ws:
                ws.send_json({"message": "hi"})
                msg1 = ws.receive_json()
                assert msg1["type"] == "text"
                assert msg1["text"] == "Hello "
                msg2 = ws.receive_json()
                assert msg2["type"] == "text"
                assert msg2["text"] == "world!"
                done = ws.receive_json()
                assert done["type"] == "done"

    async def test_stream_tool_call(self) -> None:
        app = create_app()
        register_agent(app, _mock_agent("bot"))

        async def _fake_stream(a: Any, msg: str, **kw: Any) -> Any:
            yield MagicMock(type="tool_call", tool_name="search", tool_call_id="tc1")

        mock_run = MagicMock()
        mock_run.stream = _fake_stream

        with patch("exo_server.streaming._run_agent", mock_run):
            tc = TestClient(app)
            with tc.websocket_connect("/ws/chat") as ws:
                ws.send_json({"message": "find stuff"})
                msg = ws.receive_json()
                assert msg["type"] == "tool_call"
                assert msg["tool_name"] == "search"
                done = ws.receive_json()
                assert done["type"] == "done"

    async def test_empty_message(self) -> None:
        app = create_app()
        register_agent(app, _mock_agent("bot"))

        tc = TestClient(app)
        with tc.websocket_connect("/ws/chat") as ws:
            ws.send_json({"message": ""})
            msg = ws.receive_json()
            assert msg["type"] == "error"
            assert "empty" in msg["error"].lower()

    async def test_no_agents(self) -> None:
        app = create_app()

        tc = TestClient(app)
        with tc.websocket_connect("/ws/chat") as ws:
            ws.send_json({"message": "hi"})
            msg = ws.receive_json()
            assert msg["type"] == "error"
            assert "no agents" in msg["error"].lower()

    async def test_agent_not_found(self) -> None:
        app = create_app()
        register_agent(app, _mock_agent("bot"))

        tc = TestClient(app)
        with tc.websocket_connect("/ws/chat") as ws:
            ws.send_json({"message": "hi", "agent_name": "nope"})
            msg = ws.receive_json()
            assert msg["type"] == "error"
            assert "nope" in msg["error"]

    async def test_stream_error_during_events(self) -> None:
        app = create_app()
        register_agent(app, _mock_agent("bot"))

        async def _fake_stream(a: Any, msg: str, **kw: Any) -> Any:
            raise RuntimeError("stream broke")
            yield  # pragma: no cover

        mock_run = MagicMock()
        mock_run.stream = _fake_stream

        from starlette.testclient import TestClient

        with patch("exo_server.streaming._run_agent", mock_run):
            tc = TestClient(app)
            with tc.websocket_connect("/ws/chat") as ws:
                ws.send_json({"message": "hi"})
                msg = ws.receive_json()
                assert msg["type"] == "error"
                assert "stream broke" in msg["error"]
                done = ws.receive_json()
                assert done["type"] == "done"


# ---------------------------------------------------------------------------
# SSE GET /stream
# ---------------------------------------------------------------------------


class TestSSEStream:
    async def test_stream_text_events(self) -> None:
        app = create_app()
        register_agent(app, _mock_agent("bot"))

        async def _fake_stream(a: Any, msg: str, **kw: Any) -> Any:
            yield MagicMock(type="text", text="Hello ")
            yield MagicMock(type="text", text="world!")

        mock_run = MagicMock()
        mock_run.stream = _fake_stream

        with patch("exo_server.streaming._run_agent", mock_run):
            async with _build_client(app) as client:
                resp = await client.get("/stream", params={"message": "hi"})

        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]
        lines = [ln for ln in resp.text.strip().split("\n") if ln.startswith("data:")]
        assert len(lines) >= 3  # 2 text events + [DONE]
        first = json.loads(lines[0].removeprefix("data: "))
        assert first["type"] == "text"
        assert first["text"] == "Hello "
        assert "data: [DONE]" in resp.text

    async def test_stream_no_agent(self) -> None:
        """GET /stream with no agents registered must return 404, not 200 with an error body."""
        app = create_app()

        async with _build_client(app) as client:
            resp = await client.get("/stream", params={"message": "hi"})

        assert resp.status_code == 404

    async def test_stream_with_agent_name(self) -> None:
        app = create_app()
        register_agent(app, _mock_agent("a"))
        register_agent(app, _mock_agent("b"))

        async def _fake_stream(a: Any, msg: str, **kw: Any) -> Any:
            yield MagicMock(type="text", text=f"from {a.name}")

        mock_run = MagicMock()
        mock_run.stream = _fake_stream

        with patch("exo_server.streaming._run_agent", mock_run):
            async with _build_client(app) as client:
                resp = await client.get("/stream", params={"message": "hi", "agent_name": "b"})

        lines = [ln for ln in resp.text.strip().split("\n") if ln.startswith("data:")]
        first = json.loads(lines[0].removeprefix("data: "))
        assert first["text"] == "from b"

    async def test_stream_done_marker(self) -> None:
        app = create_app()
        register_agent(app, _mock_agent("bot"))

        async def _fake_stream(a: Any, msg: str, **kw: Any) -> Any:
            yield MagicMock(type="text", text="hi")

        mock_run = MagicMock()
        mock_run.stream = _fake_stream

        with patch("exo_server.streaming._run_agent", mock_run):
            async with _build_client(app) as client:
                resp = await client.get("/stream", params={"message": "hi"})

        assert "data: [DONE]" in resp.text


# ---------------------------------------------------------------------------
# Route registration
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


class TestRouteRegistration:
    def test_ws_route_exists(self) -> None:
        app = create_app()
        routes = _all_paths(app)
        assert "/ws/chat" in routes

    def test_sse_route_exists(self) -> None:
        app = create_app()
        routes = _all_paths(app)
        assert "/stream" in routes
