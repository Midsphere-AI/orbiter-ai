"""Tests for exo_server.app — FastAPI app with /chat endpoint."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient
from pydantic import ValidationError

from exo_server.app import (
    ChatRequest,
    ChatResponse,
    InjectRequest,
    _get_agent,
    create_app,
    register_agent,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_agent(name: str = "test-agent") -> Any:
    """Create a mock agent with a name attribute."""
    agent = MagicMock()
    agent.name = name
    return agent


def _mock_run_result(output: str = "hello", steps: int = 1) -> Any:
    """Create a mock RunResult."""
    result = MagicMock()
    result.output = output
    result.steps = steps
    result.usage = MagicMock(input_tokens=10, output_tokens=5, total_tokens=15)
    return result


def _build_client(app: Any) -> AsyncClient:
    """Build an httpx async test client for the FastAPI app."""
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


# ---------------------------------------------------------------------------
# ChatRequest / ChatResponse models
# ---------------------------------------------------------------------------


class TestChatRequest:
    def test_defaults(self) -> None:
        req = ChatRequest(message="hi")
        assert req.message == "hi"
        assert req.agent_name is None
        assert req.stream is False

    def test_with_all_fields(self) -> None:
        req = ChatRequest(message="hello", agent_name="bot", stream=True)
        assert req.agent_name == "bot"
        assert req.stream is True


class TestChatResponse:
    def test_defaults(self) -> None:
        resp = ChatResponse()
        assert resp.output == ""
        assert resp.agent_name == ""
        assert resp.steps == 0
        assert resp.usage == {}

    def test_with_data(self) -> None:
        resp = ChatResponse(output="hi", agent_name="bot", steps=2, usage={"total_tokens": 42})
        assert resp.output == "hi"
        assert resp.usage["total_tokens"] == 42


# ---------------------------------------------------------------------------
# Agent registration
# ---------------------------------------------------------------------------


class TestRegisterAgent:
    def test_register_single(self) -> None:
        app = create_app()
        agent = _mock_agent("alpha")
        register_agent(app, agent)
        agents = getattr(app.state, "exo_agents", {})
        assert "alpha" in agents
        # First agent auto-becomes default
        assert getattr(app.state, "exo_default_agent", None) == "alpha"

    def test_register_multiple(self) -> None:
        app = create_app()
        register_agent(app, _mock_agent("a"))
        register_agent(app, _mock_agent("b"))
        agents = getattr(app.state, "exo_agents", {})
        assert len(agents) == 2
        # First registered remains default
        assert getattr(app.state, "exo_default_agent", None) == "a"

    def test_register_explicit_default(self) -> None:
        app = create_app()
        register_agent(app, _mock_agent("a"))
        register_agent(app, _mock_agent("b"), default=True)
        assert getattr(app.state, "exo_default_agent", None) == "b"


class TestGetAgent:
    def test_no_agents(self) -> None:
        app = create_app()
        with pytest.raises(Exception, match="No agents registered"):
            _get_agent(app, None)

    def test_agent_not_found(self) -> None:
        app = create_app()
        register_agent(app, _mock_agent("a"))
        with pytest.raises(Exception, match="not found"):
            _get_agent(app, "nope")

    def test_resolve_by_name(self) -> None:
        app = create_app()
        agent = _mock_agent("alpha")
        register_agent(app, agent)
        assert _get_agent(app, "alpha") is agent

    def test_resolve_default(self) -> None:
        app = create_app()
        agent = _mock_agent("alpha")
        register_agent(app, agent)
        assert _get_agent(app, None) is agent

    def test_no_default_no_name(self) -> None:
        app = create_app()
        # Manually set agents but no default
        app.state.exo_agents = {"a": _mock_agent("a")}  # type: ignore[attr-defined]
        app.state.exo_default_agent = None  # type: ignore[attr-defined]
        with pytest.raises(Exception, match="No agent_name"):
            _get_agent(app, None)


# ---------------------------------------------------------------------------
# /chat endpoint — non-streaming
# ---------------------------------------------------------------------------


class TestChatEndpoint:
    async def test_chat_success(self) -> None:
        app = create_app()
        agent = _mock_agent("bot")
        register_agent(app, agent)
        mock_result = _mock_run_result("Hello world!", steps=2)

        with patch("exo_server.app._run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = mock_result
            async with _build_client(app) as client:
                resp = await client.post("/chat", json={"message": "hi"})

        assert resp.status_code == 200
        data = resp.json()
        assert data["output"] == "Hello world!"
        assert data["agent_name"] == "bot"
        assert data["steps"] == 2
        assert data["usage"]["total_tokens"] == 15

    async def test_chat_with_agent_name(self) -> None:
        app = create_app()
        register_agent(app, _mock_agent("a"))
        agent_b = _mock_agent("b")
        register_agent(app, agent_b)
        mock_result = _mock_run_result("from b")

        with patch("exo_server.app._run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = mock_result
            async with _build_client(app) as client:
                resp = await client.post("/chat", json={"message": "hi", "agent_name": "b"})

        assert resp.status_code == 200
        assert resp.json()["output"] == "from b"
        # Verify run was called with agent_b
        mock_run.assert_called_once()
        args = mock_run.call_args
        assert args[0][0] is agent_b

    async def test_chat_no_agents_503(self) -> None:
        app = create_app()
        async with _build_client(app) as client:
            resp = await client.post("/chat", json={"message": "hi"})
        assert resp.status_code == 503

    async def test_chat_agent_not_found_404(self) -> None:
        app = create_app()
        register_agent(app, _mock_agent("a"))
        async with _build_client(app) as client:
            resp = await client.post("/chat", json={"message": "hi", "agent_name": "nope"})
        assert resp.status_code == 404

    async def test_chat_run_error_500(self) -> None:
        app = create_app()
        register_agent(app, _mock_agent("bot"))

        with patch(
            "exo_server.app._run_agent",
            new_callable=AsyncMock,
            side_effect=RuntimeError("boom"),
        ):
            async with _build_client(app) as client:
                resp = await client.post("/chat", json={"message": "hi"})

        assert resp.status_code == 500
        assert "boom" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# /chat endpoint — streaming SSE
# ---------------------------------------------------------------------------


class TestChatStreaming:
    # The /chat streaming path delegates to _sse_iter in exo_server.streaming,
    # so mock the _run_agent there (not in exo_server.app).
    _STREAM_PATCH = "exo_server.streaming._run_agent"

    async def test_stream_text_events(self) -> None:
        app = create_app()
        register_agent(app, _mock_agent("bot"))

        # Mock stream events
        async def _fake_stream(agent: Any, message: str, **kw: Any) -> Any:
            event1 = MagicMock(type="text", text="Hello ")
            event2 = MagicMock(type="text", text="world!")
            for e in [event1, event2]:
                yield e

        mock_run = MagicMock()
        mock_run.stream = _fake_stream

        with patch(self._STREAM_PATCH, mock_run):
            async with _build_client(app) as client:
                resp = await client.post("/chat", json={"message": "hi", "stream": True})

        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]
        lines = [line for line in resp.text.strip().split("\n") if line.startswith("data:")]
        # Should have: text "Hello ", text "world!", [DONE]
        assert len(lines) >= 3
        first = json.loads(lines[0].removeprefix("data: "))
        assert first["type"] == "text"
        assert first["text"] == "Hello "

    async def test_stream_tool_call_event(self) -> None:
        app = create_app()
        register_agent(app, _mock_agent("bot"))

        async def _fake_stream(agent: Any, message: str, **kw: Any) -> Any:
            event = MagicMock(type="tool_call", tool_name="search", tool_call_id="tc1")
            yield event

        mock_run = MagicMock()
        mock_run.stream = _fake_stream

        with patch(self._STREAM_PATCH, mock_run):
            async with _build_client(app) as client:
                resp = await client.post("/chat", json={"message": "hi", "stream": True})

        lines = [line for line in resp.text.strip().split("\n") if line.startswith("data:")]
        first = json.loads(lines[0].removeprefix("data: "))
        assert first["type"] == "tool_call"
        assert first["tool_name"] == "search"

    async def test_stream_done_marker(self) -> None:
        app = create_app()
        register_agent(app, _mock_agent("bot"))

        async def _fake_stream(agent: Any, message: str, **kw: Any) -> Any:
            event = MagicMock(type="text", text="hi")
            yield event

        mock_run = MagicMock()
        mock_run.stream = _fake_stream

        with patch(self._STREAM_PATCH, mock_run):
            async with _build_client(app) as client:
                resp = await client.post("/chat", json={"message": "hi", "stream": True})

        assert "data: [DONE]" in resp.text

    async def test_stream_error_event(self) -> None:
        app = create_app()
        register_agent(app, _mock_agent("bot"))

        async def _fake_stream(agent: Any, message: str, **kw: Any) -> Any:
            raise RuntimeError("stream failed")
            yield  # make it a generator  # pragma: no cover

        mock_run = MagicMock()
        mock_run.stream = _fake_stream

        with patch(self._STREAM_PATCH, mock_run):
            async with _build_client(app) as client:
                resp = await client.post("/chat", json={"message": "hi", "stream": True})

        lines = [line for line in resp.text.strip().split("\n") if line.startswith("data:")]
        error_line = json.loads(lines[0].removeprefix("data: "))
        assert "error" in error_line


# ---------------------------------------------------------------------------
# /inject endpoint
# ---------------------------------------------------------------------------


class TestInjectRequest:
    def test_empty_message_raises(self) -> None:
        with pytest.raises(ValidationError):
            InjectRequest(message="")

    def test_whitespace_only_message_raises(self) -> None:
        with pytest.raises(ValidationError):
            InjectRequest(message="   ")

    def test_valid_message(self) -> None:
        req = InjectRequest(message="hello")
        assert req.message == "hello"

    def test_with_agent_name(self) -> None:
        req = InjectRequest(message="hi", agent_name="bot")
        assert req.agent_name == "bot"


class TestInjectEndpoint:
    async def test_inject_success(self) -> None:
        app = create_app()
        agent = _mock_agent("bot")
        agent.inject_message = MagicMock()
        register_agent(app, agent)

        async with _build_client(app) as client:
            resp = await client.post("/inject", json={"message": "hello"})

        assert resp.status_code == 200
        assert resp.json() == {"status": "injected"}
        agent.inject_message.assert_called_once_with("hello")

    async def test_inject_empty_message_422(self) -> None:
        """Empty message must be rejected at the model level (422)."""
        app = create_app()
        register_agent(app, _mock_agent("bot"))

        async with _build_client(app) as client:
            resp = await client.post("/inject", json={"message": ""})

        assert resp.status_code == 422

    async def test_inject_no_agents_503(self) -> None:
        app = create_app()

        async with _build_client(app) as client:
            resp = await client.post("/inject", json={"message": "hello"})

        assert resp.status_code == 503

    async def test_inject_agent_not_found_404(self) -> None:
        app = create_app()
        register_agent(app, _mock_agent("bot"))

        async with _build_client(app) as client:
            resp = await client.post("/inject", json={"message": "hello", "agent_name": "nope"})

        assert resp.status_code == 404

    async def test_inject_value_error_422(self) -> None:
        """If agent.inject_message raises ValueError it should become HTTP 422."""
        app = create_app()
        agent = _mock_agent("bot")
        agent.inject_message = MagicMock(side_effect=ValueError("bad message"))
        register_agent(app, agent)

        async with _build_client(app) as client:
            resp = await client.post("/inject", json={"message": "hello"})

        assert resp.status_code == 422
        assert "bad message" in resp.json()["detail"]

    async def test_inject_with_agent_name(self) -> None:
        app = create_app()
        agent_a = _mock_agent("a")
        agent_b = _mock_agent("b")
        agent_a.inject_message = MagicMock()
        agent_b.inject_message = MagicMock()
        register_agent(app, agent_a)
        register_agent(app, agent_b)

        async with _build_client(app) as client:
            resp = await client.post("/inject", json={"message": "hi", "agent_name": "b"})

        assert resp.status_code == 200
        agent_b.inject_message.assert_called_once_with("hi")
        agent_a.inject_message.assert_not_called()


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


class TestCreateApp:
    def test_creates_fastapi_app(self) -> None:
        app = create_app()
        assert app.title == "Exo Server"

    def test_has_chat_route(self) -> None:
        app = create_app()
        routes = [r.path for r in app.routes if hasattr(r, "path")]  # type: ignore[union-attr]
        assert "/chat" in routes
