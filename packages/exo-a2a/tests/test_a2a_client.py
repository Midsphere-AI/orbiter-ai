"""Tests for exo.a2a.client — A2A HTTP client, ClientManager, RemoteAgent."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from exo.a2a.client import (  # pyright: ignore[reportMissingImports]
    A2AClient,
    A2AClientError,
    ClientManager,
    RemoteAgent,
    _extract_text,
)
from exo.a2a.types import (  # pyright: ignore[reportMissingImports]
    AgentCapabilities,
    AgentCard,
    ClientConfig,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_card(
    name: str = "remote-agent",
    url: str = "http://remote:9000",
    streaming: bool = False,
) -> AgentCard:
    return AgentCard(
        name=name,
        url=url,
        capabilities=AgentCapabilities(streaming=streaming),
    )


def _task_response(text: str = "hello", task_id: str = "t-1") -> dict[str, Any]:
    return {
        "task_id": task_id,
        "status": {"state": "completed"},
        "artifact": {"task_id": task_id, "text": text, "last_chunk": True},
    }


# ===========================================================================
# A2AClient — init
# ===========================================================================


class TestA2AClientInit:
    def test_with_agent_card(self) -> None:
        card = _make_card()
        client = A2AClient(card)
        assert repr(client) == "A2AClient('remote-agent')"

    def test_with_url_string(self) -> None:
        client = A2AClient("http://example.com/.well-known/agent-card")
        assert "unresolved" not in repr(client)

    def test_with_file_path(self) -> None:
        client = A2AClient("/tmp/agent-card.json")
        assert "unresolved" not in repr(client)

    def test_empty_string_raises(self) -> None:
        with pytest.raises(A2AClientError, match="cannot be empty"):
            A2AClient("")

    def test_invalid_type_raises(self) -> None:
        with pytest.raises(A2AClientError, match="must be AgentCard"):
            A2AClient(42)  # type: ignore[arg-type]

    def test_default_config(self) -> None:
        client = A2AClient(_make_card())
        assert client._config.timeout == 600.0


# ===========================================================================
# A2AClient — agent card resolution
# ===========================================================================


class TestA2AClientResolveCard:
    async def test_already_resolved(self) -> None:
        card = _make_card()
        client = A2AClient(card)
        resolved = await client.resolve_agent_card()
        assert resolved is card

    async def test_resolve_from_file(self, tmp_path: Path) -> None:
        card_data = {"name": "file-agent", "url": "http://localhost:8000"}
        card_file = tmp_path / "agent-card.json"
        card_file.write_text(json.dumps(card_data))

        client = A2AClient(str(card_file))
        resolved = await client.resolve_agent_card()
        assert resolved.name == "file-agent"
        assert resolved.url == "http://localhost:8000"

    async def test_resolve_from_file_not_found(self) -> None:
        client = A2AClient("/nonexistent/path/card.json")
        with pytest.raises(A2AClientError, match="not found"):
            await client.resolve_agent_card()

    async def test_resolve_from_file_invalid_json(self, tmp_path: Path) -> None:
        bad_file = tmp_path / "bad.json"
        bad_file.write_text("not json")
        client = A2AClient(str(bad_file))
        with pytest.raises(A2AClientError, match="Invalid agent card"):
            await client.resolve_agent_card()

    async def test_resolve_from_url(self) -> None:
        card_data = {"name": "url-agent", "url": "http://remote:9000"}
        mock_response = MagicMock()
        mock_response.json.return_value = card_data
        mock_response.raise_for_status = MagicMock()

        client = A2AClient("http://remote:9000/.well-known/agent-card")
        client._http = AsyncMock()
        client._http.get = AsyncMock(return_value=mock_response)

        resolved = await client.resolve_agent_card()
        assert resolved.name == "url-agent"

    async def test_resolve_from_url_failure(self) -> None:
        client = A2AClient("http://unreachable:9000/.well-known/agent-card")
        client._http = AsyncMock()
        client._http.get = AsyncMock(side_effect=httpx.ConnectError("refused"))

        with pytest.raises(A2AClientError, match="Failed to fetch"):
            await client.resolve_agent_card()

    async def test_resolve_caches_result(self, tmp_path: Path) -> None:
        card_data = {"name": "cached", "url": "http://localhost:8000"}
        card_file = tmp_path / "card.json"
        card_file.write_text(json.dumps(card_data))

        client = A2AClient(str(card_file))
        first = await client.resolve_agent_card()
        second = await client.resolve_agent_card()
        assert first is second


# ===========================================================================
# A2AClient — send_task
# ===========================================================================


class TestA2AClientSendTask:
    async def test_send_task_success(self) -> None:
        card = _make_card()
        client = A2AClient(card)

        resp_data = _task_response()
        mock_resp = MagicMock()
        mock_resp.json.return_value = resp_data
        mock_resp.raise_for_status = MagicMock()
        client._http = AsyncMock()
        client._http.post = AsyncMock(return_value=mock_resp)

        result = await client.send_task("test input")
        assert result["task_id"] == "t-1"
        assert result["artifact"]["text"] == "hello"

    async def test_send_task_with_task_id(self) -> None:
        card = _make_card()
        client = A2AClient(card)

        mock_resp = MagicMock()
        mock_resp.json.return_value = _task_response(task_id="custom-id")
        mock_resp.raise_for_status = MagicMock()
        client._http = AsyncMock()
        client._http.post = AsyncMock(return_value=mock_resp)

        result = await client.send_task("hi", task_id="custom-id")
        assert result["task_id"] == "custom-id"
        # Verify task_id was in the request payload
        call_kwargs = client._http.post.call_args
        assert call_kwargs.kwargs["json"]["task_id"] == "custom-id"

    async def test_send_task_failure(self) -> None:
        card = _make_card()
        client = A2AClient(card)
        client._http = AsyncMock()
        client._http.post = AsyncMock(side_effect=httpx.ConnectError("refused"))

        with pytest.raises(A2AClientError, match="Task request failed"):
            await client.send_task("test")


# ===========================================================================
# A2AClient — streaming
# ===========================================================================


class TestA2AClientStreaming:
    async def test_streaming_not_supported(self) -> None:
        card = _make_card(streaming=False)
        client = A2AClient(card)
        with pytest.raises(A2AClientError, match="does not support streaming"):
            await client.send_task_streaming("test")

    async def test_streaming_success(self) -> None:
        card = _make_card(streaming=True)
        client = A2AClient(card)

        events = [
            {"task_id": "s-1", "status": {"state": "working"}},
            {"task_id": "s-1", "text": "result", "last_chunk": True},
            {"task_id": "s-1", "status": {"state": "completed"}},
        ]
        ndjson = "\n".join(json.dumps(e) for e in events)

        mock_resp = MagicMock()
        mock_resp.text = ndjson
        mock_resp.raise_for_status = MagicMock()
        client._http = AsyncMock()
        client._http.post = AsyncMock(return_value=mock_resp)

        result = await client.send_task_streaming("test")
        assert len(result) == 3
        assert result[0]["status"]["state"] == "working"
        assert result[2]["status"]["state"] == "completed"

    async def test_streaming_failure(self) -> None:
        card = _make_card(streaming=True)
        client = A2AClient(card)
        client._http = AsyncMock()
        client._http.post = AsyncMock(side_effect=httpx.ConnectError("refused"))

        with pytest.raises(A2AClientError, match="Stream request failed"):
            await client.send_task_streaming("test")


# ===========================================================================
# A2AClient — lifecycle
# ===========================================================================


class TestA2AClientLifecycle:
    async def test_close(self) -> None:
        client = A2AClient(_make_card())
        client._http = AsyncMock()
        await client.close()
        client._http.aclose.assert_awaited_once()

    def test_repr_resolved(self) -> None:
        client = A2AClient(_make_card("my-agent"))
        assert repr(client) == "A2AClient('my-agent')"

    def test_repr_unresolved_url(self) -> None:
        client = A2AClient("http://example.com/card")
        assert "http://example.com/card" in repr(client)

    def test_repr_no_source(self) -> None:
        """When agent_card is AgentCard, repr shows its name."""
        card = _make_card("named")
        client = A2AClient(card)
        assert "named" in repr(client)


# ===========================================================================
# ClientManager
# ===========================================================================


class TestClientManager:
    def test_get_client_returns_client(self) -> None:
        mgr = ClientManager(_make_card())
        client = mgr.get_client()
        assert isinstance(client, A2AClient)

    def test_same_thread_same_client(self) -> None:
        mgr = ClientManager(_make_card())
        c1 = mgr.get_client()
        c2 = mgr.get_client()
        assert c1 is c2

    def test_different_threads_different_clients(self) -> None:
        mgr = ClientManager(_make_card())
        c1 = mgr.get_client()
        results: list[A2AClient] = []

        def worker() -> None:
            results.append(mgr.get_client())

        t = threading.Thread(target=worker)
        t.start()
        t.join()

        assert len(results) == 1
        assert results[0] is not c1

    async def test_shutdown(self) -> None:
        mgr = ClientManager(_make_card())
        client = mgr.get_client()
        client._http = AsyncMock()
        await mgr.shutdown()
        client._http.aclose.assert_awaited_once()
        assert repr(mgr) == "ClientManager(clients=0)"

    def test_repr(self) -> None:
        mgr = ClientManager(_make_card())
        assert repr(mgr) == "ClientManager(clients=0)"
        mgr.get_client()
        assert repr(mgr) == "ClientManager(clients=1)"

    def test_custom_config(self) -> None:
        config = ClientConfig(timeout=30.0)
        mgr = ClientManager(_make_card(), config=config)
        client = mgr.get_client()
        assert client._config.timeout == 30.0


# ===========================================================================
# RemoteAgent
# ===========================================================================


class TestRemoteAgentInit:
    def test_basic_creation(self) -> None:
        agent = RemoteAgent(name="remote", agent_card=_make_card())
        assert agent.name == "remote"

    def test_with_config(self) -> None:
        config = ClientConfig(timeout=30.0)
        agent = RemoteAgent(name="remote", agent_card=_make_card(), config=config)
        assert agent._client._config.timeout == 30.0

    def test_repr(self) -> None:
        agent = RemoteAgent(name="r1", agent_card=_make_card("target"))
        assert "r1" in repr(agent)
        assert "target" in repr(agent)


class TestRemoteAgentRun:
    async def test_run_success(self) -> None:
        card = _make_card()
        agent = RemoteAgent(name="proxy", agent_card=card)

        resp = _task_response("world")
        mock_resp = MagicMock()
        mock_resp.json.return_value = resp
        mock_resp.raise_for_status = MagicMock()
        agent._client._http = AsyncMock()
        agent._client._http.post = AsyncMock(return_value=mock_resp)

        output = await agent.run("hello")
        assert output.text == "world"
        assert output.tool_calls == []

    async def test_run_empty_response(self) -> None:
        card = _make_card()
        agent = RemoteAgent(name="proxy", agent_card=card)

        resp = {"task_id": "t-1", "status": {"state": "completed"}}
        mock_resp = MagicMock()
        mock_resp.json.return_value = resp
        mock_resp.raise_for_status = MagicMock()
        agent._client._http = AsyncMock()
        agent._client._http.post = AsyncMock(return_value=mock_resp)

        output = await agent.run("hello")
        assert output.text == ""

    async def test_run_extracts_result_field(self) -> None:
        card = _make_card()
        agent = RemoteAgent(name="proxy", agent_card=card)

        resp = {"task_id": "t-1", "status": {"state": "completed"}, "result": "from result"}
        mock_resp = MagicMock()
        mock_resp.json.return_value = resp
        mock_resp.raise_for_status = MagicMock()
        agent._client._http = AsyncMock()
        agent._client._http.post = AsyncMock(return_value=mock_resp)

        output = await agent.run("hello")
        assert output.text == "from result"


class TestRemoteAgentDescribe:
    def test_describe_sync(self) -> None:
        """describe() is synchronous to match local Agent.describe() signature."""
        card = _make_card("target-agent", "http://remote:9000")
        agent = RemoteAgent(name="local-proxy", agent_card=card)
        desc = agent.describe()
        assert desc["name"] == "local-proxy"
        assert desc["remote_name"] == "target-agent"
        assert desc["url"] == "http://remote:9000"

    def test_describe_unresolved(self) -> None:
        """describe() on an unresolved agent returns minimal info."""
        agent = RemoteAgent(name="proxy", agent_card="http://remote:9000/card")
        desc = agent.describe()
        assert desc["name"] == "proxy"
        assert desc["remote_name"] is None
        assert "http://remote:9000/card" in desc["url"]

    async def test_describe_async(self) -> None:
        """describe_async() resolves the card and returns full info."""
        card = _make_card("target-agent", "http://remote:9000")
        agent = RemoteAgent(name="local-proxy", agent_card=card)
        desc = await agent.describe_async()
        assert desc["name"] == "local-proxy"
        assert desc["remote_name"] == "target-agent"
        assert desc["url"] == "http://remote:9000"


class TestRemoteAgentClose:
    async def test_close(self) -> None:
        agent = RemoteAgent(name="r", agent_card=_make_card())
        agent._client._http = AsyncMock()
        await agent.close()
        agent._client._http.aclose.assert_awaited_once()


# ===========================================================================
# _extract_text helper
# ===========================================================================


class TestExtractText:
    def test_from_artifact(self) -> None:
        resp = {"artifact": {"text": "hello", "last_chunk": True}}
        assert _extract_text(resp) == "hello"

    def test_from_result(self) -> None:
        resp = {"result": "world"}
        assert _extract_text(resp) == "world"

    def test_empty_response(self) -> None:
        assert _extract_text({}) == ""

    def test_artifact_empty_text(self) -> None:
        resp = {"artifact": {"text": ""}, "result": "fallback"}
        assert _extract_text(resp) == "fallback"

    def test_non_dict_artifact(self) -> None:
        resp = {"artifact": "not-a-dict", "result": "ok"}
        assert _extract_text(resp) == "ok"
