"""Tests for exo.a2a.client — A2A HTTP client and RemoteAgent."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from exo.a2a.client import (  # pyright: ignore[reportMissingImports]
    A2AClient,
    A2AClientError,
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

        with pytest.raises(A2AClientError, match="Task POST to A2A peer"):
            await client.send_task("test")


# ===========================================================================
# A2AClient — streaming
# ===========================================================================


class TestA2AClientCollect:
    """Tests for send_task_collect — buffered NDJSON event collector."""

    async def test_collect_not_supported(self) -> None:
        card = _make_card(streaming=False)
        client = A2AClient(card)
        with pytest.raises(A2AClientError, match="does not support streaming"):
            await client.send_task_collect("test")

    async def test_collect_success(self) -> None:
        card = _make_card(streaming=True)
        client = A2AClient(card)

        events = [
            {"task_id": "s-1", "status": {"state": "working"}},
            {"task_id": "s-1", "text": "result", "last_chunk": True},
            {"task_id": "s-1", "status": {"state": "completed"}},
        ]
        lines = [json.dumps(e) for e in events]

        async def _aiter_lines():
            for line in lines:
                yield line

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.aiter_lines = _aiter_lines
        client._http = AsyncMock()
        client._http.post = AsyncMock(return_value=mock_resp)

        result = await client.send_task_collect("test")
        assert len(result) == 3
        assert result[0]["status"]["state"] == "working"
        assert result[2]["status"]["state"] == "completed"

    async def test_collect_failure(self) -> None:
        card = _make_card(streaming=True)
        client = A2AClient(card)
        client._http = AsyncMock()
        client._http.post = AsyncMock(side_effect=httpx.ConnectError("refused"))

        with pytest.raises(A2AClientError, match="Stream request to A2A peer"):
            await client.send_task_collect("test")


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


# ===========================================================================
# Error DX — malformed JSON / schema mismatch
# ===========================================================================


class TestMalformedNDJSON:
    """send_task_collect wraps json.JSONDecodeError as A2AClientError."""

    async def test_malformed_ndjson_line_raises(self) -> None:
        card = _make_card(streaming=True)
        client = A2AClient(card)

        raw_lines = ['{"ok": 1}', "not-valid-json", '{"ok": 3}']

        async def _aiter_lines():
            for line in raw_lines:
                yield line

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.aiter_lines = _aiter_lines
        client._http = AsyncMock()
        client._http.post = AsyncMock(return_value=mock_resp)

        with pytest.raises(A2AClientError, match="Malformed NDJSON"):
            await client.send_task_collect("test")

    async def test_malformed_ndjson_carries_context(self) -> None:
        card = _make_card(streaming=True)
        client = A2AClient(card)

        async def _aiter_lines():
            yield "not-json"

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.aiter_lines = _aiter_lines
        client._http = AsyncMock()
        client._http.post = AsyncMock(return_value=mock_resp)

        with pytest.raises(A2AClientError) as exc_info:
            await client.send_task_collect("test")
        err = exc_info.value
        assert err.context is not None
        assert "url" in err.context
        assert err.hint is not None


class TestMalformedAgentCardJSON:
    """_resolve_from_url wraps json.JSONDecodeError and ValidationError as A2AClientError."""

    async def test_json_decode_error_raises_client_error(self) -> None:
        import json as _json

        client = A2AClient("http://remote:9000/.well-known/agent-card")
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.side_effect = _json.JSONDecodeError("bad", "", 0)
        client._http = AsyncMock()
        client._http.get = AsyncMock(return_value=mock_resp)

        with pytest.raises(A2AClientError, match="Invalid agent card JSON"):
            await client.resolve_agent_card()

    async def test_schema_mismatch_raises_client_error(self) -> None:
        """AgentCard with missing required field 'name' raises A2AClientError."""
        client = A2AClient("http://remote:9000/.well-known/agent-card")
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        # Return JSON that fails AgentCard validation (missing required 'name')
        mock_resp.json.return_value = {"not_a_valid_field": "bad"}
        client._http = AsyncMock()
        client._http.get = AsyncMock(return_value=mock_resp)

        with pytest.raises(A2AClientError):
            await client.resolve_agent_card()


# ===========================================================================
# Context manager — async with A2AClient / RemoteAgent
# ===========================================================================


class TestAsyncContextManager:
    async def test_a2a_client_aenter_returns_self(self) -> None:
        client = A2AClient(_make_card())
        client._http = AsyncMock()
        async with client as c:
            assert c is client

    async def test_a2a_client_aexit_calls_close(self) -> None:
        client = A2AClient(_make_card())
        client._http = AsyncMock()
        async with client:
            pass
        client._http.aclose.assert_awaited_once()

    async def test_remote_agent_aenter_returns_self(self) -> None:
        agent = RemoteAgent(name="r", agent_card=_make_card())
        agent._client._http = AsyncMock()
        async with agent as a:
            assert a is agent

    async def test_remote_agent_aexit_calls_close(self) -> None:
        agent = RemoteAgent(name="r", agent_card=_make_card())
        agent._client._http = AsyncMock()
        async with agent:
            pass
        agent._client._http.aclose.assert_awaited_once()


# ===========================================================================
# ClientConfig — max_retries / retry_delay defaults
# ===========================================================================


class TestClientConfigRetryFields:
    def test_default_max_retries(self) -> None:
        config = ClientConfig()
        assert config.max_retries == 1

    def test_custom_max_retries(self) -> None:
        config = ClientConfig(max_retries=3)
        assert config.max_retries == 3

    def test_default_retry_delay(self) -> None:
        config = ClientConfig()
        assert config.retry_delay == 0.5


# ===========================================================================
# Retry actually fires on TransportError (Finding 1)
# ===========================================================================


class TestRetryOnTransportError:
    """TransportError must propagate out of the inner function so retry_async fires."""

    async def test_resolve_from_url_retries_on_transport_error(self) -> None:
        """resolve_agent_card retries on TransportError, succeeds on second attempt."""
        card_data = {"name": "retry-agent", "url": "http://remote:9000"}

        mock_ok = MagicMock()
        mock_ok.json.return_value = card_data
        mock_ok.raise_for_status = MagicMock()

        call_count = 0

        async def _flaky_get(url: str, **kwargs: Any) -> Any:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise httpx.ConnectError("transient failure")
            return mock_ok

        config = ClientConfig(max_retries=3, retry_delay=0.01)
        client = A2AClient("http://remote:9000/.well-known/agent-card", config)
        client._http = AsyncMock()
        client._http.get = _flaky_get

        resolved = await client.resolve_agent_card()
        assert resolved.name == "retry-agent"
        assert call_count == 2  # failed once, then succeeded

    async def test_send_task_retries_on_transport_error(self) -> None:
        """send_task retries on TransportError, succeeds on second attempt."""
        card = _make_card(url="http://remote:9000")
        resp_data = _task_response("ok")

        mock_ok = MagicMock()
        mock_ok.json.return_value = resp_data
        mock_ok.raise_for_status = MagicMock()

        call_count = 0

        async def _flaky_post(url: str, **kwargs: Any) -> Any:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise httpx.ConnectError("transient failure")
            return mock_ok

        config = ClientConfig(max_retries=3, retry_delay=0.01)
        client = A2AClient(card, config)
        client._http = AsyncMock()
        client._http.post = _flaky_post

        result = await client.send_task("hello")
        assert result["artifact"]["text"] == "ok"
        assert call_count == 2

    async def test_resolve_from_url_exhausts_retries_raises_client_error(self) -> None:
        """When all retries are exhausted, A2AClientError is raised (not TransportError)."""
        config = ClientConfig(max_retries=2, retry_delay=0.01)
        client = A2AClient("http://remote:9000/.well-known/agent-card", config)
        client._http = AsyncMock()
        client._http.get = AsyncMock(side_effect=httpx.ConnectError("always fails"))

        with pytest.raises(A2AClientError, match="Failed to fetch"):
            await client.resolve_agent_card()

    async def test_send_task_exhausts_retries_raises_client_error(self) -> None:
        """When all retries are exhausted, A2AClientError is raised (not TransportError)."""
        card = _make_card(url="http://remote:9000")
        config = ClientConfig(max_retries=2, retry_delay=0.01)
        client = A2AClient(card, config)
        client._http = AsyncMock()
        client._http.post = AsyncMock(side_effect=httpx.ConnectError("always fails"))

        with pytest.raises(A2AClientError, match="Task POST to A2A peer"):
            await client.send_task("hello")


# ===========================================================================
# Empty card.url guard (Finding 3)
# ===========================================================================


class TestEmptyCardUrl:
    async def test_send_task_empty_url_raises(self) -> None:
        """send_task raises A2AClientError when card.url is empty."""
        card = _make_card(url="")
        client = A2AClient(card)
        with pytest.raises(A2AClientError, match="empty URL"):
            await client.send_task("hello")

    async def test_send_task_collect_empty_url_raises(self) -> None:
        """send_task_collect raises A2AClientError when card.url is empty."""
        card = _make_card(url="", streaming=True)
        client = A2AClient(card)
        with pytest.raises(A2AClientError, match="empty URL"):
            await client.send_task_collect("hello")


# ===========================================================================
# SSRF guard — card.url origin validation (Finding 1)
# ===========================================================================


class TestSSRFOriginValidation:
    """_resolve_from_url rejects card.url that differs in origin from the discovery URL."""

    def test_same_origin_passes(self) -> None:
        assert A2AClient._same_origin(
            "http://remote:9000/.well-known/agent-card",
            "http://remote:9000/",
        )

    def test_different_host_fails(self) -> None:
        assert not A2AClient._same_origin(
            "http://remote:9000/.well-known/agent-card",
            "http://internal:9000/",
        )

    def test_different_port_fails(self) -> None:
        assert not A2AClient._same_origin(
            "http://remote:9000/.well-known/agent-card",
            "http://remote:8080/",
        )

    def test_different_scheme_fails(self) -> None:
        assert not A2AClient._same_origin(
            "https://remote:443/.well-known/agent-card",
            "http://remote:443/",
        )

    def test_default_http_port_normalised(self) -> None:
        """http on port 80 is the same origin as http with no explicit port."""
        assert A2AClient._same_origin(
            "http://remote:80/.well-known/agent-card",
            "http://remote/",
        )

    async def test_resolve_from_url_rejects_cross_origin_card_url(self) -> None:
        """_resolve_from_url raises A2AClientError when card.url differs in origin."""
        client = A2AClient("http://trusted:9000/.well-known/agent-card")

        # Agent card advertises a different host — SSRF attempt.
        card_data = {
            "name": "evil",
            "url": "http://internal-service:80/",
            "capabilities": {},
        }
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = card_data
        client._http = AsyncMock()
        client._http.get = AsyncMock(return_value=mock_resp)

        with pytest.raises(A2AClientError, match="SSRF"):
            await client.resolve_agent_card()

    async def test_resolve_from_url_accepts_same_origin_card_url(self) -> None:
        """_resolve_from_url accepts card.url that matches the discovery origin."""
        client = A2AClient("http://trusted:9000/.well-known/agent-card")

        card_data = {
            "name": "ok",
            "url": "http://trusted:9000/",
            "capabilities": {},
        }
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = card_data
        client._http = AsyncMock()
        client._http.get = AsyncMock(return_value=mock_resp)

        card = await client.resolve_agent_card()
        assert card.name == "ok"

    async def test_resolve_from_url_allows_empty_card_url(self) -> None:
        """Empty card.url bypasses the SSRF check (caught later by send_task)."""
        client = A2AClient("http://trusted:9000/.well-known/agent-card")

        card_data = {"name": "no-url", "url": "", "capabilities": {}}
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = card_data
        client._http = AsyncMock()
        client._http.get = AsyncMock(return_value=mock_resp)

        card = await client.resolve_agent_card()
        assert card.url == ""
