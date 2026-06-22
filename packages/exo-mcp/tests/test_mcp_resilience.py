"""Resilience tests for exo-mcp: timeouts, partial failures, and error quality.

Covers the P0/P1 findings from the error-DX audit:
- Handshake / list_tools / call_tool timeouts → actionable MCPClientError (not hangs)
- connect_all() partial failure: good servers stay connected, bad ones aggregated
- Unknown transport in mcp.json → MCPExecutionError with hint
"""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from exo.mcp.client import (  # pyright: ignore[reportMissingImports]
    MCPClient,
    MCPClientError,
    MCPServerConfig,
    MCPServerConnection,
    MCPTransport,
)
from exo.mcp.execution import (  # pyright: ignore[reportMissingImports]
    MCPExecutionError,
    load_mcp_config,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


@asynccontextmanager
async def _mock_transport():
    """Minimal mock transport that yields (read, write)."""
    yield (MagicMock(), MagicMock())


def _make_stdio_config(name: str = "test-server", timeout: float = 30.0) -> MCPServerConfig:
    return MCPServerConfig(
        name=name,
        transport=MCPTransport.STDIO,
        command="python",
        args=["-m", "fake_server"],
        timeout=timeout,
        session_timeout=30.0,
    )


def _hanging_coroutine():
    """A coroutine that sleeps forever — simulates a hung MCP server."""

    async def _hang(*args: Any, **kwargs: Any) -> Any:
        await asyncio.sleep(9999)

    return _hang


# ---------------------------------------------------------------------------
# Timeout: session.initialize() / handshake
# ---------------------------------------------------------------------------


class TestHandshakeTimeout:
    """session.initialize() must time out and raise an actionable MCPClientError."""

    async def test_handshake_timeout_raises_mcp_client_error(self) -> None:
        """A hanging initialize() should raise MCPClientError, not block forever."""
        cfg = _make_stdio_config(timeout=0.05)
        conn = MCPServerConnection(cfg)

        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_session.initialize = _hanging_coroutine()

        with (
            patch.object(MCPServerConnection, "_create_streams", return_value=_mock_transport()),
            patch("exo.mcp.client.ClientSession", return_value=mock_session),
            pytest.raises(MCPClientError) as exc_info,
        ):
            await conn.connect()

        err = exc_info.value
        # Must name the server
        assert "test-server" in str(err)
        # Must mention the timeout duration
        assert "0.05" in str(err) or "timed out" in str(err).lower()
        # Must have an actionable hint
        assert err.hint is not None
        assert len(err.hint) > 10
        # Should not leave the connection in a connected state
        assert not conn.is_connected

    async def test_handshake_timeout_hint_names_the_server(self) -> None:
        """The hint should reference the server name so devs know which server failed."""
        cfg = _make_stdio_config(name="my-broken-server", timeout=0.05)
        conn = MCPServerConnection(cfg)

        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_session.initialize = _hanging_coroutine()

        with (
            patch.object(MCPServerConnection, "_create_streams", return_value=_mock_transport()),
            patch("exo.mcp.client.ClientSession", return_value=mock_session),
            pytest.raises(MCPClientError) as exc_info,
        ):
            await conn.connect()

        err = exc_info.value
        # Context must carry the server name
        assert err.context is not None
        assert err.context.get("server") == "my-broken-server"

    async def test_handshake_timeout_cleans_up_connection(self) -> None:
        """After a handshake timeout the connection must not be in a connected state."""
        cfg = _make_stdio_config(timeout=0.05)
        conn = MCPServerConnection(cfg)

        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_session.initialize = _hanging_coroutine()

        with (
            patch.object(MCPServerConnection, "_create_streams", return_value=_mock_transport()),
            patch("exo.mcp.client.ClientSession", return_value=mock_session),
            pytest.raises(MCPClientError),
        ):
            await conn.connect()

        assert conn.is_connected is False
        assert conn.session is None


# ---------------------------------------------------------------------------
# Timeout: list_tools
# ---------------------------------------------------------------------------


class TestListToolsTimeout:
    """session.list_tools() must time out and raise an actionable MCPClientError."""

    async def _connected_conn(self) -> MCPServerConnection:
        """Helper: return a connected MCPServerConnection."""
        cfg = _make_stdio_config(timeout=30.0)
        conn = MCPServerConnection(cfg)

        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        init_result = MagicMock()
        mock_session.initialize = AsyncMock(return_value=init_result)
        # list_tools will be replaced per test
        mock_session.list_tools = AsyncMock()

        with (
            patch.object(MCPServerConnection, "_create_streams", return_value=_mock_transport()),
            patch("exo.mcp.client.ClientSession", return_value=mock_session),
        ):
            await conn.connect()

        # Inject the hanging list_tools after connection
        conn._session = mock_session  # type: ignore[attr-defined]
        return conn

    async def test_list_tools_timeout_raises_actionable_error(self) -> None:
        """A hanging list_tools() should raise MCPClientError with hint, not block forever."""
        cfg = MCPServerConfig(
            name="slow-lister",
            transport=MCPTransport.STDIO,
            command="python",
            timeout=30.0,
            session_timeout=0.05,
        )
        conn = MCPServerConnection(cfg)

        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_session.initialize = AsyncMock(return_value=MagicMock())
        mock_session.list_tools.side_effect = _hanging_coroutine()

        with (
            patch.object(MCPServerConnection, "_create_streams", return_value=_mock_transport()),
            patch("exo.mcp.client.ClientSession", return_value=mock_session),
        ):
            await conn.connect()
            with pytest.raises(MCPClientError) as exc_info:
                await conn.list_tools()

        err = exc_info.value
        assert "slow-lister" in str(err)
        assert "list_tools" in str(err).lower() or "timed out" in str(err).lower()
        assert err.hint is not None
        assert err.context is not None
        assert err.context.get("server") == "slow-lister"


# ---------------------------------------------------------------------------
# Timeout: call_tool
# ---------------------------------------------------------------------------


class TestCallToolTimeout:
    """session.call_tool() must time out and raise an actionable MCPClientError."""

    async def test_call_tool_timeout_raises_actionable_error(self) -> None:
        """A hanging call_tool() should raise MCPClientError, not block forever."""
        cfg = MCPServerConfig(
            name="slow-tool-server",
            transport=MCPTransport.STDIO,
            command="python",
            timeout=30.0,
            session_timeout=0.05,
        )
        conn = MCPServerConnection(cfg)

        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_session.initialize = AsyncMock(return_value=MagicMock())
        mock_session.call_tool.side_effect = _hanging_coroutine()

        with (
            patch.object(MCPServerConnection, "_create_streams", return_value=_mock_transport()),
            patch("exo.mcp.client.ClientSession", return_value=mock_session),
        ):
            await conn.connect()
            with pytest.raises(MCPClientError) as exc_info:
                await conn.call_tool("my_tool", {"x": 1})

        err = exc_info.value
        assert "slow-tool-server" in str(err)
        assert err.hint is not None
        assert err.context is not None
        assert err.context.get("server") == "slow-tool-server"
        assert err.context.get("tool") == "my_tool"

    async def test_call_tool_timeout_names_the_tool(self) -> None:
        """The error message must include the specific tool name that timed out."""
        cfg = MCPServerConfig(
            name="srv",
            transport=MCPTransport.STDIO,
            command="python",
            timeout=30.0,
            session_timeout=0.05,
        )
        conn = MCPServerConnection(cfg)

        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_session.initialize = AsyncMock(return_value=MagicMock())
        mock_session.call_tool.side_effect = _hanging_coroutine()

        with (
            patch.object(MCPServerConnection, "_create_streams", return_value=_mock_transport()),
            patch("exo.mcp.client.ClientSession", return_value=mock_session),
        ):
            await conn.connect()
            with pytest.raises(MCPClientError) as exc_info:
                await conn.call_tool("search_files", {"query": "test"})

        err = exc_info.value
        assert "search_files" in str(err)


# ---------------------------------------------------------------------------
# connect_all() partial failure aggregation
# ---------------------------------------------------------------------------


class TestConnectAllPartialFailure:
    """connect_all() must: connect good servers, report bad ones, not abort everything."""

    async def test_one_failing_server_does_not_prevent_other_connections(self) -> None:
        """If server 'bad' fails, server 'good' must still be connected."""
        client = MCPClient()
        client.add_server(_make_stdio_config("good"))
        client.add_server(_make_stdio_config("bad"))

        good_session = AsyncMock()
        good_session.__aenter__ = AsyncMock(return_value=good_session)
        good_session.__aexit__ = AsyncMock(return_value=None)
        good_session.initialize = AsyncMock(return_value=MagicMock())
        list_result = MagicMock()
        list_result.tools = []
        good_session.list_tools = AsyncMock(return_value=list_result)

        call_count = 0

        def _session_factory(*args: Any, **kwargs: Any) -> AsyncMock:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return good_session
            # second call → raise an error to simulate a bad server
            raise ConnectionError("refused")

        @asynccontextmanager
        async def _transport_factory():
            yield (MagicMock(), MagicMock())

        with (
            patch.object(
                MCPServerConnection,
                "_create_streams",
                side_effect=lambda *a, **kw: _transport_factory(),
            ),
            patch("exo.mcp.client.ClientSession", side_effect=_session_factory),
            pytest.raises(MCPClientError) as exc_info,
        ):
            await client.connect_all()

        err = exc_info.value
        # The error should mention the failure
        assert err.hint is not None
        # The good server should still be connected
        good_conn = client.get_connection("good")
        assert good_conn is not None and good_conn.is_connected

    async def test_connect_all_raises_mcp_client_error_on_failure(self) -> None:
        """connect_all() must raise MCPClientError (not a generic exception) on failure."""
        client = MCPClient()
        client.add_server(_make_stdio_config("bad-server"))

        with (
            patch.object(
                MCPServerConnection,
                "_create_streams",
                side_effect=OSError("connection refused"),
            ),
            pytest.raises(MCPClientError) as exc_info,
        ):
            await client.connect_all()

        err = exc_info.value
        # Must name which server failed
        assert "bad-server" in str(err)
        assert err.hint is not None

    async def test_connect_all_multi_failure_summary(self) -> None:
        """When multiple servers fail, the error summarizes all failures."""
        client = MCPClient()
        client.add_server(_make_stdio_config("srv-a"))
        client.add_server(_make_stdio_config("srv-b"))

        with (
            patch.object(
                MCPServerConnection,
                "_create_streams",
                side_effect=OSError("connection refused"),
            ),
            pytest.raises(MCPClientError) as exc_info,
        ):
            await client.connect_all()

        err = exc_info.value
        # Must mention both server names
        assert "srv-a" in str(err) or "srv-a" in (err.hint or "")
        assert "srv-b" in str(err) or "srv-b" in (err.hint or "")

    async def test_connect_all_succeeds_when_all_servers_ok(self) -> None:
        """connect_all() must not raise when all servers connect successfully."""
        client = MCPClient()
        client.add_server(_make_stdio_config("s1"))
        client.add_server(_make_stdio_config("s2"))

        good_session = AsyncMock()
        good_session.__aenter__ = AsyncMock(return_value=good_session)
        good_session.__aexit__ = AsyncMock(return_value=None)
        good_session.initialize = AsyncMock(return_value=MagicMock())

        with (
            patch.object(
                MCPServerConnection,
                "_create_streams",
                side_effect=lambda *a, **kw: _mock_transport(),
            ),
            patch("exo.mcp.client.ClientSession", return_value=good_session),
        ):
            await client.connect_all()  # must not raise

        assert client.get_connection("s1") is not None
        assert client.get_connection("s2") is not None

    async def test_connect_all_empty_client_is_noop(self) -> None:
        """connect_all() with no registered servers must succeed silently."""
        client = MCPClient()
        await client.connect_all()  # no exception


# ---------------------------------------------------------------------------
# Unknown transport in mcp.json (execution.py P0 #1)
# ---------------------------------------------------------------------------


class TestUnknownTransportInConfig:
    """An unknown 'transport' value in mcp.json must raise MCPExecutionError with a hint."""

    def test_unknown_transport_raises_execution_error(self, tmp_path: Path) -> None:
        """'grpc' transport in mcp.json should raise MCPExecutionError, not ValueError."""
        config = {
            "mcpServers": {
                "my-server": {
                    "transport": "grpc",
                    "command": "python",
                    "args": ["-m", "my_server"],
                }
            }
        }
        config_path = tmp_path / "mcp.json"
        config_path.write_text(json.dumps(config))

        with pytest.raises(MCPExecutionError) as exc_info:
            load_mcp_config(config_path)

        err = exc_info.value
        # Must name the invalid transport value
        assert "grpc" in str(err)
        # Must have an actionable hint with valid values
        assert err.hint is not None
        assert "stdio" in err.hint
        # Must carry context
        assert err.context is not None
        assert err.context.get("transport") == "grpc"

    def test_unknown_transport_hint_lists_valid_options(self, tmp_path: Path) -> None:
        """The hint must enumerate the valid transport choices."""
        config = {
            "mcpServers": {
                "x": {"transport": "websocket", "command": "node", "args": ["server.js"]}
            }
        }
        config_path = tmp_path / "mcp.json"
        config_path.write_text(json.dumps(config))

        with pytest.raises(MCPExecutionError) as exc_info:
            load_mcp_config(config_path)

        hint = exc_info.value.hint or ""
        # All three valid transports should appear in the hint
        assert "stdio" in hint
        assert "sse" in hint
        assert "streamable_http" in hint

    def test_valid_transports_still_load(self, tmp_path: Path) -> None:
        """Valid transport values must not raise."""
        for transport in ("stdio", "sse", "streamable_http"):
            if transport == "stdio":
                server_cfg = {"transport": transport, "command": "python"}
            else:
                server_cfg = {"transport": transport, "url": "http://localhost/mcp"}
            config = {"mcpServers": {"srv": server_cfg}}
            config_path = tmp_path / f"mcp_{transport}.json"
            config_path.write_text(json.dumps(config))
            cfgs = load_mcp_config(config_path)
            assert len(cfgs) == 1
            assert cfgs[0].transport.value == transport


# ---------------------------------------------------------------------------
# Actionable error quality checks
# ---------------------------------------------------------------------------


class TestErrorQuality:
    """Spot checks that errors produced by key code paths are legible + actionable."""

    async def test_connect_unknown_server_is_actionable(self) -> None:
        """Connecting to an unregistered server must raise MCPClientError (not KeyError)."""
        client = MCPClient()
        client.add_server(_make_stdio_config("registered"))

        with pytest.raises(MCPClientError, match="No server registered"):
            await client.connect("unknown")

    async def test_mcp_client_error_is_exo_error_subclass(self) -> None:
        """MCPClientError must descend from ExoError so framework catches it uniformly."""
        from exo.types import ExoError  # pyright: ignore[reportMissingImports]

        assert issubclass(MCPClientError, ExoError)

    def test_mcp_execution_error_carries_hint(self, tmp_path: Path) -> None:
        """MCPExecutionError for a missing file must include a hint."""
        missing = tmp_path / "nonexistent.json"

        with pytest.raises(MCPExecutionError) as exc_info:
            load_mcp_config(missing)

        err = exc_info.value
        assert err.hint is not None
        assert len(err.hint) > 5
