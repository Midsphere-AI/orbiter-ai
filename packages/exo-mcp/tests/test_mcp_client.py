"""Tests for MCP client — server connection."""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import timedelta
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

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_session(
    tools: list[Any] | None = None,
    call_result: Any = None,
) -> AsyncMock:
    """Create a mock MCP ClientSession."""
    session = AsyncMock()
    session.initialize = AsyncMock(return_value=MagicMock(serverInfo={"name": "test"}))
    tool_list = MagicMock()
    tool_list.tools = tools or []
    session.list_tools = AsyncMock(return_value=tool_list)
    session.call_tool = AsyncMock(return_value=call_result or MagicMock(content=[]))
    # Make session work as async context manager
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=None)
    return session


@asynccontextmanager
async def _mock_transport():
    """Mock transport that yields (read_stream, write_stream)."""
    read = MagicMock()
    write = MagicMock()
    yield (read, write)


@asynccontextmanager
async def _mock_transport_3tuple():
    """Mock transport that yields (read, write, get_session_id) for streamable-http."""
    read = MagicMock()
    write = MagicMock()
    get_session_id = MagicMock(return_value="session-123")
    yield (read, write, get_session_id)


def _make_stdio_config(name: str = "test-stdio", **kw: Any) -> MCPServerConfig:
    return MCPServerConfig(
        name=name,
        transport=MCPTransport.STDIO,
        command="python",
        args=["-m", "test_server"],
        **kw,
    )


def _make_sse_config(name: str = "test-sse", **kw: Any) -> MCPServerConfig:
    return MCPServerConfig(
        name=name,
        transport=MCPTransport.SSE,
        url="http://localhost:8080/sse",
        **kw,
    )


def _make_http_config(name: str = "test-http", **kw: Any) -> MCPServerConfig:
    return MCPServerConfig(
        name=name,
        transport=MCPTransport.STREAMABLE_HTTP,
        url="http://localhost:8080/mcp",
        **kw,
    )


# ===========================================================================
# MCPTransport
# ===========================================================================


class TestMCPTransport:
    def test_values(self) -> None:
        assert MCPTransport.STDIO == "stdio"
        assert MCPTransport.SSE == "sse"
        assert MCPTransport.STREAMABLE_HTTP == "streamable_http"

    def test_from_string(self) -> None:
        assert MCPTransport("stdio") == MCPTransport.STDIO
        assert MCPTransport("sse") == MCPTransport.SSE
        assert MCPTransport("streamable_http") == MCPTransport.STREAMABLE_HTTP


# ===========================================================================
# MCPServerConfig
# ===========================================================================


class TestMCPServerConfig:
    def test_stdio_config(self) -> None:
        cfg = _make_stdio_config()
        assert cfg.name == "test-stdio"
        assert cfg.transport == MCPTransport.STDIO
        assert cfg.command == "python"
        assert cfg.args == ["-m", "test_server"]
        cfg.validate()

    def test_sse_config(self) -> None:
        cfg = _make_sse_config()
        assert cfg.transport == MCPTransport.SSE
        assert cfg.url == "http://localhost:8080/sse"
        cfg.validate()

    def test_http_config(self) -> None:
        cfg = _make_http_config()
        assert cfg.transport == MCPTransport.STREAMABLE_HTTP
        assert cfg.url == "http://localhost:8080/mcp"
        cfg.validate()

    def test_stdio_missing_command(self) -> None:
        cfg = MCPServerConfig(name="bad", transport=MCPTransport.STDIO)
        with pytest.raises(MCPClientError, match="requires 'command'"):
            cfg.validate()

    def test_sse_missing_url(self) -> None:
        cfg = MCPServerConfig(name="bad", transport=MCPTransport.SSE)
        with pytest.raises(MCPClientError, match="requires 'url'"):
            cfg.validate()

    def test_http_missing_url(self) -> None:
        cfg = MCPServerConfig(name="bad", transport=MCPTransport.STREAMABLE_HTTP)
        with pytest.raises(MCPClientError, match="requires 'url'"):
            cfg.validate()

    def test_defaults(self) -> None:
        cfg = MCPServerConfig(name="x", transport="stdio", command="node")
        assert cfg.args == []
        assert cfg.env is None
        assert cfg.cwd is None
        assert cfg.headers is None
        assert cfg.timeout == 30.0
        assert cfg.sse_read_timeout == 300.0
        assert cfg.cache_tools is False
        assert cfg.session_timeout == 120.0

    def test_custom_values(self) -> None:
        cfg = MCPServerConfig(
            name="x",
            transport="sse",
            url="http://x",
            headers={"X-Token": "abc"},
            timeout=10,
            sse_read_timeout=60,
            cache_tools=True,
            session_timeout=30,
        )
        assert cfg.headers == {"X-Token": "abc"}
        assert cfg.timeout == 10
        assert cfg.cache_tools is True

    def test_repr(self) -> None:
        cfg = _make_stdio_config()
        assert "test-stdio" in repr(cfg)
        assert "stdio" in repr(cfg)


# ===========================================================================
# MCPServerConnection — unit tests with mocked transport
# ===========================================================================


class TestMCPServerConnectionInit:
    def test_creation(self) -> None:
        cfg = _make_stdio_config()
        conn = MCPServerConnection(cfg)
        assert conn.name == "test-stdio"
        assert conn.config is cfg
        assert conn.session is None
        assert conn.init_result is None
        assert conn.is_connected is False

    def test_repr_disconnected(self) -> None:
        conn = MCPServerConnection(_make_stdio_config())
        assert "disconnected" in repr(conn)


class TestMCPServerConnectionConnect:
    async def test_connect_success(self) -> None:
        cfg = _make_stdio_config()
        conn = MCPServerConnection(cfg)
        session = _mock_session()

        with (
            patch.object(MCPServerConnection, "_create_streams", return_value=_mock_transport()),
            patch("exo.mcp.client.ClientSession", return_value=session),
        ):
            await conn.connect()

        assert conn.is_connected is True
        assert conn.init_result is not None
        assert "connected" in repr(conn)

    async def test_connect_idempotent(self) -> None:
        cfg = _make_stdio_config()
        conn = MCPServerConnection(cfg)
        session = _mock_session()

        with (
            patch.object(MCPServerConnection, "_create_streams", return_value=_mock_transport()),
            patch("exo.mcp.client.ClientSession", return_value=session),
        ):
            await conn.connect()
            await conn.connect()  # second call is no-op

        assert conn.is_connected

    async def test_connect_failure_raises(self) -> None:
        cfg = _make_stdio_config()
        conn = MCPServerConnection(cfg)

        with (
            patch.object(
                MCPServerConnection,
                "_create_streams",
                side_effect=ConnectionError("refused"),
            ),
            pytest.raises(MCPClientError, match="Failed to connect"),
        ):
            await conn.connect()

        assert conn.is_connected is False

    async def test_connect_validates_config(self) -> None:
        cfg = MCPServerConfig(name="bad", transport=MCPTransport.STDIO)
        conn = MCPServerConnection(cfg)
        with pytest.raises(MCPClientError, match="requires 'command'"):
            await conn.connect()


class TestMCPServerConnectionListTools:
    async def test_list_tools(self) -> None:
        cfg = _make_stdio_config()
        conn = MCPServerConnection(cfg)
        mock_tool = MagicMock(name="test_tool")
        session = _mock_session(tools=[mock_tool])

        with (
            patch.object(MCPServerConnection, "_create_streams", return_value=_mock_transport()),
            patch("exo.mcp.client.ClientSession", return_value=session),
        ):
            await conn.connect()
            tools = await conn.list_tools()

        assert len(tools) == 1
        assert tools[0] is mock_tool

    async def test_list_tools_not_connected(self) -> None:
        conn = MCPServerConnection(_make_stdio_config())
        with pytest.raises(MCPClientError, match="not connected"):
            await conn.list_tools()

    async def test_list_tools_cached(self) -> None:
        cfg = _make_stdio_config(cache_tools=True)
        conn = MCPServerConnection(cfg)
        mock_tool = MagicMock(name="cached_tool")
        session = _mock_session(tools=[mock_tool])

        with (
            patch.object(MCPServerConnection, "_create_streams", return_value=_mock_transport()),
            patch("exo.mcp.client.ClientSession", return_value=session),
        ):
            await conn.connect()
            tools1 = await conn.list_tools()
            tools2 = await conn.list_tools()

        assert tools1 == tools2
        # list_tools called once on server (cache hit on second call)
        assert session.list_tools.call_count == 1

    async def test_invalidate_cache(self) -> None:
        cfg = _make_stdio_config(cache_tools=True)
        conn = MCPServerConnection(cfg)
        session = _mock_session(tools=[MagicMock()])

        with (
            patch.object(MCPServerConnection, "_create_streams", return_value=_mock_transport()),
            patch("exo.mcp.client.ClientSession", return_value=session),
        ):
            await conn.connect()
            await conn.list_tools()
            conn.invalidate_tools_cache()
            await conn.list_tools()

        # After invalidation, re-fetches
        assert session.list_tools.call_count == 2


class TestMCPServerConnectionCallTool:
    async def test_call_tool(self) -> None:
        cfg = _make_stdio_config()
        conn = MCPServerConnection(cfg)
        expected = MagicMock(content=[MagicMock(text="hello")])
        session = _mock_session(call_result=expected)

        with (
            patch.object(MCPServerConnection, "_create_streams", return_value=_mock_transport()),
            patch("exo.mcp.client.ClientSession", return_value=session),
        ):
            await conn.connect()
            result = await conn.call_tool("greet", {"name": "world"})

        assert result is expected
        session.call_tool.assert_called_once_with(
            name="greet", arguments={"name": "world"}, progress_callback=None
        )

    async def test_call_tool_not_connected(self) -> None:
        conn = MCPServerConnection(_make_stdio_config())
        with pytest.raises(MCPClientError, match="not connected"):
            await conn.call_tool("x")


class TestMCPServerConnectionCleanup:
    async def test_cleanup(self) -> None:
        cfg = _make_stdio_config()
        conn = MCPServerConnection(cfg)
        session = _mock_session()

        with (
            patch.object(MCPServerConnection, "_create_streams", return_value=_mock_transport()),
            patch("exo.mcp.client.ClientSession", return_value=session),
        ):
            await conn.connect()
            assert conn.is_connected
            await conn.cleanup()

        assert conn.is_connected is False

    async def test_async_context_manager(self) -> None:
        cfg = _make_stdio_config()
        conn = MCPServerConnection(cfg)
        session = _mock_session()

        with (
            patch.object(MCPServerConnection, "_create_streams", return_value=_mock_transport()),
            patch("exo.mcp.client.ClientSession", return_value=session),
        ):
            async with conn:
                assert conn.is_connected
            assert conn.is_connected is False


class TestMCPServerConnectionTransports:
    def test_create_streams_stdio(self) -> None:
        cfg = _make_stdio_config()
        conn = MCPServerConnection(cfg)
        with patch("exo.mcp.transport.stdio_client") as mock_stdio:
            mock_stdio.return_value = _mock_transport()
            conn._create_streams()
            mock_stdio.assert_called_once()
            params = mock_stdio.call_args[0][0]
            assert params.command == "python"

    def test_create_streams_sse(self) -> None:
        cfg = _make_sse_config()
        conn = MCPServerConnection(cfg)
        with patch("exo.mcp.transport.sse_client") as mock_sse:
            mock_sse.return_value = _mock_transport()
            conn._create_streams()
            mock_sse.assert_called_once_with(
                url="http://localhost:8080/sse",
                headers=None,
                timeout=30.0,
                sse_read_timeout=300.0,
            )

    def test_create_streams_streamable_http(self) -> None:
        cfg = _make_http_config()
        conn = MCPServerConnection(cfg)
        with patch("exo.mcp.transport.streamablehttp_client") as mock_http:
            mock_http.return_value = _mock_transport_3tuple()
            conn._create_streams()
            mock_http.assert_called_once_with(
                url="http://localhost:8080/mcp",
                headers={},
                timeout=timedelta(seconds=30.0),
                sse_read_timeout=timedelta(seconds=300.0),
                terminate_on_close=True,
            )


# ===========================================================================
# MCPClient — high-level multi-server manager
# ===========================================================================


class TestMCPClientInit:
    def test_creation(self) -> None:
        client = MCPClient()
        assert client.server_names == []
        assert repr(client) == "MCPClient(servers=0, connected=0)"

    def test_add_server(self) -> None:
        client = MCPClient()
        result = client.add_server(_make_stdio_config("a"))
        assert result is client  # chaining
        assert client.server_names == ["a"]

    def test_add_multiple_servers(self) -> None:
        client = MCPClient()
        client.add_server(_make_stdio_config("a"))
        client.add_server(_make_sse_config("b"))
        assert set(client.server_names) == {"a", "b"}

    def test_remove_server(self) -> None:
        client = MCPClient()
        client.add_server(_make_stdio_config("a"))
        client.remove_server("a")
        assert client.server_names == []

    def test_remove_nonexistent(self) -> None:
        client = MCPClient()
        client.remove_server("nope")  # no error

    def test_add_validates(self) -> None:
        client = MCPClient()
        with pytest.raises(MCPClientError):
            client.add_server(MCPServerConfig(name="bad", transport="stdio"))


class TestMCPClientConnect:
    async def test_connect_single(self) -> None:
        client = MCPClient()
        client.add_server(_make_stdio_config("s1"))
        session = _mock_session()

        with (
            patch(
                "exo.mcp.client.MCPServerConnection._create_streams",
                return_value=_mock_transport(),
            ),
            patch("exo.mcp.client.ClientSession", return_value=session),
        ):
            conn = await client.connect("s1")
            assert conn.is_connected
            assert client.get_connection("s1") is conn

    async def test_connect_unknown_server(self) -> None:
        client = MCPClient()
        with pytest.raises(MCPClientError, match="No server registered"):
            await client.connect("unknown")

    async def test_connect_reuses_cached(self) -> None:
        client = MCPClient()
        client.add_server(_make_stdio_config("s1"))
        session = _mock_session()

        with (
            patch(
                "exo.mcp.client.MCPServerConnection._create_streams",
                return_value=_mock_transport(),
            ),
            patch("exo.mcp.client.ClientSession", return_value=session),
        ):
            conn1 = await client.connect("s1")
            conn2 = await client.connect("s1")
            assert conn1 is conn2

    async def test_connect_all(self) -> None:
        client = MCPClient()
        client.add_server(_make_stdio_config("s1"))
        client.add_server(_make_sse_config("s2"))
        session = _mock_session()

        with (
            patch.object(
                MCPServerConnection,
                "_create_streams",
                side_effect=lambda *a, **kw: _mock_transport(),
            ),
            patch("exo.mcp.client.ClientSession", return_value=session),
        ):
            await client.connect_all()
            assert client.get_connection("s1") is not None
            assert client.get_connection("s2") is not None


class TestMCPClientDisconnect:
    async def test_disconnect_single(self) -> None:
        client = MCPClient()
        client.add_server(_make_stdio_config("s1"))
        session = _mock_session()

        with (
            patch(
                "exo.mcp.client.MCPServerConnection._create_streams",
                return_value=_mock_transport(),
            ),
            patch("exo.mcp.client.ClientSession", return_value=session),
        ):
            await client.connect("s1")
            await client.disconnect("s1")
            assert client.get_connection("s1") is None

    async def test_disconnect_all(self) -> None:
        client = MCPClient()
        client.add_server(_make_stdio_config("s1"))
        client.add_server(_make_sse_config("s2"))
        session = _mock_session()

        with (
            patch.object(
                MCPServerConnection,
                "_create_streams",
                side_effect=lambda *a, **kw: _mock_transport(),
            ),
            patch("exo.mcp.client.ClientSession", return_value=session),
        ):
            await client.connect_all()
            await client.disconnect_all()
            assert client.get_connection("s1") is None
            assert client.get_connection("s2") is None


class TestMCPClientToolOperations:
    async def test_list_tools(self) -> None:
        client = MCPClient()
        client.add_server(_make_stdio_config("s1"))
        mock_tool = MagicMock(name="tool1")
        session = _mock_session(tools=[mock_tool])

        with (
            patch(
                "exo.mcp.client.MCPServerConnection._create_streams",
                return_value=_mock_transport(),
            ),
            patch("exo.mcp.client.ClientSession", return_value=session),
        ):
            tools = await client.list_tools("s1")
            assert len(tools) == 1

    async def test_call_tool(self) -> None:
        client = MCPClient()
        client.add_server(_make_stdio_config("s1"))
        expected = MagicMock(content=[])
        session = _mock_session(call_result=expected)

        with (
            patch(
                "exo.mcp.client.MCPServerConnection._create_streams",
                return_value=_mock_transport(),
            ),
            patch("exo.mcp.client.ClientSession", return_value=session),
        ):
            result = await client.call_tool("s1", "my_tool", {"a": 1})
            assert result is expected

    async def test_list_tools_auto_connects(self) -> None:
        client = MCPClient()
        client.add_server(_make_stdio_config("s1"))
        session = _mock_session(tools=[MagicMock()])

        with (
            patch(
                "exo.mcp.client.MCPServerConnection._create_streams",
                return_value=_mock_transport(),
            ),
            patch("exo.mcp.client.ClientSession", return_value=session),
        ):
            # list_tools without explicit connect — should auto-connect
            tools = await client.list_tools("s1")
            assert len(tools) == 1
            assert client.get_connection("s1") is not None


class TestMCPClientContextManager:
    async def test_async_context_manager(self) -> None:
        client = MCPClient()
        client.add_server(_make_stdio_config("s1"))
        session = _mock_session()

        with (
            patch(
                "exo.mcp.client.MCPServerConnection._create_streams",
                return_value=_mock_transport(),
            ),
            patch("exo.mcp.client.ClientSession", return_value=session),
        ):
            async with client:
                assert client.get_connection("s1") is not None
            # After exit, disconnected
            assert client.get_connection("s1") is None

    async def test_repr_with_connections(self) -> None:
        client = MCPClient()
        client.add_server(_make_stdio_config("s1"))
        client.add_server(_make_sse_config("s2"))
        session = _mock_session()

        with (
            patch(
                "exo.mcp.client.MCPServerConnection._create_streams",
                return_value=_mock_transport(),
            ),
            patch("exo.mcp.client.ClientSession", return_value=session),
        ):
            await client.connect("s1")
            assert "servers=2" in repr(client)
            assert "connected=1" in repr(client)
