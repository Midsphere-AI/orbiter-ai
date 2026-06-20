"""MCP client with multiple transport support and server instance caching."""

from __future__ import annotations

import asyncio
import logging
from contextlib import AsyncExitStack
from datetime import timedelta
from enum import StrEnum
from typing import Any

from exo.types import ExoError  # pyright: ignore[reportMissingImports]
from mcp import ClientSession
from mcp.client.sse import sse_client
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.client.streamable_http import streamablehttp_client
from mcp.types import CallToolResult, InitializeResult
from mcp.types import Tool as MCPTool

logger = logging.getLogger(__name__)


class MCPClientError(ExoError):
    """Error raised by MCP client operations."""


class MCPTransport(StrEnum):
    """Transport types for MCP server connections."""

    STDIO = "stdio"
    SSE = "sse"
    STREAMABLE_HTTP = "streamable_http"


class MCPServerConfig:
    """Configuration for an MCP server connection.

    Args:
        name: Human-readable server name.
        transport: Transport type (stdio, sse, streamable_http).
        command: Executable for stdio transport.
        args: Command-line arguments for stdio transport.
        env: Environment variables for stdio transport.
        cwd: Working directory for stdio transport.
        url: Server URL for sse/streamable_http transports.
        headers: HTTP headers for sse/streamable_http transports.
        timeout: Connection timeout in seconds.
        sse_read_timeout: SSE read timeout in seconds.
        cache_tools: Whether to cache the tools list.
        session_timeout: ClientSession read timeout in seconds.
    """

    __slots__ = (
        "args",
        "cache_tools",
        "command",
        "cwd",
        "env",
        "headers",
        "large_output_tools",
        "name",
        "session_timeout",
        "sse_read_timeout",
        "timeout",
        "transport",
        "url",
    )

    def __init__(
        self,
        name: str,
        transport: MCPTransport | str = MCPTransport.STDIO,
        *,
        command: str | None = None,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
        url: str | None = None,
        headers: dict[str, str] | None = None,
        timeout: float = 30.0,
        sse_read_timeout: float = 300.0,
        cache_tools: bool = False,
        session_timeout: float | None = 120.0,
        large_output_tools: list[str] | None = None,
    ) -> None:
        self.name = name
        self.transport = MCPTransport(transport)
        self.command = command
        self.args = args or []
        self.env = env
        self.cwd = cwd
        self.url = url
        self.headers = headers
        self.timeout = timeout
        self.sse_read_timeout = sse_read_timeout
        self.cache_tools = cache_tools
        self.session_timeout = session_timeout
        self.large_output_tools: list[str] = list(large_output_tools) if large_output_tools else []

    def validate(self) -> None:
        """Validate config fields for the chosen transport."""
        if self.transport == MCPTransport.STDIO:
            if not self.command:
                raise MCPClientError(f"Server '{self.name}': stdio transport requires 'command'")
        elif self.transport in (MCPTransport.SSE, MCPTransport.STREAMABLE_HTTP) and not self.url:
            raise MCPClientError(f"Server '{self.name}': {self.transport} transport requires 'url'")

    def to_dict(self) -> dict[str, Any]:
        """Serialize all config fields to a plain dict.

        Returns:
            A JSON-serializable dict suitable for reconstruction via ``from_dict()``.
        """
        return {
            "name": self.name,
            "transport": str(self.transport),
            "command": self.command,
            "args": self.args,
            "env": self.env,
            "cwd": self.cwd,
            "url": self.url,
            "headers": self.headers,
            "timeout": self.timeout,
            "sse_read_timeout": self.sse_read_timeout,
            "cache_tools": self.cache_tools,
            "session_timeout": self.session_timeout,
            "large_output_tools": self.large_output_tools,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MCPServerConfig:
        """Reconstruct an MCPServerConfig from a dict produced by ``to_dict()``.

        Args:
            data: Dict as produced by ``to_dict()``.

        Returns:
            A reconstructed ``MCPServerConfig`` instance.
        """
        return cls(
            name=data["name"],
            transport=data.get("transport", MCPTransport.STDIO),
            command=data.get("command"),
            args=data.get("args"),
            env=data.get("env"),
            cwd=data.get("cwd"),
            url=data.get("url"),
            headers=data.get("headers"),
            timeout=data.get("timeout", 30.0),
            sse_read_timeout=data.get("sse_read_timeout", 300.0),
            cache_tools=data.get("cache_tools", False),
            session_timeout=data.get("session_timeout", 120.0),
            large_output_tools=data.get("large_output_tools"),
        )

    def __repr__(self) -> str:
        return f"MCPServerConfig(name={self.name!r}, transport={self.transport!r})"


class MCPServerConnection:
    """A live connection to an MCP server.

    Manages the async context stack (transport + session) and provides
    list_tools() and call_tool() methods.
    """

    __slots__ = (
        "_cache_dirty",
        "_cleanup_lock",
        "_config",
        "_exit_stack",
        "_init_result",
        "_session",
        "_tools_cache",
    )

    def __init__(self, config: MCPServerConfig) -> None:
        self._config = config
        self._exit_stack = AsyncExitStack()
        self._session: ClientSession | None = None
        self._init_result: InitializeResult | None = None
        self._cleanup_lock = asyncio.Lock()
        self._cache_dirty = True
        self._tools_cache: list[MCPTool] | None = None

    @property
    def name(self) -> str:
        return self._config.name

    @property
    def config(self) -> MCPServerConfig:
        return self._config

    @property
    def session(self) -> ClientSession | None:
        return self._session

    @property
    def init_result(self) -> InitializeResult | None:
        return self._init_result

    @property
    def is_connected(self) -> bool:
        return self._session is not None

    async def connect(self) -> None:
        """Open transport and initialize the MCP session."""
        if self._session is not None:
            logger.debug("Reusing existing connection to MCP server '%s'", self.name)
            return  # already connected

        self._config.validate()
        try:
            transport = await self._exit_stack.enter_async_context(self._create_streams())
            read, write, *_ = transport
            session_timeout = (
                timedelta(seconds=self._config.session_timeout)
                if self._config.session_timeout
                else None
            )
            session = await self._exit_stack.enter_async_context(
                ClientSession(read, write, session_timeout)
            )
            self._init_result = await session.initialize()
            self._session = session
            logger.debug(
                "Connected to MCP server '%s' (transport=%s)", self.name, self._config.transport
            )
        except Exception as exc:
            logger.error("Error connecting to MCP server %r: %s", self.name, exc)
            await self.cleanup()
            raise MCPClientError(f"Failed to connect to server '{self.name}': {exc}") from exc

    def _create_streams(self) -> Any:
        """Create the appropriate transport streams based on config."""
        cfg = self._config
        if cfg.transport == MCPTransport.STDIO:
            params = StdioServerParameters(
                command=cfg.command or "",
                args=cfg.args,
                env=cfg.env,
                cwd=cfg.cwd,
                encoding="utf-8",
                encoding_error_handler="strict",
            )
            return stdio_client(params)
        if cfg.transport == MCPTransport.SSE:
            return sse_client(
                url=cfg.url or "",
                headers=cfg.headers,
                timeout=cfg.timeout,
                sse_read_timeout=cfg.sse_read_timeout,
            )
        # STREAMABLE_HTTP
        return streamablehttp_client(
            url=cfg.url or "",
            headers=cfg.headers or {},
            timeout=timedelta(seconds=cfg.timeout),
            sse_read_timeout=timedelta(seconds=cfg.sse_read_timeout),
            terminate_on_close=True,
        )

    async def list_tools(self) -> list[MCPTool]:
        """List available tools from the server."""
        if not self._session:
            raise MCPClientError(f"Server '{self.name}' not connected. Call connect() first.")
        if self._config.cache_tools and not self._cache_dirty and self._tools_cache:
            logger.debug(
                "Tools cache hit for server '%s' (%d tools)", self.name, len(self._tools_cache)
            )
            return self._tools_cache
        self._cache_dirty = False
        self._tools_cache = (await self._session.list_tools()).tools
        logger.debug("Listed %d tools from server '%s'", len(self._tools_cache), self.name)
        return self._tools_cache

    def invalidate_tools_cache(self) -> None:
        """Mark the tools cache as dirty so the next list_tools() re-fetches."""
        self._cache_dirty = True

    async def call_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any] | None = None,
        *,
        progress_callback: Any | None = None,
    ) -> CallToolResult:
        """Invoke a tool on the server.

        Args:
            tool_name: Name of the tool to invoke.
            arguments: Tool arguments to pass.
            progress_callback: Optional async callable invoked on each progress
                notification from the server: ``(progress, total, message) -> None``.
        """
        if not self._session:
            raise MCPClientError(f"Server '{self.name}' not connected. Call connect() first.")
        logger.debug("Calling tool '%s' on server '%s'", tool_name, self.name)
        return await self._session.call_tool(
            name=tool_name, arguments=arguments, progress_callback=progress_callback
        )

    async def cleanup(self) -> None:
        """Close the transport and session."""
        async with self._cleanup_lock:
            self._session = None
            try:
                await self._exit_stack.aclose()
            except Exception as exc:
                logger.debug("Error closing MCP server %r: %s", self.name, exc)
            self._exit_stack = AsyncExitStack()

    async def __aenter__(self) -> MCPServerConnection:
        await self.connect()
        return self

    async def __aexit__(
        self, exc_type: type | None, exc_val: BaseException | None, exc_tb: Any
    ) -> None:
        await self.cleanup()

    def __repr__(self) -> str:
        status = "connected" if self.is_connected else "disconnected"
        return f"MCPServerConnection(name={self.name!r}, status={status})"


class MCPClient:
    """High-level MCP client managing multiple server connections.

    Provides server instance caching/reuse with session isolation.
    Servers are identified by name and cached after first connection.

    Usage::

        client = MCPClient()
        client.add_server(MCPServerConfig(
            name="my-server",
            transport="stdio",
            command="python",
            args=["-m", "my_mcp_server"],
        ))
        async with client:
            tools = await client.list_tools("my-server")
            result = await client.call_tool("my-server", "tool_name", {"arg": "val"})
    """

    __slots__ = ("_configs", "_connections")

    def __init__(self) -> None:
        self._configs: dict[str, MCPServerConfig] = {}
        self._connections: dict[str, MCPServerConnection] = {}

    def add_server(self, config: MCPServerConfig) -> MCPClient:
        """Register a server configuration. Returns self for chaining."""
        config.validate()
        self._configs[config.name] = config
        return self

    def remove_server(self, name: str) -> None:
        """Remove a server configuration (does not disconnect)."""
        self._configs.pop(name, None)

    @property
    def server_names(self) -> list[str]:
        """Names of all registered servers."""
        return list(self._configs)

    def get_connection(self, name: str) -> MCPServerConnection | None:
        """Get a cached connection by server name, or None."""
        return self._connections.get(name)

    async def connect(self, name: str) -> MCPServerConnection:
        """Connect to a specific server. Re-uses cached connection if alive."""
        if name in self._connections and self._connections[name].is_connected:
            logger.debug("Reusing connection to server '%s'", name)
            return self._connections[name]
        config = self._configs.get(name)
        if not config:
            raise MCPClientError(f"No server registered with name '{name}'")
        conn = MCPServerConnection(config)
        await conn.connect()
        self._connections[name] = conn
        logger.debug("Established new connection to server '%s'", name)
        return conn

    async def connect_all(self) -> None:
        """Connect to all registered servers."""
        for name in self._configs:
            await self.connect(name)

    async def disconnect(self, name: str) -> None:
        """Disconnect a specific server."""
        conn = self._connections.pop(name, None)
        if conn:
            logger.debug("Disconnecting from server '%s'", name)
            await conn.cleanup()

    async def disconnect_all(self) -> None:
        """Disconnect all servers."""
        for name in list(self._connections):
            await self.disconnect(name)

    async def list_tools(self, name: str) -> list[MCPTool]:
        """List tools from a specific server (connects if needed)."""
        conn = await self.connect(name)
        return await conn.list_tools()

    async def call_tool(
        self,
        server_name: str,
        tool_name: str,
        arguments: dict[str, Any] | None = None,
        *,
        progress_callback: Any | None = None,
    ) -> CallToolResult:
        """Call a tool on a specific server (connects if needed)."""
        conn = await self.connect(server_name)
        return await conn.call_tool(tool_name, arguments, progress_callback=progress_callback)

    async def __aenter__(self) -> MCPClient:
        await self.connect_all()
        return self

    async def __aexit__(
        self, exc_type: type | None, exc_val: BaseException | None, exc_tb: Any
    ) -> None:
        await self.disconnect_all()

    def __repr__(self) -> str:
        n_configs = len(self._configs)
        n_connected = sum(1 for c in self._connections.values() if c.is_connected)
        return f"MCPClient(servers={n_configs}, connected={n_connected})"
