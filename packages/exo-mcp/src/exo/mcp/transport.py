"""Shared MCP transport stream construction.

Provides :func:`create_transport_streams`, a free function that builds the
appropriate async context manager for any supported MCP transport type.
Both ``exo-mcp`` and ``exo-mcp-cli`` call this function so the 3-branch
switch lives in exactly one place.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any

from exo.types import ExoError  # pyright: ignore[reportMissingImports]
from mcp.client.sse import sse_client
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.client.streamable_http import streamablehttp_client


class MCPTransportError(ExoError):
    """Raised when an unsupported or misconfigured transport is requested."""


def create_transport_streams(
    transport: str,
    *,
    # stdio
    command: str = "",
    args: list[str] | None = None,
    env: dict[str, str] | None = None,
    cwd: str | None = None,
    encoding: str = "utf-8",
    encoding_error_handler: str = "strict",
    # sse / streamable_http
    url: str = "",
    headers: dict[str, str] | None = None,
    timeout: float = 30.0,
    sse_read_timeout: float = 300.0,
    # streamable_http-only
    terminate_on_close: bool = True,
    # websocket (exo-mcp-cli extension)
    allow_websocket: bool = False,
) -> Any:
    """Return the appropriate async context manager for *transport*.

    The returned object is an async context manager that yields
    ``(read_stream, write_stream[, ...])`` — callers unpack at least the
    first two elements.

    Args:
        transport: One of ``"stdio"``, ``"sse"``, ``"streamable_http"``,
            or ``"websocket"`` (only when *allow_websocket* is ``True``).
        command: Executable path for stdio transport.
        args: CLI arguments for stdio transport.
        env: Extra environment variables for stdio transport.
        cwd: Working directory for stdio transport.
        encoding: Character encoding for stdio transport.
        encoding_error_handler: Error handler for stdio encoding errors.
        url: Server URL for network transports.
        headers: HTTP headers for SSE / streamable-HTTP transports.
        timeout: Connection timeout in seconds.
        sse_read_timeout: Maximum time to wait for an SSE event (seconds).
        terminate_on_close: Whether to send a ``DELETE`` on close for
            streamable-HTTP sessions.
        allow_websocket: Set to ``True`` to enable WebSocket transport
            (requires ``mcp.client.websocket``; used only by exo-mcp-cli).

    Returns:
        An async context manager yielding transport streams.

    Raises:
        MCPTransportError: If *transport* is not recognised.
    """
    if transport == "stdio":
        params = StdioServerParameters(
            command=command,
            args=args or [],
            env=env,
            cwd=cwd,
            encoding=encoding,
            encoding_error_handler=encoding_error_handler,
        )
        return stdio_client(params)

    if transport == "sse":
        return sse_client(
            url=url,
            headers=headers,
            timeout=timeout,
            sse_read_timeout=sse_read_timeout,
        )

    if transport == "streamable_http":
        return streamablehttp_client(
            url=url,
            headers=headers or {},
            timeout=timedelta(seconds=timeout),
            sse_read_timeout=timedelta(seconds=sse_read_timeout),
            terminate_on_close=terminate_on_close,
        )

    if transport == "websocket" and allow_websocket:
        from mcp.client.websocket import websocket_client  # lazy import

        return websocket_client(url=url)

    raise MCPTransportError(
        f"Unsupported transport: {transport!r}",
        hint=f"Valid transports are: stdio, sse, streamable_http. Got {transport!r}.",
    )
