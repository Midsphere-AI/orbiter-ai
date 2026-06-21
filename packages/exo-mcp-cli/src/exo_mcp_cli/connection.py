"""Lightweight async connection manager for MCP servers.

Provides an async context manager that yields a connected
``ClientSession`` for any supported transport.
"""

from __future__ import annotations

import re
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from typing import Any

from mcp import ClientSession

from exo.mcp.transport import (  # pyright: ignore[reportMissingImports]
    MCPTransportError,
    create_transport_streams,
)
from exo.types import ExoError  # pyright: ignore[reportMissingImports]
from exo_mcp_cli.config import ServerEntry, substitute_env_vars
from exo_mcp_cli.vault import Vault

_VAULT_PATTERN = re.compile(r"\$\{vault:([^}]+)\}")


class MCPConnectionError(ExoError):
    """Raised when connecting to an MCP server fails."""


# ---------------------------------------------------------------------------
# Credential resolution
# ---------------------------------------------------------------------------


def _resolve_value(value: str, vault: Vault | None) -> str:
    """Resolve ``${vault:NAME}`` and ``${ENV}`` references in a string."""
    result = substitute_env_vars(value)
    if vault and _VAULT_PATTERN.search(result):
        result = vault.resolve(result)
    return result


def _resolve_dict(d: dict[str, str] | None, vault: Vault | None) -> dict[str, str] | None:
    """Resolve references in all values of a string dict."""
    if d is None:
        return None
    return {k: _resolve_value(v, vault) for k, v in d.items()}


def _resolve_entry(entry: ServerEntry, vault: Vault | None) -> ServerEntry:
    """Return a copy of *entry* with all credential references resolved."""
    from dataclasses import replace

    return replace(
        entry,
        url=_resolve_value(entry.url, vault) if entry.url else entry.url,
        command=_resolve_value(entry.command, vault) if entry.command else entry.command,
        headers=_resolve_dict(entry.headers, vault),
        env=_resolve_dict(entry.env, vault),
    )


# ---------------------------------------------------------------------------
# Connection context manager
# ---------------------------------------------------------------------------


def _create_transport(entry: ServerEntry) -> Any:
    """Create the appropriate transport streams for *entry*.

    Delegates to :func:`exo.mcp.transport.create_transport_streams`.
    WebSocket support is enabled via *allow_websocket=True* because the
    CLI (unlike the framework client) accepts that transport.
    """
    try:
        return create_transport_streams(
            entry.transport,
            command=entry.command or "",
            args=entry.args,
            env=entry.env,
            cwd=entry.cwd,
            url=entry.url or "",
            headers=entry.headers,
            timeout=entry.timeout,
            # sse_read_timeout: ServerEntry has no dedicated field; fall back
            # to the connection timeout so the call signature stays consistent.
            sse_read_timeout=entry.timeout,
            allow_websocket=True,
        )
    except MCPTransportError as exc:
        raise MCPConnectionError(str(exc)) from exc


@asynccontextmanager
async def connect_to_server(
    entry: ServerEntry,
    vault: Vault | None = None,
) -> AsyncIterator[ClientSession]:
    """Connect to an MCP server and yield a ready ``ClientSession``.

    Resolves ``${vault:...}`` and ``${ENV}`` credential references in
    the server entry before connecting.

    Usage::

        async with connect_to_server(entry, vault) as session:
            tools = await session.list_tools()
    """
    resolved = _resolve_entry(entry, vault)
    resolved.validate()

    stack = AsyncExitStack()
    try:
        transport = await stack.enter_async_context(_create_transport(resolved))
        read, write, *_ = transport
        session = await stack.enter_async_context(ClientSession(read, write))
        await session.initialize()
        yield session
    except MCPConnectionError:
        raise
    except Exception as exc:
        raise MCPConnectionError(
            f"Failed to connect to '{entry.name}' ({entry.transport}): {exc}"
        ) from exc
    finally:
        await stack.aclose()
