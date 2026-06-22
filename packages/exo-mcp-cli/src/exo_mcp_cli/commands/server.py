"""Server management commands: list, add, remove, test."""

from __future__ import annotations

import asyncio
import time
from typing import Annotated

import typer

from exo_mcp_cli.config import (
    MCPConfigError,
    ServerEntry,
    add_server,
    default_config_path,
    remove_server,
)
from exo_mcp_cli.connection import MCPConnectionError, connect_to_server
from exo_mcp_cli.output import _render_exc, console, print_error, print_servers_table, print_success

server_app = typer.Typer(
    name="server",
    help="Manage MCP server configurations.",
    no_args_is_help=True,
)


@server_app.command("list")
def server_list(ctx: typer.Context) -> None:
    """List all configured MCP servers."""
    from exo_mcp_cli.main import resolve_config

    _, servers = resolve_config(ctx)
    print_servers_table(servers)


@server_app.command("add")
def server_add(
    ctx: typer.Context,
    name: Annotated[str, typer.Argument(help="Server name.")],
    transport: Annotated[
        str,
        typer.Option(
            "--transport", "-t", help="Transport type (stdio, sse, streamable_http, websocket)."
        ),
    ] = "stdio",
    command: Annotated[
        str | None,
        typer.Option("--command", help="Executable for stdio transport."),
    ] = None,
    args: Annotated[
        list[str] | None,
        typer.Option("--arg", help="Command arguments (repeatable)."),
    ] = None,
    url: Annotated[
        str | None,
        typer.Option("--url", help="Server URL for sse/streamable_http/websocket."),
    ] = None,
    header: Annotated[
        list[str] | None,
        typer.Option(
            "--header", help="Header as KEY=VALUE (repeatable). Values are auto-encrypted."
        ),
    ] = None,
    env: Annotated[
        list[str] | None,
        typer.Option(
            "--env",
            help="Environment variable as KEY=VALUE (repeatable). Values are auto-encrypted.",
        ),
    ] = None,
    timeout: Annotated[
        float,
        typer.Option("--timeout", help="Connection timeout in seconds."),
    ] = 30.0,
) -> None:
    """Add or update an MCP server configuration.

    Sensitive values in --header and --env are automatically stored in the
    encrypted vault and replaced with ${vault:...} references in mcp.json.
    """
    from pathlib import Path

    from exo_mcp_cli.config import find_config
    from exo_mcp_cli.main import get_vault

    config_path_str = ctx.obj.get("config_path")
    if config_path_str:
        # Explicit --config: use the given path (may not exist yet — that's fine for add).
        config_path = Path(config_path_str)
    else:
        # No explicit path: prefer an existing config (same discovery as list/remove)
        # so that `server add` writes to the same file that `server list` reads.
        try:
            existing = find_config(None)
        except MCPConfigError:
            existing = None
        config_path = existing if existing is not None else default_config_path()

    # Parse headers and auto-vault sensitive values
    parsed_headers: dict[str, str] | None = None
    if header:
        parsed_headers = {}
        vault = get_vault(ctx)
        for h in header:
            if "=" not in h:
                print_error(f"Invalid header format (expected KEY=VALUE): {h}")
                raise typer.Exit(code=1)
            k, v = h.split("=", 1)
            vault_key = f"{name}_header_{k}"
            vault.set(vault_key, v)
            parsed_headers[k] = f"${{vault:{vault_key}}}"

    # Parse env vars and auto-vault
    parsed_env: dict[str, str] | None = None
    if env:
        parsed_env = {}
        vault = get_vault(ctx)
        for e in env:
            if "=" not in e:
                print_error(f"Invalid env format (expected KEY=VALUE): {e}")
                raise typer.Exit(code=1)
            k, v = e.split("=", 1)
            vault_key = f"{name}_env_{k}"
            vault.set(vault_key, v)
            parsed_env[k] = f"${{vault:{vault_key}}}"

    entry = ServerEntry(
        name=name,
        transport=transport,
        command=command,
        args=args or [],
        env=parsed_env,
        url=url,
        headers=parsed_headers,
        timeout=timeout,
    )

    try:
        entry.validate()
    except MCPConfigError as exc:
        print_error(str(exc))
        raise typer.Exit(code=1) from exc

    add_server(config_path, entry)
    print_success(f"Server '{name}' added to {config_path}")
    if parsed_headers or parsed_env:
        console.print("[dim]Sensitive values stored in encrypted vault.[/dim]")


@server_app.command("remove")
def server_remove(
    ctx: typer.Context,
    name: Annotated[str, typer.Argument(help="Server name to remove.")],
) -> None:
    """Remove an MCP server from the configuration."""
    from exo_mcp_cli.main import resolve_config

    path, _servers = resolve_config(ctx)
    if path is None:
        print_error(
            "No config file found. "
            "Create one with `exo-mcp server add <name> ...` or use --config to point to an existing file."
        )
        raise typer.Exit(code=1)
    if not remove_server(path, name):
        print_error(f"Server '{name}' not found in config.")
        raise typer.Exit(code=1)
    print_success(f"Server '{name}' removed from {path}")


@server_app.command("test")
def server_test(
    ctx: typer.Context,
    name: Annotated[str, typer.Argument(help="Server name to test.")],
) -> None:
    """Test connectivity to an MCP server by sending a ping."""
    from exo_mcp_cli.main import get_server, get_vault

    entry = get_server(ctx, name)
    vault = get_vault(ctx)

    async def _test() -> None:
        console.print(f"Connecting to [cyan]{name}[/cyan] ({entry.transport})...")
        start = time.monotonic()
        async with connect_to_server(entry, vault) as session:
            connect_time = time.monotonic() - start
            console.print(f"  Connected in {connect_time:.1f}s")

            ping_start = time.monotonic()
            await asyncio.wait_for(session.send_ping(), timeout=entry.timeout)
            ping_time = time.monotonic() - ping_start
            console.print(f"  Ping: {ping_time * 1000:.0f}ms")

            # Show server info
            tools = await asyncio.wait_for(session.list_tools(), timeout=entry.timeout)
            console.print(f"  Tools: {len(tools.tools)}")

        print_success(f"Server '{name}' is reachable.")

    try:
        asyncio.run(_test())
    except MCPConnectionError as exc:
        print_error(str(exc))
        raise typer.Exit(code=1) from exc
    except Exception as exc:
        print_error(f"Connection failed: {_render_exc(exc)}")
        if ctx.obj.get("verbose"):
            console.print_exception()
        raise typer.Exit(code=1) from exc
