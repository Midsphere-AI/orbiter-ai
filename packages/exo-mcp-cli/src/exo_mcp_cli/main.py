"""exo-mcp CLI — standalone tool for interacting with MCP servers.

Usage::

    exo-mcp server list
    exo-mcp tool list my-server
    exo-mcp tool call my-server search --arg query=hello
    exo-mcp auth set api-key sk-abc123
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated

import typer

from exo_mcp_cli.config import (
    MCPConfigError,
    ServerEntry,
    find_config,
    load_config,
)
from exo_mcp_cli.output import print_error
from exo_mcp_cli.vault import Vault

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

app = typer.Typer(
    name="exo-mcp",
    help="Standalone CLI for interacting with MCP servers.",
    no_args_is_help=True,
)


@app.callback()
def main(
    ctx: typer.Context,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Enable verbose output."),
    ] = False,
    config: Annotated[
        str | None,
        typer.Option("--config", "-c", help="Path to mcp.json config file."),
    ] = None,
) -> None:
    """exo-mcp — interact with MCP servers from the command line."""
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    ctx.obj["config_path"] = config
    if verbose:
        logging.basicConfig(level=logging.DEBUG, format="%(name)s: %(message)s")


# ---------------------------------------------------------------------------
# Context helpers (used by all commands)
# ---------------------------------------------------------------------------


def resolve_config(ctx: typer.Context) -> tuple[Path | None, dict[str, ServerEntry]]:
    """Resolve config from context. Returns (path, servers)."""
    config_path = ctx.obj.get("config_path")
    try:
        path = find_config(config_path)
    except MCPConfigError as exc:
        print_error(str(exc))
        raise typer.Exit(code=1) from exc
    if path is None:
        return None, {}
    try:
        servers = load_config(path)
    except MCPConfigError as exc:
        print_error(str(exc))
        raise typer.Exit(code=1) from exc
    return path, servers


def get_server(ctx: typer.Context, name: str) -> ServerEntry:
    """Look up a server by name from the config, or exit with error."""
    path, servers = resolve_config(ctx)
    if name not in servers:
        available = ", ".join(servers) if servers else "(none)"
        location = f" in {path}" if path else ""
        print_error(f"Server '{name}' not found{location}. Available: {available}")
        raise typer.Exit(code=1)
    return servers[name]


def get_vault(ctx: typer.Context) -> Vault:
    """Get or create the Vault instance for this session."""
    if "vault" not in ctx.obj:
        ctx.obj["vault"] = Vault()
    return ctx.obj["vault"]


# ---------------------------------------------------------------------------
# Register subcommand groups
# ---------------------------------------------------------------------------

from exo_mcp_cli.commands.auth import auth_app  # noqa: E402
from exo_mcp_cli.commands.prompt import prompt_app  # noqa: E402
from exo_mcp_cli.commands.resource import resource_app  # noqa: E402
from exo_mcp_cli.commands.server import server_app  # noqa: E402
from exo_mcp_cli.commands.tool import tool_app  # noqa: E402

app.add_typer(server_app, name="server")
app.add_typer(tool_app, name="tool")
app.add_typer(resource_app, name="resource")
app.add_typer(prompt_app, name="prompt")
app.add_typer(auth_app, name="auth")
