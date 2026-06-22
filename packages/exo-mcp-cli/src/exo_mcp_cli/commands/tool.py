"""Tool interaction commands: list, call.

Supports injected arguments via ``--inject`` / ``-i`` and the
``EXO_MCP_TOOL_INJECT`` environment variable, mirroring the
``injected_tool_args`` pattern from the Exo agent runtime.
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Annotated, Any

import typer

from exo_mcp_cli.connection import MCPConnectionError, connect_to_server
from exo_mcp_cli.output import (
    _render_exc,
    console,
    print_error,
    print_json,
    print_tool_result,
    print_tools_table,
)

tool_app = typer.Typer(
    name="tool",
    help="List and execute MCP tools.",
    no_args_is_help=True,
)

_INJECT_ENV_KEY = "EXO_MCP_TOOL_INJECT"


def _build_arguments(
    arg: list[str] | None,
    json_args: str | None,
    inject: list[str] | None,
) -> dict[str, Any]:
    """Build kwargs dict from env, --inject, --json, and --arg flags.

    Precedence (lowest → highest):
        ``EXO_MCP_TOOL_INJECT`` → ``--inject`` → ``--json`` → ``--arg``
    """
    arguments: dict[str, Any] = {}

    # 0. EXO_MCP_TOOL_INJECT env var (lowest priority — JSON object)
    env_inject = os.environ.get(_INJECT_ENV_KEY)
    if env_inject:
        try:
            env_parsed = json.loads(env_inject)
            if isinstance(env_parsed, dict):
                arguments.update(env_parsed)
        except json.JSONDecodeError:
            console.print(
                f"[yellow]Warning: {_INJECT_ENV_KEY} is not valid JSON — ignored.[/yellow]"
            )

    # 1. --inject flag args (override env inject)
    if inject:
        for item in inject:
            if "=" not in item:
                print_error(f"Expected KEY=VALUE for --inject, got: {item}")
                raise typer.Exit(code=1)
            k, v = item.split("=", 1)
            arguments[k] = v

    # 2. JSON args
    if json_args:
        try:
            parsed = json.loads(json_args)
        except json.JSONDecodeError as exc:
            print_error(f"Invalid JSON: {exc}")
            raise typer.Exit(code=1) from exc
        if not isinstance(parsed, dict):
            print_error("--json must be a JSON object")
            raise typer.Exit(code=1)
        arguments.update(parsed)

    # 3. Key=value args (highest priority)
    if arg:
        for item in arg:
            if "=" not in item:
                print_error(f"Invalid argument format (expected KEY=VALUE): {item}")
                raise typer.Exit(code=1)
            k, v = item.split("=", 1)
            arguments[k] = v

    return arguments


@tool_app.command("list")
def tool_list(
    ctx: typer.Context,
    server: Annotated[str, typer.Argument(help="Server name.")],
    as_json: Annotated[
        bool,
        typer.Option("--json", help="Output as JSON."),
    ] = False,
) -> None:
    """List available tools from an MCP server."""
    from exo_mcp_cli.main import get_server, get_vault

    entry = get_server(ctx, server)
    vault = get_vault(ctx)

    async def _run() -> None:
        async with connect_to_server(entry, vault) as session:
            result = await session.list_tools()
            if as_json:
                data = []
                for t in result.tools:
                    item = {"name": t.name, "description": t.description}
                    schema = getattr(t, "inputSchema", None)
                    if schema:
                        item["inputSchema"] = schema
                    data.append(item)
                print_json(data)
            else:
                print_tools_table(result.tools)

    try:
        asyncio.run(_run())
    except MCPConnectionError as exc:
        print_error(str(exc))
        raise typer.Exit(code=1) from exc
    except Exception as exc:
        print_error(f"Failed to list tools: {_render_exc(exc)}")
        if ctx.obj.get("verbose"):
            console.print_exception()
        raise typer.Exit(code=1) from exc


@tool_app.command("call")
def tool_call(
    ctx: typer.Context,
    server: Annotated[str, typer.Argument(help="Server name.")],
    tool: Annotated[str, typer.Argument(help="Tool name.")],
    arg: Annotated[
        list[str] | None,
        typer.Option("--arg", "-a", help="Argument as KEY=VALUE (repeatable)."),
    ] = None,
    json_args: Annotated[
        str | None,
        typer.Option("--json", "-j", help="Arguments as a JSON string."),
    ] = None,
    inject: Annotated[
        list[str] | None,
        typer.Option(
            "--inject",
            "-i",
            help="Injected argument as KEY=VALUE (repeatable). "
            "Auto-filled, not specified by LLM. Overrides EXO_MCP_TOOL_INJECT.",
        ),
    ] = None,
    raw: Annotated[
        bool,
        typer.Option("--raw", help="Output raw JSON result."),
    ] = False,
) -> None:
    """Execute a tool on an MCP server.

    Supports injected arguments via --inject/-i and EXO_MCP_TOOL_INJECT env var.
    Precedence (lowest to highest): EXO_MCP_TOOL_INJECT → --inject → --json → --arg.
    """
    from exo_mcp_cli.main import get_server, get_vault

    entry = get_server(ctx, server)
    vault = get_vault(ctx)

    arguments = _build_arguments(arg, json_args, inject)

    async def _run() -> None:
        async with connect_to_server(entry, vault) as session:
            result = await session.call_tool(tool, arguments or None)
            if raw:
                data = {
                    "isError": getattr(result, "isError", False),
                    "content": [],
                }
                for item in getattr(result, "content", []):
                    text = getattr(item, "text", None)
                    if text is not None:
                        data["content"].append({"type": "text", "text": text})
                    else:
                        data["content"].append({"type": str(type(item).__name__)})
                print_json(data)
            else:
                print_tool_result(result)

    try:
        asyncio.run(_run())
    except MCPConnectionError as exc:
        print_error(str(exc))
        raise typer.Exit(code=1) from exc
    except Exception as exc:
        print_error(f"Tool call failed: {_render_exc(exc)}")
        if ctx.obj.get("verbose"):
            console.print_exception()
        raise typer.Exit(code=1) from exc
