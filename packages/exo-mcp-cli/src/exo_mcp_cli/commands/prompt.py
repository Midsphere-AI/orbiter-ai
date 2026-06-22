"""Prompt interaction commands: list, get."""

from __future__ import annotations

import asyncio
from typing import Annotated

import typer

from exo_mcp_cli.connection import MCPConnectionError, connect_to_server
from exo_mcp_cli.output import (
    _render_exc,
    console,
    print_error,
    print_json,
    print_prompt_messages,
    print_prompts_table,
)

prompt_app = typer.Typer(
    name="prompt",
    help="List and retrieve MCP prompts.",
    no_args_is_help=True,
)


@prompt_app.command("list")
def prompt_list(
    ctx: typer.Context,
    server: Annotated[str, typer.Argument(help="Server name.")],
    as_json: Annotated[
        bool,
        typer.Option("--json", help="Output as JSON."),
    ] = False,
) -> None:
    """List available prompts from an MCP server."""
    from exo_mcp_cli.main import get_server, get_vault

    entry = get_server(ctx, server)
    vault = get_vault(ctx)

    async def _run() -> None:
        async with connect_to_server(entry, vault) as session:
            result = await session.list_prompts()
            if as_json:
                data = []
                for p in result.prompts:
                    args_list = getattr(p, "arguments", None) or []
                    data.append(
                        {
                            "name": p.name,
                            "description": getattr(p, "description", None),
                            "arguments": [
                                {
                                    "name": getattr(a, "name", ""),
                                    "description": getattr(a, "description", None),
                                    "required": getattr(a, "required", False),
                                }
                                for a in args_list
                            ],
                        }
                    )
                print_json(data)
            else:
                print_prompts_table(result.prompts)

    try:
        asyncio.run(_run())
    except MCPConnectionError as exc:
        print_error(str(exc))
        raise typer.Exit(code=1) from exc
    except Exception as exc:
        print_error(f"Failed to list prompts: {_render_exc(exc)}")
        if ctx.obj.get("verbose"):
            console.print_exception()
        raise typer.Exit(code=1) from exc


@prompt_app.command("get")
def prompt_get(
    ctx: typer.Context,
    server: Annotated[str, typer.Argument(help="Server name.")],
    name: Annotated[str, typer.Argument(help="Prompt name.")],
    arg: Annotated[
        list[str] | None,
        typer.Option("--arg", "-a", help="Argument as KEY=VALUE (repeatable)."),
    ] = None,
    as_json: Annotated[
        bool,
        typer.Option("--json", help="Output as JSON."),
    ] = False,
) -> None:
    """Get a prompt from an MCP server with optional arguments."""
    from exo_mcp_cli.main import get_server, get_vault

    entry = get_server(ctx, server)
    vault = get_vault(ctx)

    # Parse arguments
    arguments: dict[str, str] | None = None
    if arg:
        arguments = {}
        for a in arg:
            if "=" not in a:
                print_error(f"Invalid argument format (expected KEY=VALUE): {a}")
                raise typer.Exit(code=1)
            k, v = a.split("=", 1)
            arguments[k] = v

    async def _run() -> None:
        async with connect_to_server(entry, vault) as session:
            result = await session.get_prompt(name, arguments)
            if as_json:
                msgs = []
                for msg in getattr(result, "messages", []):
                    content = getattr(msg, "content", None)
                    text = getattr(content, "text", None) if content else None
                    msgs.append(
                        {
                            "role": getattr(msg, "role", "?"),
                            "content": text or str(content),
                        }
                    )
                print_json(
                    {
                        "description": getattr(result, "description", None),
                        "messages": msgs,
                    }
                )
            else:
                print_prompt_messages(result)

    try:
        asyncio.run(_run())
    except MCPConnectionError as exc:
        print_error(str(exc))
        raise typer.Exit(code=1) from exc
    except Exception as exc:
        print_error(f"Failed to get prompt: {_render_exc(exc)}")
        if ctx.obj.get("verbose"):
            console.print_exception()
        raise typer.Exit(code=1) from exc
