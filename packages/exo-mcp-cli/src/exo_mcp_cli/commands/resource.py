"""Resource interaction commands: list, read."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Annotated

import typer

from exo_mcp_cli.connection import MCPConnectionError, connect_to_server
from exo_mcp_cli.output import (
    console,
    print_error,
    print_json,
    print_resource_contents,
    print_resources_table,
    print_success,
)

resource_app = typer.Typer(
    name="resource",
    help="List and read MCP resources.",
    no_args_is_help=True,
)


@resource_app.command("list")
def resource_list(
    ctx: typer.Context,
    server: Annotated[str, typer.Argument(help="Server name.")],
    as_json: Annotated[
        bool,
        typer.Option("--json", help="Output as JSON."),
    ] = False,
) -> None:
    """List available resources from an MCP server."""
    from exo_mcp_cli.main import get_server, get_vault

    entry = get_server(ctx, server)
    vault = get_vault(ctx)

    async def _run() -> None:
        async with connect_to_server(entry, vault) as session:
            result = await session.list_resources()
            if as_json:
                data = []
                for r in result.resources:
                    data.append(
                        {
                            "name": getattr(r, "name", None),
                            "uri": str(getattr(r, "uri", "")),
                            "mimeType": getattr(r, "mimeType", None),
                            "description": getattr(r, "description", None),
                        }
                    )
                print_json(data)
            else:
                print_resources_table(result.resources)

    try:
        asyncio.run(_run())
    except MCPConnectionError as exc:
        print_error(str(exc))
        raise typer.Exit(code=1) from exc
    except Exception as exc:
        print_error(f"Failed to list resources: {exc}")
        if ctx.obj.get("verbose"):
            console.print_exception()
        raise typer.Exit(code=1) from exc


@resource_app.command("read")
def resource_read(
    ctx: typer.Context,
    server: Annotated[str, typer.Argument(help="Server name.")],
    uri: Annotated[str, typer.Argument(help="Resource URI to read.")],
    output: Annotated[
        str | None,
        typer.Option("--output", "-o", help="Write output to file instead of stdout."),
    ] = None,
) -> None:
    """Read a resource from an MCP server."""
    from exo_mcp_cli.main import get_server, get_vault

    entry = get_server(ctx, server)
    vault = get_vault(ctx)

    async def _run() -> None:
        import base64

        from pydantic import AnyUrl

        async with connect_to_server(entry, vault) as session:
            result = await session.read_resource(AnyUrl(uri))
            if output:
                contents = getattr(result, "contents", [])
                if len(contents) == 1:
                    # Single-blob: write to the exact path given.
                    out_path = Path(output)
                    item = contents[0]
                    text = getattr(item, "text", None)
                    if text is not None:
                        out_path.write_text(text, encoding="utf-8")
                    else:
                        blob = getattr(item, "blob", "")
                        out_path.write_bytes(base64.b64decode(blob))
                    print_success(f"Written to {out_path}")
                else:
                    # Multi-blob: index filenames as <base>.<n>[.<ext>] so no item
                    # is silently overwritten.
                    base_path = Path(output)
                    stem = base_path.stem
                    suffix = base_path.suffix
                    written: list[Path] = []
                    for idx, item in enumerate(contents):
                        indexed_path = base_path.with_name(f"{stem}.{idx}{suffix}")
                        text = getattr(item, "text", None)
                        if text is not None:
                            indexed_path.write_text(text, encoding="utf-8")
                        else:
                            blob = getattr(item, "blob", "")
                            indexed_path.write_bytes(base64.b64decode(blob))
                        written.append(indexed_path)
                    for p in written:
                        print_success(f"Written to {p}")
            else:
                print_resource_contents(result)

    try:
        asyncio.run(_run())
    except MCPConnectionError as exc:
        print_error(str(exc))
        raise typer.Exit(code=1) from exc
    except Exception as exc:
        print_error(f"Failed to read resource: {exc}")
        if ctx.obj.get("verbose"):
            console.print_exception()
        raise typer.Exit(code=1) from exc
