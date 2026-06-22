"""Rich formatting utilities for CLI output."""

from __future__ import annotations

import json
from typing import Any

from rich.console import Console
from rich.json import JSON as RichJSON  # noqa: N811
from rich.table import Table

from exo._internal.errors import unwrap_exception_group  # pyright: ignore[reportMissingImports]
from exo_mcp_cli.config import ServerEntry

console = Console()


# ---------------------------------------------------------------------------
# Error / success helpers
# ---------------------------------------------------------------------------


def _render_exc(exc: BaseException) -> str:
    """Unwrap a single-child ExceptionGroup and return a clean string.

    Prevents raw ExceptionGroup repr leaking into user-facing error messages
    when asyncio task machinery wraps a lone failure in a group.
    """
    return str(unwrap_exception_group(exc))


def print_error(message: str) -> None:
    """Print a red error message."""
    console.print(f"[red]Error: {message}[/red]")


def print_success(message: str) -> None:
    """Print a green success message."""
    console.print(f"[green]{message}[/green]")


def print_json(data: Any) -> None:
    """Pretty-print JSON data with syntax highlighting."""
    console.print(RichJSON(json.dumps(data, indent=2, default=str)))


# ---------------------------------------------------------------------------
# Server table
# ---------------------------------------------------------------------------


def print_servers_table(servers: dict[str, ServerEntry]) -> None:
    """Print a table of configured MCP servers."""
    if not servers:
        console.print("[dim]No servers configured.[/dim]")
        return
    table = Table(title="MCP Servers")
    table.add_column("Name", style="cyan", no_wrap=True, min_width=8)
    table.add_column("Transport", no_wrap=True)
    table.add_column("Endpoint")
    for entry in servers.values():
        endpoint = entry.url or (
            f"{entry.command} {' '.join(entry.args)}" if entry.command else "-"
        )
        table.add_row(entry.name, entry.transport, endpoint)
    console.print(table)


# ---------------------------------------------------------------------------
# Tool table / result
# ---------------------------------------------------------------------------


def print_tools_table(tools: list[Any]) -> None:
    """Print a table of MCP tools."""
    if not tools:
        console.print("[dim]No tools available.[/dim]")
        return
    table = Table(title="Tools")
    table.add_column("Name", style="cyan", no_wrap=True)
    table.add_column("Description")
    table.add_column("Parameters", style="dim")
    for tool in tools:
        desc = (tool.description or "")[:80]
        params = ""
        schema = getattr(tool, "inputSchema", None)
        if schema and isinstance(schema, dict):
            props = schema.get("properties", {})
            required = set(schema.get("required", []))
            parts = []
            for pname in props:
                marker = "*" if pname in required else ""
                parts.append(f"{pname}{marker}")
            params = ", ".join(parts)
        table.add_row(tool.name, desc, params)
    console.print(table)


def print_tool_result(result: Any) -> None:
    """Format and print a CallToolResult."""
    if getattr(result, "isError", False):
        for item in getattr(result, "content", []):
            text = getattr(item, "text", None) or str(item)
            console.print(f"[red]{text}[/red]")
        return
    for item in getattr(result, "content", []):
        text = getattr(item, "text", None)
        if text is not None:
            console.print(text)
        else:
            console.print(str(item))


# ---------------------------------------------------------------------------
# Resource table / contents
# ---------------------------------------------------------------------------


def print_resources_table(resources: list[Any]) -> None:
    """Print a table of MCP resources."""
    if not resources:
        console.print("[dim]No resources available.[/dim]")
        return
    table = Table(title="Resources")
    table.add_column("Name", style="cyan", no_wrap=True)
    table.add_column("URI", no_wrap=True)
    table.add_column("MIME Type", style="dim", no_wrap=True)
    table.add_column("Description")
    for res in resources:
        name = getattr(res, "name", "-")
        uri = str(getattr(res, "uri", "-"))
        mime = getattr(res, "mimeType", None) or "-"
        desc = (getattr(res, "description", None) or "")[:60]
        table.add_row(name, uri, mime, desc)
    console.print(table)


def print_resource_contents(result: Any) -> None:
    """Print the contents of a read resource."""
    for item in getattr(result, "contents", []):
        text = getattr(item, "text", None)
        if text is not None:
            console.print(text)
        else:
            blob = getattr(item, "blob", None)
            if blob:
                console.print(f"[dim]Binary content ({len(blob)} bytes)[/dim]")
            else:
                console.print(str(item))


# ---------------------------------------------------------------------------
# Prompt table / messages
# ---------------------------------------------------------------------------


def print_prompts_table(prompts: list[Any]) -> None:
    """Print a table of MCP prompts."""
    if not prompts:
        console.print("[dim]No prompts available.[/dim]")
        return
    table = Table(title="Prompts")
    table.add_column("Name", style="cyan", no_wrap=True)
    table.add_column("Description")
    table.add_column("Arguments", style="dim")
    for prompt in prompts:
        desc = (getattr(prompt, "description", None) or "")[:60]
        args_list = getattr(prompt, "arguments", None) or []
        parts = []
        for arg in args_list:
            name = getattr(arg, "name", "?")
            req = getattr(arg, "required", False)
            parts.append(f"{name}{'*' if req else ''}")
        table.add_row(prompt.name, desc, ", ".join(parts))
    console.print(table)


def print_prompt_messages(result: Any) -> None:
    """Print prompt messages with role labels."""
    desc = getattr(result, "description", None)
    if desc:
        console.print(f"[dim]{desc}[/dim]\n")
    for msg in getattr(result, "messages", []):
        role = getattr(msg, "role", "?")
        color = "blue" if role == "user" else "green"
        content = getattr(msg, "content", None)
        text = getattr(content, "text", None) if content else str(content)
        console.print(f"[bold {color}]{role}:[/bold {color}] {text}")
