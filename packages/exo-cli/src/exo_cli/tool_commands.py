"""Tool offloading commands: list, call, schema.

Discover and execute Exo native tools from the command line.
Agents can shell out to these commands instead of using LLM tool
calling, keeping tool schemas out of the context window.

Tool discovery: the ``--from`` flag accepts either a Python dotted
module path (``myapp.tools``) or a file path (``./tools.py``).
Module-level ``Tool`` instances and items in a ``tools`` list are
collected automatically.

Usage::

    exo tool list --from myapp.tools
    exo tool call search_web --from myapp.tools -j '{"query": "python"}'
    exo tool call greet --from myapp.tools -a name=Alice
    exo tool schema search_web --from myapp.tools
"""

from __future__ import annotations

import asyncio
import importlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.table import Table

from exo.tool import Tool

console = Console()

tool_app = typer.Typer(
    name="tool",
    help="Discover and execute Exo native tools (tool offloading).",
    no_args_is_help=True,
)


# ---------------------------------------------------------------------------
# Tool discovery
# ---------------------------------------------------------------------------


def _collect_tools_from_module(module: Any) -> dict[str, Tool]:
    """Scan a module for Tool instances.

    Checks all module-level attributes for Tool instances, and also
    inspects ``tools`` if it's a list/tuple.
    """
    found: dict[str, Tool] = {}

    for attr_name in dir(module):
        obj = getattr(module, attr_name, None)
        if isinstance(obj, Tool) and obj.name not in found:
            found[obj.name] = obj

    # Also check a conventional ``tools`` list
    tools_list = getattr(module, "tools", None)
    if isinstance(tools_list, (list, tuple)):
        for obj in tools_list:
            if isinstance(obj, Tool) and obj.name not in found:
                found[obj.name] = obj

    return found


def _load_module(source: str) -> Any:
    """Import a module from a dotted path or file path."""
    path = Path(source)
    if path.exists() and path.suffix == ".py":
        module_name = f"_exo_tool_{path.stem}"
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise typer.BadParameter(f"Cannot import {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        except Exception as exc:
            del sys.modules[module_name]
            raise typer.BadParameter(f"Error importing {path}: {exc}") from exc
        return module

    # Dotted module path
    try:
        return importlib.import_module(source)
    except ImportError as exc:
        raise typer.BadParameter(f"Cannot import '{source}': {exc}") from exc


def _discover(source: str) -> dict[str, Tool]:
    """Load module and return discovered tools."""
    module = _load_module(source)
    tools = _collect_tools_from_module(module)
    if not tools:
        raise typer.BadParameter(f"No tools found in '{source}'")
    return tools


def _get_tool(source: str, name: str) -> Tool:
    """Discover tools from source and return the named one."""
    tools = _discover(source)
    if name not in tools:
        available = ", ".join(sorted(tools.keys()))
        raise typer.BadParameter(f"Tool '{name}' not found. Available: {available}")
    return tools[name]


# ---------------------------------------------------------------------------
# Argument parsing and type coercion
# ---------------------------------------------------------------------------


def _coerce_value(value: str, schema: dict[str, Any]) -> Any:
    """Coerce a string value based on JSON Schema type info."""
    json_type = schema.get("type", "string")

    if json_type == "integer":
        try:
            return int(value)
        except ValueError:
            return value
    elif json_type == "number":
        try:
            return float(value)
        except ValueError:
            return value
    elif json_type == "boolean":
        return value.lower() in ("true", "1", "yes")
    elif json_type in ("array", "object"):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value

    return value


def _build_arguments(
    tool: Tool,
    arg: list[str] | None,
    json_args: str | None,
    inject: list[str] | None = None,
) -> dict[str, Any]:
    """Build kwargs dict from env, --inject, --json, and --arg flags, coercing types.

    Precedence: EXO_TOOL_INJECT (lowest) �� --inject → --json → --arg (highest).
    Injected args act like ``injected_tool_args`` in Agent — they provide
    values the LLM doesn't need to specify.
    """
    arguments: dict[str, Any] = {}
    properties = tool.parameters.get("properties", {})

    # 0. EXO_TOOL_INJECT env var (lowest priority — JSON object)
    env_inject = os.environ.get("EXO_TOOL_INJECT")
    if env_inject:
        try:
            env_parsed = json.loads(env_inject)
            if isinstance(env_parsed, dict):
                arguments.update(env_parsed)
        except json.JSONDecodeError:
            pass  # Silently ignore malformed env var

    # 1. --inject flag args (override env inject)
    if inject:
        for a in inject:
            if "=" not in a:
                raise typer.BadParameter(f"Expected KEY=VALUE for --inject, got: {a}")
            k, v = a.split("=", 1)
            prop_schema = properties.get(k, {})
            arguments[k] = _coerce_value(v, prop_schema)

    # 2. JSON args
    if json_args:
        try:
            parsed = json.loads(json_args)
        except json.JSONDecodeError as exc:
            raise typer.BadParameter(f"Invalid JSON: {exc}") from exc
        if not isinstance(parsed, dict):
            raise typer.BadParameter("--json must be a JSON object")
        arguments.update(parsed)

    # 3. Key=value args (highest priority)
    if arg:
        for a in arg:
            if "=" not in a:
                raise typer.BadParameter(f"Expected KEY=VALUE, got: {a}")
            k, v = a.split("=", 1)
            prop_schema = properties.get(k, {})
            arguments[k] = _coerce_value(v, prop_schema)

    return arguments


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def _format_result(result: Any) -> str:
    """Format a tool result for terminal output."""
    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        return json.dumps(result, indent=2, default=str)
    if isinstance(result, list):
        # ContentBlock list — extract text
        parts = []
        for item in result:
            text = getattr(item, "text", None)
            if text is not None:
                parts.append(text)
            else:
                parts.append(str(item))
        return "\n".join(parts) if parts else "[]"
    return str(result)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

_FROM_OPTION = Annotated[
    str | None,
    typer.Option(
        "--from",
        "-f",
        help="Python module path or .py file containing tools. Default: EXO_TOOL_SOURCE env var.",
    ),
]


def _resolve_source(source: str | None) -> str:
    """Resolve tool source from flag or EXO_TOOL_SOURCE env var."""
    resolved = source or os.environ.get("EXO_TOOL_SOURCE")
    if not resolved:
        console.print(
            "[red]Error: --from required or set EXO_TOOL_SOURCE environment variable.[/red]"
        )
        raise typer.Exit(code=1)
    return resolved


@tool_app.command("list")
def tool_list(
    source: _FROM_OPTION = None,
    as_json: Annotated[
        bool,
        typer.Option("--json", help="Output as JSON."),
    ] = False,
) -> None:
    """List available tools from a module."""
    tools = _discover(_resolve_source(source))

    if as_json:
        data = []
        for t in sorted(tools.values(), key=lambda t: t.name):
            item: dict[str, Any] = {"name": t.name, "description": t.description}
            item["parameters"] = t.parameters
            data.append(item)
        console.print_json(json.dumps(data))
        return

    table = Table(title="Exo Tools", show_lines=False)
    table.add_column("Name", style="cyan", no_wrap=True)
    table.add_column("Description")
    table.add_column("Parameters", style="dim")

    for t in sorted(tools.values(), key=lambda t: t.name):
        props = t.parameters.get("properties", {})
        required = set(t.parameters.get("required", []))
        params = []
        for pname, pschema in props.items():
            ptype = pschema.get("type", "?")
            marker = "*" if pname in required else ""
            params.append(f"{pname}{marker}:{ptype}")
        table.add_row(t.name, t.description, ", ".join(params))

    console.print(table)


@tool_app.command("call")
def tool_call(
    name: Annotated[str, typer.Argument(help="Tool name.")],
    source: _FROM_OPTION = None,
    arg: Annotated[
        list[str] | None,
        typer.Option("--arg", "-a", help="Argument as KEY=VALUE (repeatable)."),
    ] = None,
    json_args: Annotated[
        str | None,
        typer.Option("--json", "-j", help="Arguments as a JSON object string."),
    ] = None,
    inject: Annotated[
        list[str] | None,
        typer.Option(
            "--inject",
            "-i",
            help="Injected argument as KEY=VALUE (repeatable). "
            "Like injected_tool_args — visible in schema but auto-filled.",
        ),
    ] = None,
    raw: Annotated[
        bool,
        typer.Option("--raw", "-r", help="Output raw JSON result."),
    ] = False,
) -> None:
    """Execute a tool and print its result."""
    t = _get_tool(_resolve_source(source), name)
    arguments = _build_arguments(t, arg, json_args, inject)

    async def _run() -> Any:
        return await t.execute(**arguments)

    try:
        result = asyncio.run(_run())
    except Exception as exc:
        console.print(f"[red]Error:[/red] {exc}", highlight=False)
        raise typer.Exit(code=1) from exc

    if raw:
        console.print_json(json.dumps({"result": result}, default=str))
    else:
        console.print(_format_result(result), highlight=False)


@tool_app.command("schema")
def tool_schema(
    name: Annotated[str, typer.Argument(help="Tool name.")],
    source: _FROM_OPTION = None,
) -> None:
    """Show the full JSON schema for a tool."""
    t = _get_tool(_resolve_source(source), name)
    console.print_json(json.dumps(t.to_schema(), indent=2))
