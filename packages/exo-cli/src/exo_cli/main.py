"""Exo CLI — command-line agent runner.

Entry point for the ``exo`` command. Supports agent/swarm execution
from YAML config files with environment variable override, model
selection, verbosity control, and streaming output.

Config file search order (first found wins):
    1. ``--config`` / ``-c`` flag (explicit path)
    2. ``.exo.yaml`` in current directory
    3. ``exo.config.yaml`` in current directory

Usage::

    exo run --config agents.yaml "What is 2+2?"
    exo run -m openai:gpt-4o "Hello"
    exo --verbose run "Explain Python decorators"
    exo start worker --redis-url redis://localhost:6379
    exo task list --status running
    exo task status <task_id>
    exo task cancel <task_id>
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Annotated, Any
from urllib.parse import urlparse

import typer
from rich.console import Console
from rich.table import Table

from exo.types import ExoError  # pyright: ignore[reportMissingImports]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config file discovery
# ---------------------------------------------------------------------------

_DEFAULT_CONFIG_NAMES = (".exo.yaml", "exo.config.yaml")


class CLIError(ExoError):
    """Raised for CLI-level errors (config not found, parse failures)."""


def find_config(directory: str | Path | None = None) -> Path | None:
    """Search *directory* (default: cwd) for a config file.

    Returns the first matching path or ``None`` if no config exists.
    """
    base = Path(directory) if directory else Path.cwd()
    for name in _DEFAULT_CONFIG_NAMES:
        candidate = base / name
        if candidate.is_file():
            return candidate
    return None


def load_config(path: str | Path) -> dict[str, Any]:
    """Load and validate a YAML config file.

    Delegates to :func:`exo.loader.load_yaml` for variable substitution,
    then validates the top-level structure.

    Raises:
        CLIError: If the file doesn't exist or isn't valid YAML dict.
    """
    p = Path(path)
    if not p.is_file():
        raise CLIError(f"Config file not found: {p}")

    from exo.loader import LoaderError, load_yaml  # lazy import

    try:
        data = load_yaml(p)
    except LoaderError as exc:
        raise CLIError(f"Invalid config: {exc}") from exc
    return data


def resolve_config(config_path: str | None) -> dict[str, Any] | None:
    """Resolve config from explicit path or auto-discovery.

    Returns:
        Parsed config dict, or ``None`` if no config is available.
    """
    if config_path:
        return load_config(config_path)
    found = find_config()
    if found:
        return load_config(found)
    return None


# ---------------------------------------------------------------------------
# Typer CLI app
# ---------------------------------------------------------------------------

app = typer.Typer(
    name="exo",
    help="Exo — multi-agent framework CLI.",
    no_args_is_help=True,
)

console = Console()


@app.callback()
def main(
    ctx: typer.Context,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Enable verbose output."),
    ] = False,
) -> None:
    """Exo CLI — run agents from the command line."""
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose


@app.command()
def run(
    ctx: typer.Context,
    input_text: Annotated[
        str,
        typer.Argument(help="Input text to send to the agent."),
    ],
    config: Annotated[
        str | None,
        typer.Option("--config", "-c", help="Path to YAML config file."),
    ] = None,
    model: Annotated[
        str | None,
        typer.Option("--model", "-m", help="Model string (e.g. openai:gpt-4o)."),
    ] = None,
    stream: Annotated[
        bool,
        typer.Option("--stream", "-s", help="Enable streaming output."),
    ] = False,
) -> None:
    """Run an agent or swarm with the given input."""
    verbose: bool = ctx.obj.get("verbose", False)

    logger.debug(
        "CLI command=%s args=%r",
        "run",
        {"input": input_text, "config": config, "model": model, "stream": stream},
    )

    # Resolve config
    cfg = resolve_config(config)
    if verbose and cfg:
        console.print(f"[dim]Loaded config with keys: {list(cfg.keys())}[/dim]")

    # Store parsed state in context for downstream use (loader, console, etc.)
    ctx.obj["config"] = cfg
    ctx.obj["model"] = model
    ctx.obj["stream"] = stream
    ctx.obj["input"] = input_text

    if not cfg:
        console.print("[yellow]No config file found. Use --config or create .exo.yaml[/yellow]")
        raise typer.Exit(code=1)

    if verbose:
        console.print(f"[dim]Model: {model or 'auto'}[/dim]")
        console.print(f"[dim]Streaming: {stream}[/dim]")

    console.print(f"[green]Running with input:[/green] {input_text}")


# ---------------------------------------------------------------------------
# Subcommand group: start
# ---------------------------------------------------------------------------

start_app = typer.Typer(
    name="start",
    help="Start long-running services.",
    no_args_is_help=True,
)
app.add_typer(start_app, name="start")


def _mask_redis_url(url: str) -> str:
    """Return a masked version of the Redis URL showing only the host."""
    try:
        parsed = urlparse(url)
        host = parsed.hostname or "unknown"
        port = parsed.port or 6379
        return f"redis://{host}:{port}/***"
    except Exception:
        return "redis://***"


@start_app.command("worker")
def start_worker(
    redis_url: Annotated[
        str | None,
        typer.Option("--redis-url", help="Redis connection URL (default: EXO_REDIS_URL env var)."),
    ] = None,
    concurrency: Annotated[
        int,
        typer.Option("--concurrency", help="Number of concurrent task executions."),
    ] = 1,
    queue: Annotated[
        str,
        typer.Option("--queue", help="Redis Streams queue name."),
    ] = "exo:tasks",
    worker_id: Annotated[
        str | None,
        typer.Option("--worker-id", help="Unique worker ID (auto-generated if not set)."),
    ] = None,
) -> None:
    """Start a distributed worker that claims and executes agent tasks."""
    logger.debug(
        "CLI command=%s args=%r",
        "start worker",
        {
            "redis_url": bool(redis_url),
            "concurrency": concurrency,
            "queue": queue,
            "worker_id": worker_id,
        },
    )
    url = redis_url or os.environ.get("EXO_REDIS_URL")
    if not url:
        console.print(
            "[red]Error: --redis-url required or set EXO_REDIS_URL environment variable.[/red]"
        )
        raise typer.Exit(code=1)

    from exo.distributed.worker import Worker  # pyright: ignore[reportMissingImports]

    worker = Worker(
        url,
        worker_id=worker_id,
        concurrency=concurrency,
        queue_name=queue,
    )

    # Print startup banner
    console.print("[bold green]Exo Worker Starting[/bold green]")
    console.print(f"  Worker ID:   {worker.worker_id}")
    console.print(f"  Redis URL:   {_mask_redis_url(url)}")
    console.print(f"  Queue:       {queue}")
    console.print(f"  Concurrency: {concurrency}")
    console.print()
    console.print("[dim]Press Ctrl+C to stop.[/dim]")

    asyncio.run(worker.start())


# ---------------------------------------------------------------------------
# Subcommand group: task
# ---------------------------------------------------------------------------

task_app = typer.Typer(
    name="task",
    help="Inspect and manage distributed tasks.",
    no_args_is_help=True,
)
app.add_typer(task_app, name="task")


def _resolve_redis_url(redis_url: str | None) -> str:
    """Resolve Redis URL from flag or environment variable."""
    url = redis_url or os.environ.get("EXO_REDIS_URL")
    if not url:
        console.print(
            "[red]Error: --redis-url required or set EXO_REDIS_URL environment variable.[/red]"
        )
        raise typer.Exit(code=1)
    return url


def _format_timestamp(ts: float | None) -> str:
    """Format a Unix timestamp as a human-readable string."""
    if ts is None:
        return "-"
    from datetime import UTC, datetime

    return datetime.fromtimestamp(ts, tz=UTC).strftime("%Y-%m-%d %H:%M:%S UTC")


def _format_duration(started: float | None, completed: float | None) -> str:
    """Format duration between two timestamps."""
    if started is None:
        return "-"
    if completed is None:
        return "running..."
    secs = completed - started
    if secs < 1:
        return f"{secs * 1000:.0f}ms"
    if secs < 60:
        return f"{secs:.1f}s"
    return f"{secs / 60:.1f}m"


def _status_color(status: str) -> str:
    """Return a Rich color name for a task status."""
    colors: dict[str, str] = {
        "pending": "yellow",
        "running": "blue",
        "completed": "green",
        "failed": "red",
        "cancelled": "dim",
        "retrying": "magenta",
    }
    return colors.get(status, "white")


@task_app.command("status")
def task_status(
    task_id: Annotated[
        str,
        typer.Argument(help="Task ID to inspect."),
    ],
    redis_url: Annotated[
        str | None,
        typer.Option("--redis-url", help="Redis connection URL (default: EXO_REDIS_URL env var)."),
    ] = None,
) -> None:
    """Show status details for a specific task."""
    logger.debug("CLI command=%s args=%r", "task status", {"task_id": task_id})
    url = _resolve_redis_url(redis_url)

    async def _show() -> None:
        from exo.distributed.store import TaskStore  # pyright: ignore[reportMissingImports]

        store = TaskStore(url)
        await store.connect()
        try:
            result = await store.get_status(task_id)
        finally:
            await store.disconnect()

        if result is None:
            console.print(f"[yellow]Task not found: {task_id}[/yellow]")
            raise typer.Exit(code=1)

        color = _status_color(result.status)
        console.print(f"[bold]Task {result.task_id}[/bold]")
        console.print(f"  Status:      [{color}]{result.status}[/{color}]")
        console.print(f"  Worker:      {result.worker_id or '-'}")
        console.print(f"  Started:     {_format_timestamp(result.started_at)}")
        console.print(f"  Completed:   {_format_timestamp(result.completed_at)}")
        console.print(f"  Duration:    {_format_duration(result.started_at, result.completed_at)}")
        console.print(f"  Retries:     {result.retries}")
        if result.error:
            console.print(f"  Error:       [red]{result.error}[/red]")
        if result.result:
            import json

            preview = json.dumps(result.result)
            if len(preview) > 200:
                preview = preview[:200] + "..."
            console.print(f"  Result:      {preview}")

    asyncio.run(_show())


@task_app.command("cancel")
def task_cancel(
    task_id: Annotated[
        str,
        typer.Argument(help="Task ID to cancel."),
    ],
    redis_url: Annotated[
        str | None,
        typer.Option("--redis-url", help="Redis connection URL (default: EXO_REDIS_URL env var)."),
    ] = None,
) -> None:
    """Cancel a running distributed task."""
    logger.debug("CLI command=%s args=%r", "task cancel", {"task_id": task_id})
    url = _resolve_redis_url(redis_url)

    async def _cancel() -> None:
        from exo.distributed.broker import TaskBroker  # pyright: ignore[reportMissingImports]

        broker = TaskBroker(url)
        await broker.connect()
        try:
            await broker.cancel(task_id)
        finally:
            await broker.disconnect()

        console.print(f"[green]Task {task_id} cancelled.[/green]")

    asyncio.run(_cancel())


@task_app.command("list")
def task_list(
    status: Annotated[
        str | None,
        typer.Option(
            "--status",
            help="Filter by status (pending, running, completed, failed, cancelled, retrying).",
        ),
    ] = None,
    limit: Annotated[
        int,
        typer.Option("--limit", help="Maximum number of tasks to display."),
    ] = 100,
    redis_url: Annotated[
        str | None,
        typer.Option("--redis-url", help="Redis connection URL (default: EXO_REDIS_URL env var)."),
    ] = None,
) -> None:
    """List recent distributed tasks."""
    logger.debug("CLI command=%s args=%r", "task list", {"status": status, "limit": limit})
    url = _resolve_redis_url(redis_url)

    # Validate status filter if provided.
    from exo.distributed.models import TaskStatus  # pyright: ignore[reportMissingImports]

    status_filter: TaskStatus | None = None
    if status is not None:
        try:
            status_filter = TaskStatus(status)
        except ValueError as err:
            valid = ", ".join(s.value for s in TaskStatus)
            console.print(f"[red]Invalid status: {status}. Valid values: {valid}[/red]")
            raise typer.Exit(code=1) from err

    async def _list() -> None:
        from exo.distributed.store import TaskStore  # pyright: ignore[reportMissingImports]

        store = TaskStore(url)
        await store.connect()
        try:
            results = await store.list_tasks(status=status_filter, limit=limit)
        finally:
            await store.disconnect()

        if not results:
            console.print("[dim]No tasks found.[/dim]")
            return

        table = Table(title="Distributed Tasks")
        table.add_column("Task ID", style="cyan", no_wrap=True)
        table.add_column("Status", no_wrap=True)
        table.add_column("Worker", no_wrap=True)
        table.add_column("Started", no_wrap=True)
        table.add_column("Duration", no_wrap=True)

        for r in results:
            color = _status_color(r.status)
            table.add_row(
                r.task_id,
                f"[{color}]{r.status}[/{color}]",
                r.worker_id or "-",
                _format_timestamp(r.started_at),
                _format_duration(r.started_at, r.completed_at),
            )

        console.print(table)

    asyncio.run(_list())


# ---------------------------------------------------------------------------
# Subcommand group: worker
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Subcommand group: tool (tool offloading)
# ---------------------------------------------------------------------------

from exo_cli.tool_commands import tool_app  # noqa: E402

app.add_typer(tool_app, name="tool")


worker_app = typer.Typer(
    name="worker",
    help="Manage and monitor distributed workers.",
    no_args_is_help=True,
)
app.add_typer(worker_app, name="worker")


@worker_app.command("list")
def worker_list(
    redis_url: Annotated[
        str | None,
        typer.Option("--redis-url", help="Redis connection URL (default: EXO_REDIS_URL env var)."),
    ] = None,
) -> None:
    """List all active distributed workers and their health."""
    logger.debug("CLI command=%s args=%r", "worker list", {})
    url = _resolve_redis_url(redis_url)

    async def _list_workers() -> None:
        from exo.distributed.health import (  # pyright: ignore[reportMissingImports]
            get_worker_fleet_status,
        )

        workers = await get_worker_fleet_status(url)

        if not workers:
            console.print("[dim]No active workers found.[/dim]")
            return

        table = Table(title="Distributed Workers")
        table.add_column("Worker ID", style="cyan", no_wrap=True)
        table.add_column("Status", no_wrap=True)
        table.add_column("Hostname", no_wrap=True)
        table.add_column("Tasks", no_wrap=True)
        table.add_column("Failed", no_wrap=True)
        table.add_column("Current Task", no_wrap=True)
        table.add_column("Concurrency", no_wrap=True)
        table.add_column("Last Heartbeat", no_wrap=True)

        for w in workers:
            status_color = "green" if w.alive else "red"
            status_text = w.status if w.alive else "dead"
            table.add_row(
                w.worker_id,
                f"[{status_color}]{status_text}[/{status_color}]",
                w.hostname,
                str(w.tasks_processed),
                str(w.tasks_failed),
                w.current_task_id or "-",
                str(w.concurrency),
                _format_timestamp(w.last_heartbeat),
            )

        console.print(table)

    asyncio.run(_list_workers())
