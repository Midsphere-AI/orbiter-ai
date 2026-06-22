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
    exo chat --config agents.yaml
    exo batch --config agents.yaml inputs.jsonl
    exo start worker --redis-url redis://localhost:6379
    exo task list --status running
    exo task status <task_id>
    exo task cancel <task_id>
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Coroutine
from pathlib import Path
from typing import Annotated, Any
from urllib.parse import urlparse

import typer
from rich.console import Console
from rich.table import Table

from exo._internal.errors import unwrap_exception_group  # pyright: ignore[reportMissingImports]
from exo.types import ExoError  # pyright: ignore[reportMissingImports]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared async boundary
# ---------------------------------------------------------------------------


def _cli_run(coro: Coroutine[Any, Any, Any], *, verbose: bool = False) -> None:
    """Run *coro* under ``asyncio.run`` with a clean error boundary.

    Handles:
    - ``KeyboardInterrupt`` → prints "Interrupted." and exits 130.
    - ``ExoError`` → prints the structured teaching block and exits 1.
    - ``BaseExceptionGroup`` → unwraps to the real cause, prints it,
      and exits 1 (re-raises the full group when *verbose* is True).
    """
    try:
        asyncio.run(coro)
    except KeyboardInterrupt:
        console.print("\n[dim]Interrupted.[/dim]")
        raise typer.Exit(code=130) from None
    except ExoError as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc
    except BaseExceptionGroup as eg:
        real = unwrap_exception_group(eg)
        console.print(f"[bold red]Error:[/bold red] {real}")
        if verbose:
            raise
        raise typer.Exit(code=1) from real


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
        raise CLIError(
            f"Config file not found: {p}",
            hint="Use --config to specify a path, or create .exo.yaml in the current directory.",
        )

    from exo.loader import LoaderError, load_yaml  # lazy import

    try:
        data = load_yaml(p)
    except LoaderError as exc:
        raise CLIError(
            f"Invalid config: {exc}",
            context={"path": str(p)},
            hint="Check that the file is valid YAML and contains a top-level 'agents' key.",
        ) from exc
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

    # Resolve config path once; reuse for both metadata display and agent loading.
    config_path = config or str(find_config() or "")
    if not config_path:
        console.print("[yellow]No config file found. Use --config or create .exo.yaml[/yellow]")
        raise typer.Exit(code=1)

    cfg = load_config(config_path)
    if verbose:
        console.print(f"[dim]Loaded config with keys: {list(cfg.keys())}[/dim]")
        console.print(f"[dim]Model: {model or 'auto'}[/dim]")
        console.print(f"[dim]Streaming: {stream}[/dim]")

    console.print(f"[green]Running with input:[/green] {input_text}")

    # Load agents from config (same file, already validated above)
    from exo_cli.loader import AgentLoadError, load_yaml_agents  # lazy import

    try:
        agents = load_yaml_agents(Path(config_path))
    except AgentLoadError as exc:
        console.print(f"[bold red]Error:[/bold red] loading agents: {exc}")
        raise typer.Exit(code=1) from exc

    if not agents:
        console.print("[bold red]Error:[/bold red] no agents defined in config.")
        raise typer.Exit(code=1)

    # Pick the first agent (config determines which agent to run)
    agent = next(iter(agents.values()))

    # Override model if --model flag provided
    if model:
        agent.model = model

    from exo_cli.executor import ExecutionResult, LocalExecutor  # lazy import

    # Pass main.console so primary output goes to stdout; executor diagnostics
    # also land on stdout for the run command (streaming already does this).
    executor = LocalExecutor(agent=agent, verbose=verbose, console=console)

    async def _run() -> None:
        if stream:
            async for chunk in executor.stream(input_text):
                console.print(chunk, end="")
            console.print()  # final newline
        else:
            result: ExecutionResult = await executor.execute(input_text)
            executor.print_result(result)

    _cli_run(_run(), verbose=verbose)


# ---------------------------------------------------------------------------
# chat command
# ---------------------------------------------------------------------------


@app.command()
def chat(
    ctx: typer.Context,
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
    """Start an interactive chat session with an agent."""
    verbose: bool = ctx.obj.get("verbose", False)

    # Resolve config path once; reuse for both metadata display and agent loading.
    config_path = config or str(find_config() or "")
    if not config_path:
        console.print("[yellow]No config file found. Use --config or create .exo.yaml[/yellow]")
        raise typer.Exit(code=1)

    cfg = load_config(config_path)
    if verbose:
        console.print(f"[dim]Loaded config with keys: {list(cfg.keys())}[/dim]")

    from exo_cli.loader import AgentLoadError, load_yaml_agents  # lazy import

    try:
        agents = load_yaml_agents(Path(config_path))
    except AgentLoadError as exc:
        console.print(f"[bold red]Error:[/bold red] loading agents: {exc}")
        raise typer.Exit(code=1) from exc

    if not agents:
        console.print("[bold red]Error:[/bold red] no agents defined in config.")
        raise typer.Exit(code=1)

    if model:
        for agent in agents.values():
            agent.model = model

    from exo.runner import run as run_fn  # pyright: ignore[reportMissingImports]
    from exo_cli.console import InteractiveConsole  # lazy import

    stream_fn = getattr(run_fn, "stream", None) if stream else None
    repl = InteractiveConsole(
        agents=agents,
        run_fn=run_fn,
        stream_fn=stream_fn,
        streaming=stream,
        debug=verbose,
    )

    _cli_run(repl.start(), verbose=verbose)


# ---------------------------------------------------------------------------
# batch command
# ---------------------------------------------------------------------------


@app.command()
def batch(
    ctx: typer.Context,
    inputs_file: Annotated[
        str,
        typer.Argument(help="Path to inputs file (.json, .jsonl, or .csv)."),
    ],
    config: Annotated[
        str | None,
        typer.Option("--config", "-c", help="Path to YAML config file."),
    ] = None,
    model: Annotated[
        str | None,
        typer.Option("--model", "-m", help="Model string (e.g. openai:gpt-4o)."),
    ] = None,
    concurrency: Annotated[
        int,
        typer.Option("--concurrency", "-n", help="Maximum concurrent executions."),
    ] = 4,
    output_format: Annotated[
        str,
        typer.Option("--output-format", "-f", help="Output format: jsonl or csv."),
    ] = "jsonl",
    timeout: Annotated[
        float,
        typer.Option("--timeout", "-t", help="Per-item timeout in seconds (0 = no timeout)."),
    ] = 0.0,
) -> None:
    """Run an agent against a batch of inputs from a file."""
    verbose: bool = ctx.obj.get("verbose", False)

    # Resolve config path once; reuse for both metadata display and agent loading.
    config_path = config or str(find_config() or "")
    if not config_path:
        console.print("[yellow]No config file found. Use --config or create .exo.yaml[/yellow]")
        raise typer.Exit(code=1)

    cfg = load_config(config_path)
    if verbose:
        console.print(f"[dim]Loaded config with keys: {list(cfg.keys())}[/dim]")

    from exo_cli.batch import (  # lazy import
        BatchError,
        batch_execute,
        load_batch_items,
        results_to_csv,
        results_to_jsonl,
    )
    from exo_cli.loader import AgentLoadError, load_yaml_agents  # lazy import

    try:
        agents = load_yaml_agents(Path(config_path))
    except AgentLoadError as exc:
        console.print(f"[bold red]Error:[/bold red] loading agents: {exc}")
        raise typer.Exit(code=1) from exc

    if not agents:
        console.print("[bold red]Error:[/bold red] no agents defined in config.")
        raise typer.Exit(code=1)

    agent = next(iter(agents.values()))
    if model:
        agent.model = model

    try:
        items = load_batch_items(inputs_file)
    except BatchError as exc:
        console.print(f"[bold red]Error:[/bold red] loading inputs: {exc}")
        raise typer.Exit(code=1) from exc

    console.print(f"[green]Running batch:[/green] {len(items)} items, concurrency={concurrency}")

    async def _run_batch() -> None:
        result = await batch_execute(
            agent,
            items,
            concurrency=concurrency,
            timeout=timeout,
        )
        console.print(f"[green]Done:[/green] {result.summary()}")
        if output_format == "csv":
            console.print(results_to_csv(result))
        else:
            console.print(results_to_jsonl(result))

    _cli_run(_run_batch(), verbose=verbose)


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
            "[bold red]Error:[/bold red] --redis-url required or set EXO_REDIS_URL environment variable."
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

    _cli_run(worker.start())


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
            "[bold red]Error:[/bold red] --redis-url required or set EXO_REDIS_URL environment variable."
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

    _cli_run(_show())


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

    _cli_run(_cancel())


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
            console.print(
                f"[bold red]Error:[/bold red] invalid status: {status}. Valid values: {valid}"
            )
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

    _cli_run(_list())


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

    _cli_run(_list_workers())


@worker_app.command("status")
def worker_status(
    worker_id: Annotated[
        str,
        typer.Argument(help="Worker ID to inspect."),
    ],
    redis_url: Annotated[
        str | None,
        typer.Option("--redis-url", help="Redis connection URL (default: EXO_REDIS_URL env var)."),
    ] = None,
) -> None:
    """Show detailed health status for a specific worker."""
    logger.debug("CLI command=%s args=%r", "worker status", {"worker_id": worker_id})
    url = _resolve_redis_url(redis_url)

    async def _show_worker() -> None:
        from exo.distributed.health import (  # pyright: ignore[reportMissingImports]
            get_worker_fleet_status,
        )

        workers = await get_worker_fleet_status(url)
        match = next((w for w in workers if w.worker_id == worker_id), None)
        if match is None:
            console.print(f"[yellow]Worker not found: {worker_id}[/yellow]")
            raise typer.Exit(code=1)

        status_color = "green" if match.alive else "red"
        status_text = match.status if match.alive else "dead"
        console.print(f"[bold]Worker {match.worker_id}[/bold]")
        console.print(f"  Status:        [{status_color}]{status_text}[/{status_color}]")
        console.print(f"  Hostname:      {match.hostname}")
        console.print(f"  Concurrency:   {match.concurrency}")
        console.print(f"  Tasks done:    {match.tasks_processed}")
        console.print(f"  Tasks failed:  {match.tasks_failed}")
        console.print(f"  Current task:  {match.current_task_id or '-'}")
        console.print(f"  Last heartbeat: {_format_timestamp(match.last_heartbeat)}")

    _cli_run(_show_worker())


@worker_app.command("stop")
def worker_stop(
    worker_id: Annotated[
        str,
        typer.Argument(help="Worker ID to stop gracefully."),
    ],
    redis_url: Annotated[
        str | None,
        typer.Option("--redis-url", help="Redis connection URL (default: EXO_REDIS_URL env var)."),
    ] = None,
) -> None:
    """Send a graceful stop signal to a worker.

    The worker will finish its current task and then exit.
    Use ``exo start worker`` to start a new one.
    """
    logger.debug("CLI command=%s args=%r", "worker stop", {"worker_id": worker_id})
    url = _resolve_redis_url(redis_url)

    async def _stop_worker() -> None:
        from exo.distributed.broker import TaskBroker  # pyright: ignore[reportMissingImports]

        broker = TaskBroker(url)
        await broker.connect()
        try:
            await broker.stop_worker(worker_id)
        finally:
            await broker.disconnect()

        console.print(f"[green]Stop signal sent to worker {worker_id}.[/green]")
        console.print("[dim]The worker will finish its current task and then exit.[/dim]")

    _cli_run(_stop_worker())
