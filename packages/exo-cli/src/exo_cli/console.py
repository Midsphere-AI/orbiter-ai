"""Interactive REPL console for Exo CLI.

Provides a read-eval-print loop for chatting with agents, with
slash-command support, streaming output, and Rich formatting.

Usage::

    console = Console(agents={"helper": agent}, run_fn=run)
    await console.start()
"""

from __future__ import annotations

import asyncio
import sys
from collections.abc import AsyncIterator, Callable, Coroutine
from typing import Any, Protocol

from rich.console import Console as RichConsole
from rich.panel import Panel
from rich.table import Table

from exo.types import ExoError  # pyright: ignore[reportMissingImports]

# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class ConsoleError(ExoError):
    """Raised for interactive console configuration errors."""


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


class StreamEvent(Protocol):
    """Minimal protocol for streaming events."""

    @property
    def text(self) -> str: ...


RunFn = Callable[..., Coroutine[Any, Any, Any]]
StreamFn = Callable[..., AsyncIterator[StreamEvent]]

# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

_HELP_TEXT = (
    "[bold]Available commands:[/bold]\n"
    "  /help          Show this help message\n"
    "  /exit, /quit   Exit the console\n"
    "  /agents        List available agents\n"
    "  /switch <name> Switch to a different agent\n"
    "  /clear         Clear the screen\n"
    "  /info          Show current agent info"
)


def parse_command(text: str) -> tuple[str, str]:
    """Parse a slash command from *text*.

    Returns:
        ``(command, argument)`` tuple.  *command* is the slash-command
        name (lowercase, without ``/``) and *argument* is the rest of
        the line.  For non-command input returns ``("", text)``.
    """
    stripped = text.strip()
    if not stripped.startswith("/"):
        return "", stripped
    parts = stripped.split(maxsplit=1)
    cmd = parts[0][1:].lower()  # strip leading '/'
    arg = parts[1] if len(parts) > 1 else ""
    return cmd, arg


def format_agents_table(agents: dict[str, Any]) -> Table:
    """Build a Rich :class:`Table` listing available agents."""
    table = Table(title="Available Agents")
    table.add_column("Name", style="magenta")
    table.add_column("Model", style="cyan")
    table.add_column("Description", style="green")

    for name, agent in agents.items():
        model = getattr(agent, "model", "") or ""
        desc = ""
        if hasattr(agent, "describe"):
            info = agent.describe()
            desc = info.get("instructions", "")[:60] if isinstance(info, dict) else ""
        table.add_row(name, str(model), desc)
    return table


# ---------------------------------------------------------------------------
# Console REPL
# ---------------------------------------------------------------------------


class InteractiveConsole:
    """Interactive REPL for chatting with Exo agents.

    Parameters:
        agents: Mapping of agent name → agent instance.
        run_fn: Async callable ``run(agent, input, **kw) -> result``.
        stream_fn: Optional async-iterator factory for streaming responses.
        console: Optional Rich console (default: stderr for clean piping).
        streaming: Enable streaming output when a stream_fn is provided.
        debug: When ``True``, re-raise exceptions in ``_execute`` so the
            caller (``_cli_run``) can show a full traceback.
    """

    __slots__ = (
        "_agents",
        "_console",
        "_current_name",
        "_debug",
        "_is_tty",
        "_run_fn",
        "_stream_fn",
        "_streaming",
    )

    def __init__(
        self,
        *,
        agents: dict[str, Any],
        run_fn: RunFn,
        stream_fn: StreamFn | None = None,
        console: RichConsole | None = None,
        streaming: bool = False,
        debug: bool = False,
    ) -> None:
        if not agents:
            raise ConsoleError(
                "At least one agent is required",
                hint="Load at least one agent before starting the chat.",
            )
        self._agents = dict(agents)
        self._run_fn = run_fn
        self._stream_fn = stream_fn
        self._console = console or RichConsole(stderr=True)
        self._streaming = streaming and stream_fn is not None
        self._debug = debug
        self._current_name = next(iter(self._agents))
        # Check once at init; stdin tty-ness doesn't change during a session.
        self._is_tty: bool = sys.stdin.isatty()

    # -- properties ----------------------------------------------------------

    @property
    def current_agent_name(self) -> str:
        """Name of the currently selected agent."""
        return self._current_name

    @property
    def current_agent(self) -> Any:
        """The currently selected agent instance."""
        return self._agents[self._current_name]

    @property
    def agents(self) -> dict[str, Any]:
        """Copy of the agents registry."""
        return dict(self._agents)

    # -- command handlers ----------------------------------------------------

    def _handle_help(self) -> None:
        self._console.print(Panel(_HELP_TEXT, title="Help"))

    def _handle_agents(self) -> None:
        self._console.print(format_agents_table(self._agents))

    def _handle_switch(self, arg: str) -> bool:
        """Switch to agent *arg*.  Returns True if switched."""
        name = arg.strip()
        if not name:
            self._console.print("[yellow]Usage: /switch <agent_name>[/yellow]")
            return False
        if name not in self._agents:
            self._console.print(f"[red]Unknown agent: {name}[/red]")
            return False
        self._current_name = name
        self._console.print(f"[green]Switched to agent: [bold]{name}[/bold][/green]")
        return True

    def _handle_info(self) -> None:
        agent = self.current_agent
        self._console.print(f"[bold]Agent:[/bold] {self._current_name}")
        model = getattr(agent, "model", None)
        if model:
            self._console.print(f"[bold]Model:[/bold] {model}")
        if hasattr(agent, "describe"):
            info = agent.describe()
            if isinstance(info, dict):
                for key, val in info.items():
                    if key not in ("name", "model") and val:
                        display = str(val)[:80]
                        self._console.print(f"[bold]{key}:[/bold] {display}")

    def _handle_clear(self) -> None:
        self._console.clear()

    # -- input ---------------------------------------------------------------

    async def _read_input(self) -> str | None:
        """Read one line from stdin.  Returns ``None`` on EOF."""
        if self._is_tty:
            prompt = f"[bold cyan]{self._current_name}[/bold cyan]> "
            self._console.print(prompt, end="")
        try:
            line = await asyncio.to_thread(sys.stdin.readline)
        except (EOFError, KeyboardInterrupt):
            return None
        if not line:  # EOF
            return None
        return line.rstrip("\n")

    # -- execution -----------------------------------------------------------

    async def _execute(self, user_input: str) -> None:
        """Send *user_input* to the current agent and display the result."""
        agent = self.current_agent
        try:
            if self._streaming and self._stream_fn is not None:
                async for event in self._stream_fn(agent, user_input):
                    self._console.print(event.text, end="")
                self._console.print()  # final newline
            else:
                result = await self._run_fn(agent, user_input)
                output = getattr(result, "output", None) or str(result)
                self._console.print(f"[green]{self._current_name}:[/green] {output}")
        except Exception as exc:
            self._console.print(f"[bold red]Error:[/bold red] {exc}")
            if self._debug:
                raise

    # -- main loop -----------------------------------------------------------

    async def start(self) -> None:
        """Run the interactive REPL until exit."""
        self._console.print(
            Panel(
                f"Exo Console — chatting with [bold]{self._current_name}[/bold]\n"
                "Type /help for commands, /exit to quit.",
                style="blue",
            )
        )

        while True:
            raw = await self._read_input()
            if raw is None:
                break
            text = raw.strip()
            if not text:
                continue

            cmd, arg = parse_command(text)

            if cmd in ("exit", "quit"):
                self._console.print("[dim]Goodbye![/dim]")
                break
            if cmd == "help":
                self._handle_help()
                continue
            if cmd == "agents":
                self._handle_agents()
                continue
            if cmd == "switch":
                self._handle_switch(arg)
                continue
            if cmd == "info":
                self._handle_info()
                continue
            if cmd == "clear":
                self._handle_clear()
                continue
            if cmd:
                self._console.print(f"[yellow]Unknown command: /{cmd}[/yellow]")
                continue

            await self._execute(text)
