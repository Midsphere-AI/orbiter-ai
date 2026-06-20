"""Local agent executor for Exo CLI.

Wraps :func:`exo.runner.run` and :func:`exo.runner.run.stream`
with Rich output formatting, error handling, timeout support, and
provider resolution from CLI configuration.

Usage::

    executor = LocalExecutor(agent=agent)
    result = await executor.execute("Hello!")

    # Streaming:
    async for chunk in executor.stream("Hello!"):
        print(chunk)
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Sequence
from typing import Any

from rich.console import Console as RichConsole
from rich.panel import Panel

from exo.types import ExoError  # pyright: ignore[reportMissingImports]

# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class ExecutorError(ExoError):
    """Raised for execution-level errors."""


# ---------------------------------------------------------------------------
# Result wrapper
# ---------------------------------------------------------------------------


class ExecutionResult:
    """Wraps a run result with CLI-friendly accessors.

    Parameters:
        output: Agent output text.
        steps: Number of LLM call steps.
        elapsed: Wall-clock seconds.
        usage: Token usage dict (prompt_tokens, output_tokens, total_tokens).
        raw: The underlying ``RunResult`` object (if available).
    """

    __slots__ = ("_elapsed", "_output", "_raw", "_steps", "_usage")

    def __init__(
        self,
        *,
        output: str,
        steps: int = 0,
        elapsed: float = 0.0,
        usage: dict[str, int] | None = None,
        raw: Any = None,
    ) -> None:
        self._output = output
        self._steps = steps
        self._elapsed = elapsed
        self._usage = usage or {}
        self._raw = raw

    @property
    def output(self) -> str:
        return self._output

    @property
    def steps(self) -> int:
        return self._steps

    @property
    def elapsed(self) -> float:
        return self._elapsed

    @property
    def usage(self) -> dict[str, int]:
        return dict(self._usage)

    @property
    def raw(self) -> Any:
        return self._raw

    def summary(self) -> str:
        """Human-readable one-line summary."""
        parts = [f"{self._steps} step(s)"]
        if self._elapsed > 0:
            parts.append(f"{self._elapsed:.1f}s")
        total = self._usage.get("total_tokens", 0)
        if total:
            parts.append(f"{total} tokens")
        return ", ".join(parts)

    def __repr__(self) -> str:
        return f"ExecutionResult(output={self._output!r}, steps={self._steps})"


# ---------------------------------------------------------------------------
# Local executor
# ---------------------------------------------------------------------------


class LocalExecutor:
    """Executes agents locally via :func:`exo.runner.run`.

    Parameters:
        agent: An ``Agent`` (or ``Swarm``) instance.
        provider: LLM provider.  When ``None``, auto-resolved.
        timeout: Per-execution timeout in seconds (0 = no timeout).
        max_retries: Retry attempts for transient LLM errors.
        console: Rich console for formatted output (default: stderr).
        verbose: When ``True``, print timing and usage details.
    """

    __slots__ = (
        "_agent",
        "_console",
        "_max_retries",
        "_provider",
        "_timeout",
        "_verbose",
    )

    def __init__(
        self,
        *,
        agent: Any,
        provider: Any = None,
        timeout: float = 0.0,
        max_retries: int = 3,
        console: RichConsole | None = None,
        verbose: bool = False,
    ) -> None:
        self._agent = agent
        self._provider = provider
        self._timeout = timeout
        self._max_retries = max_retries
        self._console = console or RichConsole(stderr=True)
        self._verbose = verbose

    # -- properties ----------------------------------------------------------

    @property
    def agent(self) -> Any:
        return self._agent

    @property
    def timeout(self) -> float:
        return self._timeout

    @property
    def verbose(self) -> bool:
        return self._verbose

    # -- execute (non-streaming) ---------------------------------------------

    async def execute(
        self,
        input: str,
        *,
        messages: Sequence[Any] | None = None,
    ) -> ExecutionResult:
        """Run the agent and return an :class:`ExecutionResult`.

        Raises:
            ExecutorError: On timeout or agent failure.
        """
        from exo.runner import run  # lazy import

        t0 = time.monotonic()
        try:
            coro = run(
                self._agent,
                input,
                messages=messages,
                provider=self._provider,
                max_retries=self._max_retries,
            )
            if self._timeout > 0:
                raw = await asyncio.wait_for(coro, timeout=self._timeout)
            else:
                raw = await coro
        except TimeoutError as exc:
            raise ExecutorError(f"Execution timed out after {self._timeout:.1f}s") from exc
        except Exception as exc:
            raise ExecutorError(f"Agent execution failed: {exc}") from exc
        elapsed = time.monotonic() - t0

        # Extract usage from RunResult
        usage_dict: dict[str, int] = {}
        usage_obj = getattr(raw, "usage", None)
        if usage_obj is not None:
            usage_dict = {
                "prompt_tokens": getattr(usage_obj, "prompt_tokens", 0) or 0,
                "output_tokens": getattr(usage_obj, "output_tokens", 0) or 0,
                "total_tokens": getattr(usage_obj, "total_tokens", 0) or 0,
            }

        result = ExecutionResult(
            output=getattr(raw, "output", "") or "",
            steps=getattr(raw, "steps", 0) or 0,
            elapsed=elapsed,
            usage=usage_dict,
            raw=raw,
        )

        if self._verbose:
            self._console.print(f"[dim]{result.summary()}[/dim]")

        return result

    # -- stream --------------------------------------------------------------

    async def stream(
        self,
        input: str,
        *,
        messages: Sequence[Any] | None = None,
    ) -> AsyncIterator[str]:
        """Stream agent output, yielding text chunks.

        Raises:
            ExecutorError: On failure during streaming.
        """
        from exo.runner import run as run_fn  # lazy import

        stream_fn = getattr(run_fn, "stream", None)
        if stream_fn is None:
            raise ExecutorError("Streaming not available (run.stream not found)")

        try:
            async for event in stream_fn(
                self._agent,
                input,
                messages=messages,
                provider=self._provider,
            ):
                text = getattr(event, "text", None)
                if text:
                    yield text
        except Exception as exc:
            raise ExecutorError(f"Streaming failed: {exc}") from exc

    # -- display helpers -----------------------------------------------------

    def print_result(self, result: ExecutionResult) -> None:
        """Pretty-print an execution result to the console."""
        name = getattr(self._agent, "name", "agent")
        self._console.print(
            Panel(result.output, title=f"[bold]{name}[/bold]", border_style="green")
        )
        if self._verbose:
            self._console.print(f"[dim]{result.summary()}[/dim]")

    def print_error(self, error: Exception) -> None:
        """Display an error to the console."""
        self._console.print(f"[bold red]Error:[/bold red] {error}")

    def __repr__(self) -> str:
        name = getattr(self._agent, "name", "?")
        return f"LocalExecutor(agent={name!r}, timeout={self._timeout})"
