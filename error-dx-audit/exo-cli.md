# exo-cli — Error DX & Resilience Audit

## Counts
- raise sites: 51 (24 ExoError-subclass raises, 19 `typer.Exit`/`typer.BadParameter` CLI signals, 1 raw `ValueError`, 7 re-raises with chaining)
- error classes total / not inheriting ExoError: 4 total (`CLIError`, `ExecutorError`, `AgentLoadError`, `BatchError`) / **1 offender** — `console.py:121` `raise ValueError("At least one agent is required")` (plain `ValueError`, not an `ExoError` subclass)
- `except Exception` sites: 9 ; swallow-and-pass: 1 (`tool_commands.py:169` — `EXO_TOOL_INJECT` malformed JSON silently ignored) ; drop-cause: 0
- CancelledError/KeyboardInterrupt handlers: 1 (`console.py:193` catches `KeyboardInterrupt` in `_read_input`, returns `None` → clean loop exit). No explicit `CancelledError` handler at any `asyncio.run()` call site.
- top-level boundary renders ExoError cleanly? **partial** — only the `run` command has a try/except around `asyncio.run()`; `chat`, `batch`, `start worker`, `task status/cancel/list`, and `worker list` all call `asyncio.run()` bare with no surrounding error handler.

---

## Findings (prioritized)

### P0 — Raw tracebacks / ugly failures

**[P0] | checklist#1,7 | `main.py:277` (`chat`), `main.py:369` (`batch`), `main.py:450` (`start worker`), `main.py:559` (`task status`), `main.py:589` (`task cancel`), `main.py:659` (`task list`), `main.py:731` (`worker list`)**
All seven of these `asyncio.run()` call sites have **no surrounding try/except**. Any `ExoError`, `ExceptionGroup`, connection error, or `KeyboardInterrupt` from the async internals will produce a raw Python traceback dumped directly to the terminal. The `run` command (line 210) is the only command that has a boundary. The remaining six commands are unguarded.
**Fix:** Wrap each bare `asyncio.run(...)` in a shared helper, e.g.:
```python
def _cli_run(coro, *, verbose: bool = False) -> None:
    try:
        asyncio.run(coro)
    except KeyboardInterrupt:
        console.print("\n[dim]Interrupted.[/dim]")
        raise typer.Exit(code=130)
    except ExoError as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        raise typer.Exit(code=1)
    except BaseExceptionGroup as eg:
        real = unwrap_exception_group(eg)
        console.print(f"[bold red]Error:[/bold red] {real}")
        if verbose:
            raise
        raise typer.Exit(code=1)
```
This is the single highest-value fix.

**[P0] | checklist#7 | `executor.py:192` (`execute`)**
`except Exception as exc: raise ExecutorError(f"Agent execution failed: {exc}") from exc` — catches the outer exception but does **not** call `unwrap_exception_group` before stringifying. An `ExceptionGroup` from `asyncio.gather` inside the runner will stringify as `ExceptionGroup('', [TimeoutError(...)])` in the error message. The real cause is there on `__cause__` but the message shown to the user is confusing.
**Fix:** call `unwrap_exception_group(exc)` before building the `ExecutorError` message:
```python
from exo._internal.errors import unwrap_exception_group
real = unwrap_exception_group(exc)
raise ExecutorError(f"Agent execution failed: {real}", ...) from real
```

**[P0] | checklist#6 | `main.py:277` (chat `asyncio.run(repl.start())`)**
Ctrl-C during an interactive chat session produces a raw Python traceback because `asyncio.run()` re-raises `KeyboardInterrupt` and there is no surrounding handler. `console.py:193` handles Ctrl-C inside `_read_input` (in `asyncio.to_thread`), but a Ctrl-C during an active LLM call (while `_execute` is awaiting) propagates up through `asyncio.run()` unhandled.
**Fix:** covered by the `_cli_run` helper above (catch `KeyboardInterrupt`, print "Interrupted.", exit 130).

---

### P1 — Unactionable messages / missing context / no --debug toggle

**[P1] | checklist#2,3 | `executor.py:191,193` | `ExecutorError` messages lack `context=` and `hint=`**
`"Execution timed out after 3.0s"` and `"Agent execution failed: <stringified exc>"` contain no `context=` (which agent? which model?) and no `hint=` (what should the developer do?). `ExecutorError` is an `ExoError` subclass so these fields are free.
**Fix:**
```python
raise ExecutorError(
    f"Execution timed out after {self._timeout:.1f}s",
    context={"agent": getattr(self._agent, "name", "?"), "timeout": self._timeout},
    hint="Increase the --timeout flag or simplify the input.",
) from exc
```

**[P1] | checklist#2,3 | `executor.py:249` | streaming `ExecutorError` also lacks context/hint**
`"Streaming failed: {exc}"` — same pattern as above.
**Fix:** add `context={"agent": name}` and `hint="Check that the model supports streaming and that your API key is set."`.

**[P1] | checklist#3 | `main.py:77` (`load_config`) | `CLIError` lacks context/hint**
`"Config file not found: {p}"` — no hint that explains how to supply one.
**Fix:** add `hint="Use --config to specify a path, or create .exo.yaml in the current directory."`.

**[P1] | checklist#3 | `main.py:84` (`load_config`) | `CLIError` "Invalid config" lacks field name**
`"Invalid config: {exc}"` — the underlying YAML error string from `LoaderError` may be adequate, but no `context={"path": str(p)}` is attached.

**[P1] | checklist#8 | `executor.py:262-264` (`print_error`) | does not render ExoError teaching block**
`print_error` does `f"[bold red]Error:[/bold red] {error}"` — calling `str()` on an `ExoError` does include `where:` and `→ hint` lines (because `ExoError.__str__` formats them), BUT only if those fields were populated. Since they currently are not (see P1 finding above), the teaching block is never shown. Once context/hint are added, this method will work correctly — no separate fix needed, but it's blocked on the P1 findings above.

**[P1] | checklist#2 | `console.py:213-214` (`_execute`) | catches any `Exception` and prints `str(exc)` with no `hint=` or `where=`**
During interactive chat, errors are printed as `Error: <str(exc)>`. If the agent raises an `ExoError` with context and hint populated, this renders correctly (via `ExoError.__str__`). But if it raises a plain `Exception` or `ExceptionGroup`, the user gets an opaque message with no guidance. No `--debug` flag is threaded into `InteractiveConsole` to optionally show a traceback.
**Fix:** add a `debug: bool = False` parameter to `InteractiveConsole`; re-raise in debug mode, otherwise print the structured message.

**[P1] | checklist#2 | No `--debug` flag propagated to any async boundary**
The `--verbose`/`-v` flag is defined in `main.py:main()` and passed to `LocalExecutor`, but it only controls timing/usage output — not whether a full traceback is shown on error. There is no `--debug` mode that reveals the underlying traceback. Developers cannot inspect async root causes without patching the source.
**Fix:** Honour `verbose` in the `_cli_run` helper: `if verbose: raise` after printing the clean message.

**[P1] | checklist#4 | `batch.py:256-261` (`_run_one`) | `except Exception` converts error to string, drops structured cause**
Item-level failures are caught and stored as `str(exc)` in `ItemResult.error`. The `ExoError.__str__` formatting will be used so hint/context survive stringify. However, the structured cause (`__cause__`) is silently dropped and there is no way to surface it from the `BatchResult` output.
**Fix:** store the exception type and consider a `traceback` field (debug-only); at minimum log the original at `DEBUG`.

**[P1] | checklist#3 | `loader.py:54,67` | `AgentLoadError` messages lack context/hint**
`"Cannot create module spec for {path}"` and `"No create_agent() function in {path}"` — both would benefit from `hint=`. E.g. for the second: `hint="Define a top-level create_agent() -> Agent function in the module."`.

---

### P2 — Polish

**[P2] | checklist#1 | `console.py:121` | `raise ValueError("At least one agent is required")`**
`InteractiveConsole.__init__` raises a plain `ValueError`. This crosses the package boundary (the REPL is constructed in `main.py:chat`). Should be `CLIError` or a new `ConsoleError(ExoError)` with `hint="Load at least one agent before starting the chat."`.

**[P2] | checklist#5 | `loader.py:67` | missing `from` on chained raise**
`raise AgentLoadError(f"No create_agent() function in {path}")` — no `from exc`; but there is no prior exception at this point, so chaining is not applicable. Already OK.

**[P2] | checklist#4 | `tool_commands.py:169` | silent ignore of malformed `EXO_TOOL_INJECT`**
`except json.JSONDecodeError: pass  # Silently ignore malformed env var` — developer sets `EXO_TOOL_INJECT=not-json` and gets no feedback, then their injected args are silently absent. At minimum, log a `logger.warning`; better: print a `[yellow]Warning: EXO_TOOL_INJECT is not valid JSON; ignoring.[/yellow]`.

**[P2] | checklist#2,3 | `batch.py:241` | `BatchError("concurrency must be >= 1")` lacks hint**
Add `hint="Pass --concurrency with a value of 1 or greater."`.

**[P2] | checklist#2 | `main.py:427-431` | `start worker` Redis URL error message is plain red, not ExoError**
`console.print("[red]Error: ...[/red]")` is fine for CLI display, but the pattern is inconsistent — some paths print and exit, others raise `CLIError`. Consider `_cli_exit_error()` helper for uniformity.

**[P2] | checklist#9 | logging pattern**
`main.py` and all other `exo_cli` files use `import logging; logger = logging.getLogger(__name__)` — correct for a non-`_internal/` package. Consistent throughout.

---

## Resilience gaps

- **`executor.py:187` — no timeout on streaming** (`stream()` method has no `asyncio.wait_for`; only `execute()` does). A stalled LLM stream will hang the CLI indefinitely with no recourse.
  File: `packages/exo-cli/src/exo_cli/executor.py:221-249`.
- **`main.py:527-534` / `task_status` and `task_cancel`** — `TaskStore.connect()` and `TaskBroker.connect()` have no timeout; a Redis connection hang will freeze the CLI with no feedback. Add `asyncio.wait_for(..., timeout=10)` around the connect calls or rely on the underlying client timeout.
- **`main.py:450` (`start worker`)** — `asyncio.run(worker.start())` is the long-running process entry; Ctrl-C will dump a KeyboardInterrupt traceback instead of a clean "Worker stopped." banner.
- **`batch.py:265` — `asyncio.gather(*tasks)` with `return_exceptions=False` (default)** — if multiple items somehow raise outside `_run_one`'s try/except (e.g., a `CancelledError`), gather will raise an `ExceptionGroup` that propagates to `asyncio.run(_run_batch())` unhandled.

---

## Effort estimate

**M** — The highest-value fix (a shared `_cli_run` helper wrapping the 7 unguarded `asyncio.run()` calls, plus `unwrap_exception_group` in `execute()`) is a focused ~80-line change; adding `context=`/`hint=` to the remaining ~12 raise sites and wiring `--verbose` as a debug toggle adds another ~60 lines. Total: roughly one focused session.
