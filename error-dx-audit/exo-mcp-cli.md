# exo-mcp-cli — Error DX & Resilience Audit

## Counts

- raise sites: 19 (in exo-mcp-cli src; excludes typer.Exit re-raises and vault/transport in exo-mcp)
- error classes total / not inheriting ExoError: 2 / 0
  - `MCPConfigError(ExoError)` — config.py:36 ✓
  - `MCPConnectionError(ExoError)` — connection.py:27 ✓
  - (VaultError and MCPTransportError live in exo-mcp, also inherit ExoError)
- `except Exception` sites: 8 ; swallow-and-pass: 1 (tool.py:54–55) ; drop-cause: 0
- `CancelledError`/`KeyboardInterrupt` handlers: 1 (vault.py:85 in exo-mcp — catches `KeyboardInterrupt` and converts to `VaultError`, safe)
- top-level boundary renders errors cleanly? **partial** — no raw traceback escapes (clean by default), traceback only on `--verbose`; but `except Exception` at async boundaries does NOT unwrap `ExceptionGroup`, so nested asyncio task failures in stdio subprocess launch or SSE/streamable_http connect surface as `"Failed to connect to 'x': ..."` with `str(exc)` of a raw `ExceptionGroup` object as the message body.

---

## Findings (prioritized)

**[P0] checklist#7 | connection.py:122–125 | ExceptionGroup leaks into error message string**
`connect_to_server` catches bare `Exception` and does `str(exc)` to build the `MCPConnectionError` message. When `asyncio.run()` unwinds an `ExceptionGroup` from MCP's internal task machinery (e.g., stdio subprocess failure, SSE timeout), `str(exc)` is the group's repr — a wall of noise injected directly into the "clean" error message.
Fix: call `unwrap_exception_group(exc)` (from `exo._internal.errors`) on `exc` before `str(exc)`. Import it with `# pyright: ignore[reportMissingImports]`.

**[P0] checklist#7 | commands/server.py:191–194, commands/tool.py:125–128, commands/prompt.py:71–74, commands/resource.py:66–69, commands/resource.py:135–138 | ExceptionGroup in fallback `except Exception` handler (5 sites)**
Every command module's outer `except Exception as exc` does `print_error(f"... {exc}")`. If the inner `asyncio.run()` propagates an `ExceptionGroup` that wasn't caught by the `MCPConnectionError` guard, `str(exc)` produces a multi-line group repr that renders as a garbled error. Identical fix: apply `unwrap_exception_group` before printing. Same 5 call sites share the exact same pattern — create a helper `_render_exc(exc)` in `output.py` that calls unwrap and stringifies.

**[P0] checklist#6 | connection.py:122–125 | `CancelledError` swallowed by `except Exception` in async context manager**
`CancelledError` is a subclass of `BaseException` in Python 3.8+ and NOT caught by `except Exception` — this is actually safe. However `asyncio.run()` itself will convert a `CancelledError` inside the async function to a `RuntimeError("Event loop is closed")` which will fall into the `except Exception` fallback in every command and print an opaque message. No specific fix needed beyond the ExceptionGroup handling above, but document this.

**[P1] checklist#2 | vault.py:86 (exo-mcp) | `VaultError("Vault passphrase required")` — missing hint**
When a user hits Ctrl-C at the passphrase prompt, or runs in a non-interactive pipeline, this error surfaces as just "Vault passphrase required" with no actionable hint.
Fix: add `hint="Set EXO_MCP_VAULT_KEY env var to provide the passphrase non-interactively."` to the `VaultError(...)` constructor call.

**[P1] checklist#2 | vault.py:121 (exo-mcp) | `VaultError("Wrong passphrase or corrupted vault")` — missing hint**
A user who mistyped their passphrase or moved the vault file gets a dead-end message.
Fix: add `hint="Re-run with the correct passphrase, or delete ~/.exo-mcp/credentials.vault to start fresh (you will lose stored secrets)."`.

**[P1] checklist#2 | connection.py:123–125 | `MCPConnectionError` lacks context= and hint=**
`"Failed to connect to '{entry.name}' ({entry.transport}): {exc}"` — transport type and name are embedded in the message string rather than in `context=`, and there's no `hint=`.
Fix: move identifiers to `context={"server": entry.name, "transport": entry.transport}`, keep the short message, and add `hint="Check the server is running, the command/URL is correct in mcp.json, and credentials are set."`.

**[P1] checklist#2 | config.py:128,130 | `validate()` MCPConfigErrors lack hint=**
`"Server 'x': stdio transport requires 'command'"` and `"Server 'x': {transport} transport requires 'url'"` tell users what's missing but not how to add it.
Fix: add `hint="Run `exo-mcp server add <name> --command <cmd>` to supply a command."` / `hint="Run `exo-mcp server add <name> --url <url>` to supply a URL."`.

**[P1] checklist#2 | config.py:198 | `MCPConfigError("Config file not found: {p}")` — missing hint**
When an explicit `--config` path is wrong, the user sees the path but no next step.
Fix: add `hint="Check the path or omit --config to use ./mcp.json or ~/.exo-mcp/mcp.json."`.

**[P1] checklist#3 | connection.py:92 | `MCPConnectionError(str(exc))` drops transport context**
In `_create_transport`, `MCPTransportError` is caught and re-raised as `MCPConnectionError(str(exc))`. The server name and transport are available in the calling scope (`_resolve_entry` result) but are not threaded through.
Fix: pass `context={"transport": transport}` (the `transport` string is in scope within `_create_transport`); the caller in `connect_to_server` can `.with_context(server=entry.name)`.

**[P1] checklist#2 | commands/server.py:129 | `except Exception` catches `MCPConfigError` from `entry.validate()`**
`server_add` has `try: entry.validate() / except Exception as exc: print_error(str(exc))`. Because `MCPConfigError` is an `ExoError`, `str(exc)` will render multi-line (with `where:` and `→` lines if they had been set) but the guard is wider than it needs to be — any unexpected error from `ServerEntry.__post_init__` (e.g., programmer error) silently shows as a clean user error instead of bubbling.
Fix: narrow to `except MCPConfigError`.

**[P1] checklist#2 | commands/tool.py:54–55 | Silent swallow of malformed `EXO_MCP_TOOL_INJECT`**
`except json.JSONDecodeError: pass` silently discards a malformed env-var value. A user who fat-fingered `EXO_MCP_TOOL_INJECT` gets no feedback and wonders why injected args are missing.
Fix: emit a `console.print("[yellow]Warning: EXO_MCP_TOOL_INJECT is not valid JSON — ignored.[/yellow]")` instead of a bare `pass`.

**[P1] checklist#2 | main.py:88–90 | `get_server` "not found" error — missing `--config` context**
`"Server 'name' not found. Available: ..."` doesn't mention which config file was searched, so if a user has both `./mcp.json` and `~/.exo-mcp/mcp.json`, they can't tell which one was loaded.
Fix: include the config path in context: `print_error(f"Server '{name}' not found in {path}. Available: {available}")`.

**[P2] checklist#3 | config.py:227 | `MCPConfigError` missing context= for config path**
`"Failed to parse config '{path}': {exc}"` has the path in the message but not in `context=`. Consistent with the charter to carry identifiers in `context=` so structured renderers can extract them.
Fix: add `context={"config": str(path)}` and shorten the message to `f"Failed to parse config: {exc}"`.

**[P2] checklist#9 | vault.py (exo-mcp) | No logging at vault load/decrypt errors**
`VaultError` is raised on `InvalidToken` and JSON decode failures without any `logger.debug(...)` call. A developer debugging a pipeline can't tell if the vault was even reached.
Fix: add `logger.debug("Vault decrypt failed", exc_info=True)` before the `raise VaultError(...)` at lines 121 and 125.

**[P2] checklist#2 | server_remove (server.py:149) | "No config file found." — missing hint**
Fix: add `hint="Create a config with `exo-mcp server add <name> ...` or use --config to point to an existing file."`.

---

## Resilience gaps

| File:Line | System | Gap |
|-----------|--------|-----|
| `connection.py:115–119` | MCP stdio subprocess | `session.initialize()` has no timeout guard beyond the `ServerEntry.timeout` passed to transport — but `ClientSession.initialize()` itself is not covered by that timeout on stdio (only on the transport layer). A hung subprocess blocks indefinitely. Fix: wrap `await session.initialize()` in `asyncio.wait_for(..., timeout=entry.timeout)`. |
| `commands/server.py:176–183` | MCP server test | `session.send_ping()` and `session.list_tools()` are unguarded after connection. If the server becomes unresponsive mid-session, the CLI hangs. Fix: wrap both in `asyncio.wait_for(..., timeout=entry.timeout)`. |
| `config.py:253–254` | File I/O (save_config) | `path.write_text(...)` is unguarded. A permissions error or full disk raises a bare `OSError` that escapes to the top-level typer handler as an unformatted exception. Fix: wrap in `try/except OSError as exc: raise MCPConfigError(f"Failed to write config '{path}': {exc}") from exc`. |
| `exo-mcp/vault.py:138–139` | File I/O (vault save) | `self._path.write_bytes(...)` is similarly unguarded; a vault save failure (disk full, permissions) raises bare `OSError`. Fix: wrap and re-raise as `VaultError`. |
| `commands/resource.py:104,107,120,123` | File I/O (resource write-to-file) | `out_path.write_text/write_bytes(...)` unguarded — permissions error or full disk raises bare `OSError` inside the async run, falls to `except Exception` with message `"Failed to read resource: [Errno 13] Permission denied"`. Fix: catch `OSError` around each write and raise a clean `MCPConnectionError` or just `print_error`. |

---

## Effort estimate

**S** — All errors already inherit `ExoError`, no raw tracebacks escape to users (boundary pattern is consistently applied), and the taxonomy is clean. The work is almost entirely adding `hint=`/`context=` to ~10 raise sites, one `unwrap_exception_group` fix, one swallow-to-warning conversion, and 3–4 I/O guard wrappers. Estimated 2–3 hours.
