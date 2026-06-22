# exo-mcp — Error DX & Resilience Audit

## Counts
- raise sites: 29
- error classes total / not inheriting ExoError: 5 / 0
  - `MCPClientError(ExoError)` — client.py:22
  - `MCPExecutionError(MCPClientError)` — execution.py:22
  - `MCPTransportError(ExoError)` — transport.py:20
  - `MCPToolError(ToolError)` — tools.py:20 (ToolError → ExoError, so fully in-tree)
  - `MCPServerError(ExoError)` — server.py:17
  - `VaultError(ExoError)` — vault.py:34
  - *(6 classes total; all descend from ExoError)*
- `except Exception` sites: 7 ; swallow-and-pass: 1 (`server.py:249–250`) ; drop-cause: 1 (`tools.py:397`)
- CancelledError handlers: 2 — server.py:223 (correct, re-raises) and server.py:249 (swallowed in shutdown `pass` block — acceptable)
- I/O call sites lacking asyncio-level timeout: 4 (`session.initialize()`, `session.list_tools()`, `session.call_tool()`, lazy-reconnect `conn.connect()` in tools.py)

---

## Findings (prioritized)

| # | Priority | Checklist | File:Line | What's wrong | Concrete fix |
|---|----------|-----------|-----------|--------------|--------------|
| 1 | P0 | 5 | execution.py:112 | `MCPTransport(cfg.get("transport", "stdio"))` raises a raw `ValueError` when the JSON contains an unknown transport string (e.g. `"grpc"`). This `ValueError` is not caught by the surrounding `except (json.JSONDecodeError, OSError)` block, so it escapes as a bare Python exception with no server name, no file path, and no hint. | Wrap the `MCPTransport(...)` call in a `try/except ValueError` and re-raise as `MCPExecutionError(f"Server '{name}': unknown transport {cfg.get('transport')!r} — valid values: stdio, sse, streamable_http", hint="Check the 'transport' field in mcp.json.") from exc` |
| 2 | P0 | 8 | client.py:309 | `session.initialize()` has no asyncio-level timeout guard. If an MCP server starts but hangs during the MCP handshake, this call will block forever. The `session_timeout` param is passed to `ClientSession` as an *idle* read timeout, not a handshake timeout — it does not fire until data starts flowing. | Wrap with `asyncio.wait_for(session.initialize(), timeout=self._config.timeout)` and map `TimeoutError` → `MCPClientError(..., hint="The server started but didn't complete the MCP handshake within {timeout}s — verify the command is a valid MCP server.")` |
| 3 | P0 | 8 | client.py:344 | `session.list_tools()` has no asyncio-level timeout. A hung server will block the agent's tool-loading phase indefinitely. | Wrap with `asyncio.wait_for(..., timeout=self._config.session_timeout or 120)` and raise `MCPClientError` with context `{"server": self.name}` and a hint to increase `session_timeout`. |
| 4 | P0 | 8 | client.py:370–371 | `session.call_tool(...)` has no asyncio-level timeout. Hangs here block the entire agent turn. During distributed/worker execution this is especially harmful. | Same pattern: `asyncio.wait_for(self._session.call_tool(...), timeout=self._config.session_timeout or 120)` → `MCPClientError` with `context={"server": self.name, "tool": tool_name}`. |
| 5 | P0 | 8 | tools.py:317–324 | Lazy reconnect `conn.connect()` has no asyncio-level timeout. A hung server during distributed-worker reconnect can stall a worker task indefinitely with no timeout or cleanup guarantee beyond the outer exception handler. | Add `asyncio.wait_for(conn.connect(), timeout=self._server_config.timeout)` inside the reconnect block; map timeout → `MCPToolError(..., context={"server": self._server_name, "tool": self._original_name}, hint="Check the MCP server is reachable from the worker node.")`. |
| 6 | P0 | 8 | client.py:457–459 | `connect_all()` is purely serial. One slow-starting server (e.g. stdio process that takes 3s to boot) blocks all subsequent servers. On failure the error surface is a single bare `MCPClientError` with no indication of which servers succeeded. | Use `asyncio.gather(*[self.connect(n) for n in self._configs], return_exceptions=True)`, collect failures, re-raise as a single `MCPClientError` listing which servers failed and which succeeded. |
| 7 | P1 | 2, 3 | client.py:317 | `"Failed to connect to server '{self.name}': {exc}"` embeds `str(exc)` — which may be a raw mcp-library or asyncio message — but gives no hint. For stdio transport the most common cause is "command not found". | Add `hint=f"For stdio transport, verify '{self._config.command}' is installed and on PATH. For HTTP transport, check the URL is reachable."` and `context={"server": self.name, "transport": str(self._config.transport)}`. |
| 8 | P1 | 2, 3 | transport.py:107 | `MCPTransportError(f"Unsupported transport: {transport!r}")` names the bad value but omits the valid choices. | Add `hint=f"Valid transports are: stdio, sse, streamable_http. Got {transport!r}."` |
| 9 | P1 | 2 | execution.py:97 | `"MCP config file not found: {path}"` — no hint to guide the developer. | Add `hint=f"Create {path} or pass the correct path to load_mcp_config(). See docs for the expected mcp.json format."` |
| 10 | P1 | 2, 3 | tools.py:320–322 | Lazy-reconnect error `"MCP server reconnection failed for server '{self._server_name}'"` drops the underlying exception message. The cause is chained (`from exc`) but the top-level message is sparse. | Add `context={"server": self._server_name, "tool": self._original_name}` and `hint="Check the server command is on PATH and the server_config is valid."` |
| 11 | P1 | 2 | vault.py:114 | `"Vault file is corrupted: {self._path}"` — no hint on what to do. | Add `hint=f"Delete {self._path} and re-add secrets with vault.set(), or restore a backup."` |
| 12 | P1 | 2 | vault.py:121 | `"Wrong passphrase or corrupted vault"` — no hint. | Add `hint="Re-run with the correct passphrase (EXO_MCP_VAULT_KEY env var) or delete the vault file to start over."` |
| 13 | P1 | 2, 3 | server.py:49 | `"MCP server '{name}' not registered"` — no list of registered servers and no hint. | Add `hint=f"Registered servers: {sorted(self._classes)}. Use @mcp_server() to register a class."` |
| 14 | P1 | 2 | server.py:192, 208 | `"MCP server not initialized"` — doesn't name the server, no hint. | Include `server_name` in message; add `hint="The @mcp_server decorator must be applied and __init__ must call the decorated __init__ before run() or run_async() is called."` |
| 15 | P1 | 2 | server.py:216 | `"No async runner for transport {transport!r}"` — no hint, no valid alternatives. | Add `hint="Valid async transports for FastMCP are: stdio, sse. Check the transport argument passed to run_async()."` |
| 16 | P1 | 5 | execution.py:107 | `raise MCPExecutionError(f"Expected 'mcpServers' to be a dict in '{path}'")` — no `from` (the `data` was just parsed, no previous exc to chain, so technically not a bug). But could include the actual type for diagnosability. | Change message to include `type(servers_raw).__name__` so developers see what they got: `f"Expected 'mcpServers' to be a dict in '{path}', got {type(servers_raw).__name__}"`. |
| 17 | P1 | 5 | vault.py:88 | `raise VaultError("Vault passphrase cannot be empty")` — no `from exc`; the `if not pwd:` check has no prior exception. Acceptable pattern. But hint is missing. | Add `hint="Set EXO_MCP_VAULT_KEY env var to a non-empty passphrase, or enter a non-empty passphrase at the prompt."` |
| 18 | P2 | 9 | client.py:380–381 | `cleanup()` catches `Exception` and logs at `DEBUG`. Cleanup failures are often silent to the developer since debug logging is off by default. | Promote to `WARNING` level so connection-cleanup failures surface in production logs. |
| 19 | P2 | 7 | client.py:314–317 | `connect()` catches bare `Exception` (fine for `CancelledError` safety) but the re-raised `MCPClientError` message embeds `str(exc)` which may include an `ExceptionGroup` from asyncio transport setup. | Pass `exc` through `unwrap_exception_group(exc)` before embedding in the message, giving a cleaner single-line cause. |
| 20 | P2 | 3 | tools.py:373 | `"MCP tool '{self._original_name}' returned error: {error_text}"` — no `context=` dict. | Add `context={"server": self._server_name, "tool": self._original_name}` to the `MCPToolError` so downstream handlers see structured fields. |

---

## Resilience gaps

| File:Line | I/O site | Gap |
|-----------|----------|-----|
| client.py:309 | `session.initialize()` | No asyncio timeout; a hanging MCP handshake blocks forever |
| client.py:344 | `session.list_tools()` | No asyncio timeout; hung server blocks tool-loading |
| client.py:370 | `session.call_tool(...)` | No asyncio timeout; hung server stalls an agent turn |
| tools.py:317 | lazy `conn.connect()` in `MCPToolWrapper.execute()` | No asyncio timeout; distributed-worker reconnect can stall |
| client.py:457–459 | `connect_all()` serial loop | Serial connect; one slow server blocks the rest; partial failures not aggregated |
| client.py:380 | `_exit_stack.aclose()` in `cleanup()` | No timeout on transport teardown; could hang if subprocess is zombie |

---

## Effort estimate

**M** — all error class lineage is already correct (everything → ExoError); the work is adding `hint=`/`context=` to ~15 raise sites, fixing one unguarded `ValueError` escape, and wrapping 4 I/O hot paths with `asyncio.wait_for`; no architectural changes needed.
