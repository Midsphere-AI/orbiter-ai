# exo-core — Error DX & Resilience Audit

## Counts
- raise sites: 178
- error classes total / not inheriting ExoError: 30 / 3
  - `MaxToolCallsExceeded(RuntimeError)` — `packages/exo-core/src/exo/ptc.py:100`
  - `_PTCBaseExceptionTrap(Exception)` — `packages/exo-core/src/exo/ptc.py:104` (internal trap, acceptable)
  - Multiple `ValueError` raises in `agent.py` (inject_message, to_dict, _serialize_tool, _deserialize_tool, _import_object), `config.py` (all validators), `parallel.py:145` — these cross public package boundaries with plain Python builtins instead of `AgentError`/`ExoError`
- `except Exception` sites: 60 ; swallow-and-pass: 0 (none silently pass) ; drop-cause (no `from`): 8 (see findings)
- `CancelledError` handlers: 9 explicit; `BaseException` catchers: 3 (two in agent.py, one in ptc.py)
- I/O call sites lacking timeout/retry: 3 (see Resilience gaps)

---

## Findings (prioritized)

### P0 — Silent failure / cause loss / swallowed cancellation / crash with useless message

**[P0] | #6 | `packages/exo-core/src/exo/agent.py:2851` | BaseException swallows CancelledError in `_execute_tools`**
Inside `_run_one` (the per-tool task function), after `GuardAbortError` re-raise, there is `except BaseException as exc:` that converts any exception — including `asyncio.CancelledError` — into a `ToolResult` with an error string. If the outer `asyncio.TaskGroup` cancels child tasks, the cancellation is silently absorbed and the tool slot gets an error string instead of the cancel propagating. Fix: split into `except asyncio.CancelledError: raise` then `except Exception as exc:`, or check `isinstance(exc, asyncio.CancelledError)` before converting.

**[P0] | #6 | `packages/exo-core/src/exo/agent.py:1777` | BaseException swallows CancelledError in `spawn_self` tool**
The `except BaseException as exc: return tool_error(...)` block in the `spawn_self` TaskGroup catcher converts `CancelledError` into a string tool result. If the parent agent is cancelled while children run, the cancellation is eaten. Fix: `except* HandlerError as eg:` or check and re-raise `CancelledError`.

**[P0] | #5 | `packages/exo-core/src/exo/runner.py:1043` | Cause flattened: `raise AgentError(str(exc)) from exc`**
In `_resolve_provider`, the `ModelError` re-raise wraps `str(exc)` not the original error class name or structured fields: `raise AgentError(str(exc)) from exc`. This is actually `from exc` so the chain is preserved — however the message degrades to a string rather than carrying `context=/hint=`. Medium priority, but the `ModelError` detail (which model, which provider) is lost in the string conversion.

**[P0] | #4 | `packages/exo-core/src/exo/runner.py:802-803` | Silent swallow in MCP progress queue drain**
`except Exception: break` at line 802 in the MCP progress queue drain silently swallows any exception without logging. A `QueueEmpty` is expected here but the broad catch hides any real errors. Fix: catch `asyncio.QueueEmpty` specifically.

**[P0] | #2,#3 | `packages/exo-core/src/exo/ptc.py:100` | `MaxToolCallsExceeded(RuntimeError)` escapes package boundary**
This error class inherits `RuntimeError` (not `ExoError`) and carries no `context=` or `hint=`. When the PTC tool call limit is exceeded, the developer receives a raw `RuntimeError` with no hint about what the limit is or how to raise it. Fix: `class MaxToolCallsExceeded(ExoError)` with `hint="Increase ptc_max_tool_calls= on the Agent(...) constructor"` and `context={"limit": max_tool_calls, "agent": agent_name}`.

**[P0] | #2,#3 | `packages/exo-core/src/exo/config.py:26-131` | All validators raise `ValueError` not `AgentError`**
`parse_model_string`, `validate_planning_model`, `validate_budget_awareness`, `validate_injected_tool_args`, `validate_max_spawn_children` all raise plain `ValueError`. These are called from `Agent.__init__`, so a typo in `model=` or `context_pressure=` reaches the developer as an unadorned `ValueError` with no hint. Fix: convert to `AgentError` with `hint=` showing the valid format/range.

---

### P1 — Unactionable message or missing context on a common path

**[P1] | #2,#3 | `packages/exo-core/src/exo/agent.py:2267,2287` | `inject_message` / `inject_ephemeral` raise `ValueError` not `AgentError`**
Public API methods that raise bare `ValueError("inject_message content must be non-empty")`. Should be `AgentError` with `hint="Pass a non-empty string to inject_message()"`.

**[P1] | #2,#3 | `packages/exo-core/src/exo/agent.py:3173-3190` | `to_dict()` raises `ValueError` not `AgentError` on 6 cases**
All serialization guards raise bare `ValueError`. These are developer-facing; they should be `AgentError` with hint showing the workaround (e.g. "Use a module-level function instead of a callable instruction").

**[P1] | #2,#3 | `packages/exo-core/src/exo/agent.py:3352-3448` | Serialization helpers raise `ValueError` not `AgentError`**
`_serialize_tool`, `_deserialize_tool`, `_import_object` all raise plain `ValueError`. These surface during `Agent.from_dict()` / `Agent.to_dict()`. Fix: `AgentError` with `hint=` and `context={"tool": name}`.

**[P1] | #2 | `packages/exo-core/src/exo/agent.py:2729` | Context-length error message lacks hint**
```python
raise AgentError(f"Context length exceeded on agent '{self.name}': {exc}") from exc
```
No `hint=` explaining the fix (reduce `context_limit=`, enable `overflow="summarize"`, or use `context_mode="compact"`). Very common path on long conversations.

**[P1] | #2 | `packages/exo-core/src/exo/agent.py:2753` | Retry exhaustion message lacks hint**
`AgentError(f"Agent '{self.name}' failed after {max_retries} retries: {last_error}")` has no `hint=`. Developers don't know if they should check an API key, increase `max_retries`, or look for a rate limit. Fix: add `hint="Check your provider API key, model quota, and network connectivity. Increase max_retries= for transient failures."` and `context={"agent": self.name, "model": self.model, "retries": max_retries}`.

**[P1] | #2 | `packages/exo-core/src/exo/_internal/call_runner.py:168` | Loop detection error lacks hint**
`CallRunnerError("Endless loop detected: same tool calls repeated {consecutive} times (threshold={threshold})")` — informative but no `hint=` saying what to do. Fix: add `hint="Increase loop_threshold= if repetition is expected, or add logic to the tool/instructions to vary tool arguments across calls."` and `context={"agent": agent.name, "signature": signature[:80]}`.

**[P1] | #2 | `packages/exo-core/src/exo/_internal/handlers.py:170` | Max-handoffs message lacks hint**
`HandlerError(f"Max handoffs ({self.max_handoffs}) exceeded in swarm")` — no `hint=` on how to increase the limit or restructure the flow. Fix: add `hint="Increase max_handoffs= on the Swarm, or use workflow mode if handoff chaining is not needed."`.

**[P1] | #7 | `packages/exo-core/src/exo/_internal/agent_group.py:104-106` | ExceptionGroup not unwrapped at group boundary**
`ParallelGroup.run()` catches `except* GroupError as eg` and raises a new `GroupError` with "; "-joined messages. The individual agent names appear in the message but the first exception's cause is the re-raised `ExceptionGroup` (`from eg`), not the original error. Developers see `GroupError: ParallelGroup 'X' failed: ...` but lose per-agent structured context. Fix: use `unwrap_exception_group` or preserve individual causes.

**[P1] | #7 | `packages/exo-core/src/exo/_internal/handlers.py:432-434` | Same as above in `GroupHandler._run_parallel`**
Parallel agent group re-raises with `from eg` (ExceptionGroup as cause), which surfaces to the developer as nested noise. Fix: same `unwrap_exception_group` or flatten with `describe_exception_group`.

**[P1] | #3 | `packages/exo-core/src/exo/swarm.py:659,1068` | Handoff-limit errors lack swarm context**
`SwarmError(f"Max handoffs ({self.max_handoffs}) exceeded in swarm")` — no `context=` naming which swarm or which agent triggered the excess. Fix: add `context={"swarm": self.name, "agent": current_agent_name}`.

**[P1] | #3 | `packages/exo-core/src/exo/_internal/nested.py:101,141` | NestedSwarmError drops inner cause type**
`raise NestedSwarmError(f"SwarmNode '{self.name}' failed: {exc}") from exc` — the `from exc` is present (good), but the message body is just `str(exc)`. The developer sees `NestedSwarmError: SwarmNode 'X' failed: ...` where the inner detail could be a `CallRunnerError` with agent context. Fix: use `unwrap_exception_group` on the inner error before re-wrapping.

**[P1] | #5 | `packages/exo-core/src/exo/agent.py:2833` | Tool execution exception drops ExoError context**
In `_execute_tools`, `except (ToolError, Exception) as exc: result = _tool_error(..., str(exc))` converts all exceptions (including structured `ExoError` with `context=` fields) to a plain string. The `hint=` and `context=` of any `ToolError` subclass are stripped away. Fix: check `isinstance(exc, ExoError)` and preserve `str(exc)` (which already renders context) vs bare exception.

**[P1] | #2 | `packages/exo-core/src/exo/parallel.py:283` | `RuntimeError` created inline, not `SubAgentError`**
`error=RuntimeError("Agent wrapper exited without setting result")` in the fallback slot — creates a raw `RuntimeError` as a value, not raised. Since it ends up inside `SubAgentResult.error`, it's accessible but inconsistent with the rest of the error taxonomy.

---

### P2 — Polish

**[P2] | #2 | `packages/exo-core/src/exo/_internal/output_parser.py:115` | OutputParseError missing hint**
`OutputParseError(f"Invalid JSON in arguments for tool '{tool_name}': {exc}")` — no `hint=` telling the developer the LLM produced malformed JSON and they should check the tool schema description. Fix: add `hint="The LLM produced non-JSON arguments. Check the tool's schema (parameters) and description. Structured output_type= may help for strict outputs."`.

**[P2] | #1 | `packages/exo-core/src/exo/tool.py:382` | `except Exception: _hints = {}` drops type-hint resolution failure silently in `FunctionTool.__init__`**
The hint-resolution failure for `_tool_context_param` detection is swallowed with `_hints = {}` (no logging). The function-level failure in `_generate_schema` is already warned about; this second site in `__init__` is silent. Fix: add a `_log.debug(...)` line.

**[P2] | #2 | `packages/exo-core/src/exo/agent.py:2739` | Auth error hint is hardcoded, not model-aware**
`"Check your API key / environment variable."` doesn't name which env var (e.g. `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`). Fix: derive from `agent.provider_name` to give the specific var name.

**[P2] | #9 | `packages/exo-core/src/exo/agent.py:145` | `_build_child_context` swallows fork failure**
`except Exception: child_context = parent.context` — falls back to sharing context, which is intentional, but no log message makes it invisible in production. Fix: add `_log.debug("context fork failed for child '%s', sharing parent context: %s", child_name, exc)`.

**[P2] | #5 | `packages/exo-core/src/exo/agent.py:1948` | Background child exception drops cause type**
`handler.handle_error(task_id, str(exc))` — converts the exception to a string, losing the class and any ExoError context fields. Fix: `handler.handle_error(task_id, f"{type(exc).__name__}: {exc}")`.

**[P2] | #2 | `packages/exo-core/src/exo/swarm.py:1007,1011` | `resume()` errors lack agent/swarm context**
`SwarmError("resume() is only supported in workflow mode")` — no `context=` with swarm name or mode. Fix: add `context={"swarm": self.name, "mode": str(self.mode)}`.

**[P2] | #1 | `packages/exo-core/src/exo/ptc.py:104` | `_PTCBaseExceptionTrap(Exception)` is an internal non-ExoError**
This is deliberately internal and never escapes the package boundary (it's caught within PTCTool.execute). No action needed if it stays internal; note for awareness.

+3 more P2 of "swarm error messages without hint=" kind (swarm.py:83, 690, 1113).

---

## Resilience gaps

**`packages/exo-core/src/exo/agent.py:2704-2710` — LLM `provider.complete()` call**
The `max_retries` backoff is exponential (`2**attempt`) but has no jitter and no overall timeout guard. For providers with long connection timeouts, a single call can hang indefinitely (no `asyncio.wait_for`). External system: any LLM provider HTTP endpoint.

**`packages/exo-core/src/exo/agent.py:2113-2121` — `add_mcp_server()` / `MCPServerConnection.connect()`**
No timeout on `conn.connect()`. A slow or hung MCP server stalls the agent init indefinitely. Fix: `asyncio.wait_for(conn.connect(), timeout=30)`.

**`packages/exo-core/src/exo/agent.py:2406` — `_memory_persistence.store.add(HumanMemory(...))`**
Memory write before the LLM call has no timeout. If the SQLite store is locked (WAL mode, concurrent write), the agent blocks the full run without a deadline. Fix: `asyncio.wait_for(..., timeout=5.0)` with a fallback log-and-continue.

**`packages/exo-core/src/exo/_internal/planner.py:188` — `except Exception: pass` (implicit)**
At line 188, context-window look-up `except Exception: pass` — drops the exception without even logging it. While this is a minor lookup, the silent drop means context-window config issues are invisible.

---

## Effort estimate

**L** (Large). The taxonomy problem (30+ `ValueError`/`RuntimeError` raises across `config.py`, `agent.py`, `parallel.py`, `ptc.py` plus the `MaxToolCallsExceeded` class) requires touching every config validator and the serialization layer. The cancellation safety bugs in `_execute_tools` and `spawn_self` require careful surgery in the hot execution path. Adding `hint=`/`context=` to the ~12 most common error paths (retry exhaustion, context length, loop detection, handoff limit) is straightforward but numerous. Resilience gaps (MCP connect timeout, memory write timeout) are self-contained additions.
