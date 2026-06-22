# exo-a2a — Error DX & Resilience Audit

## Counts
- raise sites: 11 (10 in client.py, 1 in server.py)
- error classes total / not inheriting ExoError: 2 / 0
  - `A2AClientError(ExoError)` — `packages/exo-a2a/src/exo/a2a/client.py:22` ✓
  - `A2AServerError(ExoError)` — `packages/exo-a2a/src/exo/a2a/server.py:24` ✓
- `except Exception` sites: 3 ; swallow-and-pass: 0 ; drop-cause: 0
  - `server.py:254` — `execute_task` failure handler (returns 500 JSON, logs with exc_info) ✓
  - `server.py:337` — `_generate()` streaming failure handler (emits FAILED event, logs with exc_info) — **misses CancelledError** (P0)
  - `client.py:110` — `(json.JSONDecodeError, Exception)` in `_resolve_from_file` ✓ (chained from exc)
- `CancelledError` handlers: 0 explicit — no cancellation handling anywhere
- I/O call sites lacking timeout/retry: 3
  - `client.py:95` — agent card `GET` (timeout set, no retry)
  - `client.py:135` — task `POST /` (timeout set, no retry)
  - `client.py:174` — streaming `POST /stream` (timeout set, no retry)

---

## Findings (prioritized)

### P0 — Crash / cancel swallowed / server stability

**[P0] | #6 | `server.py:337` | `except Exception` in `_generate()` does not catch `CancelledError` — but the real risk is the opposite: generator abandonment**
The streaming `_generate()` async generator has `except Exception as exc` to emit a FAILED NDJSON event on error — which correctly excludes `CancelledError`. However, `CancelledError` *is* `BaseException`, not `Exception`. If the client disconnects mid-stream and Starlette/asyncio cancels the generator, `CancelledError` is thrown into `_generate()` at `await server._executor.execute(...)`. It is not caught, so it propagates up through `StreamingResponse` unchecked — which can surface as an unhandled asyncio exception in server logs and leave the task store in WORKING state permanently.
Fix: wrap the entire generator body in `try/finally` (not `except`) to save FAILED/CANCELED state on any exit, and let `CancelledError` propagate naturally:
```python
async def _generate() -> AsyncIterator[str]:
    try:
        ...
        result = await server._executor.execute(...)
        yield ...  # artifact + completed events
    except asyncio.CancelledError:
        # Log, optionally emit CANCELED event, then re-raise
        logger.warning("A2A stream cancelled: task_id=%s", task_id)
        yield json.dumps(TaskStatusUpdateEvent(task_id=task_id,
            status=A2ATaskStatus(state=TaskState.CANCELED)).model_dump()) + "\n"
        raise
    except Exception as exc:
        ...  # existing FAILED path
```

**[P0] | #8 | `client.py:183` | `json.loads()` in `send_task_collect` NDJSON loop is outside any try/except**
After the HTTP POST succeeds (lines 173–177), the `except httpx.HTTPError` block closes. Lines 179–184 then iterate over NDJSON lines calling `json.loads(line)` with no error handling. A malformed line (truncated stream, gateway error body, server bug) raises a bare `json.JSONDecodeError` that escapes to the caller as a raw Python exception — not an `A2AClientError`. This violates checklist #1 (raw exception escapes package boundary) and #2 (unactionable — developer has no idea which line or which peer caused it).
Fix:
```python
try:
    events.append(json.loads(line))
except json.JSONDecodeError as exc:
    raise A2AClientError(
        f"Malformed NDJSON line from A2A peer at {url}/stream",
        context={"url": f"{url}/stream", "task_id": task_id, "line": line[:120]},
        hint="The remote agent returned invalid JSON in its streaming response. Check server logs at the peer.",
    ) from exc
```

**[P0] | #8 | `client.py:97` | `resp.json()` and `AgentCard(**resp.json())` in `_resolve_from_url` are inside `except httpx.HTTPError` but `json.JSONDecodeError` / `pydantic.ValidationError` are not HTTPError**
`httpx.Response.json()` raises `json.JSONDecodeError` (a `ValueError`) on malformed bodies. `AgentCard(**data)` raises `pydantic.ValidationError` on a wrong schema. Both escape the `try/except httpx.HTTPError` block at line 98 as raw Python exceptions — not `A2AClientError`. A developer whose agent card endpoint returns a non-JSON 200 body would see a raw `JSONDecodeError` with no context about which URL was involved.
Fix: broaden the except block or add an inner catch:
```python
        except httpx.HTTPError as exc:
            raise A2AClientError(
                f"HTTP error fetching agent card from {url}: {exc}",
                context={"url": url},
                hint="Verify the agent card endpoint is reachable and returns valid JSON.",
            ) from exc
        except (json.JSONDecodeError, Exception) as exc:
            raise A2AClientError(
                f"Invalid agent card JSON at {url}: {exc}",
                context={"url": url},
                hint="The /.well-known/agent-card endpoint must return valid AgentCard JSON.",
            ) from exc
```
Same issue at `client.py:137` — `resp.json()` in `send_task` can return `JSONDecodeError` from a 200 body with non-JSON content.

---

### P1 — Unactionable messages or missing context

**[P1] | #2,#3 | `client.py:99` | `A2AClientError(f"Failed to fetch agent card from {url}: {exc}")` — no `hint=` or `context=`**
The URL is embedded in the message string but not in `context=` (so it cannot be programmatically extracted). No `hint=` telling the developer what to check.
Fix:
```python
raise A2AClientError(
    f"Failed to fetch agent card from {url}.",
    context={"url": url},
    hint="Verify the A2A peer is running and the URL points to a valid /.well-known/agent-card endpoint.",
) from exc
```

**[P1] | #2,#3 | `client.py:139` | `A2AClientError(f"Task request failed: {exc}")` — peer URL absent from message**
A developer looking at this error cannot tell which peer was called. The URL is available (`card.url`) but not included.
Fix:
```python
raise A2AClientError(
    f"Task POST to A2A peer at {url}/ failed.",
    context={"url": f"{url}/", "task_id": task_id or "(auto)"},
    hint="Check that the remote A2A server is running and reachable. Increase ClientConfig(timeout=...) for slow agents.",
) from exc
```

**[P1] | #2,#3 | `client.py:177` | `A2AClientError(f"Stream request failed: {exc}")` — same missing URL/hint**
Same pattern as above. The `url/stream` endpoint and `task_id` should appear in `context=`, with a hint to check `ServingConfig(streaming=True)` on the peer.

**[P1] | #2,#3 | `client.py:166` | `A2AClientError("Remote agent does not support streaming")` — no context**
No `context=` naming the agent or its URL, and no `hint=` on how to enable streaming on the server side.
Fix:
```python
raise A2AClientError(
    "Remote agent does not support streaming.",
    context={"agent_name": card.name, "url": card.url},
    hint="Enable streaming on the remote server: ServingConfig(streaming=True). "
         "Or use send_task() for a non-streaming request.",
)
```

**[P1] | #2,#3 | `client.py:58` | `A2AClientError("agent_card string cannot be empty")` — no hint**
Minimal fix: add `hint="Pass a URL ('https://...') or a local file path to the agent card JSON."`.

**[P1] | #2,#3 | `client.py:81` | `A2AClientError("No agent card source to resolve")` — internal state message**
This is an internal consistency guard (should never reach a user normally), but if it does, the message is cryptic. Add `hint="Construct A2AClient with an AgentCard instance, a URL, or a file path."`.

**[P1] | #8 | `client.py:64` | `httpx.AsyncClient` created in `__init__` with no async context manager on `A2AClient`/`RemoteAgent`**
The `httpx.AsyncClient` is created eagerly and closed only via an explicit `await client.close()` call. If the caller forgets `close()` (or an exception interrupts their code), the underlying connection pool leaks. Neither `A2AClient` nor `RemoteAgent` implement `__aenter__`/`__aexit__`, so they cannot be used with `async with`.
Fix: add `async with` support:
```python
async def __aenter__(self) -> "A2AClient":
    return self

async def __aexit__(self, *args: object) -> None:
    await self.close()
```
And do the same for `RemoteAgent`, delegating to `self._client`.

**[P1] | #8 | No retry / backoff on any HTTP call**
All three client HTTP call sites (`_resolve_from_url`, `send_task`, `send_task_collect`) make exactly one HTTP attempt. Transient network errors (connect timeout, brief DNS blip) cause immediate failure with no recovery. `ClientConfig` has a `timeout` field but no `max_retries` or `retry_backoff` fields.
Fix: add `max_retries: int = 1` and `retry_delay: float = 0.5` to `ClientConfig`, then wrap each `await self._http.*` call with a simple exponential backoff loop for `httpx.TransportError` (network-layer errors, not 4xx/5xx HTTP status errors which are not retried).

**[P1] | #2 | `server.py:183` | `A2AServerError("fastapi is required for A2AServer: pip install fastapi")` — good, but missing `hint=`**
The install command is in the message body rather than `hint=`. Minor improvement: move to `hint=`:
```python
raise A2AServerError(
    "fastapi is not installed — A2AServer requires it.",
    hint="Install with: pip install fastapi uvicorn",
) from exc
```

---

### P2 — Polish

**[P2] | #9 | `server.py:254,337` | `except Exception` logs use `exc_info=True` — good, but `str(exc)` for an `ExoError` in `reason=` includes full context block**
When a wrapped `ExoError` (with `context=` and `hint=`) is the agent failure, `str(exc)` renders the multi-line formatted string (including `→ hint` lines) into `A2ATaskStatus.reason`. This is verbose in the JSON response and could expose internal implementation details to remote callers. Consider using `exc.message` (just the one-liner) for the wire-facing `reason` field when `isinstance(exc, ExoError)`.

**[P2] | #1 | `client.py:110` | `except (json.JSONDecodeError, Exception)` — redundant**
`Exception` already covers `json.JSONDecodeError`. Should be just `except Exception as exc`. Not a bug, but noisy.

**[P2] | #3 | `client.py:93` | `A2AClientError(f"Invalid URL: {url}")` — bare URL in message, no hint**
Add `hint="Provide a full URL including scheme, e.g. 'https://myagent.example.com/.well-known/agent-card'."`.

---

## Resilience gaps

| File:line | Issue |
|---|---|
| `client.py:64` | `httpx.AsyncClient` created in `__init__`, not in an `async with` block; leaked if `close()` never called |
| `client.py:95` | Agent-card fetch: single attempt, no retry on `TransportError` (connect / DNS) |
| `client.py:135` | Task POST: single attempt, no retry; long-running agents + flaky networks = avoidable failure |
| `client.py:174` | Streaming POST: single attempt, no retry |
| `server.py:313` | `await server._executor.execute(...)` inside `_generate()`: if this hangs indefinitely the streaming HTTP response never completes. No per-request timeout; relies solely on the agent's own timeout. A starlette-level `anyio.fail_after()` or `asyncio.wait_for()` wrapper would protect the server. |
| `server.py:229` | `execute_task`: same — no server-side request timeout around `_executor.execute()`. |

---

## Effort estimate

**S** — The package is small (4 source files, ~365 lines across client + server). All P0s are localized fixes (one `try/except` addition each), P1s are message enrichment (`context=`, `hint=`, async context manager). No architectural changes required. Realistic fix time: 2–3 hours including tests.
