# exo-server — Error DX & Resilience Audit

## Counts
- raise sites: 3 (all `HTTPException`, none `ExoError` subclass)
- error classes total / not inheriting ExoError: 0 defined / 0 (no custom error classes exist at all)
- `except Exception` sites: 3 ; swallow-and-pass: 1 (streaming.py:50–51 catches and yields error payload, swallowing the exception entirely) ; drop-cause: 2 (`streaming.py:50` yields string, no `__cause__`; `app.py:204` does `raise HTTPException(…) from exc` so cause is preserved)
- CancelledError handlers: 0 (not handled anywhere — client disconnect in WebSocket is caught as `WebSocketDisconnect`, but agent run is not cancelled)
- ExoError→HTTP response mapping present? **no** — no FastAPI exception handler registered; `ExoError` is never imported

## Findings (prioritized)

### P0

**[P0] | #7 (async-noise collapse) | app.py:204–205** — Non-streaming `/chat` catches the raw agent exception and forwards `str(exc)` as the HTTP `detail`. If the agent raises an `ExceptionGroup` (the common asyncio/TaskGroup case), the client sees a multi-line nested group traceback as a JSON string. No `unwrap_exception_group` call is made. Fix: import `unwrap_exception_group` from `exo._internal.errors`; apply it before converting to `HTTPException`; additionally check for `ExoError` and surface `.message`, `.hint`, and `.context` as a structured JSON body.

**[P0] | #4 (no silent swallow) | streaming.py:50–51** — `_iter_events` catches `except Exception as exc` and yields `{"type": "error", "error": str(exc)}` — then execution falls through to the caller which yields `data: [DONE]`. The exception is fully swallowed with no logging; the agent run may have been left in an unclean state. Same `ExceptionGroup` problem as above: a raw nested group becomes the client error string. Fix: log the exception at `ERROR` level before yielding the error payload; apply `unwrap_exception_group` to normalise; re-raise after yielding (or at minimum, don't silently discard).

**[P0] | #7 (async-noise collapse) | app.py:143–156 (`_sse_stream`)** — The `/chat?stream=True` SSE path in `app.py` has its own copy of `_iter_events` logic (not the one in `streaming.py`) and the same `except Exception as exc: yield f"data: {json.dumps({'error': str(exc)})}\n\n"` pattern with no unwrapping and no logging. Duplicate of the P0 above. Fix: consolidate into `streaming._iter_events` and apply the fix there only once.

**[P0] | #6 (cancellation safety) | streaming.py:99–106, app.py:203** — When a WebSocket client disconnects mid-stream, `WebSocketDisconnect` is caught and the handler returns silently — but the `async for event in stream_fn(agent, message)` generator is abandoned without being cancelled or closed. The underlying agent task may continue running. Similarly, if the HTTP client disconnects during the non-streaming `/chat` call, the agent coroutine is abandoned. Fix: wrap the agent call in `asyncio.shield`/cancel, or use `asyncio.Task` with explicit cancellation on disconnect; for the streaming generator, use `aclose()` in a `finally` block.

### P1

**[P1] | #1 (taxonomy) | app.py, streaming.py** — No `ExoError` subclass is defined or imported. All agent failures surface to HTTP clients as raw `str(exc)` embedded in a 500 `detail` or an SSE error event. There is no structured mapping of `ExoError → {"status", "message", "hint", "context"}`. Fix: add a FastAPI exception handler in `create_app()`:
```python
from exo.types import ExoError
from exo._internal.errors import unwrap_exception_group
from fastapi.responses import JSONResponse

@app.exception_handler(ExoError)
async def exo_error_handler(request, exc):
    return JSONResponse(
        status_code=422,
        content={"message": exc.message, "hint": exc.hint, "context": exc.context},
    )
```
Also add a `BaseException` handler that applies `unwrap_exception_group` before re-raising as a 500 with only `exc.message` (no full traceback) in the body.

**[P1] | #2 (actionable messages) | app.py:116** — `raise HTTPException(status_code=503, detail="No agents registered")` gives no hint. Fix: add `"→ Call register_agent(app, my_agent) before starting the server."` to the detail.

**[P1] | #2 (actionable messages) | app.py:128** — `raise HTTPException(status_code=400, detail="No agent_name specified and no default agent")` gives no fix hint. Fix: `"→ Pass agent_name in the request body, or set default=True when calling register_agent()."`.

**[P1] | #3 (context) | app.py:204–205** — The `HTTPException` carries only `str(exc)`, discarding any `context=` dict from an `ExoError`. A developer gets a message string but loses the `where:` / `→` hint fields that `ExoError.__str__` embeds. Fix: handle `ExoError` specifically and include `.context` and `.hint` as separate top-level JSON fields, not just a flattened string.

**[P1] | #9 (consistent logging) | app.py:204, streaming.py:50** — No logging at the `except` sites. Errors are either forwarded to the client (as opaque strings) or silently swallowed. The server operator has no server-side record of agent failures. Fix: add `logger.exception("Agent error", exc_info=exc)` at each `except` site before converting to an HTTP response.

**[P1] | #8 (resilience) | streaming.py:109–113 (`_sse_iter`)** — `_sse_iter` calls `_iter_events` which may yield an error event, but after that it always yields `data: [DONE]`. The SSE `/stream` endpoint has no timeout: a long-running or hung agent will hold the connection open indefinitely. Fix: wrap the stream with `asyncio.timeout()` (Python 3.11+) and yield an error event + `[DONE]` on `TimeoutError`.

### P2

**[P2] | #1 (taxonomy) | app.py:138–141** — `_sse_stream` yields `{'error': 'Streaming not available'}` as a data string when `run.stream` is None, then silently returns. This is a configuration issue that should be a proper error (or ideally detected at startup). Fix: raise a structured error at `create_app()` time if streaming is unavailable, rather than silently degrading at request time.

**[P2] | #8 (resilience) | sessions.py (all mutation endpoints)** — The in-memory session store uses a plain `dict` in `app.state` with no concurrency protection. Under concurrent requests, `_get_store` + mutate + `_set_store` is not atomic. Fix: use `asyncio.Lock` per-app, or move to a proper store. This is a correctness/resilience note, not a crash hazard under uvicorn's default single-worker mode.

**[P2] | #5 (chaining) | streaming.py:50–51** — The caught exception is converted to a string and yielded; the original exception is not chained anywhere. Fix: at minimum log with `exc_info=True` to preserve the traceback server-side.

**[P2] | #2 (actionable messages) | streaming.py:94–96** — WebSocket "Agent not found" / "No agents registered" errors are plain strings. Fix: include the list of known agent names in the error body to help callers self-correct.

**[P2] | #4 (no silent swallow) | streaming.py:81** — `except (WebSocketDisconnect, ValueError, KeyError): return` on initial JSON parse silently discards malformed messages with no log. A developer sending bad JSON gets no feedback server-side. Fix: log at `WARNING` level before returning.

## Resilience gaps

| Location | Gap |
|---|---|
| `streaming.py:40–51` (`_iter_events`) | No timeout on agent run; a hung agent holds the stream open forever |
| `app.py:203` (`/chat` non-streaming) | No timeout on `await _run_agent(...)` — a slow agent blocks the request indefinitely |
| `streaming.py:99–106` (WebSocket) | Agent generator not explicitly closed/cancelled on client disconnect; underlying task may leak |
| `streaming.py:109–113` (`_sse_iter`) | SSE stream has no keep-alive heartbeat; long silent runs cause proxy/browser timeouts with no client-side signal |
| `sessions.py` (mutation endpoints) | No concurrency lock on the in-memory session dict; race conditions under concurrent writes |

## Effort estimate

**S** — The package is tiny (5 files, ~350 LOC). The highest-value fix is adding a single FastAPI exception handler + `unwrap_exception_group` call in `create_app()`, plus adding `asyncio.timeout()` wrappers in the two agent-invocation sites. Estimated 2–3 hours for all P0/P1 items.
