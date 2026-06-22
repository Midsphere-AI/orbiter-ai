# exo-distributed — Error DX & Resilience Audit

## Counts
- raise sites: 14
- error classes total / not inheriting ExoError: 0 / 5 (offenders: temporal.py:146 `ImportError`, temporal.py:204 `RuntimeError`, temporal.py:236 `RuntimeError`, worker.py:149-168 `ValueError` ×4, worker.py:195 `ImportError`)
- `except Exception` sites: 7 ; swallow-and-pass: 0 ; drop-cause: 2 (worker.py:573 suppress(Exception) around teardown, _alerts.py:212 logs but drops `__cause__` on callback failure)
- CancelledError handlers: 4 (all safe — worker.py:316/333 explicitly re-raise, worker.py:273/579 use `contextlib.suppress(CancelledError)` on task cleanup paths where swallowing is correct)
- I/O call sites lacking timeout/retry/reconnect: 9

---

## Findings (prioritized)

**[P0] | #8 | worker.py:347–740 | Task timeout_seconds never enforced**
`TaskPayload.timeout_seconds` (default 300 s) is stored but the worker never wraps `_run_agent()` in `asyncio.wait_for` or `asyncio.timeout`. A stuck agent runs indefinitely, permanently consuming a worker slot and blocking the PEL entry. Fix: wrap `result_text = await self._run_agent(agent, task, token)` with `asyncio.wait_for(..., timeout=task.timeout_seconds)` and ack the task on timeout expiry.

**[P0] | #8 | worker.py:515–561 | Cascading Redis failure leaves task stuck in PEL**
The `except Exception` handler calls `self._store.set_status(FAILED)` then `self._broker.ack/nack()` — both go through `_client()` which raises `ExoError` if Redis is down. If those secondary Redis calls also throw, neither `ack` nor `nack` is executed, leaving the message stuck in the consumer PEL (Pending Entries List) forever. This is a task-loss vector. Fix: wrap the error-path Redis calls in their own try/except; always attempt ack in a `finally` clause that logs and suppresses infrastructure failures.

**[P0] | #8 | broker.py:118 | nack reads original message from stream before ack-ing — race**
`nack()` calls `r.xrange(...)` to fetch the payload, then `r.xack()`, then `r.xadd()`. If Redis restarts between xrange and xack, the message is lost with no fallback. More critically: if `r.xrange()` returns empty (line 125 `else` branch), the task is acked with no re-queue — it silently vanishes. Fix: treat the empty-xrange branch as an error (log at ERROR level and raise or mark FAILED), not a warning.

**[P0] | #5 | temporal.py:204, 236 | `RuntimeError` not wrapped in ExoError, cause dropped**
```python
# temporal.py:204
raise RuntimeError(msg)          # bare RuntimeError, no __cause__
# temporal.py:236
raise RuntimeError(msg)          # same
```
These cross the package boundary as raw `RuntimeError`, losing the ExoError structure and any chain. Fix: `raise ExoError(msg, hint="Call connect() before execute_task()/start_temporal_worker().") from None`.

**[P0] | #1 | temporal.py:146 | ImportError not wrapped in ExoError**
```python
raise ImportError(msg)  # temporal.py:146
```
Bare `ImportError` escapes to callers who can't distinguish it from a missing stdlib module. The Worker.__init__ at worker.py:195 has the same pattern. Fix: `raise ExoError(msg, hint="pip install exo-distributed[temporal]")`.

**[P1] | #8 | worker.py:589–604 | `_listen_for_cancel` opens raw Redis connection per task, no reconnect on error**
A new `aioredis.from_url(...)` connection is created per task. If Redis drops mid-subscription, the inner `pubsub.get_message(timeout=1.0)` will raise a `ConnectionError` that is not caught anywhere in the loop, crashing the coroutine without setting `token.cancelled`. The `finally` block still runs `r.aclose()` (connection cleanup is safe), but the cancel signal is then permanently missed for the rest of the task's life. Fix: wrap the loop body in `except (aioredis.ConnectionError, aioredis.RedisError): await asyncio.sleep(1); continue`.

**[P1] | #8 | worker.py:299–326 | `_heartbeat_loop` opens raw Redis connection with no reconnect backoff**
A separate `aioredis.from_url(...)` connection is created at start. The inner `except Exception` retries immediately after `asyncio.sleep(heartbeat_ttl / 3)`, but on a persistent Redis outage the connection object is never recreated — every iteration will fail with `ConnectionError` forever. If the connection object becomes invalid, heartbeats are silently lost, and the worker appears dead to `WorkerHealthCheck`. Fix: recreate the connection inside the loop on failure (or use a reconnect-capable client).

**[P1] | #8 | client.py:87–118 | `TaskHandle.result()` polls forever with no timeout guard**
The caller's `timeout` parameter is stored in `TaskPayload.timeout_seconds` (enforced nowhere — see P0 above), but `TaskHandle.result()` has its own infinite `while True` poll loop with no deadline. If the worker crashes without updating task status to FAILED/CANCELLED, `result()` blocks the caller indefinitely. Fix: accept an optional `timeout` kwarg and wrap with `asyncio.wait_for`.

**[P1] | #8 | client.py:184–214 | `distributed()` leaks Redis connections if `broker.submit()` raises**
```python
await broker.connect()
await store.connect()
await subscriber.connect()
# ... build payload ...
await broker.submit(payload)   # if this throws, broker/store/subscriber never disconnected
```
There is no `try/finally` or `async with` around the submit. Fix: wrap the block in `try/except` and call `await broker.disconnect(); await store.disconnect(); await subscriber.disconnect()` on failure before re-raising.

**[P1] | #8 | events.py:163–198 | `EventSubscriber.subscribe()` loops forever if task never reaches terminal state**
The subscriber polls Pub/Sub until a `StatusEvent` with `"completed"/"error"/"cancelled"` arrives. If the worker crashes without publishing a terminal event, the subscriber hangs indefinitely (no timeout). Fix: add a configurable `deadline` param and break out of the loop with a `TimeoutError` after `deadline` seconds without a terminal event.

**[P1] | #2 | worker.py:149–168 | `ValueError` for grouped-vs-flat conflict carries no hint about the grouped API**
```python
raise ValueError(
    "Cannot specify both 'redis=' and 'redis_url='. "
    "Use RedisConfig(url=...) or the flat 'redis_url=' kwarg, not both."
)
```
These are already somewhat actionable, but (a) they are raw `ValueError`, not `ExoError`, and (b) `hint=` would make them teachable. Fix: convert to `ExoError(msg, hint="Prefer redis=RedisConfig(url=...) for grouped config.")`.

**[P1] | #2 | client.py:111/117/175 | ExoError messages miss hint= and context=**
```python
raise ExoError(f"Task {self._task_id} failed: {task_result.error or 'unknown error'}")
raise ExoError(f"Task {self._task_id} was cancelled")
raise ExoError("redis_url must be provided or EXO_REDIS_URL environment variable must be set")
```
All three lack `hint=` and `context=`. Fix for example: `raise ExoError(msg, context={"task_id": self._task_id}, hint="Check worker logs for the root failure; use task.status() for details.")`.

**[P1] | #3 | worker.py:89 | ExoError for unknown message role missing context=**
```python
raise ExoError(f"Unknown message role: {role!r}")
```
No `context={"role": role, "task_id": ...}`, no `hint=`. Fix: add `context={"role": role}` and `hint="Valid roles: user, assistant, system, tool."`.

**[P1] | #8 | broker.py:51–57 | `connect()` raises raw `aioredis.ResponseError` on non-BUSYGROUP errors**
Non-BUSYGROUP Redis errors from `xgroup_create` escape as raw `aioredis.ResponseError` with no context about which stream/group failed or what the developer should do. Fix: catch and re-raise as `ExoError(f"Failed to create consumer group {self._group_name!r} on stream {self._queue_name!r}: {exc}", hint="Ensure Redis is reachable and the stream key is not occupied by a different type.") from exc`.

**[P2] | #9 | worker.py:438–443 | ImportError for exo-memory logged at WARNING, not INFO, with no task context in message**
The log message `"Task %s: exo-memory is not installed; ..."` is correct but severity is WARNING, which inflates logs on expected optional-dep-missing paths. P2 only.

**[P2] | #8 | health.py:146–180 | `get_worker_fleet_status` creates a bare connection with no timeout**
`aioredis.from_url(...)` has no `socket_connect_timeout` or `socket_timeout`. A hung Redis will block the scan indefinitely. Fix: pass `socket_connect_timeout=5, socket_timeout=5`.

**[P2] | #8 | health.py:67–117 | `WorkerHealthCheck.check()` creates a synchronous Redis connection with no timeout**
```python
r = redis.from_url(self._redis_url, decode_responses=True)
data: dict[str, str] = r.hgetall(key)
```
No socket timeout; will block the calling thread if Redis is unreachable. Fix: pass `socket_timeout=5` to `redis.from_url`.

**[P2] | #3 | events.py:82 | ExoError for unknown event type missing context**
```python
raise ExoError(f"Unknown event type: {event_type!r}")
```
No `hint=` listing the valid set. Fix: add `hint=f"Valid types: {list(_EVENT_TYPE_MAP)}"`.

---

## Resilience gaps

| File:line | System | Gap |
|-----------|--------|-----|
| worker.py:347 | Task execution | `timeout_seconds` stored in payload but never enforced — tasks can run forever |
| worker.py:515–561 | Redis / PEL | ack/nack in error handler can also throw, leaving task stuck in PEL permanently |
| worker.py:589 | Redis Pub/Sub | Per-task cancel listener has no reconnect on `ConnectionError`; cancel signal lost |
| worker.py:299 | Redis | Heartbeat uses a stale connection object on Redis reconnect; no conn recreation |
| broker.py:118 | Redis Streams | nack's empty-xrange branch silently drops the task (warns, but no ack means PEL stuck) |
| client.py:87 | Redis polling | `result()` polls forever with no deadline — hangs on crashed worker |
| client.py:184 | Redis | Connections leaked if `broker.submit()` raises (no try/finally) |
| events.py:163 | Redis Pub/Sub | `subscribe()` loops forever if no terminal event published (worker crash scenario) |
| health.py:67,160 | Redis | Sync and async health check connections have no `socket_timeout` |

---

## Effort estimate

**L** — 9 resilience gaps (including the critical PEL/timeout/leak triad), 5 bare-exception raise sites, and 6+ ExoError quality upgrades across 6 files; the task-timeout and PEL-cleanup fixes alone require careful `asyncio.wait_for` wiring and retry-safe finally logic.
