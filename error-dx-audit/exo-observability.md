# exo-observability — Error DX & Resilience Audit

## Counts
- raise sites: 11
- error classes total / not inheriting ExoError: 0 custom error classes defined — all 11 raises use stdlib types (`RuntimeError`, `ImportError`, `ValueError`, `TypeError`, `KeyError`, `AttributeError`)
- `except Exception` sites: 2 ; swallow-and-pass (no log): 0 ; drop-cause: 0
- `except BaseException` sites: 4 (all in `tracing.py` `@traced` wrappers — correct: record + re-raise)
- CancelledError handlers: 0 (no bare `except BaseException` that could accidentally swallow; all 4 re-raise immediately)
- I/O call sites lacking timeout: 1 (`BatchSpanProcessor` created without `export_timeout_millis`)

---

## Findings (prioritized)

### P0 — Tracer hook callbacks are unguarded: a span SDK error aborts the agent run

**[P0] | checklist#4 | `tracer.py:110-212` (all `_on_*` async hook methods)**

`HookManager` explicitly states: *"exceptions from hooks are **not** suppressed — a failing hook aborts the run"* (`hooks.py:35-36`). Every `Tracer` hook callback (`_on_start`, `_on_finished`, `_on_error`, `_on_pre_llm`, `_on_post_llm`, `_on_pre_tool`, `_on_post_tool`, `_on_context_window`) calls OTel span methods directly with zero `try/except` protection.

If `span.start_span()`, `span.set_attribute()`, `span.set_status()`, `span.end()`, or `_otel_context.detach()` raises for any reason (SDK bug, attribute value type error, corrupted context, etc.), the user's agent run dies with a raw OTel exception. This is the defining fail-open violation for an observability subsystem.

**Concrete fix:** Wrap each hook callback body in `try/except Exception` + `_log.warning(..., exc_info=True)`. Because these are hook callbacks, catching and logging (not re-raising) is exactly correct here — instrumentation failure must never kill the observed process:

```python
async def _on_start(self, *, agent: Any = None, input: Any = None, **_: Any) -> None:
    try:
        name = getattr(agent, "name", "agent")
        ...
        self._open("agent", f"agent.run {name}", attrs)
    except Exception:
        _log.warning("tracer._on_start failed (tracing degraded)", exc_info=True)
```

Apply the same pattern to all 8 hook callbacks.

---

### P0 — `_end_frame` can raise and crash the hook callback

**[P0] | checklist#4 | `tracer.py:117-127`**

`_end_frame` is called from `_close_until` and `_on_post_llm`/`_on_post_tool`. Its `try/finally` block ensures `span.end()` runs, but if `span.record_exception()` or `span.set_status()` raises (before the `finally`), the exception propagates out of the hook callback with no guard. This is a secondary surface of the same P0 above — fully resolved by adding per-callback `try/except`.

---

### P1 — `langfuse.py` raises bare `RuntimeError` (not ExoError, no hint about what to do next)

**[P1] | checklist#1,#2 | `backends/langfuse.py:42-44`**

```python
raise RuntimeError(
    "langfuse SDK is not installed; install 'langfuse' to use score_client()"
) from exc
```

The message is actually useful (it names the install command), but it should be an `ExoError` subclass that escapes the package boundary, and the error does not tell the user which Exo extra to install. The `_LangfuseScoreClient.__init__` raises eagerly on construction; callers who guard via `try/except` expect a known type.

**Concrete fix:**

```python
from exo.types import ExoError
raise ExoError(
    "langfuse SDK is not installed; cannot publish eval scores.",
    hint="Install it with: pip install langfuse  (or: pip install 'exo-ai[langfuse]')",
    context={"operation": "score_client"},
) from exc
```

---

### P1 — `langsmith.py` raises bare `ImportError` (not ExoError)

**[P1] | checklist#1,#2 | `backends/langsmith.py:44-47`**

```python
raise ImportError(
    "langsmith SDK is required for dataset_client(); "
    "install with: pip install langsmith"
) from exc
```

Same issue: good message body but wrong type. Callers can't distinguish this from a framework-internal import failure.

**Concrete fix:** Raise `ExoError` with `hint="pip install langsmith"` and `context={"operation": "dataset_client"}`.

---

### P1 — `health.py` raises bare `KeyError` (not ExoError, no hint)

**[P1] | checklist#1,#2,#3 | `health.py:149`**

```python
raise KeyError(msg)
```

No context about which agent/check was involved, no hint about valid check names. Callers can't distinguish from a dict miss.

**Concrete fix:**

```python
from exo.types import ExoError
raise ExoError(
    f"Unknown health check: {name!r}",
    context={"check": name, "available": self.list_checks()},
    hint="Use HealthRegistry.list_checks() to see registered check names.",
) from None
```

---

### P1 — `backends/__init__.py` raises bare `TypeError` (not ExoError, no hint)

**[P1] | checklist#1,#2 | `backends/__init__.py:154`**

```python
raise TypeError(f"unsupported tracing spec: {spec!r}")
```

No hint about what valid specs look like. A developer who passes the wrong type gets no guidance.

**Concrete fix:**

```python
raise ExoError(
    f"Unsupported tracing spec: {spec!r}",
    context={"spec_type": type(spec).__name__},
    hint="Pass a string ('langfuse', 'otlp'), list of strings, bool, TracingConfig, or None.",
) from None
```

---

### P1 — `config.py` raises bare `ValueError` in namespace validator (no ExoError, no hint pointing to docs)

**[P1] | checklist#1,#2 | `config.py:175,181,196,206,223,229` (6 raises)**

The `_explode_namespace_configs` validator raises `ValueError` for wrong types and conflicting flat/grouped fields. The messages are already fairly clear (e.g., `"'logging' must be a LoggingConfig or dict, got <class 'str'>"`), but they escape as raw `ValueError`, not `ExoError`, and carry no `hint` pointing to the correct API or docs.

**Concrete fix (example for one — apply pattern to all 6):**

```python
raise ExoError(
    f"'logging' must be a LoggingConfig or dict, got {type(logging_cfg)}",
    hint="Pass LoggingConfig(level='DEBUG') or a plain dict like {'level': 'DEBUG'}.",
    context={"field": "logging"},
)
```

---

### P1 — `cost.py:stamp_cost_on_current_span` silently swallows `ImportError` with bare `pass`

**[P1] | checklist#4 | `cost.py:304-305`**

```python
except ImportError:
    pass
```

This is a deliberate fail-open (OTel not installed → do nothing), which is correct conceptually. However, a bare `pass` with no log makes it invisible to developers trying to debug why cost attributes don't appear on spans. Should be at least a `_log.debug(...)`.

**Concrete fix:**

```python
except ImportError:
    _log.debug("stamp_cost_on_current_span: opentelemetry not installed, skipping")
```

---

### P2 — `BatchSpanProcessor` created with no timeout configuration

**[P2] | checklist#8 | `backends/base.py:98`**

```python
return BatchSpanProcessor(exporter)
```

`BatchSpanProcessor` defaults to `export_timeout_millis=30000` (30 s). A slow/hung OTLP collector will hold up the span-export thread for 30 seconds before timing out. During process shutdown this can extend `force_flush` / `shutdown` time noticeably. It does not block the main agent thread (export is async), but a custom lower timeout is a resilience best-practice for the exporter loop.

**Concrete fix:** Allow passing `export_timeout_millis` through `OTLPBackend`:

```python
return BatchSpanProcessor(exporter, export_timeout_millis=5000)
```

Or expose it as an `OTLPBackend` field so callers can tune it.

---

### P2 — `_LangfuseScoreClient.score()` makes a network call with no error guard

**[P2] | checklist#8 | `backends/langfuse.py:60-67`**

`self._client.score(...)` calls Langfuse's SDK which makes a network round-trip. There is no `try/except` around it. A Langfuse API timeout or 5xx response will surface as an unhandled SDK exception to the caller of `score()`. The caller in `exo-eval` presumably does guard, but this leaves a trap.

**Concrete fix:** Wrap in `try/except Exception` + `_log.warning(...)` inside `score()`:

```python
try:
    self._client.score(**kwargs)
except Exception as exc:
    _log.warning("langfuse.score failed (score not recorded): %s", exc)
```

---

### P2 — `_BraintrustScoreClient.log_score` swallows `Exception` without exc_info

**[P2] | checklist#9 | `backends/braintrust.py:72-73`**

```python
except Exception as exc:  # pragma: no cover — SDK runtime errors
    _log.warning("braintrust.log_score failed: %s", exc)
```

This is a correct fail-open pattern, but logging without `exc_info=True` loses the traceback, which makes SDK runtime errors nearly impossible to debug.

**Concrete fix:** Add `exc_info=True`:

```python
_log.warning("braintrust.log_score failed: %s", exc, exc_info=True)
```

---

### P2 — `health.py:run_all` `except Exception` drops cause in HealthResult message

**[P2] | checklist#5 | `health.py:134-139`**

```python
except Exception:
    logger.error("health check %r raised an exception", name, exc_info=True)
    results[name] = HealthResult(
        status=HealthStatus.UNHEALTHY,
        message=f"Check {name!r} raised an exception",
    )
```

The logging call is correct (uses `exc_info=True`). The `HealthResult.message` is generic — it doesn't include the exception type or message, so callers reading JSON output can't see what failed without also reading logs.

**Concrete fix:** Include exception text in the message:

```python
except Exception as exc:
    logger.error("health check %r raised: %s", name, exc, exc_info=True)
    results[name] = HealthResult(
        status=HealthStatus.UNHEALTHY,
        message=f"Check {name!r} raised {type(exc).__name__}: {exc}",
    )
```

---

## Resilience gaps

| File:line | System | Gap |
|---|---|---|
| `backends/base.py:98` | OTLP exporter | `BatchSpanProcessor` constructed with default 30 s export timeout; no user-facing knob to lower it. A hung collector delays process shutdown. |
| `backends/langfuse.py:60-67` | Langfuse API (network) | `_LangfuseScoreClient.score()` network call has no try/except — SDK errors propagate unguarded to callers. |
| `tracer.py:158-261` (all `_on_*`) | OTel SDK (all backends) | Hook callbacks have no exception guard at all; any OTel SDK failure aborts the agent run. **Primary resilience gap for the whole package.** |

No `CancelledError` exposure found (the `except BaseException` sites in `tracing.py` all immediately re-raise, which is correct).

---

## Effort estimate

**M** — The P0 fix (wrapping 8 hook callbacks in `try/except Exception + log`) is a 30-line mechanical change, but then 6+ `ValueError` → `ExoError` conversions in `config.py`, 3 `ExoError` taxonomy fixes in backends, and the Langfuse/Braintrust network-call guards add up to roughly half a day of careful work with test runs.
