# Error DX Charter

> The rubric every package follows when we harden error handling. The **primary
> goal is developer experience**: because Exo is purely async, failures tend to
> surface as overwhelming walls of `ExceptionGroup` / asyncio-internal frames
> with messages that say *what broke* but never *what to do about it*. This
> charter makes every Exo error **legible, contextual, and actionable**.
>
> **Co-goal:** while we are in every package's error paths anyway, we also harden
> **resilience** across all packages — retries/backoff/timeouts on I/O,
> cancellation safety, resource cleanup, and partial-failure handling. Error DX is
> the headline; resilience is the second deliverable, applied everywhere we touch.

## The three rules

Every error a developer can hit must satisfy:

1. **Legible** — the failure that reaches the developer is one clean cause, not a
   nested `ExceptionGroup` or a traceback full of asyncio plumbing. Collapse the
   noise at the boundary; keep the real cause via `__cause__`.
2. **Contextual** — it names *where* it happened: which agent, tool, model,
   config field, swarm node, and what state it was in. "Tool failed" → "Tool
   `web_search` on agent `researcher` failed while parsing its result".
3. **Actionable** — it ends with `→ <what the developer can do>`: the missing
   argument, the env var to set, the valid options, the doc link. No error is
   "done" until a reader knows their next move.

## The mechanism: structured `ExoError`

Base class in `exo-core/src/exo/types.py`. Fully backward compatible —
`ExoError("msg")` still behaves like a plain exception.

```python
raise ProviderError(
    "Anthropic request failed after 3 retries.",
    context={"model": "anthropic:claude-opus-4", "agent": agent.name},
    hint="Check ANTHROPIC_API_KEY is set and the model id is spelled correctly.",
    doc="https://docs.exo.dev/providers#anthropic",
)
```

renders as:

```
Anthropic request failed after 3 retries.
  where: model='anthropic:claude-opus-4'  agent='researcher'
  → Check ANTHROPIC_API_KEY is set and the model id is spelled correctly.
  docs: https://docs.exo.dev/providers#anthropic
```

- `context` — dict of the involved identifiers/state. Keep keys short and stable.
- `hint` — the **action**, imperative voice. This is the field that turns a
  stack trace into a fix.
- `doc` — optional pointer to docs/example. Omit if none applies.
- The underlying error always goes on `__cause__` via `raise ... from exc` — the
  structured fields summarize, they don't replace the chain.

## Taming async noise

Use the helpers in `exo/_internal/errors.py` (added in Phase 0):

- `unwrap_exception_group(exc)` — collapse single-child `ExceptionGroup`s (the
  common `asyncio.gather`/`TaskGroup` case) down to the real exception. Multi-error
  groups are summarized, not flattened away.
- Boundary points (`run`, `run.stream`, `run.sync`, Swarm/group entry) translate
  raw `ExceptionGroup`/`CancelledError`/asyncio internals into a single clean
  `ExoError` subclass before it reaches the developer.

## Cancellation

- **Never swallow `CancelledError`.** A broad `except Exception` is fine (it does
  not catch `CancelledError`), but any `except BaseException` / bare re-raise path
  must re-raise cancellation untouched.
- Cleanup belongs in `finally`, not in an `except` that might eat the cancel.

## The per-package checklist (what each audit/fix agent applies)

1. **Taxonomy** — every raised exception is an `ExoError` subclass. No raw
   `Exception` / `RuntimeError` / `ValueError` escapes a package boundary.
   (Known offenders to fix: `EmbeddingError(Exception)`,
   `MaxToolCallsExceeded(RuntimeError)`, `RailAbortError`.)
2. **Actionable messages** — every `raise` has a `hint` (or a message that
   already states the fix). This is the headline metric.
3. **Context** — errors carry the involved agent/tool/model/field via `context=`.
4. **No silent swallow** — every `except` re-raises, handles meaningfully, or logs
   with context. Zero `except: pass`. (29 sites today.)
5. **Chaining** — `raise New(...) from exc`; never drop the cause.
6. **Cancellation safety** — per the rule above.
7. **Async-noise collapse** — boundaries unwrap ExceptionGroups.
8. **Resilience (co-goal)** — retries+backoff+timeouts on I/O
   (provider/Redis/MCP/HTTP); resource cleanup via `async with`/`finally` (no
   leaked connections or tasks); partial-failure handling in parallel/swarm paths.
   Exhaustion surfaces as an actionable `ProviderError`/`TimeoutError`, never a raw
   library exception.
9. **Consistent logging** — follow the two logging patterns in CLAUDE.md; log at
   the point of handling, not at every re-raise.

## Conventions to preserve

- Keep deprecated aliases / old spellings (repo-wide convention from the DX work).
- Match the existing teaching-error voice (see `swarm.py`'s "did you mean" hints,
  `loader.py` "Available agents: [...]").
- Run the package's tests after every change; full `uv run pytest` after each wave.
