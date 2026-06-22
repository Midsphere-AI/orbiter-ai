# exo-eval — Error DX & Resilience Audit

## Counts
- raise sites: 14
- error classes total / not inheriting ExoError: 3 total / 1 offender
  - `EvalSyncError(Exception)` at `sync.py:45`
- `except Exception` sites: 5 ; swallow-and-pass: 0 ; drop-cause: 4
- CancelledError handlers: 0 (no explicit handling; `except Exception` doesn't catch it — safe by default)
- I/O call sites lacking timeout/retry: 5
  - `LLMAsJudgeScorer.score` — judge call, no timeout/retry (`llm_scorer.py:99`)
  - `GeneralReflector.analyze` — judge call, no timeout/retry (`reflection.py:183`)
  - `RefinementLoop._execute` — execute_fn call, no timeout/retry (`ralph/runner.py:287`)
  - `RefinementLoop._analyze` — scorer calls, no timeout/retry (`ralph/runner.py:307`)
  - `PlatformSync._push_one_score` / `push_scores` — external platform client, no timeout/retry (`sync.py:186-194`)

---

## Findings (prioritized)

### P0

**[P0] | #7 (partial-failure) | base.py:168-172 | Case failure is silently dropped with only a warning log — no failed-case record**

In `Evaluator.evaluate`, failed tasks are caught via `isinstance(result, BaseException)`, logged with a bare `%s` format (no case_id, no cause repr), and then `continue`-d. The case is not recorded as a failed `EvalCaseResult`; it simply disappears from `case_results`. A caller running 100 cases and getting back 97 results has no way to know which 3 failed or why.

Fix: create a sentinel `EvalCaseResult` with a special failed status and attach the exception as `details={"error": str(exc), "exc_type": type(exc).__name__}`, or raise `EvalError` after the full run with the list of failed case IDs and their causes.

```python
# current (drops cause, loses case_id):
if isinstance(result, BaseException):
    logger.warning("Scorer task failed, skipping case: %s", result)
    continue
```

---

**[P0] | #5 (chaining) | ralph/runner.py:290-292 | `_execute` converts exceptions to strings — cause is permanently lost**

`RefinementLoop._execute` wraps the execute_fn call in `except Exception as exc: return (str(exc), False)`. The exception is stringified and becomes the `output` for that iteration. The real exception type, traceback, and `__cause__` are all discarded. If the judge LLM later reflects on the failure, it sees only a string like `"HTTPStatusError: 503"` with no structured context.

The same pattern appears in `stream()` at `ralph/runner.py:248-251`.

Fix: log the exc with `exc_info=True` and re-raise into `state.record_failure()`, or at minimum store the exception in `LoopState` so it can be surfaced in `RefinementResult`.

---

**[P0] | #4 (no silent swallow) | ralph/runner.py:307-310 | Scorer exceptions in `_analyze` are silently swallowed with only a WARNING log**

```python
except Exception as exc:
    logger.warning("Scorer failed case=%s: %s", case_id, exc, exc_info=True)
```

The scorer name is not logged, the exception is not chained, and no failed scorer entry is recorded in the returned `scores` dict. A scorer silently returning no score is indistinguishable from a scorer returning 0.0, which can corrupt pass@k and score_threshold logic.

Fix: record a `ScorerResult` with `score=0.0` and a details dict carrying the error string, similar to how `TimeCostScorer` surfaces its own errors.

---

### P1

**[P1] | #1 (taxonomy) | sync.py:45 | `EvalSyncError` inherits `Exception`, not `ExoError`**

`EvalSyncError` is the public error for dataset-pull and score-push failures — the most likely errors a developer will encounter when wiring up Langfuse/Braintrust/LangSmith. It inherits bare `Exception`, so it gets no `context=`, `hint=`, or `doc=` fields.

Fix: `class EvalSyncError(ExoError): ...` and add hints like "Pass a backend that exposes `score_client()` or supply `score_client=` directly to `PlatformSync`."

---

**[P1] | #2 (actionable messages) | llm_scorer.py:93-97 | `LLMAsJudgeScorer.score` raises bare `ValueError` with no `hint=` and wrong taxonomy**

```python
raise ValueError(msg)
```

The message is decent but the class is `ValueError` (not `EvalError`), so it escapes the package boundary as a raw stdlib exception. The hint text is embedded in the message but not in the `hint=` field, making it invisible in structured logging.

Fix: `raise EvalError(msg, hint="Pass judge=(async callable) at construction time.")`.

---

**[P1] | #2 (actionable messages) | scorers.py:33 | `FormatValidationScorer.__init__` raises bare `ValueError`**

```python
raise ValueError(msg)
```

The message lists valid formats, which is good, but the class is raw `ValueError`. Should be `EvalError` to stay on-taxonomy.

---

**[P1] | #3 (context) | base.py:125-129 | Config-validation `EvalError` raises have no `context=` or `hint=`**

```python
raise EvalError(msg)  # "parallel must be >= 1"
raise EvalError(msg)  # "repeat_times must be >= 1"
```

Both use `EvalError` (good taxonomy) but carry no `hint=` and no context about the caller's values. Should include the bad value in context and a corrective hint.

Fix:
```python
raise EvalError(
    f"parallel must be >= 1, got {parallel}",
    context={"parallel": parallel},
    hint="Set parallel to at least 1 (e.g. parallel=4).",
)
```

---

**[P1] | #3 (context) | ralph/config.py:56-58 | `ValidationConfig.__post_init__` raises bare `ValueError`, no context or hint**

```python
raise ValueError(msg)
```

`StopConditionConfig.__post_init__` at `config.py:82-84` has the same issue. Both should be `EvalError` with `context={"value": ...}` and a corrective `hint=`.

---

**[P1] | #2+#3 | sync.py:119-121 and sync.py:169-172 | `EvalSyncError` raises carry good messages but no `context=` or `hint=`**

```python
raise EvalSyncError(
    "No dataset client is configured.  Pass a backend that exposes ..."
)
```

The message is actionable but the hint text is buried inside the message string rather than in `hint=` (once `EvalSyncError` inherits `ExoError`). Once taxonomy is fixed, move the "how to fix" sentence to `hint=`.

---

**[P1] | #7 (partial-failure) | ralph/runner.py:220 | `stream()` raises bare `ValueError` for missing `stream_execute_fn`**

```python
raise ValueError("stream_execute_fn required for streaming")
```

Should be `EvalError` with `hint="Use RefinementLoop.from_agent(agent, scorers) to wire both execute_fn and stream_execute_fn automatically."`.

---

**[P1] | #5 (chaining) | ralph/runner.py:248-251 | `stream()` `except Exception` converts exc to string, drops cause**

Same pattern as `_execute`: `output = str(exc)` destroys the cause. Add `logger.warning("...", exc_info=True)` at minimum.

---

**[P1] | #8 (resilience) | llm_scorer.py:99 | LLM judge call has no timeout**

`response = await self._judge(prompt)` — if the judge callable hangs (network issue, provider timeout), the entire `RefinementLoop` iteration hangs indefinitely. No `asyncio.wait_for` wrapper.

Fix: add an optional `timeout: float = 0.0` parameter to `LLMAsJudgeScorer` and wrap the judge call:
```python
if self._timeout > 0:
    response = await asyncio.wait_for(self._judge(prompt), timeout=self._timeout)
else:
    response = await self._judge(prompt)
```

---

**[P1] | #8 (resilience) | reflection.py:183 | GeneralReflector judge call has no timeout**

`response = await self._judge(prompt)` — same as above. No timeout, no retry, any network failure propagates as an uncaught exception through `_learn` into the loop.

---

### P2

**[P2] | #9 (logging) | base.py:171 | Warning log drops case_id**

```python
logger.warning("Scorer task failed, skipping case: %s", result)
```

`result` is the exception. The case_id is not logged, making it impossible to correlate the failure to the dataset. The tasks are built as `_run(case, r)` — the case_id is computed inside `_run` and is not available at the gather level. Consider restructuring so case_id is captured in the coroutine name or a wrapper.

---

**[P2] | #2 | trajectory.py:144-196 | `TrajectoryItem.__post_init__` raises bare `ValueError` for style conflicts**

These are constructor-guard errors (good pattern), but they're `ValueError` not `EvalError`/`TrajectoryError`. Should be `TrajectoryError` with `hint="Use one construction style: either flat kwargs or grouped namespace objects, not both."`.

---

**[P2] | #8 | sync.py:186-194 | `_push_one_score` has no timeout around platform client calls**

Platform push calls (`client.score(...)` / `client.log_score(...)`) are synchronous duck-typed calls with no timeout. If the platform client blocks, `push_scores` hangs.

---

**[P2] | #2 | trajectory_scorers.py:44-49 | `get_scorer` raises bare `KeyError` with no hint**

```python
return _SCORER_REGISTRY[name]
```

A missing name raises `KeyError: 'bad_name'` with no hint about available scorers.

Fix:
```python
if name not in _SCORER_REGISTRY:
    raise EvalError(
        f"Scorer {name!r} is not registered.",
        context={"name": name},
        hint=f"Available scorers: {list_scorers()}. Register custom scorers with @scorer_register(name).",
    )
return _SCORER_REGISTRY[name]
```

---

## Resilience gaps

| Site | File:Line | System |
|---|---|---|
| LLM judge call — no timeout/retry | `llm_scorer.py:99` | All LLM-judge scorers; hangs entire eval case |
| Reflector judge call — no timeout/retry | `reflection.py:183` | GeneralReflector in every refinement loop iteration |
| execute_fn call — cause destroyed on failure | `ralph/runner.py:287-292` | RefinementLoop Phase 1; errors become opaque strings |
| Scorer calls in `_analyze` — silently dropped | `ralph/runner.py:307-310` | RefinementLoop Phase 2; corrupts score accounting |
| Platform push — no timeout | `sync.py:186-194` | PlatformSync push_scores; sync to Langfuse/Braintrust/LangSmith |
| **KEY** Case failures in `Evaluator.evaluate` — no failed-case record | `base.py:168-172` | Core eval loop; failed cases silently disappear from results |

**Partial-failure verdict:** The `Evaluator` class does *not* abort on a single failure (uses `gather(..., return_exceptions=True)`), which is correct. However, the failure is silently discarded rather than recorded as a failed case, so the caller cannot detect, triage, or retry it. The run continues but the failure is invisible.

The `RefinementLoop` similarly continues on iteration failure but converts the exception to a string, permanently losing cause information.

---

## Effort estimate

**M** — Six targeted changes cover all P0/P1 issues: fix `EvalSyncError` taxonomy (1 line), fix 4 `ValueError` raises to `EvalError`, record failed cases in `Evaluator.evaluate` (replaces ~4 lines), add `exc_info=True` logging in `_execute`/`stream` (2 lines each), add optional judge timeout to `LLMAsJudgeScorer` and `GeneralReflector`, and fix `get_scorer` to raise actionable error.
