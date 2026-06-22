# exo-search — Error DX & Resilience Audit

## Counts
- raise sites: 18 (pipeline.py: 8, config.py: 7, embeddings.py: 3)
- error classes total / not inheriting ExoError: 0 custom classes defined / all `raise` sites use raw `ValueError` or re-raise raw `BaseException` (offenders: `pipeline.py:47` `ValueError`, `config.py:185,212,238,262,280,295,334` all `ValueError`)
- `except Exception` sites: 21 ; swallow-and-pass: 0 ; drop-cause: 11 (sites that catch `Exception as exc` and don't chain via `from exc`)
- CancelledError handlers: 0 (no explicit handling — `query_planner.py:315` catches `except Exception:` in `_timed_agents`, which is fine for `Exception` but `asyncio.wait_for` raises `TimeoutError` which IS an `Exception` — the `asyncio.CancelledError` is `BaseException` so actually safe here)
- I/O call sites lacking timeout/retry: 3 (`_fetch_via_jina` self-hosted path at `web_fetcher.py:275` uses `timeout=15` but no retry; `_fetch_page_fallback` at `web_fetcher.py:313` timeout=30 but no retry; `search_and_collect` at `searxng.py:295` calls `asyncio.gather(*tasks)` — no `return_exceptions`, propagates raw exception)

## Findings (prioritized)

### P0 — Crash / silent corruption

**[P0] checklist#7 | pipeline.py:105,122,231,351,373,508 | classifier failure raises raw `BaseException` with no context**
When the classifier LLM fails (network error, provider crash), the bare `raise classification_r` re-raises whatever the agent framework threw — an `ExceptionGroup`, a provider-specific `httpx.HTTPError`, or similar asyncio noise — with zero context about which pipeline stage failed or what the query was. The whole search crashes and the developer sees a wall of async frames.
Fix: wrap in `SearchError(f"Classifier failed for query {query!r}", context={"stage": "classifier", "mode": mode, "query": query}, hint="Check model credentials and network. Try mode='speed' to skip the classifier for fast fallback.") from unwrap_exception_group(classification_r)`.

**[P0] checklist#7 | pipeline.py:231,508 | `verify_r` (citation verifier) re-raised raw after `return_exceptions=True`**
When `verify_citations` throws (LLM failure, parse error), the raw exception is re-raised without context. Unlike the `contradict_r` case (which is gracefully degraded to `None`), a citation verifier failure aborts the entire pipeline even though the writer has already produced an answer.
Fix: degrade gracefully like contradiction detection does — log warning, set `verification = CitationVerification()`, and continue rather than crashing.

**[P0] checklist#5 | pipeline.py:105,122,351,373 | cause dropped on re-raise**
`raise classification_r` and `raise verify_r` re-raise the exception directly without `from` chaining. The cause IS the exception itself so this doesn't lose it technically, but it also loses the opportunity to wrap it in an `ExoError` with context. The raw exception often arrives as a single-child `ExceptionGroup` from the asyncio gather that should be unwrapped first via `unwrap_exception_group`.
Fix: `raise SearchError(...) from unwrap_exception_group(exc)`.

**[P0] checklist#7 | query_planner.py:315 | `_timed_agents()` swallows ALL exceptions including timeout**
```python
except Exception:
    return []
```
`asyncio.wait_for` raises `asyncio.TimeoutError` (subclass of `TimeoutError`, subclass of `Exception`). This is intentionally swallowed — the parallel agents are "best-effort." However, ANY exception from `parallel_research` is silently dropped here with no log. If `adaptive_research` (the structured path) also fails and `asyncio.gather` on line 328 raises, there's no `return_exceptions=True` — the error propagates raw.
Fix: add `_log.debug("timed agents failed/timed out: %s", exc)` in the except clause; add `return_exceptions=True` to the `asyncio.gather(structured_task, agent_task)` call at line 328 and handle failures.

**[P0] checklist#7 | searxng.py:295-310 | `search_and_collect` uses `asyncio.gather` without `return_exceptions=True`**
```python
tasks = [asyncio.to_thread(_search, q, categories, engines) for q in queries]
all_results = await asyncio.gather(*tasks)
```
If one query thread raises, the entire gather raises, collapsing all parallel searches. This propagates as a raw `ExceptionGroup` to callers (`deep_researcher._execute_step`, `query_planner.adaptive_research`). Those callers don't expect raw groups.
Fix: add `return_exceptions=True`; filter out exception entries in the result loop.

### P1 — Unactionable / missing context

**[P1] checklist#1,2 | config.py:185,212,238,262,280,295,334 | 7× raw `ValueError` escape package boundary**
All 7 `ValueError` raises in `SearchConfig.__post_init__` (namespace conflict detection and source validation) inherit raw `ValueError`, not `ExoError`. They have good messages but no `hint=` pointing to the specific fix, and no `context=` with the conflicting fields.
Fix: replace with `SearchConfigError(ExoError)` subclass; add `context={"conflicting_fields": conflicts}` and `hint="Pass either the grouped config or flat kwargs, not both."`.

**[P1] checklist#2 | pipeline.py:47 | `_normalize_mode` raises raw `ValueError` without `ExoError`**
Good teaching message but escapes the boundary as a raw `ValueError`. A user calling `search("q", mode="fasst")` sees `ValueError: Unknown search mode: 'fasst'. Valid modes are: ...` — correct content but wrong class.
Fix: `raise SearchError(f"Unknown search mode: {mode!r}. Valid modes are: {valid_list}. Example: search('query', mode='balanced')", hint="Use one of the valid mode strings listed above.")`.

**[P1] checklist#3 | deep_researcher.py:184 | plan parse failure loses context**
```python
except Exception as exc:
    _log.warning("plan parse failed (%s), falling back to single step", exc)
    plan = ResearchPlan(steps=[...])
```
Falls back silently. The `exc` cause is logged but never attached to anything. The developer sees "plan parse failed" in logs but no structured error or context about the model used.
Fix: This fallback is reasonable (graceful degradation), but should log `context={"model": config.model, "raw_output": raw[:200]}`.

**[P1] checklist#3 | deep_researcher.py:312 | extraction failure logs step_id but drops model context**
```python
except Exception as exc:
    _log.warning("extraction failed for step %s: %s", step.step_id, exc)
    extracted_info = "; ".join(r.title for r in results[:5])
```
Step ID is logged but not the model used (`config.fast_model`), which is the most actionable field for debugging.
Fix: add `model=config.fast_model` to warning log arguments.

**[P1] checklist#2,3 | researcher.py:417 | `run(agent, ...)` in `_run_worker` failure logged but no search angle context**
`parallel_research` gathers worker failures and logs `"sub-researcher '%s' failed: %s"` but the logged message for the failure is whatever the underlying `ExoError`/provider error says, with no angle or model attached.
Fix: log angle + model in the failure message: `_log.warning("sub-researcher '%s' (model=%s) failed: %s", active_angles[i], research_model, result)`.

**[P1] checklist#2,3 | classifier.py:51 | classifier parse failure logged but original output lost**
```python
except Exception:
    _log.warning("classifier parse failed, using fallback")
```
The raw `result.output` (the malformed LLM output) is not logged. Makes debugging impossible without source-level tracing.
Fix: `except Exception as exc: _log.warning("classifier parse failed (%s), raw=%r, using fallback", exc, result.output[:200])`.

**[P1] checklist#2 | query_planner.py:170 | parse failure swallows exception entirely**
```python
except Exception:
    plan = QueryPlan(queries=[query], sufficient=False)
```
The exception is swallowed with no logging at all. A malformed LLM response causes a silent fallback.
Fix: `except Exception as exc: _log.warning("query_plan parse failed: %s — falling back to raw query", exc)`.

**[P1] checklist#5 | web_fetcher.py:63,74,88 | PDF extraction failures drop cause**
```python
except Exception as exc:
    _log.warning("pdf download failed for %s: %s", url, exc)
    return ""
```
All three PDF extraction failure paths log the exception message as a string but don't chain `__cause__`. Fine for an internal tool returning `""`, but the URL is the key context and `exc` type could matter (timeout vs SSL vs 403).
Fix: minor — add `exc_info=True` to preserve the traceback in the log.

**[P1] checklist#9 | _utils.py:7-8 | logging inconsistency — uses `import logging` not `get_logger`**
`_utils.py` uses `import logging; logger = logging.getLogger(__name__)` while all other exo-core-internal files use `from exo.observability.logging import get_logger`. The `_utils` module is internal to the package and should use the correct pattern.
Fix: `from exo.observability.logging import get_logger  # pyright: ignore[reportMissingImports]` → `_log = get_logger(__name__)`.

### P2 — Polish

**[P2] checklist#2 | serper.py:71 | failure returns `[]` with no hint about API key validity**
`except Exception as exc: _log.warning("serper request failed: %s", exc); return []` — this is fine for resilience but the log doesn't distinguish 401 (bad key) from 429 (rate limit) from timeout. Structured callers can't act differently.
Fix: check `exc.code` / `str(exc)` and log at `error` level for 401, `warning` for 429/timeout.

**[P2] checklist#2 | jina.py:75,130 | Jina failures return `[]`/`""` with no hint about key vs network**
Same pattern as serper — all failure modes collapsed to empty result.

**[P2] checklist#7 | pipeline.py:191,456 | `suggest_task` created but never cancelled on pipeline failure**
`suggest_task = asyncio.create_task(generate_suggestions(...))` — if the pipeline fails between task creation and `await suggest_task`, the task leaks and runs in the background. No `try/finally` to cancel it.
Fix: wrap the pipeline body in `try/except` with `suggest_task.cancel()` in a `finally` block.

**[P2] checklist#3 | server.py:382 | string-matching on raw error for user-facing message**
```python
if "Invalid JSON" in raw or "Extra data" in raw:
    msg = "The AI model returned malformed output..."
```
Fragile string matching on error text. Will miss novel error messages. Should use `isinstance` checks on structured `ExoError` subclasses once those are in place.

**[P2] checklist#1 | embeddings.py:135,141,150 | `EmbeddingError` from exo-models escapes exo-search boundary**
`_GeminiEmbeddings.embed_batch` raises `EmbeddingError` which is caught in `rerank_search_results` at line 251 (`except Exception as exc`) and degraded to keyword scoring. The catch is correct but `EmbeddingError` is not an `ExoError` subclass (tracked separately in exo-models audit). This is cross-package boundary leakage.

## Resilience gaps

| Site | System | Issue |
|------|--------|-------|
| `searxng.py:295` `search_and_collect` | SearXNG/Serper/Jina | `asyncio.gather(*tasks)` without `return_exceptions=True` — one query failure crashes all parallel searches |
| `query_planner.py:328` `asyncio.gather(structured_task, agent_task)` | parallel research | Missing `return_exceptions=True`; if structured path raises, the exception propagates raw as an asyncio group |
| `web_fetcher.py:480-499` `_timed_fetch` | Jina reader | `asyncio.wait_for` exceptions from the PDF path (e.g., `TimeoutError`) are NOT caught — propagates to `asyncio.gather(..., return_exceptions=True)` in `enrich_results`, which handles it correctly, but the outer `_timed_fetch` may also raise for non-PDF paths on line 489 if `asyncio.wait_for` itself throws non-`TimeoutError` |
| `pipeline.py:191,456` `suggest_task` | suggestion generator | Background task is never cancelled if pipeline fails mid-flight; task leaks |
| `_fetch_page_fallback` `web_fetcher.py:313` | direct HTTP fetch | `timeout=30` but no retry on transient failures (vs. SearXNG which has 3 retries) |
| `jina.py:124` `jina_reader_fetch` | Jina Reader | Retries on `_RETRYABLE` string matches — but `TimeoutError` string may not match `"timed out"` on all platforms; fragile retry detection |

## Effort estimate

**M** — The critical P0s (raw re-raise in pipeline, `search_and_collect` missing `return_exceptions`, unlogged query planner swallow) are mechanical 1-3 line fixes; the P1 taxonomy work (add a `SearchError(ExoError)` subclass, thread it through `ValueError` raises in config and pipeline) is ~30-40 lines. No architectural changes needed. The suggest_task leak and the `_timed_agents` logging gaps are also small. One wave of fixes covers all P0s and most P1s; a second wave completes the P2 polish.
