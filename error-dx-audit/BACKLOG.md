# Error DX & Resilience — Master Backlog (Phase 1 result)

Aggregated from the 18 per-package audits in this directory, scored against
`error-dx-charter.md`. Primary goal: legible/contextual/**actionable** errors.
Co-goal: resilience. Sorted for the Phase 2 fix waves.

## Headline numbers

- **18/18 packages audited.** ~470 raise sites reviewed.
- **~45 P0 findings** (silent failures, swallowed cancellation, security fail-open,
  process/connection leaks, raw tracebacks at the user).
- Recurring structural defects (fix once, apply everywhere):
  1. **Structured-field drop** — several `ExoError` subclasses (`ModelError`,
     `GuardrailError`, `RetrievalError`, `RailAbortError`) define their own
     `__init__` that does NOT forward `context=/hint=/doc=` to `ExoError`, so the
     whole teaching mechanism is dead in those packages. **This is the single
     highest-leverage fix** — it unblocks actionable messages everywhere.
  2. **Taxonomy leaks** — raw `ValueError`/`RuntimeError`/`KeyError`/`ImportError`
     cross package boundaries (exo-core config/agent, exo-models `EmbeddingError`,
     exo-memory, exo-eval `EvalSyncError`, exo-train ×10, exo-observability ×9,
     exo-context `state.py:49`).
  3. **`except Exception`/`BaseException` that can eat `CancelledError`** on async
     paths (exo-core `agent.py:2851`, exo-a2a `server.py:337`).
  4. **No async boundary cleanup** — CLIs/server stringify raw `ExceptionGroup`s
     instead of `unwrap_exception_group` (exo-cli, exo-mcp-cli, exo-server).
  5. **Missing timeouts on I/O** — MCP sessions, provider calls, Redis, pgvector,
     embeddings, subprocess all lack `asyncio.wait_for`.

## P0 findings by package (must-fix)

**exo-core**
- `agent.py:2851` — `except BaseException` on the tool-execution hot path swallows
  `CancelledError` into a tool-error string. Split `except asyncio.CancelledError: raise` first.
- +4 more P0 (raw `ValueError` boundary escapes in config/agent/parallel; drop-cause sites).

**exo-guardrail** *(security)*
- `llm_backend.py:186` — JSON parse failure **fails OPEN** (returns SAFE). Must return HIGH.
- `llm_backend.py:190` — unknown `risk_level` string falls back to SAFE. Fail safe.
- `llm_backend.py:193` — `has_risk=False` trusted even when risk_level non-SAFE.

**exo-distributed** *(resilience-critical)*
- `worker.py:347` — `task.timeout_seconds` stored but never enforced → stuck agent runs forever.
- `worker.py:515` — cascading Redis failure in the FAILED path leaves task stuck in PEL permanently.
- `broker.py:118` — `nack()` empty-xrange branch still `xack`s → task silently vanishes.
- `temporal.py:204,236` — bare `RuntimeError` crosses boundary, no cause chain.

**exo-models**
- `ModelError.__init__` swallows `hint=/context=` → no provider error can be actionable.
- 3 provider I/O sites + Veo **unbounded polling loop** with no timeout.

**exo-memory**
- `encrypted.py:140` — `except Exception: continue` silently drops undecryptable items, no log.
- `backends/{sqlite,postgres}.py` — `RuntimeError` escapes; Postgres store has zero try/except (raw asyncpg leaks).

**exo-sandbox**
- `tools.py:381` & `tools.py:574` — timeout `proc.kill()` without `await proc.wait()` → **zombie subprocess leak**.
- `e2b.py:185` — `_kill_sandbox()` swallows failure → leaked remote sandbox.
- `kubernetes.py:107` — kubeconfig loader exception swallowed, cause dropped.

**exo-mcp**
- `client.py:309/344/370` — handshake/list_tools/call_tool have **no asyncio timeout** → hang forever.
- `execution.py:112` — raw `ValueError` on unknown transport escapes boundary.
- `connect_all()` serial, no partial-failure aggregation.

**exo-observability**
- `tracer.py:158–261` — 8 hook callbacks unguarded; `HookManager` does NOT suppress → a telemetry SDK error **crashes the user's agent run** (fail-open violation).

**exo-eval**
- `base.py:168` — failed eval cases silently dropped (no failed-case record): 100 in → 97 out, caller blind.
- `ralph/runner.py:290` — exceptions stringified, cause chain destroyed.
- `ralph/runner.py:307` — scorer failures swallowed, corrupts score_threshold logic.

**exo-search**
- `pipeline.py` (×6) — classifier/verifier re-raise raw `BaseException`/`ExceptionGroup`, no degrade.
- `query_planner.py:315` + `searxng.py:295` — bare `asyncio.gather` (no `return_exceptions`) → one failure crashes all parallel searches.

**exo-cli**
- `main.py` — 7 of 8 `asyncio.run()` calls unguarded → raw traceback/ExceptionGroup dumped at user; Ctrl-C ugly.

**exo-a2a**
- `server.py:337` — `_generate()` doesn't catch `CancelledError` → client disconnect leaves task stuck WORKING forever.
- `client.py:97,137,183` — `JSONDecodeError`/`ValidationError` escape the `httpx.HTTPError` catch as raw.

**exo-server**
- `app.py:204` / `streaming.py:50` — agent exceptions stringified to client (raw ExceptionGroup) or silently swallowed; client disconnect leaks the agent run.

**exo-context**
- `info.py:135` — `except Exception: pass` eats token-tracker errors, drops trajectory silently.
- `workspace.py:310` — blocking file I/O on the event loop; cancel mid-persist leaves inconsistent state.

**exo-mcp-cli**
- `connection.py:122` (+5 command fallbacks) — `str(ExceptionGroup)` renders garbled repr into error message; needs `unwrap_exception_group`.

**exo-skills**
- `skills.py:194` — no per-file isolation; one unreadable skill aborts the whole scan.
- `_clone_github` — git `CalledProcessError`/`FileNotFoundError` escapes unwrapped.

**exo-train** *(experimental)*
- `verl.py:417` — optional-dep `ImportError` not actionable; install hint stripped by re-wrap.
- `verl.py:493` — `raise ... from None` drops cause; GPU/OOM errors surface as bare strings.

**exo-retrieval**
- No P0 swallows, but `RetrievalError` has a broken constructor (takes `operation/details`, ignores ExoError fields); all RAG LLM/DB calls unguarded (P1-heavy).

## Phase 2 — fix waves (dependency order)

- **Wave 0 (do first, in core):** fix the structured-field-drop pattern. Establish a
  canonical `ExoError.__init__` forwarding contract + a tiny `retry`/`timeout` helper
  and a shared "fail-open-but-log" idiom. Everything downstream depends on this.
- **Wave A — exo-core:** CancelledError hot-path fix, config/agent taxonomy, boundary
  unwrap in run/run.stream.
- **Wave B — exo-models, exo-context, exo-memory, exo-mcp, exo-sandbox, exo-observability, exo-guardrail:**
  parallel. Guardrail fail-open + observability hook-guard + sandbox zombie leak are P0.
- **Wave C — exo-retrieval, exo-search, exo-cli, exo-distributed, exo-eval, exo-a2a, exo-skills, exo-mcp-cli:**
  parallel. Distributed PEL/timeout triad is the heaviest (L). **DONE** — all 8 packages
  fixed (P0/P1 + cheap P2s), each green individually and combined (Wave C 1947 passed,
  exo-core 1820 passed; 0 new ruff/pyright). NOTE: the four `temporal.py` findings here are
  **MOOT** — that file was deleted and is being rebuilt as a `temporal/` package by the
  separate Temporal-parity effort; that subtree was left untouched.
- **Wave D — exo-server, exo-train:** parallel (both experimental, both small/S).

## Effort roll-up

L: exo-core, exo-distributed. M: models, memory, mcp, context, retrieval, search,
eval, sandbox, observability, skills, train. S: guardrail, a2a, mcp-cli, server.
Rough total: a handful of focused days, front-loaded by the Wave-0 shared fixes.
