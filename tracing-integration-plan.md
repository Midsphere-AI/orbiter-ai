# Observability & Eval Integration Plan (Langfuse / LangSmith / OTLP / Phoenix / Braintrust)

**Status:** ✅ IMPLEMENTED on `chore/distribution-cleanup` (Phases 1–4 complete, suite green: 6325 passed).
**Author context:** Launch-readiness push — make tracing + eval integrations deep across exo from day one.

## Implementation summary (what shipped)
- **Spine** (`exo-observability/observability/tracer.py`, `backends/`): `Tracer` registers on the agent
  hook lifecycle and emits a nested OTel span tree (`agent.run` → `gen_ai.chat` → `tool.execute` →
  `context.compress`). `Agent(tracing=...)` + `EXO_TRACING` + env auto-detect. `TracingBackend` protocol +
  generic `OTLPBackend` + in-process `MemoryBackend`. **Tracer binds to the concrete provider returned by
  `_ensure_provider`** (not the global `get_tracer`) — robust against stale/proxy global providers.
- **Cost** (`observability/cost.py`): `CostTracker`, `estimate_cost`, 16-model `PRICE_TABLE`,
  `stamp_cost_on_current_span`; per-call cost stamped onto the `gen_ai.chat` span via `COST_*` semconv.
- **Vendor adapters** (`backends/{langfuse,langsmith,phoenix,braintrust}.py`): self-register on import;
  OTLP exporter wiring + native `score_client()`/`dataset_client()` seams. Behind `exo-ai[<vendor>]` extras.
- **Eval bridge** (`exo-eval/sync.py`): `PlatformSync` — pull dataset → `Evaluator` → push scores
  (duck-typed Langfuse `.score` / Braintrust `.log_score` / LangSmith datasets); `export_trajectory`.
- Degrades to no-op when OTel/extras absent; bare `Agent(...)` stays zero-overhead.

---
**Original plan below (for reference).**

---

## 1. Goal

Make exo observable end-to-end with first-class hooks into the eval/observability ecosystem
(Langfuse, LangSmith, OTLP-native backends, Phoenix/Arize, Braintrust), without bolting a separate
SDK onto each subsystem. One switch should instrument the **whole stack** — agents, swarms,
sub-agents, tools, RAG retrieval, guardrail rails, memory ops — because they all flow through the
same lifecycle hooks.

**Locked decisions:**
- **Architecture:** OTel spine first (vendor-neutral), native adapters on top.
- **Scope:** cost tracking + eval dataset sync + Phoenix/Arize + one more eval platform (Braintrust).
- **Default state:** **on-by-default when a backend is detected** (env keys present) — see §6.
- **Process:** parallel sonnet sub-agents, one per vendor adapter, after the spine lands.

---

## 2. Current state (what exists today)

### exo-observability (`packages/exo-observability/src/exo/observability/`)
- `tracing.py` — OTel-based `SpanLike` protocol, `NullSpan` no-op fallback, `span()`/`aspan()`/`@traced`.
  Uses the global OTel tracer (`get_tracer("exo")`) when `opentelemetry-api>=1.20` is present; no-ops otherwise.
- `semconv.py` — 40+ GenAI/Agent/Tool/Task/**Cost** semantic-convention constants already reserved
  (`GEN_AI_USAGE_INPUT_TOKENS`, `COST_*`, `AGENT_NAME`, `TOOL_NAME`, …).
- `config.py` — `TraceBackend` enum (`OTLP`/`MEMORY`/`CONSOLE`), `TracingConfig(enabled, backend, endpoint, sample_rate)`.
- `metrics.py` — dual-path OTel/in-memory `MetricsCollector`, `record_agent_run()`, `record_tool_step()`, `Timer`.
- `logging.py`, `health.py` — structured logging + health registry.

**Gaps:** spans are never emitted during a run (no lifecycle wiring); `TracingConfig.endpoint`/`sample_rate`
are read-only with **no exporter behind them**; `CostTracker` referenced in README but **not implemented**;
no Langfuse/LangSmith/anything adapters.

### exo-core hooks/events (`packages/exo-core/src/exo/`)
- `hooks.py` — `HookPoint` enum: `START`, `FINISHED`, `ERROR`, `PRE_LLM_CALL`, `POST_LLM_CALL`,
  `PRE_TOOL_CALL`, `POST_TOOL_CALL`, `CONTEXT_WINDOW`. `HookManager.add/run/has_hooks`. Hooks are async,
  run sequentially in registration order; exceptions abort the run.
- Hook payloads (verified):
  | Hook | Payload |
  |---|---|
  | `START` | `agent`, `input` |
  | `PRE_LLM_CALL` | `agent`, `messages` |
  | `POST_LLM_CALL` | `agent`, `response` (`response.usage`, `.tool_calls`, `.finish_reason`) |
  | `PRE_TOOL_CALL` | `agent`, `tool_name`, `arguments` |
  | `POST_TOOL_CALL` | `agent`, `tool_name`, `result` (`.content`, `.error`, `.success`) |
  | `FINISHED` | `agent`, `output` |
  | `ERROR` | `agent`, `error` |
  | `CONTEXT_WINDOW` | `agent`, `messages`, `budget`, `current_fill_ratio` |
- Registration: `Agent.__init__` builds `self.hook_manager = HookManager()` and adds any `hooks=[(point, fn)]`.
- Dispatch sites: LLM call `Agent._call_llm()` (`agent.py` ~2617-2675), tools `Agent._execute_tools()`
  (`agent.py` ~2677-2781, parallel via `asyncio.TaskGroup`).
- `Usage` model (`types.py` ~170): `input_tokens`, `output_tokens`, `total_tokens`. Populated on
  `ModelResponse.usage`, aggregated into `RunResult.usage` and `state.total_usage`.
- Streaming events (`run.stream()`): `UsageEvent` (per-step usage + model), `StepEvent`, `ToolResultEvent`
  (duration_ms), `ErrorEvent`, etc. — all carry `agent_name` for attribution.
- `token_counter.py` — provider-aware `TokenCounter` for cost estimation.

### exo-eval (`packages/exo-eval/src/exo/eval/`)
- `Evaluator.evaluate(target, dataset)` — async, semaphore-bounded parallelism, pass@k, applies all `Scorer`s + `EvalCriteria`.
- `Scorer` ABC + registry (`@scorer_register`, `get_scorer`, `list_scorers`); rule-based + LLM-judge scorers.
- Data models: `ScorerResult(scorer_name, score, status, details)`, `EvalCaseResult`, `EvalResult(case_results, summary, pass_at_k)`.
- `TrajectoryDataset` — capture + `to_json()`/`to_csv()`; `TrajectoryItem` (SAR pattern).
- Refinement loop (Ralph) with `RefinementIterationEvent`/`RefinementStopEvent`.

**Gaps:** no external platform publishing (no Langfuse scores/datasets, no LangSmith `evaluate()`),
no link between eval results and trace IDs.

---

## 3. Architecture: OTel spine + native adapters

```
                         ┌─────────────────────────────────────┐
   Agent / Swarm / Tools │  exo-core hooks (already exist)      │
   / RAG / Guardrail     │  START PRE/POST_LLM PRE/POST_TOOL …  │
                         └───────────────┬─────────────────────┘
                                         │ registers callbacks
                         ┌───────────────▼─────────────────────┐
   LAYER 1 (spine)       │  Tracer  (observability/tracer.py)   │
   vendor-neutral OTel   │  builds span tree, stamps semconv    │
                         │  + CostTracker stamps COST_* attrs   │
                         └───────────────┬─────────────────────┘
                                         │ OTLP spans
            ┌────────────────┬───────────┼────────────┬──────────────┐
   LAYER 2  │ langfuse.py    │ langsmith │ phoenix.py  │ braintrust   │  backends/
   adapters │ OTLP+scores    │ OTLP+ds   │ OTLP only   │ OTLP+scores  │  (extras)
            └────────────────┴───────────┴────────────┴──────────────┘
                                         │ trace IDs
                         ┌───────────────▼─────────────────────┐
   LAYER 3 (eval bridge) │  exo-eval/sync.py                    │
                         │  dataset pull → Evaluator → push     │
                         │  ScorerResult→score, Trajectory→trace│
                         └─────────────────────────────────────┘
```

**Why OTel-first:** Langfuse (`/api/public/otel`) and LangSmith both ingest OpenTelemetry natively, as do
Phoenix/Arize, Braintrust, Opik, W&B Weave, MLflow, Jaeger, Tempo. One span tree → every backend. Native
SDKs are used only for what OTLP doesn't carry: **scores, datasets, prompt management, `evaluate()`**.

---

## 4. Module layout

```
exo-observability/src/exo/observability/
  tracing.py              # existing span/aspan/traced — KEEP
  tracer.py          NEW  # Tracer: owns span tree, registers on hook_manager, nests sub-agents
  cost.py            NEW  # CostTracker + model price table; stamps COST_* + RunResult cost
  backends/          NEW
    __init__.py           #   resolve_backends(spec) -> list[TracingBackend]; env auto-detect (§6)
    base.py               #   TracingBackend protocol + generic OTLPBackend
    langfuse.py           #   OTLP exporter (auth header) + native SDK bridge: scores, prompts
    langsmith.py          #   OTLP exporter + native bridge: datasets, evaluate()
    phoenix.py            #   OTLP target only (3rd impl → validates the seam)
    braintrust.py         #   OTLP + scores/datasets bridge
  config.py               # EXTEND TraceBackend enum; make TracingConfig drive a real exporter

exo-eval/src/exo/eval/
  sync.py            NEW  # PlatformSync: dataset pull/push, ScorerResult->score, Trajectory exporters
```

`exo-ai` extras: `[tracing]` (OTel + OTLP exporter base), `[langfuse]`, `[langsmith]`, `[phoenix]`,
`[braintrust]`. Each native dep isolated like provider SDKs in exo-models.

---

## 5. The plugin seam (`backends/base.py`)

```python
@runtime_checkable
class TracingBackend(Protocol):
    name: str
    def build_span_processor(self) -> "SpanProcessor | None": ...   # OTLP exporter wiring
    def is_available(self) -> bool: ...                              # extra installed + keys present
    # optional native capabilities (None if unsupported):
    def score_client(self) -> "ScoreSink | None": ...               # scores → traces
    def dataset_client(self) -> "DatasetClient | None": ...         # dataset pull/push
```

`resolve_backends(spec)` accepts `"langfuse"`, `["langfuse","phoenix"]`, a `TracingConfig`, or `None`
(env auto-detect). Returns configured backends; unavailable ones degrade to no-op + a single `_log.debug`.

---

## 6. Activation & defaults (on-by-default when detected)

Resolution order (first match wins):
1. Explicit `Agent(tracing=...)` / `run(..., tracing=...)` argument.
2. `EXO_TRACING` env (`langfuse` | `langsmith` | `otlp` | `langfuse,phoenix` | `off`).
3. **Auto-detect:** if a known backend's keys are present in env
   (`LANGFUSE_PUBLIC_KEY`+`LANGFUSE_SECRET_KEY`, `LANGSMITH_API_KEY`, `PHOENIX_*`, `BRAINTRUST_API_KEY`)
   **and** the matching extra is importable → enable that backend automatically.
4. Otherwise → `NullSpan`, zero overhead.

**Guardrails on auto-enable** (so it's not surprising/leaky):
- `EXO_TRACING=off` (or `tracing=False`) hard-disables, beating auto-detect.
- On first auto-enable, emit one `_log.info`: `"exo: tracing auto-enabled via <backend> (set EXO_TRACING=off to disable)"`.
- Auto-detect only fires when the extra is actually installed — presence of keys alone never pulls a missing dep.
- Document prominently in the observability README + meta-package README.

---

## 7. Phases & execution

### Phase 1 — Spine (GATE; vendor-agnostic; single focused agent, not parallel)
- `tracer.py`: `Tracer` registers one async callback per `HookPoint`. Span tree:
  - `START` → root `agent.run` span (attrs: `AGENT_NAME`, session/task/user ids).
  - `PRE_LLM_CALL`/`POST_LLM_CALL` → `gen_ai.chat` span; stamp `response.usage` → `GEN_AI_USAGE_*`, model, finish_reason.
  - `PRE_TOOL_CALL`/`POST_TOOL_CALL` → `tool.execute {tool_name}` span; args, result/error, duration.
  - `ERROR` → `record_exception` + error status on current span.
  - `CONTEXT_WINDOW` → `context.compress` span (before/after counts, fill ratio).
  - Sub-agents / Swarm nest via OTel context propagation (contextvars) — no extra wiring.
- `config.py`: extend `TraceBackend` (`LANGFUSE`/`LANGSMITH`/`PHOENIX`/`BRAINTRUST`); build a real OTLP
  `BatchSpanProcessor` + exporter from `endpoint`/`sample_rate`/`backend`.
- `backends/base.py` + `backends/__init__.py`: protocol + `resolve_backends` + env auto-detect (§6).
- Entry points: `Agent(tracing=...)`, `run(..., tracing=...)`, `EXO_TRACING`.
- **Tests:** in-memory OTel `SpanExporter` + `MockProvider`; assert span-tree shape, parent/child nesting,
  usage attrs, error status. No network. (Skip gracefully if `opentelemetry-sdk` absent.)

### Phase 2 — Cost tracking (parallelizable after P1)
- `cost.py`: `CostTracker` + model→price table (`provider:model` → input/output $/1M tokens).
  Compute per-call + cumulative cost from `Usage`; stamp `COST_*` semconv onto spans; surface total on `RunResult`.
- Vendor-neutral — every backend inherits cost because it rides the spans.
- **Tests:** known usage → expected cost; unknown model → no crash, `cost=None`.

### Phase 3 — Native adapters (PARALLEL — one sonnet sub-agent per file)
- `backends/langfuse.py` — OTLP endpoint + Basic-auth header from keys; native SDK bridge for **scores**
  (attach `ScorerResult` to trace) + **prompt management** fetch.
- `backends/langsmith.py` — OTLP wiring; native bridge for **datasets** + `evaluate()` interop.
- `backends/phoenix.py` — OTLP target only (proves the seam = 3rd implementation).
- `backends/braintrust.py` — OTLP + scores/datasets bridge.
- **Tests per adapter:** `is_available()` false when extra/keys missing → no-op; span-processor built when present
  (monkeypatched exporter, no real network).

### Phase 4 — Eval bridge (parallel with P3)
- `exo-eval/sync.py`: `PlatformSync` — pull dataset from {Langfuse, LangSmith, Braintrust} → run `Evaluator`
  → push `ScorerResult`s back as scores keyed to emitted trace IDs. `TrajectoryDataset → {langfuse, langsmith}`
  trace-format exporters.
- **Tests:** round-trip with a fake client; `ScorerResult`→score payload shape.

**Sequencing:** Phase 1 is the gate (do alone, get it green). Then fan out Phases 2/3/4 as concurrent
sonnet sub-agents — one per adapter/file per the CLAUDE.md parallel-subagent pattern. After each fan-out,
run the affected package's tests; finish with a full `uv run pytest` to catch cross-package breakage.

---

## 8. Conventions to honor (from CLAUDE.md)
- exo-core `_internal` logging: `from exo.observability.logging import get_logger  # pyright: ignore[reportMissingImports]`.
  Other packages: `import logging; logger = logging.getLogger(__name__)`.
- Ruff line-length 100, `datetime.UTC`. Pydantic v2. `asyncio_mode=auto`. Unique test file names across packages.
- Model strings `"provider:model"`. Never make real API calls in tests — `MockProvider` + monkeypatched exporters.
- Native SDK deps isolated behind optional extras; core stays dep-light.

## 9. Risks / open questions
- **OTLP span attribute size limits** for big prompts/results — may need truncation knob (`max_attr_chars`).
- **Sub-agent nesting across `asyncio.TaskGroup`** (parallel tools / parallel sub-agents) — verify OTel
  context propagates into task-group children; may need explicit `context.attach`.
- **Langfuse vs LangSmith trace semantics** differ (generations vs runs) — generic GenAI semconv should map,
  but confirm each renders the tree correctly with a live smoke test before launch.
- **Double-export** when multiple backends share the OTLP pipeline — ensure independent span processors.

---

## 10. Definition of done
- `EXO_TRACING=langfuse` (or keys auto-detected) → a bare `Agent("...").run()` produces a correct nested
  trace (run → llm → tools → sub-agents) in Langfuse, with token usage + cost.
- Same for LangSmith, Phoenix, Braintrust via config swap only.
- `exo-eval` can pull a dataset, evaluate, and push scores linked to traces.
- Full `uv run pytest` green; no real network in tests; zero overhead when tracing is off.
