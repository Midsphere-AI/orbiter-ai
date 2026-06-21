# Exo DX Simplicity Audit

**North-star goal:** a programming rookie should read a signature or an example and *just get it* — minimal concepts, no jargon, autocomplete that works, one obvious way to do each thing, and errors that teach.

This document has two parts:
- **Part 1 — Constructor param bloat → namespaces** (the original audit).
- **Part 2 — Broader DX friction** (naming/jargon, stringly-typed APIs, imports & autocomplete, choice overload, hello-world friction & error quality).

**Research only — no code changed.** All findings carry `file:line` citations.

---

# Part 1 — Constructor Param Bloat → Namespaces

**Goal:** Replace flat, dozens-of-kwargs constructors with grouped **namespaces** (e.g. `Agent.context`, `Agent.memory`, `Agent.planner`, `Agent.tools`, `Agent.subagents`) so developers configure capabilities through one obvious, discoverable handle per concern instead of memorizing a wall of top-level params.

This part surveys every package, ranks the offenders, and proposes a consistent namespace pattern to apply repo-wide.

---

## 1. The headline problem: `Agent.__init__` (41 params)

`packages/exo-core/src/exo/agent.py:712` — **41 keyword-only params.** This is the front door of the framework and the single most important thing to fix. Today a power user faces a flat list mixing identity, model, context, memory, planning, tools, subagents, PTC, guardrails, and dead no-ops.

### Current params → proposed namespaces

| Namespace | Current flat params | Count |
|---|---|---|
| *(top-level, keep flat)* | `name`, `model`, `instructions`, `tools`, `handoffs` | 5 |
| `Agent.model` | `temperature`, `max_tokens`, `output_type` (+ `model` string) | 3 |
| `Agent.context` | `context`, `context_mode`, `context_limit`, `overflow`, `cache`, `budget_awareness` | 6 |
| `Agent.memory` | `store`, `memory` | 2 |
| `Agent.planner` | `planning_enabled`, `planning_model`, `planning_instructions` | 3 |
| `Agent.tools` | `bare_tools`, `injected_tool_args`, `tool_gate`, `skills`, `tool_resolver`, `emit_mcp_progress` | 6 |
| `Agent.subagents` | `allow_self_spawn`, `subagents`, `max_spawn_depth`, `max_spawn_children` | 4 |
| `Agent.ptc` | `ptc`, `ptc_timeout`, `ptc_max_output_bytes`, `ptc_max_tool_calls`, `ptc_extra_args` | 5 |
| `Agent.guardrails` | `rails`, `hitl_tools`, `human_input_handler` | 3 |
| `Agent.hooks` | `hooks` | 1 |
| `runtime` | `max_steps` | 1 |
| **DELETE (dead no-ops)** | `allow_parallel_subagents`, `max_parallel_subagents` | 2 |

**Notable redundancies the refactor should also kill:**
- `allow_self_spawn` **and** `subagents` control the same thing — two params, one behavior.
- `context_mode` vs (`context_limit`/`overflow`/`cache`) are mutually exclusive with a runtime guard — collapses into one `context=` handle.
- `allow_parallel_subagents`/`max_parallel_subagents` are explicitly no-ops (agent.py:773–776) — pure deletion.
- `model` string is `"provider:model"` but `ModelConfig` splits `provider`+`model_name` — a format inconsistency to reconcile.

### Recommended pattern: hybrid (grouped configs in, namespace attrs out)

Accept either a bare shorthand **or** a typed config object per namespace, and expose the resolved config as an attribute namespace post-construction:

```python
agent = Agent(
    name="researcher",
    model="openai:gpt-4o",
    instructions="...",
    tools=[search, browse],                                   # flat shorthand stays
    context=ContextConfig(limit=50, overflow="summarize"),    # or context="copilot"
    memory=MemoryConfig(store="sqlite"),                      # or memory=False
    planner=PlannerConfig(enabled=True, model="openai:gpt-4o-mini"),
    subagents=SubagentsConfig(max_depth=2, max_children=3),   # or subagents=False
    ptc=PTCConfig(enabled=True, timeout=120),                # or ptc=True
    guardrails=GuardrailsConfig(hitl_tools=["deploy"], rails=[my_rail]),
)

# inspectable namespaces post-construction
agent.context.limit        # 50
agent.planner.enabled      # True
agent.subagents.max_depth  # 2
```

**Backward-compat is the constraint.** These flat kwargs are everywhere in examples/tests/benchmarks and MUST keep working: `name`, `model`, `instructions`, `tools` (as `list[Tool]`), `max_steps`, `temperature`, `memory=None` (disable idiom), `ptc=True` (bool form), `planning_enabled`. Bool/scalar shorthands resolve into the config objects internally. `AgentConfig` (config.py:171) and the internal `spawn_self` construction (agent.py:1316–1335) mirror these params and need parallel updates.

---

## 2. Severity ranking — all packages

| Rank | Surface | File:Line | Params | Verdict |
|---|---|---|---|---|
| 🔴 1 | `Agent.__init__` | exo-core `agent.py:712` | **41** | **CRITICAL** — the whole point |
| 🔴 2 | `SearchConfig` | exo-search `config.py:14` | **~28** | **CRITICAL** — flat dataclass, 6 implicit sub-systems |
| 🔴 3 | `MCPServerConfig` | exo-mcp `client.py:67` | **13** | **HIGH** — transport-conditional params conflated (correctness hazard) |
| 🔴 4 | `TrajectoryItem` | exo-eval `trajectory.py:27` | **13** | **HIGH** — SAR structure documented but flat |
| 🟠 5 | `ServingConfig` | exo-a2a `types.py:112` | **10** | **MEDIUM** — bind vs protocol-advertise mixed |
| 🟠 6 | `LoopState` | exo-eval `ralph/config.py:111` | 10 | runtime-state, lower priority |
| 🟠 7 | `ServerEntry` | exo-mcp-cli `config.py:40` | **9** | **MEDIUM** — near-dup of MCPServerConfig, same flaw |
| 🟠 8 | `ObservabilityConfig` | exo-observability `config.py:22` | **9** | **MEDIUM** — `trace_` prefixes ARE latent namespaces |
| 🟠 9 | `Swarm.__init__` | exo-core `swarm.py:60` | 9 | **MEDIUM** — context quartet leaked from Agent |
| 🟡 10 | `E2BSandbox` / `KubernetesSandbox` | exo-sandbox `e2b.py:55` | 9 / 8 | LOW — clean base/backend split already |
| 🟡 11 | `ModelConfig` | exo-core `config.py:144` | 7 + hidden extras | MEDIUM — `extra="allow"` hides Google creds |
| 🟡 12 | `Worker.__init__` | exo-distributed `worker.py:114` | 7 | MEDIUM — Redis config should be one object |
| 🟡 13 | `HTTPEmbeddings` | exo-models `embeddings.py:396` | 7 | MEDIUM — `*_field` trio confusing |
| 🟡 14 | `ContextConfig` | exo-context `config.py:29` | 7 | acceptable; mirrors Agent's context params |
| 🟢 | exo-retrieval, exo-guardrail, exo-cli, exo-skills | — | ≤7, mostly DI | **FINE as-is** |

---

## 3. Per-package detail

### 🔴 exo-search — `SearchConfig` (worst non-Agent offender)
`config.py:14` — **~28 flat dataclass fields** spanning six implicit subsystems with zero grouping. The prefixes and comment blocks (`# Performance tuning`, `# Deep research settings`, `# Verification settings`, `# Writing settings`, `# Revision loop settings`) literally document the namespaces the author wanted but couldn't express.

Proposed namespaces:
- `models` → `model`, `fast_model`, `embedding_model`, `context_window_tokens`
- `sources` / connectors → `searxng_url`, `searxng_timeout`, `jina_reader_url`, `jina_api_key`, `serper_api_key`, `sources`, `max_results`
- `research` → `research_mode`, `max_iterations`, `max_deep_research_steps`, `deep_research_enrich_per_step`, `max_content_chars`
- `writer` → `system_instructions`, `max_writer_words`, `max_writer_sources`, `claim_first_writing`, `use_reasoning_preamble`
- `verification` → `llm_verification`, `llm_verify_source_chars`
- `revision` → `max_revision_rounds`, `revision_threshold`

```python
SearchConfig(
    models=SearchModels(model="openai:gpt-4o", fast_model="openai:gpt-4o-mini"),
    sources=SourcesConfig(searxng_url="...", serper_api_key="..."),
    research=ResearchConfig(mode="quality", max_iterations=25),
    writer=WriterConfig(max_words=2000, claim_first=True),
)
```

### 🔴 exo-mcp — `MCPServerConfig` (13) + exo-mcp-cli `ServerEntry` (9)
Both interleave **stdio-only** params (`command`/`args`/`env`/`cwd`) with **http-only** params (`url`/`headers`) at the same level — wrong-transport params silently no-op. Namespacing into `stdio=StdioConfig(...)` / `http=HttpConfig(...)` / `timeouts=TimeoutConfig(...)` is a *correctness* fix, not just cosmetics. The two classes are near-duplicates and should share the same `StdioConfig`/`HttpConfig` types (lift them to exo-mcp, reuse in the CLI).

### 🔴 exo-eval — `TrajectoryItem` (13)
The docstring already names the State-Action-Reward structure; the dataclass is flat. Group into `state=TrajectoryState(...)`, `action=TrajectoryAction(...)`, `reward=TrajectoryReward(...)` + identity fields. Self-documenting, shrinks `from_dict`. (`LoopState`/`RalphResult`/`StopConditionConfig` are lower-priority runtime/config records.)

### 🟠 exo-a2a — `ServingConfig` (10)
Primary user-facing publish config. Split `bind=BindConfig(host/port/endpoint)` from `advertise=AdvertiseConfig(streaming/version/transports/modes)`. Note: `AgentCard` (9 fields) should stay flat — it's a protocol wire-format type whose flat JSON is spec-mandated.

### 🟠 exo-observability — `ObservabilityConfig` (9)
The `log_*`, `trace_*`, `metrics_*` prefixes are latent namespaces. Lift to `logging=LoggingConfig(...)`, `tracing=TracingConfig(...)`, `metrics_enabled` + service identity. Cleanest mechanical win in the repo.

### 🟠 exo-core — `Swarm.__init__` (9)
The `context_mode`/`context_limit`/`overflow`/`cache` quartet (with its mutual-exclusion guard) is leaked Agent API. Collapse to a single `context=` param. Also worth a shared `StreamConfig(detailed, max_steps, event_types)` across `run.stream` and `Swarm.stream` where the same trio is duplicated. `LoopNode` (6, three mutually-exclusive mode params) would benefit from factory classmethods (`LoopNode.count/.items/.while_`).

### 🟡 exo-models / exo-context / exo-memory
- `ModelConfig` (config.py:144): collapse `provider`+`model_name` → one `model: str` (fix format inconsistency); surface the `extra="allow"` Google creds into an explicit `google=GoogleModelConfig(...)`.
- `HTTPEmbeddings` (embeddings.py:396): group the confusing `input_field`/`output_field`/`vector_field` into `schema=HTTPResponseSchema(...)`.
- `ContextConfig`/`SummaryConfig`: acceptable; main issue is the duplicated `keep_recent` default (5 vs 4) — converge on one source of truth. These overlap directly with `Agent.context.*` and should be the same type the Agent namespace wraps.

### 🟡 exo-distributed — `Worker.__init__` (7)
`redis_url`/`queue_name`/`heartbeat_ttl` recur across `Worker`, `TaskBroker`, `TaskStore`, `EventPublisher`. Extract one shared `RedisConfig` and inject it everywhere.

### 🟢 Already fine (no action)
- **exo-retrieval**: retrievers take 2 injected deps + 1–2 tuning kwargs (`Retriever`, `HybridRetriever`, `AgenticRetriever`, backends). Dependency injection already *is* the namespace pattern here.
- **exo-guardrail**: `BaseGuardrail` (2), `LLMGuardrailBackend` (5, all distinct). Clean.
- **exo-cli**, **exo-skills**, **exo-sandbox**: largest are 6–9 params with clean base/extension splits or internal-only construction. Defer.

---

## 4. Recommended namespace convention (apply repo-wide)

To keep the philosophy consistent everywhere:

1. **One frozen config dataclass per concern**, named `<Concern>Config` (or the concern noun, e.g. `PlannerConfig`, `RedisConfig`, `StdioConfig`).
2. **Top-level constructors accept `concern=<Config> | <shorthand>`** — bool/scalar/string shorthands resolve into the config internally so simple cases stay one-liners.
3. **Resolved configs are exposed as read attributes** (`agent.context`, `agent.planner`) for inspection/serialization.
4. **Share config types across packages** instead of duplicating (`StdioConfig`/`HttpConfig` between exo-mcp & exo-mcp-cli; `RedisConfig` across exo-distributed; `ContextConfig` between exo-context & Agent).
5. **Mutually-exclusive param sets** (transport, loop mode, context_mode-vs-shorthand) become either discriminated config objects or factory classmethods — never flat sibling kwargs guarded at runtime.
6. **Delete dead params** rather than carry them (Agent's parallel-subagent no-ops).

---

## 5. Suggested execution order

1. **`Agent` namespaces** (exo-core) — the flagship; everything else mirrors its conventions. Establishes the `ContextConfig`/`MemoryConfig`/`PlannerConfig`/`SubagentsConfig`/`PTCConfig`/`GuardrailsConfig`/`ToolsConfig` types.
2. **`Swarm`** — reuse Agent's `context=` handle; drop the quartet.
3. **`SearchConfig`** — biggest standalone win.
4. **`MCPServerConfig` + `ServerEntry`** — shared transport configs (correctness fix).
5. **`ObservabilityConfig`, `ServingConfig`, `TrajectoryItem`, `Worker`/`RedisConfig`** — mechanical, independent, parallelizable across sub-agents.
6. **`ModelConfig`/`HTTPEmbeddings`** — reconcile model-string format + surface hidden extras.

Each step keeps flat shorthands working for backward compat; the namespaces are additive grouping, not breaking renames.

---

# Part 2 — Broader DX Friction (rookie-readiness)

Param grouping (Part 1) is necessary but not sufficient. Five further dimensions decide whether a beginner succeeds. Each was audited across all packages.

## 2.0 The single recurring root cause: `Any`

The same defect surfaced in four independent audits. The hottest entry points are typed `Any`, which **destroys IDE autocomplete** — the one tool rookies rely on most:

| Surface | file:line | Today | Should be |
|---|---|---|---|
| `run(agent, …)` agent / provider | `runner.py:73,77` | `Any` | `Agent \| Swarm`, `ModelProvider \| None` |
| `run.sync` / `run.stream` | `runner.py:934-935` | bolted-on attrs, `# type: ignore` — **invisible to autocomplete** | typed `_RunCallable` Protocol exposing `.sync`/`.stream` |
| `Agent(store=…, memory=…, context=…, context_mode=…)` | `agent.py:737-740` | `Any` | concrete unions / `Literal[...]` |
| `Agent.memory` / `Agent.context` attrs | `agent.py:857,~891` | `Any` | `AgentMemory \| None`, `Context \| None` |
| `add_mcp_server(config=…)` | `agent.py:1671` | `Any` | `MCPServerConfig` |

**Verified firsthand:** `run.sync`/`run.stream` are attached as function attributes with `# type: ignore[attr-defined]` (runner.py:934-935) → a rookie typing `run.` gets no `.sync`/`.stream` suggestion. Fixing `Any` is the highest-leverage, lowest-risk DX win in the repo because it improves every dimension below at once.

## 2.1 Naming & jargon — opaque concept names

Exo's public vocabulary is full of metaphors and research acronyms a beginner cannot decode from the name alone. Top renames (keep old names as deprecated aliases):

| Rank | Current | Plain-English rename | Why |
|---|---|---|---|
| 1 | `ptc` / `PTCTool` / `PTCExecutor` | `batch_tools` / `ToolBatch…` | Acronym hides "LLM writes code to batch tool calls" |
| 2 | `Neuron` / `neuron_registry` / `*Neuron` | `PromptSection` / `*Section` | Sounds like ML internals; it's just a prompt block |
| 3 | `Ralph` / `RalphRunner` / `RalphNode` | `RefinementLoop` / `Refinement…` | Proper name, zero semantic signal |
| 4 | `Rail` / `RailManager` / `rails=` | `Guard` / `GuardManager` / `guards=` | Collides with the `Guardrail` package concept |
| 5 | `HITL` / `hitl_tools` | `approval` / `approval_tools` | Spell out the research acronym |
| 6 | `handoffs` / `handoff` | `transfers` / `transfer` (or `delegate`) | Unusual verb for agent-to-agent control transfer |
| 7 | `context_mode="pilot"/"copilot"/"navigator"` | `"large"/"balanced"/"compact"` | Aviation metaphors → `pilot` secretly means "big context" |
| 8 | `budget_awareness` | `context_pressure` / `context_alert` | "Budget" reads financial; it's context-fullness |
| 9 | `MemoryEvolutionStrategy` / `ACEStrategy` / `ReMeStrategy` | `MemoryCuration…` / descriptive names | Genetics metaphor + paper acronyms |
| 10 | `large_output=True` ↔ `ToolResultOffloader` | `offload_large_output` ↔ `LargeOutputHandler` | The decorator flag and the mechanism share no vocabulary |

**Naming inconsistencies (same concept, different names):**
- System prompt: `instructions` (Agent) vs `system_prompt` (eval/train) vs `system_instructions` (search) → standardize on **`instructions`**.
- Model string: `model` vs `model_name` vs `provider_name` → **`model`** for full string, `provider`+`model_id` for split parts (also reconcile with `ModelConfig` splitting them — Part 1).
- Memory shorthand: `store=` vs `memory=` → **`store=`**.
- Vector store ABC duplicated: `VectorStoreBase` (exo-models) vs `VectorStore` (exo-retrieval), with two separate `InMemoryVectorStore` impls → one canonical ABC.
- `GoogleProvider`/`GeminiProvider`/`VertexProvider` are 3 public names for ~1 class → document or collapse.
- `TaskLoopQueue` (exo-core steering) collides conceptually with `Task*` (exo-distributed) → rename to `SteeringQueue`.
- British spelling `DataSynthesiser`/`TemplateSynthesiser` (exo-train) breaks the otherwise-American codebase → `…Synthesizer`.

## 2.2 Stringly-typed APIs — silent failures, zero discoverability

Meaningful "choice" params are bare `str`, so typos fail silently or late, and valid values aren't visible in the signature. Worst offenders ranked by rookie pain:

| # | API | file:line | Failure on typo | Fix |
|---|---|---|---|---|
| 1 | `Agent(context_mode="Pilot")` | `agent.py:250-256` | **Totally silent** — `else:` falls through to copilot defaults, no error ever | `Literal[…]` + raise on unknown |
| 2 | `run.stream(event_types={"texts"})` | `runner.py:176` | **Silent zero output** — no match, no error (15 valid strings, undiscoverable) | export `EventType` enum / `EVENT_TYPES` set |
| 3 | `Swarm(mode="handOff")` | `swarm.py:65` | Deferred — `SwarmError` only at `.run()`, not construction | `Literal[…]` + validate in `__init__` |
| 4 | `Agent(store="redis")` | `agent.py:737` | **Silent wrong behavior** — unknown string treated as a SQLite *file path* (`store="redis"` writes a file named `redis`) | typed union + validate |
| 5 | `search(mode="Quality")` | `search/__init__.py:31` | Silent degradation — falls through to quality branch; `ResearchMode` enum exists but the public fn ignores it | accept `ResearchMode` |
| 6 | `BaseGuardrail(events=["pre_llm_call"])` | `guardrail/base.py:45` | Deferred to `.attach()`; accepts raw strings though `HookPoint` enum exists | accept `list[HookPoint]` |
| 7 | `SearchConfig(sources=["academics"])` | `search/config.py:25` | Silent miss — `"x" in sources` membership test, plural typo → no results | `SearchSource` enum |
| 8 | `model="openai:fake-xyz"` | `models/provider.py:99` | Provider validated immediately (good), but model name only fails at first API call with cryptic SDK error | soft-warn against `MODEL_CONTEXT_WINDOWS` |
| 9 | `model="claude-3-5-sonnet"` (no `provider:`) | `config.py:10-25` | **Silently assumes `openai:`** → confusing auth error | require colon or warn |

**Already good** (use as the template): `overflow=` → `OverflowStrategy(StrEnum)` raises immediately listing valid values; `get_provider()` → immediate `ModelError` with the available-provider list; `StatusEvent.status` is a `Literal`.

## 2.3 Imports & entry points — the front door leaks

`from exo import …` exposes 39 names but **omits the types a rookie hits in the first 5 minutes**. `exo/types.py` has 27 public symbols; the top-level re-exports only 9.

Must-have-but-buried (should be `from exo import …`):
- `RunResult`, `Usage` — the return value of `run()` and its `.output`/`.usage`.
- The entire streaming event set (`StreamEvent`, `TextEvent`, `ToolCallEvent`, `StepEvent`, `ToolResultEvent`, `ErrorEvent`, `StatusEvent`, `UsageEvent`, …) — needed the moment anyone calls `run.stream`.
- `AgentError` (in `exo.agent`), `HookPoint` & `HookManager` (in `exo.hooks`) — 8+ test files already do `from exo.agent import Agent, AgentError` because it isn't re-exported.
- Message types: `Message`, `UserMessage`, `AssistantMessage`, `SystemMessage`, `ToolResult`, `ToolCall`.
- Memory backends not re-exported even at package level: `from exo.memory.backends.sqlite import SQLiteMemoryStore` (should be `from exo.memory import SQLiteMemoryStore`).

**Top fixes:** (1) bulk re-export `exo.types` at top level — zero-cost, no optional deps; (2) give `run` a typed Protocol so `.sync`/`.stream` autocomplete (~15 lines); (3) add `AgentError`/`HookPoint` to `exo.__all__`; (4) re-export the 3 memory backends at `exo.memory`.

## 2.4 Choice overload — too many ways to do one thing

The user's core complaint, quantified. Ranked "decision paralysis":

| # | Question a rookie asks | Competing answers | Keep ONE |
|---|---|---|---|
| 1 | **How do I run my agent?** | `await run(agent,…)` → `RunResult.output` vs `agent.run(…, provider=)` → `AgentOutput.text` (**different return types & field names!**) vs `run.sync` vs `run.stream` | `run()` / `run.sync` / `run.stream`; make `agent.run()` private (`_run`) |
| 2 | **How do I set memory?** | `store=` vs `memory=` vs `store=None` vs `store=False` vs `memory=None` (all 5 reachable) | `store=`; deprecate `memory=`; disable = `store=False` only |
| 3 | **How do I set context?** | `context_limit`/`overflow`/`cache` vs `context_mode` (opaque) vs `context=` object vs nothing | inline trio for simple; `context=ContextConfig` for power; drop `context_mode` |
| 4 | **How do I run agents in parallel?** | `ParallelGroup` vs `run_parallel()` vs `spawn_self()` vs `Swarm(mode="team")` | document use-case→API map (they genuinely differ); don't add more |
| 5 | **Self-spawn on/off?** | `subagents=False` vs `allow_self_spawn=False` (same flag, 1 line apart) | `subagents=`; deprecate `allow_self_spawn` |
| 6 | **How do I make a tool?** | `@tool` vs `FunctionTool(fn)` (identical) vs `Tool` ABC | `@tool` for fns, `Tool` ABC for custom; drop `FunctionTool` from public exports |
| 7 | **Dead params** | `allow_parallel_subagents`, `max_parallel_subagents` — **no-ops** still validated, stored, serialized & documented as real in `AgentConfig` | delete |
| 8 | **`AgentConfig` vs `Agent()`** | a frozen Pydantic model that has **diverged** from the real constructor (missing `store`/`subagents`/`context_limit`/`ptc`/…) | delete or regenerate from the constructor |

Also: legacy alias properties leaking implementation (`ContextConfig._enable_snapshots`↔`cache`, `_token_budget_trigger`↔`token_pressure`).

## 2.5 Hello-world friction & error quality

**The good news:** a true 3-line program already works —
```python
from exo import Agent, run
agent = Agent(name="bot", model="openai:gpt-4o-mini")
print(run.sync(agent, "Hello!").output)
```
**The bad news:** it's surfaced nowhere. `README.md:43-63` leads with the `async def main(): … asyncio.run(main())` version — forcing a beginner to learn asyncio before their first query, when `run.sync` hides all of it.

Friction blockers:
- `model=` default is **`openai:gpt-4o`** (`agent.py:716`), ~30× pricier than `gpt-4o-mini`, which every example actually uses. Flip the default.
- `name=` is the lone required arg; `Agent()` raises. Consider defaulting it.
- A bare `Agent(name="bot")` silently pre-registers **9 hidden tools** (todo/knowledge/file/spawn) → extra tokens every call, no visible hint.

**Error messages that don't teach** (rate: does it tell the user what to do?):

| Scenario | file:line | Current | Teaches? |
|---|---|---|---|
| Bad provider name | `runner.py:898` | `_resolve_provider` **catches the helpful `ModelError` ("Provider 'x' not registered. Available: […]") and returns None** → caller raises generic `"requires a provider"` | ❌ (verified) |
| `tools="get_weather"` (string not list) | `agent.py:1023` | `AttributeError: 'str' object has no attribute 'name'` | ❌ |
| `model=42` (wrong type) | `config.py:22` | `TypeError: argument of type 'int' is not iterable` | ❌ |
| Forgot `await run(...)` | runner | `coroutine ... was never awaited` at exit; no hint that `run.sync` exists | ❌ |
| Wrong flow separator `a -> b` | `swarm.py:98` | `Flow references unknown agent 'agent_a -> agent_b'` | ❌ |
| Missing `ANTHROPIC_API_KEY` | `anthropic.py:375` | `Set ANTHROPIC_API_KEY or pass api_key=…` | ✅ (template) |
| Missing `OPENAI_API_KEY` | `openai.py` | SDK message is fine but **wrapped 3× by retry spam** under a `CallRunnerError` — 401 is non-retryable | ⚠️ |
| `max_steps=0` | `agent.py:758` | `max_steps must be >= 1, got 0` | ✅ |

Fixes (high ROI, mostly localized): rewrite README quickstart to the 3-liner; re-raise the swallowed `ModelError`; validate `tools`/`model` types with teaching messages; don't retry 401s; warn on `@tool` with no docstring; list available agents in Swarm flow errors.

---

# Consolidated priority — the rookie-readiness backlog

Ordered by (impact on a beginner) ÷ (risk/effort). Backward-compat via aliases throughout.

**Tier 0 — do first, broad payoff, low risk**
1. **Kill `Any` on the hot path** — type `run`/`run.sync`/`run.stream` (Protocol), `Agent(store/memory/context/context_mode)`, `agent`/`provider`. Fixes autocomplete everywhere. (2.0)
2. **Re-export the first-5-minutes API at top level** — `RunResult`, `Usage`, streaming events, `AgentError`, `HookPoint`, message types; memory backends at `exo.memory`. (2.3)
3. **README + docstrings lead with `run.sync` 3-liner**; flip default model to `gpt-4o-mini`. (2.5)
4. **Teaching errors** — re-raise swallowed provider error; validate `tools`/`model` types; no-retry 401s. (2.5)

**Tier 1 — the "one obvious way" cleanups**
5. Make `agent.run()` private; `run()` is the only invocation API (unifies `RunResult` vs `AgentOutput`). (2.4 #1)
6. Collapse memory to `store=`; deprecate `memory=`. Collapse `subagents=`/`allow_self_spawn`. **Delete** the no-op parallel-subagent params + the diverged `AgentConfig`. Drop `FunctionTool` from public exports. (2.4)
7. `Literal`/enum the silent stringly-typed params (`context_mode`, `store`, `mode`, `event_types`, search `mode`/`sources`); export an `EventType` enum. (2.2)

**Tier 2 — vocabulary (aliased renames; do alongside Part 1 namespaces)**
8. Rename the opaque concepts: `ptc`→`batch_tools`, `Neuron`→`PromptSection`, `Ralph`→`Refinement…`, `Rail`→`Guard`, `hitl_tools`→`approval_tools`, `context_mode` values→size words. Standardize `instructions`/`model`/vector-store vocabulary. (2.1)

**Tier 3 — Part 1 namespace refactor** (`Agent` first → establishes config types reused everywhere), then `SearchConfig`, `MCPServerConfig`+`ServerEntry`, `ObservabilityConfig`, `ServingConfig`, `TrajectoryItem`, `Worker`/`RedisConfig`.

Each tier is independently shippable and keeps old spellings working as deprecated aliases.
