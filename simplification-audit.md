# Exo Simplification Audit — Feature Inventory & Keep/Cut Decisions

> Phase 1 deliverable: per-package feature inventory (how each feature is offered, how
> well features compose) plus a cross-package keep/cut/merge plan. Produced by 13
> parallel Sonnet sub-agents, one per package group, then synthesized.

## TL;DR — the 7 structural problems

1. **Embeddings + vector stores are implemented 3 times** — `exo-retrieval`, `exo-memory`, and `exo-search` each ship their own `OpenAIEmbeddings` / `VertexEmbeddings` and their own `_cosine_similarity`. ChromaDB is wired twice (`exo-retrieval.ChromaVectorStore`, `exo-memory.ChromaVectorMemoryStore`).
2. **`exo-web` re-implements ~6 sibling packages internally** instead of depending on them: Crews (vs `exo.Swarm`), memory service (vs `exo-memory`), evaluators+safety (vs `exo-eval`+`exo-guardrail`), knowledge/RAG (vs `exo-retrieval`), cost/observability (vs `exo-observability`), MCP health (vs `exo-mcp`). It depends only on `exo-core`+`exo-models`.
3. **Two CLIs, two servers.** `exo` vs `exo-mcp` (copy-pasted tool-call UX); `exo-server` vs `exo-web` (two FastAPI agent servers, distinction undocumented).
4. **`exo-core` has too many ways to do one thing.** `run()` vs a ~700-line duplicate `run.stream()`; 7 swarm node types with duck-typed dispatch; 4-way context config; 3 parallel-subagent mechanisms (one a no-op).
5. **`exo-harness` overlaps `exo-core` orchestration** (`ParallelGroup`/`Swarm`/`task_controller`) and has **zero non-test consumers**. Its `TimeoutMiddleware` is non-functional.
6. **`exo-train` is ~60% aspirational** — `GaussianMutationStrategy` returns simulated loss; VeRL path needs GPU infra and has deprecation bugs.
7. **Aspirational/half-wired features ship as if real** — `exo-web` plugins marketplace, neuron pipelines, plans/PEV, context-state inspector; `exo-observability` cost/SLO/prompt-logger/span-consumer (zero callers); various dead `@tool`s in `exo-search`.

---

## Cross-package consolidation plan (highest leverage)

| # | Consolidation | Action | Removes |
|---|---|---|---|
| C1 | **One embeddings/vector layer** | Make `exo-retrieval` the canonical home for `Embeddings` ABC + OpenAI/Vertex/HTTP impls + `_cosine_similarity` + Chroma/pgvector. `exo-memory` and `exo-search` import from it. | ~3 embedding impls, 2 extra `_cosine_similarity`, 1 duplicate Chroma store |
| C2 | **`exo-web` delegates to packages** | Replace web-internal reimplementations with the real packages (Swarm, exo-memory, exo-eval, exo-guardrail, exo-retrieval, exo-observability, exo-mcp). | ~2,600+ LOC in exo-web |
| C3 | **One CLI tool-call util** | Extract `_build_arguments`/arg-coercion to `exo-mcp` (or exo-core); both CLIs import it. Move `Vault` from `exo-mcp-cli` into `exo-mcp`. | copy-pasted arg parser, siloed vault |
| C4 | **One server story** | Demote `exo-server` to a documented "minimal embed helper"; `exo-web` is the platform. De-dup its SSE/WS/`_get_agent` copies; fix/remove `/inject`. | duplicate server surface |
| C5 | **One trajectory type** | Move `TrajectoryItem`/`TrajectoryDataset` to `exo-eval`; `exo-train` imports it. | trajectory type fork |
| C6 | **One token-estimate util** | Single `estimate_tokens` (in observability or core); `exo-memory` stops shipping its private copy. | duplicate token heuristic |
| C7 | **Distributed-only semconv/alerts** | Move `DIST_*`/`STREAM_*` semconv, `AlertManager`, `BaggagePropagator` from `exo-observability` into `exo-distributed`. | dead constants in shared pkg |

---

## Per-package decisions

### exo-core — SIMPLIFY (keep as foundation)
- **CUT:** `SerialGroup` (== `Swarm(mode="workflow")`), `Swarm(mode="team")` (thin wrapper over handoff), `RalphNode` (depends on undefined `RalphRunner`, ignores all params), `allow_parallel_subagents` (no-op flag), bulk of `task_controller` (`TaskManager`/`TaskScheduler`/`TaskEventBus`/`IntentRecognizer` — only `TaskLoopQueue` is wired).
- **SIMPLIFY:** de-duplicate `run.stream()` against `call_runner` (~700 LOC duplication, biggest maintenance risk); collapse 4-way context config to one API; default `memory=None` (stop auto-creating stores).
- **MOVE:** `Rail` ABC → `exo-guardrail`.
- **KEEP:** Agent/Tool/`@tool`, `run`/`run.sync`/`run.stream`, hooks, `Swarm(workflow|handoff)`, `ParallelGroup`, `BranchNode`, `LoopNode`, `SwarmNode`, PTC, spawn-self, HITL, skills, checkpoints, loader, token counter, events.

### exo-models — KEEP (trim duplication)
- **MERGE:** `VertexProvider` into `GeminiProvider` (identical bodies; differ only in `__init__` auth) → one `GoogleProvider`; fold `_media.py` into `_google_common.py`.
- **SIMPLIFY:** add `use_cache` opt-out (Anthropic caching is always-on); fix `-> Any` return types to `-> ModelResponse`; replace `"dummy"` api-key fallback with construction-time error; standardize `GOOGLE_API_KEY` vs `GEMINI_API_KEY`; make silent media-drops raise.
- **CUT:** vestigial private re-exports in provider `__all__`; either expand or drop the 9-entry `context_windows.py`.

### exo-web — AGGRESSIVE CUT (heaviest package)
- **CUT (dead/aspirational, ~2,600 LOC):** Crews (duplicates Swarm, rolls own httpx provider dispatch), Plugins marketplace (hardcoded catalog, no real install), Neuron pipelines (returns placeholder strings, not wired to runtime), Plans/PEV (only builds prompt strings server-side), Context-state inspector (fabricates fork/merge tree), Benchmarks (600 LOC wrapping eval), Annotations (difflib cache, no integration), embedded docs (ship externally).
- **MERGE → sibling packages:** memory service→`exo-memory`, evaluators+safety→`exo-eval`+`exo-guardrail`, knowledge→`exo-retrieval`, cost→`exo-observability`, MCP health→`exo-mcp`; unify Threads+Conversations (and playground store); pick one of config-versions/agent-templates.
- **FIX FIRST (security, from audit.md):** B-4 webhook `url_token` never validated; B-1 sandbox `.replace()` escaping.
- **KEEP:** auth/sessions/CSRF/team, projects, agent CRUD+runtime bridge, workflow engine+canvas, scheduler, deployments+widget, artifacts, prompt templates, retention, audit log, notifications, FTS search.

### exo-retrieval — KEEP (becomes canonical embed/vector layer per C1)
- **CUT:** `GraphRetriever`+`TripleExtractor` (1 LLM call/chunk, brittle, synthetic invalid chunks), `ChromaVectorStore` (dup of exo-memory's; buggy substring filter).
- **SIMPLIFY:** drop `TokenChunker` silent whitespace fallback; fix `AgenticRetriever` round-0 rewrite; auto-init `PgVectorStore` on first add; fix dimension-default mismatch (1536 vs 768).
- **KEEP:** Embeddings ABC + OpenAI/Vertex/HTTP, InMemory + pgvector stores, chunkers/parsers, Vector/Sparse/Hybrid retrievers, `LLMReranker`, tools. Add it to the `exo-ai` meta-package.

### exo-memory — SIMPLIFY (stop duplicating retrieval)
- **CUT:** `MemoryStatus` DRAFT/DISCARD lifecycle (always ACCEPTED — dead), `OpenAIEmbeddingProvider` (dup of `OpenAIEmbeddings`), `SentenceTransformerEmbeddingProvider` (unexported, no callers), `ReMeStrategy` (overlaps `MemoryOrchestrator`).
- **MERGE → exo-retrieval (C1):** `VectorMemoryStore`, `ChromaVectorMemoryStore`, `OpenAIEmbeddings`, `VertexEmbeddings`.
- **FIX:** `EncryptedMemoryStore` silently-broken keyword search + dropped `category`; `MemoryEventEmitter` dropped `category`.
- **KEEP:** MemoryItem/MemoryStore, ShortTerm/LongTerm, SQLite/Postgres backends, EncryptedStore (decorator), persistence+snapshots, SearchManager, `MemUpdateChecker`, evolution ABC, `ACEStrategy`, migrations.

### exo-context — SIMPLIFY (large dead surface)
- **CUT:** `SummarizeProcessor` (only sets a flag nobody reads), `RoundWindowProcessor` (dup windowing), `MessageOffloader` (injects unrecoverable markers — corrupts history), `make_config`/`AutomationMode` presets, `enable_retrieval` field (no wiring), module-level tool singletons, `Context.add_tokens`/`_token_usage` (parallel to `TokenTracker`), `DialogueCompressor.model` (dead param).
- **SIMPLIFY:** drop legacy config field-set (keep simplified API only); the speculative neurons (Todo/Knowledge/Workspace/Skill/Fact/Entity — nothing populates their state).
- **MERGE:** `KnowledgeStore` into `Workspace`; `Checkpoint`/`CheckpointStore` with exo-core's `WorkflowCheckpoint` (structurally identical); `SummaryConfig` with exo-memory.
- **KEEP:** `Context` fork/merge, `ContextState`, `ProcessorPipeline`, `ToolResultOffloader` (has recovery path), `ContextWindowHook`, `Workspace`, `TokenTracker`, `Neuron`/`PromptBuilder` core, `DynamicVariableRegistry`.

### exo-harness — DONE (dissolved into exo-core)
- Package deleted. `is_harness` duck-typing branches removed from `runner.py`.
- **CUT:** `Harness` ABC, `HarnessContext`, `HarnessNode`, `HarnessError`, `HarnessEvent`,
  `HarnessCheckpoint`, `CheckpointAdapter`, `SessionState`, `Middleware` ABC,
  `TimeoutMiddleware`, `CostTrackingMiddleware`, `/tmp` log-file side effects,
  `AssistantMessage` injection footgun, `_ForkedSessionState`.
- **SALVAGED → `exo.parallel`:** `run_parallel(tasks, ...)` and `stream_parallel(tasks, ...)`
  as free functions in `packages/exo-core/src/exo/parallel.py`. Types
  `SubAgentTask`, `SubAgentResult`, `SubAgentStatus`, `SubAgentError` live there.
  All exported from the top-level `exo` namespace.
- Custom orchestration = a plain async function calling `run()` / `run.stream()`.

### exo-skills — KEEP (justified split)
- Split exists only to make `watchfiles` optional — sound. (ABC/types already live in `exo-core/skills.py`.)
- **MERGE:** duplicated `_diff_snapshots` → shared `_utils.py`.
- **FIX:** `asyncio.get_event_loop()` → `get_running_loop()`; add GitHub auth token; fix `branch=None` clone path.
- **Doc:** "Skills = static markdown bundles; MCP = runtime servers" to stop users duplicating effort.

### exo-cli — FIX (a core command is a stub)
- **FIX:** `exo run` loads config, prints input, exits — **never invokes the agent**. Wire to `LocalExecutor.execute()` or remove.
- **CUT:** `PluginManager` (no callsites), `InteractiveConsole` (no `exo chat` command) unless wired.
- **MOVE:** `exo start worker`/`task`/`worker` → `exo-distributed` (owns the runtime).
- **MERGE (C3):** tool-call arg parser shared with exo-mcp-cli.
- **KEEP:** `LocalExecutor`, `BatchLoader`/`batch_execute` (add `exo batch`), agent loaders.

### exo-mcp-cli — KEEP (move shared bits down)
- **MOVE:** `Vault` → `exo-mcp` (framework MCP should reuse it); arg parser shared (C3).
- **FIX:** `resource read` multi-blob bug (loop overwrites same output file); re-vaulting silently overwrites; PBKDF2 480k iters makes every invocation slow.
- **KEEP:** server/tool/resource/prompt/auth commands, multi-transport, `${vault:}`/`${ENV}` substitution, standalone (no exo-core dep) design.

### exo-server — DEMOTE
- Superseded by `exo-web`. Document as "single-process embed helper."
- **CUT:** `/inject` (`agent.inject_message()` doesn't exist on public Agent — runtime `AttributeError`), in-memory sessions/workspace routes (false persistence expectations).
- **SIMPLIFY:** SSE+WS do the same thing — share `_iter_events`, drop duplicate `_sse_stream` and double `_get_agent`.

### exo-mcp — KEEP (trim)
- **CUT:** `MCPServerRegistry` (unused global), `call_tool_with_retry` (dead for agent path), `load_tools_from_client` (thin loop).
- **MERGE (C3):** transport construction + `substitute_env_vars` shared with exo-mcp-cli.
- **SIMPLIFY:** `@mcp_server` decorator (fragile `__init__` patching).
- **KEEP:** `MCPClient`/`MCPServerConnection`, `MCPToolWrapper` (reconnect/progress), filter, namespacing, config loading.

### exo-sandbox — SIMPLIFY (and unify with exo-web)
- **CUT:** `LocalSandbox` (no isolation == `tool.execute()`), `SandboxBuilder` (lazy `__getattr__` footgun).
- **FIX:** `KubernetesSandbox` broken pod-exec fallback (hardcoded `raise ImportError`), unused ClusterIP Service per start, broken stop/restart contract.
- **MERGE:** `ShellTool`(allowlist)+`TerminalTool`(blacklist) → one tool with a `mode`; `CodeTool` with exo-web's `services/sandbox.py` (two independent Python-exec impls — pick the more complete web one).
- **KEEP:** `Sandbox` ABC, `E2BSandbox`, `FilesystemTool`.

### exo-guardrail — SIMPLIFY
- **CUT:** `GuardrailResult.modified_data` (never populated), `BaseGuardrail(backend=None)` silent no-op, `confidence` field (never used in decisions) — or wire it into thresholds.
- **MERGE:** `GuardrailResult` into `RiskAssessment` (two models, same data); de-dup `_extract_latest_user_message`.
- **FIX (security):** `LLMGuardrailBackend` **fails open** (returns SAFE on LLM/import failure) — add `fail_open: bool`, default to BLOCK.
- **KEEP:** `GuardrailBackend` ABC, `RiskLevel`/`RiskAssessment`, `GuardrailError`, `BaseGuardrail`, `PatternBackend`, `UserInputGuardrail`. (Receive `Rail` ABC from exo-core.)

### exo-observability — SIMPLIFY (trim unused)
- **CUT (zero callers):** `cost.py` `CostTracker`, `prompt_logger.py`, `slo.py` `SLOTracker`, `SpanConsumer`, `EventLoopCheck` (broken by design), eventually the `exo.log` shim (3rd access path).
- **MOVE → exo-distributed (C7):** `DIST_*`/`STREAM_*` semconv, `AlertManager`, `BaggagePropagator`.
- **FIX:** `metrics.py` creates a fresh OTel counter on every call (cache instruments); dual config path.
- **KEEP:** logging, tracing, core metrics, `GEN_AI_*`/`AGENT_*`/`TOOL_*` semconv, `HealthRegistry` (for web `/health`).

### exo-eval — KEEP (tighten)
- **CUT:** `OutputRelevanceScorer` (word-overlap is meaningless).
- **SIMPLIFY:** half-populated scorer registry (register all or remove); `LabelDistributionScorer` (expose only `summarize()`); fold reflection into Ralph.
- **FIX:** `LLMAsJudgeScorer(judge=None)` silent `0.0`; `RalphRunner._plan()` feedback resets instead of chaining; `SchemaValidationScorer` ignores most JSON-Schema keywords; declare `exo-core` dependency.
- **KEEP:** `Evaluator`/`Scorer`/pass@k, rule-based scorers, LLM judges, `RalphRunner`. Receives `TrajectoryItem` (C5).

### exo-train — DEMOTE TO OPTIONAL / PRE-RELEASE
- ~60% aspirational. **Do not ship in default `uv sync`.**
- **CUT / label demo:** `GaussianMutationStrategy` (simulated `exp(-lr*epoch)` loss, fake accuracy), and thus most of `evolution.py` except the ABC + pipeline shell.
- **FIX:** `asyncio.get_event_loop()` deprecation in `verl.py`; `FileCheckpointStore` non-atomic write + `json.dumps(default=str)` silent loss; misleading `VeRLTrainer.check_config()` dict-merge.
- **KEEP (real):** `Trainer` ABC, `VeRLTrainer` (gate behind `[verl]` extra), `Operator` family, `InstructionOptimizer`/`ToolOptimizer` (textual-gradient prompt opt — the real value), `SynthesisPipeline`.
- **Consider split:** `exo-train-core` (optimizers/operators/synthesis) + `exo-train-rl` (VeRL).

### exo-a2a — KEEP (simplify)
- **CUT:** unimplemented `TransportMode.GRPC`/`WEBSOCKET` and `push_notifications` flag (false surface area); `ClientManager` thread-local (misfit in async framework).
- **FIX:** `send_task_streaming` buffers full response then splits — not real streaming; `AgentCard.url` uses `port=0`; rename Protocol `TaskStore` → `A2ATaskStore` (collides with exo-distributed's class).
- **KEEP:** AgentCard/discovery, A2AServer/AgentExecutor, A2AClient/RemoteAgent, InMemoryTaskStore.

### exo-distributed — KEEP (most production-ready; surface it)
- Real, complete, tested; wired via exo-cli. **Not** in CLAUDE.md or the meta-package.
- **FIX:** `_log` → `logger` (convention); guard `MemoryPersistence` import; **document the tool-serialization gap** — `Agent.to_dict()` does not serialize `@tool` functions, so workers must have tools registered locally (biggest user footgun).
- **ADD:** to `exo-ai` as `[distributed]` extra; doc when to use vs `exo-harness` (cross-process queue vs in-process parallel).
- **KEEP:** everything (Worker/TaskBroker/TaskStore, `distributed()`/`TaskHandle`, pub/sub+replay, cancellation, fleet health, metrics/alerts, Temporal).

### exo (meta-package) — KEEP (add extras)
- Pure dependency bundle (correct). Dist name `exo-ai`, namespace `_exo_meta`.
- **ADD:** optional-extras system — `[distributed]`, `[a2a]`, `[retrieval]`, `[all]` — and include `exo-retrieval`/`exo-distributed`.
- **SIMPLIFY:** move `exo-a2a` from hard dep to `[a2a]` extra (drops httpx for everyone).

---

## Suggested execution order (Phase 2)

1. **Security fixes first** (independent, urgent): exo-web B-1/B-4; exo-guardrail fail-open.
2. **Dead-code deletion** (low risk, high signal): exo-web aspirational features, exo-core no-op flags/`SerialGroup`/`team`/`RalphNode`, exo-observability zero-caller modules, exo-search dead `@tool`s, exo-context dead processors.
3. **Stub fixes** (correctness): `exo run`, exo-mcp-cli `resource read`, exo-sandbox k8s fallback, exo-eval `judge=None`/Ralph feedback.
4. **Consolidations C1–C7** (cross-package, needs care): embeddings/vector layer, then exo-web delegation, then CLI/server/trajectory/token/semconv merges.
5. **Packaging** (meta extras, exo-train demotion, exo-distributed surfacing).
