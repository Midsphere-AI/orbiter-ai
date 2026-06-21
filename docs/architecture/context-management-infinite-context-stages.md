# Context Management and Infinite Context Stages

Status: Accepted  
Story: US-003  
Date: 2026-03-09

## Goal

Define an evidence-backed path for Exo's long-context work. The decision for this story is that Exo should stage "infinite context" as provider-agnostic, branch-scoped summaries, checkpoints, retrieval, and artifact offloading. It should not promise unlimited raw-history replay or depend on opaque vendor state as its source of truth.

## Current Inventory And Exact Touchpoints

| Surface | Current touchpoints | What exists today |
| --- | --- | --- |
| Core agent prompt assembly | `packages/exo-core/src/exo/agent.py` in `_run_inner()`, `_apply_context_windowing()`, `_inject_long_term_knowledge()`, `_offload_large_result()`, `branch()`, and `_make_spawn_self_tool()` | The real runtime path already does history loading, transient summarization, trimming, vector injection, branch copy, and large-result offload. |
| Streaming visibility | `packages/exo-core/src/exo/runner.py`, `packages/exo-core/src/exo/types.py` (`ContextEvent`) | Streamed runs emit context actions today, but only for the transient helper path. |
| Short-term persistence | `packages/exo-memory/src/exo/memory/persistence.py`, `base.py`, `short_term.py` | Human, AI, and tool results are persisted and replayed into later runs. |
| Long-term / vector search | `packages/exo-memory/src/exo/memory/long_term.py`, `backends/vector.py` | Exo has keyword long-term memory plus in-memory and Chroma-backed vector search. |
| Context primitives | `packages/exo-context/src/exo/context/context.py`, `state.py`, `checkpoint.py`, `token_tracker.py` | Branchable state, in-memory checkpoints, and token tracking already exist as standalone primitives. |
| Prompt-building / processors / workspace | `packages/exo-context/src/exo/context/prompt_builder.py`, `processor.py`, `workspace.py`, `_internal/knowledge.py`, `tools.py`, `neuron.py` | Exo has a richer context engine package, but the main `Agent.run()` path does not use it end-to-end. |
| Evidence tests | `tests/integration/test_context_summarization.py`, `tests/integration/test_context_vector_injection.py`, `tests/integration/test_branching_isolation.py`, `tests/integration/test_spawn_memory_isolation.py`, `packages/exo-context/tests/test_context_integration.py` | These prove the current behavior and current limits more reliably than the guides alone. |

## What Exo Does Today

### 1. Summarization And Trimming

- `Agent._run_inner()` loads persisted history through `MemoryPersistence.load_history()`, appends the new user turn, and then calls `_apply_context_windowing()` before the provider call.
- `_apply_context_windowing()` always applies operations in this order:
  1. aggressive trim when `offload_threshold` is exceeded
  2. LLM summarization when `summary_threshold` or `force_summarize` is hit
  3. final history-round trimming
- The "offload" path is only a destructive trim to the last `summary_threshold` non-system messages. It does not create a checkpoint, artifact, or persisted summary.
- Summarization reuses `exo-memory`'s `generate_summary()` and injects a transient `SystemMessage` with `[Conversation Summary]`.
- That summary is not persisted to short-term memory, long-term memory, checkpoints, or workspace. `tests/integration/test_context_summarization.py` explicitly documents this as the current behavior.
- Token budget handling is reactive. The agent waits for a completed model call, reads `usage.input_tokens`, and only then forces an early summarization pass for the next step.
- There is no pre-call budget gate, no persisted summary chain, no retrieval of old summaries, and no deterministic "still over budget after compaction" stop result yet.
- Because history windowing runs after summary injection, a low `history_rounds` setting can trim away the just-created summary. The current tests avoid that by keeping `history_rounds` high.

### 2. Vector Injection

- `_inject_long_term_knowledge()` searches `memory.long_term` with the current user input and injects up to 5 hits as a `<knowledge>` block in the system message.
- `VectorMemoryStore` and `ChromaVectorMemoryStore` do semantic search; `LongTermMemory` uses keyword matching.
- Retrieval is immediate and stateless. There is no reranking step, no retrieved-summary cache, no source scoring surfaced back to the caller, and no branch-aware filter.
- The injected knowledge is plain text. It is not tied to checkpoint versions, summary artifacts, or workspace citations.

### 3. Branch Isolation

- `Context.fork()` creates a child context with inherited reads and isolated writes by chaining `ContextState(parent=...)`.
- `Context.merge()` merges only child-local state plus the net token delta since fork time.
- `spawn_self()` gives the child a fresh short-term memory store, shares the parent's long-term memory store, and attempts to fork the parent's context.
- `Agent.branch()` copies raw persisted conversation items up to a chosen message id into a new `conversation_id`, which gives short-term-memory isolation by `metadata.task_id`.
- The current isolation boundary is incomplete:
  - short-term conversation history is isolated
  - local context-state writes are isolated when a real child `Context` instance is used
  - long-term memory is still shared
  - workspace objects can still be shared by reference if they live in inherited context state
- Core checkpoints are also incomplete for branch continuation. `Context.snapshot()` writes only to the in-memory `CheckpointStore` owned by that `Context` instance.

### 4. Memory Integration

- If the caller does not pass `memory=...`, `Agent` auto-creates `AgentMemory(short_term=ShortTermMemory(), long_term=default_store)` when `exo-memory` is importable.
- `MemoryPersistence` hooks persist AI responses and tool results; the user turn is saved before the provider call.
- `ShortTermMemory` already knows how to scope by user/session/task, keep the last N conversation rounds, and remove incomplete trailing AI/tool-call pairs.
- The core/runtime memory-management path uses hook-based message persistence plus transient `_apply_context_windowing()`.

### 5. Workspace And Artifact Behavior

- Every agent registers `retrieve_artifact`.
- `_execute_tools()` offloads large string tool results when `tool.large_output` is set or the string exceeds `EXO_LARGE_OUTPUT_THRESHOLD`.
- `_offload_large_result()` lazily creates `Workspace(workspace_id=f"agent_{self.name}")`, stores the content, and returns a pointer string for the model to use with `retrieve_artifact(...)`.
- The `Workspace` type itself is more capable than the agent integration:
  - version history
  - filesystem persistence when `storage_path` is set
  - observer callbacks
  - optional `KnowledgeStore` auto-indexing
  - path-traversal checks for persisted artifacts
- The agent offload path does not configure `storage_path` or `knowledge_store`, so current tool-result artifacts are process-local and not automatically searchable.
- Context tools like `get_knowledge`, `grep_knowledge`, and `search_knowledge` only work if something explicitly stores `workspace` and `knowledge_store` in `ctx.state`. `Agent.run()` does not wire that up today.

### 6. Rich Context Engine Pieces Exist But Are Not The Main Runtime

- `PromptBuilder`, neurons, and `ProcessorPipeline` are implemented and tested in `exo-context`.
- The main agent runtime still bypasses them in favor of bespoke helpers in `exo-core/src/exo/agent.py`.
- That matters for staging: replacing the whole prompt assembly stack in one go would be a much larger refactor than the PRD allows.

## External Research And Vendor Patterns

| Source | Approach | Evidence for Exo |
| --- | --- | --- |
| [OpenAI Prompt Caching](https://platform.openai.com/docs/guides/prompt-caching) and [stateful Responses API patterns](https://platform.openai.com/docs/guides/conversation-state?api-mode=responses) | Reuse stable prompt prefixes and let the server carry forward response state. | Exact-prefix caching lowers cost and latency, but it does not solve portability, branch isolation, or provider-independent replay. Exo can use cache-friendly prompt shapes, but its durable context state still needs explicit artifacts. |
| [OpenAI Codex context management](https://openai.com/index/introducing-codex/) | Codex keeps hidden reasoning items across turns and periodically compacts with a dedicated `responses/compact` step. | The useful pattern is explicit compaction checkpoints. The part Exo should not copy is dependence on opaque vendor-only compacted state as the only durable record. |
| [Anthropic prompt caching](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching) and [long-context prompting tips](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/long-context-tips) | Stable prefixes, structured prompt layout, and careful placement of long documents and queries. | Exo should keep static instructions/tools ahead of dynamic compaction state so vendor caches work, and its prompt builder should preserve clear sections for summaries, retrieved context, and recent turns. |
| [Anthropic Claude Code subagents](https://docs.anthropic.com/en/docs/claude-code/sub-agents) | Subagents run in separate context windows and keep the main thread cleaner. | This is strong support for branch-scoped compaction and isolation instead of one giant shared transcript. |
| [Google Gemini long context](https://ai.google.dev/gemini-api/docs/long-context) and [context caching](https://ai.google.dev/gemini-api/docs/context-caching) | Very large input windows postpone compaction pressure, while cached prefixes reduce repeated costs. | Bigger context windows change thresholds, not architecture. Exo still needs persisted summaries/checkpoints because long sessions, branches, and resumed runs outlive a single prompt window. |
| [MemGPT](https://arxiv.org/abs/2310.08560) | OS-like virtual memory for LLM agents with a working context and external memory. | The memory hierarchy is relevant, but autonomous memory paging is too large a jump for Exo's first staged implementation. |
| [LongMem](https://arxiv.org/abs/2306.07174) | Retrieval from an external long-term memory bank instead of replaying the whole sequence. | Exo should treat summaries and checkpoints as searchable memory objects, not just one latest summary blob. |
| [LongLLMLingua](https://arxiv.org/abs/2310.06839) | Learned prompt compression to shrink long prompts while preserving salient content. | Compression is a possible later optimization, but only after Exo has persisted artifacts and regression coverage. |
| [LoCoMo](https://arxiv.org/abs/2402.17753) | A long-conversation benchmark showing that long context alone does not guarantee reliable long-term recall. | Exo should define bounded stage goals and regression tests instead of marketing "infinite context" as solved. |

## Decision

Exo should stage context management around explicit, branch-scoped compaction artifacts:

- a persisted summary chain
- a persisted checkpoint chain
- retrieval over those artifacts
- workspace offloading that produces retrievable artifact summaries

It should not treat raw conversation replay as the only state, and it should not treat vendor-managed hidden state as the canonical store.

### Stage 1: Persisted Summary + Checkpoint Foundation

This is the next implementation stage Exo should take.

Rules:

- Persist summaries or checkpoints after successful turns instead of keeping summaries transient.
- Keep compaction branch-scoped. The unit of isolation is the conversation/branch id, not the whole agent name.
- Assemble the next prompt from:
  - the latest persisted summary or checkpoint
  - at most the 2 most recent raw turns
  - up to the top 3 retrieved relevant summaries
  - the current user turn
- Reuse the existing `ContextEvent` surface so compaction remains observable.
- Keep the artifact format provider-agnostic and serializable across local, distributed, and web paths.

Why this is the right cut:

- It directly fixes the biggest current gap: summaries are transient and disappear on the next reload.
- It stays close to the current `Agent.run()` execution path instead of forcing a wholesale runtime rewrite to `PromptBuilder`/`ProcessorPipeline`.
- It creates a durable record that later stories can load without replaying the entire raw transcript.

### Stage 2: Branch Inheritance + Workspace Alignment

After stage 1 exists, Exo should unify branch continuation and artifact retrieval.

Rules:

- Child branches may read inherited parent checkpoints and summaries at fork time.
- Child branches must write only child-scoped summary/checkpoint updates.
- Large tool-result offloads should move from process-local workspace state to persisted workspace storage.
- Offloaded artifacts should produce retrievable summary/index entries so they can participate in later compaction.
- Checkpoint and context-state inspection should read the same underlying branch-scoped artifact model as the core runtime.

Why this is the right second step:

- Current short-term branch isolation is real, but long-term memory and workspace behavior still leak across scopes.

### Stage 3: Retrieval-Aware Budget Enforcement + Optional Compression

Only after persisted artifacts and branch scopes exist should Exo add more aggressive compaction controls.

Rules:

- Add a pre-call budget policy that compacts before the next model call when usage is projected above the configured limit.
- If retrieval-aware compaction still cannot get under budget, stop with a deterministic budget-limit result.
- Treat vendor prompt caching and large context windows as cost/performance optimizations, not correctness mechanisms.
- Evaluate the combined behavior with long-conversation benchmarks and Exo integration tests before attempting more autonomous memory policies.

Why this stays third:

- Compression before persistence is hard to debug and easy to regress.
- Learned or vendor-native compaction methods become safer once Exo already has auditable summaries/checkpoints to compare against.

## Example Long-Running Conversation Flow

1. A root conversation runs for 8 turns. Raw turns are stored in short-term memory as they are today.
2. After turn 8, the runtime crosses the configured threshold and persists:
   - summary `S1` for turns 1-6
   - checkpoint `C1` with token usage, branch id, and artifact references
   Raw turns 7-8 stay uncompressed.
3. The next user turn arrives. Prompt assembly loads:
   - `C1` or `S1`
   - the 2 most recent raw turns (7-8)
   - up to 3 retrieved relevant summaries/artifact summaries
   - the new user turn
   No full transcript replay is needed.
4. A tool returns a 30 KB report. Exo writes artifact `A14` to persisted workspace storage, stores a short retrieval summary for it, and keeps only a pointer in the raw turn log.
5. The user creates a branch from this point. The child branch reads `C1` and inherited summaries but writes its own `S1-child`, `C1-child`, and artifact summaries. The parent branch is unchanged.
6. The next day, resuming the child branch loads the child's latest checkpoint, the 2 most recent child turns, and any retrieved parent/child summaries relevant to the new request. The runtime still does not need the whole raw transcript.

## Explicitly Out Of Scope

- "True infinite context" where Exo guarantees perfect recall of every historical raw token forever.
- Vendor-specific opaque compacted state blobs as Exo's only durable source of truth.
- Autonomous MemGPT-style memory management that lets the model invent its own paging policy in the first implementation slice.
- Cross-branch write-back where child or sibling branches automatically mutate parent summaries, parent checkpoints, or shared workspace history.

## Implementation Guidance For Follow-On Stories

- Keep stage 1 in the current `Agent.run()` / `runner.stream()` execution path and add persisted artifacts there first.
- Do not combine stage 1 with a full migration to `PromptBuilder`, `ProcessorPipeline`, or a brand new web storage model.
- Use the existing integration tests as the seed matrix and extend them to cover:
  - persisted-summary reload
  - checkpoint continuation
  - retrieved-summary selection
  - branch-scoped inheritance and non-write-back
  - workspace artifact retrieval after compaction

## References

- OpenAI: [Prompt Caching](https://platform.openai.com/docs/guides/prompt-caching)
- OpenAI: [Conversation state for the Responses API](https://platform.openai.com/docs/guides/conversation-state?api-mode=responses)
- OpenAI: [Introducing Codex](https://openai.com/index/introducing-codex/)
- Anthropic: [Prompt caching](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching)
- Anthropic: [Long context prompting tips](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/long-context-tips)
- Anthropic: [Claude Code subagents](https://docs.anthropic.com/en/docs/claude-code/sub-agents)
- Google: [Gemini long context](https://ai.google.dev/gemini-api/docs/long-context)
- Google: [Gemini context caching](https://ai.google.dev/gemini-api/docs/context-caching)
- Research: [MemGPT](https://arxiv.org/abs/2310.08560)
- Research: [LongMem](https://arxiv.org/abs/2306.07174)
- Research: [LongLLMLingua](https://arxiv.org/abs/2310.06839)
- Research: [LoCoMo](https://arxiv.org/abs/2402.17753)
