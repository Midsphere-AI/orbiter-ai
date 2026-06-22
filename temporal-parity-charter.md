# Temporal Parity Charter

**Status:** Proposed — pending implementation
**Branch target:** continue on `chore/distribution-cleanup` (or a fresh `feat/temporal-parity`)
**Scope:** `packages/exo-distributed/` (Temporal integration) + a determinism audit of `packages/exo-core/src/exo/_internal/`
**Goal:** Move exo's Temporal support from a thin wrapper to **100% feature parity** — every Temporal primitive available out of the box, mapped onto an exo concept.

---

## 1. The core reframe

Today the entire agent run is buried inside **one activity** (`execute_agent_activity` in
`temporal.py`). Temporal can only observe "an activity ran." Every missing feature —
resumption, signals, queries, child workflows, per-step retry — requires lifting the agent's
**step structure** out of the activity and into the **workflow**, so Temporal can see and
control each step.

The rule that governs everything:

> **Workflow code is deterministic** (replayed from event history). **Activities hold all
> non-determinism** (LLM calls, tool I/O, wall-clock time, randomness, network).

exo's current loop interleaves the two. Parity means splitting them:

```
Workflow (deterministic orchestrator)          Activities (all I/O)
─────────────────────────────────────          ────────────────────
 AgentWorkflow.run()                            llm_turn_activity()      ← one LLM call
   loop:                                        tool_call_activity()     ← one tool exec
     turn = await execute_activity(llm_turn)    memory_load/save_activity()
     if turn.tool_calls:                        guardrail_activity()
       results = await gather(                   embed_activity()
         execute_activity(tool_call) ...)
     handle signals / queries / updates
     if history too big: continue_as_new()
```

This is the difference between "Temporal as a retrying job queue" (today) and "Temporal as a
durable execution engine" (the goal).

---

## 2. Current state (what exists today)

`packages/exo-distributed/src/exo/distributed/temporal.py` (~250 lines):

- `execute_agent_activity` — runs an entire `Agent`/`Swarm` via `run.stream`, collects
  `TextEvent` text, heartbeats every 10 events.
- `AgentExecutionWorkflow` — one `@workflow.run` that calls the one activity with a
  `start_to_close_timeout` and `heartbeat_timeout`.
- `TemporalExecutor` — `connect`/`disconnect`/`execute_task`/`start_temporal_worker`.
- Wired into `worker.py` via `executor="temporal"` (`Worker.__init__`).

**What works:** worker-crash survival via heartbeat; basic durable run.
**What's overstated:** the docstring claims "full state recovery" — but the activity replays
from step 0 on retry (in-memory `text_parts`, heartbeat details never read). It's
restartable, not resumable.

**Honest coverage today: ~20% of Temporal's surface.**

---

## 3. Feature-by-feature parity matrix

| Temporal feature | exo concept it maps to | What it takes | Phase |
|---|---|---|---|
| Activity retry policy | per-tool / per-LLM-call retry | `RetryPolicy` on each `execute_activity`; `TaskPayload.retry_policy` + per-tool overrides | 1 |
| Heartbeat details (coarse, manual resumption) | resume mid-run after crash **without** moving the loop into the workflow | checkpoint agent state into `activity.heartbeat(state)`, read `activity.info().heartbeat_details` on retry. **This is NOT Temporal's signature checkpoint-restore** (see §3.1) — it's a manual, coarse cousin. It only makes today's whole-agent-in-one-activity design crash-resumable; the real thing needs Phase 2. | 1 |
| **Automatic checkpoint-restore** (the headline feature) | agent loop resumes exactly where it stopped, zero checkpoint code | requires the loop to live in the **workflow** (each LLM turn / tool call = a recorded history event). Replay reconstructs state for free. **Only achievable in Phase 2.** | 2 |
| Activity timeouts (4 kinds) | task/tool timeouts | map `schedule_to_close` / `start_to_close` / `schedule_to_start` / `heartbeat` to a `TimeoutConfig` namespace | 1 |
| Activity cancellation | `CancellationToken` | Temporal raises `CancelledError` into the activity on heartbeat → flip the token. Replaces today's "checked once before start" | 1 |
| Data converter / payload codec | encrypt agent configs/outputs at rest | agent payloads carry prompts, keys, PII; Temporal stores full history. A `PayloadCodec` (AES) is effectively mandatory for production | 1 |
| Failure converter | map `ExoError` hierarchy → Temporal failures | retry/non-retry decisions respect exo's error taxonomy (ties into the error-DX effort) | 1 |
| Worker + client interceptors | tracing + metrics | wire exo OTel (`aspan`, `BaggagePropagator` already in `worker.py`) into Temporal interceptors so traces span workflow→activity | 1 |
| TLS / mTLS / API key | Temporal Cloud | today hardcoded insecure localhost → a `ConnectionConfig` (TLS, API key, cloud namespace) | 1 |
| Worker tuning (slot suppliers, max concurrent, thread/process pools) | `Worker(concurrency=...)` | map exo worker concurrency onto Temporal tuner | 1 |
| Workflow Signals | human-in-the-loop, steering, mid-run input injection | `@workflow.signal` → exo's message-injection default maps 1:1 | 2 |
| Workflow Queries | live progress / current state | `@workflow.query` returns step count, last tool, partial output — feeds the streaming story without Redis | 2 |
| Workflow Updates | validated synchronous steering | `@workflow.update` + validator → change model mid-run, approve a tool call, with a return value | 2 |
| Continue-As-New | long agents exceeding history limits | after N steps / ~10MB history, serialize `agent.to_dict()` + messages and continue | 2 |
| Timers / durable sleep / `wait_condition` | scheduled retries, backoff, "wait then resume" | `workflow.sleep()` for delayed tool retries and human-response waits | 2 |
| Local activities | cheap deterministic-ish steps (formatting, regex guardrail) | lower overhead than full activities for sub-second pure steps | 2 |
| Async activity completion | long external tool calls (human approval, external job) | a tool returns a task token, completed out-of-band | 2 |
| Child workflows | **`Swarm` → parent; each `Agent` → child** | parent `SwarmWorkflow` spawns `AgentWorkflow` children; sub-agent orchestration gets per-agent durability | 3 |
| Schedules | cron agents / recurring jobs | `client.create_schedule(...)` → a `ScheduledAgent` API; replaces ad-hoc cron | 3 |
| Search attributes + memo | task queryability (by agent, model, tenant) | typed search attributes on start → ops query "all running gpt-4o swarms for tenant X" | 3 |
| Saga / compensation | rollback partial Swarm / tool side-effects | compensation stack in the workflow; register undo per tool | 3 |
| Replayer | determinism testing | replay recorded histories to catch non-deterministic agent-loop changes — critical | 3 |
| Versioning / patching | safe runtime upgrades | `workflow.patched()` guards when the agent loop changes, so in-flight runs don't break | 3 |
| Nexus (cross-namespace) | multi-tenant / cross-service agents | optional, advanced — Temporal's newest primitive | 3 (stretch) |

### 3.0 Second-pass audit — features the first matrix missed

A deliberate sweep of the full SDK + platform surface (not just the marketing headlines)
surfaced these. They are real parity gaps, not nice-to-haves.

| Temporal feature | exo concept it maps to | What it takes | Phase |
|---|---|---|---|
| **Pydantic data converter** (`temporalio.contrib.pydantic`) | clean typed serialization of `TaskPayload`, agent configs, events | replace the hand-rolled `json.dumps(model_dump())` / `json.loads` with the pydantic payload converter. exo is pydantic-v2 everywhere — this is the correct serialization spine, not an option | 1 |
| **Workflow-level timeouts** (`workflow_execution_timeout`, `workflow_run_timeout`, `workflow_task_timeout`) | overall task deadline vs per-run vs per-decision | distinct from the 4 activity timeouts already in the matrix; surface in `TimeoutConfig` | 1 |
| **Workflow-level retry policy** | retry the whole agent run, not just a step | `RetryPolicy` at `start_workflow`; separate knob from activity retry | 1 |
| **Codec server** (companion to PayloadCodec) | view encrypted payloads in Web UI / `tctl` | ship/operate a codec server alongside the `PayloadCodec`, or payloads are opaque to ops | 1 |
| **Python workflow sandbox + passthrough modules** | let exo's `_internal` run as workflow code at all | the SDK reimport-sandbox bans non-deterministic stdlib; `_internal` trips it. Configure passthrough modules / `with_unrestricted_imports`. See §5(5) | 2 |
| **Side effects** (`workflow.side_effect` / `mutable_side_effect`) | deterministic capture of unavoidable non-determinism | broader than `workflow.uuid4()`/`now()`; for any one-off non-deterministic read in the loop | 2 |
| **Cancellation scopes + shielding** | guaranteed compensation/cleanup on cancel | makes saga (§3 Phase 3) actually fire its undo when a run is cancelled mid-flight | 2 |
| **Signal-with-start / Update-with-Start** | "ensure this agent session exists, then steer it" | lazy-init pattern on the client | 2 |
| **Time-skipping test environment** (`start_time_skipping`) | test durable timers / schedules / `wait_condition` | without it, timer tests wait real wall-clock. Pairs with the Replayer | 3 |
| **Worker Build-ID versioning / deployments** | safe rolling upgrades of long-lived workflows | distinct from `workflow.patched()` — the modern recommended mechanism. Charter previously conflated the two | 3 |
| **Reset + batch operations** | ops: reset a workflow to a past event; bulk cancel/signal/terminate | client/ops-plane; lower priority but part of "everything" | 3 |

### 3.1 The checkpoint-restore distinction (read this before promising "durability")

Temporal's signature pitch — *"your code never loses its place"* — is **automatic** and lives
**only at the workflow layer**. Workflows persist every step to event history; on crash,
Temporal replays the history to reconstruct state and resumes exactly where it stopped. You
write **zero checkpoint code.**

**Activities do not get this.** A crashed activity is **re-run from the start** (hence
activities must be idempotent). The only way to resume *inside* an activity is **manual**
`activity.heartbeat(details)` checkpointing — coarse, and you write it yourself.

The consequence for exo: **today the entire agent run is one activity**, so the current design
can *never* offer the real automatic checkpoint-restore — only the manual heartbeat cousin
(Phase 1). The headline feature requires the agent loop to move into the **workflow** so each
LLM turn / tool call becomes a replayable history event (Phase 2). The current `temporal.py`
docstring already claims "full state recovery" — that claim is **false until Phase 2 lands**;
Phase 1 should downgrade the wording to "crash-resumable via heartbeat checkpointing."

---

## 4. The architecture in three layers

**1. `AgentWorkflow` — the durable agent loop.** A deterministic re-implementation of exo's
run loop where each LLM turn and tool call is an `execute_activity`. The hard, central piece:
the `_internal/call_runner.py` → `message_builder` → `output_parser` → `handlers` chain must
be split so **message assembly / routing (deterministic) stays in the workflow** and **LLM /
tool I/O moves to activities.**

**2. `SwarmWorkflow` — orchestration.** Maps exo's Swarm DAG (`_internal/graph.py`,
`agent_group.py`) onto child workflows; `ParallelGroup` → `asyncio.gather` over child handles.

**3. `TemporalExecutor` — control plane.** Grows the most: signals/queries/updates
passthrough, schedules API, codec, interceptors, `ConnectionConfig`, async completion.

---

## 5. The hard problems (where parity bites)

1. **Determinism of exo's agent loop is the whole ballgame.** Anything in the orchestration
   path that does I/O, reads wall-clock time, uses `random`/`uuid`, or depends on dict
   iteration order breaks replay. `_internal` was not written under this constraint. True
   100% requires the audit in §7.

2. **State serialization.** Resumption and Continue-As-New need full agent state — messages,
   memory, checkpoints, sub-agent state — round-trippable. `to_dict`/`from_dict` and
   checkpoints exist (good) but must become **complete and stable across versions.**

3. **History size + payload limits.** Temporal caps history (~50K events / 50MB). Long runs
   blow it → mandatory Continue-As-New, and large tool outputs need the **claim-check pattern**
   (store blob externally, pass a reference). Ties into the codec.

4. **Versioning.** Once workflows are durable and long-lived, every change to the agent loop
   needs `workflow.patched()` guards (or Build-ID worker versioning) plus a Replayer test, or
   deploys break in-flight runs.

5. **The Python workflow sandbox (§5.5).** Temporal's Python SDK runs workflow code inside a
   reimport sandbox that bans non-deterministic stdlib (`time`, `random`, threading, most I/O)
   and reimports modules per workflow. Moving any part of exo's `_internal` into the workflow
   layer (Phase 2) will trip this immediately — `_internal` imports observability, logging,
   registry, and provider plumbing. You must mark those as **passthrough modules** (loaded once,
   not sandbox-reimported) and ensure nothing in the deterministic path actually *calls*
   non-deterministic stdlib. This is the single most likely Phase-2 blocker and should be
   spiked early with a one-activity-one-turn prototype before committing to the full split.

---

## 6. Phasing

### Phase 1 — make the current wrapper production-real (no re-architecture)
**Status: COMPLETE** (branch `chore/distribution-cleanup`). `temporal.py` refactored into a
`temporal/` package; all green (ruff + pyright clean, full repo suite passes, ~150 new tests).
Ships ~40% of parity without touching `_internal`, and corrects the false durability claims.
- `RetryPolicy` on the activity + `TaskPayload.retry_policy`. ✅
- Heartbeat-detail resumption (checkpoint into heartbeat, restore on retry). ✅ *(coarse: logs
  prior progress on retry; the single-activity design re-runs from the start — see §3.1)*
- Real activity cancellation wired to `CancellationToken`. ✅ *(Temporal→activity direction)*
- `ConnectionConfig` — TLS / mTLS / API key / cloud namespace. ✅
- `PayloadCodec` (AES) + claim-check for large payloads. ✅
- Failure converter mapping `ExoError` → retryable/non-retryable. ✅
- OTel interceptor (workflow→activity trace continuity). ✅
- Worker tuning knobs mapped from exo concurrency. ✅

**~~Known Phase-1 gap~~ — CLOSED in Phase 2:** cancellation is now **bidirectional**.
`TemporalExecutor.execute_task()` awaits the workflow result and a concurrent watch on the
`CancellationToken` (`_await_with_cancel`); a cancel flipped on the token calls `handle.cancel()`
so the in-flight Temporal run actually stops. `CancellationToken` gained an awaitable `wait()`
(backed by an `asyncio.Event`, fully backward compatible with the sync `cancelled`/`cancel()`
API) so the watch is event-driven, not a busy-poll.

### Phase 2 — durable agent loop (`AgentLoopWorkflow`, step-as-activity)
**Status: CORE COMPLETE** (branch `chore/distribution-cleanup`). Shipped as an *opt-in* path
(`TemporalExecutor(durable=True)`) **additively** alongside the single-activity workflow — all
green (ruff + pyright clean; 568 exo-distributed tests pass, incl. real end-to-end
`WorkflowEnvironment.start_time_skipping()` runs of the durable loop, signals, queries, updates,
continue-as-new and cancellation).

Design that sidesteps the §5/§7 hazards without a `_internal` rewrite: the **workflow** is a thin
deterministic orchestrator that manipulates only the serialized message history (plain JSON dicts)
+ counters — it imports no `exo._internal`, calls no `uuid`/`time`/`sleep`, and dispatches
activities **by name** so its import graph never drags exo into the replay sandbox. Memory is kept
off in the durable loop (history lives in workflow state), which also sidesteps the `uuid4`
conversation-id determinism hazard from §7. Each activity reconstructs the agent from the frozen
`agent_config` and does exactly one unit of I/O, reusing exo's real runtime per step:
- `agent_llm_turn` → one LLM call via `Agent._call_llm` (preserves `PRE/POST_LLM_CALL` hooks +
  guardrails + retry). ✅
- `agent_tool_call` → one tool via `Agent._execute_tools` (preserves `PRE/POST_TOOL_CALL` hooks,
  `injected_tool_args` stripping, `ToolContext`, `large_output` offloading). ✅
- **Per-step checkpoint-restore** — a crashed worker resumes from the last completed step (the
  real headline feature), because every turn/tool result is a recorded history event. ✅
- **Signals** — `inject_message` (mid-run message injection, maps to exo's injection default) +
  `cancel` (graceful stop at the next step boundary). ✅
- **Queries** — `get_progress` returns live `{step, total_steps, message_count, last_text, done,
  cancel_requested}`. ✅
- **Updates** — `steer` (validated mid-run instruction with ack) + `set_model` (swap model for
  subsequent turns), each with a `@*.validator`. ✅
- **Continue-As-New** — rolls over on a history-length threshold, carrying messages + accumulated
  `total_steps`. ✅
- Sandbox handled via `build_workflow_runner()` (passthrough of the `exo` package); §5.5 blocker
  resolved by dispatching activities by name + not importing `_internal` into the workflow. ✅

**Deferred within Phase 2** (do not claim these yet): in-loop context-window summarization /
memory-persistence / planning pre-pass parity (currently loop-level concerns not yet folded into
the durable path); Timers / `wait_condition`; local activities; async activity completion;
cancellation scopes + shielding; signal-with-start / update-with-start. Swarm in the durable loop
is rejected with a clear error (Phase 3).

### Phase 3 — orchestration + ops
- `SwarmWorkflow` child workflows.
- Schedules API (`ScheduledAgent`).
- Search attributes + memo.
- Saga / compensation.
- Replayer test suite + versioning discipline.
- Nexus (stretch).

---

## 7. Determinism audit checklist for `_internal/`

Before Phase 2, audit the orchestration path for replay hazards. Each must move to an activity
or a Temporal-safe equivalent (`workflow.now()`, `workflow.uuid4()`, deterministic ordering):

- [ ] Wall-clock reads (`time.time()`, `datetime.now()`) in the loop / state machine.
- [ ] `random` / `uuid4` used for IDs in `state.py`, `call_runner.py`.
- [ ] Network/file I/O outside the LLM/tool boundary (memory loads, registry lookups).
- [ ] Non-deterministic iteration order (dict/set ordering affecting tool dispatch in `handlers.py`).
- [ ] `asyncio` primitives that assume real concurrency (`gather` over tools is fine *if* mapped to activities; raw `create_task` against I/O is not).
- [ ] Loop-detection / state transitions in `call_runner.py` that depend on timing.
- [ ] Anything in `message_builder.py` that reads live config/env mid-build rather than from the frozen payload.

Deliverable of the audit: a list of "lift to activity" vs "make deterministic" decisions per
module, gated by a Replayer test that fails on any non-determinism.

---

## 8. Acceptance criteria for "100% parity"

- Every row in §3 has a working code path and a test.
- A long agent run survives a worker kill mid-tool and **resumes from the last completed step**
  (not from step 0).
- A running workflow accepts a signal (inject message), answers a query (live state), and an
  update (steer) — all from the `TemporalExecutor` client.
- A `Swarm` runs as parent + child workflows with per-agent durability.
- A Replayer test suite guards the agent loop against non-deterministic regressions.
- Temporal Cloud connection (TLS + API key) works; payloads are encrypted at rest.
