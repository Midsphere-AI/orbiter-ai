# Agent Runtime Control Contracts

Status: Accepted  
Story: US-002  
Date: 2026-03-08

## Goal

Define the runtime-facing contract for planning, budget awareness, hidden tool-argument injection, sub-agent controls, HITL gating, MCP progress emission, and tool-result metadata before implementation work starts. The contract must round-trip cleanly across core `Agent` objects, runner execution, serialized configs, distributed workers, Temporal payloads, and Exo Web storage/runtime.

## Current Inventory And Exact Touchpoints

| Surface | Current touchpoints | Why this story must touch them |
| --- | --- | --- |
| Core agent object | `packages/exo-core/src/exo/agent.py` in `Agent.__init__`, `get_tool_schemas()`, `_execute_tools()`, `_make_spawn_self_tool()`, `to_dict()`, and `from_dict()` | All new config fields live on `Agent`, tool-schema injection happens here, and sub-agent/HITL behavior must be enforced here. |
| Core runner and events | `packages/exo-core/src/exo/runner.py` in `run()` and `_stream()`; `packages/exo-core/src/exo/types.py` in `StatusEvent`, `ToolResult`, `ToolResultEvent`, and `MCPProgressEvent` | Planning must run before executor turns, MCP progress gating is emitted here, and tool-result metadata must be surfaced here. |
| Serialization and parsing | `packages/exo-core/src/exo/config.py` (`AgentConfig`) and `packages/exo-core/src/exo/_internal/output_parser.py` (`parse_tool_arguments()`) | The new fields must survive `Agent.to_dict()/from_dict()` and injected args must be split away from executable kwargs here. |
| Distributed execution | `packages/exo-distributed/src/exo/distributed/models.py` (`TaskPayload.agent_config`), `worker.py`, `events.py`, and `temporal.py` | Distributed workers and Temporal rebuild agents from serialized config. Event transport must carry the new metadata and optional MCP progress without drift. |

## Decisions

### 1. Agent Config Fields

These fields are added to the core agent contract and serialized config:

| Field | Type | Default | Validation | Runtime contract |
| --- | --- | --- | --- | --- |
| `planning_enabled` | `bool` | `false` | none | Run a planner phase before the executor phase. |
| `planning_model` | `str \| null` | `null` | Must parse as a normal Exo model string when set. | When null, planning uses the executor model/provider. |
| `planning_instructions` | `str` | `""` | none | When empty, use Exo’s built-in planner prompt. |
| `budget_awareness` | `str \| null` | `null` | Only `per-message` or `limit:<0-100>` are valid. | Controls context-budget warnings and hard stops. |
| `hitl_tools` | `list[str]` | `[]` | Every name must exist on the agent after tool registration. | Matching tool calls pause for approval before execution. |
| `emit_mcp_progress` | `bool` | `true` | none | `false` suppresses `MCPProgressEvent` emission only. |
| `injected_tool_args` | `dict[str, str]` | `{}` | Keys must be valid JSON object property names; values are descriptions shown in tool schema. | Adds schema-only optional fields to every exposed tool. |
| `allow_parallel_subagents` | `bool` | `false` | none | Enables a dedicated parallel-subagent tool. |
| `max_parallel_subagents` | `int` | `3` | Must be `1..7`. | Caps the number of child jobs in a single parallel-subagent call. |

Exact code touchpoints:

- `packages/exo-core/src/exo/agent.py`
- `packages/exo-core/src/exo/config.py`
- `packages/exo-core/tests/test_serialization.py`

### 2. Planning Contract

`planning_enabled` introduces a planner phase, not a second persisted conversation.

Rules:

- The planner runs before the first executor model call.
- The planner may use the same tool schemas as the executor.
- Planner messages, planner tool chatter, and planner tool results never enter the executor `messages` list.
- Planner output is reduced to a structured internal plan artifact and optional run/debug metadata.
- `planning_model` is a pure override for the planner phase only; it never mutates `agent.model`.
- `planning_instructions` scopes only to the planner phase.

Exact touchpoints:

- `packages/exo-core/src/exo/runner.py`: add a planner pre-pass before the normal execution loop.
- `packages/exo-core/src/exo/agent.py`: surface planner fields on the agent and keep them in `to_dict()/from_dict()`.

Decision: do not reuse `agent_plans` for runtime planner transcripts. `agent_plans` is a user-facing, persistent web feature; runtime planning is ephemeral execution state.

### 3. Budget Awareness Contract

`budget_awareness` has exactly two enabled modes:

- `per-message`
- `limit:<0-100>`

Rules:

- `null` disables budget-aware behavior.
- `per-message` adds an advisory string to every assistant message and tool result that is surfaced back to the model.
- `limit:<n>` forces a compaction pass before the next model call whenever the context exceeds the configured percentage of the window.
- If compaction still cannot get the prompt below the configured limit, the run ends with a deterministic budget-limit message instead of making another provider call.

Exact touchpoints:

- `packages/exo-core/src/exo/runner.py`: current token tracking and context windowing already live here.
- `packages/exo-core/src/exo/agent.py`: `_get_context_window_tokens()` and `_update_system_token_info()` already exist and should be reused.
- `packages/exo-core/src/exo/types.py`: the deterministic stop message should remain a normal result, not a hidden exception path.

### 4. Injected Tool Args Contract

`injected_tool_args` is schema-only and must never pollute executable kwargs.

Rules:

- `Agent.get_tool_schemas()` deep-copies each tool schema and merges every configured injected field as an optional property.
- The underlying `Tool.parameters` object is never mutated.
- `parse_tool_arguments()` must split raw model-returned args into:
  - executable `arguments`
  - `injected_args`
- Only executable `arguments` are passed to `tool.execute(**kwargs)`.
- `injected_args` are copied into tool-result metadata and approval records when relevant.

Exact touchpoints:

- `packages/exo-core/src/exo/agent.py`: `get_tool_schemas()`
- `packages/exo-core/src/exo/_internal/output_parser.py`: split injected args away from executable args
- `packages/exo-core/src/exo/types.py`: carry injected args in tool-result metadata

Decision: `injected_tool_args` stays `dict[str, str]` rather than full JSON Schema so it round-trips cleanly through JSON, DB text storage, and distributed payloads without introducing per-provider schema drift in the first implementation.

### 5. HITL Tool Contract

`hitl_tools` is an allowlist of tool names already present on the agent.

Rules:

- Validation happens at agent construction, deserialization, and Exo Web save time.
- If a tool call targets a HITL-marked tool, Exo creates an approval request before executing the tool.
- The tool call returns one of three deterministic approval states:
  - `approved`
  - `rejected`
  - `timed_out`
- Approval state is recorded in tool-result metadata and in the streaming event.

Exact touchpoints:

- `packages/exo-core/src/exo/agent.py`: gate tool execution inside `_execute_tools()`
- `packages/exo-core/src/exo/types.py`: add approval fields to tool-result metadata

Decision: introduce a run-scoped approval record in exo-core that serves tool approvals, keyed by run and tool-call id.

### 6. Parallel Sub-Agent Contract

There are two sub-agent paths:

- `spawn_self(...)` for a single child
- a new parallel-subagent tool enabled by `allow_parallel_subagents`

Parent config:

- `allow_parallel_subagents=false` means the parallel-subagent tool is absent.
- `max_parallel_subagents` caps the number of child jobs per call and may never exceed `7`.

Child-call contract:

- `task: str` is required.
- `additional_context: str = ""` is optional.
- `tool_names: list[str] | null = null` optionally narrows the child tool set.
- `output_schema: dict[str, Any] | null = null` optionally declares a JSON Schema contract for the child result.
- `system_prompt: str = ""` is allowed on the parallel-subagent tool for per-child framing.

Rules:

- `tool_names`, when set, must be a strict subset of the parent’s child-safe tools.
- Child overrides affect only the child execution context and never mutate the parent agent.
- `output_schema` is JSON Schema, not an import path to a Pydantic class, because it must round-trip through tool calls and distributed payloads.
- Every child result is validated before aggregation.
- Aggregated parallel results are structured as per-job success/error entries, never free-form concatenated text.

Exact touchpoints:

- `packages/exo-core/src/exo/agent.py`: `_make_spawn_self_tool()`
- `packages/exo-core/src/exo/_internal/agent_group.py`: existing `asyncio.TaskGroup` pattern is the right concurrency primitive
- `packages/exo-core/src/exo/types.py`: add structured child-result metadata as needed

Decision: keep child tool subsets and `output_schema` as call-time contracts, not agent-config fields. The parent agent config only decides whether parallel sub-agents are enabled and how many can run.

### 7. Tool-Result Metadata Contract

Add a new metadata object carried by both `ToolResult` and `ToolResultEvent`.

Proposed shape:

```python
class ToolResultMetadata(BaseModel):
    status: Literal["success", "error", "rejected", "timed_out"]
    started_at: float
    completed_at: float
    execution_time_ms: float
    approval_status: Literal["not_required", "approved", "rejected", "timed_out"]
    approval_id: str | None = None
    injected_args: dict[str, Any] = Field(default_factory=dict)
    offloaded_artifact_id: str | None = None
```

Rules:

- `ToolResult.metadata` becomes the canonical runtime record.
- `ToolResultEvent.metadata` mirrors the same object.
- Existing `success`, `error`, and `duration_ms` fields on `ToolResultEvent` stay for backward compatibility in the first implementation slice.
- When a large tool result is offloaded, record the artifact id in metadata instead of requiring callers to parse the pointer string.

Exact touchpoints:

- `packages/exo-core/src/exo/types.py`
- `packages/exo-core/src/exo/agent.py`
- `packages/exo-core/src/exo/runner.py`
- `packages/exo-distributed/src/exo/distributed/events.py`

### 8. MCP Progress Contract

`emit_mcp_progress` defaults to `true` for backward compatibility.

Rules:

- `false` suppresses only `MCPProgressEvent` emission.
- Tool execution still proceeds identically.
- Suppression happens in the runner after tool execution has drained MCP progress queues.

Exact touchpoints:

- `packages/exo-core/src/exo/runner.py`: current MCP queue draining is unconditional and must become flag-aware
- `packages/exo-distributed/src/exo/distributed/events.py`: `_EVENT_TYPE_MAP` currently omits `mcp_progress`
- `packages/exo-distributed/src/exo/distributed/temporal.py`: current Temporal activity collects only `TextEvent` and silently drops progress and tool-result events

Decision: `emit_mcp_progress` is purely an emission toggle. It is not a tool behavior toggle and it must not change the tool call/result contract.

## Serialized Agent-Config Example

Defaults:

- `planning_enabled=false`
- `planning_model=null`
- `planning_instructions=""`
- `budget_awareness=null`
- `hitl_tools=[]`
- `emit_mcp_progress=true`
- `injected_tool_args={}`
- `allow_parallel_subagents=false`
- `max_parallel_subagents=3`

Configured example:

```json
{
  "name": "ops-supervisor",
  "model": "openai:gpt-4o",
  "instructions": "Plan first, execute carefully, and escalate dangerous actions.",
  "max_steps": 12,
  "temperature": 0.2,
  "max_tokens": 4096,
  "planning_enabled": true,
  "planning_model": "openai:gpt-4o-mini",
  "planning_instructions": "Return a short numbered execution plan before acting.",
  "budget_awareness": "limit:70",
  "hitl_tools": ["deploy_service", "rotate_credentials"],
  "emit_mcp_progress": true,
  "injected_tool_args": {
    "ui_request_id": "Opaque UI correlation id exposed only in tool schemas.",
    "run_origin": "Short label for the caller surface, such as playground or workflow."
  },
  "allow_parallel_subagents": true,
  "max_parallel_subagents": 4
}
```

Behavior notes:

- If `planning_enabled` were `false`, `planning_model` and `planning_instructions` would be ignored.
- If `budget_awareness` were `null`, neither advisory strings nor hard budget stops would run.
- If `emit_mcp_progress` were `false`, the same MCP tools would run but no `mcp_progress` events would be emitted.
- If `allow_parallel_subagents` were `false`, the parallel-subagent tool would not be registered even if `max_parallel_subagents` remained `4`.

## Rejected Designs

These are explicit non-goals for implementation stories that depend on this memo.

- Reject planner-history mixing. Planner prompts, planner tool calls, and planner tool results must not be appended to executor message history.
- Reject raw tool-kwarg pollution from injected fields. Injected schema-only fields are captured separately and never passed into `tool.execute(**kwargs)`.
- Reject parallel-subagent counts above `7`. Validation belongs in core agent construction, deserialization, and per-call runtime checks.
- Reject `hitl_tools` names that are not present on the agent. Save-time and deserialize-time validation should fail fast.

## Implementation Order For Follow-On Stories

1. Extend `Agent`, `AgentConfig`, and `Agent.to_dict()/from_dict()` with the new scalar/list/map fields.
2. Add planner pre-pass, budget-awareness enforcement, and injected-arg splitting in core runner/parsing.
3. Add tool-result metadata and approval-aware tool execution in core types and agent execution.
4. Bring distributed worker, event transport, and Temporal execution to parity, including `mcp_progress`.

This order minimizes interface churn because every later story can depend on one serialized agent contract and one tool-result metadata shape.
