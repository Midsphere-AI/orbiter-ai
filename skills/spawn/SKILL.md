---
name: exo:spawn
description: "Use when an Exo agent needs to dynamically spawn copies of itself for parallel sub-tasks — allow_self_spawn, spawn_self() tool, max_spawn_depth, max_spawn_children, depth guards, memory isolation, context forking. Triggers on: spawn_self, allow_self_spawn, self spawn, max_spawn_depth, max_spawn_children, parallel sub-task, agent clone, dynamic sub-agent."
---

> **Branch:** These skills are written for the `rename/orbiter-to-exo` branch. The Exo APIs referenced here may differ on other branches.

# Exo Spawn — Self-Spawning Agents

## When To Use This Skill

Use this skill when the developer needs to:
- Let an agent dynamically spawn copies of itself at runtime
- Delegate parallel sub-tasks from within an agent's execution
- Configure spawn depth limits and per-call child caps
- Understand memory isolation between parent and spawned children
- Understand context forking for spawned agents

## Decision Guide

1. **Does the agent need to delegate parallel work to itself?** → Sub-agents are ON by default. No change needed. Set `allow_self_spawn=False` to opt out.
2. **How deep should recursive spawning go?** → Set `max_spawn_depth` (default 3; root agent at depth 0)
3. **How many children per single spawn_self call?** → Set `max_spawn_children` (default 4). The LLM receives an error if it passes more tasks than this limit.
4. **Does the child need to share long-term knowledge?** → Automatic: shared long-term memory, fresh short-term
5. **Should children be able to spawn further?** → Default: No. Children get `allow_self_spawn=False`. For multi-level, parent must have `max_spawn_depth > 2`.

## Reference

### Enabling Self-Spawn

```python
from exo import Agent

agent = Agent(
    name="coordinator",
    model="openai:gpt-4o",
    instructions=(
        "You are a research coordinator. When asked about multiple topics, "
        "use spawn_self to delegate each topic as a separate sub-task. "
        "After all spawn_self calls return, synthesize the results."
    ),
    tools=[search, summarize],
    # allow_self_spawn=True by default — spawn_self(tasks) tool auto-registered
    max_spawn_depth=2,           # Root (depth 0) + 1 level of children (depth 1)
    max_spawn_children=4,        # Up to 4 children per spawn_self call (default)
)
```

**Agent parameters:**
- `allow_self_spawn: bool = True` — When `True` (default), registers the `spawn_self(tasks: list[str]) -> str` tool. Set `allow_self_spawn=False` to disable sub-agent spawning.
- `max_spawn_depth: int = 3` — Maximum recursive depth. Default allows root + 2 levels.
- `max_spawn_children: int = 4` — Maximum number of tasks the LLM can pass in a single `spawn_self` call. If exceeded, the tool returns an error string instead of spawning.

### spawn_self(tasks: list[str]) -> str

Auto-registered tool that the LLM calls to spawn child agents in parallel:

**What happens when the LLM calls `spawn_self(tasks)`:**

1. **Empty check:** If `tasks` is empty, returns error string (no exception)
2. **Children cap check:** If `len(tasks) > max_spawn_children`, returns error string (no exception)
3. **Depth check:** If `parent._spawn_depth >= max_spawn_depth`, returns error string (no exception)
4. **Provider check:** Verifies the parent's `_current_provider` is available
5. **Child tools built once:** Parent's tools **minus** `spawn_self` and context tools
6. **Parallel child creation and execution** via `asyncio.TaskGroup` — one child agent per task, all running concurrently:
   - Same `model`, `instructions`, `temperature`, `max_tokens`, `max_steps`
   - Fresh `ShortTermMemory` (conversation isolation)
   - Shared `LongTermMemory` instance (knowledge accumulates across spawns)
   - Forked context via `parent.context.fork(child_name)` (reads inherited, writes isolated)
   - `allow_self_spawn=False` — children cannot spawn further
   - `_spawn_depth = parent._spawn_depth + 1`
7. **Returns:**
   - **Single task** (`len(tasks) == 1`): raw `result.text` string
   - **Multiple tasks**: formatted as `[Task 1]: result\n\n[Task 2]: result\n\n...`
   - **Per-child errors**: if an individual child fails, its slot becomes `[child N error] <message>` (1-indexed)

### Child Agent Properties

| Property | Inherited From Parent? | Details |
|----------|----------------------|---------|
| `model` | Yes | Same model string |
| `instructions` | Yes | Same system prompt |
| `tools` | Yes, minus exclusions | Excludes `spawn_self` and `_is_context_tool` tools |
| `temperature` | Yes | Same value |
| `max_tokens` | Yes | Same value |
| `max_steps` | Yes | Same value |
| `allow_self_spawn` | No | Always `False` on spawned children (prevents recursive spawning unless depth allows) |
| `_spawn_depth` | Incremented | `parent._spawn_depth + 1` |
| Short-term memory | No | Fresh `ShortTermMemory()` instance |
| Long-term memory | Yes | Shared instance (same reference) |
| Context | Forked | `parent.context.fork(child_name)` |

### Child Naming

Children are named: `{parent.name}_spawn_{uuid_hex[:8]}`

Example: `coordinator_spawn_a3f1b2c4`

### Memory Isolation

```
Parent Agent
├── ShortTermMemory: [msg1, msg2, msg3, ...]  ← parent's conversation
├── LongTermMemory: shared_store               ← shared across all
│
└── spawn_self(tasks=["Research topic A", "Research topic B"])
    │
    ├── Child A (tasks[0]: "Research topic A")   ← runs in parallel
    │   ├── ShortTermMemory: []                  ← fresh, empty
    │   └── LongTermMemory: shared_store         ← same reference
    │
    └── Child B (tasks[1]: "Research topic B")   ← runs in parallel
        ├── ShortTermMemory: []                  ← fresh, empty
        └── LongTermMemory: shared_store         ← same reference
```

- **Short-term:** Each child gets a brand new `ShortTermMemory` — no access to parent's or sibling's conversation
- **Long-term:** All share the same `LongTermMemory` instance — writes by child A are visible to child B via `search()`

### Error Handling

All errors return strings (no exceptions raised to the LLM):

```
"[spawn_self error] Empty tasks list. Provide at least one task."
"[spawn_self error] Too many tasks (6). Maximum is 4 per call."
"[spawn_self error] Maximum spawn depth (2) reached. Cannot spawn further sub-agents."
"[spawn_self error] No provider available for spawned agent."
```

Per-child errors (when an individual child raises during execution):

```
"[child 1 error] Connection timeout"
"[child 3 error] Rate limit exceeded"
```

The LLM sees these as tool results and can react accordingly. Per-child errors are embedded in the formatted output — other children's results are still returned.

### Provider Lifecycle

```python
# Inside Agent.run():
self._current_provider = provider  # Set before execution
try:
    return await self._run_inner(...)
finally:
    self._current_provider = None  # Cleaned up after
```

The `spawn_self` tool closure captures `parent` and accesses `parent._current_provider`. This is safe because spawn_self only runs during the parent's `run()` execution.

### Depth Guard

```
max_spawn_depth=3 (default)

Depth 0: Root agent        → can spawn (0 < 3)
Depth 1: Child of root     → can spawn (1 < 3) IF allow_self_spawn=True on child
Depth 2: Grandchild        → can spawn (2 < 3) IF allow_self_spawn=True
Depth 3: Great-grandchild  → blocked (3 >= 3), returns error string
```

**Important:** By default, children get `allow_self_spawn=False`, so only the root can spawn. Children cannot spawn further by default.

## Patterns

### Parallel Research Coordinator

```python
agent = Agent(
    name="researcher",
    model="openai:gpt-4o",
    instructions=(
        "You research topics by delegating to sub-agents. "
        "Pass all sub-topics in a single spawn_self call as a list. "
        "After spawn_self returns, synthesize findings into a report."
    ),
    tools=[web_search, fetch_page],
    allow_self_spawn=True,
    max_spawn_depth=2,
    max_spawn_children=6,   # Allow up to 6 parallel children
    memory=None,    # Disable memory for stateless research
    context=None,   # Disable context for simplicity
)

result = await run(
    agent,
    "Compare AI chip architectures: NVIDIA H100, Google TPU v5, AMD MI300X",
    provider=provider,
)
# LLM calls spawn_self(tasks=["Research NVIDIA H100", "Research Google TPU v5", "Research AMD MI300X"])
# All 3 children run in parallel via asyncio.TaskGroup
# Parent receives "[Task 1]: ...\n\n[Task 2]: ...\n\n[Task 3]: ..." and synthesizes
```

### Data Processing Fan-Out

```python
agent = Agent(
    name="processor",
    instructions=(
        "Process data items by spawning sub-agents. "
        "Use spawn_self(tasks=['Process item: X', 'Process item: Y', ...]) "
        "to handle multiple items in parallel."
    ),
    tools=[validate, transform, store],
    allow_self_spawn=True,
    max_spawn_depth=2,
    max_spawn_children=8,   # Higher cap for batch processing
)
```

### Spawn with Explicit Memory Control

```python
from exo.memory.base import AgentMemory
from exo.memory.short_term import ShortTermMemory
from exo.memory.backends.sqlite import SQLiteMemoryStore

# Shared long-term store for knowledge accumulation (SQLite is the default backend)
shared_lt = SQLiteMemoryStore()

agent = Agent(
    name="coordinator",
    memory=AgentMemory(
        short_term=ShortTermMemory(),
        long_term=shared_lt,
    ),
    max_spawn_children=4,
)
# Children will share shared_lt but get fresh ShortTermMemory
```

## Gotchas

- **Children get `allow_self_spawn=False`** — only the parent can spawn. Children cannot spawn further by default.
- **Children within a single `spawn_self` call run in parallel** via `asyncio.TaskGroup`. All children from one call execute concurrently, not sequentially.
- **`max_spawn_children` caps tasks per call, not total children** — the LLM can make multiple `spawn_self` calls across different tool-call steps, each spawning up to `max_spawn_children` children. The cap applies per invocation.
- **Single vs. multiple return format differs** — a single-task call returns the raw result string; multiple tasks return the `[Task N]: result` formatted output. Design instructions accordingly.
- **Per-child errors don't abort siblings** — if one child fails, its slot gets `[child N error] ...` but other children continue and their results are still returned.
- **Context tools are excluded** from children's tool sets — children get fresh context tool bindings via their own `__init__`
- **`retrieve_artifact` is also excluded** — auto-registered by child's `__init__` if any `large_output=True` tool exists
- **Guide the LLM via instructions** — the LLM decides when to spawn. Be explicit: "pass all countries as a list to spawn_self"
- **Memory=None propagates** — if parent has `memory=None`, children also get `memory=None`
- **Context fork fallback** — if `parent.context.fork()` fails, child shares parent's context (fallback, not isolation)
- **Provider is shared** — parent and all children use the same provider instance. Rate limits and quotas are shared.
