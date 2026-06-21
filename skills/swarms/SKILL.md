---
name: exo:swarms
description: "Use when building multi-agent systems with Exo — Swarm orchestration, flow DSL, workflow/handoff/team modes, ParallelGroup, BranchNode, LoopNode, SwarmNode, RalphNode, agent delegation. Triggers on: swarm, multi-agent, workflow, handoff, team mode, flow DSL, ParallelGroup, SwarmNode, RalphNode, agent pipeline, delegation, nested swarm."
---

> **Branch:** These skills are written for the `rename/orbiter-to-exo` branch. The Exo APIs referenced here may differ on other branches.

# Exo Swarms — Multi-Agent Orchestration

## When To Use This Skill

Use this skill when the developer needs to:
- Orchestrate multiple agents in a pipeline, delegation chain, or team
- Use the flow DSL to define execution order
- Choose between workflow, handoff, and team modes
- Use ParallelGroup for concurrent agent execution within a workflow
- Add conditional branching (BranchNode) or iteration (LoopNode)
- Nest swarms inside swarms via `SwarmNode` (with streaming support)
- Plug a `RalphRunner` into a swarm via `RalphNode`
- Stream events from multi-agent execution

## Decision Guide

1. **Fixed pipeline where output chains to next agent?** → `mode="workflow"` with flow DSL `"a >> b >> c"`
2. **Agents decide dynamically who runs next?** → `mode="handoff"` — agents have `handoffs=[...]` and delegate via tool calls
3. **One lead agent coordinating specialist workers?** → `mode="team"` — first agent = lead, gets auto-generated `delegate_to_{name}` tools for each worker
4. **Need agents to run in parallel within a workflow?** → `ParallelGroup(agents=[a, b])` as a node (for sequential chaining within a group, use `Swarm(mode="workflow")` directly)
5. **Need conditional routing?** → `BranchNode` evaluates state and jumps to a target agent
6. **Need iteration?** → `LoopNode` repeats body agents until condition or max_iterations
7. **Need a swarm as a node in another swarm?** → `SwarmNode(swarm=inner)` — supports `run()` and `stream()` with context isolation
8. **Need a Ralph iterative loop as a node?** → `RalphNode(runner=ralph_runner)` — streams `RalphIterationEvent` + inner events

## Reference

### Swarm Constructor

```python
from exo import Swarm, Agent

swarm = Swarm(
    agents=[agent_a, agent_b, agent_c],   # Required: list of Agent instances
    flow="a >> b >> c",                     # Optional: flow DSL (defaults to list order)
    mode="workflow",                        # "workflow" | "handoff" | "team"
    max_handoffs=10,                        # Handoff mode only: safety cap
    context_mode="copilot",                 # Optional: propagate context mode to all agents
)
```

**Parameters:**
- `agents: list[Agent]` — Must have at least one agent. Names must be unique.
- `flow: str | None` — Flow DSL string. If omitted, agents run in list order.
- `mode: str` — Execution mode (default `"workflow"`).
- `max_handoffs: int` — Maximum handoff transitions (default 10, handoff mode only).
- `context_mode: Any` — When set, creates a Context from this mode and assigns it to all member agents.

### Mode Comparison

| Mode | Execution Pattern | When Output Chains | Agent Decides Next? |
|------|------------------|-------------------|-------------------|
| `workflow` | Sequential pipeline per flow DSL | Yes — each agent's output becomes next agent's input | No — fixed order |
| `handoff` | Dynamic delegation | Yes — output forwarded to handoff target | Yes — agent triggers handoff via tool call |
| `team` | Lead + workers | No — lead gets worker results as tool returns | Yes — lead calls `delegate_to_{worker}(task)` |

### Workflow Mode

Agents execute in topological order. Each agent's output becomes the next agent's input.

```python
researcher = Agent(name="researcher", instructions="Research the topic deeply.")
writer = Agent(name="writer", instructions="Write a report from the research.")
editor = Agent(name="editor", instructions="Edit the report for clarity.")

swarm = Swarm(
    agents=[researcher, writer, editor],
    flow="researcher >> writer >> editor",
    mode="workflow",
)

result = await swarm.run("AI safety trends in 2026", provider=provider)
# researcher runs first → output fed to writer → output fed to editor
# result is editor's output
```

### Handoff Mode

The first agent in the flow runs. It can delegate to other agents via handoff targets.

```python
triage = Agent(
    name="triage",
    instructions="Route the query to the right specialist.",
    handoffs=[billing, technical, general],
)
billing = Agent(name="billing", instructions="Handle billing questions.")
technical = Agent(name="technical", instructions="Handle technical questions.")
general = Agent(name="general", instructions="Handle general questions.")

swarm = Swarm(
    agents=[triage, billing, technical, general],
    mode="handoff",
    max_handoffs=5,
)

result = await swarm.run("My invoice is wrong", provider=provider)
# triage runs → decides to handoff to billing → billing handles it
```

**Handoff chain:** Agent A can hand off to Agent B, who can hand off to Agent C, etc. — up to `max_handoffs`.

### Team Mode

First agent is the lead. All other agents are workers. The lead automatically gets `delegate_to_{worker_name}(task: str)` tools.

```python
lead = Agent(
    name="lead",
    instructions="You coordinate a team. Delegate research to researcher and coding to coder.",
)
researcher = Agent(name="researcher", instructions="Deep research.", tools=[search])
coder = Agent(name="coder", instructions="Write Python code.", tools=[run_code])

swarm = Swarm(
    agents=[lead, researcher, coder],
    mode="team",
)

result = await swarm.run("Build a sentiment analysis tool", provider=provider)
# lead gets delegate_to_researcher(task) and delegate_to_coder(task) tools
# lead decides when and what to delegate
```

**Delegate tool behavior:** Each `delegate_to_{name}(task)` tool runs the target worker agent with the given task and returns its text output to the lead.

### Flow DSL

```python
# Sequential pipeline
flow = "a >> b >> c"

# Agents run in list order when flow is omitted
swarm = Swarm(agents=[a, b, c])  # same as "a >> b >> c"
```

**Validation:**
- All node names in the flow must match agent names
- Cyclic flows are rejected (topological sort)
- Unknown agent names raise `SwarmError`

### ParallelGroup

Groups act as composite nodes in a workflow:

```python
from exo import ParallelGroup, Agent, Swarm

analyst_a = Agent(name="analyst_a", instructions="Analyze market data.")
analyst_b = Agent(name="analyst_b", instructions="Analyze competitor data.")
synthesizer = Agent(name="synthesizer", instructions="Combine analyses.")

# Parallel: both analysts run concurrently
parallel = ParallelGroup(agents=[analyst_a, analyst_b])

# Use in a workflow
swarm = Swarm(
    agents=[parallel, synthesizer],
    mode="workflow",
)

result = await swarm.run("Analyze the AI market", provider=provider)
```

**ParallelGroup:** Runs all agents concurrently, combines outputs with a separator (default `"\n\n"`) or a custom `aggregate_fn`.

Has `is_group = True` and its own `run()` / `stream()` methods.

### Swarm.run()

```python
result = await swarm.run(
    "User query",
    messages=None,              # Optional prior conversation history
    provider=provider,          # LLM provider for all agents
    max_retries=3,              # Retry attempts for transient errors
    checkpoint=True,            # Enable workflow checkpointing (or pass WorkflowCheckpointStore)
)
# Returns RunResult with .output (final agent's text) and .steps
```

### Swarm.stream()

```python
async for event in swarm.stream(
    "User query",
    provider=provider,
    detailed=True,              # Emit StepEvent, UsageEvent, ToolResultEvent, StatusEvent
    max_steps=20,               # Max LLM-tool round-trips per agent
    event_types={"text", "status"},  # Filter event types
):
    if isinstance(event, TextEvent):
        print(event.text, end="")
    elif isinstance(event, StatusEvent):
        print(f"\n[{event.agent_name}] {event.message}")
```

**StatusEvent in swarms:**
- `"Workflow executing agent 'writer'"` — workflow mode agent transitions
- `"Handoff from 'triage' to 'billing'"` — handoff mode transitions
- `"Team lead 'lead' starting execution"` — team mode start
- `"Branch 'router' routing to 'specialist'"` — branch decisions

All events include `agent_name` showing which agent produced them.

### Workflow Checkpointing

Save state before each node for resumable workflows:

```python
from exo import WorkflowCheckpointStore

store = WorkflowCheckpointStore()
result = await swarm.run("Query", provider=provider, checkpoint=store)

# Inspect checkpoints
for cp in store.list():
    print(f"Node: {cp.node_name}, completed: {cp.completed_nodes}")
```

Pass `checkpoint=True` for an internal store (useful for testing), or pass your own `WorkflowCheckpointStore`.

## Patterns

### Research-Write-Review Pipeline

```python
researcher = Agent(
    name="researcher",
    instructions="Find relevant papers and data on the topic.",
    tools=[search, fetch_paper],
)
writer = Agent(
    name="writer",
    instructions="Write a comprehensive report from the research.",
)
reviewer = Agent(
    name="reviewer",
    instructions="Review the report. Output only the final polished version.",
)

pipeline = Swarm(
    agents=[researcher, writer, reviewer],
    flow="researcher >> writer >> reviewer",
)

result = await pipeline.run("Quantum computing advances in 2026", provider=provider)
```

### Customer Support Triage

```python
router = Agent(
    name="router",
    instructions="Classify the query and hand off to the right department.",
    handoffs=[billing, technical, account],
)

swarm = Swarm(
    agents=[router, billing, technical, account],
    mode="handoff",
    max_handoffs=3,
)
```

### Parallel Analysis with Synthesis

```python
market = Agent(name="market", instructions="Analyze market trends.")
tech = Agent(name="tech", instructions="Analyze technology landscape.")
synth = Agent(name="synth", instructions="Synthesize both analyses into recommendations.")

swarm = Swarm(
    agents=[ParallelGroup(agents=[market, tech]), synth],
    mode="workflow",
)
```

### SwarmNode (Nested Swarm)

Wraps a Swarm so it can be used as a node in another Swarm. Supports both `run()` and `stream()` with context isolation.

```python
from exo import Swarm, SwarmNode, Agent

# Inner pipeline
inner = Swarm(agents=[a, b], flow="a >> b")
node = SwarmNode(swarm=inner, name="inner_pipeline")

# Outer pipeline uses inner as a node
outer = Swarm(agents=[c, node, d], flow="c >> inner_pipeline >> d")

# Streaming works — inner events flow through
async for event in outer.stream("Query", provider=provider):
    print(f"[{event.agent_name}] {event.type}")
```

**Key properties:**
- `is_swarm = True` — triggers Swarm's duck-typing detection
- `stream()` delegates to inner swarm's `stream()` with context isolation
- `run()` delegates to inner swarm's `run()` — no message history forwarding
- Inner swarm creates its own `RunState` — no shared mutable state

### RalphNode (Ralph Loop in a Swarm)

Wraps a `RalphRunner` so it can be placed in a Swarm's agent list. The `is_group = True` marker triggers the Swarm's stream delegation path.

```python
from exo import RalphNode, Swarm, Agent
from exo.eval.ralph import RalphRunner

researcher = Agent(name="researcher", instructions="Research deeply.")
ralph = RalphNode(
    runner=RalphRunner.from_agent(researcher, scorers=[quality_scorer]),
    name="research_loop",
)
summarizer = Agent(name="summarizer", instructions="Summarize findings.")

swarm = Swarm(
    agents=[ralph, summarizer],
    flow="research_loop >> summarizer",
)

# Stream yields RalphIterationEvent, inner agent events, RalphStopEvent,
# then summarizer's events
async for event in swarm.stream("Analyze market trends", provider=provider):
    print(event)
```

**Key properties:**
- `is_group = True` — triggers Swarm's `getattr(agent, "is_group", False)` check
- `stream()` delegates to `RalphRunner.stream()` with the node's name
- `run()` delegates to `RalphRunner.run()`, wraps result in `RunResult`
- Provider and messages are ignored — Ralph manages its own execution context

## Gotchas

- **Agent names must be unique** within a Swarm — duplicates raise `SwarmError`
- **Flow DSL node names must exactly match agent names** — `"a >> b"` requires agents named `"a"` and `"b"`
- **Cyclic flows are rejected** — the topological sort detects cycles and raises `SwarmError`
- **Team mode requires at least 2 agents** (lead + at least one worker)
- **Delegate tools are temporary** — in team mode, they're added to the lead before execution and removed after
- **`max_handoffs=10` default** prevents infinite delegation loops — increase if needed for deep handoff chains
- **Workflow output chaining** means each agent receives the previous agent's text output as its input — not the original query
- **context_mode on Swarm** propagates to ALL member agents, overriding their individual context settings
