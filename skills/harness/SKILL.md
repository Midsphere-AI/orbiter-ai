---
name: exo:harness
description: "Use when running multiple Exo agents concurrently or building custom orchestration pipelines — run_parallel, stream_parallel, SubAgentTask, SubAgentResult, SubAgentStatus, SubAgentError, fan-out, parallel dispatch, custom async orchestration. Triggers on: run_parallel, stream_parallel, parallel agents, SubAgentTask, SubAgentResult, fan-out, parallel sub-agent, concurrent agents, orchestration pipeline, harness, Harness, HarnessContext, exo-harness."
---

# Exo Parallel Agent Dispatch

> **Note:** The `exo-harness` package (Harness ABC, HarnessContext, middleware chain, etc.) has been
> dissolved into exo-core as of the distribution-cleanup branch. Custom orchestration is now a plain
> async function calling `run()` / `run.stream()`. The batteries-included parallel dispatch capability
> has been salvaged as `run_parallel` / `stream_parallel` in `exo.parallel`.

## Running multiple agents concurrently

```python
from exo import run_parallel, stream_parallel
from exo.parallel import SubAgentTask, SubAgentResult, SubAgentStatus, SubAgentError

tasks = [
    SubAgentTask(agent=agent_a, input="Summarise topic X"),
    SubAgentTask(agent=agent_b, input="Summarise topic X", timeout=10.0),
]

# Non-streaming: wait for all, return list of SubAgentResult
results = await run_parallel(tasks, continue_on_error=True)
for r in results:
    print(r.agent_name, r.status, r.output)

# Streaming: multiplexed StreamEvent instances in arrival order
async for event in stream_parallel(tasks):
    print(event)
```

### `SubAgentTask`

```python
@dataclass(frozen=True)
class SubAgentTask:
    agent: Any            # Agent, Swarm, or any object with run()/stream()
    input: MessageContent # User query
    name: str | None = None          # Label override; defaults to agent.name
    messages: Sequence[Message] | None = None  # Prior history (None = fresh)
    provider: Any = None  # LLM provider override for this task
    timeout: float | None = None     # Per-task timeout in seconds
```

### `run_parallel(tasks, *, provider=None, continue_on_error=False, max_concurrency=None)`

- Returns `list[SubAgentResult]` in the same order as `tasks`.
- `continue_on_error=False` (default): raises `SubAgentError` on first failure.
  The exception carries `.results` (all results) and `.failed_agents` (names list).
- `continue_on_error=True`: runs all tasks; failed tasks have `status=FAILED` and a non-None `error`.
- `max_concurrency=N`: semaphore limits concurrent tasks to N at a time.
- `provider`: default provider; overridden per-task by `SubAgentTask.provider`.

### `stream_parallel(tasks, *, provider=None, continue_on_error=False, max_concurrency=None, queue_size=256)`

- Async generator yielding `StreamEvent` instances from all tasks interleaved in arrival order.
- Each event carries `agent_name` identifying which agent produced it.
- `StatusEvent(status="starting")` and `StatusEvent(status="completed")` bracket each agent.
- Failed agents produce an `ErrorEvent` in the stream.

### `SubAgentResult`

```python
@dataclass(frozen=True)
class SubAgentResult:
    agent_name: str
    status: SubAgentStatus          # SUCCESS | FAILED | CANCELLED | TIMED_OUT
    output: str = ""                # Accumulated text output (empty on failure)
    result: RunResult | None = None # Full RunResult on success
    error: BaseException | None = None  # Original exception on failure
    elapsed_seconds: float = 0.0
```

## Custom orchestration (no Harness ABC needed)

For sequential pipelines, routing, or any custom orchestration logic, use a plain async function:

```python
from exo import run

async def router(user_input: str, provider) -> str:
    """Classify then route to a specialist agent."""
    classification = await run(classifier, user_input, provider=provider)
    category = classification.output.strip()
    specialist = {"billing": billing_agent, "tech": tech_agent}.get(category, general_agent)
    result = await run(specialist, user_input, provider=provider)
    return result.output

# Or with streaming:
async def streaming_pipeline(user_input: str, provider):
    research = await run(researcher, user_input, provider=provider)
    async for event in run.stream(writer, research.output, provider=provider):
        yield event
```

This is the preferred pattern — it's plain Python, fully debuggable, and composes with
`run_parallel` / `stream_parallel` for fan-out steps.
