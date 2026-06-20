# exo-harness

> Orchestration harness for composable agent runtime behavior.

`exo-harness` provides a structured way to coordinate multiple Exo agents under a single execution context. Subclass `Harness`, implement one async generator method, and use the provided `HarnessContext` to route between agents, share state, and stream multiplexed events — all with normal Python control flow. It sits one layer above individual `Agent` and `Swarm` instances in the Exo stack.

## Installation

```bash
pip install exo-harness
# or
uv add exo-harness
```

## Quick start

```python
from exo import Agent, run
from exo.harness import Harness, HarnessContext, SubAgentTask, TimeoutMiddleware

classifier = Agent(name="classifier", model="openai:gpt-4o-mini",
                   instructions="Output exactly one word: 'code' or 'docs'.")
code_agent = Agent(name="code", model="openai:gpt-4o-mini",
                   instructions="You are a coding assistant.")
docs_agent = Agent(name="docs", model="openai:gpt-4o-mini",
                   instructions="You are a documentation assistant.")


class Router(Harness):
    async def execute(self, ctx: HarnessContext):
        label = await ctx.run_agent(self.agents["classifier"], ctx.input)
        target = self.agents[label.output.strip()]
        async for event in ctx.stream_agent(target, ctx.input):
            yield event


harness = Router(
    name="router",
    agents=[classifier, code_agent, docs_agent],
    middleware=[TimeoutMiddleware(60.0)],
)

result = await run(harness, "How do I reverse a list in Python?")
print(result.output)
```

### Parallel sub-agents

```python
from exo.harness import SubAgentTask

class Parallel(Harness):
    async def execute(self, ctx: HarnessContext):
        tasks = [
            SubAgentTask(agent=self.agents["search"], input=ctx.input),
            SubAgentTask(agent=self.agents["summarize"], input=ctx.input),
        ]
        async for event in ctx.stream_agents_parallel(tasks):
            yield event
```

## What's inside

- **`Harness`** — abstract base class; implement `execute(ctx)` to define orchestration logic
- **`HarnessContext`** — runtime handle with `run_agent()`, `stream_agent()`, `run_agents_parallel()`, `stream_agents_parallel()`, `emit()`, and `checkpoint()`
- **`SubAgentTask`** / **`SubAgentResult`** — typed task spec and result for parallel execution
- **`SessionState`** — dirty-tracked mutable dict persisted across harness steps
- **`Middleware`** / **`TimeoutMiddleware`** / **`CostTrackingMiddleware`** — composable event-stream wrappers
- **`HarnessCheckpoint`** / **`CheckpointAdapter`** — snapshot and restore execution state across process restarts

## Part of [Exo](https://github.com/midsphere-ai/exo)

Get the full framework with `pip install exo-ai`.

---

MIT © Midsphere AI
