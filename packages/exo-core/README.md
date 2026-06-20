# exo-core

> The core agent framework powering the Exo stack: Agent, Tool, Runner, Swarm, and the primitives everything else builds on.

exo-core defines `Agent` — the autonomous unit that loops over LLM calls and tool dispatches — together with the `@tool` decorator, the async `run()` entry point, and multi-agent `Swarm` orchestration. It is the single dependency that every other Exo package inherits. exo-core itself depends only on Pydantic, keeping it lightweight enough to embed anywhere.

## Installation

```bash
pip install exo-core
# or
uv add exo-core
```

## Quick start

```python
import asyncio
from exo import Agent, run, tool

@tool
async def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

agent = Agent(
    name="calc",
    model="openai:gpt-4o-mini",
    instructions="You are a calculator assistant.",
    tools=[add],
)

async def main() -> None:
    result = await run(agent, "What is 17 + 25?")
    print(result.output)

asyncio.run(main())
```

Use `run.sync()` for a blocking call or `run.stream()` for an async generator of `StreamEvent` objects.

## What's inside

- **`Agent`** — the core autonomous unit: model, instructions, tools, handoffs, hooks, planning, memory, and spawn behaviour in a single class
- **`run`** — async entry point with `run.sync()` (blocking) and `run.stream()` (streaming) variants; handles retries and loop detection
- **`tool` / `Tool` / `FunctionTool`** — decorator and ABCs for defining LLM-callable tools with auto-generated JSON schemas
- **`Swarm`** — multi-agent orchestration with `ParallelGroup` and `SerialGroup` execution primitives and a flow DSL (`"a >> b >> c"`)
- **`ToolContext`** — per-call context passed to every tool, carrying agent state and injected arguments
- **`TokenCounter` / `count_tokens`** — provider-aware token counting without making LLM calls

## Part of [Exo](https://github.com/midsphere-ai/exo)

Foundation of the Exo stack — everything else builds on this. Get the full framework with `pip install exo-ai`.

---

MIT © Midsphere AI
