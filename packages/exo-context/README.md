# exo-context

> Context engine for Exo agents: hierarchical state, composable prompt building, event-driven processors, and versioned workspace storage.

exo-context manages everything that surrounds an LLM call. It provides a `Context` object with fork/merge semantics for nested tasks, a `Neuron`-based prompt assembly pipeline, event-driven `ContextProcessor` hooks that fire before and after LLM calls, and a `Workspace` for persisting artifacts to disk. It sits directly above exo-core in the dependency graph; exo-memory, exo-retrieval, and the agent harness all build on it.

## Installation

```bash
pip install exo-context
# or
uv add exo-context
```

## Quick start

```python
import asyncio
from exo import Agent, run
from exo.context import Context, ContextConfig, OverflowStrategy

config = ContextConfig(
    limit=30,
    overflow=OverflowStrategy.SUMMARIZE,
    keep_recent=5,
    cache=True,
)
ctx = Context(task_id="session-001", config=config)

agent = Agent(
    name="assistant",
    model="openai:gpt-4o-mini",
    instructions="You are a helpful assistant.",
    context=ctx,
)

async def main() -> None:
    result = await run(agent, "Summarize the last three messages.")
    print(result.output)

asyncio.run(main())
```

Context can also be set via shorthand on `Agent` directly: `Agent(name="bot", context_limit=30, overflow="summarize")`.

## What's inside

- **`Context`** — per-task context with hierarchical fork/merge, token tracking, and checkpoint support
- **`ContextConfig` / `OverflowStrategy`** — declarative configuration for context limits and overflow handling (`summarize`, `truncate`, `none`)
- **`Neuron`** — ABC for composable prompt fragments; nine built-in neurons cover system info, task state, history, todos, knowledge, workspace summaries, skills, facts, and entities
- **`ProcessorPipeline`** — ordered chain of `ContextProcessor` instances that fire on `pre_llm_call` and `post_tool_call` events; built-ins include `SummarizeProcessor`, `RoundWindowProcessor`, `DialogueCompressor`, and `ToolResultOffloader`
- **`Workspace`** — filesystem-backed artifact store with `ArtifactType` classification, full version history, and observer callbacks
- **`PromptBuilder`** — assembles the final prompt from prioritized neurons and context state

## Part of [Exo](https://github.com/midsphere-ai/exo)

Context layer of the Exo stack — sits above exo-core and beneath exo-memory, exo-retrieval, and the harness. Get the full framework with `pip install exo-ai`.

---

MIT © Midsphere AI
