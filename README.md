<div align="center">

# Exo

### Smart agents by default, not by configuration.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-3776AB.svg?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-22c55e.svg?style=flat-square)](LICENSE)
[![UV Workspace](https://img.shields.io/badge/uv-workspace-DE5FE9.svg?style=flat-square&logo=uv&logoColor=white)](https://docs.astral.sh/uv/)
[![Pydantic v2](https://img.shields.io/badge/pydantic-v2-E92063.svg?style=flat-square&logo=pydantic&logoColor=white)](https://docs.pydantic.dev/)

[Docs](https://midsphere-ai.github.io/exo/) · [Examples](examples/) · [API Reference](https://midsphere-ai.github.io/exo/reference/)

</div>

---

## Why Exo

Most frameworks hand you a blank agent and a long to-do list: wire up memory, manage context windows, handle token budgets, write summarization logic. You spend days on plumbing before the agent does anything useful.

- **Cognitive architecture out of the box.** Every agent ships with dynamic prompt assembly (neurons), automatic context compression, memory, workspace indexing, and budget awareness — none of it requires configuration.
- **Opt-in, not opt-out.** Don't need memory? Pass `memory=None`. Don't need the context engine? Pass `context=None`. Defaults are rich; stripping back is trivial.
- **Async-first, provider-agnostic.** OpenAI, Anthropic, Gemini, Vertex AI — swap with a string. Core APIs are async with a `run.sync()` escape hatch for scripts.
- **Modular monorepo.** 20 focused packages. Install the full stack with `exo-ai` or pull in only what you need (`exo-core`, `exo-models`, …).
- **Production-ready primitives.** Multi-agent swarms, MCP tool integration, sandboxed code execution, RAG pipelines, A2A communication, evaluation, observability — all first-class packages.

---

## Install

```bash
pip install exo-ai          # full framework
pip install exo-core        # just the agent runtime
```

Requires Python 3.11+.

---

## Quick start

```python
import asyncio
from exo import Agent, run, tool

@tool
def get_weather(city: str) -> str:
    """Return the current weather for a city."""
    return f"Sunny, 24 °C in {city}."

agent = Agent(
    name="assistant",
    model="openai:gpt-4o-mini",
    tools=[get_weather],
    instructions="You are a helpful travel assistant.",
)

async def main():
    result = await run(agent, "What's the weather like in Tokyo?")
    print(result.output)

asyncio.run(main())
```

`run.sync()` is available for synchronous contexts; `run.stream()` yields events as they arrive.

---

## The Exo stack

**Foundation**

| Package | Purpose |
|---|---|
| `exo-core` | Agent, Tool, `@tool`, `run` / `run.sync` / `run.stream`, Swarm, hooks, events |
| `exo-context` | Context engine: neurons, prompt assembly, processors, workspace |
| `exo-models` | LLM providers — OpenAI, Anthropic, Gemini, Vertex AI |
| `exo-observability` | Structured logging, tracing, metrics, cost tracking |

**Capabilities**

| Package | Purpose |
|---|---|
| `exo-memory` | Memory backends: in-memory, SQLite, Postgres, vector search |
| `exo-mcp` | MCP (Model Context Protocol) client and tool integration |
| `exo-sandbox` | Sandboxed execution environments for safe tool running |
| `exo-retrieval` | RAG pipeline — embeddings, vector stores, hybrid search, reranking |
| `exo-search` | AI-powered search with deep research, citations, and multi-turn chat |
| `exo-guardrail` | Pluggable security guardrails — prompt injection and jailbreak detection |
| `exo-harness` | Composable orchestration harness with middleware and session state |
| `exo-skills` | Dynamic capability packages with hot-reload and GitHub skill sources |
| `exo-eval` | Evaluation and scoring framework for agent outputs |
| `exo-a2a` | Agent-to-Agent protocol for inter-agent communication |
| `exo-train` | Training framework: data synthesis, evolution, VeRL integration |
| `exo-distributed` | Distributed execution: Redis task queue, workers, event streaming |

**Apps & tooling**

| Package | Purpose |
|---|---|
| `exo-server` | Lightweight web UI and API server for agents |
| `exo-cli` | CLI agent runner |
| `exo-mcp-cli` | Standalone CLI for interacting with MCP servers |
| `exo` / `exo-ai` | Meta-package: bundles core + capabilities in one install |

---

## Examples

The [`examples/`](examples/) directory contains runnable scripts organised by topic:

- [`examples/quickstart/`](examples/quickstart/) — hello-world and first-tool examples
- [`examples/multi_agent/`](examples/multi_agent/) — Swarm, parallel groups, A2A
- [`examples/advanced/`](examples/advanced/) — context modes, custom neurons, harness
- [`examples/distributed/`](examples/distributed/) — Redis-backed worker pools

---

## Documentation

Full docs at **[midsphere-ai.github.io/exo](https://midsphere-ai.github.io/exo/)** — getting started guide, concept explanations, API reference, and cookbook recipes.

---

## License

MIT © [Midsphere AI](https://github.com/midsphere-ai)
