<div align="center">

# exo-ai

### Smart agents by default, not by configuration.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-3776AB.svg?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-22c55e.svg?style=flat-square)](LICENSE)

[Docs](https://midsphere-ai.github.io/exo/) · [GitHub](https://github.com/midsphere-ai/exo) · [API Reference](https://midsphere-ai.github.io/exo/reference/)

</div>

---

`exo-ai` is the meta-package for **Exo** — a modular multi-agent framework for Python. It bundles the full production stack in a single install.

## Install

```bash
pip install exo-ai
```

Requires Python 3.11+.

## What's included

`exo-ai` pulls in:

| Package | Role |
|---|---|
| `exo-core` | Agent, Tool, `@tool`, `run` / `run.sync` / `run.stream`, Swarm |
| `exo-models` | LLM providers — OpenAI, Anthropic, Gemini, Vertex AI |
| `exo-memory` | Memory backends: in-memory, SQLite, Postgres, vector search |
| `exo-mcp` | MCP (Model Context Protocol) client and tool integration |
| `exo-sandbox` | Sandboxed execution environments for safe tool running |
| `exo-observability` | Structured logging, tracing, metrics, cost tracking |
| `exo-eval` | Evaluation and scoring framework for agent outputs |
| `exo-a2a` | Agent-to-Agent protocol for inter-agent communication |

Need only part of the stack? Every package is installable independently — e.g. `pip install exo-core`.

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

`run.sync()` is available for synchronous scripts; `run.stream()` yields events as they arrive.

## More

Full package list, architecture guide, and cookbook recipes:
**[github.com/midsphere-ai/exo](https://github.com/midsphere-ai/exo)**

Full documentation:
**[midsphere-ai.github.io/exo](https://midsphere-ai.github.io/exo/)**

## License

MIT © [Midsphere AI](https://github.com/midsphere-ai)
