# exo-a2a

> Agent-to-Agent protocol for inter-agent communication.

`exo-a2a` lets Exo agents talk to each other over HTTP using a standard discovery and task protocol. Serve any Exo agent as a self-describing A2A endpoint, then call it from another process — or another machine — using `RemoteAgent`, which presents the same `run()` interface as a local agent. It sits at the inter-process boundary of the Exo stack, enabling multi-process and cross-service agent graphs.

## Installation

```bash
pip install exo-a2a
# or
uv add exo-a2a
```

## Quick start

### Serve an agent

```python
import uvicorn
from exo import Agent
from exo.a2a import A2AServer, AgentExecutor, ServingConfig

agent = Agent(name="summarizer", model="openai:gpt-4o-mini",
              instructions="Summarize the user's input in one sentence.")

server = A2AServer(
    AgentExecutor(agent),
    ServingConfig(host="0.0.0.0", port=8080),
)
app = server.build_app()

uvicorn.run(app, host="0.0.0.0", port=8080)
```

### Call it remotely

```python
from exo.a2a import RemoteAgent

remote = RemoteAgent(
    name="summarizer",
    agent_card="http://localhost:8080/.well-known/agent-card",
)

result = await remote.run("Exo is a modular multi-agent framework for Python.")
print(result.text)
```

## What's inside

- **`A2AServer`** — FastAPI app that exposes `POST /` (task execution), `GET /.well-known/agent-card` (discovery), and an optional `POST /stream` endpoint
- **`AgentExecutor`** — wraps any Exo `Agent` for A2A task execution
- **`RemoteAgent`** — agent-compatible wrapper that calls a remote A2A server; drop-in replacement for local agents in swarms or harnesses
- **`A2AClient`** — low-level async HTTP client with `send_task()` and `send_task_streaming()`
- **`AgentCard`** / **`AgentCapabilities`** / **`AgentSkill`** — Pydantic models for agent discovery metadata
- **`TaskState`** / **`TaskStatus`** / **`TaskStatusUpdateEvent`** / **`TaskArtifactUpdateEvent`** — task lifecycle types for status tracking and streaming

## Part of [Exo](https://github.com/midsphere-ai/exo)

Get the full framework with `pip install exo-ai`.

---

MIT © Midsphere AI
