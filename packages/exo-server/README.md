# exo-server

> Lightweight FastAPI server for exposing Exo agents over HTTP and SSE.

`exo-server` wraps any Exo `Agent` or `Swarm` in a production-ready HTTP API with a `/chat` endpoint, session management, and Server-Sent Events streaming. It is the minimal server layer between your agent logic and any HTTP client — no frontend included.

## Installation

```bash
pip install exo-server
# or
uv add exo-server
```

## Quick start

```python
from exo import Agent
from exo_server import create_app, register_agent

agent = Agent(name="assistant", model="openai:gpt-4o-mini")

app = create_app()
register_agent(app, agent, default=True)

# Run with: uvicorn myapp:app --port 8000
```

```bash
# Non-streaming chat
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello!", "stream": false}'

# SSE streaming
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Tell me a story", "stream": true}'
```

## What's inside

- **`create_app()`** — factory that returns a configured `FastAPI` instance with all routes mounted
- **`register_agent(app, agent)`** — register one or more agents; set `default=True` for the fallback when no `agent_name` is specified
- **`serve(host, port)`** — convenience wrapper that calls `uvicorn.run()` directly
- **`/chat` endpoint** — synchronous JSON response or SSE stream depending on the `stream` flag
- **`/inject` endpoint** — inject a message into a running agent's context before its next LLM call
- **Session routes** — `CreateSessionRequest`, `AppendMessageRequest`, `Session` for stateful multi-turn conversations

## Part of [Exo](https://github.com/midsphere-ai/exo)

Get the full framework with `pip install exo-ai`.

---

MIT © Midsphere AI
