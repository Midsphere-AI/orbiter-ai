# exo-cli

> Command-line agent runner for the Exo multi-agent framework.

`exo-cli` gives you a single `exo` command to run Exo agents from a terminal, start distributed workers, and inspect task queues — no Python script required. It sits at the top of the Exo stack and delegates all agent execution to `exo-core` and `exo-distributed`.

## Installation

```bash
pip install exo-cli
# or
uv add exo-cli
```

## Quick start

```bash
# Run an agent defined in a config file
exo run --config agents.yaml "Summarise today's news"

# Select a model inline (no config file needed for quick tests)
exo run -m openai:gpt-4o "What is 2+2?"

# Stream the response token-by-token
exo run --stream "Write a haiku about distributed systems"

# Start a distributed worker (reads EXO_REDIS_URL env var)
exo start worker --concurrency 4

# Inspect tasks
exo task list --status running
exo task status <task_id>
exo task cancel <task_id>

# View active workers
exo worker list
```

Config files are discovered automatically (`.exo.yaml` or `exo.config.yaml` in the current directory) or supplied with `--config`.

## What's inside

- **`exo run`** — execute an agent or swarm with arbitrary text input; supports `--stream` for SSE output and `--model` for inline model selection
- **`exo start worker`** — launch a Redis-backed distributed worker with configurable concurrency and queue name
- **`exo task list/status/cancel`** — inspect and control tasks across the distributed queue
- **`exo worker list`** — display heartbeat, concurrency, and per-worker task counts for the fleet
- **`exo tool`** — tool offloading sub-commands for advanced operator workflows
- **Config auto-discovery** — resolves `.exo.yaml` or `exo.config.yaml` from the working directory before falling back to `--config`

## Part of [Exo](https://github.com/midsphere-ai/exo)

Get the full framework with `pip install exo-ai`.

---

MIT © Midsphere AI
