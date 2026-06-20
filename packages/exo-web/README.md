# exo-web

> Exo Web — full-stack AI agent platform with a visual canvas UI and REST/WebSocket API.

`exo-web` is the complete platform interface for Exo. It ships an Astro 5 frontend with a ReactFlow agent-graph canvas alongside a FastAPI backend that handles agent execution, session persistence, scheduling, and sandboxed code running. It is the highest-level package in the Exo stack and depends on `exo-core` and `exo-models`.

## Installation

```bash
pip install exo-web
# or
uv add exo-web
```

## Quick start

```bash
# Install frontend dependencies (first time only)
cd packages/exo-web
npm install

# Start Astro dev server + FastAPI backend concurrently
npm run dev

# Backend only (port 8000)
uv run uvicorn exo_web.app:app --reload --host 127.0.0.1 --port 8000

# Production frontend build
npm run build
```

Key environment variables: `EXO_DATABASE_URL`, `EXO_SECRET_KEY`, `EXO_DEBUG`.

## What's inside

- **Visual canvas** — ReactFlow-based agent-graph editor (`src/islands/Canvas/`) for building and wiring agent workflows without writing code
- **FastAPI backend** (`src/exo_web/app.py`) — 50+ API routes under `/api/v1/` covering agents, sessions, deployments, artifacts, benchmarks, and webhooks
- **Workflow engine** (`engine.py`) — topological-sort executor with per-node retry and parallel branch support
- **Services layer** (`services/`) — agent runtime, sandbox execution, scheduler, and memory backends
- **Astro pages** (`src/pages/`) — server-rendered pages for agents, applications, deployments, artifacts, docs, and benchmarks
- **SQLite persistence** — async `aiosqlite` with WAL mode; migrations run automatically at startup

## Part of [Exo](https://github.com/midsphere-ai/exo)

Get the full framework with `pip install exo-ai`.

---

MIT © Midsphere AI
