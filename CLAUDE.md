# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What is Exo

Exo is a modular multi-agent framework for building LLM-powered applications in Python. It's a UV workspace monorepo with 21 packages. Requires Python 3.11+.

## Common Commands

```bash
# Install all workspace packages (editable mode)
uv sync

# Run all tests (~2,900 tests, asyncio_mode=auto)
uv run pytest

# Run tests for a single package
uv run pytest packages/exo-core/tests/

# Run a single test file
uv run pytest packages/exo-core/tests/test_agent.py

# Run a single test
uv run pytest packages/exo-core/tests/test_agent.py::test_function_name

# Lint (with auto-fix)
uv run ruff check packages/ --fix

# Format check
uv run ruff format --check packages/

# Type-check a package
uv run pyright packages/exo-core/

# Verify installation
uv run python -c "from exo import Agent, run, tool; print('OK')"
```

### exo-web (dual Node+Python package)

```bash
cd packages/exo-web

# Install frontend deps
npm install

# Dev server (runs Astro + FastAPI concurrently)
npm run dev

# Astro typecheck
npx astro check

# Run backend only
uv run uvicorn exo_web.app:app --reload
```

## Architecture

UV workspace monorepo. Packages live in `packages/`. The dependency graph flows upward from `exo-core`:

```
exo-core (foundation, only depends on pydantic)
    ↑
exo-models (OpenAI, Anthropic, Gemini, Vertex AI providers)
    ↑
exo-context, exo-memory, exo-mcp, exo-sandbox, exo-observability, exo-guardrail
    ↑
exo-retrieval, exo-search, exo-cli, exo-server, exo-eval, exo-a2a, exo-train, exo-web,
exo-harness, exo-skills, exo-mcp-cli
    ↑
exo (meta-package, re-exports everything)
```

There are no `orbiter-*` mirror packages present in `packages/` at this time. (They were planned as thin re-export wrappers for a public `orbiter` distribution but have not been created yet — do not assume they exist.)

### Key Packages

- **exo-core** (`packages/exo-core/src/exo/`): `Agent`, `Tool`, `@tool` decorator, `run`/`run.sync`/`run.stream`, `Swarm`, hooks, events, config, registry. The `_internal/` subpackage has the agent runtime internals (see below).
- **exo-models** (`packages/exo-models/`): LLM provider implementations. Provider SDKs are isolated here — core has zero heavy deps.
- **exo-guardrail** (`packages/exo-guardrail/`): Security guardrails — pattern-based and LLM-based prompt injection/jailbreak detection with pluggable backends.
- **exo-retrieval** (`packages/exo-retrieval/`): RAG pipeline — embeddings (OpenAI, Vertex, HTTP), vector stores (pgvector, ChromaDB), hybrid search, reranking, knowledge graph, agentic retrieval.
- **exo-search** (`packages/exo-search/`): AI search engine with query classification, parallel research agents, result reranking, citation generation, and 3 quality modes (speed/balanced/quality).
- **exo-harness** (`packages/exo-harness/`): Composable orchestration harness — `Harness` ABC, `HarnessContext`, middleware (timeout, cost tracking), `SessionState` for multi-step agent workflows. Supports parallel sub-agents via `run_agents_parallel()`/`stream_agents_parallel()` with event multiplexing, per-agent log files (`/tmp/`), and `AssistantMessage` output injection.
- **exo-skills** (`packages/exo-skills/`): Dynamic capability packages — `SkillRegistry`, skill markdown files with front-matter, hot-reload, GitHub skill sources.
- **exo-mcp-cli** (`packages/exo-mcp-cli/`): Standalone CLI for MCP server interaction — `mcp.json` config, encrypted vault, credential management, server add/remove/test, tool list/call.
- **exo-web** (`packages/exo-web/`): Full platform UI. Hybrid package — Astro 5.x frontend (`src/pages/`, `src/islands/`) + FastAPI backend (`src/exo_web/`). Has its own `package.json` AND `pyproject.toml`.

### exo-core `_internal/` — Agent Runtime Internals

The `_internal/` subpackage is the engine room. Understanding the call chain is critical for working on agent execution:

| Module | Role |
|---|---|
| `call_runner.py` | Entry point from `runner.py` — state tracking, loop detection |
| `message_builder.py` | Assembles the LLM message list from agent state, neurons, history |
| `handlers.py` | Tool call dispatch, parallel execution with `except*` ExceptionGroup handling |
| `output_parser.py` | Parses LLM responses into tool calls and text output |
| `state.py` | `RunNode`/`RunState` state machine — RUNNING/SUCCESS/FAILED/TIMEOUT transitions |
| `planner.py` | Planning pre-pass (isolated context, plan injection) |
| `agent_group.py` | `ParallelGroup`/`SerialGroup` execution for Swarm workflows |
| `graph.py` | DAG algorithms for Swarm flow resolution |
| `branch_node.py` / `loop_node.py` | Conditional routing and iteration nodes for workflow mode |
| `nested.py` | `SwarmNode`/`RalphNode` — nested orchestration primitives |
| `background.py` | Background task submission, result/error lifecycle |
| `task_controller/` | Sub-package: event bus, intent recognizer, manager, scheduler, task loop queue |

**Execution flow:** `run()` → `runner.py` → `call_runner()` → `message_builder.build_messages()` → LLM call → `output_parser` → `handlers` (tool dispatch) → loop back to LLM or return result.

### exo-web Backend Structure

- `app.py` — FastAPI app entry point, middleware, route registration
- `config.py` — Settings dataclass (env vars: `EXO_DATABASE_URL`, `EXO_SECRET_KEY`, `EXO_DEBUG`)
- `database.py` — `get_db()` async context manager, WAL mode, foreign keys
- `engine.py` — Workflow execution engine (topological sort, node execution, retry)
- `migrations/` — Sequential SQL files, run automatically on startup via lifespan
- `routes/` — 50+ APIRouter modules, all under `/api/v1/` prefix
- `services/` — Business logic layer (agent runtime, sandbox, scheduler, memory)
- `middleware/` — CSRF, rate limiting, security headers, API version redirect

### exo-web Frontend Structure

- Astro 5.x pages in `src/pages/`, layouts in `src/layouts/`
- React islands in `src/islands/` (e.g., ReactFlow canvas)
- Tailwind CSS v4 via `@tailwindcss/vite`
- `cn()` utility at `src/utils/merge.ts` for class merging

## Code Conventions

- **Ruff**: line-length 100, rules `E,F,I,N,W,UP,B,SIM,RUF`, ignore `E501`. Use `datetime.UTC` not `timezone.utc`.
- **Pyright**: basic mode, Python 3.11 target.
- **Async-first**: all core APIs are async. Tests use `asyncio_mode = "auto"` (no `@pytest.mark.asyncio` needed).
- **Pydantic v2**: for all schemas and validation.
- **Test file names must be unique** across all packages (pytest `--import-mode=importlib`).
- **Tests use MockProvider** — never make real API calls. Integration tests live in `tests/integration/` (marked with `@pytest.mark.integration` or `@pytest.mark.marathon`).
- **Model strings**: format `"provider:model"` (e.g., `"openai:gpt-4o-mini"`).
- **FastAPI Depends()**: use `# noqa: B008` for ruff on function defaults.
- **CSRF**: auto-injected via fetch monkey-patch in PageLayout — no manual header needed in frontend.
- **API routes**: define static paths (`/search`) before param routes (`/{id}`) to prevent FastAPI mismatching.

### Logging conventions (two patterns, do NOT mix)

- **exo-core internal files** (`_internal/`): `from exo.observability.logging import get_logger  # pyright: ignore[reportMissingImports]` → `_log = get_logger(__name__)`
- **All other packages** (exo-models, exo-mcp, etc.): `import logging` → `logger = logging.getLogger(__name__)`

## Adding a New Package to the Workspace

1. Create `packages/<name>/` with `pyproject.toml` and `src/` layout
2. Update root `pyproject.toml`: add to `[tool.uv.workspace].members`, `[dependency-groups].dev`, and `[tool.uv.sources]`
3. Run `uv sync`

## Important File Locations

- Root config: `pyproject.toml` (workspace definition, ruff, pyright, pytest config)
- Public API exports: `packages/exo-core/src/exo/__init__.py`
- Provider resolution: `packages/exo-models/`
- Web app entry: `packages/exo-web/src/exo_web/app.py`
- DB migrations: `packages/exo-web/src/exo_web/migrations/`
- Handle types (keep in sync): `packages/exo-web/src/islands/Canvas/handleTypes.ts` ↔ `routes/tools.py` (`_NODE_HANDLE_MAP`)

---

## Audit & Ongoing Work

An 83-finding audit report lives at `audit.md`. It covers Bugs & Security, Logical Issues, Non-Completeness, Inconsistencies, and Duplications across all packages.

### Audit Fix Priority (exo-web)

| ID | File | Issue |
|---|---|---|
| **B-1** | `services/sandbox.py:87-88` | Sandbox escape: `_build_runner_script` uses `.replace()` for escaping — misses `\r`, `\t`, `\0`, Unicode. Fix: use `repr()` or `json.dumps()`. |
| **B-2** | `config.py:14` | Hardcoded default secret key with no production guard. Fix: startup assertion in lifespan. |
| **B-4** | `routes/webhooks.py:146-236` | Webhook trigger has no auth — `url_token` stored but never validated. Fix: compare against DB. |
| **B-8** | — | Token leaked in logs. |
| **L-2** | — | `stream_agent` bypasses tool loop. |
| **B-15** | — | Async callable instructions never awaited. |

### Approach: Parallel Sub-agents

For multi-file work across multiple packages, use `Agent` tool with multiple concurrent `general-purpose` sub-agents — one per package. This pattern cuts wall-clock time by ~3x.

## graphify

This project has a graphify knowledge graph at graphify-out/.

Rules:
- Before answering architecture or codebase questions, read graphify-out/GRAPH_REPORT.md for god nodes and community structure
- If graphify-out/wiki/index.md exists, navigate it instead of reading raw files
- After modifying code files in this session, run `python3 -c "from graphify.watch import _rebuild_code; from pathlib import Path; _rebuild_code(Path('.'))"` to keep the graph current
