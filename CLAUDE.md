# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What is Exo

Exo is a modular multi-agent framework for building LLM-powered applications in Python. It's a UV workspace monorepo with 19 packages. Requires Python 3.11+.

A bare `Agent(...)` is batteries-included by default — memory, context management, checkpoints, message injections, and sub-agent orchestration are all on by default with no extra configuration required.

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

## Architecture

UV workspace monorepo. Packages live in `packages/`. The dependency graph flows upward from `exo-core`:

```
exo-core (foundation, only depends on pydantic)
    ↑
exo-models (OpenAI, Anthropic, Gemini/Vertex AI providers; canonical embeddings/vector layer)
    ↑
exo-context, exo-memory, exo-mcp, exo-sandbox, exo-observability, exo-guardrail
    ↑
exo-retrieval, exo-search, exo-cli, exo-distributed, exo-eval, exo-a2a,
exo-skills, exo-mcp-cli
    ↑
exo-server [experimental], exo-train [experimental]
    ↑
exo (meta-package, re-exports everyday framework; extras for distributed/a2a/search)
```

There are no `orbiter-*` mirror packages present in `packages/` at this time. (They were planned as thin re-export wrappers for a public `orbiter` distribution but have not been created yet — do not assume they exist.)

### Key Packages

- **exo-core** (`packages/exo-core/src/exo/`): `Agent`, `Tool`, `@tool` decorator, `run`/`run.sync`/`run.stream`, `Swarm`, hooks, events, config, registry. The `_internal/` subpackage has the agent runtime internals (see below).
- **exo-models** (`packages/exo-models/`): LLM provider implementations. Provider SDKs are isolated here — core has zero heavy deps. Includes `GoogleProvider` (unified Gemini + Vertex AI), and the canonical `exo.models.embeddings` / `exo.models.vector` layer for embeddings and vector store abstractions.
- **exo-guardrail** (`packages/exo-guardrail/`): Security guardrails — pattern-based and LLM-based prompt injection/jailbreak detection with pluggable backends.
- **exo-retrieval** (`packages/exo-retrieval/`): RAG pipeline — embeddings (OpenAI, Vertex, HTTP), vector stores (pgvector, ChromaDB), hybrid search, reranking, knowledge graph, agentic retrieval.
- **exo-distributed** (`packages/exo-distributed/`): Production-ready distributed execution — Redis Streams task queue, `Worker`, `TaskBroker`, `TaskStore`, event streaming, health monitoring, cancellation, and optional Temporal workflow integration. Use `exo-ai[distributed]` to pull in.
- **exo-search** (`packages/exo-search/`): AI search engine with query classification, parallel research agents, result reranking, citation generation, and 3 quality modes (speed/balanced/quality).
- **exo-skills** (`packages/exo-skills/`): Dynamic capability packages — `SkillRegistry`, skill markdown files with front-matter, hot-reload, GitHub skill sources.
- **exo-mcp-cli** (`packages/exo-mcp-cli/`): Standalone CLI for MCP server interaction — `mcp.json` config, encrypted vault, credential management, server add/remove/test, tool list/call.
- **exo-server** (`packages/exo-server/`): **Experimental.** Minimal FastAPI embed helper for serving Exo agents over HTTP. Not included in the meta-package.
- **exo-train** (`packages/exo-train/`): **Experimental/pre-release.** Training framework — data synthesis, evolution, VeRL integration (heavy GPU deps behind `[verl]` optional extra). Not included in the meta-package.

### Meta-package extras (`exo-ai`)

Install the meta-package with extras for optional capabilities:

```
pip install exo-ai                  # core + models + memory + mcp + sandbox + observability + eval + retrieval
pip install "exo-ai[distributed]"   # + exo-distributed (Redis task queue / workers)
pip install "exo-ai[a2a]"           # + exo-a2a (agent-to-agent protocol, drops httpx for base users)
pip install "exo-ai[search]"        # + exo-search (AI search engine)
pip install "exo-ai[all]"           # everything above combined
```

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
| `agent_group.py` | `ParallelGroup` execution for Swarm workflows |
| `graph.py` | DAG algorithms for Swarm flow resolution |
| `branch_node.py` / `loop_node.py` | Conditional routing and iteration nodes for workflow mode |
| `nested.py` | `SwarmNode`/`RalphNode` — nested orchestration primitives |
| `background.py` | Background task submission, result/error lifecycle |

**Execution flow:** `run()` → `runner.py` → `call_runner()` → `message_builder.build_messages()` → LLM call → `output_parser` → `handlers` (tool dispatch) → loop back to LLM or return result.

## Code Conventions

- **Ruff**: line-length 100, rules `E,F,I,N,W,UP,B,SIM,RUF`, ignore `E501`. Use `datetime.UTC` not `timezone.utc`.
- **Pyright**: basic mode, Python 3.11 target.
- **Async-first**: all core APIs are async. Tests use `asyncio_mode = "auto"` (no `@pytest.mark.asyncio` needed).
- **Pydantic v2**: for all schemas and validation.
- **Test file names must be unique** across all packages (pytest `--import-mode=importlib`).
- **Tests use MockProvider** — never make real API calls. Integration tests live in `tests/integration/` (marked with `@pytest.mark.integration` or `@pytest.mark.marathon`).
- **Model strings**: format `"provider:model"` (e.g., `"openai:gpt-4o-mini"`).
- **FastAPI Depends()**: use `# noqa: B008` for ruff on function defaults.
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

---

## Audit & Ongoing Work

An 83-finding audit report lives at `audit.md`. It covers Bugs & Security, Logical Issues, Non-Completeness, Inconsistencies, and Duplications across all packages.

### Approach: Parallel Sub-agents

For multi-file work across multiple packages, use `Agent` tool with multiple concurrent `general-purpose` sub-agents — one per package. This pattern cuts wall-clock time by ~3x. Prefer **sonnet** sub-agents (`model: sonnet`) for the DX work below.

## DX Simplification Plan ("start" trigger)

**When the user says "start" (with no other context), begin executing this plan.** Goal: make Exo's DX *extremely simple — usable by programming rookies*. Remove overlapping/duplicate features, cut choices, fix how choices are made.

**The plan lives in `namespace-dx-audit.md` at repo root** (two parts, all findings carry `file:line`):
- **Part 1** — constructor param bloat → group into namespaces (`Agent.context`, `Agent.memory`, `Agent.planner`, `Agent.tools`, `Agent.subagents`, `Agent.ptc`, `Agent.guardrails`). Worst offenders: `Agent.__init__` (41 params), `SearchConfig` (~28), `MCPServerConfig`+`ServerEntry` (13/9), `TrajectoryItem` (13), `ServingConfig` (10), `ObservabilityConfig` (9), `Swarm` (9).
- **Part 2** — broader friction: pervasive `Any` typing kills autocomplete; jargon (`ptc`/`Neuron`/`Ralph`/`Rail`/`HITL`/`context_mode="pilot"`); stringly-typed silent failures; leaky top-level exports; choice overload (`run()` vs `agent.run()` return-type split, `store=` vs `memory=`, dead no-op params, diverged `AgentConfig`); hello-world friction + non-teaching errors.

**Execution order — ship tiers sequentially, each independently green (~5,950 tests, `uv run pytest`); keep old spellings as deprecated aliases:**
- **Tier 0 (do first; ~1 session, parallel-friendly, low risk, mostly additive):** kill `Any` on the hot path (`run`/`run.sync`/`run.stream` via typed Protocol; `Agent(store/memory/context/context_mode)`; `agent`/`provider`); re-export first-5-minutes API at top level (`RunResult`, `Usage`, streaming events, `AgentError`, `HookPoint`, message types; memory backends at `exo.memory`); README/docstrings lead with the `run.sync` 3-liner + flip default model to `gpt-4o-mini`; teaching errors (re-raise the swallowed provider `ModelError` in `runner.py:898`; validate `tools`/`model` types; don't retry 401s).
- **Tier 1:** one obvious way — make `agent.run()` private (unify `RunResult`/`AgentOutput`); collapse `store=`/`memory=`; delete no-op `allow_parallel_subagents`/`max_parallel_subagents` + diverged `AgentConfig`; `Literal`/enum the silent strings; drop `FunctionTool` from public exports.
- **Tier 2:** vocabulary renames (aliased): `ptc`→`batch_tools`, `Neuron`→`PromptSection`, `Ralph`→`Refinement*`, `Rail`→`Guard`, `hitl_tools`→`approval_tools`, `context_mode` values→size words; standardize `instructions`/`model`/vector-store names.
- **Tier 3:** Part 1 namespace refactor — `Agent` first (establishes the `*Config` types reused everywhere), then fan out the other 7 config classes (`SearchConfig`, `MCPServerConfig`+`ServerEntry`, `ObservabilityConfig`, `ServingConfig`, `TrajectoryItem`, `Worker`/`RedisConfig`).

**Why multiple sessions:** sub-agents parallelize independent files, but Tier 1/3 core changes serialize on `agent.py`/`runner.py` and each tier gates on a green test run + failure triage in the main context. Tier 0 ≈ this session; Tiers 1–3 ≈ one focused session each.

## graphify

This project has a graphify knowledge graph at graphify-out/.

Rules:
- Before answering architecture or codebase questions, read graphify-out/GRAPH_REPORT.md for god nodes and community structure
- If graphify-out/wiki/index.md exists, navigate it instead of reading raw files
- After modifying code files in this session, run `python3 -c "from graphify.watch import _rebuild_code; from pathlib import Path; _rebuild_code(Path('.'))"` to keep the graph current
