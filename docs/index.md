# Exo

**A modern, modular multi-agent framework for building LLM-powered applications in Python.**

Exo is a modern multi-agent framework designed around composability, type safety, and a clean async-first API.

## Install

```bash
pip install exo            # everything
pip install exo-core       # minimal: agent, tools, runner, swarm
```

Requires Python 3.11+.

## Quick Start

```python
from exo import Agent, run, tool


@tool
async def get_weather(city: str) -> str:
    """Return the current weather for a city."""
    return f"Sunny, 22°C in {city}."


agent = Agent(
    name="weather-bot",
    model="openai:gpt-4o-mini",
    instructions="You are a helpful weather assistant.",
    tools=[get_weather],
)

result = run.sync(agent, "What's the weather in Tokyo?")
print(result.output)
```

## Feature Highlights

**Composable Agents** -- Build agents from simple building blocks: a model string, instructions, tools, and hooks. Agents are plain Python objects with no hidden state machines.

**Type-Safe Tools** -- The `@tool` decorator auto-generates JSON schemas from function signatures and docstrings. Full type checking from your IDE to the LLM and back.

**Three Execution Modes** -- `run()` (async), `run.sync()` (blocking), `run.stream()` (real-time streaming with `TextEvent` and `ToolCallEvent`).

**Multi-Agent Orchestration** -- Swarms support three modes: `workflow` (sequential pipeline), `handoff` (agent-driven delegation), and `team` (lead-worker with auto-generated delegate tools). Compose with a flow DSL: `"researcher >> writer >> reviewer"`.

**Context Engine** -- Hierarchical state management, composable prompt building via neurons, event-driven processor pipelines, workspace for artifact storage with versioning.

**Memory System** -- Typed short/long-term memory with SQLite, Postgres, and vector backends. LLM-based extraction and summarization.

**Lifecycle Hooks** -- Intercept every stage: `PRE_LLM_CALL`, `POST_LLM_CALL`, `PRE_TOOL_CALL`, `POST_TOOL_CALL`, `START`, `FINISHED`, `ERROR`.

**Multiple Providers** -- OpenAI, Anthropic, Gemini, and Vertex AI built-in. Model strings: `"openai:gpt-4o"`, `"anthropic:claude-sonnet-4-20250514"`. Extensible via `ModelProvider` ABC.

**Structured Output** -- Constrain agent output to Pydantic models with automatic schema injection.

**Human-in-the-Loop** -- Pause agents for human input, confirmation, or review.

**MCP Support** -- Model Context Protocol client/server for tool interoperability.

**Config-Driven** -- Load agents and swarms from YAML configuration files.

**Tracing** -- OpenTelemetry-based observability with `@traced` decorator and prompt logging.

**Evaluation** -- Rule-based and LLM-as-judge scorers, reflection, pass@k evaluation.

**Training** -- Trajectory collection, data synthesis, evolution, VeRL/RLHF integration.

## Package Overview

Exo is organized as a UV workspace monorepo with 13 packages:

| Package | Description |
|---------|-------------|
| **exo-core** | Agent, Tool, Runner, Swarm, Config, Events, Hooks, Registry |
| **exo-models** | LLM providers -- OpenAI, Anthropic, Gemini, Vertex AI |
| **exo-context** | Context engine, neurons, prompt builder, workspace |
| **exo-memory** | Short/long-term memory, SQLite, Postgres, vector search |
| **exo-mcp** | Model Context Protocol client/server |
| **exo-sandbox** | Local + Kubernetes sandboxed execution |
| **exo-trace** | OpenTelemetry tracing, span decorators |
| **exo-eval** | Evaluators, scorers, reflection framework |
| **exo-a2a** | Agent-to-Agent protocol (server + client) |
| **exo-cli** | CLI runner, interactive console, batch processing |
| **exo-server** | FastAPI server, sessions, WebSocket streaming |
| **exo-train** | Trajectory dataset, trainers, data synthesis |
| **exo** | Meta-package that installs core + all extras |

## Documentation

### Getting Started

- **[Installation](getting-started/installation.md)** -- pip install, UV workspace, environment variables
- **[Quickstart](getting-started/quickstart.md)** -- build and run your first agent in 5 minutes
- **[Core Concepts](getting-started/concepts.md)** -- Agent, Tool, Runner, Swarm, messages, hooks, events
- **[Your First Agent](getting-started/first-agent.md)** -- step-by-step multi-agent tutorial

### Guides

- **[Agents](guides/agents.md)** -- creating and configuring agents
- **[Tools](guides/tools.md)** -- `@tool` decorator, `Tool` ABC, schema generation
- **[Running Agents](guides/running.md)** -- `run()`, `run.sync()`, `run.stream()`
- **[Streaming](guides/streaming.md)** -- real-time streaming with events
- **[Multi-Agent Swarms](guides/multi-agent.md)** -- workflow, handoff, team modes
- **[Agent Groups](guides/agent-groups.md)** -- parallel and serial execution groups
- **[Context Engine](guides/context/index.md)** -- state, neurons, processors, workspace
- **[Memory](guides/memory.md)** -- short/long-term memory, backends
- **[Hooks](guides/hooks.md)** -- lifecycle interception
- **[Human-in-the-Loop](guides/human-in-the-loop.md)** -- human input and confirmation
- **[Structured Output](guides/structured-output.md)** -- Pydantic model constraints
- **[Model Providers](guides/models.md)** -- OpenAI, Anthropic, custom providers
- **[Config-Driven](guides/config-driven.md)** -- YAML agent loading
- **[Skills](guides/skills.md)** -- skill registry and loading
- **[MCP](guides/mcp.md)** -- Model Context Protocol
- **[Tracing](guides/tracing.md)** -- OpenTelemetry observability
- **[Evaluation](guides/evaluation.md)** -- scoring and reflection
- **[Training](guides/training.md)** -- trajectories, synthesis, VeRL
- **[Server](guides/server.md)** -- HTTP server deployment
- **[CLI](guides/cli.md)** -- command-line interface
- **[Full Guide Index](guides/index.md)** -- all 24 guides

### Reference

- **[Architecture](architecture/index.md)** -- design philosophy, dependency graph, execution flow
- **[API Reference](reference/index.md)** -- complete reference for all public APIs (90+ pages)
- **[Migration Guide](migration-guide.md)** -- migrating from AWorld to Exo
- **[Contributing](contributing/index.md)** -- development setup, code style, testing
- **[Changelog](changelog.md)** -- version history
