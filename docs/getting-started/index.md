# Getting Started

Welcome to Exo. This section walks you from zero to a working multi-agent system.

## Learning Path

Follow these pages in order for the best experience:

### 1. [Installation](installation.md)

Set up Exo in your project. Covers `pip install`, UV workspace development, Python version requirements, and environment variables for LLM providers.

### 2. [Quickstart](quickstart.md)

Build and run your first agent in under 5 minutes. A weather-bot example that covers `@tool`, `Agent`, `run.sync()`, streaming, and multi-turn conversations.

### 3. [Core Concepts](concepts.md)

Understand the building blocks of Exo: Agent, Tool, Runner, Swarm, message types, RunResult, and streaming events. This is the reference you will come back to most often.

### 4. [Your First Agent](first-agent.md)

A step-by-step tutorial that builds a real multi-agent system from scratch. Define tools, create agents, inspect results, add handoffs, set up a Swarm workflow, and use structured output.

## What You Will Build

By the end of this section you will know how to:

- Install Exo and configure LLM providers
- Define typed tools with the `@tool` decorator
- Create agents with instructions and tools
- Run agents synchronously, asynchronously, and with streaming
- Build multi-turn conversations by passing message history
- Orchestrate multiple agents with Swarm (workflow, handoff, and team modes)
- Validate agent output with Pydantic structured output

## Prerequisites

- Python 3.13 or later
- An API key for at least one LLM provider (OpenAI or Anthropic)
- Basic familiarity with `async`/`await` in Python (helpful but not required -- `run.sync()` provides a blocking API)

## Next Steps

Once you have finished the Getting Started section, explore:

- **[Guides](../guides/context/index.md)** -- Deep dives into context engine, memory, tracing
- **[Architecture](../architecture/index.md)** -- How Exo is designed internally
- **[API Reference](../reference/index.md)** -- Complete API documentation
