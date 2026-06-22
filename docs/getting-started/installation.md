# Installation

## Requirements

- **Python 3.13+** -- Exo uses modern Python features (`types.UnionType`, `asyncio.TaskGroup`, `ExceptionGroup`); the project targets Python 3.13 or later.
- **An LLM API key** -- At minimum, you need an API key from OpenAI or Anthropic.

## Install with pip

### Meta-package (recommended)

The `exo-ai` meta-package installs exo-core plus all standard extras:

```bash
pip install exo-ai
```

### Minimal install

If you only need the core agent framework (Agent, Tool, Runner, Swarm) without LLM provider packages:

```bash
pip install exo-core
```

### With LLM providers

To use OpenAI and Anthropic models, add the models package:

```bash
pip install exo-core exo-models
```

### Individual packages

Install only what you need:

```bash
pip install exo-core       # Agent, Tool, Runner, Swarm, Config, Events, Hooks
pip install exo-models     # OpenAI + Anthropic providers
pip install exo-context    # Context engine, neurons, prompt builder
pip install exo-memory     # Short/long-term memory, vector search
pip install exo-mcp        # Model Context Protocol client/server
pip install "exo-ai[tracing]"  # OpenTelemetry tracing (OTLP exporter)
pip install exo-eval       # Evaluators, scorers, reflection
pip install exo-sandbox    # Sandboxed execution environments
pip install exo-a2a        # Agent-to-Agent protocol
pip install exo-cli        # CLI entry point
pip install exo-server     # FastAPI server + WebSocket streaming
pip install exo-train      # Trajectory dataset + trainers
```

## Environment Variables

Set API keys for the LLM providers you plan to use:

```bash
# OpenAI (required for "openai:gpt-4o", "openai:gpt-4o-mini", etc.)
export OPENAI_API_KEY="sk-..."

# Anthropic (required for "anthropic:claude-sonnet-4-20250514", etc.)
export ANTHROPIC_API_KEY="sk-ant-..."
```

You can also pass API keys programmatically when constructing a provider:

```python
from exo.models.provider import get_provider

provider = get_provider("openai:gpt-4o", api_key="sk-...")
```

Or set them in a `.env` file and load with your preferred method (e.g., `python-dotenv`).

## Development Setup (UV Workspace)

If you want to contribute to Exo or work with the full monorepo:

```bash
# Clone the repository
git clone https://github.com/midsphere-ai/exo && cd exo

# Install UV if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Sync all workspace packages (installs all 13 packages in editable mode)
uv sync

# Verify the installation
uv run python -c "from exo import Agent, run, tool; print('OK')"
```

### Running Tests

```bash
# Run all tests
uv run pytest

# Run tests for a specific package
uv run pytest packages/exo-core/tests/
uv run pytest packages/exo-models/tests/

# Run with verbose output
uv run pytest -v
```

### Linting and Type Checking

```bash
# Lint with ruff
uv run ruff check packages/

# Type-check with pyright
uv run pyright packages/exo-core/
uv run pyright packages/exo-models/
```

## Verify Your Installation

Run this minimal script to confirm everything is working:

```python
from exo import Agent, tool

@tool
def hello(name: str) -> str:
    """Say hello."""
    return f"Hello, {name}!"

agent = Agent(
    name="greeter",
    model="openai:gpt-4o-mini",
    instructions="You are a friendly greeter.",
    tools=[hello],
)

print(agent.describe())
# {'name': 'greeter', 'model': 'openai:gpt-4o-mini', 'tools': ['hello'],
#  'handoffs': [], 'max_steps': 10, 'output_type': None}
```

This script only constructs the agent and does not make any LLM calls, so it works without an API key. If this runs without errors, your installation is correct.

## Next Steps

Continue to the [Quickstart](quickstart.md) to build and run your first agent.
