# Package Structure

Exo is a UV workspace monorepo with 13 packages. This document describes the conventions for package layout, namespace packages, and how to add new packages.

## Standard Package Layout

Every package follows this structure:

```
packages/exo-<name>/
+-- pyproject.toml
+-- src/
|   +-- exo/
|       +-- __init__.py          # extend_path + re-exports
|       +-- <name>/              # (for sub-namespace packages)
|       |   +-- __init__.py      # public API
|       |   +-- module.py
|       |   +-- _internal/       # private implementation details
|       |       +-- __init__.py
|       |       +-- helper.py
|       +-- module.py            # (for top-level namespace)
+-- tests/
    +-- __init__.py
    +-- test_module.py
```

### exo-core (top-level namespace)

`exo-core` contributes modules directly to the `exo` namespace:

```
packages/exo-core/
+-- src/
|   +-- exo/
|       +-- __init__.py      # extend_path, re-exports Agent, Swarm, Tool, tool, run
|       +-- types.py         # Message types, RunResult, StreamEvent
|       +-- config.py        # AgentConfig, ModelConfig, etc.
|       +-- registry.py      # Registry[T]
|       +-- events.py        # EventBus
|       +-- hooks.py         # Hook, HookPoint, HookManager
|       +-- tool.py          # Tool ABC, @tool decorator, FunctionTool
|       +-- agent.py         # Agent class
|       +-- swarm.py         # Swarm class
|       +-- runner.py        # run(), run.sync(), run.stream()
|       +-- _internal/
|           +-- __init__.py
|           +-- message_builder.py
|           +-- output_parser.py
|           +-- call_runner.py
|           +-- state.py
|           +-- graph.py
+-- tests/
    +-- test_types.py
    +-- test_config.py
    +-- test_agent.py
    +-- ...
```

### Sub-namespace packages (exo-models, exo-context, etc.)

Other packages contribute to sub-namespaces like `exo.models`, `exo.context`:

```
packages/exo-models/
+-- src/
|   +-- exo/
|       +-- __init__.py          # extend_path (namespace extension)
|       +-- models/
|           +-- __init__.py      # re-exports: ModelProvider, get_provider, etc.
|           +-- types.py         # ModelError, ModelResponse, StreamChunk
|           +-- provider.py      # ModelProvider ABC, model_registry
|           +-- openai.py        # OpenAIProvider
|           +-- anthropic.py     # AnthropicProvider
+-- tests/
    +-- test_model_types.py      # Note: prefixed to avoid collision with exo-core
    +-- ...
```

## Namespace Packages

Exo uses `pkgutil.extend_path()` to create namespace packages. This allows multiple PyPI packages to contribute to the same `exo` import path.

### How It Works

Each package's `exo/__init__.py` calls `extend_path()`:

```python
"""Exo Core: Agent, Tool, Runner, Config, Events, Hooks, Swarm."""

from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)
__version__ = "0.1.0"

# Public API re-exports
from exo.agent import Agent
from exo.runner import run
from exo.swarm import Swarm
from exo.tool import FunctionTool, Tool, tool
```

When Python imports `exo`, it finds the `exo/` directory from `exo-core`. The `extend_path()` call tells Python to also search other installed packages for `exo/` directories. This means `exo-models` can add `exo/models/` and it becomes importable as `from exo.models import ...`.

### Important Details

- **Every package needs `__init__.py`** in its `exo/` directory with `extend_path()`
- **Re-exports only in `__init__.py`** -- no logic, no complex imports
- **`__all__` is recommended** when the export list might be ambiguous
- **The meta-package** (`packages/exo/`) uses a `_exo_meta` dummy package for hatchling compatibility

## `__init__.py` Pattern

```python
"""<Package description>."""

from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)
__version__ = "0.1.0"

# Public API — these are what users import
from exo.agent import Agent
from exo.runner import run
# ...

__all__ = [
    "Agent",
    "run",
    # ...
]
```

### Rules

- `__init__.py` is the **public API surface** -- only export what users need
- Use `__all__` if the export list is ambiguous
- **Never put logic in `__init__.py`** -- only imports and re-exports
- Subpackages (`exo.models`, `exo.context`, etc.) follow the same pattern

## pyproject.toml Template

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "exo-<name>"
version = "0.1.0"
description = "<description>"
requires-python = ">=3.13"
dependencies = [
    "exo-core>=0.1.0",
]

[tool.hatch.build.targets.wheel]
packages = ["src/exo"]

[tool.uv.sources]
exo-core = { workspace = true }
```

### Key Points

- **Build system:** hatchling (consistent across all packages)
- **Python version:** 3.13+ (project floor; `asyncio.TaskGroup`, `StrEnum` need 3.11+)
- **Workspace sources:** Use `{ workspace = true }` for workspace dependencies so UV resolves them locally
- **Wheel packages:** Point to `src/exo` so the namespace package structure is preserved

## Naming Conventions

| Thing | Convention | Example |
|-------|-----------|---------|
| Package directory | `exo-<name>` | `exo-core`, `exo-models` |
| PyPI package name | `exo-<name>` | `pip install exo-models` |
| Import path | `exo.<name>` | `from exo.models import ...` |
| Module files | `snake_case.py` | `message_builder.py`, `output_parser.py` |
| Internal modules | `_internal/<name>.py` | `_internal/call_runner.py` |

## Dependency Rules

1. **`exo-core` has ZERO heavy dependencies** -- only `pydantic`
2. **Provider SDKs** live in `exo-models` only (e.g., `openai`, `anthropic`)
3. **Optional heavy deps** declared as extras (e.g., `exo-memory[qdrant]`)
4. **No lateral dependencies** between same-level packages (e.g., `exo-models` does not depend on `exo-memory`)
5. **`_internal/` modules are never imported from outside their package**

## Adding a New Package

1. Create the directory structure:
   ```bash
   mkdir -p packages/exo-foo/src/exo/foo
   mkdir -p packages/exo-foo/tests
   ```

2. Create `packages/exo-foo/pyproject.toml` (see template above)

3. Create `packages/exo-foo/src/exo/__init__.py`:
   ```python
   from pkgutil import extend_path
   __path__ = extend_path(__path__, __name__)
   ```

4. Create `packages/exo-foo/src/exo/foo/__init__.py`:
   ```python
   """Exo Foo: <description>."""
   # Public API re-exports
   ```

5. Create `packages/exo-foo/tests/__init__.py` (empty)

6. Add to workspace root `pyproject.toml` members list

7. Run `uv sync`

## Test File Uniqueness

Test file names must be **unique across all packages**. pytest collects `tests/` from multiple packages into one module namespace.

```
# BAD -- collision between exo-core and exo-models
packages/exo-core/tests/test_types.py
packages/exo-models/tests/test_types.py     # NAME COLLISION

# GOOD -- prefixed names
packages/exo-core/tests/test_types.py
packages/exo-models/tests/test_model_types.py
```
