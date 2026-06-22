"""Agent discovery and loading for the Exo CLI.

Scans directories for agent definitions in three formats:

1. **Python files** (``.py``) — modules containing a ``create_agent()``
   factory function that returns an ``Agent`` instance.
2. **YAML files** (``.yaml``) — agent configs loaded via
   :func:`exo.loader.load_agents`.
3. **Markdown files** (``.md``) — agent definitions with YAML front-matter
   (``name``, ``model``, ``instructions``) and body as system prompt.

Usage::

    registry = scan_directory("/path/to/agents")
    agent = registry["my_agent"]
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

from exo.agent import Agent
from exo.types import ExoError  # pyright: ignore[reportMissingImports]

# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class AgentLoadError(ExoError):
    """Raised when agent loading or validation fails."""


# ---------------------------------------------------------------------------
# Individual file loaders
# ---------------------------------------------------------------------------


def load_python_agent(path: Path) -> dict[str, Any]:
    """Load agents from a Python file.

    The module must define a ``create_agent()`` callable that returns
    an ``Agent`` (or dict of name → Agent for multi-agent files).

    Returns:
        Dict mapping agent name → agent instance.
    """
    module_name = f"_exo_agent_{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise AgentLoadError(
            f"Cannot create module spec for {path}",
            hint="Verify the file exists and is a valid Python source file.",
        )

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        del sys.modules[module_name]
        raise AgentLoadError(f"Error executing {path}: {exc}") from exc

    factory = getattr(module, "create_agent", None)
    if factory is None:
        del sys.modules[module_name]
        raise AgentLoadError(
            f"No create_agent() function in {path}",
            hint="Define a top-level create_agent() -> Agent function in the module.",
        )

    try:
        result = factory()
    except Exception as exc:
        raise AgentLoadError(f"create_agent() in {path} raised: {exc}") from exc
    finally:
        # Remove the transient module from sys.modules so repeated loads of files
        # with the same stem don't shadow each other and test runs stay isolated.
        sys.modules.pop(module_name, None)

    if isinstance(result, dict):
        return result
    # Single agent — use name attribute or file stem
    name = getattr(result, "name", path.stem)
    return {name: result}


def load_yaml_agents(path: Path) -> dict[str, Any]:
    """Load agents from a YAML config file.

    Delegates to :func:`exo.loader.load_agents`.
    """
    from exo.loader import LoaderError, load_agents

    try:
        return load_agents(path)
    except LoaderError as exc:
        raise AgentLoadError(f"YAML load failed for {path}: {exc}") from exc


def _parse_front_matter(text: str) -> tuple[dict[str, str], str]:
    """Extract YAML front-matter and body from a markdown file.

    Returns:
        Tuple of (front-matter dict, body string).
    """
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, text

    end = -1
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            end = i
            break
    if end < 0:
        return {}, text

    meta: dict[str, str] = {}
    for line in lines[1:end]:
        colon = line.find(":")
        if colon < 0:
            continue
        key = line[:colon].strip().lower()
        val = line[colon + 1 :].strip()
        meta[key] = val

    body = "\n".join(lines[end + 1 :]).strip()
    return meta, body


def load_markdown_agent(path: Path) -> dict[str, Any]:
    """Load an agent from a Markdown file with front-matter.

    Front-matter fields:
        - ``name``: Agent name (defaults to file stem).
        - ``model``: Model string (e.g. ``openai:gpt-4o``).
        - ``instructions``: Explicit instructions (overrides body).
        - ``temperature``: Float temperature value.
        - ``max_tokens``: Integer max tokens.
        - ``max_steps``: Integer max steps.

    The markdown body (after front-matter) is used as ``instructions``
    unless an explicit ``instructions`` field is provided.
    """
    text = path.read_text(encoding="utf-8")
    meta, body = _parse_front_matter(text)

    name = meta.get("name", path.stem)
    kwargs: dict[str, Any] = {"name": name}

    if "model" in meta:
        kwargs["model"] = meta["model"]

    # Explicit instructions take priority over body
    if "instructions" in meta:
        kwargs["instructions"] = meta["instructions"]
    elif body:
        kwargs["instructions"] = body

    if "temperature" in meta:
        try:
            kwargs["temperature"] = float(meta["temperature"])
        except ValueError as exc:
            raise AgentLoadError(f"Invalid temperature in {path}: {meta['temperature']}") from exc

    if "max_tokens" in meta:
        try:
            kwargs["max_tokens"] = int(meta["max_tokens"])
        except ValueError as exc:
            raise AgentLoadError(f"Invalid max_tokens in {path}: {meta['max_tokens']}") from exc

    if "max_steps" in meta:
        try:
            kwargs["max_steps"] = int(meta["max_steps"])
        except ValueError as exc:
            raise AgentLoadError(f"Invalid max_steps in {path}: {meta['max_steps']}") from exc

    return {name: Agent(**kwargs)}


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

_AGENT_EXTENSIONS: dict[str, Any] = {
    ".py": load_python_agent,
    ".yaml": load_yaml_agents,
    ".md": load_markdown_agent,
}


def discover_agent_files(directory: str | Path) -> list[Path]:
    """Scan *directory* for agent definition files.

    Returns files with extensions ``.py``, ``.yaml``, and ``.md``
    sorted by name for deterministic ordering.  Only immediate children
    are scanned (no recursive walk).
    """
    base = Path(directory)
    if not base.is_dir():
        raise AgentLoadError(f"Not a directory: {base}")
    files: list[Path] = []
    for p in sorted(base.iterdir()):
        if p.is_file() and p.suffix in _AGENT_EXTENSIONS:
            files.append(p)
    return files


def validate_agent(name: str, agent: Any) -> None:
    """Validate that *agent* looks like a usable agent instance.

    Checks for required attributes: ``name`` and ``run``.

    Raises:
        AgentLoadError: If validation fails.
    """
    if not hasattr(agent, "name"):
        raise AgentLoadError(f"Agent '{name}' missing 'name' attribute")
    if not hasattr(agent, "run"):
        raise AgentLoadError(f"Agent '{name}' missing 'run' method")


def scan_directory(
    directory: str | Path,
    *,
    validate: bool = True,
) -> dict[str, Any]:
    """Discover and load all agents from *directory*.

    Scans for ``.py``, ``.yaml``, and ``.md`` files, loads each via
    the appropriate loader, validates agent instances, and returns a
    merged dict of name → agent.

    Args:
        directory: Path to scan for agent files.
        validate: Whether to validate loaded agents (default ``True``).

    Returns:
        Dict mapping agent name → agent instance.

    Raises:
        AgentLoadError: On discovery, loading, or validation errors.
    """
    files = discover_agent_files(directory)
    agents: dict[str, Any] = {}

    for path in files:
        loader_fn = _AGENT_EXTENSIONS.get(path.suffix)
        if loader_fn is None:
            continue

        loaded = loader_fn(path)
        for name, agent in loaded.items():
            if validate:
                validate_agent(name, agent)
            if name in agents:
                raise AgentLoadError(f"Duplicate agent name '{name}' (from {path}, already loaded)")
            agents[name] = agent

    return agents
