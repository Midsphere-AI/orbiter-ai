"""YAML agent & swarm loader with variable substitution.

Load agent/swarm definitions from YAML files, supporting ``${ENV_VAR}``
(environment) and ``${vars.KEY}`` (internal) variable substitution,
swarm topology patterns (workflow, handoff, team), and agent factory
dispatch (builtin vs. custom classes).

Usage::

    agents = load_agents("agents.yaml")
    swarm  = load_swarm("agents.yaml")
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml

from exo.agent import Agent
from exo.swarm import Swarm
from exo.types import ExoError

_VAR_RE = re.compile(r"\$\{([^}]+)\}")


class LoaderError(ExoError):
    """Raised for YAML loading or validation errors."""


# ---------------------------------------------------------------------------
# Variable substitution
# ---------------------------------------------------------------------------


def _substitute(value: Any, env: dict[str, Any], vars_: dict[str, Any]) -> Any:
    """Recursively substitute ``${ENV_VAR}`` and ``${vars.KEY}`` in *value*."""
    if isinstance(value, str):
        # Full-string match → preserve original type
        m = _VAR_RE.fullmatch(value)
        if m:
            return _resolve_ref(m.group(1), env, vars_)
        # Partial match → string interpolation
        return _VAR_RE.sub(lambda m: str(_resolve_ref(m.group(1), env, vars_)), value)
    if isinstance(value, dict):
        return {k: _substitute(v, env, vars_) for k, v in value.items()}
    if isinstance(value, list):
        return [_substitute(v, env, vars_) for v in value]
    return value


def _resolve_ref(ref: str, env: dict[str, Any], vars_: dict[str, Any]) -> Any:
    """Resolve a single ``${ref}`` to its value."""
    if ref.startswith("vars."):
        key = ref[5:]
        if key in vars_:
            return vars_[key]
        return f"${{{ref}}}"
    # Environment variable
    val = env.get(ref)
    if val is not None:
        return val
    return f"${{{ref}}}"


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------

_AGENT_FACTORIES: dict[str, type] = {}


def register_agent_class(name: str, cls: type) -> None:
    """Register a custom agent class for YAML ``type:`` dispatch."""
    _AGENT_FACTORIES[name] = cls


def _build_agent(name: str, spec: dict[str, Any]) -> Any:
    """Build an agent from a YAML agent spec dict."""
    agent_type = spec.pop("type", "builtin")

    if agent_type != "builtin" and agent_type in _AGENT_FACTORIES:
        cls = _AGENT_FACTORIES[agent_type]
        return cls(name=name, **spec)

    # Builtin Agent construction
    kwargs: dict[str, Any] = {"name": name}
    if "model" in spec:
        kwargs["model"] = spec["model"]
    if "instructions" in spec:
        kwargs["instructions"] = spec["instructions"]
    if "system_prompt" in spec:
        kwargs["instructions"] = spec["system_prompt"]
    if "temperature" in spec:
        kwargs["temperature"] = float(spec["temperature"])
    if "max_tokens" in spec:
        kwargs["max_tokens"] = int(spec["max_tokens"])
    if "max_steps" in spec:
        kwargs["max_steps"] = int(spec["max_steps"])
    for field in (
        "planning_enabled",
        "planning_model",
        "planning_instructions",
        "budget_awareness",
        "hitl_tools",
        "bare_tools",
        "emit_mcp_progress",
        "injected_tool_args",
        # PTC settings — forwarded so YAML-defined agents can enable PTC.
        "ptc",
        "ptc_timeout",
        "ptc_max_output_bytes",
        "ptc_max_tool_calls",
        "ptc_extra_args",
    ):
        if field in spec:
            kwargs[field] = spec[field]
    return Agent(**kwargs)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load and substitute a YAML file, returning the raw dict."""
    p = Path(path)
    if not p.exists():
        raise LoaderError(f"YAML file not found: {p}")
    text = p.read_text(encoding="utf-8")
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise LoaderError(f"Expected YAML dict, got {type(data).__name__}")
    vars_ = data.pop("vars", {}) or {}
    return _substitute(data, dict(os.environ), vars_)  # type: ignore[return-value]


def load_agents(path: str | Path) -> dict[str, Any]:
    """Load agents from a YAML file.

    Returns:
        Dict mapping agent name → Agent (or custom class) instance.
    """
    data = load_yaml(path)
    agents_spec: dict[str, Any] = data.get("agents", {})
    if not agents_spec:
        raise LoaderError("No 'agents' section in YAML")
    agents: dict[str, Any] = {}
    for name, spec in agents_spec.items():
        agents[name] = _build_agent(name, dict(spec))
    return agents


def load_swarm(path: str | Path) -> Swarm:
    """Load a swarm (with agents) from a YAML file.

    If the YAML has no ``swarm`` section, creates a workflow-mode
    swarm with agents in declaration order.

    Returns:
        Configured ``Swarm`` instance.
    """
    data = load_yaml(path)
    agents_spec: dict[str, Any] = data.get("agents", {})
    if not agents_spec:
        raise LoaderError("No 'agents' section in YAML")

    agents: dict[str, Any] = {}
    for name, spec in agents_spec.items():
        agents[name] = _build_agent(name, dict(spec))

    swarm_spec = data.get("swarm", {})
    mode = swarm_spec.get("type", "workflow")
    flow = swarm_spec.get("flow")
    order = swarm_spec.get("order")
    max_handoffs = swarm_spec.get("max_handoffs", 10)

    # Build agent list in specified order
    if order:
        agent_list = []
        for n in order:
            if n not in agents:
                raise LoaderError(f"Swarm order references unknown agent '{n}'")
            agent_list.append(agents[n])
    else:
        agent_list = list(agents.values())

    # Wire handoff edges if specified
    edges = swarm_spec.get("edges", [])
    for src, dst in edges:
        if src not in agents or dst not in agents:
            raise LoaderError(f"Handoff edge references unknown agent: {src} → {dst}")
        a = agents[src]
        if dst not in a.handoffs and dst in agents:
            a.handoffs[dst] = agents[dst]

    return Swarm(
        agents=agent_list,
        flow=flow,
        mode=mode,
        max_handoffs=max_handoffs,
    )
