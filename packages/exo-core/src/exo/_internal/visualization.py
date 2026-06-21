"""Mermaid flowchart generation from Swarm workflows.

Generates valid Mermaid ``graph TD`` syntax by inspecting the Swarm's
agents and flow topology.  Node shapes reflect type:

- Regular agents: ``[name]`` (rectangle)
- BranchNode: ``{name}`` (diamond / rhombus)
- LoopNode: ``{{name}}`` (hexagon)
- SwarmNode: ``[[name]]`` (subroutine)
- ParallelGroup: ``subgraph`` containing member nodes
"""

from __future__ import annotations

import re
from typing import Any

from exo._internal.graph import Graph, parse_flow_dsl

# Mermaid IDs must be alphanumeric (plus underscore).
_UNSAFE_RE = re.compile(r"[^a-zA-Z0-9_]")


def _safe_id(name: str) -> str:
    """Convert an agent name to a valid Mermaid node ID."""
    return _UNSAFE_RE.sub("_", name)


def _escape_label(label: str) -> str:
    """Escape characters that are special in Mermaid labels."""
    return label.replace('"', "#quot;")


def _build_graph(swarm: Any) -> Graph:
    """Reconstruct the topology graph from a Swarm instance."""
    if swarm.flow:
        return parse_flow_dsl(swarm.flow)

    # No DSL — build a linear chain from flow_order.
    graph = Graph()
    for i, name in enumerate(swarm.flow_order):
        graph.add_node(name)
        if i > 0:
            graph.add_edge(swarm.flow_order[i - 1], name)
    return graph


def to_mermaid(swarm: Any) -> str:
    """Generate a Mermaid flowchart from a :class:`~exo.swarm.Swarm`.

    Parameters:
        swarm: A ``Swarm`` instance.

    Returns:
        A string containing valid ``graph TD`` Mermaid syntax.
    """
    graph = _build_graph(swarm)
    lines: list[str] = ["graph TD"]

    # Track which node IDs represent groups (subgraphs) — edges to/from
    # a group target the subgraph ID, not individual members.
    group_nodes: set[str] = set()

    # --- Node declarations ---------------------------------------------------
    for name in graph.nodes:
        agent = swarm.agents.get(name)
        nid = _safe_id(name)
        label = _escape_label(name)

        if agent is None:
            # Defensive: node referenced in DSL but missing from agents dict.
            lines.append(f"    {nid}[{label}]")
            continue

        if getattr(agent, "is_branch", False):
            lines.append(f"    {nid}{{{label}}}")
        elif getattr(agent, "is_loop", False):
            lines.append(f"    {nid}{{{{{label}}}}}")
        elif getattr(agent, "is_swarm", False):
            lines.append(f"    {nid}[[{label}]]")
        elif getattr(agent, "is_group", False):
            group_nodes.add(name)
            lines.append(f"    subgraph {nid}[{label}]")
            for sub_name in getattr(agent, "agent_order", []):
                sub_id = _safe_id(sub_name)
                sub_label = _escape_label(sub_name)
                lines.append(f"        {sub_id}[{sub_label}]")
            lines.append("    end")
        else:
            lines.append(f"    {nid}[{label}]")

    # --- Edges from the flow graph -------------------------------------------
    # Collect existing edge targets from each source so we can avoid
    # duplicating branch / loop edges added later.
    edge_set: set[tuple[str, str]] = set()

    for src, tgt in graph.edges:
        src_agent = swarm.agents.get(src)
        src_id = _safe_id(src)
        tgt_id = _safe_id(tgt)

        if getattr(src_agent, "is_branch", False):
            # Label edges with true / false based on branch targets.
            if tgt == src_agent.if_true:
                lines.append(f"    {src_id} -->|true| {tgt_id}")
            elif tgt == src_agent.if_false:
                lines.append(f"    {src_id} -->|false| {tgt_id}")
            else:
                lines.append(f"    {src_id} --> {tgt_id}")
        else:
            lines.append(f"    {src_id} --> {tgt_id}")

        edge_set.add((src, tgt))

    # --- Extra branch edges (targets outside the flow graph) -----------------
    for name in graph.nodes:
        agent = swarm.agents.get(name)
        if agent is None or not getattr(agent, "is_branch", False):
            continue
        nid = _safe_id(name)
        for label, target in [("true", agent.if_true), ("false", agent.if_false)]:
            if (name, target) not in edge_set and target in swarm.agents:
                tgt_id = _safe_id(target)
                lines.append(f"    {nid} -->|{label}| {tgt_id}")
                edge_set.add((name, target))

    # --- Loop body edges -----------------------------------------------------
    for name in graph.nodes:
        agent = swarm.agents.get(name)
        if agent is None or not getattr(agent, "is_loop", False):
            continue
        nid = _safe_id(name)
        body_names: list[str] = getattr(agent, "body", [])
        prev_id = nid
        for body_name in body_names:
            if body_name not in swarm.agents:
                continue
            body_id = _safe_id(body_name)
            # Declare body node if not already in the flow graph.
            if body_name not in {n for n in graph.nodes}:
                body_label = _escape_label(body_name)
                lines.append(f"    {body_id}[{body_label}]")
            if (name if prev_id == nid else "", body_name) not in edge_set:
                lines.append(f"    {prev_id} -->|body| {body_id}")
            prev_id = body_id
        # Loop-back edge from last body node to loop node.
        if body_names:
            last_body_id = _safe_id(body_names[-1])
            lines.append(f"    {last_body_id} -.->|loop| {nid}")

    return "\n".join(lines)
