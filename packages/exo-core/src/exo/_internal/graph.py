"""Graph utilities for agent execution ordering.

Provides a simple directed graph, topological sort (Kahn's algorithm),
cycle detection, and a flow DSL parser for defining agent pipelines.
"""

from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass, field

from exo.types import ExoError


class GraphError(ExoError):
    """Raised for graph-related errors (cycles, invalid DSL, etc.)."""


@dataclass
class Graph:
    """Simple directed graph using adjacency lists.

    Nodes are strings (typically agent names). Edges are directed
    from source to target.
    """

    _adjacency: dict[str, list[str]] = field(default_factory=dict)

    def add_node(self, name: str) -> None:
        """Add a node (idempotent)."""
        if name not in self._adjacency:
            self._adjacency[name] = []

    def add_edge(self, source: str, target: str) -> None:
        """Add a directed edge from *source* to *target*.

        Both nodes are created implicitly if they don't exist.
        Duplicate edges are ignored.
        """
        self.add_node(source)
        self.add_node(target)
        if target not in self._adjacency[source]:
            self._adjacency[source].append(target)

    @property
    def nodes(self) -> list[str]:
        """Return all node names in insertion order."""
        return list(self._adjacency)

    @property
    def edges(self) -> list[tuple[str, str]]:
        """Return all edges as ``(source, target)`` tuples."""
        result: list[tuple[str, str]] = []
        for src, targets in self._adjacency.items():
            for tgt in targets:
                result.append((src, tgt))
        return result

    def successors(self, name: str) -> list[str]:
        """Return direct successors of *name*."""
        if name not in self._adjacency:
            raise GraphError(f"Unknown node: {name!r}")
        return list(self._adjacency[name])

    def in_degree(self, name: str) -> int:
        """Return the in-degree of *name*."""
        if name not in self._adjacency:
            raise GraphError(f"Unknown node: {name!r}")
        count = 0
        for targets in self._adjacency.values():
            count += targets.count(name)
        return count


def topological_sort(graph: Graph) -> list[str]:
    """Topological sort using Kahn's algorithm.

    Returns:
        Ordered list of node names.

    Raises:
        GraphError: If the graph contains a cycle.
    """
    all_nodes = graph.nodes
    if not all_nodes:
        return []

    in_deg: dict[str, int] = {n: 0 for n in all_nodes}
    for _src, tgt in graph.edges:
        in_deg[tgt] += 1

    queue: deque[str] = deque(sorted(n for n in all_nodes if in_deg[n] == 0))
    order: list[str] = []

    while queue:
        current = queue.popleft()
        order.append(current)
        for succ in sorted(graph.successors(current)):
            in_deg[succ] -= 1
            if in_deg[succ] == 0:
                queue.append(succ)

    if len(order) != len(all_nodes):
        raise GraphError("Cycle detected in graph")

    return order


# ---------------------------------------------------------------------------
# Flow DSL parser
# ---------------------------------------------------------------------------

_PARALLEL_RE = re.compile(r"\(([^)]+)\)")


def parse_flow_dsl(dsl: str) -> Graph:
    """Parse a flow DSL string into a :class:`Graph`.

    Syntax::

        "a >> b >> c"           # linear chain
        "(a | b) >> c"          # parallel group then c
        "a >> (b | c) >> d"     # a feeds into parallel b,c which feed into d

    ``>>`` denotes sequential dependency.  ``(x | y)`` denotes a parallel
    group — all members share the same predecessors and successors.

    Returns:
        A :class:`Graph` with appropriate edges.

    Raises:
        GraphError: If the DSL string is empty or malformed.
    """
    dsl = dsl.strip()
    if not dsl:
        raise GraphError("Empty flow DSL")

    # Tokenise: split by ">>" preserving parallel groups
    raw_stages = [s.strip() for s in dsl.split(">>")]
    if not raw_stages or any(s == "" for s in raw_stages):
        raise GraphError(f"Malformed flow DSL: {dsl!r}")

    stages: list[list[str]] = []
    for raw in raw_stages:
        match = _PARALLEL_RE.fullmatch(raw)
        if match:
            members = [m.strip() for m in match.group(1).split("|")]
            if not members or any(m == "" for m in members):
                raise GraphError(f"Empty member in parallel group: {raw!r}")
            stages.append(members)
        else:
            if not raw:
                raise GraphError(f"Malformed flow DSL: {dsl!r}")
            stages.append([raw])

    graph = Graph()
    for stage in stages:
        for name in stage:
            graph.add_node(name)

    for i in range(len(stages) - 1):
        for src in stages[i]:
            for tgt in stages[i + 1]:
                graph.add_edge(src, tgt)

    return graph
