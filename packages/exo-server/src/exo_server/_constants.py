"""Shared state-key constants for exo_server.

These keys are used to store agent registry and default-agent name
on the FastAPI ``app.state`` object.  Importing from a single module
prevents the duplication that previously existed between app.py,
agents.py, and streaming.py.
"""

from __future__ import annotations

#: Key under which the ``{name: agent}`` registry dict is stored on app state.
AGENTS_KEY: str = "exo_agents"

#: Key under which the default agent name (``str | None``) is stored on app state.
DEFAULT_AGENT_KEY: str = "exo_default_agent"
