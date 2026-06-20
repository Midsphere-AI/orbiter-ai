"""Agent management and workspace routes.

Provides endpoints for listing registered agents, inspecting agent details,
and accessing workspace artifacts (files) associated with agents.

Usage::

    from exo_server.agents import agent_router

    app = FastAPI()
    app.include_router(agent_router)
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------


class AgentInfo(BaseModel):
    """Summary information about a registered agent."""

    model_config = ConfigDict(frozen=True)

    name: str
    model: str = ""
    is_default: bool = False
    tools: list[str] = Field(default_factory=list)
    handoffs: list[str] = Field(default_factory=list)
    max_steps: int = 0
    temperature: float = 0.0
    max_tokens: int | None = None


class WorkspaceFile(BaseModel):
    """Metadata about a file/artifact in an agent's workspace."""

    model_config = ConfigDict(frozen=True)

    name: str
    artifact_type: str = "text"
    version_count: int = 1


class WorkspaceFileContent(BaseModel):
    """Full content of a workspace file."""

    model_config = ConfigDict(frozen=True)

    name: str
    content: str
    artifact_type: str = "text"
    version_count: int = 1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_AGENTS_KEY = "exo_agents"
_DEFAULT_AGENT_KEY = "exo_default_agent"


def _get_agents(state: Any) -> dict[str, Any]:
    """Retrieve the agent registry from app state."""
    return getattr(state, _AGENTS_KEY, {})


def _get_default_name(state: Any) -> str | None:
    """Retrieve the default agent name from app state."""
    return getattr(state, _DEFAULT_AGENT_KEY, None)


def _get_workspace(agent: Any) -> Any | None:
    """Retrieve the workspace from an agent's context, if available."""
    ctx = getattr(agent, "context", None)
    if ctx is None:
        return None
    return getattr(ctx, "workspace", None)


def _agent_info(agent: Any, *, is_default: bool) -> AgentInfo:
    """Build an AgentInfo from an agent object."""
    tools_dict: dict[str, Any] = getattr(agent, "tools", {})
    handoffs_dict: dict[str, Any] = getattr(agent, "handoffs", {})
    return AgentInfo(
        name=getattr(agent, "name", ""),
        model=getattr(agent, "model", ""),
        is_default=is_default,
        tools=list(tools_dict.keys()) if isinstance(tools_dict, dict) else [],
        handoffs=list(handoffs_dict.keys()) if isinstance(handoffs_dict, dict) else [],
        max_steps=getattr(agent, "max_steps", 0) or 0,
        temperature=getattr(agent, "temperature", 0.0) or 0.0,
        max_tokens=getattr(agent, "max_tokens", None),
    )


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

agent_router = APIRouter(prefix="/agents", tags=["agents"])


@agent_router.get("", response_model=list[AgentInfo])
async def list_agents(req: Request) -> Any:
    """List all registered agents."""
    agents = _get_agents(req.app.state)
    default_name = _get_default_name(req.app.state)
    return [_agent_info(agent, is_default=(name == default_name)) for name, agent in agents.items()]


@agent_router.get("/{agent_name}", response_model=AgentInfo)
async def get_agent(req: Request, agent_name: str) -> Any:
    """Get details for a specific agent."""
    agents = _get_agents(req.app.state)
    agent = agents.get(agent_name)
    if agent is None:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found")
    default_name = _get_default_name(req.app.state)
    return _agent_info(agent, is_default=(agent_name == default_name))


@agent_router.get("/{agent_name}/workspace", response_model=list[WorkspaceFile])
async def list_workspace_files(req: Request, agent_name: str) -> Any:
    """List files in an agent's workspace."""
    agents = _get_agents(req.app.state)
    agent = agents.get(agent_name)
    if agent is None:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found")
    workspace = _get_workspace(agent)
    if workspace is None:
        return []
    artifacts: list[Any] = workspace.list()
    return [
        WorkspaceFile(
            name=getattr(a, "name", ""),
            artifact_type=str(getattr(a, "artifact_type", "text")),
            version_count=len(getattr(a, "versions", [])),
        )
        for a in artifacts
    ]


@agent_router.get("/{agent_name}/workspace/{file_name:path}", response_model=WorkspaceFileContent)
async def read_workspace_file(req: Request, agent_name: str, file_name: str) -> Any:
    """Read the content of a specific workspace file."""
    agents = _get_agents(req.app.state)
    agent = agents.get(agent_name)
    if agent is None:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found")
    workspace = _get_workspace(agent)
    if workspace is None:
        raise HTTPException(status_code=404, detail="Agent has no workspace")
    artifact = workspace.get(file_name)
    if artifact is None:
        raise HTTPException(status_code=404, detail=f"File '{file_name}' not found")
    return WorkspaceFileContent(
        name=artifact.name,
        content=getattr(artifact, "content", ""),
        artifact_type=str(getattr(artifact, "artifact_type", "text")),
        version_count=len(getattr(artifact, "versions", [])),
    )
