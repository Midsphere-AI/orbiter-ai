"""Exo Server: FastAPI app with /chat endpoint.

Provides a web API for running Exo agents via HTTP.
Supports both synchronous request/response and streaming SSE.

Usage::

    from exo_server.app import create_app, register_agent

    app = create_app()
    register_agent(app, my_agent)

    # Run with: uvicorn exo_server.app:app
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

from exo.runner import run as _run_agent
from exo_server._constants import AGENTS_KEY, DEFAULT_AGENT_KEY
from exo_server.agents import agent_router
from exo_server.sessions import session_router
from exo_server.streaming import _sse_iter, stream_router

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class ChatRequest(BaseModel):
    """Request body for the /chat endpoint.

    Attributes:
        message: The user's input message.
        agent_name: Name of the agent to invoke (optional; uses default if omitted).
        stream: Whether to stream the response via SSE.
    """

    model_config = ConfigDict(frozen=True)

    message: str
    agent_name: str | None = None
    stream: bool = False


class InjectRequest(BaseModel):
    """Request body for the /inject endpoint.

    Attributes:
        message: The message to inject into the running agent's context.
            Must be a non-empty string.
        agent_name: Name of the agent to inject into (optional; uses default if omitted).
    """

    model_config = ConfigDict(frozen=True)

    message: str
    agent_name: str | None = None

    @field_validator("message")
    @classmethod
    def _message_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("message must not be empty")
        return v


class ChatResponse(BaseModel):
    """Non-streaming response from the /chat endpoint.

    Attributes:
        output: The agent's text response.
        agent_name: Name of the agent that produced the response.
        steps: Number of LLM call steps taken.
        usage: Token usage statistics.
    """

    model_config = ConfigDict(frozen=True)

    output: str = ""
    agent_name: str = ""
    steps: int = 0
    usage: dict[str, int] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Agent registry (per-app state)
# ---------------------------------------------------------------------------


def register_agent(app: FastAPI, agent: Any, *, default: bool = False) -> None:
    """Register an agent with the FastAPI app.

    Parameters:
        app: The FastAPI application instance.
        agent: An ``Agent`` (or ``Swarm``) instance with a ``name`` attribute.
        default: If ``True``, set this agent as the default for requests
            that don't specify ``agent_name``.
    """
    agents: dict[str, Any] = getattr(app.state, AGENTS_KEY, {})
    name = getattr(agent, "name", "agent")
    agents[name] = agent
    app.state.exo_agents = agents  # type: ignore[attr-defined]
    if default or len(agents) == 1:
        app.state.exo_default_agent = name  # type: ignore[attr-defined]


def _get_agent(app: FastAPI, name: str | None) -> Any:
    """Resolve an agent by name, falling back to the default."""
    agents: dict[str, Any] = getattr(app.state, AGENTS_KEY, {})
    if not agents:
        raise HTTPException(status_code=503, detail="No agents registered")

    if name is not None:
        agent = agents.get(name)
        if agent is None:
            raise HTTPException(status_code=404, detail=f"Agent '{name}' not found")
        return agent

    default_name: str | None = getattr(app.state, DEFAULT_AGENT_KEY, None)
    if default_name and default_name in agents:
        return agents[default_name]

    raise HTTPException(status_code=400, detail="No agent_name specified and no default agent")


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def serve(host: str = "0.0.0.0", port: int = 8000, agents: list[Any] | None = None) -> None:
    """Start the Exo Server with uvicorn.

    Parameters:
        host: Host address to bind to.
        port: Port to listen on.
        agents: Optional list of ``Agent`` (or ``Swarm``) instances to register
            before the server starts.  The first agent in the list becomes the
            default.  Without agents every ``/chat`` request returns 503, so
            pass at least one agent here (or call ``register_agent`` on the
            app returned by ``create_app()`` before handing it to uvicorn).
    """
    import uvicorn  # pyright: ignore[reportMissingImports]

    logger.info("Starting Exo Server on %s:%d", host, port)
    app = create_app()
    for agent in agents or []:
        register_agent(app, agent)
    uvicorn.run(app, host=host, port=port)


def create_app() -> FastAPI:
    """Create a configured FastAPI application with the /chat endpoint."""
    app = FastAPI(title="Exo Server", version="0.1.0")
    app.include_router(agent_router)
    app.include_router(session_router)
    app.include_router(stream_router)

    @app.post("/chat", response_model=ChatResponse)
    async def chat(request: ChatRequest) -> Any:
        """Run an agent and return the response.

        When ``stream=True``, returns Server-Sent Events instead of JSON.
        Delegates to the shared ``_sse_iter`` helper in ``streaming.py``.
        """
        agent = _get_agent(app, request.agent_name)

        if request.stream:
            return StreamingResponse(
                _sse_iter(agent, request.message),
                media_type="text/event-stream",
            )

        # Non-streaming: call run() directly
        try:
            result = await _run_agent(agent, request.message)
        except Exception as exc:
            logger.exception("Unhandled error in /chat: %s", exc)
            raise HTTPException(status_code=500, detail="internal error") from exc

        usage_obj = getattr(result, "usage", None)
        usage_dict: dict[str, int] = {}
        if usage_obj is not None:
            usage_dict = {
                "input_tokens": getattr(usage_obj, "input_tokens", 0) or 0,
                "output_tokens": getattr(usage_obj, "output_tokens", 0) or 0,
                "total_tokens": getattr(usage_obj, "total_tokens", 0) or 0,
            }

        return ChatResponse(
            output=getattr(result, "output", "") or "",
            agent_name=getattr(agent, "name", ""),
            steps=getattr(result, "steps", 0) or 0,
            usage=usage_dict,
        )

    @app.post("/inject")
    async def inject_message(request: InjectRequest) -> dict[str, str]:
        """Inject a message into a running agent's context.

        The message is picked up before the agent's next LLM call.
        Returns 422 if the message is empty (validated by ``InjectRequest``).
        Returns 404 / 503 if the named agent is not registered.
        """
        agent = _get_agent(app, request.agent_name)
        try:
            agent.inject_message(request.message)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {"status": "injected"}

    return app
