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

import json
import logging
from collections.abc import AsyncIterator
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from exo.runner import run as _run_agent
from exo_server.agents import agent_router
from exo_server.sessions import session_router
from exo_server.streaming import stream_router

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
        agent_name: Name of the agent to inject into (optional; uses default if omitted).
    """

    model_config = ConfigDict(frozen=True)

    message: str
    agent_name: str | None = None


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

_AGENTS_KEY = "exo_agents"
_DEFAULT_AGENT_KEY = "exo_default_agent"


def register_agent(app: FastAPI, agent: Any, *, default: bool = False) -> None:
    """Register an agent with the FastAPI app.

    Parameters:
        app: The FastAPI application instance.
        agent: An ``Agent`` (or ``Swarm``) instance with a ``name`` attribute.
        default: If ``True``, set this agent as the default for requests
            that don't specify ``agent_name``.
    """
    agents: dict[str, Any] = getattr(app.state, _AGENTS_KEY, {})
    name = getattr(agent, "name", "agent")
    agents[name] = agent
    app.state.exo_agents = agents  # type: ignore[attr-defined]
    if default or len(agents) == 1:
        app.state.exo_default_agent = name  # type: ignore[attr-defined]


def _get_agent(app: FastAPI, name: str | None) -> Any:
    """Resolve an agent by name, falling back to the default."""
    agents: dict[str, Any] = getattr(app.state, _AGENTS_KEY, {})
    if not agents:
        raise HTTPException(status_code=503, detail="No agents registered")

    if name is not None:
        agent = agents.get(name)
        if agent is None:
            raise HTTPException(status_code=404, detail=f"Agent '{name}' not found")
        return agent

    default_name: str | None = getattr(app.state, _DEFAULT_AGENT_KEY, None)
    if default_name and default_name in agents:
        return agents[default_name]

    raise HTTPException(status_code=400, detail="No agent_name specified and no default agent")


# ---------------------------------------------------------------------------
# SSE streaming helper
# ---------------------------------------------------------------------------


async def _sse_stream(agent: Any, message: str) -> AsyncIterator[str]:
    """Yield SSE-formatted events from the agent's stream."""
    stream_fn = getattr(_run_agent, "stream", None)
    if stream_fn is None:
        yield f"data: {json.dumps({'error': 'Streaming not available'})}\n\n"
        return

    try:
        async for event in stream_fn(agent, message):
            event_type = getattr(event, "type", "text")
            if event_type == "text":
                payload = {"type": "text", "text": getattr(event, "text", "")}
            else:
                payload = {
                    "type": "tool_call",
                    "tool_name": getattr(event, "tool_name", ""),
                    "tool_call_id": getattr(event, "tool_call_id", ""),
                }
            yield f"data: {json.dumps(payload)}\n\n"
    except Exception as exc:
        yield f"data: {json.dumps({'error': str(exc)})}\n\n"

    yield "data: [DONE]\n\n"


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def serve(host: str = "0.0.0.0", port: int = 8000) -> None:
    """Start the Exo Server with uvicorn.

    Parameters:
        host: Host address to bind to.
        port: Port to listen on.
    """
    import uvicorn  # pyright: ignore[reportMissingImports]

    logger.info("Starting Exo Server on %s:%d", host, port)
    app = create_app()
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
        """
        agent = _get_agent(app, request.agent_name)

        if request.stream:
            return StreamingResponse(
                _sse_stream(agent, request.message),
                media_type="text/event-stream",
            )

        # Non-streaming: call run() directly
        try:
            result = await _run_agent(agent, request.message)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

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
        """
        agent = _get_agent(app, request.agent_name)
        agent.inject_message(request.message)
        return {"status": "injected"}

    return app
