"""WebSocket and SSE streaming for Exo Server.

Provides real-time streaming of agent output via WebSocket and
Server-Sent Events (SSE) as a fallback for non-WebSocket clients.

WebSocket protocol:
    Client sends: ``{"message": "...", "agent_name": "..."}``
    Server sends: ``{"type": "text", "text": "..."}`` or
                  ``{"type": "tool_call", "tool_name": "...", "tool_call_id": "..."}``
    Server sends: ``{"type": "done"}`` when complete
    Server sends: ``{"type": "error", "error": "..."}`` on failure

SSE endpoint:
    GET ``/stream?message=...&agent_name=...`` returns ``text/event-stream``
    with the same JSON payloads, ending with ``data: [DONE]``.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse

from exo.runner import run as _run_agent
from exo_server._constants import AGENTS_KEY, DEFAULT_AGENT_KEY

logger = logging.getLogger(__name__)

stream_router = APIRouter()


async def _iter_events(agent: Any, message: str) -> AsyncIterator[dict[str, Any]]:
    """Iterate over agent stream events as JSON-serialisable dicts.

    Only ``"text"`` and ``"tool_call"`` events are forwarded to the client;
    all other event types (step, status, usage, error, reasoning,
    tool_result, tool_call_delta, etc.) are silently dropped so that
    consumers never receive a payload mislabelled as ``tool_call``.
    """
    stream_fn = getattr(_run_agent, "stream", None)
    if stream_fn is None:
        yield {"type": "error", "error": "Streaming not available"}
        return

    try:
        async for event in stream_fn(agent, message):
            event_type = getattr(event, "type", None)
            if event_type == "text":
                yield {"type": "text", "text": getattr(event, "text", "")}
            elif event_type == "tool_call":
                yield {
                    "type": "tool_call",
                    "tool_name": getattr(event, "tool_name", ""),
                    "tool_call_id": getattr(event, "tool_call_id", ""),
                }
            # All other event types (step, status, usage, error, reasoning,
            # tool_result, tool_call_delta, context, …) are intentionally
            # dropped — this server only surfaces text and tool_call to clients.
    except Exception as exc:
        logger.exception("Error streaming agent response: %s", exc)
        yield {"type": "error", "error": str(exc)}


def _resolve_agent(app_state: Any, name: str | None) -> Any:
    """Resolve agent from app state without raising HTTPException."""
    agents: dict[str, Any] = getattr(app_state, AGENTS_KEY, {})
    if not agents:
        return None

    if name is not None:
        return agents.get(name)

    default_name: str | None = getattr(app_state, DEFAULT_AGENT_KEY, None)
    if default_name and default_name in agents:
        return agents[default_name]

    return None


@stream_router.websocket("/ws/chat")
async def ws_chat(websocket: WebSocket) -> None:
    """WebSocket endpoint for real-time agent streaming.

    Expects a JSON message: ``{"message": "...", "agent_name": "..."}``
    Sends back streaming events and a final ``{"type": "done"}`` message.

    Origin allowlist (opt-in):
        Set ``app.state.exo_ws_allowed_origins`` to a list of allowed Origin
        header values (e.g. ``["https://app.example.com"]``).  When the list
        is empty or absent the check is skipped — default-permissive for local
        development.  Example::

            app.state.exo_ws_allowed_origins = ["https://prod.example.com"]

    # SECURITY: WebSocket connections inherit cookies/credentials from the
    # browser, making them vulnerable to cross-site WebSocket hijacking when
    # the Origin header is not validated. Enable the allowlist in production.
    """
    # SECURITY: Origin check — permissive by default; activate via app state.
    allowed_origins: list[str] | None = getattr(
        websocket.app.state, "exo_ws_allowed_origins", None  # type: ignore[union-attr]
    )
    if allowed_origins:
        origin = websocket.headers.get("origin", "")
        if origin not in allowed_origins:
            await websocket.close(code=1008, reason="Origin not allowed")
            return

    await websocket.accept()

    try:
        data = await websocket.receive_json()
    except (WebSocketDisconnect, ValueError, KeyError):
        return

    message: str = data.get("message", "")
    agent_name: str | None = data.get("agent_name")

    if not message:
        await websocket.send_json({"type": "error", "error": "Empty message"})
        await websocket.close()
        return

    agent = _resolve_agent(websocket.app.state, agent_name)  # type: ignore[union-attr]
    if agent is None:
        detail = "No agents registered" if not agent_name else f"Agent '{agent_name}' not found"
        await websocket.send_json({"type": "error", "error": detail})
        await websocket.close()
        return

    try:
        async for payload in _iter_events(agent, message):
            await websocket.send_json(payload)
        # Send the "done" marker and close inside the try so that a
        # WebSocketDisconnect during these final sends is caught cleanly.
        await websocket.send_json({"type": "done"})
        await websocket.close()
    except WebSocketDisconnect:
        return


async def _sse_iter(agent: Any, message: str) -> AsyncIterator[str]:
    """Yield SSE-formatted lines from agent stream events."""
    async for payload in _iter_events(agent, message):
        yield f"data: {json.dumps(payload)}\n\n"
    yield "data: [DONE]\n\n"


@stream_router.get("/stream")
async def sse_stream(
    req: Request,
    message: str = Query(..., description="The user message"),
    agent_name: str | None = Query(None, description="Agent to invoke"),
) -> StreamingResponse:
    """SSE fallback endpoint for non-WebSocket clients.

    Returns ``text/event-stream`` with JSON payloads matching
    the WebSocket protocol, ending with ``data: [DONE]``.
    Returns HTTP 404 (before the stream begins) when the agent is not found.
    """
    from fastapi import HTTPException  # local import avoids circular deps

    agent = _resolve_agent(req.app.state, agent_name)
    if agent is None:
        raise HTTPException(status_code=404, detail="Agent not found")

    return StreamingResponse(
        _sse_iter(agent, message),
        media_type="text/event-stream",
    )
