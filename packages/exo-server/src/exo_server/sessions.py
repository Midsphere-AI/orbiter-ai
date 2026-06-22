"""Session management API routes.

Provides CRUD endpoints for chat sessions. Each session groups a conversation
with a specific agent, storing messages exchanged during the interaction.

Usage::

    from exo_server.sessions import session_router

    app = FastAPI()
    app.include_router(session_router)
"""

from __future__ import annotations

import time
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class SessionMessage(BaseModel):
    """A single message within a session."""

    model_config = ConfigDict(frozen=True)

    role: str
    content: str
    timestamp: float = Field(default_factory=time.time)


class Session(BaseModel):
    """A chat session grouping a conversation with an agent.

    ``Session`` is intentionally mutable (no ``frozen=True``) so that
    endpoints can update ``title``, ``agent_name``, ``updated_at``, and
    ``messages`` in place using normal attribute assignment.  Every mutation
    goes through the route handlers in this module; the model is never
    shared across threads so there is no concurrency hazard.
    """

    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:16])
    agent_name: str = ""
    title: str = ""
    messages: list[SessionMessage] = Field(default_factory=list)
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)


class CreateSessionRequest(BaseModel):
    """Request body for creating a new session."""

    model_config = ConfigDict(frozen=True)

    agent_name: str = ""
    title: str = ""


class UpdateSessionRequest(BaseModel):
    """Request body for updating a session."""

    model_config = ConfigDict(frozen=True)

    title: str | None = None
    agent_name: str | None = None


class AppendMessageRequest(BaseModel):
    """Request body for appending a message to a session."""

    model_config = ConfigDict(frozen=True)

    role: str
    content: str


class SessionSummary(BaseModel):
    """Lightweight session info for list responses."""

    model_config = ConfigDict(frozen=True)

    id: str
    agent_name: str
    title: str
    message_count: int
    created_at: float
    updated_at: float


# ---------------------------------------------------------------------------
# In-memory session store (per-app via state)
# ---------------------------------------------------------------------------

_SESSIONS_KEY = "exo_sessions"

#: Monotonically incrementing counter used to give distinct ``created_at``
#: / ``updated_at`` values even on fast CI where ``time.time()`` resolution
#: is coarse.  Only used internally by ``_now()``.
_tick: int = 0


def _now() -> float:
    """Return a strictly increasing timestamp.

    Combines ``time.time()`` with a monotonic tiebreaker so that two
    sessions created in the same clock tick still have distinct
    ``created_at`` values and the "newest first" sort is deterministic.
    """
    global _tick
    _tick += 1
    return time.time() + _tick * 1e-9


def _get_store(state: Any) -> dict[str, Session]:
    """Retrieve the sessions dict from app state."""
    return getattr(state, _SESSIONS_KEY, {})


def _set_store(state: Any, store: dict[str, Session]) -> None:
    """Persist the sessions dict to app state."""
    setattr(state, _SESSIONS_KEY, store)


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

session_router = APIRouter(prefix="/sessions", tags=["sessions"])


@session_router.post("", response_model=Session, status_code=201)
async def create_session(req: Request, body: CreateSessionRequest) -> Any:
    """Create a new chat session."""
    store = _get_store(req.app.state)
    now = _now()
    session = Session(agent_name=body.agent_name, title=body.title, created_at=now, updated_at=now)
    store[session.id] = session
    _set_store(req.app.state, store)
    return session


@session_router.get("", response_model=list[SessionSummary])
async def list_sessions(req: Request) -> Any:
    """List all sessions, newest first."""
    store = _get_store(req.app.state)
    summaries = [
        SessionSummary(
            id=s.id,
            agent_name=s.agent_name,
            title=s.title,
            message_count=len(s.messages),
            created_at=s.created_at,
            updated_at=s.updated_at,
        )
        for s in store.values()
    ]
    return sorted(summaries, key=lambda s: s.created_at, reverse=True)


@session_router.get("/{session_id}", response_model=Session)
async def get_session(req: Request, session_id: str) -> Any:
    """Retrieve a single session by ID."""
    store = _get_store(req.app.state)
    session = store.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
    return session


@session_router.patch("/{session_id}", response_model=Session)
async def update_session(req: Request, session_id: str, body: UpdateSessionRequest) -> Any:
    """Update session metadata (title, agent_name)."""
    store = _get_store(req.app.state)
    session = store.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
    if body.title is not None:
        session.title = body.title
    if body.agent_name is not None:
        session.agent_name = body.agent_name
    session.updated_at = _now()
    return session


@session_router.delete("/{session_id}", status_code=204)
async def delete_session(req: Request, session_id: str) -> None:
    """Delete a session."""
    store = _get_store(req.app.state)
    if session_id not in store:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
    del store[session_id]


@session_router.post("/{session_id}/messages", response_model=SessionMessage, status_code=201)
async def append_message(req: Request, session_id: str, body: AppendMessageRequest) -> Any:
    """Append a message to a session's conversation history."""
    store = _get_store(req.app.state)
    session = store.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
    msg = SessionMessage(role=body.role, content=body.content)
    session.messages.append(msg)
    session.updated_at = _now()
    return msg


@session_router.get("/{session_id}/messages", response_model=list[SessionMessage])
async def list_messages(req: Request, session_id: str) -> Any:
    """List all messages in a session."""
    store = _get_store(req.app.state)
    session = store.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
    return session.messages
