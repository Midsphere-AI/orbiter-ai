"""Tests for exo_server.sessions — session management CRUD routes."""

from __future__ import annotations

from typing import Any

from httpx import ASGITransport, AsyncClient

from exo_server.app import create_app
from exo_server.sessions import (
    AppendMessageRequest,
    CreateSessionRequest,
    Session,
    SessionMessage,
    SessionSummary,
    UpdateSessionRequest,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_client(app: Any) -> AsyncClient:
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class TestSessionMessage:
    def test_defaults(self) -> None:
        msg = SessionMessage(role="user", content="hello")
        assert msg.role == "user"
        assert msg.content == "hello"
        assert msg.timestamp > 0

    def test_custom_timestamp(self) -> None:
        msg = SessionMessage(role="assistant", content="hi", timestamp=1.0)
        assert msg.timestamp == 1.0


class TestSession:
    def test_defaults(self) -> None:
        s = Session()
        assert len(s.id) == 16
        assert s.agent_name == ""
        assert s.title == ""
        assert s.messages == []
        assert s.created_at > 0

    def test_with_fields(self) -> None:
        s = Session(id="abc", agent_name="bot", title="Chat 1")
        assert s.id == "abc"
        assert s.agent_name == "bot"
        assert s.title == "Chat 1"


class TestRequestModels:
    def test_create_request(self) -> None:
        req = CreateSessionRequest(agent_name="bot", title="Test")
        assert req.agent_name == "bot"

    def test_update_request_partial(self) -> None:
        req = UpdateSessionRequest(title="New Title")
        assert req.title == "New Title"
        assert req.agent_name is None

    def test_append_message_request(self) -> None:
        req = AppendMessageRequest(role="user", content="hello")
        assert req.role == "user"

    def test_session_summary(self) -> None:
        s = SessionSummary(
            id="x", agent_name="bot", title="t", message_count=3, created_at=1.0, updated_at=2.0
        )
        assert s.message_count == 3


# ---------------------------------------------------------------------------
# Create session
# ---------------------------------------------------------------------------


class TestCreateSession:
    async def test_create_returns_201(self) -> None:
        app = create_app()
        async with _build_client(app) as client:
            resp = await client.post("/sessions", json={"agent_name": "bot", "title": "Chat"})
        assert resp.status_code == 201
        data = resp.json()
        assert data["agent_name"] == "bot"
        assert data["title"] == "Chat"
        assert "id" in data
        assert data["messages"] == []

    async def test_create_defaults(self) -> None:
        app = create_app()
        async with _build_client(app) as client:
            resp = await client.post("/sessions", json={})
        assert resp.status_code == 201
        assert resp.json()["agent_name"] == ""

    async def test_create_multiple(self) -> None:
        app = create_app()
        async with _build_client(app) as client:
            r1 = await client.post("/sessions", json={"title": "A"})
            r2 = await client.post("/sessions", json={"title": "B"})
        assert r1.json()["id"] != r2.json()["id"]


# ---------------------------------------------------------------------------
# List sessions
# ---------------------------------------------------------------------------


class TestListSessions:
    async def test_empty_list(self) -> None:
        app = create_app()
        async with _build_client(app) as client:
            resp = await client.get("/sessions")
        assert resp.status_code == 200
        assert resp.json() == []

    async def test_lists_created_sessions(self) -> None:
        app = create_app()
        async with _build_client(app) as client:
            await client.post("/sessions", json={"title": "A"})
            await client.post("/sessions", json={"title": "B"})
            resp = await client.get("/sessions")
        data = resp.json()
        assert len(data) == 2
        assert "message_count" in data[0]

    async def test_newest_first(self) -> None:
        """Sessions must be listed newest-first with a strictly greater created_at."""
        app = create_app()
        async with _build_client(app) as client:
            await client.post("/sessions", json={"title": "First"})
            await client.post("/sessions", json={"title": "Second"})
            resp = await client.get("/sessions")
        data = resp.json()
        # _now() assigns strictly increasing timestamps so this is always strict.
        assert data[0]["created_at"] > data[1]["created_at"]


# ---------------------------------------------------------------------------
# Get session
# ---------------------------------------------------------------------------


class TestGetSession:
    async def test_get_existing(self) -> None:
        app = create_app()
        async with _build_client(app) as client:
            create_resp = await client.post("/sessions", json={"title": "X"})
            sid = create_resp.json()["id"]
            resp = await client.get(f"/sessions/{sid}")
        assert resp.status_code == 200
        assert resp.json()["title"] == "X"

    async def test_get_not_found(self) -> None:
        app = create_app()
        async with _build_client(app) as client:
            resp = await client.get("/sessions/nonexistent")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Update session
# ---------------------------------------------------------------------------


class TestUpdateSession:
    async def test_update_title(self) -> None:
        app = create_app()
        async with _build_client(app) as client:
            create_resp = await client.post("/sessions", json={"title": "Old"})
            sid = create_resp.json()["id"]
            resp = await client.patch(f"/sessions/{sid}", json={"title": "New"})
        assert resp.status_code == 200
        assert resp.json()["title"] == "New"

    async def test_update_agent_name(self) -> None:
        app = create_app()
        async with _build_client(app) as client:
            create_resp = await client.post("/sessions", json={})
            sid = create_resp.json()["id"]
            resp = await client.patch(f"/sessions/{sid}", json={"agent_name": "bot2"})
        assert resp.json()["agent_name"] == "bot2"

    async def test_update_not_found(self) -> None:
        app = create_app()
        async with _build_client(app) as client:
            resp = await client.patch("/sessions/nope", json={"title": "X"})
        assert resp.status_code == 404

    async def test_partial_update(self) -> None:
        app = create_app()
        async with _build_client(app) as client:
            create_resp = await client.post(
                "/sessions", json={"title": "Keep", "agent_name": "bot"}
            )
            sid = create_resp.json()["id"]
            resp = await client.patch(f"/sessions/{sid}", json={"title": "Changed"})
        data = resp.json()
        assert data["title"] == "Changed"
        assert data["agent_name"] == "bot"  # unchanged


# ---------------------------------------------------------------------------
# Delete session
# ---------------------------------------------------------------------------


class TestDeleteSession:
    async def test_delete_existing(self) -> None:
        app = create_app()
        async with _build_client(app) as client:
            create_resp = await client.post("/sessions", json={"title": "Del"})
            sid = create_resp.json()["id"]
            del_resp = await client.delete(f"/sessions/{sid}")
            assert del_resp.status_code == 204
            # Verify it's gone
            get_resp = await client.get(f"/sessions/{sid}")
            assert get_resp.status_code == 404

    async def test_delete_not_found(self) -> None:
        app = create_app()
        async with _build_client(app) as client:
            resp = await client.delete("/sessions/nope")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Messages
# ---------------------------------------------------------------------------


class TestAppendMessage:
    async def test_append_message(self) -> None:
        app = create_app()
        async with _build_client(app) as client:
            create_resp = await client.post("/sessions", json={})
            sid = create_resp.json()["id"]
            resp = await client.post(
                f"/sessions/{sid}/messages", json={"role": "user", "content": "hello"}
            )
        assert resp.status_code == 201
        data = resp.json()
        assert data["role"] == "user"
        assert data["content"] == "hello"
        assert data["timestamp"] > 0

    async def test_append_multiple(self) -> None:
        app = create_app()
        async with _build_client(app) as client:
            create_resp = await client.post("/sessions", json={})
            sid = create_resp.json()["id"]
            await client.post(f"/sessions/{sid}/messages", json={"role": "user", "content": "hi"})
            await client.post(
                f"/sessions/{sid}/messages", json={"role": "assistant", "content": "hello!"}
            )
            resp = await client.get(f"/sessions/{sid}/messages")
        msgs = resp.json()
        assert len(msgs) == 2
        assert msgs[0]["role"] == "user"
        assert msgs[1]["role"] == "assistant"

    async def test_append_to_nonexistent(self) -> None:
        app = create_app()
        async with _build_client(app) as client:
            resp = await client.post(
                "/sessions/nope/messages", json={"role": "user", "content": "hello"}
            )
        assert resp.status_code == 404


class TestListMessages:
    async def test_list_empty(self) -> None:
        app = create_app()
        async with _build_client(app) as client:
            create_resp = await client.post("/sessions", json={})
            sid = create_resp.json()["id"]
            resp = await client.get(f"/sessions/{sid}/messages")
        assert resp.status_code == 200
        assert resp.json() == []

    async def test_list_not_found(self) -> None:
        app = create_app()
        async with _build_client(app) as client:
            resp = await client.get("/sessions/nope/messages")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Full lifecycle
# ---------------------------------------------------------------------------


class TestSessionLifecycle:
    async def test_full_lifecycle(self) -> None:
        """Create → update → add messages → list → get → delete."""
        app = create_app()
        async with _build_client(app) as client:
            # Create
            r = await client.post("/sessions", json={"agent_name": "bot", "title": "Chat 1"})
            assert r.status_code == 201
            sid = r.json()["id"]

            # Update title
            r = await client.patch(f"/sessions/{sid}", json={"title": "Renamed"})
            assert r.json()["title"] == "Renamed"

            # Add messages
            await client.post(f"/sessions/{sid}/messages", json={"role": "user", "content": "hi"})
            await client.post(
                f"/sessions/{sid}/messages",
                json={"role": "assistant", "content": "hello!"},
            )

            # List sessions — should show message_count=2
            r = await client.get("/sessions")
            summaries = r.json()
            assert len(summaries) == 1
            assert summaries[0]["message_count"] == 2

            # Get full session
            r = await client.get(f"/sessions/{sid}")
            session = r.json()
            assert len(session["messages"]) == 2
            assert session["agent_name"] == "bot"

            # Delete
            r = await client.delete(f"/sessions/{sid}")
            assert r.status_code == 204

            # Verify gone
            r = await client.get("/sessions")
            assert r.json() == []

    async def test_updates_timestamp(self) -> None:
        """Updating or appending messages should update `updated_at`."""
        app = create_app()
        async with _build_client(app) as client:
            r = await client.post("/sessions", json={})
            sid = r.json()["id"]
            created_at = r.json()["updated_at"]

            # Update — _now() guarantees a strictly greater value.
            r = await client.patch(f"/sessions/{sid}", json={"title": "X"})
            assert r.json()["updated_at"] > created_at
