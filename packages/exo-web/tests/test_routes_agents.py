"""Tests for the agents CRUD and sub-agent REST API."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import FastAPI
from starlette.testclient import TestClient

from exo_web.routes.agents import router
from exo_web.routes.auth import get_current_user

# ---------------------------------------------------------------------------
# Fixtures / constants
# ---------------------------------------------------------------------------

FAKE_USER = {
    "id": "user-001",
    "email": "test@example.com",
    "role": "developer",
    "created_at": "2025-01-01T00:00:00",
}

FAKE_AGENT_ROW = {
    "id": "agent-001",
    "name": "Test Agent",
    "description": "",
    "instructions": "You are helpful.",
    "model_provider": "openai",
    "model_name": "gpt-4o",
    "temperature": 0.7,
    "max_tokens": 1024,
    "max_steps": 5,
    "output_type_json": "{}",
    "tools_json": "[]",
    "handoffs_json": "[]",
    "hooks_json": "{}",
    "knowledge_base_ids": "[]",
    "persona_role": "",
    "persona_goal": "",
    "persona_backstory": "",
    "autonomous_mode": 0,
    "context_automation_level": "copilot",
    "context_max_tokens_per_step": None,
    "context_max_total_tokens": None,
    "context_memory_type": "conversation",
    "context_workspace_enabled": 0,
    "project_id": "proj-001",
    "user_id": "user-001",
    "created_at": "2025-01-01 00:00:00",
    "updated_at": "2025-01-01 00:00:00",
}

PROJECT_ROW = {"id": "proj-001"}

SUB_AGENT_REL_ROW = {
    "id": "rel-001",
    "supervisor_id": "agent-001",
    "sub_agent_id": "agent-002",
    "sub_agent_name": "Helper",
    "sub_agent_description": "A helper agent",
    "relationship_type": "delegation",
    "routing_rule_json": "{}",
    "created_at": "2025-01-01 00:00:00",
    "updated_at": "2025-01-01 00:00:00",
}


# ---------------------------------------------------------------------------
# Apps
# ---------------------------------------------------------------------------

_app = FastAPI()
_app.include_router(router)
_app.dependency_overrides[get_current_user] = lambda: FAKE_USER

_app_no_auth = FastAPI()
_app_no_auth.include_router(router)


# ---------------------------------------------------------------------------
# Fake row helpers
# ---------------------------------------------------------------------------


class _FakeRow:
    """Minimal aiosqlite.Row-compatible object."""

    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def keys(self) -> Any:
        return self._data.keys()

    def __iter__(self) -> Any:
        return iter(self._data.items())

    def __contains__(self, key: str) -> bool:
        return key in self._data


def _fake_row(data: dict[str, Any] | None) -> _FakeRow | None:
    return _FakeRow(data) if data is not None else None


def _make_get_db(*row_sequence: dict[str, Any] | None):
    """Return a patched get_db that yields mock rows in order across calls."""
    rows = list(row_sequence)
    call_index = 0

    @asynccontextmanager
    async def _mock_get_db():
        nonlocal call_index
        mock_db = AsyncMock()

        async def _execute(sql: str, params: Any = ()) -> AsyncMock:
            nonlocal call_index
            cursor = AsyncMock()
            row = rows[call_index] if call_index < len(rows) else None
            cursor.fetchone = AsyncMock(return_value=_fake_row(row))
            cursor.fetchall = AsyncMock(return_value=[])
            cursor.rowcount = 1
            call_index += 1
            return cursor

        mock_db.execute = _execute
        mock_db.commit = AsyncMock()
        yield mock_db

    return _mock_get_db


def _make_get_db_multi(fetches: list[dict[str, Any]]):
    """get_db whose successive execute() calls return the given fetchone/fetchall.

    Each entry: {"one": <row or None>, "all": <list of rows or None>}.
    """
    idx = 0

    @asynccontextmanager
    async def _mock():
        nonlocal idx
        db = AsyncMock()

        async def _execute(sql: str, params: Any = ()) -> AsyncMock:
            nonlocal idx
            cursor = AsyncMock()
            f = fetches[idx] if idx < len(fetches) else {}
            cursor.fetchone = AsyncMock(return_value=_fake_row(f.get("one")))
            cursor.fetchall = AsyncMock(return_value=[_fake_row(r) for r in (f.get("all") or [])])
            cursor.rowcount = 1
            idx += 1
            return cursor

        db.execute = _execute
        db.commit = AsyncMock()
        yield db

    return _mock


def _no_op_config_version():
    return patch("exo_web.routes.agents.create_config_version", AsyncMock())


def _no_op_audit():
    return patch("exo_web.routes.agents.audit_log", AsyncMock())


# ---------------------------------------------------------------------------
# GET /api/v1/agents
# ---------------------------------------------------------------------------


class TestListAgents:
    def test_success(self) -> None:
        mock_result = MagicMock()
        mock_result.model_dump.return_value = {"items": [], "next_cursor": None}
        with (
            patch("exo_web.routes.agents.get_db", _make_get_db()),
            patch("exo_web.routes.agents.paginate", AsyncMock(return_value=mock_result)),
        ):
            client = TestClient(_app, raise_server_exceptions=False)
            resp = client.get("/api/v1/agents")
        assert resp.status_code == 200
        assert resp.json() == {"items": [], "next_cursor": None}

    def test_unauthenticated(self) -> None:
        with patch("exo_web.routes.auth.get_db", _make_get_db(None)):
            client = TestClient(_app_no_auth, raise_server_exceptions=False)
            resp = client.get("/api/v1/agents")
        assert resp.status_code == 401


# ---------------------------------------------------------------------------
# POST /api/v1/agents
# ---------------------------------------------------------------------------


class TestCreateAgent:
    def test_success(self) -> None:
        mock_get_db = _make_get_db(
            PROJECT_ROW,  # project ownership check
            None,  # INSERT
            FAKE_AGENT_ROW,  # SELECT new row
        )
        with (
            patch("exo_web.routes.agents.get_db", mock_get_db),
            _no_op_config_version(),
            _no_op_audit(),
        ):
            client = TestClient(_app, raise_server_exceptions=False)
            resp = client.post(
                "/api/v1/agents",
                json={"name": "Test Agent", "project_id": "proj-001"},
            )
        assert resp.status_code == 201
        data = resp.json()
        assert data["id"] == "agent-001"
        assert data["autonomous_mode"] is False

    def test_project_not_found(self) -> None:
        mock_get_db = _make_get_db(None)  # project lookup returns nothing
        with (
            patch("exo_web.routes.agents.get_db", mock_get_db),
            _no_op_config_version(),
            _no_op_audit(),
        ):
            client = TestClient(_app, raise_server_exceptions=False)
            resp = client.post(
                "/api/v1/agents",
                json={"name": "Test Agent", "project_id": "missing"},
            )
        assert resp.status_code == 404
        assert resp.json()["detail"] == "Project not found"

    def test_missing_name(self) -> None:
        with patch("exo_web.routes.agents.get_db", _make_get_db()):
            client = TestClient(_app, raise_server_exceptions=False)
            resp = client.post("/api/v1/agents", json={"project_id": "proj-001"})
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# GET /api/v1/agents/{agent_id}
# ---------------------------------------------------------------------------


class TestGetAgent:
    def test_found(self) -> None:
        with patch("exo_web.routes.agents.get_db", _make_get_db(FAKE_AGENT_ROW)):
            client = TestClient(_app, raise_server_exceptions=False)
            resp = client.get("/api/v1/agents/agent-001")
        assert resp.status_code == 200
        assert resp.json()["id"] == "agent-001"

    def test_not_found(self) -> None:
        with patch("exo_web.routes.agents.get_db", _make_get_db(None)):
            client = TestClient(_app, raise_server_exceptions=False)
            resp = client.get("/api/v1/agents/missing")
        assert resp.status_code == 404
        assert resp.json()["detail"] == "Agent not found"

    def test_wrong_user(self) -> None:
        # Ownership query filters on user_id, so a non-matching agent returns None.
        with patch("exo_web.routes.agents.get_db", _make_get_db(None)):
            client = TestClient(_app, raise_server_exceptions=False)
            resp = client.get("/api/v1/agents/agent-001")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# PUT /api/v1/agents/{agent_id}
# ---------------------------------------------------------------------------


class TestUpdateAgent:
    def test_success(self) -> None:
        mock_get_db = _make_get_db(
            FAKE_AGENT_ROW,  # _verify_ownership
            None,  # UPDATE
            FAKE_AGENT_ROW,  # SELECT after update
        )
        with (
            patch("exo_web.routes.agents.get_db", mock_get_db),
            _no_op_config_version(),
        ):
            client = TestClient(_app, raise_server_exceptions=False)
            resp = client.put(
                "/api/v1/agents/agent-001",
                json={"name": "Renamed"},
            )
        assert resp.status_code == 200
        assert resp.json()["id"] == "agent-001"

    def test_no_fields(self) -> None:
        with patch("exo_web.routes.agents.get_db", _make_get_db()):
            client = TestClient(_app, raise_server_exceptions=False)
            resp = client.put("/api/v1/agents/agent-001", json={})
        assert resp.status_code == 422
        assert resp.json()["detail"] == "No fields to update"

    def test_not_found(self) -> None:
        with (
            patch("exo_web.routes.agents.get_db", _make_get_db(None)),
            _no_op_config_version(),
        ):
            client = TestClient(_app, raise_server_exceptions=False)
            resp = client.put("/api/v1/agents/missing", json={"name": "Renamed"})
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# DELETE /api/v1/agents/{agent_id}
# ---------------------------------------------------------------------------


class TestDeleteAgent:
    def test_success(self) -> None:
        mock_get_db = _make_get_db(
            FAKE_AGENT_ROW,  # _verify_ownership
            None,  # DELETE
        )
        with (
            patch("exo_web.routes.agents.get_db", mock_get_db),
            _no_op_audit(),
        ):
            client = TestClient(_app, raise_server_exceptions=False)
            resp = client.delete("/api/v1/agents/agent-001")
        assert resp.status_code == 204

    def test_not_found(self) -> None:
        with patch("exo_web.routes.agents.get_db", _make_get_db(None)):
            client = TestClient(_app, raise_server_exceptions=False)
            resp = client.delete("/api/v1/agents/missing")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# GET /api/v1/agents/{agent_id}/sub-agents
# ---------------------------------------------------------------------------


class TestListSubAgents:
    def test_success_empty(self) -> None:
        mock_get_db = _make_get_db_multi(
            [
                {"one": FAKE_AGENT_ROW},  # _verify_ownership
                {"all": []},  # _get_sub_agents
            ]
        )
        with patch("exo_web.routes.agents.get_db", mock_get_db):
            client = TestClient(_app, raise_server_exceptions=False)
            resp = client.get("/api/v1/agents/agent-001/sub-agents")
        assert resp.status_code == 200
        assert resp.json() == []


# ---------------------------------------------------------------------------
# POST /api/v1/agents/{agent_id}/sub-agents
# ---------------------------------------------------------------------------


class TestAddSubAgent:
    def test_success(self) -> None:
        mock_get_db = _make_get_db_multi(
            [
                {"one": FAKE_AGENT_ROW},  # supervisor ownership
                {"one": FAKE_AGENT_ROW},  # sub-agent ownership
                {"one": None},  # duplicate check (none)
                {},  # INSERT
                {"one": SUB_AGENT_REL_ROW},  # joined SELECT
            ]
        )
        with patch("exo_web.routes.agents.get_db", mock_get_db):
            client = TestClient(_app, raise_server_exceptions=False)
            resp = client.post(
                "/api/v1/agents/agent-001/sub-agents",
                json={"sub_agent_id": "agent-002"},
            )
        assert resp.status_code == 201
        assert resp.json()["sub_agent_id"] == "agent-002"

    def test_self_reference(self) -> None:
        with patch("exo_web.routes.agents.get_db", _make_get_db()):
            client = TestClient(_app, raise_server_exceptions=False)
            resp = client.post(
                "/api/v1/agents/agent-001/sub-agents",
                json={"sub_agent_id": "agent-001"},
            )
        assert resp.status_code == 422
        assert resp.json()["detail"] == "Agent cannot be its own sub-agent"

    def test_duplicate(self) -> None:
        mock_get_db = _make_get_db_multi(
            [
                {"one": FAKE_AGENT_ROW},  # supervisor ownership
                {"one": FAKE_AGENT_ROW},  # sub-agent ownership
                {"one": SUB_AGENT_REL_ROW},  # duplicate check (exists)
            ]
        )
        with patch("exo_web.routes.agents.get_db", mock_get_db):
            client = TestClient(_app, raise_server_exceptions=False)
            resp = client.post(
                "/api/v1/agents/agent-001/sub-agents",
                json={"sub_agent_id": "agent-002"},
            )
        assert resp.status_code == 409
        assert resp.json()["detail"] == "Sub-agent relationship already exists"


# ---------------------------------------------------------------------------
# DELETE /api/v1/agents/{agent_id}/sub-agents/{sub_agent_id}
# ---------------------------------------------------------------------------


class TestRemoveSubAgent:
    def test_success(self) -> None:
        mock_get_db = _make_get_db_multi(
            [
                {"one": FAKE_AGENT_ROW},  # _verify_ownership
                {"one": SUB_AGENT_REL_ROW},  # relationship lookup
                {},  # DELETE
            ]
        )
        with patch("exo_web.routes.agents.get_db", mock_get_db):
            client = TestClient(_app, raise_server_exceptions=False)
            resp = client.delete("/api/v1/agents/agent-001/sub-agents/agent-002")
        assert resp.status_code == 204

    def test_not_found(self) -> None:
        mock_get_db = _make_get_db_multi(
            [
                {"one": FAKE_AGENT_ROW},  # _verify_ownership
                {"one": None},  # relationship lookup (missing)
            ]
        )
        with patch("exo_web.routes.agents.get_db", mock_get_db):
            client = TestClient(_app, raise_server_exceptions=False)
            resp = client.delete("/api/v1/agents/agent-001/sub-agents/agent-002")
        assert resp.status_code == 404
        assert resp.json()["detail"] == "Sub-agent relationship not found"


# ---------------------------------------------------------------------------
# GET /api/v1/agents/{agent_id}/effective-config
# ---------------------------------------------------------------------------


class TestEffectiveConfig:
    def test_success_merges(self) -> None:
        agent_with_tools = {**FAKE_AGENT_ROW, "tools_json": '[{"id": "web_search"}]'}
        mock_get_db = _make_get_db_multi(
            [
                {"one": agent_with_tools},  # _verify_ownership
                {"all": [SUB_AGENT_REL_ROW]},  # _get_sub_agents
            ]
        )
        with patch("exo_web.routes.agents.get_db", mock_get_db):
            client = TestClient(_app, raise_server_exceptions=False)
            resp = client.get("/api/v1/agents/agent-001/effective-config")
        assert resp.status_code == 200
        data = resp.json()
        # Own tool plus the generated sub-agent delegation tool.
        import json as _json

        tools = _json.loads(data["tools_json"])
        assert {"id": "web_search"} in tools
        assert any(t.get("type") == "sub_agent" for t in tools)
        # Instructions augmented with sub-agent prompt section.
        assert "Available Sub-Agents" in data["instructions"]
        assert len(data["sub_agents"]) == 1
