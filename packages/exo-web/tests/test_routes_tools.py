"""Tests for the tools route module.

Covers:
- Generic CRUD (list, create, get, update, delete)
- Built-in tools always merged into listings (code_interpreter)
- Custom tool endpoints (create from code, update, test execution, schema gen)
- Visual schema save
- Validation (invalid category/tool_type, missing function, no fields)
- Auth (401 when no session, 404 when project/tool missing)
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import AsyncMock, patch

from fastapi import FastAPI
from starlette.testclient import TestClient

from exo_web.routes.auth import get_current_user
from exo_web.routes.tools import router

# ---------------------------------------------------------------------------
# Minimal test app
# ---------------------------------------------------------------------------

_app = FastAPI()
_app.include_router(router)

FAKE_USER = {
    "id": "user-001",
    "email": "test@example.com",
    "role": "developer",
    "created_at": "2025-01-01T00:00:00",
}
_app.dependency_overrides[get_current_user] = lambda: FAKE_USER

# App without auth override to exercise the 401 path.
_app_no_auth = FastAPI()
_app_no_auth.include_router(router)

# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------

FAKE_TOOL_ROW: dict[str, Any] = {
    "id": "tool-001",
    "name": "my_tool",
    "description": "Does stuff",
    "category": "custom",
    "schema_json": '{"type": "object", "properties": {}}',
    "code": "",
    "tool_type": "function",
    "usage_count": 0,
    "project_id": "proj-001",
    "user_id": "user-001",
    "created_at": "2025-01-01 00:00:00",
}

PROJECT_ROW: dict[str, Any] = {"id": "proj-001"}

VALID_CODE = '''
def greet(name: str, count: int = 1) -> str:
    """Greet someone.

    Args:
        name: The person to greet.
        count: How many times.
    """
    return f"Hello {name}" * count
'''


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


def _make_get_db(*row_sequence: dict[str, Any] | list[dict[str, Any]] | None):
    """Return a patched get_db; each call to db.execute gets the next row/list."""
    items = list(row_sequence)
    call_idx = 0

    @asynccontextmanager
    async def _mock():
        nonlocal call_idx
        db = AsyncMock()

        async def _execute(sql: str, params: Any = ()) -> AsyncMock:
            nonlocal call_idx
            cursor = AsyncMock()
            item = items[call_idx] if call_idx < len(items) else None
            call_idx += 1
            if isinstance(item, list):
                cursor.fetchall = AsyncMock(return_value=[_FakeRow(r) for r in item])
                cursor.fetchone = AsyncMock(return_value=_fake_row(item[0]) if item else None)
            else:
                cursor.fetchone = AsyncMock(return_value=_fake_row(item))
                cursor.fetchall = AsyncMock(return_value=[_FakeRow(item)] if item else [])
            cursor.rowcount = 1
            return cursor

        db.execute = _execute
        db.commit = AsyncMock()
        yield db

    return _mock


# ---------------------------------------------------------------------------
# Generic CRUD: list
# ---------------------------------------------------------------------------


class TestListTools:
    def test_returns_builtin_plus_db_tools(self):
        # fetchall returns no DB rows; builtin must still be present.
        mock_db = _make_get_db([])
        with patch("exo_web.routes.tools.get_db", mock_db):
            client = TestClient(_app)
            resp = client.get("/api/v1/tools")
        assert resp.status_code == 200
        data = resp.json()
        ids = [t["id"] for t in data]
        assert "builtin:code_interpreter" in ids

    def test_includes_db_tools(self):
        mock_db = _make_get_db([FAKE_TOOL_ROW])
        with patch("exo_web.routes.tools.get_db", mock_db):
            client = TestClient(_app)
            resp = client.get("/api/v1/tools")
        assert resp.status_code == 200
        ids = [t["id"] for t in resp.json()]
        assert "builtin:code_interpreter" in ids
        assert "tool-001" in ids

    def test_filtered_by_category_excludes_builtin(self):
        # code_interpreter is in category 'code'; filtering by 'custom' drops it.
        mock_db = _make_get_db([])
        with patch("exo_web.routes.tools.get_db", mock_db):
            client = TestClient(_app)
            resp = client.get("/api/v1/tools", params={"category": "custom"})
        assert resp.status_code == 200
        ids = [t["id"] for t in resp.json()]
        assert "builtin:code_interpreter" not in ids

    def test_filtered_by_category_includes_builtin(self):
        mock_db = _make_get_db([])
        with patch("exo_web.routes.tools.get_db", mock_db):
            client = TestClient(_app)
            resp = client.get("/api/v1/tools", params={"category": "code"})
        assert resp.status_code == 200
        ids = [t["id"] for t in resp.json()]
        assert "builtin:code_interpreter" in ids

    def test_unauthenticated_returns_401(self):
        # No session row returned, so get_current_user raises 401.
        with patch("exo_web.routes.tools.get_db", _make_get_db()):
            client = TestClient(_app_no_auth, raise_server_exceptions=False)
            resp = client.get("/api/v1/tools")
        assert resp.status_code == 401


# ---------------------------------------------------------------------------
# Generic CRUD: create
# ---------------------------------------------------------------------------


class TestCreateTool:
    def test_creates_and_returns_201(self):
        # Project lookup, INSERT, SELECT created row.
        mock_db = _make_get_db(PROJECT_ROW, None, FAKE_TOOL_ROW)
        with patch("exo_web.routes.tools.get_db", mock_db):
            client = TestClient(_app)
            resp = client.post(
                "/api/v1/tools",
                json={"name": "my_tool", "project_id": "proj-001", "category": "custom"},
            )
        assert resp.status_code == 201
        assert resp.json()["id"] == "tool-001"

    def test_invalid_category_returns_422(self):
        client = TestClient(_app, raise_server_exceptions=False)
        resp = client.post(
            "/api/v1/tools",
            json={"name": "x", "project_id": "proj-001", "category": "bogus"},
        )
        assert resp.status_code == 422

    def test_invalid_tool_type_returns_422(self):
        client = TestClient(_app, raise_server_exceptions=False)
        resp = client.post(
            "/api/v1/tools",
            json={
                "name": "x",
                "project_id": "proj-001",
                "category": "custom",
                "tool_type": "bogus",
            },
        )
        assert resp.status_code == 422

    def test_project_not_found_returns_404(self):
        mock_db = _make_get_db(None)
        with patch("exo_web.routes.tools.get_db", mock_db):
            client = TestClient(_app, raise_server_exceptions=False)
            resp = client.post(
                "/api/v1/tools",
                json={"name": "x", "project_id": "missing", "category": "custom"},
            )
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Generic CRUD: get
# ---------------------------------------------------------------------------


class TestGetTool:
    def test_found_returns_200(self):
        mock_db = _make_get_db(FAKE_TOOL_ROW)
        with patch("exo_web.routes.tools.get_db", mock_db):
            client = TestClient(_app)
            resp = client.get("/api/v1/tools/tool-001")
        assert resp.status_code == 200
        assert resp.json()["name"] == "my_tool"

    def test_not_found_returns_404(self):
        mock_db = _make_get_db(None)
        with patch("exo_web.routes.tools.get_db", mock_db):
            client = TestClient(_app, raise_server_exceptions=False)
            resp = client.get("/api/v1/tools/missing")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Generic CRUD: update
# ---------------------------------------------------------------------------


class TestUpdateTool:
    def test_updates_returns_200(self):
        # Ownership SELECT, UPDATE, SELECT updated.
        mock_db = _make_get_db(FAKE_TOOL_ROW, None, FAKE_TOOL_ROW)
        with patch("exo_web.routes.tools.get_db", mock_db):
            client = TestClient(_app)
            resp = client.put("/api/v1/tools/tool-001", json={"name": "renamed"})
        assert resp.status_code == 200

    def test_no_fields_returns_422(self):
        client = TestClient(_app, raise_server_exceptions=False)
        resp = client.put("/api/v1/tools/tool-001", json={})
        assert resp.status_code == 422

    def test_invalid_category_returns_422(self):
        client = TestClient(_app, raise_server_exceptions=False)
        resp = client.put("/api/v1/tools/tool-001", json={"category": "bogus"})
        assert resp.status_code == 422

    def test_not_found_returns_404(self):
        mock_db = _make_get_db(None)
        with patch("exo_web.routes.tools.get_db", mock_db):
            client = TestClient(_app, raise_server_exceptions=False)
            resp = client.put("/api/v1/tools/missing", json={"name": "x"})
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Generic CRUD: delete
# ---------------------------------------------------------------------------


class TestDeleteTool:
    def test_deletes_returns_204(self):
        # Ownership SELECT, DELETE.
        mock_db = _make_get_db(FAKE_TOOL_ROW, None)
        with patch("exo_web.routes.tools.get_db", mock_db):
            client = TestClient(_app)
            resp = client.delete("/api/v1/tools/tool-001")
        assert resp.status_code == 204

    def test_not_found_returns_404(self):
        mock_db = _make_get_db(None)
        with patch("exo_web.routes.tools.get_db", mock_db):
            client = TestClient(_app, raise_server_exceptions=False)
            resp = client.delete("/api/v1/tools/missing")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Custom tool endpoints
# ---------------------------------------------------------------------------


class TestCreateCustomTool:
    def test_creates_and_returns_schema(self):
        created = {**FAKE_TOOL_ROW, "code": VALID_CODE}
        # Project lookup, INSERT, SELECT created row.
        mock_db = _make_get_db(PROJECT_ROW, None, created)
        with patch("exo_web.routes.tools.get_db", mock_db):
            client = TestClient(_app)
            resp = client.post(
                "/api/v1/tools/custom",
                json={"name": "greet", "project_id": "proj-001", "code": VALID_CODE},
            )
        assert resp.status_code == 201
        body = resp.json()
        assert "schema" in body
        props = body["schema"]["properties"]
        assert "name" in props
        assert "count" in props
        # name is required (no default); count is optional (default 1).
        assert body["schema"]["required"] == ["name"]
        params = {p["name"]: p for p in body["parameters"]}
        assert params["name"]["default"] is None
        assert params["count"]["default"] == "1"

    def test_no_function_returns_422(self):
        # _generate_schema_from_code raises 422 before any DB access.
        client = TestClient(_app, raise_server_exceptions=False)
        resp = client.post(
            "/api/v1/tools/custom",
            json={"name": "x", "project_id": "proj-001", "code": "x = 1"},
        )
        assert resp.status_code == 422

    def test_project_not_found_returns_404(self):
        mock_db = _make_get_db(None)
        with patch("exo_web.routes.tools.get_db", mock_db):
            client = TestClient(_app, raise_server_exceptions=False)
            resp = client.post(
                "/api/v1/tools/custom",
                json={"name": "greet", "project_id": "missing", "code": VALID_CODE},
            )
        assert resp.status_code == 404


class TestUpdateCustomTool:
    def test_updates_and_regenerates_schema(self):
        updated = {**FAKE_TOOL_ROW, "code": VALID_CODE}
        # Ownership SELECT, UPDATE, SELECT updated.
        mock_db = _make_get_db(FAKE_TOOL_ROW, None, updated)
        with patch("exo_web.routes.tools.get_db", mock_db):
            client = TestClient(_app)
            resp = client.put("/api/v1/tools/custom/tool-001", json={"code": VALID_CODE})
        assert resp.status_code == 200
        body = resp.json()
        assert "name" in body["schema"]["properties"]

    def test_not_found_returns_404(self):
        mock_db = _make_get_db(None)
        with patch("exo_web.routes.tools.get_db", mock_db):
            client = TestClient(_app, raise_server_exceptions=False)
            resp = client.put("/api/v1/tools/custom/missing", json={"code": VALID_CODE})
        assert resp.status_code == 404


class TestTestCustomTool:
    def test_executes_and_returns_output(self):
        tool_with_code = {**FAKE_TOOL_ROW, "code": VALID_CODE}
        mock_db = _make_get_db(tool_with_code)
        with patch("exo_web.routes.tools.get_db", mock_db):
            client = TestClient(_app)
            resp = client.post(
                "/api/v1/tools/custom/tool-001/test",
                json={"inputs": {"name": "Alice", "count": 1}},
            )
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert body["output"] == "Hello Alice"

    def test_no_code_returns_422(self):
        # FAKE_TOOL_ROW has empty code.
        mock_db = _make_get_db(FAKE_TOOL_ROW)
        with patch("exo_web.routes.tools.get_db", mock_db):
            client = TestClient(_app, raise_server_exceptions=False)
            resp = client.post(
                "/api/v1/tools/custom/tool-001/test",
                json={"inputs": {}},
            )
        assert resp.status_code == 422


class TestGenerateSchema:
    def test_generates_without_saving(self):
        client = TestClient(_app)
        resp = client.post("/api/v1/tools/schema", json={"code": VALID_CODE})
        assert resp.status_code == 200
        body = resp.json()
        assert "schema" in body
        assert "name" in body["schema"]["properties"]
        assert body["schema"]["required"] == ["name"]


# ---------------------------------------------------------------------------
# Visual schema save
# ---------------------------------------------------------------------------


class TestSaveVisualSchema:
    def test_saves_and_returns_201(self):
        # Project lookup, INSERT, SELECT created row.
        mock_db = _make_get_db(PROJECT_ROW, None, FAKE_TOOL_ROW)
        with patch("exo_web.routes.tools.get_db", mock_db):
            client = TestClient(_app)
            resp = client.post(
                "/api/v1/tools/schema/save",
                json={
                    "name": "visual",
                    "project_id": "proj-001",
                    "schema": {"type": "object", "properties": {}},
                },
            )
        assert resp.status_code == 201
        assert resp.json()["id"] == "tool-001"

    def test_project_not_found_returns_404(self):
        mock_db = _make_get_db(None)
        with patch("exo_web.routes.tools.get_db", mock_db):
            client = TestClient(_app, raise_server_exceptions=False)
            resp = client.post(
                "/api/v1/tools/schema/save",
                json={"name": "visual", "project_id": "missing", "schema": {}},
            )
        assert resp.status_code == 404
