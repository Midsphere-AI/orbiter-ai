"""Tests for the workflows CRUD REST API."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import AsyncMock, patch

from fastapi import FastAPI
from starlette.testclient import TestClient

from exo_web.routes.auth import get_current_user
from exo_web.routes.workflows import router

# ---------------------------------------------------------------------------
# Test app
# ---------------------------------------------------------------------------

FAKE_USER = {
    "id": "user-001",
    "email": "test@example.com",
    "role": "developer",
    "created_at": "2025-01-01T00:00:00",
}

_app = FastAPI()
_app.include_router(router)
_app.dependency_overrides[get_current_user] = lambda: FAKE_USER


# ---------------------------------------------------------------------------
# Fixtures / fakes
# ---------------------------------------------------------------------------

FAKE_WORKFLOW_ROW = {
    "id": "wf-001",
    "name": "My Workflow",
    "description": "",
    "project_id": "proj-001",
    "nodes_json": "[]",
    "edges_json": "[]",
    "viewport_json": '{"x":0,"y":0,"zoom":1}',
    "version": 1,
    "status": "draft",
    "last_run_at": None,
    "user_id": "user-001",
    "created_at": "2025-01-01 00:00:00",
    "updated_at": "2025-01-01 00:00:00",
}

PROJECT_ROW = {"id": "proj-001"}


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
    """Return a patched get_db that yields mock rows in order across execute calls."""
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


def _patch_side_effects():
    """Patch out config-version + audit-log side effects shared by mutating endpoints."""
    return (
        patch("exo_web.routes.workflows.create_config_version", AsyncMock()),
        patch("exo_web.routes.workflows.audit_log", AsyncMock()),
    )


def _client() -> TestClient:
    return TestClient(_app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# GET /api/v1/workflows  (list)
# ---------------------------------------------------------------------------


class TestListWorkflows:
    def test_success(self) -> None:
        page = AsyncMock()
        page.model_dump = lambda: {"items": [], "next_cursor": None}
        mock_paginate = AsyncMock(return_value=page)

        with (
            patch("exo_web.routes.workflows.get_db", _make_get_db()),
            patch("exo_web.routes.workflows.paginate", mock_paginate),
        ):
            resp = _client().get("/api/v1/workflows")

        assert resp.status_code == 200
        assert resp.json() == {"items": [], "next_cursor": None}

    def test_unauthenticated_returns_401(self) -> None:
        app_no_auth = FastAPI()
        app_no_auth.include_router(router)

        with patch("exo_web.routes.auth.get_db", _make_get_db(None)):
            client = TestClient(app_no_auth, raise_server_exceptions=False)
            resp = client.get("/api/v1/workflows")

        assert resp.status_code == 401


# ---------------------------------------------------------------------------
# POST /api/v1/workflows  (create)
# ---------------------------------------------------------------------------


class TestCreateWorkflow:
    def test_success(self) -> None:
        mock_get_db = _make_get_db(
            PROJECT_ROW,  # project ownership check
            None,  # INSERT workflows (no row returned)
            FAKE_WORKFLOW_ROW,  # SELECT after INSERT
        )
        cv_patch, audit_patch = _patch_side_effects()
        with (
            patch("exo_web.routes.workflows.get_db", mock_get_db),
            patch("exo_web.routes.workflows._snapshot_workflow", lambda row: {}),
            cv_patch,
            audit_patch,
        ):
            resp = _client().post(
                "/api/v1/workflows",
                json={"name": "My Workflow", "project_id": "proj-001"},
            )

        assert resp.status_code == 201
        assert resp.json()["id"] == "wf-001"

    def test_project_not_found_returns_404(self) -> None:
        mock_get_db = _make_get_db(None)  # project lookup misses
        cv_patch, audit_patch = _patch_side_effects()
        with (
            patch("exo_web.routes.workflows.get_db", mock_get_db),
            cv_patch,
            audit_patch,
        ):
            resp = _client().post(
                "/api/v1/workflows",
                json={"name": "My Workflow", "project_id": "missing"},
            )

        assert resp.status_code == 404

    def test_missing_name_returns_422(self) -> None:
        with patch("exo_web.routes.workflows.get_db", _make_get_db()):
            resp = _client().post("/api/v1/workflows", json={"project_id": "proj-001"})

        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# GET /api/v1/workflows/{workflow_id}
# ---------------------------------------------------------------------------


class TestGetWorkflow:
    def test_found(self) -> None:
        with patch("exo_web.routes.workflows.get_db", _make_get_db(FAKE_WORKFLOW_ROW)):
            resp = _client().get("/api/v1/workflows/wf-001")

        assert resp.status_code == 200
        assert resp.json()["id"] == "wf-001"

    def test_not_found_returns_404(self) -> None:
        with patch("exo_web.routes.workflows.get_db", _make_get_db(None)):
            resp = _client().get("/api/v1/workflows/missing")

        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# PUT /api/v1/workflows/{workflow_id}
# ---------------------------------------------------------------------------


class TestUpdateWorkflow:
    def test_success(self) -> None:
        mock_get_db = _make_get_db(
            FAKE_WORKFLOW_ROW,  # _verify_ownership
            None,  # UPDATE workflows (no row returned)
            FAKE_WORKFLOW_ROW,  # SELECT after UPDATE
        )
        cv_patch, audit_patch = _patch_side_effects()
        with (
            patch("exo_web.routes.workflows.get_db", mock_get_db),
            patch("exo_web.routes.workflows._snapshot_workflow", lambda row: {}),
            cv_patch,
            audit_patch,
        ):
            resp = _client().put("/api/v1/workflows/wf-001", json={"name": "Renamed"})

        assert resp.status_code == 200
        assert resp.json()["id"] == "wf-001"

    def test_no_fields_returns_422(self) -> None:
        with patch("exo_web.routes.workflows.get_db", _make_get_db()):
            resp = _client().put("/api/v1/workflows/wf-001", json={})

        assert resp.status_code == 422

    def test_not_found_returns_404(self) -> None:
        with patch("exo_web.routes.workflows.get_db", _make_get_db(None)):
            resp = _client().put("/api/v1/workflows/missing", json={"name": "Renamed"})

        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# DELETE /api/v1/workflows/{workflow_id}
# ---------------------------------------------------------------------------


class TestDeleteWorkflow:
    def test_success(self) -> None:
        mock_get_db = _make_get_db(
            FAKE_WORKFLOW_ROW,  # _verify_ownership
            None,  # DELETE
        )
        cv_patch, audit_patch = _patch_side_effects()
        with (
            patch("exo_web.routes.workflows.get_db", mock_get_db),
            cv_patch,
            audit_patch,
        ):
            resp = _client().delete("/api/v1/workflows/wf-001")

        assert resp.status_code == 204

    def test_not_found_returns_404(self) -> None:
        with patch("exo_web.routes.workflows.get_db", _make_get_db(None)):
            resp = _client().delete("/api/v1/workflows/missing")

        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# POST /api/v1/workflows/import
# ---------------------------------------------------------------------------


class TestImportWorkflow:
    def test_valid_json_fields(self) -> None:
        mock_get_db = _make_get_db(
            PROJECT_ROW,  # project ownership check
            None,  # INSERT workflows (no row returned)
            FAKE_WORKFLOW_ROW,  # SELECT after INSERT
        )
        with patch("exo_web.routes.workflows.get_db", mock_get_db):
            resp = _client().post(
                "/api/v1/workflows/import",
                json={
                    "name": "Imported",
                    "project_id": "proj-001",
                    "nodes_json": "[]",
                    "edges_json": "[]",
                    "viewport_json": '{"x":0,"y":0,"zoom":1}',
                },
            )

        assert resp.status_code == 201
        assert resp.json()["id"] == "wf-001"

    def test_invalid_nodes_json_returns_422(self) -> None:
        with patch("exo_web.routes.workflows.get_db", _make_get_db()):
            resp = _client().post(
                "/api/v1/workflows/import",
                json={
                    "name": "Imported",
                    "project_id": "proj-001",
                    "nodes_json": "not-json",
                },
            )

        assert resp.status_code == 422

    def test_project_not_found_returns_404(self) -> None:
        mock_get_db = _make_get_db(None)  # project lookup misses
        with patch("exo_web.routes.workflows.get_db", mock_get_db):
            resp = _client().post(
                "/api/v1/workflows/import",
                json={"name": "Imported", "project_id": "missing"},
            )

        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# POST /api/v1/workflows/{workflow_id}/export
# ---------------------------------------------------------------------------


class TestExportWorkflow:
    def test_success(self) -> None:
        with patch("exo_web.routes.workflows.get_db", _make_get_db(FAKE_WORKFLOW_ROW)):
            resp = _client().post("/api/v1/workflows/wf-001/export")

        assert resp.status_code == 200
        assert "attachment; filename=" in resp.headers["content-disposition"]
        assert resp.json()["name"] == "My Workflow"


# ---------------------------------------------------------------------------
# POST /api/v1/workflows/{workflow_id}/duplicate
# ---------------------------------------------------------------------------


class TestDuplicateWorkflow:
    def test_success(self) -> None:
        duplicated = {**FAKE_WORKFLOW_ROW, "id": "wf-002", "name": "My Workflow (Copy)"}
        mock_get_db = _make_get_db(
            FAKE_WORKFLOW_ROW,  # _verify_ownership (first get_db block)
            None,  # INSERT workflows (second get_db block, no row returned)
            duplicated,  # SELECT after INSERT (second get_db block)
        )
        with patch("exo_web.routes.workflows.get_db", mock_get_db):
            resp = _client().post("/api/v1/workflows/wf-001/duplicate")

        assert resp.status_code == 201
        assert resp.json()["name"] == "My Workflow (Copy)"
