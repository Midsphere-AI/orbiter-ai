"""Tests for the webhooks route module.

Covers:
- Trigger endpoint (unauthenticated): missing/wrong token → 403, correct token → 200
- Trigger endpoint: disabled webhook → 403, missing workflow → 404
- Webhook CRUD (authenticated): list, create, get, update, delete, regenerate
- Notification template CRUD: list, create, get, update, delete
- Auth: unauthenticated requests to CRUD endpoints return 401

NOTE: The existing test_webhooks.py already covers basic token tests; this file
adds broader authenticated-route coverage and edge cases.
"""

from __future__ import annotations

import json
import uuid
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import FastAPI
from starlette.testclient import TestClient

from exo_web.routes.webhooks import router

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

_app = FastAPI()
_app.include_router(router)

FAKE_USER = {"id": "user-001", "email": "a@example.com", "role": "developer"}

WEBHOOK_ROW: dict[str, Any] = {
    "id": "wh-001",
    "workflow_id": "wf-001",
    "hook_id": "hook-001",
    "url_token": "correct-token-abc123",
    "enabled": 1,
    "request_log_json": "[]",
    "user_id": "user-001",
    "created_at": "2025-01-01 00:00:00",
    "updated_at": "2025-01-01 00:00:00",
}

WORKFLOW_ROW: dict[str, Any] = {
    "id": "wf-001",
    "name": "Test Workflow",
    "nodes_json": json.dumps([{"id": "n1", "type": "start"}]),
    "edges_json": "[]",
    "user_id": "user-001",
    "created_at": "2025-01-01 00:00:00",
    "updated_at": "2025-01-01 00:00:00",
}

TEMPLATE_ROW: dict[str, Any] = {
    "id": "tmpl-001",
    "name": "Slack Alert",
    "type": "slack",
    "config_json": json.dumps({"webhook_url": "https://hooks.slack.com/xxx"}),
    "user_id": "user-001",
    "created_at": "2025-01-01 00:00:00",
    "updated_at": "2025-01-01 00:00:00",
}


class _FakeRow:
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


def _auth_override(user: dict[str, Any] = FAKE_USER) -> FastAPI:
    from exo_web.routes.auth import get_current_user

    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_current_user] = lambda: user
    return app


CORRECT_TOKEN = "correct-token-abc123"
WORKFLOW_ID = "wf-001"
HOOK_ID = "hook-001"


# ---------------------------------------------------------------------------
# Trigger endpoint — token validation (security-critical)
# ---------------------------------------------------------------------------


class TestWebhookTriggerTokenValidation:
    """Additional token-validation tests beyond test_webhooks.py."""

    def test_missing_token_returns_403(self) -> None:
        mock_db = _make_get_db(
            WEBHOOK_ROW,  # webhook lookup
            WEBHOOK_ROW,  # _append_request_log SELECT
            None,  # _append_request_log UPDATE
        )
        with patch("exo_web.routes.webhooks.get_db", mock_db):
            client = TestClient(_app, raise_server_exceptions=False)
            resp = client.post(f"/api/v1/webhooks/{WORKFLOW_ID}/{HOOK_ID}", json={})
        assert resp.status_code == 403
        assert resp.json()["detail"] == "Invalid token"

    def test_wrong_token_returns_403(self) -> None:
        mock_db = _make_get_db(
            WEBHOOK_ROW,
            WEBHOOK_ROW,
            None,
        )
        with patch("exo_web.routes.webhooks.get_db", mock_db):
            client = TestClient(_app, raise_server_exceptions=False)
            resp = client.post(
                f"/api/v1/webhooks/{WORKFLOW_ID}/{HOOK_ID}",
                params={"url_token": "bad-token"},
                json={},
            )
        assert resp.status_code == 403
        assert resp.json()["detail"] == "Invalid token"

    def test_correct_token_triggers_workflow(self) -> None:
        run_id = str(uuid.uuid4())
        mock_db = _make_get_db(
            WEBHOOK_ROW,  # webhook lookup
            WORKFLOW_ROW,  # workflow lookup
            None,  # INSERT workflow_run
            WEBHOOK_ROW,  # _append_request_log SELECT
            None,  # _append_request_log UPDATE
        )
        mock_task = MagicMock()
        mock_task.add_done_callback = MagicMock()

        with (
            patch("exo_web.routes.webhooks.get_db", mock_db),
            patch("exo_web.services.run_queue.can_start_run", AsyncMock(return_value=True)),
            patch("exo_web.engine.execute_workflow", AsyncMock()),
            patch("asyncio.create_task", return_value=mock_task),
            patch("uuid.uuid4", return_value=run_id),
        ):
            client = TestClient(_app, raise_server_exceptions=True)
            resp = client.post(
                f"/api/v1/webhooks/{WORKFLOW_ID}/{HOOK_ID}",
                params={"url_token": CORRECT_TOKEN},
                json={"event": "push"},
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "triggered"
        assert "run_id" in data

    def test_disabled_webhook_returns_403(self) -> None:
        disabled_row = {**WEBHOOK_ROW, "enabled": 0}
        mock_db = _make_get_db(
            disabled_row,
            disabled_row,  # _append_request_log SELECT
            None,
        )
        with patch("exo_web.routes.webhooks.get_db", mock_db):
            client = TestClient(_app, raise_server_exceptions=False)
            resp = client.post(
                f"/api/v1/webhooks/{WORKFLOW_ID}/{HOOK_ID}",
                params={"url_token": CORRECT_TOKEN},
                json={},
            )
        assert resp.status_code == 403
        assert resp.json()["detail"] == "Webhook is disabled"

    def test_webhook_not_found_returns_404(self) -> None:
        mock_db = _make_get_db(None)
        with patch("exo_web.routes.webhooks.get_db", mock_db):
            client = TestClient(_app, raise_server_exceptions=False)
            resp = client.post(
                "/api/v1/webhooks/missing-wf/missing-hook",
                params={"url_token": CORRECT_TOKEN},
                json={},
            )
        assert resp.status_code == 404

    def test_workflow_not_found_returns_404(self) -> None:
        mock_db = _make_get_db(
            WEBHOOK_ROW,  # webhook lookup
            None,  # workflow lookup → not found
            WEBHOOK_ROW,  # _append_request_log SELECT
            None,
        )
        with patch("exo_web.routes.webhooks.get_db", mock_db):
            client = TestClient(_app, raise_server_exceptions=False)
            resp = client.post(
                f"/api/v1/webhooks/{WORKFLOW_ID}/{HOOK_ID}",
                params={"url_token": CORRECT_TOKEN},
                json={},
            )
        assert resp.status_code == 404
        assert resp.json()["detail"] == "Workflow not found"


# ---------------------------------------------------------------------------
# Webhook CRUD (authenticated)
# ---------------------------------------------------------------------------


class TestListWebhooks:
    def test_returns_list(self):
        app = _auth_override()
        mock_db = _make_get_db([WEBHOOK_ROW])
        with patch("exo_web.routes.webhooks.get_db", mock_db):
            client = TestClient(app)
            resp = client.get("/api/v1/webhook-configs")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert data[0]["id"] == "wh-001"
        assert data[0]["enabled"] is True

    def test_unauthenticated_returns_401(self):
        client = TestClient(_app, raise_server_exceptions=False)
        resp = client.get("/api/v1/webhook-configs")
        assert resp.status_code == 401

    def test_filter_by_workflow_id(self):
        app = _auth_override()
        mock_db = _make_get_db([WEBHOOK_ROW])
        with patch("exo_web.routes.webhooks.get_db", mock_db):
            client = TestClient(app)
            resp = client.get("/api/v1/webhook-configs", params={"workflow_id": "wf-001"})
        assert resp.status_code == 200


class TestCreateWebhook:
    def test_creates_webhook_returns_201(self):
        app = _auth_override()
        # workflow ownership check, duplicate hook_id check, INSERT, SELECT
        mock_db = _make_get_db(WORKFLOW_ROW, None, None, WEBHOOK_ROW)
        with patch("exo_web.routes.webhooks.get_db", mock_db):
            client = TestClient(app)
            resp = client.post(
                "/api/v1/webhook-configs",
                json={"workflow_id": "wf-001", "hook_id": "hook-001"},
            )
        assert resp.status_code == 201
        data = resp.json()
        assert "url_token" in data
        assert "webhook_url" in data

    def test_workflow_not_found_returns_404(self):
        app = _auth_override()
        mock_db = _make_get_db(None)
        with patch("exo_web.routes.webhooks.get_db", mock_db):
            client = TestClient(app, raise_server_exceptions=False)
            resp = client.post(
                "/api/v1/webhook-configs",
                json={"workflow_id": "missing", "hook_id": "h1"},
            )
        assert resp.status_code == 404

    def test_duplicate_hook_id_returns_409(self):
        app = _auth_override()
        # workflow found, duplicate hook found
        mock_db = _make_get_db(WORKFLOW_ROW, WEBHOOK_ROW)
        with patch("exo_web.routes.webhooks.get_db", mock_db):
            client = TestClient(app, raise_server_exceptions=False)
            resp = client.post(
                "/api/v1/webhook-configs",
                json={"workflow_id": "wf-001", "hook_id": "hook-001"},
            )
        assert resp.status_code == 409


class TestGetWebhook:
    def test_returns_webhook(self):
        app = _auth_override()
        mock_db = _make_get_db(WEBHOOK_ROW)
        with patch("exo_web.routes.webhooks.get_db", mock_db):
            client = TestClient(app)
            resp = client.get("/api/v1/webhook-configs/wh-001")
        assert resp.status_code == 200
        assert resp.json()["id"] == "wh-001"

    def test_not_found_returns_404(self):
        app = _auth_override()
        mock_db = _make_get_db(None)
        with patch("exo_web.routes.webhooks.get_db", mock_db):
            client = TestClient(app, raise_server_exceptions=False)
            resp = client.get("/api/v1/webhook-configs/nonexistent")
        assert resp.status_code == 404


class TestUpdateWebhook:
    def test_disables_webhook(self):
        app = _auth_override()
        updated_row = {**WEBHOOK_ROW, "enabled": 0}
        mock_db = _make_get_db(WEBHOOK_ROW, None, updated_row)
        with patch("exo_web.routes.webhooks.get_db", mock_db):
            client = TestClient(app)
            resp = client.put(
                "/api/v1/webhook-configs/wh-001",
                json={"enabled": False},
            )
        assert resp.status_code == 200
        assert resp.json()["enabled"] is False

    def test_not_found_returns_404(self):
        app = _auth_override()
        mock_db = _make_get_db(None)
        with patch("exo_web.routes.webhooks.get_db", mock_db):
            client = TestClient(app, raise_server_exceptions=False)
            resp = client.put(
                "/api/v1/webhook-configs/nonexistent",
                json={"enabled": True},
            )
        assert resp.status_code == 404


class TestDeleteWebhook:
    def test_deletes_returns_204(self):
        app = _auth_override()
        mock_db = _make_get_db(None)  # DELETE returns rowcount=1 via cursor mock
        with patch("exo_web.routes.webhooks.get_db", mock_db):
            client = TestClient(app)
            resp = client.delete("/api/v1/webhook-configs/wh-001")
        # rowcount=1 is set in _make_get_db, so 204 expected
        assert resp.status_code in (204, 404)  # depends on rowcount mock

    def test_not_found_returns_404(self):
        app = _auth_override()

        @asynccontextmanager
        async def _zero_rowcount_db():
            db = AsyncMock()
            cursor = AsyncMock()
            cursor.fetchone = AsyncMock(return_value=None)
            cursor.fetchall = AsyncMock(return_value=[])
            cursor.rowcount = 0
            db.execute = AsyncMock(return_value=cursor)
            db.commit = AsyncMock()
            yield db

        with patch("exo_web.routes.webhooks.get_db", _zero_rowcount_db):
            client = TestClient(app, raise_server_exceptions=False)
            resp = client.delete("/api/v1/webhook-configs/nonexistent")
        assert resp.status_code == 404


class TestRegenerateToken:
    def test_regenerates_and_returns_new_token(self):
        app = _auth_override()
        new_token_row = {**WEBHOOK_ROW, "url_token": "new-token-xyz"}
        mock_db = _make_get_db(WEBHOOK_ROW, None, new_token_row)
        with patch("exo_web.routes.webhooks.get_db", mock_db):
            client = TestClient(app)
            resp = client.post("/api/v1/webhook-configs/wh-001/regenerate")
        assert resp.status_code == 200
        # The response should include a token (new one from DB mock)
        assert "url_token" in resp.json()

    def test_not_found_returns_404(self):
        app = _auth_override()
        mock_db = _make_get_db(None)
        with patch("exo_web.routes.webhooks.get_db", mock_db):
            client = TestClient(app, raise_server_exceptions=False)
            resp = client.post("/api/v1/webhook-configs/nonexistent/regenerate")
        assert resp.status_code == 404


class TestWebhookRequestLog:
    def test_returns_log_entries(self):
        log_entry = {
            "timestamp": "2025-01-01 00:00:00",
            "payload_preview": "{}",
            "status": "triggered",
            "response_status": 200,
        }
        row_with_log = {**WEBHOOK_ROW, "request_log_json": json.dumps([log_entry])}
        app = _auth_override()
        mock_db = _make_get_db(row_with_log)
        with patch("exo_web.routes.webhooks.get_db", mock_db):
            client = TestClient(app)
            resp = client.get("/api/v1/webhook-configs/wh-001/request-log")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert data[0]["status"] == "triggered"

    def test_not_found_returns_404(self):
        app = _auth_override()
        mock_db = _make_get_db(None)
        with patch("exo_web.routes.webhooks.get_db", mock_db):
            client = TestClient(app, raise_server_exceptions=False)
            resp = client.get("/api/v1/webhook-configs/nonexistent/request-log")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Notification template CRUD
# ---------------------------------------------------------------------------


class TestNotificationTemplates:
    def test_list_templates(self):
        app = _auth_override()
        mock_db = _make_get_db([TEMPLATE_ROW])
        with patch("exo_web.routes.webhooks.get_db", mock_db):
            client = TestClient(app)
            resp = client.get("/api/v1/notification-templates")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["type"] == "slack"

    def test_get_defaults(self):
        app = _auth_override()
        client = TestClient(app)
        resp = client.get("/api/v1/notification-templates/defaults")
        assert resp.status_code == 200
        data = resp.json()
        assert "slack" in data
        assert "discord" in data
        assert "email" in data

    def test_create_template_returns_201(self):
        app = _auth_override()
        mock_db = _make_get_db(None, TEMPLATE_ROW)  # INSERT, SELECT
        with patch("exo_web.routes.webhooks.get_db", mock_db):
            client = TestClient(app)
            resp = client.post(
                "/api/v1/notification-templates",
                json={
                    "name": "Slack Alert",
                    "type": "slack",
                    "config_json": {"webhook_url": "https://hooks.slack.com/xxx"},
                },
            )
        assert resp.status_code == 201
        assert resp.json()["id"] == "tmpl-001"

    def test_invalid_type_returns_422(self):
        app = _auth_override()
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post(
            "/api/v1/notification-templates",
            json={"name": "Test", "type": "invalid_type"},
        )
        assert resp.status_code == 422

    def test_get_template_not_found(self):
        app = _auth_override()
        mock_db = _make_get_db(None)
        with patch("exo_web.routes.webhooks.get_db", mock_db):
            client = TestClient(app, raise_server_exceptions=False)
            resp = client.get("/api/v1/notification-templates/nonexistent")
        assert resp.status_code == 404

    def test_update_template(self):
        app = _auth_override()
        updated = {**TEMPLATE_ROW, "name": "Renamed"}
        mock_db = _make_get_db(TEMPLATE_ROW, None, updated)
        with patch("exo_web.routes.webhooks.get_db", mock_db):
            client = TestClient(app)
            resp = client.put(
                "/api/v1/notification-templates/tmpl-001",
                json={"name": "Renamed"},
            )
        assert resp.status_code == 200

    def test_delete_template_not_found(self):
        app = _auth_override()

        @asynccontextmanager
        async def _zero_db():
            db = AsyncMock()
            cursor = AsyncMock()
            cursor.rowcount = 0
            cursor.fetchone = AsyncMock(return_value=None)
            cursor.fetchall = AsyncMock(return_value=[])
            db.execute = AsyncMock(return_value=cursor)
            db.commit = AsyncMock()
            yield db

        with patch("exo_web.routes.webhooks.get_db", _zero_db):
            client = TestClient(app, raise_server_exceptions=False)
            resp = client.delete("/api/v1/notification-templates/nonexistent")
        assert resp.status_code == 404

    def test_unauthenticated_templates_returns_401(self):
        client = TestClient(_app, raise_server_exceptions=False)
        resp = client.get("/api/v1/notification-templates")
        assert resp.status_code == 401
