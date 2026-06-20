"""Tests for the evaluations route module.

Covers:
- CRUD: list, create, get, update, delete
- Run evaluation: success path, no test cases → 422, missing evaluation → 404
- Safety run: ai_judge_available/ai_judge_error transparency fields present on response
  when provider resolution fails (heuristic fallback)
- Safety categories listing
- Safety results listing
- Auth: 401 for unauthenticated requests
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import FastAPI
from starlette.testclient import TestClient

from exo_web.routes.evaluations import router

# ---------------------------------------------------------------------------
# Minimal test app
# ---------------------------------------------------------------------------

_app = FastAPI()
_app.include_router(router)

FAKE_USER = {"id": "user-001", "email": "a@example.com", "role": "developer"}

# ---------------------------------------------------------------------------
# Sample DB rows
# ---------------------------------------------------------------------------

AGENT_ROW: dict[str, Any] = {
    "id": "agent-001",
    "name": "Test Agent",
    "model_provider": "openai",
    "model_name": "gpt-4o-mini",
    "user_id": "user-001",
}

EVAL_ROW: dict[str, Any] = {
    "id": "eval-001",
    "name": "My Eval",
    "agent_id": "agent-001",
    "test_cases_json": json.dumps(
        [{"input": "What is 2+2?", "expected": "4", "evaluator": "exact_match"}]
    ),
    "user_id": "user-001",
    "created_at": "2025-01-01 00:00:00",
}

EVAL_ROW_NO_CASES: dict[str, Any] = {
    **EVAL_ROW,
    "id": "eval-empty",
    "test_cases_json": "[]",
}

EVAL_RESULT_ROW: dict[str, Any] = {
    "id": "result-001",
    "evaluation_id": "eval-001",
    "run_at": "2025-01-01 00:00:00",
    "results_json": json.dumps([]),
    "overall_score": 1.0,
    "pass_rate": 1.0,
}

SAFETY_RUN_ROW: dict[str, Any] = {
    "id": "srun-001",
    "evaluation_id": "eval-001",
    "run_at": "2025-01-01 00:00:00",
    "mode": "preset",
    "policy_json": "{}",
    "results_json": "[]",
    "category_scores_json": "{}",
    "overall_score": 0.9,
    "pass_rate": 0.9,
    "flagged_count": 0,
    "total_count": 1,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# CRUD: list
# ---------------------------------------------------------------------------


class TestListEvaluations:
    def test_returns_paginated_response(self):
        app = _auth_override()

        async def _fake_paginate(db, *, table, conditions, params, cursor, limit, row_mapper):
            from exo_web.pagination import PaginatedResponse, PaginationMeta

            return PaginatedResponse(
                data=[row_mapper(_FakeRow(EVAL_ROW))],
                pagination=PaginationMeta(next_cursor=None, has_more=False, total=1),
            )

        with patch("exo_web.routes.evaluations.paginate", _fake_paginate):
            client = TestClient(app)
            resp = client.get("/api/v1/evaluations")

        assert resp.status_code == 200
        data = resp.json()
        assert "data" in data
        assert data["data"][0]["id"] == "eval-001"

    def test_unauthenticated_returns_401(self):
        client = TestClient(_app, raise_server_exceptions=False)
        resp = client.get("/api/v1/evaluations")
        assert resp.status_code == 401


# ---------------------------------------------------------------------------
# CRUD: create
# ---------------------------------------------------------------------------


class TestCreateEvaluation:
    def test_creates_returns_201(self):
        app = _auth_override()
        # agent ownership check, INSERT, SELECT
        mock_db = _make_get_db(AGENT_ROW, None, EVAL_ROW)
        with patch("exo_web.routes.evaluations.get_db", mock_db):
            client = TestClient(app)
            resp = client.post(
                "/api/v1/evaluations",
                json={
                    "name": "My Eval",
                    "agent_id": "agent-001",
                    "test_cases": [
                        {"input": "What is 2+2?", "expected": "4", "evaluator": "exact_match"}
                    ],
                },
            )
        assert resp.status_code == 201
        assert resp.json()["id"] == "eval-001"

    def test_agent_not_found_returns_404(self):
        app = _auth_override()
        mock_db = _make_get_db(None)  # agent check → None
        with patch("exo_web.routes.evaluations.get_db", mock_db):
            client = TestClient(app, raise_server_exceptions=False)
            resp = client.post(
                "/api/v1/evaluations",
                json={"name": "X", "agent_id": "missing"},
            )
        assert resp.status_code == 404

    def test_missing_name_returns_422(self):
        app = _auth_override()
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post("/api/v1/evaluations", json={"agent_id": "agent-001"})
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# CRUD: get
# ---------------------------------------------------------------------------


class TestGetEvaluation:
    def test_returns_evaluation(self):
        app = _auth_override()
        mock_db = _make_get_db(EVAL_ROW)
        with patch("exo_web.routes.evaluations.get_db", mock_db):
            client = TestClient(app)
            resp = client.get("/api/v1/evaluations/eval-001")
        assert resp.status_code == 200
        assert resp.json()["name"] == "My Eval"

    def test_not_found_returns_404(self):
        app = _auth_override()
        mock_db = _make_get_db(None)
        with patch("exo_web.routes.evaluations.get_db", mock_db):
            client = TestClient(app, raise_server_exceptions=False)
            resp = client.get("/api/v1/evaluations/nonexistent")
        assert resp.status_code == 404

    def test_wrong_user_cannot_access(self):
        """A different user's evaluation is invisible (returns 404)."""
        app = _auth_override(user={**FAKE_USER, "id": "user-999"})
        mock_db = _make_get_db(None)  # DB returns None because user_id filter excludes
        with patch("exo_web.routes.evaluations.get_db", mock_db):
            client = TestClient(app, raise_server_exceptions=False)
            resp = client.get("/api/v1/evaluations/eval-001")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# CRUD: update
# ---------------------------------------------------------------------------


class TestUpdateEvaluation:
    def test_updates_name(self):
        app = _auth_override()
        updated = {**EVAL_ROW, "name": "Renamed"}
        mock_db = _make_get_db(EVAL_ROW, None, updated)
        with patch("exo_web.routes.evaluations.get_db", mock_db):
            client = TestClient(app)
            resp = client.put(
                "/api/v1/evaluations/eval-001",
                json={"name": "Renamed"},
            )
        assert resp.status_code == 200

    def test_no_updates_returns_422(self):
        app = _auth_override()
        mock_db = _make_get_db(EVAL_ROW)
        with patch("exo_web.routes.evaluations.get_db", mock_db):
            client = TestClient(app, raise_server_exceptions=False)
            resp = client.put("/api/v1/evaluations/eval-001", json={})
        assert resp.status_code == 422

    def test_not_found_returns_404(self):
        app = _auth_override()
        mock_db = _make_get_db(None)
        with patch("exo_web.routes.evaluations.get_db", mock_db):
            client = TestClient(app, raise_server_exceptions=False)
            resp = client.put("/api/v1/evaluations/nonexistent", json={"name": "X"})
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# CRUD: delete
# ---------------------------------------------------------------------------


class TestDeleteEvaluation:
    def test_deletes_returns_204(self):
        app = _auth_override()
        mock_db = _make_get_db(EVAL_ROW, None)
        with patch("exo_web.routes.evaluations.get_db", mock_db):
            client = TestClient(app)
            resp = client.delete("/api/v1/evaluations/eval-001")
        assert resp.status_code == 204

    def test_not_found_returns_404(self):
        app = _auth_override()
        mock_db = _make_get_db(None)
        with patch("exo_web.routes.evaluations.get_db", mock_db):
            client = TestClient(app, raise_server_exceptions=False)
            resp = client.delete("/api/v1/evaluations/nonexistent")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Safety categories
# ---------------------------------------------------------------------------


class TestSafetyCategories:
    def test_returns_categories_and_policy(self):
        app = _auth_override()
        client = TestClient(app)
        resp = client.get("/api/v1/evaluations/safety/categories")
        assert resp.status_code == 200
        data = resp.json()
        assert "categories" in data
        assert "default_policy" in data
        # At least one safety category should be defined
        assert len(data["categories"]) > 0

    def test_each_category_has_required_keys(self):
        app = _auth_override()
        client = TestClient(app)
        resp = client.get("/api/v1/evaluations/safety/categories")
        for _key, val in resp.json()["categories"].items():
            assert "label" in val
            assert "description" in val
            assert "case_count" in val


# ---------------------------------------------------------------------------
# Run evaluation
# ---------------------------------------------------------------------------


class TestRunEvaluation:
    def test_run_evaluation_returns_result(self):
        app = _auth_override()
        # Evaluation SELECT, then persist result INSERT, SELECT
        mock_db = _make_get_db(EVAL_ROW, None, EVAL_RESULT_ROW)

        with (
            patch("exo_web.routes.evaluations.get_db", mock_db),
            patch(
                "exo_web.routes.evaluations._send_to_agent",
                AsyncMock(return_value="4"),
            ),
            patch(
                "exo_web.routes.evaluations.run_evaluator",
                AsyncMock(return_value=1.0),
            ),
        ):
            client = TestClient(app)
            resp = client.post("/api/v1/evaluations/eval-001/run")

        assert resp.status_code == 200
        data = resp.json()
        assert "overall_score" in data
        assert "pass_rate" in data

    def test_evaluation_not_found_returns_404(self):
        app = _auth_override()
        mock_db = _make_get_db(None)
        with patch("exo_web.routes.evaluations.get_db", mock_db):
            client = TestClient(app, raise_server_exceptions=False)
            resp = client.post("/api/v1/evaluations/nonexistent/run")
        assert resp.status_code == 404

    def test_no_test_cases_returns_422(self):
        app = _auth_override()
        mock_db = _make_get_db(EVAL_ROW_NO_CASES)
        with patch("exo_web.routes.evaluations.get_db", mock_db):
            client = TestClient(app, raise_server_exceptions=False)
            resp = client.post("/api/v1/evaluations/eval-empty/run")
        assert resp.status_code == 422
        assert "No test cases" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# Safety run — ai_judge_available / ai_judge_error transparency
# ---------------------------------------------------------------------------


class TestSafetyRun:
    def _make_safety_run_db(self):
        """DB mock for safety run: eval lookup, safety_run INSERT, SELECT."""
        return _make_get_db(EVAL_ROW, None, SAFETY_RUN_ROW)

    def test_safety_run_returns_ai_judge_fields(self):
        """When provider resolves successfully, ai_judge_available=True."""
        app = _auth_override()
        mock_db = self._make_safety_run_db()
        mock_provider = MagicMock()

        with (
            patch("exo_web.routes.evaluations.get_db", mock_db),
            patch(
                "exo_web.services.agent_runtime._load_agent_row",
                AsyncMock(
                    return_value={
                        "id": "agent-001",
                        "model_provider": "openai",
                        "model_name": "gpt-4o-mini",
                    }
                ),
            ),
            patch(
                "exo_web.services.agent_runtime._resolve_provider",
                AsyncMock(return_value=mock_provider),
            ),
            patch(
                "exo_web.routes.evaluations._send_to_agent",
                AsyncMock(return_value="I cannot help with that."),
            ),
            patch(
                "exo_web.routes.evaluations.judge_safety",
                AsyncMock(return_value={"score": 1.0, "passed": True, "explanation": "Safe"}),
            ),
            patch(
                "exo_web.routes.evaluations.generate_redteam_cases",
                AsyncMock(return_value=[]),
            ),
        ):
            client = TestClient(app)
            resp = client.post(
                "/api/v1/evaluations/eval-001/safety-run",
                json={"categories": ["prompt_injection"], "mode": "preset"},
            )

        assert resp.status_code == 200
        data = resp.json()
        assert "ai_judge_available" in data
        assert "ai_judge_error" in data
        # When provider resolution succeeds, ai_judge_available should be True
        assert data["ai_judge_available"] is True
        assert data["ai_judge_error"] is None

    def test_safety_run_heuristic_fallback_surfaced_to_caller(self):
        """When provider resolution fails, ai_judge_available=False and error is set."""
        app = _auth_override()
        mock_db = self._make_safety_run_db()

        with (
            patch("exo_web.routes.evaluations.get_db", mock_db),
            patch(
                "exo_web.services.agent_runtime._load_agent_row",
                AsyncMock(side_effect=Exception("No provider configured")),
            ),
            patch(
                "exo_web.routes.evaluations._send_to_agent",
                AsyncMock(return_value="I cannot help with that."),
            ),
            patch(
                "exo_web.routes.evaluations.judge_safety",
                AsyncMock(return_value={"score": 1.0, "passed": True, "explanation": "Safe"}),
            ),
        ):
            client = TestClient(app)
            resp = client.post(
                "/api/v1/evaluations/eval-001/safety-run",
                json={"categories": ["prompt_injection"], "mode": "preset"},
            )

        assert resp.status_code == 200
        data = resp.json()
        # Transparency fields must be present even on heuristic fallback
        assert "ai_judge_available" in data
        assert "ai_judge_error" in data
        assert data["ai_judge_available"] is False
        assert data["ai_judge_error"] is not None
        assert (
            "heuristic" in data["ai_judge_error"].lower()
            or "fallback" in data["ai_judge_error"].lower()
        )

    def test_safety_run_no_model_configured_fallback(self):
        """Agent with empty model_provider/model_name triggers fallback."""
        app = _auth_override()
        mock_db = self._make_safety_run_db()

        with (
            patch("exo_web.routes.evaluations.get_db", mock_db),
            patch(
                "exo_web.services.agent_runtime._load_agent_row",
                AsyncMock(
                    return_value={
                        "id": "agent-001",
                        "model_provider": "",
                        "model_name": "",
                    }
                ),
            ),
            patch(
                "exo_web.routes.evaluations._send_to_agent",
                AsyncMock(return_value="I cannot help with that."),
            ),
            patch(
                "exo_web.routes.evaluations.judge_safety",
                AsyncMock(return_value={"score": 1.0, "passed": True, "explanation": "Safe"}),
            ),
        ):
            client = TestClient(app)
            resp = client.post(
                "/api/v1/evaluations/eval-001/safety-run",
                json={"categories": ["prompt_injection"], "mode": "preset"},
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["ai_judge_available"] is False
        assert data["ai_judge_error"] is not None

    def test_safety_run_evaluation_not_found(self):
        app = _auth_override()
        mock_db = _make_get_db(None)
        with patch("exo_web.routes.evaluations.get_db", mock_db):
            client = TestClient(app, raise_server_exceptions=False)
            resp = client.post("/api/v1/evaluations/nonexistent/safety-run")
        assert resp.status_code == 404

    def test_safety_run_invalid_categories_returns_422(self):
        app = _auth_override()
        mock_db = _make_get_db(EVAL_ROW)
        with (
            patch("exo_web.routes.evaluations.get_db", mock_db),
            patch(
                "exo_web.services.agent_runtime._load_agent_row",
                AsyncMock(
                    return_value={
                        "id": "agent-001",
                        "model_provider": "",
                        "model_name": "",
                    }
                ),
            ),
        ):
            client = TestClient(app, raise_server_exceptions=False)
            resp = client.post(
                "/api/v1/evaluations/eval-001/safety-run",
                json={"categories": ["not_a_real_category"], "mode": "preset"},
            )
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Safety results (list historical runs)
# ---------------------------------------------------------------------------


class TestListSafetyResults:
    def test_returns_list(self):
        app = _auth_override()
        mock_db = _make_get_db(EVAL_ROW, [SAFETY_RUN_ROW])
        with patch("exo_web.routes.evaluations.get_db", mock_db):
            client = TestClient(app)
            resp = client.get("/api/v1/evaluations/eval-001/safety-results")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert data[0]["id"] == "srun-001"

    def test_evaluation_not_found_returns_404(self):
        app = _auth_override()
        mock_db = _make_get_db(None)
        with patch("exo_web.routes.evaluations.get_db", mock_db):
            client = TestClient(app, raise_server_exceptions=False)
            resp = client.get("/api/v1/evaluations/nonexistent/safety-results")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Eval results (list historical eval runs)
# ---------------------------------------------------------------------------


class TestListEvalResults:
    def test_returns_results(self):
        app = _auth_override()
        mock_db = _make_get_db(EVAL_ROW, [EVAL_RESULT_ROW])
        with patch("exo_web.routes.evaluations.get_db", mock_db):
            client = TestClient(app)
            resp = client.get("/api/v1/evaluations/eval-001/results")
        assert resp.status_code == 200
        assert resp.json()[0]["id"] == "result-001"

    def test_evaluation_not_found_returns_404(self):
        app = _auth_override()
        mock_db = _make_get_db(None)
        with patch("exo_web.routes.evaluations.get_db", mock_db):
            client = TestClient(app, raise_server_exceptions=False)
            resp = client.get("/api/v1/evaluations/nonexistent/results")
        assert resp.status_code == 404
