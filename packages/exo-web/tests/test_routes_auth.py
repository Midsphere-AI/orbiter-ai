"""Tests for the auth router endpoints and role-based access control."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import AsyncMock, patch

import bcrypt
from fastapi import Depends, FastAPI
from starlette.testclient import TestClient

from exo_web.routes.auth import require_role, router

# ---------------------------------------------------------------------------
# Minimal test app
# ---------------------------------------------------------------------------

_app = FastAPI()
_app.include_router(router)


@_app.get("/_test/developer-only")
async def _developer_only(
    user: dict[str, Any] = Depends(require_role("developer")),  # noqa: B008
) -> dict[str, Any]:
    return user


@_app.get("/_test/admin-only")
async def _admin_only(
    user: dict[str, Any] = Depends(require_role("admin")),  # noqa: B008
) -> dict[str, Any]:
    return user


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EMAIL = "user@example.com"
USER_ID = "user-001"
SESSION_ID = "sess-001"
FAKE_PASSWORD = "correct-password-123"
FAKE_HASH = bcrypt.hashpw(FAKE_PASSWORD.encode(), bcrypt.gensalt()).decode()
CREATED_AT = "2026-01-01 00:00:00"


# ---------------------------------------------------------------------------
# Helpers
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


def _make_get_db(*row_sequence: dict[str, Any] | None, fetchall: Any = None):
    """Return a patched get_db that yields mock rows in order across calls.

    Each ``await db.execute(...)`` advances the row sequence and returns the
    next row from ``fetchone``. ``fetchall`` is a fixed list returned by every
    cursor (used by reset-password which iterates over all tokens).
    """
    rows = list(row_sequence)
    fetchall_rows = list(fetchall) if fetchall is not None else []
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
            cursor.fetchall = AsyncMock(return_value=fetchall_rows)
            cursor.rowcount = 1
            call_index += 1
            return cursor

        mock_db.execute = _execute
        mock_db.commit = AsyncMock()
        yield mock_db

    return _mock_get_db


def _user_row(role: str = "developer") -> dict[str, Any]:
    return {"id": USER_ID, "email": EMAIL, "role": role, "created_at": CREATED_AT}


def _session_user_row(role: str = "developer") -> dict[str, Any]:
    """A joined sessions/users row as returned by get_current_user."""
    return {"id": USER_ID, "email": EMAIL, "role": role, "created_at": CREATED_AT}


class _FakeSettings:
    def __init__(self, debug: bool = True, session_expiry_hours: int = 24) -> None:
        self.debug = debug
        self.session_expiry_hours = session_expiry_hours


def _client() -> TestClient:
    return TestClient(_app, raise_server_exceptions=True)


# ---------------------------------------------------------------------------
# POST /login
# ---------------------------------------------------------------------------


class TestLogin:
    def test_success_sets_cookie(self) -> None:
        login_row = {
            "id": USER_ID,
            "email": EMAIL,
            "password_hash": FAKE_HASH,
            "role": "developer",
            "created_at": CREATED_AT,
        }
        mock_get_db = _make_get_db(login_row, None)
        with (
            patch("exo_web.routes.auth.get_db", mock_get_db),
            patch("exo_web.routes.auth.settings", _FakeSettings()),
            patch("exo_web.routes.auth.audit_log", AsyncMock()),
        ):
            resp = _client().post(
                "/api/v1/auth/login",
                json={"email": EMAIL, "password": FAKE_PASSWORD},
            )

        assert resp.status_code == 200
        body = resp.json()
        assert body["id"] == USER_ID
        assert body["email"] == EMAIL
        assert "exo_session" in resp.cookies

    def test_wrong_password(self) -> None:
        login_row = {
            "id": USER_ID,
            "email": EMAIL,
            "password_hash": FAKE_HASH,
            "role": "developer",
            "created_at": CREATED_AT,
        }
        mock_get_db = _make_get_db(login_row)
        with (
            patch("exo_web.routes.auth.get_db", mock_get_db),
            patch("exo_web.routes.auth.settings", _FakeSettings()),
            patch("exo_web.routes.auth.audit_log", AsyncMock()),
        ):
            resp = _client().post(
                "/api/v1/auth/login",
                json={"email": EMAIL, "password": "wrong-password"},
            )

        assert resp.status_code == 401

    def test_unknown_user(self) -> None:
        mock_get_db = _make_get_db(None)
        with patch("exo_web.routes.auth.get_db", mock_get_db):
            resp = _client().post(
                "/api/v1/auth/login",
                json={"email": EMAIL, "password": FAKE_PASSWORD},
            )

        assert resp.status_code == 401

    def test_missing_body_fields(self) -> None:
        resp = _client().post("/api/v1/auth/login", json={"email": EMAIL})
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# POST /logout
# ---------------------------------------------------------------------------


class TestLogout:
    def test_success_clears_cookie(self) -> None:
        mock_get_db = _make_get_db({"user_id": USER_ID})
        with (
            patch("exo_web.routes.auth.get_db", mock_get_db),
            patch("exo_web.routes.auth.audit_log", AsyncMock()),
        ):
            resp = _client().post(
                "/api/v1/auth/logout",
                cookies={"exo_session": SESSION_ID},
            )

        assert resp.status_code == 204
        # Cookie deletion sets an expired exo_session cookie in the response.
        assert "exo_session" in resp.headers.get("set-cookie", "")

    def test_no_cookie_is_noop(self) -> None:
        with patch("exo_web.routes.auth.get_db", _make_get_db()):
            resp = _client().post("/api/v1/auth/logout")

        assert resp.status_code == 204


# ---------------------------------------------------------------------------
# GET /me
# ---------------------------------------------------------------------------


class TestMe:
    def test_authenticated(self) -> None:
        mock_get_db = _make_get_db(_session_user_row())
        with patch("exo_web.routes.auth.get_db", mock_get_db):
            resp = _client().get("/api/v1/auth/me", cookies={"exo_session": SESSION_ID})

        assert resp.status_code == 200
        assert resp.json()["id"] == USER_ID

    def test_no_session_cookie(self) -> None:
        resp = _client().get("/api/v1/auth/me")
        assert resp.status_code == 401

    def test_expired_session(self) -> None:
        # An expired/invalid session yields no joined row (filtered by SQL).
        mock_get_db = _make_get_db(None)
        with patch("exo_web.routes.auth.get_db", mock_get_db):
            resp = _client().get("/api/v1/auth/me", cookies={"exo_session": SESSION_ID})

        assert resp.status_code == 401


# ---------------------------------------------------------------------------
# GET /csrf
# ---------------------------------------------------------------------------


class TestCsrf:
    def test_valid_session(self) -> None:
        mock_get_db = _make_get_db({"csrf_token": "csrf-abc"})
        with patch("exo_web.routes.auth.get_db", mock_get_db):
            resp = _client().get("/api/v1/auth/csrf", cookies={"exo_session": SESSION_ID})

        assert resp.status_code == 200
        assert resp.json()["token"] == "csrf-abc"

    def test_no_session(self) -> None:
        resp = _client().get("/api/v1/auth/csrf")
        assert resp.status_code == 401


# ---------------------------------------------------------------------------
# POST /forgot-password
# ---------------------------------------------------------------------------


class TestForgotPassword:
    _MESSAGE = "If that email exists, a reset link has been sent"

    def test_known_email(self) -> None:
        mock_get_db = _make_get_db({"id": USER_ID}, None)
        with patch("exo_web.routes.auth.get_db", mock_get_db):
            resp = _client().post(
                "/api/v1/auth/forgot-password",
                json={"email": EMAIL},
            )

        assert resp.status_code == 200
        assert resp.json()["message"] == self._MESSAGE

    def test_unknown_email_same_message(self) -> None:
        mock_get_db = _make_get_db(None)
        with patch("exo_web.routes.auth.get_db", mock_get_db):
            resp = _client().post(
                "/api/v1/auth/forgot-password",
                json={"email": "nobody@example.com"},
            )

        assert resp.status_code == 200
        assert resp.json()["message"] == self._MESSAGE


# ---------------------------------------------------------------------------
# POST /reset-password
# ---------------------------------------------------------------------------


class TestResetPassword:
    def test_valid_token(self) -> None:
        token = "reset-token-value"
        token_hash = bcrypt.hashpw(token.encode(), bcrypt.gensalt()).decode()
        reset_row = _FakeRow({"id": "reset-001", "user_id": USER_ID, "token_hash": token_hash})
        # First execute (SELECT) returns fetchall list of reset rows; the
        # subsequent UPDATE/DELETE calls don't use it.
        mock_get_db = _make_get_db(fetchall=[reset_row])
        with patch("exo_web.routes.auth.get_db", mock_get_db):
            resp = _client().post(
                "/api/v1/auth/reset-password",
                json={"token": token, "new_password": "new-password-456"},
            )

        assert resp.status_code == 200
        assert resp.json()["message"] == "Password updated"

    def test_invalid_token(self) -> None:
        token_hash = bcrypt.hashpw(b"different-token", bcrypt.gensalt()).decode()
        reset_row = _FakeRow({"id": "reset-001", "user_id": USER_ID, "token_hash": token_hash})
        mock_get_db = _make_get_db(fetchall=[reset_row])
        with patch("exo_web.routes.auth.get_db", mock_get_db):
            resp = _client().post(
                "/api/v1/auth/reset-password",
                json={"token": "wrong-token", "new_password": "new-password-456"},
            )

        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# GET /profile
# ---------------------------------------------------------------------------


class TestGetProfile:
    def test_authenticated(self) -> None:
        mock_get_db = _make_get_db(_session_user_row())
        with patch("exo_web.routes.auth.get_db", mock_get_db):
            resp = _client().get("/api/v1/auth/profile", cookies={"exo_session": SESSION_ID})

        assert resp.status_code == 200
        assert resp.json()["email"] == EMAIL

    def test_unauthenticated(self) -> None:
        resp = _client().get("/api/v1/auth/profile")
        assert resp.status_code == 401


# ---------------------------------------------------------------------------
# PUT /profile
# ---------------------------------------------------------------------------


class TestUpdateProfile:
    def test_success(self) -> None:
        new_email = "new@example.com"
        mock_get_db = _make_get_db(
            _session_user_row(),  # get_current_user
            None,  # uniqueness check -> no conflict
            None,  # UPDATE
        )
        with (
            patch("exo_web.routes.auth.get_db", mock_get_db),
            patch("exo_web.routes.auth.audit_log", AsyncMock()),
        ):
            resp = _client().put(
                "/api/v1/auth/profile",
                json={"email": new_email},
                cookies={"exo_session": SESSION_ID},
            )

        assert resp.status_code == 200
        assert resp.json()["email"] == new_email

    def test_email_in_use(self) -> None:
        mock_get_db = _make_get_db(
            _session_user_row(),  # get_current_user
            {"id": "other-user"},  # uniqueness check -> conflict
        )
        with (
            patch("exo_web.routes.auth.get_db", mock_get_db),
            patch("exo_web.routes.auth.audit_log", AsyncMock()),
        ):
            resp = _client().put(
                "/api/v1/auth/profile",
                json={"email": "taken@example.com"},
                cookies={"exo_session": SESSION_ID},
            )

        assert resp.status_code == 409


# ---------------------------------------------------------------------------
# PUT /password
# ---------------------------------------------------------------------------


class TestChangePassword:
    def test_success(self) -> None:
        mock_get_db = _make_get_db(
            _session_user_row(),  # get_current_user
            {"password_hash": FAKE_HASH},  # current hash lookup
            None,  # UPDATE
            None,  # DELETE other sessions
        )
        with patch("exo_web.routes.auth.get_db", mock_get_db):
            resp = _client().put(
                "/api/v1/auth/password",
                json={
                    "current_password": FAKE_PASSWORD,
                    "new_password": "new-password-789",
                },
                cookies={"exo_session": SESSION_ID},
            )

        assert resp.status_code == 200
        assert resp.json()["message"] == "Password updated"

    def test_wrong_current_password(self) -> None:
        mock_get_db = _make_get_db(
            _session_user_row(),  # get_current_user
            {"password_hash": FAKE_HASH},  # current hash lookup
        )
        with patch("exo_web.routes.auth.get_db", mock_get_db):
            resp = _client().put(
                "/api/v1/auth/password",
                json={
                    "current_password": "not-the-password",
                    "new_password": "new-password-789",
                },
                cookies={"exo_session": SESSION_ID},
            )

        assert resp.status_code == 401


# ---------------------------------------------------------------------------
# require_role dependency
# ---------------------------------------------------------------------------


class TestRequireRole:
    def test_viewer_denied_developer_level(self) -> None:
        mock_get_db = _make_get_db(_session_user_row(role="viewer"))
        with patch("exo_web.routes.auth.get_db", mock_get_db):
            resp = _client().get(
                "/_test/developer-only",
                cookies={"exo_session": SESSION_ID},
            )

        assert resp.status_code == 403

    def test_admin_allowed_admin_level(self) -> None:
        mock_get_db = _make_get_db(_session_user_row(role="admin"))
        with patch("exo_web.routes.auth.get_db", mock_get_db):
            resp = _client().get(
                "/_test/admin-only",
                cookies={"exo_session": SESSION_ID},
            )

        assert resp.status_code == 200
        assert resp.json()["role"] == "admin"
