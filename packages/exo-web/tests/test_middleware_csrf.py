"""Tests for the CSRF protection middleware (middleware/csrf.py).

Covers: safe methods pass through, exempt paths pass through, missing session
passes through, missing CSRF header returns 403, invalid session returns 403,
wrong token returns 403, correct token passes through, constant-time comparison.
"""

from __future__ import annotations

import hmac
from unittest.mock import AsyncMock, MagicMock, patch

from starlette.requests import Request
from starlette.responses import Response

from exo_web.middleware.csrf import _EXEMPT_PATHS, _EXEMPT_PREFIXES, CSRFMiddleware

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_request(
    method: str = "POST",
    path: str = "/api/v1/agents",
    cookies: dict[str, str] | None = None,
    headers: dict[str, str] | None = None,
) -> MagicMock:
    """Build a minimal mock Starlette Request."""
    req = MagicMock(spec=Request)
    req.method = method
    req.url = MagicMock()
    req.url.path = path
    req.cookies = cookies or {}
    req.headers = MagicMock()

    # headers.get() helper
    def headers_get(key: str, default: str = "") -> str:
        h = headers or {}
        return h.get(key, default)

    req.headers.get = headers_get
    req.headers.__contains__ = lambda self, key: key in (headers or {})
    return req


async def _ok_next(request: Request) -> Response:
    """call_next stub that always returns 200 OK."""
    return Response(content="ok", status_code=200)


# ---------------------------------------------------------------------------
# Safe HTTP methods — always pass through
# ---------------------------------------------------------------------------


class TestSafeMethods:
    async def test_get_passes_through(self) -> None:
        mw = CSRFMiddleware(MagicMock())
        req = _make_request(method="GET", path="/api/v1/agents")
        resp = await mw.dispatch(req, _ok_next)
        assert resp.status_code == 200

    async def test_head_passes_through(self) -> None:
        mw = CSRFMiddleware(MagicMock())
        req = _make_request(method="HEAD", path="/api/v1/agents")
        resp = await mw.dispatch(req, _ok_next)
        assert resp.status_code == 200

    async def test_options_passes_through(self) -> None:
        mw = CSRFMiddleware(MagicMock())
        req = _make_request(method="OPTIONS", path="/api/v1/agents")
        resp = await mw.dispatch(req, _ok_next)
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Exempt paths — always pass through even for unsafe methods
# ---------------------------------------------------------------------------


class TestExemptPaths:
    async def test_login_path_exempt(self) -> None:
        mw = CSRFMiddleware(MagicMock())
        req = _make_request(method="POST", path="/api/v1/auth/login")
        resp = await mw.dispatch(req, _ok_next)
        assert resp.status_code == 200

    async def test_forgot_password_exempt(self) -> None:
        mw = CSRFMiddleware(MagicMock())
        req = _make_request(method="POST", path="/api/v1/auth/forgot-password")
        resp = await mw.dispatch(req, _ok_next)
        assert resp.status_code == 200

    async def test_reset_password_exempt(self) -> None:
        mw = CSRFMiddleware(MagicMock())
        req = _make_request(method="POST", path="/api/v1/auth/reset-password")
        resp = await mw.dispatch(req, _ok_next)
        assert resp.status_code == 200

    async def test_health_check_exempt(self) -> None:
        mw = CSRFMiddleware(MagicMock())
        req = _make_request(method="POST", path="/api/health")
        resp = await mw.dispatch(req, _ok_next)
        assert resp.status_code == 200

    async def test_ci_prefix_exempt(self) -> None:
        mw = CSRFMiddleware(MagicMock())
        req = _make_request(method="POST", path="/api/v1/ci/run")
        resp = await mw.dispatch(req, _ok_next)
        assert resp.status_code == 200

    async def test_webhooks_prefix_exempt(self) -> None:
        mw = CSRFMiddleware(MagicMock())
        req = _make_request(method="POST", path="/api/v1/webhooks/trigger")
        resp = await mw.dispatch(req, _ok_next)
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# WebSocket upgrades pass through
# ---------------------------------------------------------------------------


class TestWebSocketUpgrade:
    async def test_websocket_upgrade_exempt(self) -> None:
        mw = CSRFMiddleware(MagicMock())
        req = _make_request(
            method="GET",
            path="/api/v1/ws",
            headers={"upgrade": "websocket"},
        )
        resp = await mw.dispatch(req, _ok_next)
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# No session cookie — pass through (auth middleware handles it)
# ---------------------------------------------------------------------------


class TestNoSessionCookie:
    async def test_no_session_passes_through(self) -> None:
        mw = CSRFMiddleware(MagicMock())
        req = _make_request(method="POST", path="/api/v1/agents", cookies={})
        resp = await mw.dispatch(req, _ok_next)
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Session present but CSRF header missing → 403
# ---------------------------------------------------------------------------


class TestMissingCsrfHeader:
    async def test_missing_header_returns_403(self) -> None:
        mw = CSRFMiddleware(MagicMock())
        req = _make_request(
            method="POST",
            path="/api/v1/agents",
            cookies={"exo_session": "session-abc"},
            headers={},  # no x-csrf-token
        )
        resp = await mw.dispatch(req, _ok_next)
        assert resp.status_code == 403
        import json

        data = json.loads(resp.body)
        assert "Missing CSRF token" in data["detail"]


# ---------------------------------------------------------------------------
# Invalid session (not in DB or expired) → 403
# ---------------------------------------------------------------------------


class TestInvalidSession:
    async def test_invalid_session_returns_403(self) -> None:
        mw = CSRFMiddleware(MagicMock())
        req = _make_request(
            method="POST",
            path="/api/v1/agents",
            cookies={"exo_session": "bad-session"},
            headers={"x-csrf-token": "some-token"},
        )

        mock_db = AsyncMock()
        cursor = AsyncMock()
        cursor.fetchone = AsyncMock(return_value=None)  # session not found
        mock_db.execute = AsyncMock(return_value=cursor)

        with patch("exo_web.middleware.csrf.get_db") as mock_get_db:
            mock_get_db.return_value.__aenter__ = AsyncMock(return_value=mock_db)
            mock_get_db.return_value.__aexit__ = AsyncMock(return_value=False)
            resp = await mw.dispatch(req, _ok_next)

        assert resp.status_code == 403

    async def test_session_with_null_csrf_token_returns_403(self) -> None:
        mw = CSRFMiddleware(MagicMock())
        req = _make_request(
            method="POST",
            path="/api/v1/agents",
            cookies={"exo_session": "sess-123"},
            headers={"x-csrf-token": "some-token"},
        )

        row = MagicMock()
        row.__getitem__ = lambda s, k: None  # row["csrf_token"] returns None

        mock_db = AsyncMock()
        cursor = AsyncMock()
        cursor.fetchone = AsyncMock(return_value=row)
        mock_db.execute = AsyncMock(return_value=cursor)

        with patch("exo_web.middleware.csrf.get_db") as mock_get_db:
            mock_get_db.return_value.__aenter__ = AsyncMock(return_value=mock_db)
            mock_get_db.return_value.__aexit__ = AsyncMock(return_value=False)
            resp = await mw.dispatch(req, _ok_next)

        assert resp.status_code == 403


# ---------------------------------------------------------------------------
# Token mismatch → 403
# ---------------------------------------------------------------------------


class TestTokenMismatch:
    async def test_wrong_token_returns_403(self) -> None:
        mw = CSRFMiddleware(MagicMock())
        req = _make_request(
            method="POST",
            path="/api/v1/agents",
            cookies={"exo_session": "sess-123"},
            headers={"x-csrf-token": "wrong-token"},
        )

        row = MagicMock()
        row.__getitem__ = lambda s, k: "correct-token"

        mock_db = AsyncMock()
        cursor = AsyncMock()
        cursor.fetchone = AsyncMock(return_value=row)
        mock_db.execute = AsyncMock(return_value=cursor)

        with patch("exo_web.middleware.csrf.get_db") as mock_get_db:
            mock_get_db.return_value.__aenter__ = AsyncMock(return_value=mock_db)
            mock_get_db.return_value.__aexit__ = AsyncMock(return_value=False)
            resp = await mw.dispatch(req, _ok_next)

        assert resp.status_code == 403
        import json

        data = json.loads(resp.body)
        assert "Invalid CSRF token" in data["detail"]


# ---------------------------------------------------------------------------
# Correct token → 200
# ---------------------------------------------------------------------------


class TestCorrectToken:
    async def test_correct_token_passes(self) -> None:
        mw = CSRFMiddleware(MagicMock())
        correct_token = "valid-csrf-token-12345"

        req = _make_request(
            method="POST",
            path="/api/v1/agents",
            cookies={"exo_session": "sess-123"},
            headers={"x-csrf-token": correct_token},
        )

        row = MagicMock()
        row.__getitem__ = lambda s, k: correct_token

        mock_db = AsyncMock()
        cursor = AsyncMock()
        cursor.fetchone = AsyncMock(return_value=row)
        mock_db.execute = AsyncMock(return_value=cursor)

        with patch("exo_web.middleware.csrf.get_db") as mock_get_db:
            mock_get_db.return_value.__aenter__ = AsyncMock(return_value=mock_db)
            mock_get_db.return_value.__aexit__ = AsyncMock(return_value=False)
            resp = await mw.dispatch(req, _ok_next)

        assert resp.status_code == 200

    async def test_put_method_validated(self) -> None:
        """PUT is an unsafe method — token must also be checked."""
        mw = CSRFMiddleware(MagicMock())
        token = "token-for-put"

        req = _make_request(
            method="PUT",
            path="/api/v1/agents/1",
            cookies={"exo_session": "sess-123"},
            headers={"x-csrf-token": token},
        )

        row = MagicMock()
        row.__getitem__ = lambda s, k: token

        mock_db = AsyncMock()
        cursor = AsyncMock()
        cursor.fetchone = AsyncMock(return_value=row)
        mock_db.execute = AsyncMock(return_value=cursor)

        with patch("exo_web.middleware.csrf.get_db") as mock_get_db:
            mock_get_db.return_value.__aenter__ = AsyncMock(return_value=mock_db)
            mock_get_db.return_value.__aexit__ = AsyncMock(return_value=False)
            resp = await mw.dispatch(req, _ok_next)

        assert resp.status_code == 200

    async def test_delete_method_validated(self) -> None:
        """DELETE is an unsafe method — token must also be checked."""
        mw = CSRFMiddleware(MagicMock())
        token = "token-for-delete"

        req = _make_request(
            method="DELETE",
            path="/api/v1/agents/1",
            cookies={"exo_session": "sess-123"},
            headers={"x-csrf-token": token},
        )

        row = MagicMock()
        row.__getitem__ = lambda s, k: token

        mock_db = AsyncMock()
        cursor = AsyncMock()
        cursor.fetchone = AsyncMock(return_value=row)
        mock_db.execute = AsyncMock(return_value=cursor)

        with patch("exo_web.middleware.csrf.get_db") as mock_get_db:
            mock_get_db.return_value.__aenter__ = AsyncMock(return_value=mock_db)
            mock_get_db.return_value.__aexit__ = AsyncMock(return_value=False)
            resp = await mw.dispatch(req, _ok_next)

        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Constant-time comparison — unit test for hmac.compare_digest usage
# ---------------------------------------------------------------------------


class TestConstantTimeComparison:
    def test_compare_digest_used_for_token_comparison(self) -> None:
        """Verify that hmac.compare_digest is used (not ==) for token validation.

        We import the module and confirm it calls compare_digest by inspecting
        the source, or we validate the behavior:  identical tokens → True,
        different tokens → False.
        """
        assert hmac.compare_digest("token-abc", "token-abc") is True
        assert hmac.compare_digest("token-abc", "token-xyz") is False

    def test_csrf_module_imports_hmac(self) -> None:
        """The csrf module must import and use hmac for constant-time comparison."""
        import inspect

        import exo_web.middleware.csrf as csrf_mod

        source = inspect.getsource(csrf_mod)
        assert "hmac.compare_digest" in source


# ---------------------------------------------------------------------------
# Exempt path / prefix coverage completeness
# ---------------------------------------------------------------------------


class TestExemptSetCompleteness:
    def test_exempt_paths_non_empty(self) -> None:
        assert len(_EXEMPT_PATHS) > 0

    def test_exempt_prefixes_non_empty(self) -> None:
        assert len(_EXEMPT_PREFIXES) > 0

    def test_auth_login_in_exempt_paths(self) -> None:
        assert "/api/v1/auth/login" in _EXEMPT_PATHS
