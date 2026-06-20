"""Tests for the rate-limiting middleware (middleware/rate_limit.py).

Covers: sliding window logic, auth/agent/general limits, X-RateLimit headers,
429 response format, X-Forwarded-For trust gating, lock behavior, and cleanup.
"""

from __future__ import annotations

import time
from typing import Any
from unittest.mock import MagicMock, patch

from starlette.requests import Request
from starlette.responses import Response

from exo_web.middleware.rate_limit import (
    RateLimitMiddleware,
    _get_client_ip,
    _get_user_key,
    _is_agent_exec,
    _SlidingWindow,
)

# ---------------------------------------------------------------------------
# _SlidingWindow unit tests
# ---------------------------------------------------------------------------


class TestSlidingWindow:
    def test_first_hit_allowed(self) -> None:
        sw = _SlidingWindow()
        allowed, remaining, _reset = sw.hit("k", time.monotonic(), 5)
        assert allowed is True
        assert remaining == 4

    def test_hit_up_to_limit(self) -> None:
        sw = _SlidingWindow()
        now = time.monotonic()
        for _i in range(5):
            allowed, _, _ = sw.hit("k", now, 5)
            assert allowed is True

    def test_hit_over_limit_denied(self) -> None:
        sw = _SlidingWindow()
        now = time.monotonic()
        for _ in range(5):
            sw.hit("k", now, 5)
        allowed, remaining, _ = sw.hit("k", now, 5)
        assert allowed is False
        assert remaining == 0

    def test_hits_outside_window_expire(self) -> None:
        sw = _SlidingWindow()
        old_time = time.monotonic() - 120  # 2 minutes ago (outside 60s window)
        for _ in range(5):
            sw.hit("k", old_time, 5)
        # Now a hit at current time should be allowed (old hits have expired)
        allowed, _, _ = sw.hit("k", time.monotonic(), 5)
        assert allowed is True

    def test_cleanup_removes_stale_keys(self) -> None:
        sw = _SlidingWindow()
        old = time.monotonic() - 120
        sw.hit("stale", old, 10)
        assert "stale" in sw._hits
        sw.cleanup(time.monotonic())
        assert "stale" not in sw._hits

    def test_cleanup_keeps_recent_keys(self) -> None:
        sw = _SlidingWindow()
        now = time.monotonic()
        sw.hit("active", now, 10)
        sw.cleanup(now)
        assert "active" in sw._hits

    def test_reset_seconds_positive(self) -> None:
        sw = _SlidingWindow()
        _, _, reset = sw.hit("k", time.monotonic(), 5)
        assert reset > 0

    def test_remaining_decrements(self) -> None:
        sw = _SlidingWindow()
        now = time.monotonic()
        _, r1, _ = sw.hit("k", now, 5)
        _, r2, _ = sw.hit("k", now, 5)
        assert r1 == 4
        assert r2 == 3


# ---------------------------------------------------------------------------
# _is_agent_exec helper
# ---------------------------------------------------------------------------


class TestIsAgentExec:
    def test_workflow_run_detected(self) -> None:
        assert _is_agent_exec("/api/v1/workflows/abc/run") is True

    def test_workflow_debug_detected(self) -> None:
        assert _is_agent_exec("/api/v1/workflows/abc/debug") is True

    def test_non_exec_path_not_detected(self) -> None:
        assert _is_agent_exec("/api/v1/agents") is False

    def test_get_workflow_not_detected(self) -> None:
        assert _is_agent_exec("/api/v1/workflows/abc") is False


# ---------------------------------------------------------------------------
# _get_client_ip
# ---------------------------------------------------------------------------


class TestGetClientIp:
    def _req(
        self,
        client_host: str = "1.2.3.4",
        forwarded_for: str | None = None,
    ) -> MagicMock:
        req = MagicMock(spec=Request)
        req.client = MagicMock()
        req.client.host = client_host
        req.headers = MagicMock()
        req.headers.get = lambda key, default=None: (
            forwarded_for if key == "x-forwarded-for" else default
        )
        return req

    def test_direct_ip_when_trust_proxy_false(self) -> None:
        req = self._req(client_host="1.2.3.4", forwarded_for="9.9.9.9")
        with patch("exo_web.middleware.rate_limit._TRUST_PROXY", False):
            ip = _get_client_ip(req)
        assert ip == "1.2.3.4"

    def test_forwarded_for_used_when_trust_proxy_true(self) -> None:
        req = self._req(client_host="1.2.3.4", forwarded_for="9.9.9.9, 10.0.0.1")
        with patch("exo_web.middleware.rate_limit._TRUST_PROXY", True):
            ip = _get_client_ip(req)
        assert ip == "9.9.9.9"

    def test_forwarded_for_ignored_when_trust_proxy_false(self) -> None:
        req = self._req(client_host="1.2.3.4", forwarded_for="9.9.9.9")
        with patch("exo_web.middleware.rate_limit._TRUST_PROXY", False):
            ip = _get_client_ip(req)
        assert ip != "9.9.9.9"

    def test_no_client_returns_unknown(self) -> None:
        req = MagicMock(spec=Request)
        req.client = None
        req.headers = MagicMock()
        req.headers.get = lambda key, default=None: None
        with patch("exo_web.middleware.rate_limit._TRUST_PROXY", False):
            ip = _get_client_ip(req)
        assert ip == "unknown"


# ---------------------------------------------------------------------------
# _get_user_key
# ---------------------------------------------------------------------------


class TestGetUserKey:
    def _req(self, session_id: str | None = None, client_host: str = "1.2.3.4") -> MagicMock:
        req = MagicMock(spec=Request)
        req.cookies = {"exo_session": session_id} if session_id else {}
        req.client = MagicMock()
        req.client.host = client_host
        req.headers = MagicMock()
        req.headers.get = lambda key, default=None: None
        return req

    def test_session_key_when_cookie_present(self) -> None:
        req = self._req(session_id="mysession")
        with patch("exo_web.middleware.rate_limit._TRUST_PROXY", False):
            key = _get_user_key(req)
        assert key == "session:mysession"

    def test_ip_key_when_no_cookie(self) -> None:
        req = self._req(session_id=None, client_host="5.6.7.8")
        with patch("exo_web.middleware.rate_limit._TRUST_PROXY", False):
            key = _get_user_key(req)
        assert key.startswith("ip:")
        assert "5.6.7.8" in key


# ---------------------------------------------------------------------------
# Helpers for middleware dispatch tests
# ---------------------------------------------------------------------------


def _make_request(
    method: str = "GET",
    path: str = "/api/v1/agents",
    cookies: dict[str, str] | None = None,
    headers: dict[str, str] | None = None,
    client_host: str = "127.0.0.1",
) -> MagicMock:
    req = MagicMock(spec=Request)
    req.method = method
    req.url = MagicMock()
    req.url.path = path
    req.cookies = cookies or {}
    req.client = MagicMock()
    req.client.host = client_host

    hdr_dict = headers or {}

    def hdr_get(key: str, default: str = "") -> str:
        return hdr_dict.get(key, default)

    req.headers = MagicMock()
    req.headers.get = hdr_get
    return req


async def _ok_next(request: Any) -> Response:
    return Response(content="ok", status_code=200)


# ---------------------------------------------------------------------------
# Non-API paths pass through (no rate limiting)
# ---------------------------------------------------------------------------


class TestNonApiPaths:
    async def test_static_path_not_limited(self) -> None:
        mw = RateLimitMiddleware(MagicMock())
        req = _make_request(path="/static/app.js")
        resp = await mw.dispatch(req, _ok_next)
        assert resp.status_code == 200

    async def test_health_check_not_limited(self) -> None:
        mw = RateLimitMiddleware(MagicMock())
        req = _make_request(path="/api/health")
        resp = await mw.dispatch(req, _ok_next)
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Auth endpoint rate limiting
# ---------------------------------------------------------------------------


class TestAuthRateLimit:
    async def test_auth_requests_allowed_within_limit(self) -> None:
        from exo_web.middleware.rate_limit import _SlidingWindow

        # Fresh window so previous test hits don't affect us.
        with (
            patch("exo_web.middleware.rate_limit._auth_window", _SlidingWindow()),
            patch("exo_web.middleware.rate_limit._user_window", _SlidingWindow()),
        ):
            mw = RateLimitMiddleware(MagicMock())

            with patch("exo_web.middleware.rate_limit.settings") as mock_settings:
                mock_settings.rate_limit_auth = 10
                mock_settings.rate_limit_agent = 10
                mock_settings.rate_limit_general = 200

                req = _make_request(method="POST", path="/api/v1/auth/login")
                resp = await mw.dispatch(req, _ok_next)

            assert resp.status_code == 200
            assert "X-RateLimit-Limit" in resp.headers

    async def test_auth_requests_blocked_over_limit(self) -> None:
        from exo_web.middleware.rate_limit import _SlidingWindow

        fresh_auth = _SlidingWindow()
        fresh_user = _SlidingWindow()

        with (
            patch("exo_web.middleware.rate_limit._auth_window", fresh_auth),
            patch("exo_web.middleware.rate_limit._user_window", fresh_user),
            patch("exo_web.middleware.rate_limit.settings") as mock_settings,
            patch("exo_web.middleware.rate_limit._TRUST_PROXY", False),
        ):
            mock_settings.rate_limit_auth = 2
            mock_settings.rate_limit_agent = 10
            mock_settings.rate_limit_general = 200

            mw = RateLimitMiddleware(MagicMock())
            req = _make_request(method="POST", path="/api/v1/auth/login", client_host="99.0.0.1")

            # Exhaust limit
            for _ in range(2):
                await mw.dispatch(req, _ok_next)

            # This one should be blocked
            resp = await mw.dispatch(req, _ok_next)

        assert resp.status_code == 429
        assert "Retry-After" in resp.headers

    async def test_rate_limit_headers_on_auth_response(self) -> None:
        from exo_web.middleware.rate_limit import _SlidingWindow

        with (
            patch("exo_web.middleware.rate_limit._auth_window", _SlidingWindow()),
            patch("exo_web.middleware.rate_limit._user_window", _SlidingWindow()),
            patch("exo_web.middleware.rate_limit.settings") as mock_settings,
        ):
            mock_settings.rate_limit_auth = 100
            mock_settings.rate_limit_agent = 10
            mock_settings.rate_limit_general = 200

            mw = RateLimitMiddleware(MagicMock())
            req = _make_request(method="POST", path="/api/v1/auth/login")
            resp = await mw.dispatch(req, _ok_next)

        assert "X-RateLimit-Limit" in resp.headers
        assert "X-RateLimit-Remaining" in resp.headers
        assert "X-RateLimit-Reset" in resp.headers


# ---------------------------------------------------------------------------
# Agent execution rate limiting
# ---------------------------------------------------------------------------


class TestAgentExecRateLimit:
    async def test_agent_exec_allowed_within_limit(self) -> None:
        from exo_web.middleware.rate_limit import _SlidingWindow

        with (
            patch("exo_web.middleware.rate_limit._auth_window", _SlidingWindow()),
            patch("exo_web.middleware.rate_limit._user_window", _SlidingWindow()),
            patch("exo_web.middleware.rate_limit.settings") as mock_settings,
        ):
            mock_settings.rate_limit_auth = 5
            mock_settings.rate_limit_agent = 50
            mock_settings.rate_limit_general = 200

            mw = RateLimitMiddleware(MagicMock())
            req = _make_request(
                method="POST",
                path="/api/v1/workflows/abc/run",
                cookies={"exo_session": "sess-u1"},
            )
            resp = await mw.dispatch(req, _ok_next)

        assert resp.status_code == 200

    async def test_agent_exec_blocked_over_limit(self) -> None:
        from exo_web.middleware.rate_limit import _SlidingWindow

        fresh_user = _SlidingWindow()

        with (
            patch("exo_web.middleware.rate_limit._auth_window", _SlidingWindow()),
            patch("exo_web.middleware.rate_limit._user_window", fresh_user),
            patch("exo_web.middleware.rate_limit.settings") as mock_settings,
        ):
            mock_settings.rate_limit_auth = 5
            mock_settings.rate_limit_agent = 2
            mock_settings.rate_limit_general = 200

            mw = RateLimitMiddleware(MagicMock())
            req = _make_request(
                method="POST",
                path="/api/v1/workflows/abc/run",
                cookies={"exo_session": "sess-agent-test"},
            )

            for _ in range(2):
                await mw.dispatch(req, _ok_next)

            resp = await mw.dispatch(req, _ok_next)

        assert resp.status_code == 429


# ---------------------------------------------------------------------------
# General API rate limiting
# ---------------------------------------------------------------------------


class TestGeneralRateLimit:
    async def test_general_allowed_within_limit(self) -> None:
        from exo_web.middleware.rate_limit import _SlidingWindow

        with (
            patch("exo_web.middleware.rate_limit._auth_window", _SlidingWindow()),
            patch("exo_web.middleware.rate_limit._user_window", _SlidingWindow()),
            patch("exo_web.middleware.rate_limit.settings") as mock_settings,
        ):
            mock_settings.rate_limit_auth = 5
            mock_settings.rate_limit_agent = 10
            mock_settings.rate_limit_general = 200

            mw = RateLimitMiddleware(MagicMock())
            req = _make_request(
                method="GET",
                path="/api/v1/projects",
                cookies={"exo_session": "sess-general"},
            )
            resp = await mw.dispatch(req, _ok_next)

        assert resp.status_code == 200
        assert "X-RateLimit-Limit" in resp.headers

    async def test_general_blocked_over_limit(self) -> None:
        from exo_web.middleware.rate_limit import _SlidingWindow

        fresh_user = _SlidingWindow()

        with (
            patch("exo_web.middleware.rate_limit._auth_window", _SlidingWindow()),
            patch("exo_web.middleware.rate_limit._user_window", fresh_user),
            patch("exo_web.middleware.rate_limit.settings") as mock_settings,
        ):
            mock_settings.rate_limit_auth = 5
            mock_settings.rate_limit_agent = 10
            mock_settings.rate_limit_general = 1  # very low limit

            mw = RateLimitMiddleware(MagicMock())
            req = _make_request(
                method="GET",
                path="/api/v1/projects",
                cookies={"exo_session": "sess-tight"},
            )

            await mw.dispatch(req, _ok_next)  # consume the 1 allowed hit
            resp = await mw.dispatch(req, _ok_next)

        assert resp.status_code == 429

    async def test_429_response_body(self) -> None:
        import json

        from exo_web.middleware.rate_limit import _rate_limited_response

        resp = _rate_limited_response(5, 30)
        assert resp.status_code == 429
        data = json.loads(resp.body)
        assert "Too many requests" in data["detail"]
        assert resp.headers["Retry-After"] == "30"
        assert resp.headers["X-RateLimit-Limit"] == "5"
        assert resp.headers["X-RateLimit-Remaining"] == "0"


# ---------------------------------------------------------------------------
# X-Forwarded-For only trusted when EXO_TRUST_PROXY is set
# ---------------------------------------------------------------------------


class TestTrustProxy:
    def test_xff_ignored_without_trust_proxy(self) -> None:
        req = _make_request(
            method="POST",
            path="/api/v1/auth/login",
            headers={"x-forwarded-for": "1.1.1.1"},
            client_host="127.0.0.1",
        )
        with patch("exo_web.middleware.rate_limit._TRUST_PROXY", False):
            ip = _get_client_ip(req)
        assert ip == "127.0.0.1"

    def test_xff_used_with_trust_proxy(self) -> None:
        req = _make_request(
            method="POST",
            path="/api/v1/auth/login",
            headers={"x-forwarded-for": "1.1.1.1"},
            client_host="127.0.0.1",
        )
        with patch("exo_web.middleware.rate_limit._TRUST_PROXY", True):
            ip = _get_client_ip(req)
        assert ip == "1.1.1.1"

    def test_xff_multi_value_takes_first(self) -> None:
        req = _make_request(
            method="POST",
            path="/api/v1/auth/login",
            headers={"x-forwarded-for": "2.2.2.2, 3.3.3.3"},
            client_host="127.0.0.1",
        )
        with patch("exo_web.middleware.rate_limit._TRUST_PROXY", True):
            ip = _get_client_ip(req)
        assert ip == "2.2.2.2"
