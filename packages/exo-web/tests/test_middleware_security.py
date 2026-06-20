"""Tests for the security headers middleware (middleware/security.py).

Covers: CSP header presence, X-Content-Type-Options nosniff, X-Frame-Options
DENY for regular paths, SAMEORIGIN for frameable paths, and header values.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from starlette.requests import Request
from starlette.responses import Response

from exo_web.middleware.security import (
    _CSP,
    _FRAMEABLE_PATHS,
    SecurityHeadersMiddleware,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_request(path: str = "/api/v1/agents") -> MagicMock:
    req = MagicMock(spec=Request)
    req.url = MagicMock()
    req.url.path = path
    return req


async def _ok_next(request: Any) -> Response:
    return Response(content="ok", status_code=200)


# ---------------------------------------------------------------------------
# Content-Security-Policy
# ---------------------------------------------------------------------------


class TestContentSecurityPolicy:
    async def test_csp_header_present(self) -> None:
        mw = SecurityHeadersMiddleware(MagicMock())
        req = _make_request("/api/v1/test")
        resp = await mw.dispatch(req, _ok_next)
        assert "Content-Security-Policy" in resp.headers

    async def test_csp_value_matches_module_constant(self) -> None:
        mw = SecurityHeadersMiddleware(MagicMock())
        req = _make_request("/api/v1/test")
        resp = await mw.dispatch(req, _ok_next)
        assert resp.headers["Content-Security-Policy"] == _CSP

    async def test_csp_contains_default_src_self(self) -> None:
        mw = SecurityHeadersMiddleware(MagicMock())
        req = _make_request("/api/v1/test")
        resp = await mw.dispatch(req, _ok_next)
        assert "default-src 'self'" in resp.headers["Content-Security-Policy"]

    async def test_csp_contains_script_src_self(self) -> None:
        mw = SecurityHeadersMiddleware(MagicMock())
        req = _make_request("/api/v1/test")
        resp = await mw.dispatch(req, _ok_next)
        assert "script-src 'self'" in resp.headers["Content-Security-Policy"]

    async def test_csp_constant_non_empty(self) -> None:
        assert len(_CSP) > 0


# ---------------------------------------------------------------------------
# X-Content-Type-Options
# ---------------------------------------------------------------------------


class TestXContentTypeOptions:
    async def test_nosniff_header_present(self) -> None:
        mw = SecurityHeadersMiddleware(MagicMock())
        req = _make_request("/api/v1/test")
        resp = await mw.dispatch(req, _ok_next)
        assert "X-Content-Type-Options" in resp.headers

    async def test_nosniff_value(self) -> None:
        mw = SecurityHeadersMiddleware(MagicMock())
        req = _make_request("/api/v1/test")
        resp = await mw.dispatch(req, _ok_next)
        assert resp.headers["X-Content-Type-Options"] == "nosniff"

    async def test_nosniff_on_every_response(self) -> None:
        """Header must be present on multiple different paths."""
        mw = SecurityHeadersMiddleware(MagicMock())
        for path in ["/api/v1/agents", "/api/health", "/", "/static/app.js"]:
            req = _make_request(path)
            resp = await mw.dispatch(req, _ok_next)
            assert resp.headers.get("X-Content-Type-Options") == "nosniff", f"Missing on {path}"


# ---------------------------------------------------------------------------
# X-Frame-Options
# ---------------------------------------------------------------------------


class TestXFrameOptions:
    async def test_deny_for_regular_paths(self) -> None:
        mw = SecurityHeadersMiddleware(MagicMock())
        req = _make_request("/api/v1/agents")
        resp = await mw.dispatch(req, _ok_next)
        assert resp.headers["X-Frame-Options"] == "DENY"

    async def test_sameorigin_for_frameable_paths(self) -> None:
        """Paths registered in _FRAMEABLE_PATHS must get SAMEORIGIN."""
        test_path = "/widget/embed"
        _FRAMEABLE_PATHS.add(test_path)
        try:
            mw = SecurityHeadersMiddleware(MagicMock())
            req = _make_request(test_path)
            resp = await mw.dispatch(req, _ok_next)
            assert resp.headers["X-Frame-Options"] == "SAMEORIGIN"
        finally:
            _FRAMEABLE_PATHS.discard(test_path)

    async def test_deny_when_not_in_frameable_paths(self) -> None:
        mw = SecurityHeadersMiddleware(MagicMock())
        req = _make_request("/not/a/widget")
        resp = await mw.dispatch(req, _ok_next)
        assert resp.headers["X-Frame-Options"] == "DENY"

    async def test_frame_options_on_health_check(self) -> None:
        mw = SecurityHeadersMiddleware(MagicMock())
        req = _make_request("/api/health")
        resp = await mw.dispatch(req, _ok_next)
        assert "X-Frame-Options" in resp.headers


# ---------------------------------------------------------------------------
# Headers present on every response type
# ---------------------------------------------------------------------------


class TestHeadersOnAllResponses:
    async def test_headers_on_404_response(self) -> None:
        async def not_found_next(request: Any) -> Response:
            return Response(content="not found", status_code=404)

        mw = SecurityHeadersMiddleware(MagicMock())
        req = _make_request("/does/not/exist")
        resp = await mw.dispatch(req, not_found_next)
        assert resp.headers["X-Content-Type-Options"] == "nosniff"
        assert "Content-Security-Policy" in resp.headers

    async def test_headers_on_500_response(self) -> None:
        async def error_next(request: Any) -> Response:
            return Response(content="error", status_code=500)

        mw = SecurityHeadersMiddleware(MagicMock())
        req = _make_request("/api/v1/broken")
        resp = await mw.dispatch(req, error_next)
        assert resp.headers["X-Content-Type-Options"] == "nosniff"
        assert "Content-Security-Policy" in resp.headers

    async def test_headers_on_redirect_response(self) -> None:
        from starlette.responses import RedirectResponse

        async def redirect_next(request: Any) -> Response:
            return RedirectResponse(url="/new-path")

        mw = SecurityHeadersMiddleware(MagicMock())
        req = _make_request("/old-path")
        resp = await mw.dispatch(req, redirect_next)
        assert resp.headers["X-Content-Type-Options"] == "nosniff"


# ---------------------------------------------------------------------------
# CSP constant structure
# ---------------------------------------------------------------------------


class TestCspConstant:
    def test_csp_is_semicolon_separated(self) -> None:
        directives = [d.strip() for d in _CSP.split(";")]
        assert len(directives) >= 2

    def test_csp_has_style_src(self) -> None:
        assert "style-src" in _CSP

    def test_csp_has_img_src(self) -> None:
        assert "img-src" in _CSP
