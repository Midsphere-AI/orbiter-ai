"""Rate limiting middleware using in-memory sliding window counters."""

from __future__ import annotations

import asyncio
import os
import time

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from exo_web.config import settings

# Window size in seconds.
_WINDOW = 60

# Paths that use the stricter auth rate limit (per IP).
_AUTH_PATHS = {"/api/v1/auth/login"}

# Path prefixes that use the agent execution rate limit (per user).
_AGENT_EXEC_PREFIXES = (
    "/api/v1/workflows/",  # covers /{id}/run, /{id}/debug, /{id}/nodes/{id}/run
)

_AGENT_EXEC_SUFFIXES = ("/run", "/debug")


def _is_agent_exec(path: str) -> bool:
    """Return True if the path is an agent/workflow execution endpoint."""
    if not any(path.startswith(p) for p in _AGENT_EXEC_PREFIXES):
        return False
    return any(path.endswith(s) for s in _AGENT_EXEC_SUFFIXES)


class _SlidingWindow:
    """Simple sliding-window counter for rate limiting."""

    __slots__ = ("_hits",)

    def __init__(self) -> None:
        self._hits: dict[str, list[float]] = {}

    def hit(self, key: str, now: float, limit: int) -> tuple[bool, int, int]:
        """Record a hit and return (allowed, remaining, reset_seconds)."""
        window_start = now - _WINDOW
        timestamps = self._hits.get(key, [])
        # Trim old entries.
        timestamps = [t for t in timestamps if t > window_start]
        remaining = max(0, limit - len(timestamps) - 1)
        reset = int(window_start + _WINDOW - now) + 1
        if len(timestamps) >= limit:
            self._hits[key] = timestamps
            return False, 0, reset
        timestamps.append(now)
        self._hits[key] = timestamps
        return True, remaining, reset

    def cleanup(self, now: float) -> None:
        """Remove stale keys (call periodically to avoid memory growth)."""
        cutoff = now - _WINDOW
        stale = [k for k, v in self._hits.items() if not v or v[-1] < cutoff]
        for k in stale:
            del self._hits[k]


# Global counters — one for auth (keyed by IP), one for user-scoped limits.
_auth_window = _SlidingWindow()
_user_window = _SlidingWindow()

# Counter for cleanup scheduling.
_last_cleanup: float = 0.0

# Lock protecting all read-modify-write operations on the sliding windows and
# _last_cleanup.  Acquired only around the critical section — never held across
# an await of the upstream handler.
_window_lock: asyncio.Lock | None = None


def _get_window_lock() -> asyncio.Lock:
    """Return the per-event-loop window lock, creating it lazily if needed."""
    global _window_lock
    if _window_lock is None:
        _window_lock = asyncio.Lock()
    return _window_lock


# Whether to trust X-Forwarded-For headers (set EXO_TRUST_PROXY=1 when the app
# is deployed behind a trusted reverse-proxy such as nginx or a cloud LB).
# Defaulting to False prevents IP spoofing when running without a proxy.
_TRUST_PROXY: bool = os.getenv("EXO_TRUST_PROXY", "").lower() in ("1", "true", "yes")


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Apply per-IP and per-user rate limits.

    - Auth endpoints (POST /api/auth/login): rate_limit_auth per minute per IP
    - Agent execution endpoints: rate_limit_agent per minute per user
    - General API endpoints: rate_limit_general per minute per user
    """

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        global _last_cleanup
        path = request.url.path
        method = request.method

        # Only rate-limit API requests.
        if not path.startswith("/api/"):
            return await call_next(request)

        # Skip health check.
        if path == "/api/health":
            return await call_next(request)

        now = time.monotonic()
        lock = _get_window_lock()

        # --- Auth endpoint rate limit (per IP) ---
        if path in _AUTH_PATHS and method == "POST":
            client_ip = _get_client_ip(request)
            key = f"auth:{client_ip}"
            limit = settings.rate_limit_auth
            async with lock:
                # Periodic cleanup (~every 5 minutes).
                if now - _last_cleanup > 300:
                    _last_cleanup = now
                    _auth_window.cleanup(now)
                    _user_window.cleanup(now)
                allowed, remaining, reset = _auth_window.hit(key, now, limit)
            if not allowed:
                return _rate_limited_response(limit, reset)
            response = await call_next(request)
            _add_rate_headers(response, limit, remaining, reset)
            return response

        # --- Agent execution rate limit (per user) ---
        if method == "POST" and _is_agent_exec(path):
            user_key = _get_user_key(request)
            key = f"agent:{user_key}"
            limit = settings.rate_limit_agent
            async with lock:
                if now - _last_cleanup > 300:
                    _last_cleanup = now
                    _auth_window.cleanup(now)
                    _user_window.cleanup(now)
                allowed, remaining, reset = _user_window.hit(key, now, limit)
            if not allowed:
                return _rate_limited_response(limit, reset)
            response = await call_next(request)
            _add_rate_headers(response, limit, remaining, reset)
            return response

        # --- General API rate limit (per user or IP) ---
        user_key = _get_user_key(request)
        key = f"general:{user_key}"
        limit = settings.rate_limit_general
        async with lock:
            if now - _last_cleanup > 300:
                _last_cleanup = now
                _auth_window.cleanup(now)
                _user_window.cleanup(now)
            allowed, remaining, reset = _user_window.hit(key, now, limit)
        if not allowed:
            return _rate_limited_response(limit, reset)
        response = await call_next(request)
        _add_rate_headers(response, limit, remaining, reset)
        return response


def _get_client_ip(request: Request) -> str:
    """Extract the real client IP address.

    When EXO_TRUST_PROXY=1 the first entry of X-Forwarded-For is used (assumes
    a trusted reverse-proxy prepends the original client IP).  Without that
    flag X-Forwarded-For is ignored entirely to prevent IP spoofing — the
    direct TCP connection address is used instead.
    """
    if _TRUST_PROXY:
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            return forwarded.split(",")[0].strip()
    client = request.client
    return client.host if client else "unknown"


def _get_user_key(request: Request) -> str:
    """Return session ID if available, otherwise fall back to IP."""
    session_id = request.cookies.get("exo_session")
    if session_id:
        return f"session:{session_id}"
    return f"ip:{_get_client_ip(request)}"


def _rate_limited_response(limit: int, reset: int) -> JSONResponse:
    """Return a 429 response with appropriate headers."""
    response = JSONResponse(
        status_code=429,
        content={"detail": "Too many requests. Please try again later."},
    )
    response.headers["Retry-After"] = str(reset)
    response.headers["X-RateLimit-Limit"] = str(limit)
    response.headers["X-RateLimit-Remaining"] = "0"
    response.headers["X-RateLimit-Reset"] = str(reset)
    return response


def _add_rate_headers(response: Response, limit: int, remaining: int, reset: int) -> None:
    """Add rate limit headers to a successful response."""
    response.headers["X-RateLimit-Limit"] = str(limit)
    response.headers["X-RateLimit-Remaining"] = str(remaining)
    response.headers["X-RateLimit-Reset"] = str(reset)
