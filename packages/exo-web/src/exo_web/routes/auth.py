"""Authentication endpoints and middleware."""

from __future__ import annotations

import logging
import secrets
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

import bcrypt
from fastapi import APIRouter, Cookie, Depends, HTTPException, Request, Response
from pydantic import BaseModel, Field

from exo_web.config import settings
from exo_web.database import get_db
from exo_web.services.audit import audit_log

logger = logging.getLogger("exo_web")

router = APIRouter(prefix="/api/v1/auth", tags=["auth"])

SESSION_COOKIE = "exo_session"


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class LoginRequest(BaseModel):
    email: str = Field(..., min_length=1, description="User email address")
    password: str = Field(..., min_length=1, description="User password")


class UserResponse(BaseModel):
    id: str = Field(description="Unique identifier")
    email: str = Field(description="User email address")
    role: str = Field(description="Message role (system, user, assistant)")
    created_at: str = Field(description="ISO 8601 creation timestamp")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _hash_password(password: str) -> str:
    """Hash a password using bcrypt."""
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def _verify_password(password: str, hashed: str) -> bool:
    """Verify a password against a bcrypt hash."""
    return bcrypt.checkpw(password.encode(), hashed.encode())


# ---------------------------------------------------------------------------
# Protected route dependency
# ---------------------------------------------------------------------------


async def get_current_user(
    exo_session: str | None = Cookie(None),
) -> dict[str, Any]:
    """Extract the current user from the session cookie, or raise 401."""
    if not exo_session:
        raise HTTPException(status_code=401, detail="Not authenticated")

    async with get_db() as db:
        cursor = await db.execute(
            """
            SELECT u.id, u.email, u.role, u.created_at
            FROM sessions s
            JOIN users u ON u.id = s.user_id
            WHERE s.id = ? AND s.expires_at > datetime('now')
            """,
            (exo_session,),
        )
        row = await cursor.fetchone()

    if row is None:
        raise HTTPException(status_code=401, detail="Not authenticated")

    return dict(row)


# ---------------------------------------------------------------------------
# Role-based access control
# ---------------------------------------------------------------------------

_ROLE_HIERARCHY: dict[str, int] = {"viewer": 0, "developer": 1, "admin": 2}


def require_role(min_role: str):
    """Return a FastAPI dependency that enforces a minimum role level.

    Usage::

        @router.post("/admin-only")
        async def admin_endpoint(
            user: dict = Depends(require_role("admin")),
        ):
            ...
    """

    async def _check_role(
        user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
    ) -> dict[str, Any]:
        user_level = _ROLE_HIERARCHY.get(user.get("role", ""), -1)
        required_level = _ROLE_HIERARCHY.get(min_role, 999)
        if user_level < required_level:
            raise HTTPException(status_code=403, detail="FORBIDDEN")
        return user

    return _check_role


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/login", response_model=UserResponse)
async def login(body: LoginRequest, request: Request, response: Response) -> dict[str, Any]:
    """Authenticate with email + password and set a session cookie."""
    async with get_db() as db:
        cursor = await db.execute(
            "SELECT id, email, password_hash, role, created_at FROM users WHERE email = ?",
            (body.email,),
        )
        user = await cursor.fetchone()

    if user is None or not _verify_password(body.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    # Create session with CSRF token.
    session_id = str(uuid.uuid4())
    csrf_token = secrets.token_urlsafe(32)
    expires_at = (datetime.now(UTC) + timedelta(hours=settings.session_expiry_hours)).strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    async with get_db() as db:
        await db.execute(
            "INSERT INTO sessions (id, user_id, expires_at, csrf_token) VALUES (?, ?, ?, ?)",
            (session_id, user["id"], expires_at, csrf_token),
        )
        await db.commit()

    response.set_cookie(
        key=SESSION_COOKIE,
        value=session_id,
        httponly=True,
        samesite="lax",
        secure=not settings.debug,
        max_age=settings.session_expiry_hours * 3600,
        path="/",
    )

    ip = request.client.host if request.client else None
    await audit_log(user["id"], "login", "user", user["id"], ip_address=ip)

    return {
        "id": user["id"],
        "email": user["email"],
        "role": user["role"],
        "created_at": user["created_at"],
    }


@router.post("/logout", status_code=204)
async def logout(
    request: Request,
    response: Response,
    exo_session: str | None = Cookie(None),
) -> None:
    """Clear the session cookie and delete the session."""
    user_id: str | None = None
    if exo_session:
        async with get_db() as db:
            cursor = await db.execute("SELECT user_id FROM sessions WHERE id = ?", (exo_session,))
            row = await cursor.fetchone()
            if row:
                user_id = row["user_id"]
            await db.execute("DELETE FROM sessions WHERE id = ?", (exo_session,))
            await db.commit()

    if user_id:
        ip = request.client.host if request.client else None
        await audit_log(user_id, "logout", "user", user_id, ip_address=ip)

    response.delete_cookie(key=SESSION_COOKIE, path="/")


@router.get("/me", response_model=UserResponse)
async def me(
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> dict[str, Any]:
    """Return the current authenticated user."""
    return user


# ---------------------------------------------------------------------------
# Profile endpoints
# ---------------------------------------------------------------------------


class ProfileUpdateRequest(BaseModel):
    email: str = Field(..., min_length=1, description="User email address")


@router.get("/profile", response_model=UserResponse)
async def get_profile(
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> dict[str, Any]:
    """Return the full profile for the current authenticated user."""
    return user


@router.put("/profile", response_model=UserResponse)
async def update_profile(
    body: ProfileUpdateRequest,
    request: Request,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> dict[str, Any]:
    """Update the current user's profile (email)."""
    user_id = user["id"]
    new_email = body.email.strip()

    if not new_email:
        raise HTTPException(status_code=422, detail="Email must not be empty")

    # Check email uniqueness.
    async with get_db() as db:
        cursor = await db.execute(
            "SELECT id FROM users WHERE email = ? AND id != ?",
            (new_email, user_id),
        )
        existing = await cursor.fetchone()

    if existing:
        raise HTTPException(status_code=409, detail="Email already in use")

    # Update email.
    async with get_db() as db:
        await db.execute(
            "UPDATE users SET email = ? WHERE id = ?",
            (new_email, user_id),
        )
        await db.commit()

    ip = request.client.host if request.client else None
    await audit_log(
        user_id,
        "update_profile",
        "user",
        user_id,
        details={"old_email": user["email"], "new_email": new_email},
        ip_address=ip,
    )

    # Session is preserved — no session invalidation on email change.
    return {
        "id": user_id,
        "email": new_email,
        "role": user["role"],
        "created_at": user["created_at"],
    }


class PasswordChangeRequest(BaseModel):
    current_password: str = Field(..., min_length=1, description="Current password")
    new_password: str = Field(..., min_length=8, description="New password")


class MessageResponse(BaseModel):
    message: str = Field(description="Message")


@router.put("/password", response_model=MessageResponse)
async def change_password(
    body: PasswordChangeRequest,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
    exo_session: str | None = Cookie(None),
) -> dict[str, str]:
    """Change the current user's password."""
    user_id = user["id"]

    # Fetch current password hash.
    async with get_db() as db:
        cursor = await db.execute(
            "SELECT password_hash FROM users WHERE id = ?",
            (user_id,),
        )
        row = await cursor.fetchone()

    if row is None:
        raise HTTPException(status_code=404, detail="User not found")

    if not _verify_password(body.current_password, row["password_hash"]):
        raise HTTPException(status_code=401, detail="Current password is incorrect")

    # Hash and store new password, then invalidate other sessions.
    new_hash = _hash_password(body.new_password)
    async with get_db() as db:
        await db.execute(
            "UPDATE users SET password_hash = ? WHERE id = ?",
            (new_hash, user_id),
        )
        # Invalidate all sessions except the current one.
        await db.execute(
            "DELETE FROM sessions WHERE user_id = ? AND id != ?",
            (user_id, exo_session),
        )
        await db.commit()

    return {"message": "Password updated"}


class CsrfResponse(BaseModel):
    token: str = Field(description="Token")


@router.get("/csrf", response_model=CsrfResponse)
async def get_csrf_token(
    exo_session: str | None = Cookie(None),
) -> dict[str, str]:
    """Return the CSRF token for the current session."""
    if not exo_session:
        raise HTTPException(status_code=401, detail="Not authenticated")

    async with get_db() as db:
        cursor = await db.execute(
            "SELECT csrf_token FROM sessions WHERE id = ? AND expires_at > datetime('now')",
            (exo_session,),
        )
        row = await cursor.fetchone()

    if row is None or row["csrf_token"] is None:
        raise HTTPException(status_code=401, detail="Not authenticated")

    return {"token": row["csrf_token"]}


# ---------------------------------------------------------------------------
# Password reset flow
# ---------------------------------------------------------------------------


class ForgotPasswordRequest(BaseModel):
    email: str = Field(..., min_length=1, description="User email address")


class ResetPasswordRequest(BaseModel):
    token: str = Field(..., min_length=1, description="Token")
    new_password: str = Field(..., min_length=8, description="New password")


@router.post("/forgot-password", response_model=MessageResponse)
async def forgot_password(body: ForgotPasswordRequest) -> dict[str, str]:
    """Generate a password reset token and store its hash (email delivery is future work)."""
    async with get_db() as db:
        cursor = await db.execute(
            "SELECT id FROM users WHERE email = ?",
            (body.email,),
        )
        user = await cursor.fetchone()

    # Always return success to avoid leaking whether an email is registered.
    if user is None:
        return {"message": "If that email exists, a reset link has been sent"}

    # Generate reset token and store its bcrypt hash.
    token = str(uuid.uuid4())
    token_hash = bcrypt.hashpw(token.encode(), bcrypt.gensalt()).decode()
    reset_id = str(uuid.uuid4())
    expires_at = (datetime.now(UTC) + timedelta(hours=1)).strftime("%Y-%m-%d %H:%M:%S")

    async with get_db() as db:
        await db.execute(
            "INSERT INTO password_resets (id, user_id, token_hash, expires_at) VALUES (?, ?, ?, ?)",
            (reset_id, user["id"], token_hash, expires_at),
        )
        await db.commit()

    logger.debug("Password reset email sent to %s", body.email)

    return {"message": "If that email exists, a reset link has been sent"}


@router.post("/reset-password", response_model=MessageResponse)
async def reset_password(body: ResetPasswordRequest) -> dict[str, str]:
    """Reset a user's password using a valid reset token."""
    # Find all unused, non-expired reset tokens.
    async with get_db() as db:
        cursor = await db.execute(
            """
            SELECT id, user_id, token_hash
            FROM password_resets
            WHERE used = 0 AND expires_at > datetime('now')
            ORDER BY created_at DESC
            """,
        )
        rows = await cursor.fetchall()

    # Check the provided token against stored hashes.
    matched_reset: dict[str, Any] | None = None
    for row in rows:
        if bcrypt.checkpw(body.token.encode(), row["token_hash"].encode()):
            matched_reset = dict(row)
            break

    if matched_reset is None:
        raise HTTPException(status_code=400, detail="Invalid or expired reset token")

    # Update password, mark token as used, and invalidate all sessions.
    new_hash = _hash_password(body.new_password)
    user_id = matched_reset["user_id"]

    async with get_db() as db:
        await db.execute(
            "UPDATE users SET password_hash = ? WHERE id = ?",
            (new_hash, user_id),
        )
        await db.execute(
            "UPDATE password_resets SET used = 1 WHERE id = ?",
            (matched_reset["id"],),
        )
        await db.execute(
            "DELETE FROM sessions WHERE user_id = ?",
            (user_id,),
        )
        await db.commit()

    return {"message": "Password updated"}
