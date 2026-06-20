"""Deployment endpoints — deploy agents and workflows as API endpoints."""

from __future__ import annotations

import hashlib
import json
import secrets
import uuid
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from exo_web.database import get_db
from exo_web.pagination import paginate
from exo_web.routes.auth import get_current_user
from exo_web.sanitize import sanitize_html
from exo_web.services.audit import audit_log

router = APIRouter(prefix="/api/v1/deployments", tags=["deployments"])


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class DeploymentCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255, description="Display name")
    entity_type: str = Field(..., pattern="^(agent|workflow)$", description="Entity type")
    entity_id: str = Field(..., min_length=1, description="Entity id")
    rate_limit: int = Field(60, ge=1, le=10000, description="Rate limit")


class DeploymentResponse(BaseModel):
    id: str = Field(description="Unique identifier")
    name: str = Field(description="Display name")
    entity_type: str = Field(description="Entity type")
    entity_id: str = Field(description="Entity id")
    rate_limit: int = Field(description="Rate limit")
    status: str = Field(description="Current status")
    usage_count: int = Field(description="Number of times used")
    user_id: str = Field(description="Owning user identifier")
    created_at: str = Field(description="ISO 8601 creation timestamp")
    updated_at: str = Field(description="ISO 8601 last-update timestamp")


class DeploymentCreateResponse(DeploymentResponse):
    """Returned only on creation — includes the plaintext API key (shown once)."""

    api_key: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _hash_api_key(key: str) -> str:
    """SHA-256 hash of an API key for storage."""
    return hashlib.sha256(key.encode()).hexdigest()


def _row_to_dict(row: Any) -> dict[str, Any]:
    return dict(row)


# ---------------------------------------------------------------------------
# CRUD endpoints
# ---------------------------------------------------------------------------


@router.post("", status_code=201)
async def create_deployment(
    body: DeploymentCreate,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> DeploymentCreateResponse:
    """Create a new deployment and return a one-time API key."""
    # Verify entity exists
    async with get_db() as db:
        if body.entity_type == "agent":
            cur = await db.execute(
                "SELECT id FROM agents WHERE id = ? AND user_id = ?",
                (body.entity_id, user["id"]),
            )
        else:
            cur = await db.execute(
                "SELECT id FROM workflows WHERE id = ? AND user_id = ?",
                (body.entity_id, user["id"]),
            )
        if await cur.fetchone() is None:
            raise HTTPException(
                status_code=404,
                detail=f"{body.entity_type.title()} not found",
            )

        deployment_id = str(uuid.uuid4())
        api_key = f"orb_{secrets.token_urlsafe(32)}"
        api_key_hash = _hash_api_key(api_key)
        now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")

        await db.execute(
            """
            INSERT INTO deployments
                (id, name, entity_type, entity_id, api_key_hash,
                 rate_limit, status, user_id, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, 'active', ?, ?, ?)
            """,
            (
                deployment_id,
                sanitize_html(body.name),
                body.entity_type,
                body.entity_id,
                api_key_hash,
                body.rate_limit,
                user["id"],
                now,
                now,
            ),
        )
        await db.commit()

        await audit_log(
            user["id"],
            "create_deployment",
            "deployment",
            deployment_id,
            details={
                "name": body.name,
                "entity_type": body.entity_type,
                "entity_id": body.entity_id,
            },
        )

        return DeploymentCreateResponse(
            id=deployment_id,
            name=body.name,
            entity_type=body.entity_type,
            entity_id=body.entity_id,
            rate_limit=body.rate_limit,
            status="active",
            usage_count=0,
            user_id=user["id"],
            created_at=now,
            updated_at=now,
            api_key=api_key,
        )


@router.get("")
async def list_deployments(
    cursor: str | None = Query(None),
    limit: int = Query(20, ge=1, le=100),
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> dict[str, Any]:
    """List all deployments with stats and entity names."""
    async with get_db() as db:
        result = await paginate(
            db,
            table="deployments",
            conditions=["user_id = ?"],
            params=[user["id"]],
            cursor=cursor,
            limit=limit,
            row_mapper=_row_to_dict,
        )
        data = result.model_dump()

        # Resolve entity names
        for dep in data.get("data", []):
            if dep["entity_type"] == "agent":
                cur = await db.execute("SELECT name FROM agents WHERE id = ?", (dep["entity_id"],))
            else:
                cur = await db.execute(
                    "SELECT name FROM workflows WHERE id = ?", (dep["entity_id"],)
                )
            row = await cur.fetchone()
            dep["entity_name"] = row["name"] if row else "(deleted)"

        return data


@router.get("/{deployment_id}")
async def get_deployment(
    deployment_id: str,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> dict[str, Any]:
    """Get a single deployment with entity name."""
    async with get_db() as db:
        cur = await db.execute(
            "SELECT * FROM deployments WHERE id = ? AND user_id = ?",
            (deployment_id, user["id"]),
        )
        row = await cur.fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="Deployment not found")

        dep = dict(row)
        # Resolve entity name
        if dep["entity_type"] == "agent":
            cur2 = await db.execute("SELECT name FROM agents WHERE id = ?", (dep["entity_id"],))
        else:
            cur2 = await db.execute("SELECT name FROM workflows WHERE id = ?", (dep["entity_id"],))
        entity_row = await cur2.fetchone()
        dep["entity_name"] = entity_row["name"] if entity_row else "(deleted)"

        return dep


class DeploymentUpdate(BaseModel):
    status: str | None = Field(None, pattern="^(active|inactive)$", description="Current status")
    rate_limit: int | None = Field(None, ge=1, le=10000, description="Rate limit")
    name: str | None = Field(None, min_length=1, max_length=255, description="Display name")


@router.patch("/{deployment_id}")
async def update_deployment(
    deployment_id: str,
    body: DeploymentUpdate,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> dict[str, Any]:
    """Update deployment status, rate limit, or name."""
    async with get_db() as db:
        cur = await db.execute(
            "SELECT * FROM deployments WHERE id = ? AND user_id = ?",
            (deployment_id, user["id"]),
        )
        row = await cur.fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="Deployment not found")

        updates: list[str] = []
        params: list[Any] = []
        if body.status is not None:
            updates.append("status = ?")
            params.append(body.status)
        if body.rate_limit is not None:
            updates.append("rate_limit = ?")
            params.append(body.rate_limit)
        if body.name is not None:
            updates.append("name = ?")
            params.append(sanitize_html(body.name))

        if not updates:
            return dict(row)

        updates.append("updated_at = ?")
        params.append(datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S"))
        params.append(deployment_id)

        await db.execute(
            f"UPDATE deployments SET {', '.join(updates)} WHERE id = ?",
            params,
        )
        await db.commit()

        cur2 = await db.execute("SELECT * FROM deployments WHERE id = ?", (deployment_id,))
        return dict(await cur2.fetchone())  # type: ignore[arg-type]


@router.delete("/{deployment_id}", status_code=204)
async def delete_deployment(
    deployment_id: str,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> None:
    """Delete a deployment."""
    async with get_db() as db:
        cur = await db.execute(
            "SELECT id, name FROM deployments WHERE id = ? AND user_id = ?",
            (deployment_id, user["id"]),
        )
        row = await cur.fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="Deployment not found")

        await db.execute("DELETE FROM deployments WHERE id = ?", (deployment_id,))
        await db.commit()

    await audit_log(
        user["id"],
        "delete_deployment",
        "deployment",
        deployment_id,
        details={"name": row["name"]},
    )


# ---------------------------------------------------------------------------
# Widget configuration endpoints
# ---------------------------------------------------------------------------


class WidgetConfig(BaseModel):
    primary_color: str = Field("#F76F53", max_length=50, description="Primary color")
    position: str = Field(
        "bottom-right", pattern="^(bottom-right|bottom-left)$", description="Position"
    )
    welcome_message: str = Field(
        "Hi! How can I help you?", max_length=500, description="Welcome message"
    )
    avatar_url: str = Field("", max_length=2000, description="Avatar url")


class WidgetConfigUpdate(WidgetConfig):
    cors_origins: str = Field("", max_length=5000)


class WidgetConfigResponse(WidgetConfig):
    cors_origins: str = ""


class EmbedCodeResponse(BaseModel):
    embed_code: str = Field(description="Embed code")


@router.get("/{deployment_id}/widget")
async def get_widget_config(
    deployment_id: str,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> WidgetConfigResponse:
    """Get widget configuration for a deployment."""
    async with get_db() as db:
        cur = await db.execute(
            "SELECT widget_config_json, cors_origins FROM deployments WHERE id = ? AND user_id = ?",
            (deployment_id, user["id"]),
        )
        row = await cur.fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="Deployment not found")

        config = json.loads(row["widget_config_json"] or "{}")
        return WidgetConfigResponse(
            primary_color=config.get("primary_color", "#F76F53"),
            position=config.get("position", "bottom-right"),
            welcome_message=config.get("welcome_message", "Hi! How can I help you?"),
            avatar_url=config.get("avatar_url", ""),
            cors_origins=row["cors_origins"] or "",
        )


@router.put("/{deployment_id}/widget")
async def update_widget_config(
    deployment_id: str,
    body: WidgetConfigUpdate,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> WidgetConfigResponse:
    """Update widget configuration for a deployment."""
    async with get_db() as db:
        cur = await db.execute(
            "SELECT id FROM deployments WHERE id = ? AND user_id = ?",
            (deployment_id, user["id"]),
        )
        if await cur.fetchone() is None:
            raise HTTPException(status_code=404, detail="Deployment not found")

        config_json = json.dumps(
            {
                "primary_color": body.primary_color,
                "position": body.position,
                "welcome_message": sanitize_html(body.welcome_message),
                "avatar_url": sanitize_html(body.avatar_url),
            }
        )
        now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")

        await db.execute(
            "UPDATE deployments SET widget_config_json = ?, cors_origins = ?, updated_at = ? WHERE id = ?",
            (config_json, body.cors_origins.strip(), now, deployment_id),
        )
        await db.commit()

        return WidgetConfigResponse(
            primary_color=body.primary_color,
            position=body.position,
            welcome_message=body.welcome_message,
            avatar_url=body.avatar_url,
            cors_origins=body.cors_origins.strip(),
        )


@router.get("/{deployment_id}/embed-code")
async def get_embed_code(
    deployment_id: str,
    request: Request,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> EmbedCodeResponse:
    """Generate the embed code snippet for a deployment widget."""
    async with get_db() as db:
        cur = await db.execute(
            "SELECT id, widget_config_json FROM deployments WHERE id = ? AND user_id = ?",
            (deployment_id, user["id"]),
        )
        row = await cur.fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="Deployment not found")

    base_url = str(request.base_url).rstrip("/")
    config = json.loads(row["widget_config_json"] or "{}")
    primary_color = config.get("primary_color", "#F76F53")
    position = config.get("position", "bottom-right")
    welcome_message = config.get("welcome_message", "Hi! How can I help you?")
    avatar_url = config.get("avatar_url", "")

    embed_code = (
        f'<script src="{base_url}/widget.js" '
        f'data-deployment-id="{deployment_id}" '
        f'data-base-url="{base_url}" '
        f'data-primary-color="{_escape_attr(primary_color)}" '
        f'data-position="{_escape_attr(position)}" '
        f'data-welcome-message="{_escape_attr(welcome_message)}" '
        + (f'data-avatar-url="{_escape_attr(avatar_url)}" ' if avatar_url else "")
        + "defer></script>"
    )

    return EmbedCodeResponse(embed_code=embed_code)


def _escape_attr(value: str) -> str:
    """Escape a value for use in an HTML attribute."""
    return (
        value.replace("&", "&amp;").replace('"', "&quot;").replace("<", "&lt;").replace(">", "&gt;")
    )


# ---------------------------------------------------------------------------
# Runtime endpoint — POST /api/deployed/:id/run
# ---------------------------------------------------------------------------

deployed_router = APIRouter(prefix="/api/v1/deployed", tags=["deployed"])


async def _check_cors(deployment_id: str, origin: str | None) -> str | None:
    """Return the origin if it's allowed by the deployment's cors_origins, else None."""
    if not origin:
        return None
    async with get_db() as db:
        cur = await db.execute(
            "SELECT cors_origins FROM deployments WHERE id = ?",
            (deployment_id,),
        )
        row = await cur.fetchone()
        if row is None:
            return None
        allowed_raw = row["cors_origins"] or ""
        allowed = [o.strip() for o in allowed_raw.split(",") if o.strip()]
        if "*" in allowed or origin in allowed:
            return origin
    return None


@deployed_router.options("/{deployment_id}/run")
async def preflight_run(deployment_id: str, request: Request) -> JSONResponse:
    """Handle CORS preflight for widget embed requests."""
    origin = request.headers.get("origin")
    allowed_origin = await _check_cors(deployment_id, origin)
    headers: dict[str, str] = {}
    if allowed_origin:
        headers["Access-Control-Allow-Origin"] = allowed_origin
        headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
        headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
        headers["Access-Control-Max-Age"] = "3600"
    return JSONResponse(content=None, status_code=204, headers=headers)


async def _authenticate_deployment(deployment_id: str, request: Request) -> dict[str, Any]:
    """Validate API key from Authorization header and return deployment row."""
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid API key")
    api_key = auth_header[7:]
    key_hash = _hash_api_key(api_key)

    async with get_db() as db:
        cur = await db.execute(
            "SELECT * FROM deployments WHERE id = ? AND api_key_hash = ?",
            (deployment_id, key_hash),
        )
        row = await cur.fetchone()
        if row is None:
            raise HTTPException(status_code=401, detail="Invalid API key")

        deployment = dict(row)
        if deployment["status"] != "active":
            raise HTTPException(status_code=403, detail="Deployment is inactive")

        # Increment usage count
        await db.execute(
            "UPDATE deployments SET usage_count = usage_count + 1, updated_at = ? WHERE id = ?",
            (datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S"), deployment_id),
        )
        await db.commit()

    return deployment


class DeployedRunRequest(BaseModel):
    input: str = Field(..., min_length=1, description="Input text or data")
    stream: bool = Field(False, description="Stream")


@deployed_router.post("/{deployment_id}/run")
async def run_deployed(
    deployment_id: str,
    body: DeployedRunRequest,
    request: Request,
) -> Any:
    """Execute a deployed agent or workflow.

    Authenticates via Bearer token in the Authorization header.
    Supports optional SSE streaming when ``stream: true``.
    """
    deployment = await _authenticate_deployment(deployment_id, request)

    origin = request.headers.get("origin")
    allowed_origin = await _check_cors(deployment_id, origin)
    cors_headers: dict[str, str] = {}
    if allowed_origin:
        cors_headers["Access-Control-Allow-Origin"] = allowed_origin

    if deployment["entity_type"] == "agent":
        result = await _run_agent_deployment(deployment, body)
    else:
        result = await _run_workflow_deployment(deployment, body)

    # Attach CORS headers for cross-origin widget requests.
    if cors_headers and isinstance(result, StreamingResponse):
        result.headers.update(cors_headers)
    elif cors_headers and isinstance(result, dict):
        return JSONResponse(content=result, headers=cors_headers)

    return result


async def _run_agent_deployment(deployment: dict[str, Any], body: DeployedRunRequest) -> Any:
    """Run an agent deployment, with optional SSE streaming."""
    from exo.types import UserMessage
    from exo_web.services.agent_runtime import AgentRuntimeError, AgentService

    service = AgentService()
    messages = [UserMessage(content=body.input)]

    if body.stream:

        async def _sse_generator() -> Any:
            try:
                async for chunk in service.stream_agent(deployment["entity_id"], messages):
                    data = json.dumps({"type": "token", "content": chunk.delta})
                    yield f"data: {data}\n\n"
                yield f"data: {json.dumps({'type': 'done'})}\n\n"
            except AgentRuntimeError as exc:
                yield f"data: {json.dumps({'type': 'error', 'error': str(exc)})}\n\n"

        return StreamingResponse(_sse_generator(), media_type="text/event-stream")

    # Non-streaming
    try:
        result = await service.run_agent(deployment["entity_id"], messages)
        return {"output": result.content, "usage": result.usage}
    except AgentRuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


async def _run_workflow_deployment(deployment: dict[str, Any], body: DeployedRunRequest) -> Any:
    """Run a workflow deployment, with optional SSE streaming."""
    import asyncio

    from exo_web.engine import execute_workflow

    workflow_id = deployment["entity_id"]

    # Load workflow nodes/edges
    async with get_db() as db:
        cur = await db.execute("SELECT * FROM workflows WHERE id = ?", (workflow_id,))
        row = await cur.fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="Workflow not found")
        workflow = dict(row)

    nodes = json.loads(workflow.get("nodes_json", "[]"))
    edges = json.loads(workflow.get("edges_json", "[]"))
    run_id = str(uuid.uuid4())

    if body.stream:

        async def _sse_generator() -> Any:
            queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()

            async def _event_callback(event: dict[str, Any]) -> None:
                await queue.put(event)

            async def _run() -> str:
                try:
                    return await execute_workflow(
                        run_id,
                        workflow_id,
                        deployment["user_id"],
                        nodes,
                        edges,
                        event_callback=_event_callback,
                    )
                finally:
                    await queue.put(None)

            task = asyncio.create_task(_run())
            try:
                while True:
                    event = await queue.get()
                    if event is None:
                        break
                    yield f"data: {json.dumps(event)}\n\n"
                result = await task
                yield f"data: {json.dumps({'type': 'done', 'result': result})}\n\n"
            except Exception as exc:
                yield f"data: {json.dumps({'type': 'error', 'error': str(exc)})}\n\n"

        return StreamingResponse(_sse_generator(), media_type="text/event-stream")

    # Non-streaming
    try:
        result = await execute_workflow(
            run_id,
            workflow_id,
            deployment["user_id"],
            nodes,
            edges,
        )
        return {"output": result}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
