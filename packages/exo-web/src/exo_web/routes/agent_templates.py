"""Agent templates REST API — save, share, and instantiate agent configurations."""

from __future__ import annotations

import json
import logging
import uuid
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from exo_web.database import get_db
from exo_web.pagination import paginate
from exo_web.routes.auth import get_current_user
from exo_web.sanitize import sanitize_html

_log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/templates", tags=["agent_templates"])

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

# Keys to strip from agent config when saving as a template (credentials etc.)
_CREDENTIAL_KEYS = frozenset(
    {
        "api_key",
        "encrypted_api_key",
        "secret",
        "token",
        "password",
        "credentials",
    }
)


class TemplateCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255, description="Display name")
    description: str = Field("", description="Human-readable description")
    config_json: str = Field("{}", description="JSON configuration object")
    tools_required: str = Field("[]", description="Tools required")
    models_required: str = Field("[]", description="Models required")


class TemplateCreateFromAgent(BaseModel):
    name: str = Field(..., min_length=1, max_length=255, description="Display name")
    description: str = Field("", description="Human-readable description")
    agent_id: str = Field(..., min_length=1, description="Associated agent identifier")


class TemplateUpdate(BaseModel):
    name: str | None = Field(None, min_length=1, max_length=255, description="Display name")
    description: str | None = Field(None, description="Human-readable description")
    config_json: str | None = Field(None, description="JSON configuration object")
    tools_required: str | None = Field(None, description="Tools required")
    models_required: str | None = Field(None, description="Models required")


class TemplateResponse(BaseModel):
    id: str = Field(description="Unique identifier")
    name: str = Field(description="Display name")
    description: str = Field(description="Human-readable description")
    config_json: str = Field(description="JSON configuration object")
    tools_required: str = Field(description="Tools required")
    models_required: str = Field(description="Models required")
    version: int = Field(description="Version identifier")
    creator_id: str = Field(description="Creator id")
    created_at: str = Field(description="ISO 8601 creation timestamp")
    updated_at: str = Field(description="ISO 8601 last-update timestamp")


class VersionResponse(BaseModel):
    id: str = Field(description="Unique identifier")
    template_id: str = Field(description="Associated template identifier")
    version_number: int = Field(description="Version number")
    config_json: str = Field(description="JSON configuration object")
    tools_required: str = Field(description="Tools required")
    models_required: str = Field(description="Models required")
    created_at: str = Field(description="ISO 8601 creation timestamp")


class InstantiateRequest(BaseModel):
    project_id: str = Field(..., min_length=1, description="Associated project identifier")
    name: str | None = Field(None, max_length=255, description="Display name")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _row_to_dict(row: Any) -> dict[str, Any]:
    return dict(row)


def _strip_credentials(config: dict[str, Any]) -> dict[str, Any]:
    """Remove credential-bearing keys from a config dict (shallow)."""
    return {k: v for k, v in config.items() if k not in _CREDENTIAL_KEYS}


async def _verify_ownership(db: Any, template_id: str, user_id: str) -> dict[str, Any]:
    cursor = await db.execute(
        "SELECT * FROM agent_templates WHERE id = ? AND creator_id = ?",
        (template_id, user_id),
    )
    row = await cursor.fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Template not found")
    return _row_to_dict(row)


async def _create_version(
    db: Any,
    template_id: str,
    config_json: str,
    tools_required: str,
    models_required: str,
) -> int:
    """Create a new version entry. Returns the version number."""
    cursor = await db.execute(
        "SELECT COALESCE(MAX(version_number), 0) + 1 FROM agent_template_versions WHERE template_id = ?",
        (template_id,),
    )
    row = await cursor.fetchone()
    next_version: int = row[0]

    version_id = str(uuid.uuid4())
    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")
    await db.execute(
        """
        INSERT INTO agent_template_versions
            (id, template_id, version_number, config_json, tools_required, models_required, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (version_id, template_id, next_version, config_json, tools_required, models_required, now),
    )
    return next_version


def _extract_agent_config(agent: dict[str, Any]) -> dict[str, Any]:
    """Extract reusable config fields from an agent row."""
    config = {
        "instructions": agent.get("instructions", ""),
        "model_provider": agent.get("model_provider", ""),
        "model_name": agent.get("model_name", ""),
        "temperature": agent.get("temperature"),
        "max_tokens": agent.get("max_tokens"),
        "max_steps": agent.get("max_steps"),
        "output_type_json": agent.get("output_type_json", "{}"),
        "hooks_json": agent.get("hooks_json", "{}"),
    }
    return _strip_credentials(config)


# ---------------------------------------------------------------------------
# Template CRUD
# ---------------------------------------------------------------------------


@router.get("")
async def list_templates(
    search: str | None = Query(None),
    cursor: str | None = Query(None),
    limit: int = Query(20, ge=1, le=100),
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> dict[str, Any]:
    """List templates with optional search."""
    async with get_db() as db:
        conditions = ["creator_id = ?"]
        params: list[Any] = [user["id"]]

        if search:
            conditions.append("(name LIKE ? OR description LIKE ?)")
            like = f"%{search}%"
            params.extend([like, like])

        result = await paginate(
            db,
            table="agent_templates",
            conditions=conditions,
            params=params,
            cursor=cursor,
            limit=limit,
        )
        return result.model_dump()


@router.post("", response_model=TemplateResponse, status_code=201)
async def create_template(
    body: TemplateCreate,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> dict[str, Any]:
    """Create a new agent template directly from config JSON."""
    # Validate and strip credentials from config
    try:
        config = json.loads(body.config_json)
        config = _strip_credentials(config)
        clean_config = json.dumps(config)
    except (json.JSONDecodeError, TypeError):
        clean_config = body.config_json

    async with get_db() as db:
        template_id = str(uuid.uuid4())
        now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")

        await db.execute(
            """
            INSERT INTO agent_templates
                (id, name, description, config_json, tools_required, models_required, version, creator_id, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, 1, ?, ?, ?)
            """,
            (
                template_id,
                sanitize_html(body.name),
                sanitize_html(body.description),
                clean_config,
                body.tools_required,
                body.models_required,
                user["id"],
                now,
                now,
            ),
        )

        await _create_version(
            db, template_id, clean_config, body.tools_required, body.models_required
        )
        await db.commit()

        cursor = await db.execute("SELECT * FROM agent_templates WHERE id = ?", (template_id,))
        row = await cursor.fetchone()
        return _row_to_dict(row)


@router.post("/from-agent", response_model=TemplateResponse, status_code=201)
async def create_template_from_agent(
    body: TemplateCreateFromAgent,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> dict[str, Any]:
    """Save an existing agent's config as a new template (strips credentials)."""
    async with get_db() as db:
        cursor = await db.execute(
            "SELECT * FROM agents WHERE id = ? AND user_id = ?",
            (body.agent_id, user["id"]),
        )
        agent_row = await cursor.fetchone()
        if agent_row is None:
            raise HTTPException(status_code=404, detail="Agent not found")
        agent = _row_to_dict(agent_row)

        config = _extract_agent_config(agent)
        config_json = json.dumps(config)

        # Extract tools and models required
        tools_required = agent.get("tools_json", "[]")
        models_required = json.dumps(
            [f"{agent.get('model_provider', '')}:{agent.get('model_name', '')}"]
        )

        template_id = str(uuid.uuid4())
        now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")

        await db.execute(
            """
            INSERT INTO agent_templates
                (id, name, description, config_json, tools_required, models_required, version, creator_id, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, 1, ?, ?, ?)
            """,
            (
                template_id,
                sanitize_html(body.name),
                sanitize_html(body.description),
                config_json,
                tools_required,
                models_required,
                user["id"],
                now,
                now,
            ),
        )

        await _create_version(db, template_id, config_json, tools_required, models_required)
        await db.commit()

        cursor = await db.execute("SELECT * FROM agent_templates WHERE id = ?", (template_id,))
        row = await cursor.fetchone()
        return _row_to_dict(row)


@router.get("/export/{template_id}")
async def export_template(
    template_id: str,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> JSONResponse:
    """Export a template as a downloadable JSON file."""
    async with get_db() as db:
        tpl = await _verify_ownership(db, template_id, user["id"])

    export_data = {
        "name": tpl["name"],
        "description": tpl["description"],
        "config_json": tpl["config_json"],
        "tools_required": tpl["tools_required"],
        "models_required": tpl["models_required"],
        "version": tpl["version"],
        "exported_at": datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S"),
    }

    return JSONResponse(
        content=export_data,
        headers={
            "Content-Disposition": f'attachment; filename="template-{template_id}.json"',
        },
    )


@router.post("/import", response_model=TemplateResponse, status_code=201)
async def import_template(
    file: UploadFile,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> dict[str, Any]:
    """Import a template from a JSON file."""
    content = await file.read()
    try:
        data = json.loads(content)
    except (json.JSONDecodeError, TypeError) as exc:
        raise HTTPException(status_code=422, detail="Invalid JSON file") from exc

    name = data.get("name", "Imported Template")
    description = data.get("description", "")
    config_json = data.get("config_json", "{}")
    tools_required = data.get("tools_required", "[]")
    models_required = data.get("models_required", "[]")

    # Strip credentials from imported config.  A malformed config_json is a
    # hard error — we must not silently store credential-bearing or unparseable
    # data, so we surface a 400 to the caller.
    try:
        config = json.loads(config_json) if isinstance(config_json, str) else config_json
        if not isinstance(config, dict):
            raise TypeError(f"config_json must be a JSON object, got {type(config).__name__}")
        config = _strip_credentials(config)
        config_json = json.dumps(config)
    except (json.JSONDecodeError, TypeError) as exc:
        _log.warning("import_template: malformed config_json — %s", exc)
        raise HTTPException(
            status_code=400,
            detail=f"Malformed config_json in imported template: {exc}",
        ) from exc

    async with get_db() as db:
        template_id = str(uuid.uuid4())
        now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")

        await db.execute(
            """
            INSERT INTO agent_templates
                (id, name, description, config_json, tools_required, models_required, version, creator_id, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, 1, ?, ?, ?)
            """,
            (
                template_id,
                sanitize_html(str(name)),
                sanitize_html(str(description)),
                config_json,
                tools_required if isinstance(tools_required, str) else json.dumps(tools_required),
                models_required
                if isinstance(models_required, str)
                else json.dumps(models_required),
                user["id"],
                now,
                now,
            ),
        )

        await _create_version(
            db,
            template_id,
            config_json,
            tools_required if isinstance(tools_required, str) else json.dumps(tools_required),
            models_required if isinstance(models_required, str) else json.dumps(models_required),
        )
        await db.commit()

        cursor = await db.execute("SELECT * FROM agent_templates WHERE id = ?", (template_id,))
        row = await cursor.fetchone()
        return _row_to_dict(row)


@router.get("/{template_id}", response_model=TemplateResponse)
async def get_template(
    template_id: str,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> dict[str, Any]:
    """Get a single template by ID."""
    async with get_db() as db:
        return await _verify_ownership(db, template_id, user["id"])


@router.put("/{template_id}", response_model=TemplateResponse)
async def update_template(
    template_id: str,
    body: TemplateUpdate,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> dict[str, Any]:
    """Update a template. Creates a new version if config changes."""
    updates = body.model_dump(exclude_none=True)
    if not updates:
        raise HTTPException(status_code=422, detail="No fields to update")

    for field in ("name", "description"):
        if field in updates and isinstance(updates[field], str):
            updates[field] = sanitize_html(updates[field])

    # Strip credentials from config if provided.  A malformed config_json is a
    # hard error — we must not silently store credential-bearing or unparseable
    # data, so we surface a 400 to the caller.
    if "config_json" in updates:
        try:
            config = json.loads(updates["config_json"])
            if not isinstance(config, dict):
                raise TypeError(f"config_json must be a JSON object, got {type(config).__name__}")
            config = _strip_credentials(config)
            updates["config_json"] = json.dumps(config)
        except (json.JSONDecodeError, TypeError) as exc:
            _log.warning("update_template %s: malformed config_json — %s", template_id, exc)
            raise HTTPException(
                status_code=400,
                detail=f"Malformed config_json: {exc}",
            ) from exc

    async with get_db() as db:
        existing = await _verify_ownership(db, template_id, user["id"])

        # Create new version if config/tools/models changed
        new_config = updates.get("config_json", existing["config_json"])
        new_tools = updates.get("tools_required", existing["tools_required"])
        new_models = updates.get("models_required", existing["models_required"])

        config_changed = (
            new_config != existing["config_json"]
            or new_tools != existing["tools_required"]
            or new_models != existing["models_required"]
        )

        if config_changed:
            new_version = await _create_version(db, template_id, new_config, new_tools, new_models)
            updates["version"] = new_version

        updates["updated_at"] = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")
        set_clause = ", ".join(f"{k} = ?" for k in updates)
        values = [*list(updates.values()), template_id]

        await db.execute(
            f"UPDATE agent_templates SET {set_clause} WHERE id = ?",
            values,
        )
        await db.commit()

        cursor = await db.execute("SELECT * FROM agent_templates WHERE id = ?", (template_id,))
        row = await cursor.fetchone()
        return _row_to_dict(row)


@router.delete("/{template_id}", status_code=204)
async def delete_template(
    template_id: str,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> None:
    """Delete a template and all its versions."""
    async with get_db() as db:
        await _verify_ownership(db, template_id, user["id"])
        await db.execute(
            "DELETE FROM agent_template_versions WHERE template_id = ?", (template_id,)
        )
        await db.execute("DELETE FROM agent_templates WHERE id = ?", (template_id,))
        await db.commit()


# ---------------------------------------------------------------------------
# Instantiate — create agent from template
# ---------------------------------------------------------------------------


@router.post("/{template_id}/instantiate", status_code=201)
async def instantiate_template(
    template_id: str,
    body: InstantiateRequest,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> dict[str, Any]:
    """Create a new agent from a template's config."""
    async with get_db() as db:
        tpl = await _verify_ownership(db, template_id, user["id"])

        # Verify project exists and belongs to user
        cursor = await db.execute(
            "SELECT id FROM projects WHERE id = ? AND user_id = ?",
            (body.project_id, user["id"]),
        )
        if await cursor.fetchone() is None:
            raise HTTPException(status_code=404, detail="Project not found")

        try:
            config = json.loads(tpl["config_json"])
        except (json.JSONDecodeError, TypeError):
            config = {}

        agent_id = str(uuid.uuid4())
        now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")
        agent_name = body.name or f"{tpl['name']}"

        await db.execute(
            """
            INSERT INTO agents
                (id, name, description, instructions, model_provider, model_name,
                 temperature, max_tokens, max_steps, output_type_json, tools_json,
                 handoffs_json, hooks_json, project_id, user_id, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                agent_id,
                sanitize_html(agent_name),
                sanitize_html(tpl["description"]),
                config.get("instructions", ""),
                config.get("model_provider", ""),
                config.get("model_name", ""),
                config.get("temperature"),
                config.get("max_tokens"),
                config.get("max_steps"),
                config.get("output_type_json", "{}"),
                tpl["tools_required"],
                "[]",
                config.get("hooks_json", "{}"),
                body.project_id,
                user["id"],
                now,
                now,
            ),
        )
        await db.commit()

        cursor = await db.execute("SELECT * FROM agents WHERE id = ?", (agent_id,))
        row = await cursor.fetchone()
        return _row_to_dict(row)


# ---------------------------------------------------------------------------
# Version history
# ---------------------------------------------------------------------------


@router.get("/{template_id}/versions", response_model=list[VersionResponse])
async def list_versions(
    template_id: str,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> list[dict[str, Any]]:
    """List all versions of a template, newest first."""
    async with get_db() as db:
        await _verify_ownership(db, template_id, user["id"])
        cursor = await db.execute(
            "SELECT * FROM agent_template_versions WHERE template_id = ? ORDER BY version_number DESC",
            (template_id,),
        )
        rows = await cursor.fetchall()
        return [_row_to_dict(r) for r in rows]


@router.get("/{template_id}/versions/{version_id}", response_model=VersionResponse)
async def get_version(
    template_id: str,
    version_id: str,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> dict[str, Any]:
    """Get a specific version."""
    async with get_db() as db:
        await _verify_ownership(db, template_id, user["id"])
        cursor = await db.execute(
            "SELECT * FROM agent_template_versions WHERE id = ? AND template_id = ?",
            (version_id, template_id),
        )
        row = await cursor.fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="Version not found")
        return _row_to_dict(row)
