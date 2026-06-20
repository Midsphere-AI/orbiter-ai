"""Workspace export and import service.

Generates a ZIP archive containing JSON files for each exportable entity
type (agents, workflows, tools, prompt_templates, knowledge_bases, providers).
Imports accept the same ZIP format and create new entities without overwriting.
"""

from __future__ import annotations

import io
import json
import logging
import uuid
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from exo_web.config import settings
from exo_web.database import get_db

logger = logging.getLogger("exo_web")

# Entity types included in exports and their SQL queries.
# Provider exports explicitly exclude encrypted_api_key.
_EXPORT_QUERIES: dict[str, str] = {
    "agents": (
        "SELECT id, name, description, instructions, model_provider, model_name, "
        "temperature, max_tokens, max_steps, output_type_json, tools_json, "
        "handoffs_json, hooks_json, project_id, persona_role, persona_goal, "
        "persona_backstory, knowledge_base_ids, context_automation_level, "
        "context_max_tokens_per_step, context_max_total_tokens, "
        "context_memory_type, context_workspace_enabled, created_at, updated_at "
        "FROM agents WHERE user_id = ?"
    ),
    "workflows": (
        "SELECT id, name, description, project_id, nodes_json, edges_json, "
        "viewport_json, status, version, created_at, updated_at "
        "FROM workflows WHERE user_id = ?"
    ),
    "tools": (
        "SELECT id, name, description, category, schema_json, code, tool_type, "
        "project_id, created_at "
        "FROM tools WHERE user_id = ?"
    ),
    "prompt_templates": (
        "SELECT id, name, content, variables_json, created_at, updated_at "
        "FROM prompt_templates WHERE user_id = ?"
    ),
    "knowledge_bases": (
        "SELECT id, name, description, embedding_model, chunk_size, chunk_overlap, "
        "project_id, search_type, top_k, similarity_threshold, reranker_enabled, "
        "created_at, updated_at "
        "FROM knowledge_bases WHERE user_id = ?"
    ),
    "providers": (
        "SELECT id, name, provider_type, base_url, max_retries, timeout, "
        "created_at, updated_at "
        "FROM providers WHERE user_id = ?"
    ),
}

# Allowlist of table names permitted in _resolve_name to prevent SQL injection.
_RESOLVE_NAME_ALLOWED_TABLES: frozenset[str] = frozenset(
    {
        "agents",
        "workflows",
        "tools",
        "prompt_templates",
        "knowledge_bases",
        "providers",
    }
)

_EXPORT_DIR = Path(settings.upload_dir) / "exports"


def _get_export_dir() -> Path:
    """Return the export directory, creating it if needed."""
    _EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    return _EXPORT_DIR


async def create_export(user_id: str) -> dict[str, Any]:
    """Create a workspace export ZIP file.

    Returns metadata dict with export id, filename, and entity counts.
    """
    export_id = uuid.uuid4().hex[:16]
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    filename = f"exo-export-{timestamp}-{export_id}.zip"

    buf = io.BytesIO()
    counts: dict[str, int] = {}

    async with get_db() as db:
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for entity_type, query in _EXPORT_QUERIES.items():
                cursor = await db.execute(query, (user_id,))
                rows = [dict(row) for row in await cursor.fetchall()]
                counts[entity_type] = len(rows)
                zf.writestr(f"{entity_type}.json", json.dumps(rows, indent=2))

            # Write manifest with metadata
            manifest = {
                "export_id": export_id,
                "exported_at": datetime.now(UTC).isoformat(),
                "user_id": user_id,
                "entity_counts": counts,
                "version": "1.0",
            }
            zf.writestr("manifest.json", json.dumps(manifest, indent=2))

    # Save to disk
    export_dir = _get_export_dir()
    export_path = export_dir / filename
    export_path.write_bytes(buf.getvalue())

    logger.info("Created workspace export %s (%d bytes)", export_id, len(buf.getvalue()))

    return {
        "export_id": export_id,
        "filename": filename,
        "path": str(export_path),
        "entity_counts": counts,
        "exported_at": manifest["exported_at"],
    }


def get_export_path(export_id: str) -> Path | None:
    """Find an export ZIP by its id. Returns the file path or None."""
    export_dir = _get_export_dir()
    for path in export_dir.iterdir():
        if export_id in path.name and path.suffix == ".zip":
            return path
    return None


async def import_workspace(user_id: str, zip_bytes: bytes) -> dict[str, Any]:
    """Import entities from a workspace export ZIP.

    Creates new entities for each item in the ZIP. Existing entities with the
    same name get a " (Import)" suffix to avoid overwriting.

    Returns a summary of imported entity counts.
    """
    buf = io.BytesIO(zip_bytes)

    try:
        zf = zipfile.ZipFile(buf, "r")
    except zipfile.BadZipFile as exc:
        raise ValueError("Invalid ZIP file") from exc

    counts: dict[str, int] = {}

    with zf:
        async with get_db() as db:
            if "agents.json" in zf.namelist():
                data = json.loads(zf.read("agents.json"))
                counts["agents"] = await _import_agents(db, user_id, data)

            if "workflows.json" in zf.namelist():
                data = json.loads(zf.read("workflows.json"))
                counts["workflows"] = await _import_workflows(db, user_id, data)

            if "tools.json" in zf.namelist():
                data = json.loads(zf.read("tools.json"))
                counts["tools"] = await _import_tools(db, user_id, data)

            if "prompt_templates.json" in zf.namelist():
                data = json.loads(zf.read("prompt_templates.json"))
                counts["prompt_templates"] = await _import_prompt_templates(db, user_id, data)

            if "knowledge_bases.json" in zf.namelist():
                data = json.loads(zf.read("knowledge_bases.json"))
                counts["knowledge_bases"] = await _import_knowledge_bases(db, user_id, data)

            if "providers.json" in zf.namelist():
                data = json.loads(zf.read("providers.json"))
                counts["providers"] = await _import_providers(db, user_id, data)

            await db.commit()

    logger.info("Imported workspace data for user %s: %s", user_id, counts)
    return {"imported": counts}


async def _resolve_name(
    db: Any,
    table: str,
    name: str,
    user_id: str,
) -> str:
    """Return a unique name, appending ' (Import)' if the name already exists."""
    if table not in _RESOLVE_NAME_ALLOWED_TABLES:
        raise ValueError(f"Table '{table}' is not permitted in _resolve_name")
    cursor = await db.execute(
        f"SELECT COUNT(*) FROM {table} WHERE name = ? AND user_id = ?",
        (name, user_id),
    )
    row = await cursor.fetchone()
    if row[0] > 0:
        return f"{name} (Import)"
    return name


async def _import_agents(db: Any, user_id: str, data: list[dict[str, Any]]) -> int:
    """Import agent entities."""
    count = 0
    for item in data:
        new_id = uuid.uuid4().hex
        name = await _resolve_name(db, "agents", item.get("name", ""), user_id)
        await db.execute(
            "INSERT INTO agents (id, name, description, instructions, model_provider, "
            "model_name, temperature, max_tokens, max_steps, output_type_json, "
            "tools_json, handoffs_json, hooks_json, project_id, user_id, "
            "persona_role, persona_goal, persona_backstory, knowledge_base_ids, "
            "context_automation_level, context_max_tokens_per_step, "
            "context_max_total_tokens, context_memory_type, context_workspace_enabled) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                new_id,
                name,
                item.get("description", ""),
                item.get("instructions", ""),
                item.get("model_provider", ""),
                item.get("model_name", ""),
                item.get("temperature"),
                item.get("max_tokens"),
                item.get("max_steps"),
                item.get("output_type_json", "{}"),
                item.get("tools_json", "[]"),
                item.get("handoffs_json", "[]"),
                item.get("hooks_json", "{}"),
                item.get("project_id", ""),
                user_id,
                item.get("persona_role", ""),
                item.get("persona_goal", ""),
                item.get("persona_backstory", ""),
                item.get("knowledge_base_ids", "[]"),
                item.get("context_automation_level", "copilot"),
                item.get("context_max_tokens_per_step"),
                item.get("context_max_total_tokens"),
                item.get("context_memory_type", "conversation"),
                item.get("context_workspace_enabled", 0),
            ),
        )
        count += 1
    return count


async def _import_workflows(db: Any, user_id: str, data: list[dict[str, Any]]) -> int:
    """Import workflow entities."""
    count = 0
    for item in data:
        new_id = uuid.uuid4().hex
        name = await _resolve_name(db, "workflows", item.get("name", ""), user_id)
        await db.execute(
            "INSERT INTO workflows (id, name, description, project_id, nodes_json, "
            "edges_json, viewport_json, status, version, user_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                new_id,
                name,
                item.get("description", ""),
                item.get("project_id", ""),
                item.get("nodes_json", "[]"),
                item.get("edges_json", "[]"),
                item.get("viewport_json", '{"x":0,"y":0,"zoom":1}'),
                "draft",
                1,
                user_id,
            ),
        )
        count += 1
    return count


async def _import_tools(db: Any, user_id: str, data: list[dict[str, Any]]) -> int:
    """Import tool entities."""
    count = 0
    for item in data:
        new_id = uuid.uuid4().hex
        name = await _resolve_name(db, "tools", item.get("name", ""), user_id)
        await db.execute(
            "INSERT INTO tools (id, name, description, category, schema_json, code, "
            "tool_type, project_id, user_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                new_id,
                name,
                item.get("description", ""),
                item.get("category", "custom"),
                item.get("schema_json", "{}"),
                item.get("code", ""),
                item.get("tool_type", "function"),
                item.get("project_id", ""),
                user_id,
            ),
        )
        count += 1
    return count


async def _import_prompt_templates(db: Any, user_id: str, data: list[dict[str, Any]]) -> int:
    """Import prompt template entities."""
    count = 0
    for item in data:
        new_id = uuid.uuid4().hex
        name = await _resolve_name(db, "prompt_templates", item.get("name", ""), user_id)
        await db.execute(
            "INSERT INTO prompt_templates (id, name, content, variables_json, user_id) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                new_id,
                name,
                item.get("content", ""),
                item.get("variables_json", "{}"),
                user_id,
            ),
        )
        count += 1
    return count


async def _import_knowledge_bases(db: Any, user_id: str, data: list[dict[str, Any]]) -> int:
    """Import knowledge base config entities (not embeddings)."""
    count = 0
    for item in data:
        new_id = uuid.uuid4().hex
        name = await _resolve_name(db, "knowledge_bases", item.get("name", ""), user_id)
        await db.execute(
            "INSERT INTO knowledge_bases (id, name, description, embedding_model, "
            "chunk_size, chunk_overlap, project_id, user_id, search_type, top_k, "
            "similarity_threshold, reranker_enabled) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                new_id,
                name,
                item.get("description", ""),
                item.get("embedding_model", "text-embedding-3-small"),
                item.get("chunk_size", 512),
                item.get("chunk_overlap", 50),
                item.get("project_id"),
                user_id,
                item.get("search_type", "keyword"),
                item.get("top_k", 5),
                item.get("similarity_threshold", 0.0),
                item.get("reranker_enabled", 0),
            ),
        )
        count += 1
    return count


async def _import_providers(db: Any, user_id: str, data: list[dict[str, Any]]) -> int:
    """Import provider config entities (no API keys)."""
    count = 0
    for item in data:
        new_id = uuid.uuid4().hex
        name = await _resolve_name(db, "providers", item.get("name", ""), user_id)
        await db.execute(
            "INSERT INTO providers (id, name, provider_type, base_url, max_retries, "
            "timeout, user_id) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                new_id,
                name,
                item.get("provider_type", "custom"),
                item.get("base_url"),
                item.get("max_retries", 3),
                item.get("timeout", 30),
                user_id,
            ),
        )
        count += 1
    return count
