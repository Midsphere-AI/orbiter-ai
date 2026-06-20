"""Shared helpers for SQLite and Postgres memory backends."""

from __future__ import annotations

from typing import Any

from exo.memory.base import (  # pyright: ignore[reportMissingImports]
    AIMemory,
    HumanMemory,
    MemoryItem,
    MemoryMetadata,
    MemoryStatus,
    SystemMemory,
    ToolMemory,
)


def extra_fields(item: MemoryItem) -> dict[str, Any]:
    """Extract subclass-specific fields into a JSON-serialisable dict.

    Covers AIMemory tool_calls, ToolMemory fields, and SnapshotMemory fields.
    Both SQLiteMemoryStore and PostgresMemoryStore use this to populate
    the ``extra_json`` column.
    """
    data: dict[str, Any] = {}
    if hasattr(item, "tool_calls"):
        data["tool_calls"] = item.tool_calls  # type: ignore[attr-defined]
    if hasattr(item, "tool_call_id"):
        data["tool_call_id"] = item.tool_call_id  # type: ignore[attr-defined]
    if hasattr(item, "tool_name"):
        data["tool_name"] = item.tool_name  # type: ignore[attr-defined]
    if hasattr(item, "is_error"):
        data["is_error"] = item.is_error  # type: ignore[attr-defined]
    # Snapshot-specific fields
    if hasattr(item, "snapshot_version"):
        data["snapshot_version"] = item.snapshot_version  # type: ignore[attr-defined]
    if hasattr(item, "raw_item_count"):
        data["raw_item_count"] = item.raw_item_count  # type: ignore[attr-defined]
    if hasattr(item, "latest_raw_id"):
        data["latest_raw_id"] = item.latest_raw_id  # type: ignore[attr-defined]
    if hasattr(item, "latest_raw_created_at"):
        data["latest_raw_created_at"] = item.latest_raw_created_at  # type: ignore[attr-defined]
    if hasattr(item, "config_hash"):
        data["config_hash"] = item.config_hash  # type: ignore[attr-defined]
    return data


def row_to_item(
    row_id: str,
    content: str,
    memory_type: str,
    status_str: str,
    meta_dict: dict[str, Any],
    extra: dict[str, Any],
    created_at: str,
    updated_at: str,
) -> MemoryItem:
    """Reconstruct a MemoryItem from individual column values.

    The ``meta_dict`` and ``extra`` parameters must already be Python dicts
    (callers are responsible for JSON-parsing from their respective column
    types — SQLite always uses ``json.loads``, Postgres uses ``json.loads``
    only when the column value is a plain ``str``).
    """
    kwargs: dict[str, Any] = {
        "id": row_id,
        "content": content,
        "memory_type": memory_type,
        "status": MemoryStatus(status_str),
        "metadata": MemoryMetadata(**meta_dict),
        "created_at": created_at,
        "updated_at": updated_at,
    }

    if memory_type == "system":
        return SystemMemory(**kwargs)
    if memory_type == "human":
        return HumanMemory(**kwargs)
    if memory_type == "ai":
        kwargs["tool_calls"] = extra.get("tool_calls", [])
        return AIMemory(**kwargs)
    if memory_type == "tool":
        kwargs["tool_call_id"] = extra.get("tool_call_id", "")
        kwargs["tool_name"] = extra.get("tool_name", "")
        kwargs["is_error"] = extra.get("is_error", False)
        return ToolMemory(**kwargs)
    if memory_type == "snapshot":
        from exo.memory.snapshot import SnapshotMemory  # pyright: ignore[reportMissingImports]

        kwargs["snapshot_version"] = extra.get("snapshot_version", 1)
        kwargs["raw_item_count"] = extra.get("raw_item_count", 0)
        kwargs["latest_raw_id"] = extra.get("latest_raw_id", "")
        kwargs["latest_raw_created_at"] = extra.get("latest_raw_created_at", "")
        kwargs["config_hash"] = extra.get("config_hash", "")
        return SnapshotMemory(**kwargs)
    return MemoryItem(**kwargs)


def build_metadata_filter_sqlite(
    metadata: MemoryMetadata,
    clauses: list[str],
    params: list[Any],
) -> None:
    """Append SQLite ``json_extract`` clauses for the four metadata fields.

    Mutates *clauses* and *params* in-place. Only non-None fields are added.
    """
    if metadata.user_id:
        clauses.append("json_extract(metadata, '$.user_id') = ?")
        params.append(metadata.user_id)
    if metadata.session_id:
        clauses.append("json_extract(metadata, '$.session_id') = ?")
        params.append(metadata.session_id)
    if metadata.task_id:
        clauses.append("json_extract(metadata, '$.task_id') = ?")
        params.append(metadata.task_id)
    if metadata.agent_id:
        clauses.append("json_extract(metadata, '$.agent_id') = ?")
        params.append(metadata.agent_id)


def build_metadata_filter_postgres(
    metadata: MemoryMetadata,
    clauses: list[str],
    params: list[Any],
    idx: int,
) -> int:
    """Append Postgres JSONB ``->>`` clauses for the four metadata fields.

    Mutates *clauses* and *params* in-place.  Returns the updated parameter
    index so the caller can continue numbering ``$N`` placeholders.
    """
    if metadata.user_id:
        clauses.append(f"metadata->>'user_id' = ${idx}")
        params.append(metadata.user_id)
        idx += 1
    if metadata.session_id:
        clauses.append(f"metadata->>'session_id' = ${idx}")
        params.append(metadata.session_id)
        idx += 1
    if metadata.task_id:
        clauses.append(f"metadata->>'task_id' = ${idx}")
        params.append(metadata.task_id)
        idx += 1
    if metadata.agent_id:
        clauses.append(f"metadata->>'agent_id' = ${idx}")
        params.append(metadata.agent_id)
        idx += 1
    return idx
