"""Vector store configuration API routes."""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from exo_web.crypto import decrypt_api_key, encrypt_api_key
from exo_web.database import get_db
from exo_web.routes.auth import require_role

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/vector-stores", tags=["vector-stores"])

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_BACKENDS = ("sqlite_vss", "milvus", "qdrant", "chromadb", "pinecone")

_BACKEND_LABELS = {
    "sqlite_vss": "SQLite-VSS",
    "milvus": "Milvus / Zilliz",
    "qdrant": "Qdrant",
    "chromadb": "ChromaDB",
    "pinecone": "Pinecone",
}

_BACKEND_DEFAULTS: dict[str, dict[str, object]] = {
    "sqlite_vss": {"host": "", "port": None, "collection_name": ""},
    "milvus": {"host": "localhost", "port": 19530, "collection_name": "exo"},
    "qdrant": {"host": "localhost", "port": 6333, "collection_name": "exo"},
    "chromadb": {"host": "localhost", "port": 8000, "collection_name": "exo"},
    "pinecone": {"host": "", "port": None, "collection_name": "exo"},
}

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class VectorStoreCreate(BaseModel):
    backend: str = Field(
        "sqlite_vss",
        pattern=r"^(sqlite_vss|milvus|qdrant|chromadb|pinecone)$",
        description="Backend",
    )
    host: str = Field("", description="Host")
    port: int | None = Field(None, description="Port")
    api_key: str = Field("", description="API key (stored encrypted)")
    collection_name: str = Field("", description="Collection name")


class VectorStoreUpdate(BaseModel):
    backend: str | None = Field(
        None, pattern=r"^(sqlite_vss|milvus|qdrant|chromadb|pinecone)$", description="Backend"
    )
    host: str | None = Field(None, description="Host")
    port: int | None = Field(None, ge=1, le=65535, description="Port")
    api_key: str | None = Field(None, description="API key (stored encrypted)")
    collection_name: str | None = Field(None, description="Collection name")


class VectorStoreResponse(BaseModel):
    id: str = Field(description="Unique identifier")
    backend: str = Field(description="Backend")
    backend_label: str = Field(description="Backend label")
    host: str = Field(description="Host")
    port: int | None = Field(description="Port")
    api_key_set: bool = Field(description="Whether an API key is configured")
    collection_name: str = Field(description="Collection name")
    is_active: bool = Field(description="Whether this item is active")
    user_id: str = Field(description="Owning user identifier")
    created_at: str = Field(description="ISO 8601 creation timestamp")
    updated_at: str = Field(description="ISO 8601 last-update timestamp")


class TestResult(BaseModel):
    success: bool = Field(description="Whether the operation succeeded")
    error: str | None = Field(None, description="Error message if failed")
    backend: str = Field(description="Backend")
    backend_label: str = Field(description="Backend label")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _row_to_response(row: dict[str, object]) -> dict[str, object]:
    """Convert a DB row dict to a response dict."""
    return {
        "id": row["id"],
        "backend": row["backend"],
        "backend_label": _BACKEND_LABELS.get(str(row["backend"]), str(row["backend"])),
        "host": row["host"] or "",
        "port": row["port"],
        "api_key_set": bool(row.get("api_key_encrypted")),
        "collection_name": row["collection_name"] or "",
        "is_active": bool(row.get("is_active", 1)),
        "user_id": row["user_id"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.get("")
async def get_config(user: dict = Depends(require_role("admin"))) -> VectorStoreResponse:  # noqa: B008
    """Get the active vector store configuration (or create default)."""
    async with get_db() as db:
        cursor = await db.execute(
            "SELECT * FROM vector_store_config WHERE user_id = ? AND is_active = 1 ORDER BY updated_at DESC LIMIT 1",
            (user["id"],),
        )
        row = await cursor.fetchone()

        if row:
            return VectorStoreResponse(**_row_to_response(dict(row)))

        # Auto-create default SQLite-VSS config
        config_id = str(uuid.uuid4())
        now = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        await db.execute(
            """INSERT INTO vector_store_config (id, backend, host, port, api_key_encrypted, collection_name, is_active, user_id, created_at, updated_at)
               VALUES (?, 'sqlite_vss', '', NULL, '', '', 1, ?, ?, ?)""",
            (config_id, user["id"], now, now),
        )
        await db.commit()

        cursor2 = await db.execute("SELECT * FROM vector_store_config WHERE id = ?", (config_id,))
        new_row = await cursor2.fetchone()
        return VectorStoreResponse(**_row_to_response(dict(new_row)))


@router.put("")
async def update_config(
    body: VectorStoreUpdate,
    user: dict = Depends(require_role("admin")),  # noqa: B008
) -> VectorStoreResponse:
    """Update the active vector store configuration."""
    async with get_db() as db:
        # Ensure config exists
        cursor = await db.execute(
            "SELECT id FROM vector_store_config WHERE user_id = ? AND is_active = 1 ORDER BY updated_at DESC LIMIT 1",
            (user["id"],),
        )
        row = await cursor.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="No vector store configuration found")

        config_id = row["id"]
        updates = body.model_dump(exclude_none=True)

        if not updates:
            raise HTTPException(status_code=400, detail="No fields to update")

        # Handle api_key encryption
        if "api_key" in updates:
            raw_key = updates.pop("api_key")
            if raw_key:
                updates["api_key_encrypted"] = encrypt_api_key(raw_key)
            else:
                updates["api_key_encrypted"] = ""

        updates["updated_at"] = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

        set_clause = ", ".join(f"{k} = ?" for k in updates)
        values = [*updates.values(), config_id]

        await db.execute(f"UPDATE vector_store_config SET {set_clause} WHERE id = ?", values)
        await db.commit()

        cursor2 = await db.execute("SELECT * FROM vector_store_config WHERE id = ?", (config_id,))
        updated = await cursor2.fetchone()
        return VectorStoreResponse(**_row_to_response(dict(updated)))


@router.post("/test")
async def test_connection(user: dict = Depends(require_role("admin"))) -> TestResult:  # noqa: B008
    """Test the active vector store connection."""
    async with get_db() as db:
        cursor = await db.execute(
            "SELECT * FROM vector_store_config WHERE user_id = ? AND is_active = 1 ORDER BY updated_at DESC LIMIT 1",
            (user["id"],),
        )
        row = await cursor.fetchone()

    if not row:
        return TestResult(
            success=False,
            error="No vector store configured",
            backend="unknown",
            backend_label="Unknown",
        )

    config = dict(row)
    backend = str(config["backend"])
    label = _BACKEND_LABELS.get(backend, backend)

    # SQLite-VSS is always available (built-in)
    if backend == "sqlite_vss":
        return TestResult(success=True, backend=backend, backend_label=label)

    # For external backends, attempt a lightweight connectivity check
    host = config.get("host") or ""
    port = config.get("port")
    api_key_enc = config.get("api_key_encrypted") or ""

    if backend in ("milvus", "qdrant", "chromadb") and not host:
        return TestResult(
            success=False, error="Host is required", backend=backend, backend_label=label
        )

    if backend == "pinecone" and not api_key_enc:
        return TestResult(
            success=False,
            error="API key is required for Pinecone",
            backend=backend,
            backend_label=label,
        )

    # Try TCP connect for host-based backends
    if backend in ("milvus", "qdrant", "chromadb") and host:
        import asyncio

        target_port = port or _BACKEND_DEFAULTS.get(backend, {}).get("port", 443)
        try:
            _, writer = await asyncio.wait_for(
                asyncio.open_connection(host, int(str(target_port))),  # type: ignore[arg-type]
                timeout=5.0,
            )
            writer.close()
            await writer.wait_closed()
            return TestResult(success=True, backend=backend, backend_label=label)
        except Exception as exc:
            return TestResult(
                success=False,
                error=f"Connection failed: {exc}",
                backend=backend,
                backend_label=label,
            )

    # Pinecone: try HTTPS health check
    if backend == "pinecone":
        try:
            import httpx

            api_key = decrypt_api_key(api_key_enc)
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(
                    "https://api.pinecone.io/indexes",
                    headers={"Api-Key": api_key},
                )
                if resp.status_code < 400:
                    return TestResult(success=True, backend=backend, backend_label=label)
                return TestResult(
                    success=False,
                    error=f"Pinecone API returned {resp.status_code}",
                    backend=backend,
                    backend_label=label,
                )
        except Exception as exc:
            return TestResult(
                success=False,
                error=f"Pinecone check failed: {exc}",
                backend=backend,
                backend_label=label,
            )

    return TestResult(success=True, backend=backend, backend_label=label)
