"""Knowledge bases CRUD REST API and document management."""

from __future__ import annotations

import json
import logging
import uuid
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, UploadFile
from pydantic import BaseModel, Field

from exo_web.database import get_db
from exo_web.routes.auth import get_current_user
from exo_web.sanitize import sanitize_html
from exo_web.services.document_processor import chunk_text, extract_text

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/knowledge-bases", tags=["knowledge-bases"])


def _escape_like(term: str, escape_char: str = "\\") -> str:
    """Escape LIKE special characters in *term* so they match literally.

    SQLite LIKE treats ``%``, ``_``, and the escape character itself as
    special.  Escape them so user-supplied search terms are safe to embed
    inside a ``%…%`` pattern.
    """
    term = term.replace(escape_char, escape_char + escape_char)
    term = term.replace("%", escape_char + "%")
    term = term.replace("_", escape_char + "_")
    return term


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class KnowledgeBaseCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255, description="Display name")
    description: str = Field("", description="Human-readable description")
    embedding_model: str = Field("text-embedding-3-small", description="Embedding model")
    chunk_size: int = Field(512, ge=64, le=8192, description="Chunk size")
    chunk_overlap: int = Field(50, ge=0, le=4096, description="Chunk overlap")
    search_type: str = Field(
        "keyword", pattern=r"^(semantic|keyword|hybrid)$", description="Search type"
    )
    top_k: int = Field(5, ge=1, le=20, description="Top k")
    similarity_threshold: float = Field(0.0, ge=0.0, le=1.0, description="Similarity threshold")
    reranker_enabled: bool = Field(False, description="Reranker enabled")
    project_id: str | None = Field(None, description="Associated project identifier")


class KnowledgeBaseUpdate(BaseModel):
    name: str | None = Field(None, min_length=1, max_length=255, description="Display name")
    description: str | None = Field(None, description="Human-readable description")
    embedding_model: str | None = Field(None, description="Embedding model")
    chunk_size: int | None = Field(None, ge=64, le=8192, description="Chunk size")
    chunk_overlap: int | None = Field(None, ge=0, le=4096, description="Chunk overlap")
    search_type: str | None = Field(
        None, pattern=r"^(semantic|keyword|hybrid)$", description="Search type"
    )
    top_k: int | None = Field(None, ge=1, le=20, description="Top k")
    similarity_threshold: float | None = Field(
        None, ge=0.0, le=1.0, description="Similarity threshold"
    )
    reranker_enabled: bool | None = Field(None, description="Reranker enabled")


class KnowledgeBaseResponse(BaseModel):
    id: str = Field(description="Unique identifier")
    name: str = Field(description="Display name")
    description: str = Field(description="Human-readable description")
    embedding_model: str = Field(description="Embedding model")
    chunk_size: int = Field(description="Chunk size")
    chunk_overlap: int = Field(description="Chunk overlap")
    search_type: str = Field(description="Search type")
    top_k: int = Field(description="Top k")
    similarity_threshold: float = Field(description="Similarity threshold")
    reranker_enabled: bool = Field(description="Reranker enabled")
    doc_count: int = Field(description="Doc count")
    chunk_count: int = Field(description="Chunk count")
    project_id: str | None = Field(description="Associated project identifier")
    user_id: str = Field(description="Owning user identifier")
    created_at: str = Field(description="ISO 8601 creation timestamp")
    updated_at: str = Field(description="ISO 8601 last-update timestamp")


class DocumentResponse(BaseModel):
    id: str = Field(description="Unique identifier")
    kb_id: str = Field(description="Kb id")
    filename: str = Field(description="Filename")
    file_type: str = Field(description="File type")
    file_size: int = Field(description="File size")
    chunk_count: int = Field(description="Chunk count")
    metadata_json: str = Field(description="Metadata json")
    status: str = Field(description="Current status")
    created_at: str = Field(description="ISO 8601 creation timestamp")


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Query")


class SearchResult(BaseModel):
    chunk_id: str = Field(description="Chunk id")
    document_id: str = Field(description="Document id")
    filename: str = Field(description="Filename")
    chunk_index: int = Field(description="Chunk index")
    content: str = Field(description="Text content")
    score: float = Field(description="Score")


# Supported file extensions
_SUPPORTED_TYPES = {"pdf", "docx", "txt", "md", "csv", "html"}

# 50 MB upload limit
_MAX_FILE_SIZE = 50 * 1024 * 1024


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _row_to_dict(row: Any) -> dict[str, Any]:
    """Convert an aiosqlite.Row to a plain dict."""
    d = dict(row)
    if "reranker_enabled" in d:
        d["reranker_enabled"] = bool(d["reranker_enabled"])
    return d


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("", response_model=list[KnowledgeBaseResponse])
async def list_knowledge_bases(
    project_id: str | None = Query(None),
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> list[dict[str, Any]]:
    """Return all knowledge bases for the current user, optionally filtered by project."""
    async with get_db() as db:
        if project_id:
            cursor = await db.execute(
                "SELECT * FROM knowledge_bases WHERE user_id = ? AND project_id = ? ORDER BY created_at DESC",
                (user["id"], project_id),
            )
        else:
            cursor = await db.execute(
                "SELECT * FROM knowledge_bases WHERE user_id = ? ORDER BY created_at DESC",
                (user["id"],),
            )
        rows = await cursor.fetchall()
        return [_row_to_dict(r) for r in rows]


@router.post("", response_model=KnowledgeBaseResponse, status_code=201)
async def create_knowledge_base(
    body: KnowledgeBaseCreate,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> dict[str, Any]:
    """Create a new knowledge base."""
    kb_id = str(uuid.uuid4())
    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")

    async with get_db() as db:
        await db.execute(
            """
            INSERT INTO knowledge_bases
                (id, name, description, embedding_model, chunk_size, chunk_overlap,
                 search_type, top_k, similarity_threshold, reranker_enabled,
                 project_id, user_id, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                kb_id,
                sanitize_html(body.name),
                sanitize_html(body.description),
                body.embedding_model,
                body.chunk_size,
                body.chunk_overlap,
                body.search_type,
                body.top_k,
                body.similarity_threshold,
                int(body.reranker_enabled),
                body.project_id,
                user["id"],
                now,
                now,
            ),
        )
        await db.commit()

        cursor = await db.execute("SELECT * FROM knowledge_bases WHERE id = ?", (kb_id,))
        row = await cursor.fetchone()
        return _row_to_dict(row)


@router.get("/{kb_id}", response_model=KnowledgeBaseResponse)
async def get_knowledge_base(
    kb_id: str,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> dict[str, Any]:
    """Return a single knowledge base with stats."""
    async with get_db() as db:
        cursor = await db.execute(
            "SELECT * FROM knowledge_bases WHERE id = ? AND user_id = ?",
            (kb_id, user["id"]),
        )
        row = await cursor.fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="Knowledge base not found")
        return _row_to_dict(row)


@router.put("/{kb_id}", response_model=KnowledgeBaseResponse)
async def update_knowledge_base(
    kb_id: str,
    body: KnowledgeBaseUpdate,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> dict[str, Any]:
    """Update a knowledge base's config."""
    updates = body.model_dump(exclude_none=True)
    if not updates:
        raise HTTPException(status_code=422, detail="No fields to update")

    for field in ("name", "description"):
        if field in updates and isinstance(updates[field], str):
            updates[field] = sanitize_html(updates[field])

    # SQLite stores bools as integers
    if "reranker_enabled" in updates:
        updates["reranker_enabled"] = int(updates["reranker_enabled"])

    async with get_db() as db:
        cursor = await db.execute(
            "SELECT id FROM knowledge_bases WHERE id = ? AND user_id = ?",
            (kb_id, user["id"]),
        )
        if await cursor.fetchone() is None:
            raise HTTPException(status_code=404, detail="Knowledge base not found")

        updates["updated_at"] = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")
        set_clause = ", ".join(f"{k} = ?" for k in updates)
        values = [*list(updates.values()), kb_id]

        await db.execute(
            f"UPDATE knowledge_bases SET {set_clause} WHERE id = ?",
            values,
        )
        await db.commit()

        cursor = await db.execute("SELECT * FROM knowledge_bases WHERE id = ?", (kb_id,))
        row = await cursor.fetchone()
        return _row_to_dict(row)


@router.delete("/{kb_id}", status_code=204)
async def delete_knowledge_base(
    kb_id: str,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> None:
    """Delete a knowledge base and all its documents."""
    async with get_db() as db:
        cursor = await db.execute(
            "SELECT id FROM knowledge_bases WHERE id = ? AND user_id = ?",
            (kb_id, user["id"]),
        )
        if await cursor.fetchone() is None:
            raise HTTPException(status_code=404, detail="Knowledge base not found")

        await db.execute("DELETE FROM knowledge_bases WHERE id = ?", (kb_id,))
        await db.commit()


# ---------------------------------------------------------------------------
# Search endpoint
# ---------------------------------------------------------------------------


def _keyword_score(query_terms: list[str], content: str) -> float:
    """Compute a simple keyword match score (0.0-1.0) based on term frequency."""
    lower = content.lower()
    if not query_terms:
        return 0.0
    hits = sum(1 for term in query_terms if term in lower)
    return hits / len(query_terms)


@router.post("/{kb_id}/search", response_model=list[SearchResult])
async def search_knowledge_base(
    kb_id: str,
    body: SearchRequest,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> list[dict[str, Any]]:
    """Search a knowledge base and return ranked chunks with scores.

    Uses keyword matching against stored document chunks. Semantic and hybrid
    search modes will use the same keyword fallback until embedding support
    is added.
    """
    async with get_db() as db:
        # Verify KB access and get retrieval settings
        cursor = await db.execute(
            "SELECT * FROM knowledge_bases WHERE id = ? AND user_id = ?",
            (kb_id, user["id"]),
        )
        kb = await cursor.fetchone()
        if kb is None:
            raise HTTPException(status_code=404, detail="Knowledge base not found")

        kb = dict(kb)
        top_k = kb.get("top_k", 5)
        similarity_threshold = kb.get("similarity_threshold", 0.0)

        # Build keyword search: split query into terms, search with LIKE
        query_terms = [t.lower() for t in body.query.split() if t.strip()]
        if not query_terms:
            return []

        # Use LIKE for each term with OR to find chunks containing any query term.
        # Escape LIKE wildcards in user terms to prevent injection via % or _.
        escaped_terms = [_escape_like(t) for t in query_terms]
        like_clauses = " OR ".join("dc.content LIKE ? ESCAPE '\\'" for _ in escaped_terms)
        like_params = [f"%{term}%" for term in escaped_terms]

        cursor = await db.execute(
            f"""
            SELECT dc.id AS chunk_id, dc.document_id, d.filename,
                   dc.chunk_index, dc.content
            FROM document_chunks dc
            JOIN documents d ON d.id = dc.document_id
            WHERE dc.kb_id = ? AND ({like_clauses})
            """,
            [kb_id, *like_params],
        )
        rows = await cursor.fetchall()

    # Score and rank results
    scored: list[dict[str, Any]] = []
    for row in rows:
        row_dict = dict(row)
        score = _keyword_score(query_terms, row_dict["content"])
        if score >= similarity_threshold:
            scored.append({**row_dict, "score": round(score, 4)})

    # Sort by score descending, take top_k
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


# ---------------------------------------------------------------------------
# Document endpoints
# ---------------------------------------------------------------------------


async def _verify_kb_access(kb_id: str, user_id: str) -> None:
    """Raise 404 if the knowledge base doesn't exist or doesn't belong to the user."""
    async with get_db() as db:
        cursor = await db.execute(
            "SELECT id FROM knowledge_bases WHERE id = ? AND user_id = ?",
            (kb_id, user_id),
        )
        if await cursor.fetchone() is None:
            raise HTTPException(status_code=404, detail="Knowledge base not found")


def _file_extension(filename: str) -> str:
    """Return lowercase file extension without the dot."""
    return filename.rsplit(".", 1)[-1].lower() if "." in filename else ""


@router.post("/{kb_id}/documents", response_model=DocumentResponse, status_code=201)
async def upload_document(
    kb_id: str,
    file: UploadFile,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> dict[str, Any]:
    """Upload a document to a knowledge base.

    Accepts file upload, extracts text, chunks it, and stores everything.
    """
    await _verify_kb_access(kb_id, user["id"])

    # Validate filename and type
    if not file.filename:
        raise HTTPException(status_code=422, detail="Filename is required")
    ext = _file_extension(file.filename)
    if ext not in _SUPPORTED_TYPES:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported file type '.{ext}'. Supported: {', '.join(sorted(_SUPPORTED_TYPES))}",
        )

    # Read file content
    content = await file.read()
    if len(content) > _MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File too large (max 50MB)")
    if len(content) == 0:
        raise HTTPException(status_code=422, detail="File is empty")

    # Extract text
    try:
        text = extract_text(content, ext)
    except Exception:
        logger.exception("Failed to extract text from %s", file.filename)
        raise HTTPException(  # noqa: B904
            status_code=422, detail="Failed to extract text from file"
        )

    if not text.strip():
        raise HTTPException(status_code=422, detail="No text content found in file")

    # Get KB chunking config
    async with get_db() as db:
        cursor = await db.execute(
            "SELECT chunk_size, chunk_overlap FROM knowledge_bases WHERE id = ?",
            (kb_id,),
        )
        kb_row = await cursor.fetchone()
        kb_chunk_size = kb_row["chunk_size"] if kb_row else 512
        kb_chunk_overlap = kb_row["chunk_overlap"] if kb_row else 50

    # Chunk the text
    chunks = chunk_text(text, chunk_size=kb_chunk_size, chunk_overlap=kb_chunk_overlap)

    # Store document and chunks
    doc_id = str(uuid.uuid4())
    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")
    metadata = json.dumps({"original_filename": file.filename, "content_type": file.content_type})

    async with get_db() as db:
        await db.execute(
            """
            INSERT INTO documents (id, kb_id, filename, file_type, file_size, chunk_count, metadata_json, status, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                doc_id,
                kb_id,
                file.filename,
                ext,
                len(content),
                len(chunks),
                metadata,
                "completed",
                now,
            ),
        )

        # Insert chunks
        for i, chunk in enumerate(chunks):
            chunk_id = str(uuid.uuid4())
            chunk_meta = json.dumps({"chunk_index": i, "char_count": len(chunk)})
            await db.execute(
                """
                INSERT INTO document_chunks (id, document_id, kb_id, chunk_index, content, char_count, metadata_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (chunk_id, doc_id, kb_id, i, chunk, len(chunk), chunk_meta, now),
            )

        # Update KB counters
        await db.execute(
            "UPDATE knowledge_bases SET doc_count = doc_count + 1, chunk_count = chunk_count + ?, updated_at = ? WHERE id = ?",
            (len(chunks), now, kb_id),
        )
        await db.commit()

        cursor = await db.execute("SELECT * FROM documents WHERE id = ?", (doc_id,))
        row = await cursor.fetchone()
        return _row_to_dict(row)


@router.get("/{kb_id}/documents", response_model=list[DocumentResponse])
async def list_documents(
    kb_id: str,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> list[dict[str, Any]]:
    """Return all documents for a knowledge base."""
    await _verify_kb_access(kb_id, user["id"])

    async with get_db() as db:
        cursor = await db.execute(
            "SELECT * FROM documents WHERE kb_id = ? ORDER BY created_at DESC",
            (kb_id,),
        )
        rows = await cursor.fetchall()
        return [_row_to_dict(r) for r in rows]


@router.get("/{kb_id}/documents/{doc_id}/chunks")
async def list_document_chunks(
    kb_id: str,
    doc_id: str,
    limit: int = Query(3, ge=1, le=50),
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> list[dict[str, Any]]:
    """Return the first N chunks of a document (for preview)."""
    await _verify_kb_access(kb_id, user["id"])

    async with get_db() as db:
        cursor = await db.execute(
            "SELECT id, chunk_index, content, char_count FROM document_chunks WHERE document_id = ? AND kb_id = ? ORDER BY chunk_index LIMIT ?",
            (doc_id, kb_id, limit),
        )
        rows = await cursor.fetchall()
        return [_row_to_dict(r) for r in rows]


@router.delete("/{kb_id}/documents/{doc_id}", status_code=204)
async def delete_document(
    kb_id: str,
    doc_id: str,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> None:
    """Delete a document and its chunks from a knowledge base."""
    await _verify_kb_access(kb_id, user["id"])

    async with get_db() as db:
        cursor = await db.execute(
            "SELECT chunk_count FROM documents WHERE id = ? AND kb_id = ?",
            (doc_id, kb_id),
        )
        row = await cursor.fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="Document not found")

        removed_chunks = row["chunk_count"]

        # Cascade delete handles document_chunks via FK
        await db.execute("DELETE FROM documents WHERE id = ?", (doc_id,))

        # Update KB counters
        now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")
        await db.execute(
            "UPDATE knowledge_bases SET doc_count = doc_count - 1, chunk_count = chunk_count - ?, updated_at = ? WHERE id = ?",
            (removed_chunks, now, kb_id),
        )
        await db.commit()
