"""PostgreSQL/pgvector vector store backend.

Requires the ``asyncpg`` and ``pgvector`` packages::

    pip install exo-retrieval[pgvector]
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any

from exo.retrieval.types import (  # pyright: ignore[reportMissingImports]
    Chunk,
    RetrievalError,
    RetrievalResult,
)
from exo.retrieval.vector_store import VectorStore  # pyright: ignore[reportMissingImports]

logger = logging.getLogger(__name__)

# Only [A-Za-z0-9_] is allowed in metadata filter key names to prevent SQL injection.
_SAFE_KEY_RE = re.compile(r"^[A-Za-z0-9_]+$")

try:
    import asyncpg  # type: ignore[import-untyped]
except ImportError as exc:
    msg = (
        "asyncpg is required for PgVectorStore. "
        "Install it with: pip install exo-retrieval[pgvector]"
    )
    raise ImportError(msg) from exc


_DEFAULT_TABLE = "exo_vectors"


class PgVectorStore(VectorStore):
    """PostgreSQL vector store using the pgvector extension.

    Uses ``asyncpg`` for async PostgreSQL access and the ``<=>`` cosine
    distance operator for similarity search.

    Args:
        dsn: PostgreSQL connection string (e.g. ``postgresql://user:pass@host/db``).
        table: Name of the table to store vectors in.
        dimensions: Dimensionality of embedding vectors.
        pool: Optional pre-existing ``asyncpg.Pool`` to use instead of
            creating one from *dsn*.
    """

    def __init__(
        self,
        dsn: str = "",
        *,
        table: str = _DEFAULT_TABLE,
        dimensions: int = 1536,
        pool: asyncpg.Pool | None = None,  # type: ignore[type-arg]
    ) -> None:
        self._dsn = dsn
        self._table = table
        self._dimensions = dimensions
        self._pool: asyncpg.Pool | None = pool  # type: ignore[type-arg]
        self._owns_pool = pool is None
        self._pool_lock = asyncio.Lock()
        self._initialized = pool is not None  # externally-provided pools are pre-initialized

    async def _get_pool(self) -> asyncpg.Pool:  # type: ignore[type-arg]
        """Return the connection pool, creating it if necessary.

        Uses a lock to prevent duplicate pool creation under concurrent calls.
        """
        if self._pool is not None:
            return self._pool
        async with self._pool_lock:
            # Re-check inside the lock (double-checked locking pattern)
            if self._pool is not None:
                return self._pool
            dsn_hint = (self._dsn[:20] + "...") if len(self._dsn) > 20 else self._dsn
            try:
                self._pool = await asyncpg.create_pool(self._dsn, timeout=10)
            except Exception as exc:
                raise RetrievalError(
                    f"pgvector connection failed: {exc}",
                    context={"dsn_hint": dsn_hint},
                    hint=(
                        "Check DATABASE_URL is set, the server is running,"
                        " and the pgvector extension is installed."
                    ),
                ) from exc
        return self._pool

    async def _ensure_initialized(self) -> asyncpg.Pool:  # type: ignore[type-arg]
        """Return the pool, auto-initializing the schema on first use."""
        pool = await self._get_pool()
        if not self._initialized:
            async with self._pool_lock:
                if not self._initialized:
                    await self._do_initialize(pool)
                    self._initialized = True
        return pool

    async def _do_initialize(self, pool: asyncpg.Pool) -> None:  # type: ignore[type-arg]
        """Internal: create the extension and table (no locking)."""
        try:
            async with pool.acquire() as conn:
                await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
                await conn.execute(f"""
                    CREATE TABLE IF NOT EXISTS {self._table} (
                        id BIGSERIAL PRIMARY KEY,
                        document_id TEXT NOT NULL,
                        chunk_index INTEGER NOT NULL,
                        content TEXT NOT NULL,
                        start_offset INTEGER NOT NULL,
                        end_offset INTEGER NOT NULL,
                        metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                        embedding vector({self._dimensions}) NOT NULL
                    )
                """)
                await conn.execute(f"""
                    CREATE INDEX IF NOT EXISTS idx_{self._table}_document_id
                    ON {self._table} (document_id)
                """)
        except RetrievalError:
            raise
        except Exception as exc:
            raise RetrievalError(
                f"pgvector initialize failed: {exc}",
                context={"table": self._table, "dimensions": self._dimensions},
                hint=(
                    "Ensure the database user has CREATE privileges and the pgvector"
                    " extension is available."
                ),
            ) from exc

    async def initialize(self) -> None:
        """Create the pgvector extension and table if they don't exist.

        Calling this explicitly is optional — the store lazily auto-initializes
        on first use.  It remains available for eager initialization or to
        pre-warm the schema before the first query.
        """
        pool = await self._get_pool()
        async with self._pool_lock:
            await self._do_initialize(pool)
            self._initialized = True

    async def add(
        self,
        chunks: list[Chunk],
        embeddings: list[list[float]],
    ) -> None:
        """Add chunks with their embedding vectors."""
        if len(chunks) != len(embeddings):
            msg = f"Number of chunks ({len(chunks)}) and embeddings ({len(embeddings)}) must match"
            raise RetrievalError(
                msg,
                context={"table": self._table, "operation": "add"},
                hint="Ensure embed_batch() is called on the same chunk list before calling add().",
            )

        if not chunks:
            return

        pool = await self._ensure_initialized()
        try:
            async with pool.acquire() as conn:
                await conn.executemany(
                    f"""
                    INSERT INTO {self._table}
                        (document_id, chunk_index, content, start_offset, end_offset, metadata, embedding)
                    VALUES ($1, $2, $3, $4, $5, $6::jsonb, $7::vector)
                    """,
                    [
                        (
                            chunk.document_id,
                            chunk.index,
                            chunk.content,
                            chunk.start,
                            chunk.end,
                            json.dumps(chunk.metadata),
                            _vector_literal(embedding),
                        )
                        for chunk, embedding in zip(chunks, embeddings, strict=True)
                    ],
                )
        except RetrievalError:
            raise
        except Exception as exc:
            raise RetrievalError(
                f"pgvector add failed: {exc}",
                context={"table": self._table, "operation": "add"},
                hint="Check the database connection and that the table is initialized.",
            ) from exc

    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        filter: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        """Search for similar chunks using cosine distance (``<=>``).

        Returns results ranked by similarity (highest score first).
        Cosine distance is converted to similarity via ``1 - distance``.
        """
        pool = await self._ensure_initialized()

        where_clauses: list[str] = []
        params: list[Any] = [_vector_literal(query_embedding), top_k]

        if filter:
            for key, value in filter.items():
                if not _SAFE_KEY_RE.match(key):
                    raise RetrievalError(
                        f"Invalid metadata filter key {key!r}: only [A-Za-z0-9_] characters"
                        " are allowed.",
                        context={"key": key},
                        hint=(
                            "Metadata filter keys must match [A-Za-z0-9_]. "
                            "Rename the key in your document metadata or sanitize it before filtering."
                        ),
                    )
                idx = len(params) + 1
                where_clauses.append(f"metadata->>'{key}' = ${idx}")
                params.append(str(value))

        where_sql = ""
        if where_clauses:
            where_sql = "WHERE " + " AND ".join(where_clauses)

        query = f"""
            SELECT document_id, chunk_index, content, start_offset, end_offset,
                   metadata, 1 - (embedding <=> $1::vector) AS score
            FROM {self._table}
            {where_sql}
            ORDER BY embedding <=> $1::vector
            LIMIT $2
        """

        try:
            async with pool.acquire() as conn:
                rows = await conn.fetch(query, *params)
        except RetrievalError:
            raise
        except Exception as exc:
            raise RetrievalError(
                f"pgvector search failed: {exc}",
                context={"table": self._table, "operation": "search"},
                hint=(
                    "Check the database connection and that the table is initialized."
                    " Dimension mismatch between query embedding and stored embeddings will"
                    " cause errors."
                ),
            ) from exc

        results: list[RetrievalResult] = []
        for row in rows:
            metadata = row["metadata"]
            if isinstance(metadata, str):
                metadata = json.loads(metadata)
            chunk = Chunk(
                document_id=row["document_id"],
                index=row["chunk_index"],
                content=row["content"],
                start=row["start_offset"],
                end=row["end_offset"],
                metadata=metadata,
            )
            results.append(RetrievalResult(chunk=chunk, score=float(row["score"])))

        return results

    async def delete(self, document_id: str) -> None:
        """Delete all chunks belonging to a document."""
        pool = await self._ensure_initialized()
        try:
            async with pool.acquire() as conn:
                await conn.execute(
                    f"DELETE FROM {self._table} WHERE document_id = $1",
                    document_id,
                )
        except RetrievalError:
            raise
        except Exception as exc:
            raise RetrievalError(
                f"pgvector delete failed: {exc}",
                context={"table": self._table, "operation": "delete", "document_id": document_id},
                hint="Check the database connection and that the table exists.",
            ) from exc

    async def clear(self) -> None:
        """Remove all stored chunks and embeddings."""
        pool = await self._ensure_initialized()
        async with pool.acquire() as conn:
            await conn.execute(f"TRUNCATE {self._table}")

    async def close(self) -> None:
        """Close the connection pool if we own it."""
        if self._pool is not None and self._owns_pool:
            await self._pool.close()
            self._pool = None


def _vector_literal(vec: list[float]) -> str:
    """Convert a list of floats to a pgvector literal string, e.g. ``'[1,2,3]'``."""
    return "[" + ",".join(str(v) for v in vec) + "]"
