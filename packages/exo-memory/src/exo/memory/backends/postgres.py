"""Postgres-backed memory store using asyncpg."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any

import asyncpg  # pyright: ignore[reportMissingImports]

from exo.memory.backends._common import (  # pyright: ignore[reportMissingImports]
    build_metadata_filter_postgres,
    extra_fields,
)
from exo.memory.backends._common import (
    row_to_item as _row_to_item_shared,
)
from exo.memory.base import (  # pyright: ignore[reportMissingImports]
    ExoMemoryError,
    MemoryCategory,
    MemoryItem,
    MemoryMetadata,
    MemoryStatus,
)

logger = logging.getLogger(__name__)

_CREATE_TABLE = """\
CREATE TABLE IF NOT EXISTS memory_items (
    id          TEXT PRIMARY KEY,
    content     TEXT NOT NULL,
    memory_type TEXT NOT NULL,
    category    TEXT,
    status      TEXT NOT NULL DEFAULT 'accepted',
    metadata    JSONB NOT NULL DEFAULT '{}',
    extra_json  JSONB NOT NULL DEFAULT '{}',
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL,
    deleted     SMALLINT NOT NULL DEFAULT 0,
    version     INTEGER NOT NULL DEFAULT 1
)"""

_CREATE_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_memory_type ON memory_items(memory_type) WHERE deleted = 0",
    "CREATE INDEX IF NOT EXISTS idx_status ON memory_items(status) WHERE deleted = 0",
    "CREATE INDEX IF NOT EXISTS idx_created_at ON memory_items(created_at) WHERE deleted = 0",
    "CREATE INDEX IF NOT EXISTS idx_metadata_user ON memory_items((metadata->>'user_id')) WHERE deleted = 0",
    "CREATE INDEX IF NOT EXISTS idx_metadata_session ON memory_items((metadata->>'session_id')) WHERE deleted = 0",
    "CREATE INDEX IF NOT EXISTS idx_category ON memory_items(category) WHERE deleted = 0",
]


class PostgresMemoryStore:
    """Postgres-backed persistent memory store.

    Implements the MemoryStore protocol with JSONB indexes for metadata
    fields, soft deletes, and a version field for optimistic concurrency.

    Use ``async with PostgresMemoryStore(dsn) as store:`` or call
    ``await store.init()`` / ``await store.close()`` manually.
    """

    __slots__ = ("_init_lock", "_initialized", "_pool", "dsn")

    def __init__(self, dsn: str = "postgresql://localhost/exo") -> None:
        self.dsn = dsn
        self._pool: asyncpg.Pool | None = None  # type: ignore[type-arg]
        self._initialized = False
        self._init_lock: asyncio.Lock | None = None

    # -- lifecycle ------------------------------------------------------------

    async def init(self) -> None:
        """Open the connection pool and create tables if needed.

        Guarded by an asyncio.Lock to prevent double-initialization when
        multiple coroutines call init() concurrently before the first pool
        is established.
        """
        # Fast path: already initialized (no lock needed).
        if self._pool is not None:
            return
        # Lazy-create the lock inside the event loop.
        if self._init_lock is None:
            self._init_lock = asyncio.Lock()
        async with self._init_lock:
            if self._pool is not None:
                return  # another coroutine beat us
            _safe_dsn = re.sub(r":[^:@/]+@", ":***@", self.dsn)
            logger.debug("connecting to postgres dsn=%s", _safe_dsn)
            self._pool = await asyncpg.create_pool(self.dsn)
            pool = self._pool
            assert pool is not None
            async with pool.acquire() as conn:
                await conn.execute(_CREATE_TABLE)
                for idx_sql in _CREATE_INDEXES:
                    await conn.execute(idx_sql)
            self._initialized = True
            logger.debug("postgres memory store initialized")

    async def close(self) -> None:
        """Close the connection pool."""
        if self._pool is not None:
            await self._pool.close()
            self._pool = None
            self._initialized = False
            logger.debug("postgres memory store closed")

    async def __aenter__(self) -> PostgresMemoryStore:
        await self.init()
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.close()

    def _ensure_init(self) -> asyncpg.Pool:  # type: ignore[type-arg]
        if self._pool is None:
            raise ExoMemoryError(
                "PostgresMemoryStore is not initialized.",
                context={"dsn": self.dsn},
                hint=(
                    "Call `await store.init()` before use, "
                    "or use `async with PostgresMemoryStore(dsn) as store:`."
                ),
            )
        return self._pool

    # -- MemoryStore protocol -------------------------------------------------

    async def add(self, item: MemoryItem) -> None:
        """Persist a memory item (upsert — bumps version on conflict)."""
        pool = self._ensure_init()
        extra = extra_fields(item)
        try:
            async with pool.acquire() as conn:
                await conn.execute(
                    """\
                    INSERT INTO memory_items
                        (id, content, memory_type, category, status, metadata, extra_json,
                         created_at, updated_at, deleted, version)
                    VALUES ($1, $2, $3, $4, $5, $6::jsonb, $7::jsonb, $8, $9, 0, 1)
                    ON CONFLICT (id) DO UPDATE SET
                        content    = EXCLUDED.content,
                        category   = EXCLUDED.category,
                        status     = EXCLUDED.status,
                        metadata   = EXCLUDED.metadata,
                        extra_json = EXCLUDED.extra_json,
                        updated_at = EXCLUDED.updated_at,
                        version    = memory_items.version + 1,
                        deleted    = 0
                    """,
                    item.id,
                    item.content,
                    item.memory_type,
                    item.category.value if item.category is not None else None,
                    item.status.value,
                    item.metadata.model_dump_json(),
                    json.dumps(extra),
                    item.created_at,
                    item.updated_at,
                )
        except ExoMemoryError:
            raise
        except Exception as exc:
            raise ExoMemoryError(
                "Postgres add failed.",
                context={"item_id": item.id, "dsn": self.dsn},
                hint="Check the DSN and ensure the Postgres server is reachable.",
            ) from exc
        logger.debug("upserted item type=%s id=%s", item.memory_type, item.id)

    async def get(self, item_id: str) -> MemoryItem | None:
        """Retrieve a non-deleted memory item by ID."""
        pool = self._ensure_init()
        try:
            async with pool.acquire() as conn:
                row = await conn.fetchrow(
                    "SELECT * FROM memory_items WHERE id = $1 AND deleted = 0",
                    item_id,
                )
        except ExoMemoryError:
            raise
        except Exception as exc:
            raise ExoMemoryError(
                "Postgres get failed.",
                context={"item_id": item_id, "dsn": self.dsn},
                hint="Check the DSN and ensure the Postgres server is reachable.",
            ) from exc
        if row is None:
            return None
        return _row_to_item_postgres(row)

    async def search(
        self,
        *,
        query: str = "",
        metadata: MemoryMetadata | None = None,
        memory_type: str | None = None,
        category: MemoryCategory | None = None,
        status: MemoryStatus | None = None,
        limit: int = 10,
    ) -> list[MemoryItem]:
        """Search memory items with optional filters."""
        pool = self._ensure_init()
        clauses: list[str] = ["deleted = 0"]
        params: list[Any] = []
        idx = 1

        if memory_type:
            clauses.append(f"memory_type = ${idx}")
            params.append(memory_type)
            idx += 1
        if category is not None:
            clauses.append(f"category = ${idx}")
            params.append(category.value)
            idx += 1
        if status:
            clauses.append(f"status = ${idx}")
            params.append(status.value)
            idx += 1
        if query:
            clauses.append(f"content ILIKE ${idx}")
            params.append(f"%{query}%")
            idx += 1
        if metadata:
            idx = build_metadata_filter_postgres(metadata, clauses, params, idx)

        where = " AND ".join(clauses)
        sql = f"SELECT * FROM memory_items WHERE {where} ORDER BY created_at DESC LIMIT ${idx}"
        params.append(limit)

        try:
            async with pool.acquire() as conn:
                rows = await conn.fetch(sql, *params)
        except ExoMemoryError:
            raise
        except Exception as exc:
            raise ExoMemoryError(
                "Postgres search failed.",
                context={"dsn": self.dsn},
                hint="Check the DSN and ensure the Postgres server is reachable.",
            ) from exc
        logger.debug("search returned %d rows", len(rows))
        return [_row_to_item_postgres(r) for r in rows]

    async def clear(
        self,
        *,
        metadata: MemoryMetadata | None = None,
    ) -> int:
        """Soft-delete memory items matching the filter. Returns count."""
        pool = self._ensure_init()
        # Guard: an empty MemoryMetadata (all fields None) would match ALL
        # records — equivalent to passing metadata=None but likely unintentional.
        # Return 0 to avoid silently deleting everything.
        if metadata is not None and not any(
            [metadata.user_id, metadata.session_id, metadata.task_id, metadata.agent_id]
        ):
            logger.debug("clear() called with all-None metadata fields — returning 0")
            return 0
        try:
            async with pool.acquire() as conn:
                if metadata is None:
                    result = await conn.execute(
                        "UPDATE memory_items SET deleted = 1 WHERE deleted = 0"
                    )
                    count = _parse_rowcount(result)
                    logger.debug("soft-deleted all items count=%d", count)
                    return count

                clauses: list[str] = ["deleted = 0"]
                params: list[Any] = []
                build_metadata_filter_postgres(metadata, clauses, params, 1)

                where = " AND ".join(clauses)
                result = await conn.execute(
                    f"UPDATE memory_items SET deleted = 1 WHERE {where}",
                    *params,
                )
                count = _parse_rowcount(result)
                logger.debug("soft-deleted filtered items count=%d", count)
                return count
        except ExoMemoryError:
            raise
        except Exception as exc:
            raise ExoMemoryError(
                "Postgres clear failed.",
                context={"dsn": self.dsn},
                hint="Check the DSN and ensure the Postgres server is reachable.",
            ) from exc

    async def get_recent(
        self,
        n: int = 10,
        *,
        metadata: MemoryMetadata | None = None,
    ) -> list[MemoryItem]:
        """Return the *n* most recently created items, newest first.

        Optionally scoped to a metadata filter (agent_id, task_id, etc.).
        """
        pool = self._ensure_init()
        clauses: list[str] = ["deleted = 0"]
        params: list[Any] = []
        idx = 1

        if metadata:
            idx = build_metadata_filter_postgres(metadata, clauses, params, idx)

        where = " AND ".join(clauses)
        sql = f"SELECT * FROM memory_items WHERE {where} ORDER BY created_at DESC LIMIT ${idx}"
        params.append(n)

        try:
            async with pool.acquire() as conn:
                rows = await conn.fetch(sql, *params)
        except ExoMemoryError:
            raise
        except Exception as exc:
            raise ExoMemoryError(
                "Postgres get_recent failed.",
                context={"dsn": self.dsn},
                hint="Check the DSN and ensure the Postgres server is reachable.",
            ) from exc
        logger.debug("get_recent n=%d returned %d rows", n, len(rows))
        return [_row_to_item_postgres(r) for r in rows]

    # -- extras ---------------------------------------------------------------

    async def count(self, *, include_deleted: bool = False) -> int:
        """Return the number of stored items."""
        pool = self._ensure_init()
        where = "" if include_deleted else " WHERE deleted = 0"
        async with pool.acquire() as conn:
            row = await conn.fetchval(f"SELECT COUNT(*) FROM memory_items{where}")
        return row or 0

    def __repr__(self) -> str:
        return f"PostgresMemoryStore(dsn={self.dsn!r})"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_rowcount(result: str) -> int:
    """Parse the row count from asyncpg's UPDATE command status string."""
    # asyncpg returns e.g. "UPDATE 3"
    parts = result.split()
    if len(parts) >= 2:
        try:
            return int(parts[-1])
        except ValueError:
            return 0
    return 0


def _row_to_item_postgres(row: asyncpg.Record) -> MemoryItem:
    """Reconstruct a MemoryItem from an asyncpg Record.

    Postgres JSONB columns may be returned as dicts (asyncpg parses them
    automatically) or as JSON strings depending on driver configuration.
    Both cases are handled before delegating to the shared ``row_to_item``.
    """
    meta_raw = row["metadata"]
    meta_dict = json.loads(meta_raw) if isinstance(meta_raw, str) else dict(meta_raw)

    extra_raw = row["extra_json"]
    extra = json.loads(extra_raw) if isinstance(extra_raw, str) else dict(extra_raw)

    return _row_to_item_shared(
        row["id"],
        row["content"],
        row["memory_type"],
        row["status"],
        meta_dict,
        extra,
        row["created_at"],
        row["updated_at"],
        row["category"],
    )
