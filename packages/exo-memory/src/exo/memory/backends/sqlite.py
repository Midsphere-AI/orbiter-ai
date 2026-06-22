"""SQLite-backed memory store with JSON metadata indexes and soft deletes."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any

import aiosqlite  # pyright: ignore[reportMissingImports]

from exo.memory.backends._common import (  # pyright: ignore[reportMissingImports]
    build_metadata_filter_sqlite,
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
    metadata    TEXT NOT NULL DEFAULT '{}',
    extra_json  TEXT NOT NULL DEFAULT '{}',
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL,
    deleted     INTEGER NOT NULL DEFAULT 0,
    version     INTEGER NOT NULL DEFAULT 1
)"""

_CREATE_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_memory_type ON memory_items(memory_type) WHERE deleted = 0",
    "CREATE INDEX IF NOT EXISTS idx_status ON memory_items(status) WHERE deleted = 0",
    "CREATE INDEX IF NOT EXISTS idx_created_at ON memory_items(created_at) WHERE deleted = 0",
    "CREATE INDEX IF NOT EXISTS idx_category ON memory_items(category) WHERE deleted = 0",
]


def _default_db_path() -> str:
    """Return the default SQLite path from EXO_MEMORY_PATH or ~/.exo/memory.db."""
    raw = os.environ.get("EXO_MEMORY_PATH", "~/.exo/memory.db")
    return os.path.expanduser(raw)


class SQLiteMemoryStore:
    """SQLite-backed persistent memory store.

    Implements the MemoryStore protocol with JSON indexes for metadata
    fields, soft deletes, and a version field for optimistic concurrency.

    Use ``async with SQLiteMemoryStore(path) as store:`` or call
    ``await store.init()`` / ``await store.close()`` manually.

    When *db_path* is not provided, the path is read from the
    ``EXO_MEMORY_PATH`` environment variable (default:
    ``~/.exo/memory.db``).
    """

    __slots__ = ("_db", "_init_lock", "_initialized", "db_path")

    def __init__(self, db_path: str | None = None) -> None:
        self.db_path = db_path if db_path is not None else _default_db_path()
        self._db: aiosqlite.Connection | None = None
        self._initialized = False
        self._init_lock: asyncio.Lock | None = None

    # -- lifecycle ------------------------------------------------------------

    async def init(self) -> None:
        """Open the database and create tables if needed.

        Guarded by an asyncio.Lock to prevent double-initialization when
        multiple coroutines call init() concurrently before the first
        connection is established.
        """
        # Fast path: already initialized (no lock needed).
        if self._db is not None:
            return
        # Lazy-create the lock on first call (Lock must be created inside a
        # running event loop, so we can't do it in __init__).
        if self._init_lock is None:
            self._init_lock = asyncio.Lock()
        async with self._init_lock:
            if self._db is not None:
                return  # another coroutine beat us
            logger.debug("opening sqlite database path=%s", self.db_path)
            self._db = await aiosqlite.connect(self.db_path)
            db = self._db
            assert db is not None  # always set after connect()
            db.row_factory = aiosqlite.Row
            await db.execute(_CREATE_TABLE)
            for idx_sql in _CREATE_INDEXES:
                await db.execute(idx_sql)
            await db.commit()
            self._initialized = True
            logger.debug("sqlite memory store initialized")

    async def close(self) -> None:
        """Close the database connection."""
        if self._db is not None:
            await self._db.close()
            self._db = None
            self._initialized = False
            logger.debug("sqlite memory store closed")

    async def __aenter__(self) -> SQLiteMemoryStore:
        await self.init()
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.close()

    def _ensure_init(self) -> aiosqlite.Connection:
        if self._db is None:
            raise ExoMemoryError(
                "SQLiteMemoryStore is not initialized.",
                context={"db_path": self.db_path},
                hint=(
                    "Call `await store.init()` before use, "
                    "or use `async with SQLiteMemoryStore(path) as store:`."
                ),
            )
        return self._db

    # -- MemoryStore protocol -------------------------------------------------

    async def add(self, item: MemoryItem) -> None:
        """Persist a memory item (upsert — bumps version on conflict)."""
        db = self._ensure_init()
        extra = extra_fields(item)
        try:
            await db.execute(
                """\
                INSERT INTO memory_items
                    (id, content, memory_type, category, status, metadata, extra_json,
                     created_at, updated_at, deleted, version)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0, 1)
                ON CONFLICT(id) DO UPDATE SET
                    content    = excluded.content,
                    category   = excluded.category,
                    status     = excluded.status,
                    metadata   = excluded.metadata,
                    extra_json = excluded.extra_json,
                    updated_at = excluded.updated_at,
                    version    = version + 1,
                    deleted    = 0
                """,
                (
                    item.id,
                    item.content,
                    item.memory_type,
                    item.category.value if item.category is not None else None,
                    item.status.value,
                    item.metadata.model_dump_json(),
                    json.dumps(extra),
                    item.created_at,
                    item.updated_at,
                ),
            )
            await db.commit()
        except Exception as exc:
            raise ExoMemoryError(
                "SQLite add failed.",
                context={"item_id": item.id, "db_path": self.db_path},
                hint="Check disk space and file permissions on the database path.",
            ) from exc
        logger.debug("upserted item type=%s id=%s", item.memory_type, item.id)

    async def get(self, item_id: str) -> MemoryItem | None:
        """Retrieve a non-deleted memory item by ID."""
        db = self._ensure_init()
        try:
            async with db.execute(
                "SELECT * FROM memory_items WHERE id = ? AND deleted = 0",
                (item_id,),
            ) as cursor:
                row = await cursor.fetchone()
        except Exception as exc:
            raise ExoMemoryError(
                "SQLite get failed.",
                context={"item_id": item_id, "db_path": self.db_path},
                hint="Check the database file is readable and the schema is current.",
            ) from exc
        if row is None:
            return None
        return _row_to_item_sqlite(row)

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
        db = self._ensure_init()
        clauses: list[str] = ["deleted = 0"]
        params: list[Any] = []

        if memory_type:
            clauses.append("memory_type = ?")
            params.append(memory_type)
        if category is not None:
            clauses.append("category = ?")
            params.append(category.value)
        if status:
            clauses.append("status = ?")
            params.append(status.value)
        if query:
            clauses.append("content LIKE ?")
            params.append(f"%{query}%")
        if metadata:
            build_metadata_filter_sqlite(metadata, clauses, params)

        where = " AND ".join(clauses)
        sql = f"SELECT * FROM memory_items WHERE {where} ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        try:
            async with db.execute(sql, params) as cursor:
                rows = await cursor.fetchall()
        except Exception as exc:
            raise ExoMemoryError(
                "SQLite search failed.",
                context={"db_path": self.db_path},
                hint=(
                    "Check the database file is readable and the schema is current. "
                    "Run migrations if you recently upgraded exo-memory."
                ),
            ) from exc
        logger.debug("search returned %d rows", len(rows))
        return [_row_to_item_sqlite(r) for r in rows]

    async def clear(
        self,
        *,
        metadata: MemoryMetadata | None = None,
    ) -> int:
        """Soft-delete memory items matching the filter. Returns count."""
        db = self._ensure_init()
        # Guard: an empty MemoryMetadata (all fields None) would match ALL
        # records — equivalent to passing metadata=None but likely unintentional.
        # Return 0 to avoid silently deleting everything.
        if metadata is not None and not any(
            [metadata.user_id, metadata.session_id, metadata.task_id, metadata.agent_id]
        ):
            logger.debug("clear() called with all-None metadata fields — returning 0")
            return 0
        if metadata is None:
            cursor = await db.execute("UPDATE memory_items SET deleted = 1 WHERE deleted = 0")
            await db.commit()
            logger.debug("soft-deleted all items count=%d", cursor.rowcount)
            return cursor.rowcount

        clauses: list[str] = ["deleted = 0"]
        params: list[Any] = []
        build_metadata_filter_sqlite(metadata, clauses, params)

        where = " AND ".join(clauses)
        cursor = await db.execute(
            f"UPDATE memory_items SET deleted = 1 WHERE {where}",
            params,
        )
        await db.commit()
        logger.debug("soft-deleted filtered items count=%d", cursor.rowcount)
        return cursor.rowcount

    # -- extras ---------------------------------------------------------------

    async def count(self, *, include_deleted: bool = False) -> int:
        """Return the number of stored items."""
        db = self._ensure_init()
        where = "" if include_deleted else " WHERE deleted = 0"
        async with db.execute(f"SELECT COUNT(*) FROM memory_items{where}") as cursor:
            row = await cursor.fetchone()
        return row[0] if row else 0  # type: ignore[index]

    def __repr__(self) -> str:
        return f"SQLiteMemoryStore(db_path={self.db_path!r})"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _row_to_item_sqlite(row: Any) -> MemoryItem:
    """Reconstruct a MemoryItem from an aiosqlite Row.

    Parses the JSON metadata and extra_json columns (always stored as text
    in SQLite), then delegates to the shared ``row_to_item`` helper.
    """
    meta_dict = json.loads(row["metadata"])
    extra = json.loads(row["extra_json"])
    try:
        category_val = row["category"]
    except (IndexError, KeyError):
        category_val = None
    return _row_to_item_shared(
        row["id"],
        row["content"],
        row["memory_type"],
        row["status"],
        meta_dict,
        extra,
        row["created_at"],
        row["updated_at"],
        category_val,
    )
