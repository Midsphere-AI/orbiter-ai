"""Database connection management and migration runner."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import aiosqlite

from exo_web.config import settings

logger = logging.getLogger(__name__)

# Extract the file path from the database URL.
# Supports "sqlite+aiosqlite:///path" and plain "path.db" formats.
_DB_URL = settings.database_url
_DB_PATH = _DB_URL.split("///", 1)[-1] if _DB_URL.startswith("sqlite") else _DB_URL

MIGRATIONS_DIR = Path(__file__).parent / "migrations"

# ---------------------------------------------------------------------------
# Connection pool
#
# aiosqlite connections are NOT safe to share across concurrent coroutines for
# writes — the underlying sqlite3 connection is not reentrant.  We therefore
# maintain a small pool of connections managed via an asyncio.Queue.
#
# * Pool size: 5 (enough for the typical FastAPI request concurrency under WAL).
# * WAL + FK PRAGMAs are applied once when each connection is first created,
#   not on every get_db() call.
# * Connections are returned to the pool in the ``finally`` block of get_db(),
#   so they persist for the lifetime of the process.
# * The pool is initialised lazily on first use (or explicitly via
#   ``init_pool()`` from the FastAPI lifespan), so tests that never touch the
#   network still work without a live DB.
# ---------------------------------------------------------------------------

_POOL_SIZE = 5
_pool: asyncio.Queue[aiosqlite.Connection] | None = None
_pool_lock = asyncio.Lock()


async def _make_connection() -> aiosqlite.Connection:
    """Open a fresh aiosqlite connection with PRAGMAs applied."""
    db = await aiosqlite.connect(_DB_PATH)
    db.row_factory = aiosqlite.Row
    await db.execute("PRAGMA journal_mode=WAL")
    await db.execute("PRAGMA foreign_keys=ON")
    return db


async def init_pool() -> None:
    """Initialise the connection pool.  Safe to call multiple times."""
    global _pool
    async with _pool_lock:
        if _pool is not None:
            return
        q: asyncio.Queue[aiosqlite.Connection] = asyncio.Queue(maxsize=_POOL_SIZE)
        for _ in range(_POOL_SIZE):
            conn = await _make_connection()
            await q.put(conn)
        _pool = q
        logger.debug(
            "Database connection pool initialised (size=%d, path=%s)", _POOL_SIZE, _DB_PATH
        )


async def close_pool() -> None:
    """Close all pooled connections.  Call from lifespan shutdown."""
    global _pool
    async with _pool_lock:
        if _pool is None:
            return
        while not _pool.empty():
            conn = _pool.get_nowait()
            with contextlib.suppress(Exception):
                await conn.close()
        _pool = None
        logger.debug("Database connection pool closed")


async def _get_pool() -> asyncio.Queue[aiosqlite.Connection]:
    """Return the pool, initialising it lazily on first call."""
    if _pool is None:
        await init_pool()
    assert _pool is not None
    return _pool


@asynccontextmanager
async def get_db() -> AsyncIterator[aiosqlite.Connection]:
    """Yield a pooled aiosqlite connection as an async context manager.

    Used by most of the codebase via ``async with get_db() as db:``.
    For FastAPI dependencies, use :func:`get_db_dep` instead.

    The connection is borrowed from a fixed-size pool, so PRAGMAs are only
    applied once at pool-initialisation time rather than on every request.
    The connection is returned to the pool when the ``async with`` block exits,
    regardless of whether an exception was raised.
    """
    pool = await _get_pool()
    db = await pool.get()
    try:
        yield db
    finally:
        await pool.put(db)


async def get_db_dep() -> AsyncIterator[aiosqlite.Connection]:
    """FastAPI-compatible dependency that yields an aiosqlite connection.

    FastAPI expects a bare async generator, not an async context manager.
    """
    async with get_db() as db:
        yield db


async def run_migrations() -> list[str]:
    """Run all pending migrations in order.

    Migrations are sequential .sql files in the migrations/ directory,
    named like 001_create_users.sql, 002_create_projects.sql.

    Returns the list of newly applied migration filenames.
    """
    applied: list[str] = []

    async with get_db() as db:
        # Ensure the migrations tracking table exists.
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS _migrations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                applied_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
            """
        )
        await db.commit()

        # Get already-applied migrations.
        cursor = await db.execute("SELECT name FROM _migrations ORDER BY id")
        rows = await cursor.fetchall()
        already_applied = {row[0] for row in rows}

        # Find and sort migration files.
        if not MIGRATIONS_DIR.is_dir():
            return applied

        migration_files = sorted(f for f in os.listdir(MIGRATIONS_DIR) if f.endswith(".sql"))

        for filename in migration_files:
            if filename in already_applied:
                continue

            sql = (MIGRATIONS_DIR / filename).read_text()
            await db.executescript(sql)
            await db.execute("INSERT INTO _migrations (name) VALUES (?)", (filename,))
            await db.commit()
            applied.append(filename)

    return applied
