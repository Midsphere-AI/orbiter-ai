"""Versioned migration system for memory store schemas.

Provides a registry of schema migrations that can be applied to
SQLite and Postgres memory backends in order, with version tracking
via a ``_migrations`` metadata table.
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from exo.memory.base import ExoMemoryError  # pyright: ignore[reportMissingImports]

logger = logging.getLogger(__name__)


class MigrationError(ExoMemoryError):
    """Raised when a migration fails."""


@dataclass(frozen=True)
class Migration:
    """A single schema migration.

    Attributes:
        version: Monotonically increasing version number (1-based).
        description: Human-readable description of the change.
        up: Async callable that applies the migration. Receives
            a raw database connection (aiosqlite.Connection or
            asyncpg.Connection).
        down: Optional async callable that reverses the migration.
    """

    version: int
    description: str
    up: Callable[[Any], Awaitable[None]]
    down: Callable[[Any], Awaitable[None]] | None = None


@dataclass
class MigrationRegistry:
    """Registry of ordered schema migrations.

    Migrations are kept sorted by version. Use :meth:`register` to add
    new migrations and :meth:`list_pending` to find which ones still
    need to be applied.
    """

    _migrations: list[Migration] = field(default_factory=list)

    def register(self, migration: Migration) -> None:
        """Register a migration. Raises if version is duplicate."""
        for m in self._migrations:
            if m.version == migration.version:
                raise MigrationError(
                    f"Duplicate migration version {migration.version}.",
                    context={"version": migration.version},
                    hint=(
                        f"Already registered versions: "
                        f"{[m.version for m in self._migrations]}. "
                        "Each migration must have a unique, monotonically increasing version."
                    ),
                )
        self._migrations.append(migration)
        self._migrations.sort(key=lambda m: m.version)

    def list_pending(self, current_version: int) -> list[Migration]:
        """Return migrations with version > current_version, sorted ascending."""
        return [m for m in self._migrations if m.version > current_version]

    @property
    def all(self) -> list[Migration]:
        """All registered migrations, sorted by version."""
        return list(self._migrations)


# ---------------------------------------------------------------------------
# _migrations table DDL
# ---------------------------------------------------------------------------

_SQLITE_CREATE_MIGRATIONS_TABLE = """\
CREATE TABLE IF NOT EXISTS _migrations (
    version     INTEGER PRIMARY KEY,
    description TEXT NOT NULL,
    applied_at  TEXT NOT NULL
)"""

_POSTGRES_CREATE_MIGRATIONS_TABLE = """\
CREATE TABLE IF NOT EXISTS _migrations (
    version     INTEGER PRIMARY KEY,
    description TEXT NOT NULL,
    applied_at  TEXT NOT NULL
)"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def run_migrations(store: Any, registry: MigrationRegistry) -> int:
    """Apply pending migrations to a memory store backend.

    Works with both :class:`SQLiteMemoryStore` and
    :class:`PostgresMemoryStore`. The store must be initialized
    (``await store.init()``) before calling this function.

    Returns the number of migrations applied.

    Raises:
        MigrationError: If a migration fails (already-applied migrations
            are preserved).
    """
    backend = _detect_backend(store)
    if backend == "sqlite":
        return await _run_sqlite(store, registry)
    if backend == "postgres":
        return await _run_postgres(store, registry)
    raise MigrationError(
        f"Unsupported store type: {type(store).__name__}.",
        context={"store_type": type(store).__name__},
        hint=(
            "Pass a SQLiteMemoryStore or PostgresMemoryStore instance. "
            f"Got: {type(store).__name__}."
        ),
    )


# ---------------------------------------------------------------------------
# SQLite implementation
# ---------------------------------------------------------------------------


async def _run_sqlite(store: Any, registry: MigrationRegistry) -> int:
    db = store._ensure_init()
    await db.execute(_SQLITE_CREATE_MIGRATIONS_TABLE)
    await db.commit()

    current_version = await _sqlite_current_version(db)
    pending = registry.list_pending(current_version)
    if not pending:
        return 0

    applied = 0
    for migration in pending:
        try:
            await migration.up(db)
            await db.execute(
                "INSERT INTO _migrations (version, description, applied_at) "
                "VALUES (?, ?, datetime('now'))",
                (migration.version, migration.description),
            )
            await db.commit()
            applied += 1
            logger.info("Applied migration v%d: %s", migration.version, migration.description)
        except Exception as exc:
            msg = f"Migration v{migration.version} failed: {exc}"
            raise MigrationError(msg) from exc

    return applied


async def _sqlite_current_version(db: Any) -> int:
    cursor = await db.execute("SELECT MAX(version) FROM _migrations")
    row = await cursor.fetchone()
    if row is None or row[0] is None:
        return 0
    return row[0]


# ---------------------------------------------------------------------------
# Postgres implementation
# ---------------------------------------------------------------------------


async def _run_postgres(store: Any, registry: MigrationRegistry) -> int:
    pool = store._ensure_init()
    async with pool.acquire() as conn:
        await conn.execute(_POSTGRES_CREATE_MIGRATIONS_TABLE)

        current_version = await _postgres_current_version(conn)
        pending = registry.list_pending(current_version)
        if not pending:
            return 0

        applied = 0
        for migration in pending:
            try:
                # Each migration runs inside its own transaction so that a
                # failure doesn't partially apply a migration (SQLite commits
                # per migration too, achieving equivalent isolation).
                async with conn.transaction():
                    await migration.up(conn)
                    await conn.execute(
                        "INSERT INTO _migrations (version, description, applied_at) "
                        "VALUES ($1, $2, NOW()::text)",
                        migration.version,
                        migration.description,
                    )
                applied += 1
                logger.info("Applied migration v%d: %s", migration.version, migration.description)
            except Exception as exc:
                msg = f"Migration v{migration.version} failed: {exc}"
                raise MigrationError(msg) from exc

    return applied


async def _postgres_current_version(conn: Any) -> int:
    row = await conn.fetchval("SELECT MAX(version) FROM _migrations")
    return row or 0


# ---------------------------------------------------------------------------
# Backend detection
# ---------------------------------------------------------------------------


def _detect_backend(store: Any) -> str:
    """Detect whether a store is SQLite or Postgres based on its type name."""
    name = type(store).__name__
    if "SQLite" in name:
        return "sqlite"
    if "Postgres" in name:
        return "postgres"
    # Fallback: check for _db (sqlite) vs _pool (postgres) attributes
    if hasattr(store, "_db"):
        return "sqlite"
    if hasattr(store, "_pool"):
        return "postgres"
    return "unknown"


# ---------------------------------------------------------------------------
# Built-in migrations
# ---------------------------------------------------------------------------


async def _migration_v1_up(conn: Any) -> None:
    """Add the ``category`` column to ``memory_items``.

    Works for both SQLite (aiosqlite.Connection) and Postgres
    (asyncpg.Connection) — both expose ``.execute()``. SQLite does not
    support ``ADD COLUMN IF NOT EXISTS``, so a failure (column already
    present) is swallowed to keep the migration idempotent.
    """
    with contextlib.suppress(Exception):
        await conn.execute("ALTER TABLE memory_items ADD COLUMN category TEXT")


DEFAULT_REGISTRY = MigrationRegistry()
DEFAULT_REGISTRY.register(
    Migration(
        version=1,
        description="Add category column to memory_items",
        up=_migration_v1_up,
    )
)


def get_default_registry() -> MigrationRegistry:
    """Return the built-in migration registry with all bundled migrations."""
    return DEFAULT_REGISTRY
